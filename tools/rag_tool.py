# -*- coding: utf-8 -*-
"""
============================================================
RAG 工具模块 (Retrieval-Augmented Generation Tool)
============================================================
本模块实现金融研报的 RAG 功能，包括：

1. PDF 文档加载与解析
2. 语义分块（Semantic Chunking）
3. 向量嵌入与存储（使用 BGE-M3 + Milvus）
4. 相似度检索
5. 重排序（Reranking）

核心流程：
    PDF → 解析 → 分块 → 向量化 → 存入 Milvus → 查询 → 检索 → 重排 → 返回结果
============================================================
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# LlamaIndex 核心组件
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import FunctionTool
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Embedding 相关
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 向量存储
from llama_index.vector_stores.milvus import MilvusVectorStore

# LLM（OpenAI 兼容接口）
from llama_index.llms.openai import OpenAI

# 重排序
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 本地配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import settings, setup_logging

# 初始化日志
logger = logging.getLogger(__name__)


# ============================================================
# 1. 自定义轻量级重排序器（本地 BGE-Reranker）
# ============================================================
class BGEReranker(BaseNodePostprocessor):
    """
    基于 BGE-Reranker 的本地重排序器

    当 Cohere API 不可用时，使用本地 BGE-Reranker 模型进行重排序。
    BGE-Reranker 专门针对中英文优化，适合金融研报场景。

    原理：
    1. 接收初始检索的文档列表
    2. 将 Query 和每个文档拼接
    3. 使用 Cross-Encoder 计算相关性分数
    4. 按分数重新排序
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 3,
        device: str = "cuda"
    ):
        """
        初始化 BGE 重排序器

        Args:
            model_name: HuggingFace 模型 ID
            top_n: 重排序后保留的文档数量
            device: 运行设备（cuda/cpu）
        """
        super().__init__()
        self.top_n = top_n
        self.device = device
        self._model = None
        self._model_name = model_name

    def _load_model(self):
        """延迟加载模型（首次使用时才加载）"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"正在加载重排序模型: {self._model_name}")
                self._model = CrossEncoder(
                    self._model_name,
                    max_length=512,
                    device=self.device
                )
                logger.info("重排序模型加载完成")
            except Exception as e:
                logger.error(f"加载重排序模型失败: {e}")
                raise

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        对检索结果进行重排序

        Args:
            nodes: 初始检索的节点列表
            query_bundle: 用户查询

        Returns:
            重排序后的节点列表
        """
        if not nodes or query_bundle is None:
            return nodes[:self.top_n]

        # 延迟加载模型
        self._load_model()

        # 构造 Query-Document 对
        query_str = query_bundle.query_str
        pairs = [(query_str, node.node.get_content()) for node in nodes]

        # 计算相关性分数
        scores = self._model.predict(pairs)

        # 按分数排序
        scored_nodes = list(zip(nodes, scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # 更新节点分数并返回 Top-N
        result = []
        for node, score in scored_nodes[:self.top_n]:
            node.score = float(score)
            result.append(node)

        logger.info(f"重排序完成，从 {len(nodes)} 个文档中选出 {len(result)} 个")
        return result


# ============================================================
# 2. RAG 工具类
# ============================================================
class RAGTool:
    """
    RAG 工具类

    封装了完整的 RAG 流程：
    - 文档加载与解析
    - 向量化与索引构建
    - 相似度检索
    - 重排序优化

    使用方式：
        rag = RAGTool()
        rag.build_index("./data")  # 构建索引
        results = rag.query("新能源汽车出海情况")  # 查询
    """

    def __init__(
        self,
        llm=None,
        embed_model=None,
        vector_store=None,
        enable_rerank: bool = True
    ):
        """
        初始化 RAG 工具

        Args:
            llm: 大语言模型实例，None 则使用默认配置
            embed_model: 嵌入模型实例，None 则使用 BGE-M3
            vector_store: 向量存储实例，None 则使用 Milvus Lite
            enable_rerank: 是否启用重排序
        """
        self.enable_rerank = enable_rerank
        self._index = None
        self._query_engine = None
        self._documents = []

        # 初始化组件
        self._init_components(llm, embed_model, vector_store)

        logger.info("RAG 工具初始化完成")

    def _init_components(self, llm, embed_model, vector_store):
        """
        初始化 LLM、Embedding 和向量存储组件
        """
        # 1. 初始化 LLM（DeepSeek 通过 OpenAI 兼容接口）
        if llm is None:
            self.llm = OpenAI(
                api_key=settings.llm.api_key,
                base_url=settings.llm.base_url,
                model=settings.llm.model_name,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )
            logger.info(f"LLM 初始化完成: {settings.llm.model_name}")
        else:
            self.llm = llm

        # 2. 初始化 Embedding 模型（BGE-M3）
        if embed_model is None:
            self.embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding.model_name,
                cache_folder=settings.embedding.cache_folder,
                embed_batch_size=settings.embedding.batch_size,
            )
            logger.info(f"Embedding 模型初始化完成: {settings.embedding.model_name}")
        else:
            self.embed_model = embed_model

        # 3. 初始化向量存储（Milvus Lite）
        if vector_store is None:
            self.vector_store = MilvusVectorStore(
                uri=settings.milvus.uri,
                collection_name=settings.milvus.collection_name,
                dim=settings.embedding.embed_dim,
                overwrite=False,  # 不覆盖已有数据
            )
            logger.info(f"向量存储初始化完成: {settings.milvus.uri}")
        else:
            self.vector_store = vector_store

        # 4. 配置全局 Settings（LlamaIndex 使用）
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = settings.rag.chunk_size
        Settings.chunk_overlap = settings.rag.chunk_overlap

    def load_documents(self, data_dir: str) -> List[Document]:
        """
        从目录加载 PDF 文档

        Args:
            data_dir: 存放 PDF 文件的目录路径

        Returns:
            加载的文档列表
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 检查 PDF 文件
        pdf_files = list(data_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"目录中没有找到 PDF 文件: {data_dir}")
            return []

        logger.info(f"发现 {len(pdf_files)} 个 PDF 文件")

        # 使用 SimpleDirectoryReader 加载文档
        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=[".pdf"],
            recursive=True,  # 递归搜索子目录
            exclude_hidden=True,  # 排除隐藏文件
        )

        documents = reader.load_data()
        self._documents = documents

        logger.info(f"成功加载 {len(documents)} 个文档片段")

        # 打印文档元数据
        for i, doc in enumerate(documents[:3]):  # 只打印前3个
            logger.debug(f"文档 {i+1}: {doc.metadata}")

        return documents

    def _create_node_parser(self):
        """
        创建文档分块解析器

        分块策略说明：
        - SentenceSplitter: 基于句子边界分块，保留语义完整性
        - 也可选择 SemanticSplitterNodeParser: 基于语义相似度分块（更精确但更慢）
        """
        # 方案1：基于句子的分块（推荐，速度快）
        node_parser = SentenceSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )

        # 方案2：基于语义的分块（更精确，需要 Embedding 模型）
        # node_parser = SemanticSplitterNodeParser(
        #     buffer_size=1,
        #     breakpoint_percentile_threshold=95,
        #     embed_model=self.embed_model,
        # )

        return node_parser

    def build_index(self, data_dir: str = None, documents: List[Document] = None) -> VectorStoreIndex:
        """
        构建向量索引

        流程：
        1. 加载文档（如果未提供）
        2. 文档分块
        3. 向量化（使用 BGE-M3）
        4. 存入 Milvus

        Args:
            data_dir: PDF 文件目录
            documents: 已加载的文档列表（与 data_dir 二选一）

        Returns:
            构建好的向量索引
        """
        logger.info("开始构建向量索引...")

        # 1. 加载文档
        if documents is None:
            if data_dir is None:
                data_dir = settings.rag.data_dir
            documents = self.load_documents(data_dir)

        if not documents:
            raise ValueError("没有文档可供索引")

        # 2. 创建分块解析器
        node_parser = self._create_node_parser()

        # 3. 解析文档为节点（分块）
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"文档分块完成，共 {len(nodes)} 个节点")

        # 4. 创建存储上下文
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # 5. 构建向量索引
        self._index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        logger.info("向量索引构建完成")

        # 6. 创建查询引擎
        self._create_query_engine()

        return self._index

    def _create_query_engine(self):
        """
        创建查询引擎

        包含：
        - 向量检索器
        - 重排序器（可选）
        - 响应合成器
        """
        if self._index is None:
            raise ValueError("索引尚未构建，请先调用 build_index()")

        # 配置检索器
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=settings.rag.similarity_top_k,
        )

        # 配置重排序器
        node_postprocessors = []
        if self.enable_rerank:
            try:
                # 优先使用本地 BGE-Reranker
                reranker = BGEReranker(
                    model_name=settings.rag.rerank_model,
                    top_n=settings.rag.rerank_top_n,
                    device=settings.embedding.device,
                )
                node_postprocessors.append(reranker)
                logger.info("已启用 BGE 重排序器")
            except Exception as e:
                logger.warning(f"重排序器初始化失败，将跳过重排序: {e}")

        # 创建查询引擎
        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            llm=self.llm,
        )

        logger.info("查询引擎创建完成")

    def query(self, query_str: str) -> Dict[str, Any]:
        """
        执行 RAG 查询

        流程：
        1. 查询向量化
        2. 向量相似度检索
        3. 重排序（可选）
        4. LLM 生成回答

        Args:
            query_str: 用户查询字符串

        Returns:
            包含回答和来源的字典
        """
        if self._query_engine is None:
            raise ValueError("查询引擎未初始化，请先调用 build_index()")

        logger.info(f"执行查询: {query_str}")

        try:
            # 执行查询
            response = self._query_engine.query(query_str)

            # 提取来源信息
            source_nodes = []
            for node in response.source_nodes:
                source_nodes.append({
                    "content": node.node.get_content()[:500] + "...",  # 截断
                    "score": node.score,
                    "metadata": node.node.metadata,
                })

            result = {
                "answer": str(response),
                "sources": source_nodes,
                "query": query_str,
            }

            logger.info(f"查询完成，引用了 {len(source_nodes)} 个文档片段")
            return result

        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            raise

    def as_tool(self) -> FunctionTool:
        """
        将 RAG 工具转换为 LlamaIndex FunctionTool

        用于 Agent 调用

        Returns:
            FunctionTool 实例
        """
        def rag_query(query: str) -> str:
            """
            金融研报知识库检索工具

            用于回答关于上传的金融研报、行业分析报告的问题。
            当用户询问行业趋势、公司分析、市场数据等需要参考研报内容时使用。

            Args:
                query: 用户的问题，如"2023年新能源汽车出海情况"

            Returns:
                基于研报内容的回答
            """
            result = self.query(query)
            return result["answer"]

        return FunctionTool.from_defaults(
            fn=rag_query,
            name="rag_query",
            description="金融研报知识库检索。用于回答关于行业研报、公司分析、市场趋势的问题。"
        )


# ============================================================
# 3. 工具工厂函数
# ============================================================
def create_rag_tool(
    data_dir: str = None,
    enable_rerank: bool = True
) -> RAGTool:
    """
    创建 RAG 工具的便捷函数

    Args:
        data_dir: PDF 文件目录，None 则使用配置中的路径
        enable_rerank: 是否启用重排序

    Returns:
        初始化完成的 RAGTool 实例
    """
    if data_dir is None:
        data_dir = settings.rag.data_dir

    tool = RAGTool(enable_rerank=enable_rerank)

    # 检查是否有文档需要索引
    data_path = Path(data_dir)
    if data_path.exists() and list(data_path.glob("*.pdf")):
        tool.build_index(data_dir)

    return tool


# ============================================================
# 4. 测试入口
# ============================================================
if __name__ == "__main__":
    # 初始化日志
    setup_logging()

    # 创建 RAG 工具
    print("正在初始化 RAG 工具...")
    rag = create_rag_tool()

    # 交互式测试
    print("\n" + "=" * 60)
    print("RAG 工具测试（输入 'quit' 退出）")
    print("=" * 60)

    while True:
        query = input("\n请输入问题: ").strip()
        if query.lower() == "quit":
            break

        if not query:
            continue

        try:
            result = rag.query(query)
            print(f"\n回答: {result['answer']}")
            print(f"\n引用了 {len(result['sources'])} 个文档片段")
        except Exception as e:
            print(f"错误: {e}")
