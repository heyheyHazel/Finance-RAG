# -*- coding: utf-8 -*-
"""
============================================================
配置模块 (Configuration Module)
============================================================
本模块集中管理金融研报分析 Agent 的所有配置参数：
1. LLM 配置（DeepSeek-V3 通过 OpenAI 兼容接口）
2. Embedding 模型配置（BGE-M3 本地部署）
3. 向量数据库配置（Milvus Lite）
4. RAG 检索参数
5. 路径和日志配置

设计理念：
- 使用 pydantic 进行配置验证
- 支持从环境变量覆盖配置
- 集中管理，便于维护
============================================================
"""

import os
import logging
from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量（从 .env 文件）
load_dotenv()


# ============================================================
# 1. LLM 配置类
# ============================================================
class LLMConfig(BaseModel):
    """
    大语言模型配置
    使用 DeepSeek-V3 通过 OpenAI 兼容接口调用

    DeepSeek API 文档: https://platform.deepseek.com/docs
    """
    # API 配置
    api_key: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
        description="DeepSeek API Key"
    )
    base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API 基础 URL（OpenAI 兼容格式）"
    )

    # 模型配置
    model_name: str = Field(
        default="deepseek-chat",
        description="模型名称：deepseek-chat（对话）或 deepseek-reasoner（推理）"
    )

    # 生成参数
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="生成温度，0 表示确定性，越高越随机"
    )
    max_tokens: int = Field(
        default=4096,
        description="最大生成 token 数"
    )

    # Agent 特化参数
    agent_temperature: float = Field(
        default=0.0,  # Agent 调用工具时使用低温度保证准确性
        description="Agent 工具调用时的温度"
    )


# ============================================================
# 2. Embedding 配置类
# ============================================================
class EmbeddingConfig(BaseModel):
    """
    向量嵌入模型配置
    使用 BGE-M3 本地模型进行文本向量化

    BGE-M3 特点：
    - 多语言支持（中英文效果好）
    - 多粒度：支持密集检索、稀疏检索、多向量检索
    - 长文本支持：最长 8192 tokens
    """
    # 模型路径（HuggingFace 模型 ID 或本地路径）
    model_name: str = Field(
        default="BAAI/bge-m3",
        description="BGE-M3 模型名称或本地路径"
    )

    # 本地模型缓存目录
    cache_folder: Optional[str] = Field(
        default=None,
        description="模型缓存目录，None 则使用默认路径"
    )

    # 嵌入维度
    embed_dim: int = Field(
        default=1024,
        description="BGE-M3 输出向量维度"
    )

    # 批处理大小
    batch_size: int = Field(
        default=32,
        description="批量编码时的批次大小"
    )

    # 设备配置
    device: str = Field(
        default="cuda",  # AutoDL 环境使用 GPU
        description="运行设备：cuda 或 cpu"
    )

    # 归一化
    normalize: bool = Field(
        default=True,
        description="是否对向量进行 L2 归一化"
    )


# ============================================================
# 3. 向量数据库配置类
# ============================================================
class MilvusConfig(BaseModel):
    """
    Milvus 向量数据库配置
    使用 Milvus Lite 进行本地轻量级部署

    Milvus Lite 特点：
    - 无需独立部署，Python 库即可运行
    - 适合开发和中小规模数据
    - 数据持久化到本地文件
    """
    # 数据库类型
    uri: str = Field(
        default="./milvus_finreport.db",
        description="Milvus Lite 数据库文件路径"
    )

    # 集合名称
    collection_name: str = Field(
        default="financial_reports",
        description="存储研报向量的集合名称"
    )

    # 索引参数
    index_type: str = Field(
        default="IVF_FLAT",
        description="向量索引类型：IVF_FLAT, HNSW, AUTOINDEX 等"
    )

    metric_type: str = Field(
        default="COSINE",
        description="相似度度量：COSINE, L2, IP（内积）"
    )

    # 搜索参数
    nlist: int = Field(
        default=128,
        description="IVF 索引的聚类中心数量"
    )


# ============================================================
# 4. RAG 配置类
# ============================================================
class RAGConfig(BaseModel):
    """
    RAG（检索增强生成）配置
    控制 PDF 解析、分块、检索等核心参数
    """
    # 数据目录
    data_dir: str = Field(
        default="./data",
        description="研报 PDF 文件存放目录"
    )

    # 文档分块参数
    chunk_size: int = Field(
        default=1024,
        description="每个文本块的最大 token 数"
    )
    chunk_overlap: int = Field(
        default=200,
        description="相邻文本块的重叠 token 数"
    )

    # 检索参数
    similarity_top_k: int = Field(
        default=5,
        description="初始检索返回的文档数量"
    )

    # 重排序参数
    rerank_enabled: bool = Field(
        default=True,
        description="是否启用重排序"
    )
    rerank_top_n: int = Field(
        default=3,
        description="重排序后返回的文档数量"
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="重排序模型（BGE-Reranker 或 Cohere）"
    )

    # 文本分割策略
    split_by: str = Field(
        default="sentence",
        description="分割方式：sentence（句子）、paragraph（段落）、token"
    )


# ============================================================
# 5. Agent 配置类
# ============================================================
class AgentConfig(BaseModel):
    """
    Agent 配置
    控制 ReAct Agent 的行为和工具调用
    """
    # Agent 类型
    agent_type: str = Field(
        default="ReAct",
        description="Agent 类型：ReAct（推理+行动）"
    )

    # 最大迭代次数
    max_iterations: int = Field(
        default=10,
        description="Agent 最大推理迭代次数"
    )

    # 工具调用超时
    tool_timeout: int = Field(
        default=60,
        description="单个工具执行超时时间（秒）"
    )

    # 对话记忆
    memory_window: int = Field(
        default=10,
        description="保留的对话轮数"
    )

    # 系统提示词
    system_prompt: str = Field(
        default="""你是一位专业的金融分析师助手，擅长分析行业研报和财务数据。

你的能力包括：
1. **研报问答**：基于上传的金融研报内容，回答用户关于行业趋势、公司分析等问题
2. **数据分析**：使用 Python 和 Pandas 进行财务数据分析，计算增长率、利润率等指标
3. **可视化**：生成专业的数据图表，帮助用户直观理解数据

回答原则：
- 基于事实和数据，不做无根据的推测
- 引用研报原文时注明来源
- 涉及数据计算时展示计算过程
- 使用专业但易懂的语言
""",
        description="Agent 系统提示词"
    )


# ============================================================
# 6. 日志配置类
# ============================================================
class LogConfig(BaseModel):
    """
    日志配置
    用于调试和运行监控
    """
    # 日志级别
    level: str = Field(
        default="INFO",
        description="日志级别：DEBUG, INFO, WARNING, ERROR"
    )

    # 日志格式
    format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="日志输出格式"
    )

    # 日志文件
    file_path: Optional[str] = Field(
        default="./logs/agent.log",
        description="日志文件路径，None 则只输出到控制台"
    )


# ============================================================
# 7. 全局配置类
# ============================================================
class Settings(BaseModel):
    """
    全局配置聚合类
    整合所有子配置，提供统一访问入口

    使用方式：
        from config import settings
        print(settings.llm.model_name)
    """
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    class Config:
        # 允许从环境变量覆盖
        env_nested_delimiter = "__"


# ============================================================
# 8. 初始化全局配置实例
# ============================================================
settings = Settings()


# ============================================================
# 9. 日志初始化函数
# ============================================================
def setup_logging(config: LogConfig = None) -> logging.Logger:
    """
    初始化日志系统

    Args:
        config: 日志配置，None 则使用全局配置

    Returns:
        配置好的 Logger 实例

    使用方式：
        logger = setup_logging()
        logger.info("系统启动")
    """
    if config is None:
        config = settings.log

    # 创建日志目录
    if config.file_path:
        Path(config.file_path).parent.mkdir(parents=True, exist_ok=True)

    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, config.level),
        format=config.format,
        handlers=[
            # 控制台输出
            logging.StreamHandler(),
            # 文件输出（如果指定了路径）
            *([logging.FileHandler(config.file_path, encoding='utf-8')]
              if config.file_path else [])
        ]
    )

    # 降低第三方库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# ============================================================
# 10. 环境检查函数
# ============================================================
def check_environment() -> dict:
    """
    检查运行环境是否正确配置

    Returns:
        包含各项检查结果的字典
    """
    results = {
        "api_key": bool(settings.llm.api_key),
        "data_dir": Path(settings.rag.data_dir).exists(),
        "cuda_available": False,
        "models_cache": True  # 模型会在首次运行时自动下载
    }

    # 检查 CUDA
    try:
        import torch
        results["cuda_available"] = torch.cuda.is_available()
        if results["cuda_available"]:
            results["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        results["torch_installed"] = False

    return results


# ============================================================
# 11. 主入口（用于测试配置）
# ============================================================
if __name__ == "__main__":
    # 初始化日志
    logger = setup_logging()

    # 打印配置信息
    print("=" * 60)
    print("金融研报分析 Agent - 配置信息")
    print("=" * 60)
    print(f"\n[LLM 配置]")
    print(f"  模型: {settings.llm.model_name}")
    print(f"  API URL: {settings.llm.base_url}")
    print(f"  温度: {settings.llm.temperature}")

    print(f"\n[Embedding 配置]")
    print(f"  模型: {settings.embedding.model_name}")
    print(f"  维度: {settings.embedding.embed_dim}")
    print(f"  设备: {settings.embedding.device}")

    print(f"\n[Milvus 配置]")
    print(f"  URI: {settings.milvus.uri}")
    print(f"  集合: {settings.milvus.collection_name}")

    print(f"\n[RAG 配置]")
    print(f"  数据目录: {settings.rag.data_dir}")
    print(f"  分块大小: {settings.rag.chunk_size}")
    print(f"  重排序: {'启用' if settings.rag.rerank_enabled else '禁用'}")

    print(f"\n[环境检查]")
    env_check = check_environment()
    for key, value in env_check.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")

    print("\n" + "=" * 60)
