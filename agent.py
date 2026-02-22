# -*- coding: utf-8 -*-
"""
============================================================
Agent 模块 (Agent Module)
============================================================
本模块实现金融研报分析 Agent 的核心逻辑，包括：

1. ReAct Agent 初始化（Reasoning + Acting）
2. 工具注册与管理
3. 对话记忆管理（ChatMemory）
4. 多轮对话支持

ReAct 架构说明：
- Reasoning（推理）：Agent 分析用户问题，决定使用哪个工具
- Acting（行动）：调用工具执行具体任务
- Observation（观察）：获取工具执行结果
- 循环直到得出最终答案
============================================================
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json

# LlamaIndex 核心组件
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage

# LLM
from llama_index.llms.openai import OpenAI

# 本地模块
from config import settings, setup_logging
from tools.rag_tool import RAGTool, create_rag_tool
from tools.analysis_tool import AnalysisTool, create_analysis_tool

# 初始化日志
logger = logging.getLogger(__name__)


# ============================================================
# 1. Agent 工具管理器
# ============================================================
class ToolManager:
    """
    工具管理器

    负责注册、管理和分发 Agent 可用的工具

    工具类型：
    - RAG 工具：研报知识检索
    - 分析工具：数据分析、代码执行
    - 自定义工具：用户扩展
    """

    def __init__(self):
        """初始化工具管理器"""
        self._tools: List[FunctionTool] = []
        self._tool_map: Dict[str, FunctionTool] = {}
        logger.info("工具管理器初始化")

    def register_tool(self, tool: FunctionTool) -> None:
        """
        注册工具

        Args:
            tool: FunctionTool 实例
        """
        self._tools.append(tool)
        self._tool_map[tool.metadata.name] = tool
        logger.info(f"已注册工具: {tool.metadata.name}")

    def register_rag_tool(self, rag_tool: RAGTool) -> None:
        """注册 RAG 检索工具"""
        tool = rag_tool.as_tool()
        self.register_tool(tool)

    def register_analysis_tool(self, analysis_tool: AnalysisTool) -> None:
        """注册数据分析工具（包含代码执行器）"""
        # 注册通用分析工具
        analysis_fn_tool = analysis_tool.as_tool()
        self.register_tool(analysis_fn_tool)

        # 注册代码执行器
        code_tool = analysis_tool.as_code_interpreter_tool()
        self.register_tool(code_tool)

    def get_tools(self) -> List[FunctionTool]:
        """获取所有已注册的工具"""
        return self._tools

    def get_tool(self, name: str) -> Optional[FunctionTool]:
        """根据名称获取工具"""
        return self._tool_map.get(name)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tool_map.keys())

    def get_tool_descriptions(self) -> str:
        """获取所有工具的描述（用于提示词）"""
        descriptions = []
        for tool in self._tools:
            desc = f"- {tool.metadata.name}: {tool.metadata.description}"
            descriptions.append(desc)
        return "\n".join(descriptions)


# ============================================================
# 2. 对话记忆管理器
# ============================================================
class MemoryManager:
    """
    对话记忆管理器

    使用 ChatMemoryBuffer 管理对话历史，支持：
    - 保存多轮对话
    - 限制上下文长度
    - 持久化存储（可选）
    """

    def __init__(self, token_limit: int = 8192):
        """
        初始化记忆管理器

        Args:
            token_limit: 记忆的 token 上限
        """
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)
        logger.info(f"对话记忆初始化，token 上限: {token_limit}")

    def get_memory(self) -> ChatMemoryBuffer:
        """获取记忆实例"""
        return self._memory

    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到记忆

        Args:
            role: 角色（user/assistant/system）
            content: 消息内容
        """
        message = ChatMessage(role=role, content=content)
        self._memory.put(message)

    def get_chat_history(self) -> List[ChatMessage]:
        """获取完整对话历史"""
        return self._memory.get_all()

    def clear(self) -> None:
        """清空对话记忆"""
        self._memory.reset()
        logger.info("对话记忆已清空")

    def save_to_file(self, filepath: str) -> None:
        """
        保存对话历史到文件

        Args:
            filepath: 保存路径
        """
        history = self.get_chat_history()
        data = [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"对话历史已保存到: {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """
        从文件加载对话历史

        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.clear()
            for item in data:
                self.add_message(item["role"], item["content"])
            logger.info(f"已从 {filepath} 加载对话历史")
        except FileNotFoundError:
            logger.warning(f"对话历史文件不存在: {filepath}")


# ============================================================
# 3. 金融分析 Agent
# ============================================================
class FinancialAnalysisAgent:
    """
    金融研报分析 Agent

    核心能力：
    1. 基于研报的知识问答（RAG）
    2. 财务数据分析和可视化
    3. 多轮对话支持
    4. 工具调用和推理

    使用方式：
        agent = FinancialAnalysisAgent()
        agent.initialize()  # 初始化（加载模型、构建索引）

        response = agent.chat("新能源汽车出海情况如何？")
        print(response)
    """

    def __init__(
        self,
        data_dir: str = None,
        enable_rerank: bool = True,
        verbose: bool = True
    ):
        """
        初始化 Agent

        Args:
            data_dir: 研报 PDF 目录
            enable_rerank: 是否启用重排序
            verbose: 是否打印详细日志
        """
        self.data_dir = data_dir or settings.rag.data_dir
        self.enable_rerank = enable_rerank
        self.verbose = verbose

        # 组件实例
        self._llm = None
        self._rag_tool = None
        self._analysis_tool = None
        self._tool_manager = None
        self._memory_manager = None
        self._agent = None

        # 状态
        self._initialized = False

        logger.info("金融分析 Agent 创建完成（尚未初始化）")

    def _init_llm(self) -> OpenAI:
        """
        初始化大语言模型（DeepSeek）

        使用 OpenAI 兼容接口调用 DeepSeek-V3
        """
        logger.info("正在初始化 LLM...")

        self._llm = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            model=settings.llm.model_name,
            temperature=settings.llm.agent_temperature,  # Agent 使用较低温度
            max_tokens=settings.llm.max_tokens,
        )

        # 设置全局 LLM
        Settings.llm = self._llm

        logger.info(f"LLM 初始化完成: {settings.llm.model_name}")
        return self._llm

    def _init_tools(self) -> ToolManager:
        """
        初始化工具

        包括：
        - RAG 检索工具
        - 数据分析工具
        - 代码执行器
        """
        logger.info("正在初始化工具...")

        self._tool_manager = ToolManager()

        # 1. 初始化 RAG 工具
        logger.info("初始化 RAG 工具...")
        self._rag_tool = create_rag_tool(
            data_dir=self.data_dir,
            enable_rerank=self.enable_rerank
        )
        self._tool_manager.register_rag_tool(self._rag_tool)

        # 2. 初始化分析工具
        logger.info("初始化数据分析工具...")
        self._analysis_tool = create_analysis_tool()
        self._tool_manager.register_analysis_tool(self._analysis_tool)

        logger.info(f"工具初始化完成，共 {len(self._tool_manager.list_tools())} 个工具")
        return self._tool_manager

    def _init_memory(self) -> MemoryManager:
        """初始化对话记忆"""
        self._memory_manager = MemoryManager()
        return self._memory_manager

    def initialize(self) -> None:
        """
        完整初始化 Agent

        流程：
        1. 初始化 LLM
        2. 初始化工具
        3. 初始化记忆
        4. 创建 ReAct Agent
        """
        if self._initialized:
            logger.warning("Agent 已经初始化，跳过")
            return

        logger.info("=" * 60)
        logger.info("开始初始化金融分析 Agent...")
        logger.info("=" * 60)

        try:
            # 1. 初始化 LLM
            self._init_llm()

            # 2. 初始化工具
            self._init_tools()

            # 3. 初始化记忆
            self._init_memory()

            # 4. 创建 ReAct Agent
            logger.info("正在创建 ReAct Agent...")
            self._agent = ReActAgent.from_tools(
                self._tool_manager.get_tools(),
                llm=self._llm,
                memory=self._memory_manager.get_memory(),
                verbose=self.verbose,
                max_iterations=settings.agent.max_iterations,
                system_prompt=settings.agent.system_prompt,
            )

            self._initialized = True
            logger.info("=" * 60)
            logger.info("Agent 初始化完成！")
            logger.info(f"可用工具: {self._tool_manager.list_tools()}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Agent 初始化失败: {e}")
            raise

    def chat(self, message: str) -> str:
        """
        发送消息并获取回复

        Args:
            message: 用户消息

        Returns:
            Agent 的回复
        """
        if not self._initialized:
            raise RuntimeError("Agent 尚未初始化，请先调用 initialize()")

        logger.info(f"收到用户消息: {message[:50]}...")

        try:
            # 调用 Agent
            response = self._agent.chat(message)

            # 记录到对话历史
            self._memory_manager.add_message("user", message)
            self._memory_manager.add_message("assistant", str(response))

            logger.info("回复生成完成")
            return str(response)

        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"

    def chat_stream(self, message: str):
        """
        流式对话（逐字返回）

        Args:
            message: 用户消息

        Yields:
            生成的文本片段
        """
        if not self._initialized:
            raise RuntimeError("Agent 尚未初始化，请先调用 initialize()")

        logger.info(f"收到用户消息（流式）: {message[:50]}...")

        try:
            # 流式调用 Agent
            response = self._agent.chat(message)
            yield str(response)

        except Exception as e:
            logger.error(f"流式对话处理失败: {e}")
            yield f"抱歉，处理您的请求时出现错误: {str(e)}"

    def reset_memory(self) -> None:
        """重置对话记忆"""
        if self._memory_manager:
            self._memory_manager.clear()
            logger.info("对话记忆已重置")

    def get_chat_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        if not self._memory_manager:
            return []

        history = self._memory_manager.get_chat_history()
        return [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ]

    def add_document(self, filepath: str) -> bool:
        """
        添加新文档到知识库

        Args:
            filepath: PDF 文件路径

        Returns:
            是否添加成功
        """
        if not self._initialized:
            logger.warning("Agent 未初始化，无法添加文档")
            return False

        try:
            # 重新构建索引
            self._rag_tool.build_index(
                documents=None,
                data_dir=str(Path(filepath).parent)
            )
            logger.info(f"文档已添加: {filepath}")
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取 Agent 状态信息"""
        return {
            "initialized": self._initialized,
            "llm_model": settings.llm.model_name if self._initialized else None,
            "tools": self._tool_manager.list_tools() if self._tool_manager else [],
            "memory_turns": len(self.get_chat_history()) if self._memory_manager else 0,
            "data_dir": self.data_dir,
        }


# ============================================================
# 4. 简化工厂函数
# ============================================================
def create_agent(
    data_dir: str = None,
    enable_rerank: bool = True,
    verbose: bool = True
) -> FinancialAnalysisAgent:
    """
    创建并初始化 Agent 的便捷函数

    Args:
        data_dir: 研报目录
        enable_rerank: 启用重排序
        verbose: 详细日志

    Returns:
        初始化完成的 Agent 实例
    """
    agent = FinancialAnalysisAgent(
        data_dir=data_dir,
        enable_rerank=enable_rerank,
        verbose=verbose
    )
    agent.initialize()
    return agent


# ============================================================
# 5. 测试入口
# ============================================================
if __name__ == "__main__":
    # 初始化日志
    setup_logging()

    print("\n" + "=" * 60)
    print("金融研报分析 Agent - 测试模式")
    print("=" * 60)

    # 创建 Agent
    print("\n正在初始化 Agent...")
    agent = create_agent(verbose=True)

    # 打印状态
    status = agent.get_status()
    print(f"\nAgent 状态:")
    print(f"  - 初始化: {status['initialized']}")
    print(f"  - LLM: {status['llm_model']}")
    print(f"  - 工具: {status['tools']}")
    print(f"  - 数据目录: {status['data_dir']}")

    # 交互式对话
    print("\n" + "=" * 60)
    print("交互模式（输入 'quit' 退出，'reset' 重置记忆）")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n用户: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("再见！")
                break

            if user_input.lower() == "reset":
                agent.reset_memory()
                print("对话记忆已重置")
                continue

            if user_input.lower() == "status":
                print(json.dumps(agent.get_status(), ensure_ascii=False, indent=2))
                continue

            # 发送消息
            response = agent.chat(user_input)
            print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\n\n已中断，再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
