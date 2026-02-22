# -*- coding: utf-8 -*-
"""
============================================================
Streamlit Web 应用 (Web Frontend)
============================================================
本模块提供金融研报分析 Agent 的 Web 界面

功能：
- 友好的聊天界面
- 文件上传功能
- 对话历史显示
- 状态监控面板
- 图表展示

启动方式：
    streamlit run app.py

    或指定端口：
    streamlit run app.py --server.port 8501
============================================================
"""

import streamlit as st
from streamlit_chat import message
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import base64

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 本地模块
from config import settings, setup_logging, check_environment
from agent import FinancialAnalysisAgent, create_agent

# 页面配置（必须在最前面）
st.set_page_config(
    page_title="金融研报分析 Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "金融研报分析 Agent - 基于 LlamaIndex + DeepSeek + BGE-M3"
    }
)

# 初始化日志
logger = setup_logging()


# ============================================================
# 1. Session State 管理
# ============================================================
def init_session_state():
    """
    初始化 Streamlit Session State

    用于保存：
    - Agent 实例
    - 对话历史
    - 配置状态
    """
    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "messages" not in st.session_state:
        st.session_state.messages = []


# ============================================================
# 2. Agent 初始化
# ============================================================
@st.cache_resource
def get_agent(data_dir: str, enable_rerank: bool) -> FinancialAnalysisAgent:
    """
    获取或创建 Agent 实例（使用 Streamlit 缓存）

    Args:
        data_dir: 研报目录
        enable_rerank: 是否启用重排序

    Returns:
        Agent 实例
    """
    agent = create_agent(
        data_dir=data_dir,
        enable_rerank=enable_rerank,
        verbose=False  # Web 模式关闭详细日志
    )
    return agent


def initialize_agent():
    """初始化 Agent（带进度提示）"""
    with st.spinner("正在初始化 Agent，首次运行可能需要下载模型..."):
        try:
            st.session_state.agent = get_agent(
                data_dir=settings.rag.data_dir,
                enable_rerank=settings.rag.rerank_enabled
            )
            st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"Agent 初始化失败: {e}")
            logger.error(f"Agent 初始化失败: {e}")
            return False


# ============================================================
# 3. UI 组件
# ============================================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("📊 金融研报分析 Agent")
        st.markdown("---")

        # 状态面板
        st.subheader("系统状态")
        env_check = check_environment()

        status_items = {
            "API Key": "✅" if env_check["api_key"] else "❌",
            "数据目录": "✅" if env_check["data_dir"] else "⚠️",
            "CUDA": "✅" if env_check.get("cuda_available") else "⚠️ CPU"
        }

        for name, status in status_items.items():
            st.write(f"{status} {name}")

        st.markdown("---")

        # 配置面板
        st.subheader("配置")
        st.write(f"**LLM**: {settings.llm.model_name}")
        st.write(f"**Embedding**: BGE-M3")
        st.write(f"**向量库**: Milvus Lite")

        st.markdown("---")

        # 操作按钮
        st.subheader("操作")

        if st.button("🔄 重置对话", use_container_width=True):
            if st.session_state.agent:
                st.session_state.agent.reset_memory()
                st.session_state.messages = []
                st.rerun()

        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            **研报问答**：
            - 直接输入关于研报内容的问题
            - 例如："概括新能源汽车出海情况"

            **数据分析**：
            - 请求计算财务指标
            - 例如："计算 A 公司的 CAGR"

            **代码执行**：
            - 让 Agent 编写 Python 代码
            - 例如："画一个折线图对比两家公司"
            """)

        # 关于
        st.markdown("---")
        st.caption("Powered by LlamaIndex + DeepSeek + BGE-M3")


def render_chat_history():
    """渲染聊天历史"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 如果有图表，显示图表
            if "chart" in msg:
                st.image(base64.b64decode(msg["chart"]), use_column_width=True)


def handle_user_input():
    """处理用户输入"""
    if prompt := st.chat_input("输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取 Agent 回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    response = st.session_state.agent.chat(prompt)

                    # 流式显示（简单版本）
                    response_placeholder = st.empty()
                    full_response = ""

                    # 模拟打字效果
                    for chunk in response.split():
                        full_response += chunk + " "
                        response_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)

                    response_placeholder.markdown(full_response)

                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": str(response)
                    })

                except Exception as e:
                    error_msg = f"抱歉，处理请求时出错: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


# ============================================================
# 4. 文件上传功能
# ============================================================
def render_file_uploader():
    """渲染文件上传组件"""
    st.subheader("📁 上传研报")

    uploaded_files = st.file_uploader(
        "上传 PDF 研报文件",
        type=["pdf"],
        accept_multiple_files=True,
        help="支持上传多个 PDF 文件"
    )

    if uploaded_files:
        # 保存文件到数据目录
        data_dir = Path(settings.rag.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"已上传: {uploaded_file.name}")

        # 重新构建索引按钮
        if st.button("🔄 重新构建索引"):
            with st.spinner("正在构建索引..."):
                try:
                    st.session_state.agent._rag_tool.build_index(str(data_dir))
                    st.success("索引构建完成！")
                except Exception as e:
                    st.error(f"索引构建失败: {e}")


# ============================================================
# 5. 示例问题
# ============================================================
def render_example_questions():
    """渲染示例问题按钮"""
    st.subheader("💡 试试这些问题")

    examples = [
        "概括 2023 年新能源汽车的出海情况",
        "对比比亚迪和特斯拉的利润率变化",
        "分析光伏行业的增长趋势",
        "计算某公司过去三年的复合增长率",
    ]

    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # 模拟用户输入
                st.session_state.example_query = example


# ============================================================
# 6. 主页面布局
# ============================================================
def main():
    """主函数"""
    # 初始化 Session State
    init_session_state()

    # 检查 API Key
    env_check = check_environment()
    if not env_check["api_key"]:
        st.error("⚠️ 未配置 DEEPSEEK_API_KEY 环境变量")
        st.info("""
        请在终端设置环境变量：
        ```
        export DEEPSEEK_API_KEY='your-api-key'
        ```

        或创建 `.env` 文件：
        ```
        DEEPSEEK_API_KEY=your-api-key
        ```
        """)
        st.stop()

    # 初始化 Agent
    if not st.session_state.initialized:
        if not initialize_agent():
            st.stop()

    # 渲染侧边栏
    render_sidebar()

    # 主内容区
    st.title("💬 金融研报分析对话")

    # 标签页
    tab1, tab2, tab3 = st.tabs(["💬 对话", "📁 文件管理", "📊 数据分析"])

    with tab1:
        # 聊天区域
        chat_container = st.container()

        with chat_container:
            # 显示聊天历史
            render_chat_history()

            # 处理用户输入
            handle_user_input()

        # 示例问题
        st.markdown("---")
        render_example_questions()

    with tab2:
        render_file_uploader()

        # 显示已上传文件
        st.subheader("📄 已上传文件")
        data_dir = Path(settings.rag.data_dir)
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            if pdf_files:
                for pdf in pdf_files:
                    st.write(f"📕 {pdf.name}")
            else:
                st.info("暂无上传文件")
        else:
            st.info("数据目录不存在")

    with tab3:
        st.subheader("📈 数据分析工具")

        st.markdown("""
        Agent 可以执行以下数据分析任务：

        **财务指标计算**：
        - CAGR（复合年增长率）
        - 利润率、ROE、ROA
        - 同比/环比增长率

        **数据可视化**：
        - 趋势折线图
        - 对比柱状图
        - 占比饼图

        **示例请求**：
        - "用 Python 计算 CAGR，起始值 100，结束值 150，3 年"
        - "生成一个折线图展示营收增长"
        """)

        # 简易代码执行器
        st.subheader("🔧 代码执行器（高级）")
        st.caption("直接执行 Python 代码进行数据分析")

        default_code = """import pandas as pd
import numpy as np

# 示例：计算 CAGR
start_value = 100
end_value = 150
years = 3

cagr = (end_value / start_value) ** (1/years) - 1
print(f"复合年增长率: {cagr * 100:.2f}%")

# 创建数据表
df = pd.DataFrame({
    '年份': [2021, 2022, 2023],
    '营收': [100, 120, 150],
    '利润': [10, 15, 20]
})
print(df)
"""

        code = st.text_area(
            "Python 代码",
            value=default_code,
            height=200,
            key="code_input"
        )

        if st.button("▶️ 执行代码"):
            if st.session_state.agent:
                with st.spinner("执行中..."):
                    try:
                        result = st.session_state.agent._analysis_tool.execute_custom_analysis(code)
                        if result["success"]:
                            st.code(result["output"], language="python")
                            if result.get("result"):
                                st.json(result["result"])
                        else:
                            st.error(result["error"])
                    except Exception as e:
                        st.error(f"执行失败: {e}")


# ============================================================
# 7. 应用入口
# ============================================================
if __name__ == "__main__":
    main()
