# 金融研报分析 Agent

基于 LlamaIndex + DeepSeek + BGE-M3 的智能金融研报分析系统，支持研报问答和数据分析。

## 功能特性

- **RAG 知识问答**：上传 PDF 研报，基于语义检索回答问题
- **数据分析**：计算 CAGR、利润率、ROE 等财务指标
- **代码执行**：Agent 自动编写 Python 代码进行数据分析
- **图表生成**：生成折线图、柱状图等专业可视化
- **多轮对话**：支持上下文追问，记忆对话历史

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | LlamaIndex |
| LLM | DeepSeek-V3 (OpenAI 兼容接口) |
| Embedding | BGE-M3 (本地部署) |
| 向量数据库 | Milvus Lite |
| Agent 类型 | ReAct Agent |
| 前端 | Streamlit |

## 快速开始

### 1. 环境要求

- Python 3.10+
- CUDA（可选，GPU 加速）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入 DeepSeek API Key
# DEEPSEEK_API_KEY=your-api-key-here
```

获取 DeepSeek API Key：https://platform.deepseek.com/

### 4. 准备数据

```bash
# 将 PDF 研报放入 data 目录
mkdir -p data
cp your_report.pdf ./data/
```

### 5. 启动服务

**终端交互模式：**
```bash
python main.py
```

**Web 界面模式：**
```bash
streamlit run app.py
```

## 项目结构

```
rag/
├── config.py              # 配置管理（LLM、Embedding、RAG 参数）
├── agent.py               # ReAct Agent 核心
├── main.py                # 终端交互入口
├── app.py                 # Streamlit Web 前端
├── requirements.txt       # 依赖清单
├── .env.example           # 环境变量模板
├── data/                  # PDF 研报存放目录
└── tools/
    ├── __init__.py
    ├── rag_tool.py        # RAG 检索工具
    └── analysis_tool.py   # 数据分析工具
```

## 使用示例

### 研报问答

```
用户: 概括 2023 年新能源汽车的出海情况
Agent: 根据研报内容，2023 年新能源汽车出海呈现以下特点...
```

### 数据分析

```
用户: 对比 A 公司和 B 公司过去三年的利润率
Agent: 我来计算两家公司的利润率并进行对比...

| 年份 | A 公司 | B 公司 |
|------|--------|--------|
| 2021 | 10.5%  | 8.2%   |
| 2022 | 12.3%  | 9.1%   |
| 2023 | 14.8%  | 11.5%  |
```

### 代码执行

```
用户: 计算复合增长率，起始值 100，结束值 150，3 年
Agent: 三年复合增长率 (CAGR) = 14.47%
```

## 核心模块说明

### config.py - 配置管理

集中管理所有配置参数：
- `LLMConfig`: DeepSeek API 配置
- `EmbeddingConfig`: BGE-M3 模型配置
- `MilvusConfig`: 向量数据库配置
- `RAGConfig`: 检索参数配置

### tools/rag_tool.py - RAG 工具

实现完整的 RAG 流程：
1. PDF 文档加载
2. 语义分块 (SentenceSplitter)
3. 向量嵌入 (BGE-M3)
4. Milvus 存储
5. 相似度检索
6. BGE-Reranker 重排序

### tools/analysis_tool.py - 数据分析工具

提供财务分析能力：
- `FinancialMetrics`: CAGR、YOY、QOQ、利润率、ROE、ROA
- `ChartGenerator`: 折线图、柱状图、饼图
- `CodeExecutor`: 安全的 Python 代码执行

### agent.py - ReAct Agent

实现 Reasoning + Acting 架构：
- 自动选择工具
- 多轮对话记忆
- 工具调用反思

## 配置说明

### LLM 配置

```python
# config.py 或 .env
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

### RAG 配置

```python
# config.py
chunk_size = 1024        # 分块大小
chunk_overlap = 200      # 分块重叠
similarity_top_k = 5     # 检索数量
rerank_top_n = 3         # 重排序后数量
```

## 常见问题

### Q: 首次运行很慢？

首次运行需要下载 BGE-M3 模型（约 2GB），请耐心等待。模型会缓存到本地，后续启动会很快。

### Q: 如何使用 GPU 加速？

确保安装了 CUDA 版本的 PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q: Milvus 报错？

Milvus Lite 会在当前目录创建数据库文件，确保有写入权限。

## 许可证

MIT License

## 致谢

- [LlamaIndex](https://github.com/run-llama/llama_index)
- [DeepSeek](https://www.deepseek.com/)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Milvus](https://milvus.io/)
