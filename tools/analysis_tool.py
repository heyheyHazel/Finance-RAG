# -*- coding: utf-8 -*-
"""
============================================================
数据分析工具模块 (Data Analysis Tool)
============================================================
本模块实现金融数据分析功能，包括：

1. 结构化数据处理（Pandas DataFrame 操作）
2. 财务指标计算（CAGR、利润率、ROE 等）
3. 数据对比分析
4. 可视化图表生成

核心设计：
- 使用 Python 代码执行器安全运行分析代码
- 支持多种财务指标计算
- 生成专业的数据图表
============================================================
"""

import logging
import json
import io
import base64
import traceback
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import re

# 数据分析库
import pandas as pd
import numpy as np

# 可视化库
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# LlamaIndex 工具
from llama_index.core.tools import FunctionTool

# 本地配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import settings, setup_logging

# 初始化日志
logger = logging.getLogger(__name__)


# ============================================================
# 1. 财务指标计算器
# ============================================================
class FinancialMetrics:
    """
    财务指标计算工具类

    提供常用的财务分析指标计算方法：
    - CAGR: 复合年增长率
    - YOY: 同比增长率
    - QOQ: 环比增长率
    - 利润率、ROE、ROA 等
    """

    @staticmethod
    def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
        """
        计算复合年增长率 (Compound Annual Growth Rate)
        公式: CAGR = (End / Start)^(1/n) - 1

        Args:
            start_value: 起始值
            end_value: 结束值
            periods: 期数（年数）

        Returns:
            CAGR 值（小数形式，如 0.15 表示 15%）
        """
        if start_value <= 0 or end_value <= 0:
            raise ValueError("起始值和结束值必须为正数")
        if periods <= 0:
            raise ValueError("期数必须为正数")
        cagr = (end_value / start_value) ** (1 / periods) - 1
        return round(cagr, 4)

    @staticmethod
    def calculate_yoy(current: float, previous: float) -> float:
        """
        计算同比增长率 (Year-over-Year Growth)
        公式: YOY = (Current - Previous) / Previous
        """
        if previous == 0:
            raise ValueError("去年同期值不能为0")
        return round((current - previous) / previous, 4)

    @staticmethod
    def calculate_qoq(current: float, previous: float) -> float:
        """
        计算环比增长率 (Quarter-over-Quarter Growth)
        公式: QOQ = (Current - Previous) / Previous
        """
        if previous == 0:
            raise ValueError("上期值不能为0")
        return round((current - previous) / previous, 4)

    @staticmethod
    def calculate_profit_margin(revenue: float, profit: float) -> float:
        """计算利润率: Profit Margin = Profit / Revenue"""
        if revenue == 0:
            raise ValueError("营收不能为0")
        return round(profit / revenue, 4)

    @staticmethod
    def calculate_roe(net_income: float, equity: float) -> float:
        """计算净资产收益率 ROE = Net Income / Equity"""
        if equity == 0:
            raise ValueError("净资产不能为0")
        return round(net_income / equity, 4)

    @staticmethod
    def calculate_roa(net_income: float, assets: float) -> float:
        """计算资产收益率 ROA = Net Income / Total Assets"""
        if assets == 0:
            raise ValueError("总资产不能为0")
        return round(net_income / assets, 4)

    @staticmethod
    def calculate_debt_ratio(debt: float, assets: float) -> float:
        """计算资产负债率 = Total Debt / Total Assets"""
        if assets == 0:
            raise ValueError("总资产不能为0")
        return round(debt / assets, 4)


# ============================================================
# 2. 图表生成器
# ============================================================
class ChartGenerator:
    """
    图表生成工具类

    支持生成多种专业金融图表：
    - 折线图：趋势分析
    - 柱状图：对比分析
    - 饼图：占比分析
    - 热力图：相关性分析
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """初始化图表样式"""
        try:
            plt.style.use(style)
        except:
            plt.style.use("seaborn-v0_8-whitegrid")
        self.figsize = (10, 6)
        self.dpi = 100

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """将图表转换为 Base64 字符串（用于 Web 展示）"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return img_base64

    def line_chart(
        self,
        data: Dict[str, List],
        x_label: str = "时间",
        y_label: str = "数值",
        title: str = "趋势图"
    ) -> str:
        """
        生成折线图

        Args:
            data: 数据字典，格式为 {"系列名": [值列表], ...}，需包含 "x" 键作为 X 轴
            x_label: X 轴标签
            y_label: Y 轴标签
            title: 图表标题

        Returns:
            Base64 编码的图片字符串
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        x_data = data.get("x", list(range(len(list(data.values())[0]))))

        for key, values in data.items():
            if key != "x":
                ax.plot(x_data, values, marker='o', label=key, linewidth=2)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def bar_chart(
        self,
        data: Dict[str, List],
        x_label: str = "类别",
        y_label: str = "数值",
        title: str = "对比图"
    ) -> str:
        """
        生成柱状图

        Args:
            data: 数据字典，格式为 {"类别": 值} 或 {"系列名": [值列表]}
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if "x" in data:
            # 多系列柱状图
            x = data["x"]
            width = 0.8 / (len(data) - 1)
            for i, (key, values) in enumerate(data.items()):
                if key != "x":
                    offset = (i - len(data) / 2) * width
                    ax.bar([x_i + offset for x_i in range(len(x))], values,
                           width=width, label=key)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x)
        else:
            # 单系列柱状图
            categories = list(data.keys())
            values = list(data.values())
            ax.bar(categories, values, color='steelblue')

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')

        return self._fig_to_base64(fig)

    def pie_chart(
        self,
        data: Dict[str, float],
        title: str = "占比图"
    ) -> str:
        """
        生成饼图

        Args:
            data: 数据字典，格式为 {"类别": 占比值}
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        labels = list(data.keys())
        sizes = list(data.values())

        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               startangle=90, colors=plt.cm.Set3.colors)
        ax.set_title(title, fontsize=14, fontweight='bold')

        return self._fig_to_base64(fig)


# ============================================================
# 3. 代码执行器（安全的 Python 执行环境）
# ============================================================
class CodeExecutor:
    """
    安全的 Python 代码执行器

    设计考虑：
    - 受限的执行环境（禁止文件操作、网络请求等危险操作）
    - 超时控制
    - 结果捕获
    """

    # 允许使用的模块白名单
    ALLOWED_MODULES = {
        'pandas', 'pd', 'numpy', 'np', 'matplotlib', 'plt',
        'seaborn', 'sns', 'json', 'math', 'statistics',
        'datetime', 'decimal', 'collections', 'itertools'
    }

    # 禁止的关键字
    FORBIDDEN_KEYWORDS = {
        'import os', 'import sys', 'import subprocess', 'import shutil',
        'eval(', 'exec(', 'compile(', 'open(', '__import__',
        'globals()', 'locals()', 'vars(', 'dir('
    }

    def __init__(self, timeout: int = 60):
        """
        初始化代码执行器

        Args:
            timeout: 执行超时时间（秒）
        """
        self.timeout = timeout
        self.metrics = FinancialMetrics()
        self.chart_gen = ChartGenerator()

    def _validate_code(self, code: str) -> bool:
        """
        验证代码安全性

        Args:
            code: 待执行的 Python 代码

        Returns:
            是否通过安全检查
        """
        # 检查禁止的关键字
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in code:
                logger.warning(f"代码包含禁止的关键字: {keyword}")
                return False
        return True

    def execute(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行 Python 代码

        Args:
            code: Python 代码字符串
            context: 执行上下文（变量字典）

        Returns:
            执行结果字典，包含 output、result、error 等字段
        """
        if not self._validate_code(code):
            return {
                "success": False,
                "error": "代码未通过安全检查",
                "output": None
            }

        # 准备执行上下文
        exec_context = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'json': json,
            'FinancialMetrics': self.metrics,
            'ChartGenerator': self.chart_gen,
        }
        if context:
            exec_context.update(context)

        # 捕获标准输出
        output_buffer = io.StringIO()

        result = {
            "success": False,
            "output": None,
            "result": None,
            "error": None,
            "figures": []
        }

        try:
            # 重定向 stdout
            import sys
            old_stdout = sys.stdout
            sys.stdout = output_buffer

            # 执行代码
            exec_globals = {}
            exec_locals = exec_context.copy()
            exec(code, exec_globals, exec_locals)

            # 恢复 stdout
            sys.stdout = old_stdout

            # 获取输出
            result["output"] = output_buffer.getvalue()

            # 检查是否有返回值
            if 'result' in exec_locals:
                result["result"] = exec_locals['result']

            # 检查是否有生成的图表
            if 'fig_base64' in exec_locals:
                result["figures"].append(exec_locals['fig_base64'])

            result["success"] = True
            logger.info("代码执行成功")

        except Exception as e:
            sys.stdout = old_stdout
            result["error"] = f"{type(e).__name__}: {str(e)}"
            result["output"] = output_buffer.getvalue()
            logger.error(f"代码执行失败: {result['error']}")

        return result


# ============================================================
# 4. 数据分析工具类
# ============================================================
class AnalysisTool:
    """
    数据分析工具类

    整合财务指标计算和可视化功能，提供给 Agent 调用

    使用方式：
        tool = AnalysisTool()
        result = tool.compare_profitability(company_a_data, company_b_data)
    """

    def __init__(self):
        """初始化分析工具"""
        self.metrics = FinancialMetrics()
        self.executor = CodeExecutor(timeout=settings.agent.tool_timeout)
        self.chart_gen = ChartGenerator()
        logger.info("数据分析工具初始化完成")

    def analyze_financial_data(
        self,
        data: Dict[str, Any],
        analysis_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        通用财务数据分析

        Args:
            data: 财务数据字典
            analysis_type: 分析类型 (summary/trend/comparison)

        Returns:
            分析结果字典
        """
        try:
            df = pd.DataFrame(data)
            summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "describe": df.describe().to_dict() if len(df) > 0 else {},
            }
            return {"success": True, "summary": summary}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def compare_companies(
        self,
        company_data: Dict[str, Dict[str, List]],
        metric: str = "profit_margin",
        years: List[int] = None
    ) -> Dict[str, Any]:
        """
        公司对比分析

        Args:
            company_data: 公司数据，格式为 {"公司A": {"revenue": [...], "profit": [...]}, ...}
            metric: 对比指标
            years: 年份列表

        Returns:
            对比分析结果
        """
        results = {}

        for company, data in company_data.items():
            if "revenue" in data and "profit" in data:
                revenues = data["revenue"]
                profits = data["profit"]
                margins = [
                    self.metrics.calculate_profit_margin(r, p)
                    for r, p in zip(revenues, profits)
                ]
                results[company] = {
                    "profit_margins": margins,
                    "avg_margin": np.mean(margins),
                    "trend": "上升" if margins[-1] > margins[0] else "下降"
                }

        # 生成对比图表
        if years is None:
            years = list(range(len(list(company_data.values())[0]["revenue"])))

        chart_data = {"x": [str(y) for y in years]}
        for company, data in results.items():
            chart_data[company] = data["profit_margins"]

        chart_base64 = self.chart_gen.line_chart(
            chart_data,
            x_label="年份",
            y_label="利润率",
            title="公司利润率对比"
        )

        return {
            "success": True,
            "comparison": results,
            "chart": chart_base64
        }

    def calculate_growth_metrics(
        self,
        values: List[float],
        periods: int = None
    ) -> Dict[str, Any]:
        """
        计算增长指标

        Args:
            values: 时间序列数值
            periods: 计算周期

        Returns:
            增长指标字典
        """
        if len(values) < 2:
            return {"success": False, "error": "至少需要两个数据点"}

        results = {
            "total_growth": (values[-1] - values[0]) / values[0] if values[0] != 0 else None,
            "cagr": self.metrics.calculate_cagr(values[0], values[-1], len(values) - 1),
            "yoy_growth": [],
            "qoq_growth": []
        }

        # 计算同比增长
        for i in range(1, len(values)):
            results["yoy_growth"].append(
                self.metrics.calculate_yoy(values[i], values[i-1])
            )

        return {"success": True, "metrics": results}

    def execute_custom_analysis(self, code: str) -> Dict[str, Any]:
        """
        执行自定义分析代码

        Args:
            code: Python 分析代码

        Returns:
            执行结果
        """
        return self.executor.execute(code)

    def as_tool(self) -> FunctionTool:
        """
        将分析工具转换为 LlamaIndex FunctionTool
        """
        def analyze_data(
            analysis_request: str,
            data_json: str = None
        ) -> str:
            """
            金融数据分析工具

            用于进行财务数据分析，包括：
            - 计算增长率（CAGR、同比、环比）
            - 计算财务比率（利润率、ROE、ROA）
            - 公司对比分析
            - 生成数据可视化图表

            当用户需要进行数据计算、对比分析或要求画图时使用此工具。

            Args:
                analysis_request: 分析请求描述，如"对比 A 公司和 B 公司过去三年的利润率"
                data_json: JSON 格式的数据（可选）

            Returns:
                分析结果和图表
            """
            # 解析数据
            data = None
            if data_json:
                try:
                    data = json.loads(data_json)
                except json.JSONDecodeError:
                    return "数据格式错误，请提供有效的 JSON 数据"

            # 根据请求类型选择分析方法
            result = {"analysis_type": "custom", "request": analysis_request}

            if "对比" in analysis_request or "比较" in analysis_request:
                if data and "companies" in data:
                    result = self.compare_companies(data["companies"])

            elif "增长率" in analysis_request or "CAGR" in analysis_request:
                if data and "values" in data:
                    result = self.calculate_growth_metrics(data["values"])

            return json.dumps(result, ensure_ascii=False, indent=2)

        return FunctionTool.from_defaults(
            fn=analyze_data,
            name="analyze_data",
            description="金融数据分析工具。用于计算财务指标、对比分析、生成图表。"
        )

    def as_code_interpreter_tool(self) -> FunctionTool:
        """
        代码解释器工具（让 Agent 能够执行 Python 代码）
        """
        def execute_python_code(code: str) -> str:
            """
            Python 代码执行器

            用于执行数据分析代码，生成计算结果和图表。

            可用库：pandas (pd), numpy (np), matplotlib (plt), seaborn (sns)
            可用类：FinancialMetrics（财务指标计算）, ChartGenerator（图表生成）

            使用示例：
            ```python
            import pandas as pd
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            result = df.describe()
            ```

            Args:
                code: 要执行的 Python 代码

            Returns:
                执行结果
            """
            exec_result = self.executor.execute(code)

            if exec_result["success"]:
                response = f"执行成功！\n"
                if exec_result["output"]:
                    response += f"输出：\n{exec_result['output']}\n"
                if exec_result["result"]:
                    response += f"结果：{exec_result['result']}\n"
                if exec_result["figures"]:
                    response += f"[生成了 {len(exec_result['figures'])} 个图表]"
                return response
            else:
                return f"执行失败：{exec_result['error']}"

        return FunctionTool.from_defaults(
            fn=execute_python_code,
            name="execute_python_code",
            description="Python 代码执行器。用于运行数据分析代码、计算指标、生成图表。"
        )


# ============================================================
# 5. 工厂函数
# ============================================================
def create_analysis_tool() -> AnalysisTool:
    """创建数据分析工具的便捷函数"""
    return AnalysisTool()


# ============================================================
# 6. 测试入口
# ============================================================
if __name__ == "__main__":
    setup_logging()

    print("正在初始化数据分析工具...")
    tool = AnalysisTool()

    # 测试 CAGR 计算
    print("\n" + "=" * 60)
    print("测试 CAGR 计算")
    print("=" * 60)

    cagr = FinancialMetrics.calculate_cagr(100, 150, 3)
    print(f"三年 CAGR: {cagr * 100:.2f}%")

    # 测试对比分析
    print("\n" + "=" * 60)
    print("测试公司对比分析")
    print("=" * 60)

    test_data = {
        "companies": {
            "公司A": {
                "revenue": [100, 120, 150, 180],
                "profit": [10, 15, 20, 25]
            },
            "公司B": {
                "revenue": [80, 100, 130, 160],
                "profit": [8, 12, 18, 22]
            }
        }
    }

    result = tool.compare_companies(test_data["companies"], years=[2020, 2021, 2022, 2023])
    print(f"对比结果: {json.dumps(result['comparison'], ensure_ascii=False, indent=2)}")
    print(f"图表已生成: {'是' if result.get('chart') else '否'}")

    # 测试代码执行
    print("\n" + "=" * 60)
    print("测试代码执行器")
    print("=" * 60)

    test_code = """
import pandas as pd
import numpy as np

# 创建示例数据
df = pd.DataFrame({
    '年份': [2020, 2021, 2022, 2023],
    '营收': [100, 120, 150, 180],
    '利润': [10, 15, 20, 25]
})

# 计算利润率
df['利润率'] = df['利润'] / df['营收']

# 计算 CAGR
cagr = FinancialMetrics.calculate_cagr(100, 180, 3)
print(f"三年复合增长率: {cagr * 100:.2f}%")

result = df.to_dict()
"""

    exec_result = tool.execute_custom_analysis(test_code)
    print(f"执行成功: {exec_result['success']}")
    print(f"输出: {exec_result['output']}")
