# -*- coding: utf-8 -*-
"""
============================================================
主程序入口 (Main Entry Point)
============================================================
本模块提供金融研报分析 Agent 的命令行交互界面

功能：
- 终端交互式对话
- 支持多轮追问
- 命令控制（退出、重置、状态查看等）
- 丰富的输出格式

使用方式：
    python main.py

    或指定数据目录：
    python main.py --data-dir ./my_reports
============================================================
"""

import argparse
import sys
import json
from pathlib import Path

# 富文本终端输出（可选）
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 本地模块
from config import settings, setup_logging, check_environment
from agent import create_agent, FinancialAnalysisAgent

# 初始化日志
logger = setup_logging()


# ============================================================
# 1. 命令行参数解析
# ============================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="金融研报分析 Agent - 命令行交互模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py                           # 使用默认配置启动
    python main.py --data-dir ./reports      # 指定研报目录
    python main.py --no-rerank               # 禁用重排序
    python main.py --quiet                   # 安静模式

交互命令:
    /quit, /exit     - 退出程序
    /reset           - 重置对话记忆
    /status          - 查看 Agent 状态
    /history         - 查看对话历史
    /help            - 显示帮助
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=settings.rag.data_dir,
        help=f"研报 PDF 目录路径（默认: {settings.rag.data_dir}）"
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="禁用重排序功能"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式，减少输出"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细模式，显示更多信息"
    )

    return parser.parse_args()


# ============================================================
# 2. 终端 UI 类
# ============================================================
class TerminalUI:
    """
    终端用户界面

    提供美观的命令行交互体验
    """

    def __init__(self, use_rich: bool = True):
        """
        初始化终端 UI

        Args:
            use_rich: 是否使用 Rich 库美化输出
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()

    def print_welcome(self):
        """打印欢迎信息"""
        if self.use_rich:
            self.console.print(Panel.fit(
                "[bold cyan]金融研报分析 Agent[/bold cyan]\n"
                "[dim]Financial Report Analysis Agent[/dim]\n\n"
                "基于 LlamaIndex + DeepSeek + BGE-M3\n"
                "支持研报问答和数据分析",
                border_style="cyan"
            ))
        else:
            print("\n" + "=" * 60)
            print("       金融研报分析 Agent")
            print("       Financial Report Analysis Agent")
            print("=" * 60)
            print("基于 LlamaIndex + DeepSeek + BGE-M3")
            print("支持研报问答和数据分析")
            print("=" * 60 + "\n")

    def print_status(self, status: dict):
        """打印 Agent 状态"""
        if self.use_rich:
            self.console.print("\n[bold]Agent 状态:[/bold]")
            for key, value in status.items():
                self.console.print(f"  • {key}: [green]{value}[/green]")
        else:
            print("\nAgent 状态:")
            for key, value in status.items():
                print(f"  • {key}: {value}")

    def print_help(self):
        """打印帮助信息"""
        help_text = """
可用命令:
  /quit, /exit     退出程序
  /reset           重置对话记忆
  /status          查看 Agent 状态
  /history         查看对话历史
  /help            显示此帮助

使用示例:
  • 研报问答: "概括 2023 年新能源汽车的出海情况"
  • 数据分析: "对比 A 公司和 B 公司过去三年的利润率"
  • 代码执行: "用 Python 计算复合增长率"
        """
        if self.use_rich:
            self.console.print(Panel(help_text, title="[bold]帮助[/bold]", border_style="yellow"))
        else:
            print(help_text)

    def print_user_message(self, message: str):
        """打印用户消息"""
        if self.use_rich:
            self.console.print(f"\n[bold blue]用户:[/bold blue] {message}")
        else:
            print(f"\n用户: {message}")

    def print_agent_response(self, response: str):
        """打印 Agent 回复"""
        if self.use_rich:
            self.console.print(f"\n[bold green]Agent:[/bold green]")
            # 尝试解析为 Markdown
            try:
                md = Markdown(response)
                self.console.print(md)
            except:
                self.console.print(response)
        else:
            print(f"\nAgent: {response}")

    def print_error(self, error: str):
        """打印错误信息"""
        if self.use_rich:
            self.console.print(f"\n[bold red]错误:[/bold red] {error}")
        else:
            print(f"\n错误: {error}")

    def print_info(self, info: str):
        """打印提示信息"""
        if self.use_rich:
            self.console.print(f"\n[dim]{info}[/dim]")
        else:
            print(f"\n{info}")

    def print_divider(self):
        """打印分隔线"""
        if self.use_rich:
            self.console.print("\n" + "─" * 60 + "\n")
        else:
            print("\n" + "-" * 60 + "\n")


# ============================================================
# 3. 交互式会话
# ============================================================
class InteractiveSession:
    """
    交互式会话管理

    处理用户输入、命令解析、Agent 调用
    """

    def __init__(
        self,
        agent: FinancialAnalysisAgent,
        ui: TerminalUI,
        verbose: bool = False
    ):
        """
        初始化交互会话

        Args:
            agent: 金融分析 Agent 实例
            ui: 终端 UI 实例
            verbose: 是否显示详细信息
        """
        self.agent = agent
        self.ui = ui
        self.verbose = verbose
        self.running = True

    def process_command(self, command: str) -> bool:
        """
        处理命令

        Args:
            command: 用户输入的命令（以 / 开头）

        Returns:
            是否继续运行
        """
        cmd = command.lower().strip()

        if cmd in ["/quit", "/exit", "quit", "exit"]:
            self.ui.print_info("再见！感谢使用金融研报分析 Agent")
            self.running = False
            return False

        elif cmd == "/reset":
            self.agent.reset_memory()
            self.ui.print_info("对话记忆已重置")

        elif cmd == "/status":
            status = self.agent.get_status()
            self.ui.print_status(status)

        elif cmd == "/history":
            history = self.agent.get_chat_history()
            if not history:
                self.ui.print_info("暂无对话历史")
            else:
                self.ui.print_info(f"对话历史 ({len(history)} 条):")
                for i, msg in enumerate(history, 1):
                    role = "用户" if msg["role"] == "user" else "Agent"
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    print(f"  {i}. [{role}] {content}")

        elif cmd == "/help":
            self.ui.print_help()

        else:
            self.ui.print_error(f"未知命令: {command}。输入 /help 查看可用命令")

        return True

    def process_message(self, message: str) -> str:
        """
        处理用户消息并获取 Agent 回复

        Args:
            message: 用户消息

        Returns:
            Agent 回复
        """
        try:
            response = self.agent.chat(message)
            return response
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            raise

    def run(self):
        """运行交互会话"""
        self.ui.print_welcome()
        self.ui.print_info("输入 /help 查看可用命令，或直接输入问题开始对话\n")

        while self.running:
            try:
                # 获取用户输入
                user_input = input(">>> ").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    self.process_command(user_input)
                    continue

                # 处理消息
                self.ui.print_user_message(user_input)

                response = self.process_message(user_input)
                self.ui.print_agent_response(str(response))

                if self.verbose:
                    self.ui.print_divider()

            except KeyboardInterrupt:
                self.ui.print_info("\n\n已中断。输入 /quit 退出")
                continue

            except EOFError:
                self.ui.print_info("\n再见！")
                break

            except Exception as e:
                self.ui.print_error(str(e))
                if self.verbose:
                    import traceback
                    traceback.print_exc()


# ============================================================
# 4. 主函数
# ============================================================
def main():
    """
    主函数入口

    流程：
    1. 解析命令行参数
    2. 检查环境配置
    3. 初始化 Agent
    4. 启动交互会话
    """
    # 解析参数
    args = parse_args()

    # 检查环境
    print("正在检查运行环境...")
    env_status = check_environment()

    if not env_status["api_key"]:
        print("\n警告: 未配置 DEEPSEEK_API_KEY 环境变量")
        print("请设置: export DEEPSEEK_API_KEY='your-api-key'")
        print("或在 .env 文件中配置")

    if not env_status["data_dir"]:
        print(f"\n提示: 数据目录不存在，将创建: {args.data_dir}")
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if env_status.get("cuda_available"):
        print(f"✓ CUDA 可用: {env_status.get('cuda_device', 'Unknown')}")
    else:
        print("✓ 将使用 CPU 运行（建议使用 GPU 加速）")

    # 初始化 UI
    ui = TerminalUI(use_rich=not args.quiet)

    # 初始化 Agent
    print("\n正在初始化 Agent...")
    print("（首次运行可能需要下载模型，请耐心等待）\n")

    try:
        agent = create_agent(
            data_dir=args.data_dir,
            enable_rerank=not args.no_rerank,
            verbose=args.verbose
        )
    except Exception as e:
        ui.print_error(f"Agent 初始化失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # 启动交互会话
    session = InteractiveSession(
        agent=agent,
        ui=ui,
        verbose=args.verbose
    )

    try:
        session.run()
    except Exception as e:
        ui.print_error(f"会话异常: {e}")
        sys.exit(1)


# ============================================================
# 5. 程序入口
# ============================================================
if __name__ == "__main__":
    main()
