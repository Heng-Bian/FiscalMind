"""
FiscalMind主程序入口
Main entry point for FiscalMind table document agent.
"""

import argparse
import logging
from pathlib import Path
from typing import List

from fiscal_mind.agent import TableDocumentAgent
from fiscal_mind.parser import ExcelParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FiscalMind - 面向财务BP的表格分析Agent"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Excel文件路径列表"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="进入交互模式"
    )
    
    args = parser.parse_args()
    
    # 验证文件
    file_paths = []
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            continue
        if not path.suffix.lower() in ['.xlsx', '.xls']:
            logger.error(f"不支持的文件格式: {file_path}")
            continue
        file_paths.append(str(path))
    
    if not file_paths:
        logger.error("没有有效的Excel文件")
        return
    
    # 创建Agent
    logger.info("初始化FiscalMind Agent...")
    agent = TableDocumentAgent()
    
    # 加载文档
    logger.info(f"加载 {len(file_paths)} 个Excel文档...")
    agent.load_documents(file_paths)
    
    # 显示文档摘要
    print("\n" + "="*60)
    print("文档加载完成！")
    print("="*60)
    print(agent.get_document_summary())
    print("="*60 + "\n")
    
    if args.interactive:
        # 交互模式
        interactive_mode(agent)
    else:
        # 显示基本信息
        show_basic_info(agent)


def interactive_mode(agent: TableDocumentAgent):
    """交互模式"""
    print("进入交互模式。输入 'exit' 或 'quit' 退出。")
    print("输入 'help' 查看帮助信息。\n")
    
    while True:
        try:
            query = input("FiscalMind> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("再见！")
                break
            
            if query.lower() == 'help':
                show_help()
                continue
            
            # 处理查询
            print("\n处理中...")
            response = agent.query(query)
            print("\n" + "="*60)
            print("回答:")
            print("="*60)
            print(response)
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}")
            print(f"错误: {str(e)}\n")


def show_basic_info(agent: TableDocumentAgent):
    """显示基本信息"""
    print("\n使用 --interactive 或 -i 参数进入交互模式")
    print("示例: python -m fiscal_mind.main file1.xlsx file2.xlsx -i\n")


def show_help():
    """显示帮助信息"""
    help_text = """
可用命令:
    help        - 显示此帮助信息
    exit/quit   - 退出程序

查询示例:
    - "显示所有文档的统计信息"
    - "搜索关键词"
    - "显示文档摘要"
    
提示: 您可以用中文或英文提问。
    """
    print(help_text)


if __name__ == "__main__":
    main()
