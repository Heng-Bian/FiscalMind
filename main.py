"""
FiscalMind主程序入口 - 面向财务BP的专业问答系统
Main entry point for FiscalMind - Professional Q&A system for Financial Business Partners.

本程序自动加载当前目录的所有Excel文档，并使用PRR Agent提供专业的财务分析问答能力。
特别适合回答诸如"哪个大区表现更好"这类复杂的财务对比分析问题。
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI

from fiscal_mind.prr_agent import PRRAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def find_excel_files(directory: str = ".") -> List[str]:
    """
    在指定目录中查找所有Excel文件
    
    Args:
        directory: 要搜索的目录路径，默认为当前目录
        
    Returns:
        Excel文件路径列表
    """
    excel_files = []
    search_path = Path(directory).resolve()
    
    # 支持的Excel文件扩展名
    excel_extensions = {'.xlsx', '.xls'}
    
    # 遍历目录查找Excel文件
    for file_path in search_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in excel_extensions:
            excel_files.append(str(file_path))
    
    return sorted(excel_files)


def interactive_mode(agent: PRRAgent):
    """
    交互式问答模式
    
    用户可以持续输入问题，Agent将使用PRR架构(Plan-ReAct-Reflect)来分析并回答。
    特别适合财务BP的专业问题，如：
    - "哪个大区表现更好？"
    - "哪个产品的利润率最高？"
    - "销售额增长最快的区域是哪个？"
    """
    print("\n" + "="*80)
    print("欢迎使用 FiscalMind - 财务BP专业问答系统")
    print("="*80)
    print("\n系统特性:")
    print("  • 使用PRR架构 (Plan-ReAct-Reflect) 进行智能分析")
    print("  • 支持复杂的财务对比分析问题")
    print("  • 自动分解问题并逐步推理求解")
    print("\n可以尝试的问题:")
    print("  - 哪个大区表现更好？")
    print("  - 哪个产品的利润率最高？")
    print("  - 销售额增长最快的区域是哪个？")
    print("  - 对比各区域的销售额和利润")
    print("\n输入 'exit' 或 'quit' 退出，输入 'help' 查看帮助\n")
    
    while True:
        try:
            query = input("FiscalMind> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n感谢使用 FiscalMind！再见！")
                break
            
            if query.lower() in ['help', 'h', '?']:
                show_help()
                continue
            
            # 使用PRR Agent处理查询
            print("\n正在分析您的问题...")
            print("-" * 80)
            
            response = agent.query(query)
            
            print("\n回答:")
            print("="*80)
            print(response)
            print("="*80 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\n\n感谢使用 FiscalMind！再见！")
            break
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}", exc_info=True)
            print(f"\n错误: {str(e)}\n")
            print("请尝试重新表述您的问题，或输入 'help' 查看帮助\n")


def show_help():
    """显示帮助信息"""
    help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                          FiscalMind 帮助文档                                ║
╚════════════════════════════════════════════════════════════════════════════╝

【可用命令】
  help / h / ?    - 显示此帮助信息
  exit / quit / q - 退出程序

【PRR架构说明】
  FiscalMind使用Plan-ReAct-Reflect架构来处理复杂的财务分析问题：
  
  1. Plan (计划)   - 将您的问题分解为可执行的步骤
  2. ReAct (推理)  - 执行推理和行动，调用工具获取数据
  3. Reflect (反思) - 评估执行结果，决定下一步行动
  
  这种架构特别适合回答需要多步推理的复杂财务问题。

【问题示例】
  区域对比分析:
    • 哪个大区表现更好？
    • 哪个区域的利润率最高？
    • 销售额增长最快的区域是哪个？
    • 对比各大区的销售额和利润
  
  产品分析:
    • 哪个产品的销售额最高？
    • 哪个产品的利润率最好？
    • 产品销售趋势如何？
  
  综合分析:
    • 找出综合表现最好的大区
    • 对比产品和区域的整体表现
    • 分析销售数据的主要趋势

【提示】
  • 问题越具体，回答越准确
  • 可以使用中文或英文提问
  • 系统会自动识别相关的数据表和指标
  • 对于复杂问题，系统会分步骤进行分析

╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(help_text)


def main():
    """主函数"""
    print("\n" + "="*80)
    print("FiscalMind 启动中...")
    print("="*80)
    
    # 查找当前目录的所有Excel文件
    current_dir = os.getcwd()
    print(f"\n正在搜索目录: {current_dir}")
    excel_files = find_excel_files(current_dir)
    
    if not excel_files:
        print("\n⚠️  警告: 当前目录未找到任何Excel文件 (.xlsx, .xls)")
        print("\n建议:")
        print("  1. 将Excel文件放置在当前目录")
        
        # 检查examples目录是否存在
        examples_dir = Path(__file__).parent / "examples"
        if examples_dir.exists() and (examples_dir / "create_samples.py").exists():
            print("  2. 或者使用示例数据: python examples/create_samples.py")
        else:
            print("  2. 或者创建您自己的Excel数据文件")
        
        print("  3. 然后重新运行本程序")
        return
    
    print(f"\n✓ 找到 {len(excel_files)} 个Excel文件:")
    for i, file_path in enumerate(excel_files, 1):
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / 1024  # KB
        print(f"  {i}. {file_name} ({file_size:.1f} KB)")
    
    # 创建PRR Agent
    print("\n正在初始化 PRR Agent (Plan-ReAct-Reflect)...")
    llm = ChatOpenAI(model="qwen-vl-max-2025-08-13",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        api_key='YOUR_API_KEY_HERE')
    agent = PRRAgent(llm=llm)
    
    # 加载Excel文档
    print(f"正在加载 {len(excel_files)} 个Excel文档...")
    try:
        agent.load_documents(excel_files)
        print("✓ 文档加载完成！")
    except Exception as e:
        logger.error(f"加载文档时出错: {str(e)}")
        print(f"\n❌ 错误: 无法加载Excel文档")
        print(f"详细信息: {str(e)}")
        return
    
    # 显示文档摘要
    try:
        print("\n" + "="*80)
        print("文档摘要")
        print("="*80)
        summary = agent.get_document_summary()
        print(summary)
        print("="*80)
    except Exception as e:
        logger.warning(f"获取文档摘要时出错: {str(e)}")
    
    # 进入交互模式
    interactive_mode(agent)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"\n❌ 程序出错: {str(e)}")
        sys.exit(1)
