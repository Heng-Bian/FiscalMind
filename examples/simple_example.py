"""
简单示例 - 演示如何使用FiscalMind
Simple example demonstrating FiscalMind usage.
"""

from fiscal_mind.parser import ExcelParser
from fiscal_mind.meta_functions import TableMetaFunctions, TableQueryHelper
from fiscal_mind.agent import TableDocumentAgent
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

def example1_basic_parsing():
    """示例1: 基础解析功能"""
    print("\n" + "="*60)
    print("示例1: 基础Excel解析")
    print("="*60)
    
    # 创建解析器
    parser = ExcelParser()
    
    # 这里使用一个假设的路径，实际使用时需要先运行 create_samples.py
    # doc = parser.load_document('examples/financial_report.xlsx')
    # print(f"\n已加载文档，包含工作表: {doc.get_sheet_names()}")
    
    print("\n提示: 请先运行 'python examples/create_samples.py' 创建示例数据")
    print("然后取消注释上面的代码运行此示例")


def example2_meta_functions():
    """示例2: 元功能演示"""
    print("\n" + "="*60)
    print("示例2: 表格元功能")
    print("="*60)
    
    # 创建示例DataFrame
    import pandas as pd
    
    sample_data = {
        '月份': ['2024-01', '2024-02', '2024-03'],
        '收入': [100000, 120000, 115000],
        '成本': [60000, 72000, 69000],
        '利润': [40000, 48000, 46000]
    }
    df = pd.DataFrame(sample_data)
    
    print("\n示例数据:")
    print(df)
    
    # 获取表格结构
    print("\n表格结构:")
    schema = TableMetaFunctions.get_table_schema(df)
    for key, value in schema.items():
        print(f"  {key}: {value}")
    
    # 获取列统计
    print("\n'收入'列统计:")
    stats = TableMetaFunctions.get_column_statistics(df, '收入')
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 格式化为LLM上下文
    print("\nLLM上下文格式:")
    context = TableMetaFunctions.format_for_llm_context(df)
    print(context)


def example3_query_helper():
    """示例3: 查询辅助工具"""
    print("\n" + "="*60)
    print("示例3: 查询辅助工具")
    print("="*60)
    
    import pandas as pd
    
    # 创建销售数据示例
    sales_data = {
        '产品': ['产品A', '产品B', '产品A', '产品C', '产品B', '产品A'],
        '区域': ['华北', '华东', '华南', '华北', '华南', '华东'],
        '销售额': [10000, 15000, 12000, 8000, 16000, 11000],
        '数量': [100, 150, 120, 80, 160, 110]
    }
    df = pd.DataFrame(sales_data)
    
    print("\n原始数据:")
    print(df)
    
    # 按产品聚合销售额
    print("\n按产品聚合销售额:")
    agg_result = TableQueryHelper.aggregate_by_column(df, '产品', '销售额', 'sum')
    print(agg_result)
    
    # 获取Top 3销售额
    print("\nTop 3销售额:")
    top3 = TableQueryHelper.get_top_n_by_column(df, '销售额', n=3)
    print(top3)
    
    # 按关键词查找列
    print("\n查找包含'销售'的列:")
    cols = TableQueryHelper.find_column_by_keyword(df, '销售')
    print(f"  找到列: {cols}")


def example4_agent_workflow():
    """示例4: Agent工作流"""
    print("\n" + "="*60)
    print("示例4: Agent工作流（简化版）")
    print("="*60)
    
    # 创建Agent
    agent = TableDocumentAgent()
    
    print("\nAgent初始化完成")
    print("Agent工作流包含以下节点:")
    print("  1. load_context - 加载文档上下文")
    print("  2. analyze_query - 分析用户查询")
    print("  3. execute_query - 执行查询操作")
    print("  4. format_response - 格式化响应")
    
    print("\n提示: 使用以下方式加载文档并查询:")
    print("  agent.load_documents(['file1.xlsx', 'file2.xlsx'])")
    print("  response = agent.query('显示统计信息')")


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*20 + "FiscalMind 示例演示")
    print("="*70)
    
    # 运行所有示例
    example1_basic_parsing()
    example2_meta_functions()
    example3_query_helper()
    example4_agent_workflow()
    
    print("\n" + "="*70)
    print("所有示例演示完成！")
    print("\n要查看完整功能，请:")
    print("1. 运行 'python examples/create_samples.py' 创建示例Excel文件")
    print("2. 运行 'python -m fiscal_mind.main examples/*.xlsx -i' 进入交互模式")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
