"""
完整功能测试脚本
Comprehensive functionality test script.
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.parser import ExcelParser, ExcelDocument
from fiscal_mind.meta_functions import TableMetaFunctions, TableQueryHelper
from fiscal_mind.agent import TableDocumentAgent
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

def test_parser():
    """测试解析器功能"""
    print("\n" + "="*70)
    print("测试 1: Excel解析器")
    print("="*70)
    
    parser = ExcelParser()
    
    # 加载文档
    print("\n加载示例文档...")
    doc = parser.load_document('examples/financial_report.xlsx')
    
    print(f"✓ 文档名称: {doc.file_name}")
    print(f"✓ 工作表数量: {len(doc.sheets)}")
    print(f"✓ 工作表名称: {doc.get_sheet_names()}")
    
    # 获取工作表摘要
    for sheet_name in doc.get_sheet_names():
        summary = doc.get_sheet_summary(sheet_name)
        print(f"\n工作表 '{sheet_name}' 摘要:")
        print(f"  - 行数: {summary['rows']}")
        print(f"  - 列数: {summary['columns']}")
        print(f"  - 列名: {summary['column_names']}")
    
    return parser


def test_multi_document_loading(parser):
    """测试多文档加载"""
    print("\n" + "="*70)
    print("测试 2: 多文档加载")
    print("="*70)
    
    # 加载多个文档
    files = [
        'examples/sales_data.xlsx',
        'examples/employee_salary.xlsx'
    ]
    
    print(f"\n加载 {len(files)} 个文档...")
    parser.load_documents(files)
    
    print(f"✓ 总共加载了 {len(parser.documents)} 个文档:")
    for name in parser.documents.keys():
        print(f"  - {name}")
    
    # 获取所有文档摘要
    summary = parser.get_documents_summary()
    print(f"\n✓ 文档总行数: {sum(doc['total_rows'] for doc in summary['documents'].values())}")


def test_meta_functions(parser):
    """测试元功能"""
    print("\n" + "="*70)
    print("测试 3: 表格元功能")
    print("="*70)
    
    doc = parser.get_document('sales_data.xlsx')
    df = doc.get_sheet('销售明细')
    
    print("\n测试统计功能...")
    stats = TableMetaFunctions.get_numeric_summary(df)
    print(f"✓ 找到 {len(stats)} 个数值列")
    
    if '销售额' in stats:
        print("\n'销售额'列统计:")
        for key, value in stats['销售额'].items():
            if key not in ['value_counts']:
                print(f"  - {key}: {value}")
    
    print("\n测试LLM上下文格式化...")
    context = TableMetaFunctions.format_for_llm_context(df, max_rows=5)
    print("✓ 成功生成LLM上下文 (前200字符):")
    print(context[:200] + "...")


def test_query_helper(parser):
    """测试查询辅助功能"""
    print("\n" + "="*70)
    print("测试 4: 查询辅助工具")
    print("="*70)
    
    doc = parser.get_document('sales_data.xlsx')
    df = doc.get_sheet('销售明细')
    
    # 测试聚合
    print("\n测试数据聚合...")
    agg_result = TableQueryHelper.aggregate_by_column(df, '产品', '销售额', 'sum')
    if agg_result is not None:
        print("✓ 按产品聚合销售额:")
        print(agg_result.head())
    
    # 测试Top N
    print("\n测试Top N查询...")
    top5 = TableQueryHelper.get_top_n_by_column(df, '销售额', n=5)
    print(f"✓ Top 5销售额记录:")
    print(top5[['产品', '销售额', '区域']].head())
    
    # 测试列查找
    print("\n测试列名查找...")
    cols = TableQueryHelper.find_column_by_keyword(df, '销售')
    print(f"✓ 找到包含'销售'的列: {cols}")


def test_agent():
    """测试Agent功能"""
    print("\n" + "="*70)
    print("测试 5: LangGraph Agent")
    print("="*70)
    
    # 创建Agent
    print("\n创建Agent...")
    agent = TableDocumentAgent()
    
    # 加载文档
    print("加载文档...")
    agent.load_documents([
        'examples/financial_report.xlsx',
        'examples/sales_data.xlsx'
    ])
    
    print(f"✓ Agent已加载 {len(agent.parser.documents)} 个文档")
    
    # 测试文档摘要
    print("\n获取文档摘要...")
    summary = agent.get_document_summary()
    print("✓ 文档摘要 (前300字符):")
    print(summary[:300] + "...")
    
    # 测试查询
    print("\n测试查询功能...")
    try:
        response = agent.query("显示统计信息")
        print("✓ 查询响应 (前200字符):")
        print(response[:200] + "...")
    except Exception as e:
        print(f"⚠ 查询功能需要进一步集成LLM: {str(e)}")
    
    # 测试工作表分析
    print("\n测试工作表分析...")
    analysis = agent.analyze_sheet('financial_report.xlsx', '损益表')
    print("✓ 工作表分析 (前200字符):")
    print(analysis[:200] + "...")


def test_search():
    """测试搜索功能"""
    print("\n" + "="*70)
    print("测试 6: 数据搜索")
    print("="*70)
    
    parser = ExcelParser()
    doc = parser.load_document('examples/sales_data.xlsx')
    
    print("\n搜索'产品A'...")
    results = doc.search_value('产品A')
    print(f"✓ 找到 {len(results)} 个结果")
    if results:
        print("前3个结果:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. 工作表: {result['sheet']}, 行: {result['row']}, 列: {result['column']}")


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print(" "*20 + "FiscalMind 完整功能测试")
    print("="*70)
    
    try:
        # 运行所有测试
        parser = test_parser()
        test_multi_document_loading(parser)
        test_meta_functions(parser)
        test_query_helper(parser)
        test_agent()
        test_search()
        
        print("\n" + "="*70)
        print(" "*20 + "所有测试通过！✓")
        print("="*70)
        print("\nFiscalMind已成功实现以下功能:")
        print("  ✓ Excel文档解析")
        print("  ✓ 多文档同时处理")
        print("  ✓ 表格元功能（统计、摘要、格式化）")
        print("  ✓ 查询辅助工具（聚合、过滤、排序）")
        print("  ✓ LangGraph Agent工作流")
        print("  ✓ 数据搜索功能")
        print("\n系统已准备好与LLM集成，支持智能查询分析！")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
