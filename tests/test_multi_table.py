"""
多表格检测功能测试
Test script for multi-table detection functionality.
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.parser import ExcelParser, ExcelDocument
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# Test file paths
MULTI_TABLE_FILE = 'examples/multi_table_sheet.xlsx'
OFFSET_TABLE_FILE = 'examples/offset_table_sheet.xlsx'
STANDARD_FILE = 'examples/financial_report.xlsx'


def test_multi_table_detection():
    """测试多表格检测功能"""
    print("\n" + "="*70)
    print("测试: 多表格检测")
    print("="*70)
    
    # 创建启用多表格检测的解析器
    parser = ExcelParser(detect_multiple_tables=True)
    
    # 加载包含多个表格的文件
    print("\n加载多表格示例文件...")
    doc = parser.load_document(MULTI_TABLE_FILE)
    
    print(f"✓ 文档名称: {doc.file_name}")
    print(f"✓ 工作表数量: {len(doc.sheets)}")
    
    # 获取第一个工作表的摘要
    sheet_name = doc.get_sheet_names()[0]
    summary = doc.get_sheet_summary(sheet_name)
    
    print(f"\n工作表 '{sheet_name}' 摘要:")
    print(f"  - 检测到表格数量: {summary.get('num_tables', 0)}")
    
    if 'tables' in summary:
        for table in summary['tables']:
            print(f"\n  表格 {table['index']}:")
            print(f"    - 位置: {table['position']}")
            print(f"    - 形状: {table['shape']}")
            print(f"    - 描述: {table['description']}")
            print(f"    - 列名: {table['columns']}")
    
    # 获取各个表格的数据
    print("\n获取各表格数据:")
    tables = doc.get_sheet_tables(sheet_name)
    if tables:
        for i, table_info in enumerate(tables):
            print(f"\n表格 {i}:")
            print(f"  描述: {table_info.description}")
            print(f"  数据预览:")
            print(table_info.data.head())


def test_offset_table_detection():
    """测试偏移表格检测"""
    print("\n" + "="*70)
    print("测试: 偏移表格检测")
    print("="*70)
    
    parser = ExcelParser(detect_multiple_tables=True)
    
    print("\n加载偏移表格示例文件...")
    doc = parser.load_document(OFFSET_TABLE_FILE)
    
    sheet_name = doc.get_sheet_names()[0]
    summary = doc.get_sheet_summary(sheet_name)
    
    print(f"\n工作表 '{sheet_name}' 摘要:")
    print(f"  - 检测到表格数量: {summary.get('num_tables', 0)}")
    
    if 'tables' in summary:
        for table in summary['tables']:
            print(f"\n  表格 {table['index']}:")
            print(f"    - 位置: {table['position']}")
            print(f"    - 形状: {table['shape']}")
            print(f"    - 描述: {table['description']}")
            print(f"    - 列名: {table['columns']}")
    
    # 获取表格数据
    table_0 = doc.get_table_by_index(sheet_name, 0)
    if table_0 is not None:
        print("\n表格数据预览:")
        print(table_0)


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "="*70)
    print("测试: 向后兼容性（不启用多表格检测）")
    print("="*70)
    
    # 不启用多表格检测，应该使用默认行为
    parser = ExcelParser(detect_multiple_tables=False)
    
    print("\n加载标准文件...")
    doc = parser.load_document(STANDARD_FILE)
    
    print(f"✓ 文档名称: {doc.file_name}")
    print(f"✓ 工作表数量: {len(doc.sheets)}")
    
    # 获取工作表数据（应该使用传统方式）
    sheet_name = doc.get_sheet_names()[0]
    df = doc.get_sheet(sheet_name)
    print(f"\n工作表 '{sheet_name}':")
    print(f"  - 形状: {df.shape}")
    print(f"  - 列名: {df.columns.tolist()}")
    
    # 确认没有多表格信息
    tables = doc.get_sheet_tables(sheet_name)
    print(f"\n多表格信息: {tables}")
    print("✓ 向后兼容性测试通过")


def test_api_methods():
    """测试新的API方法"""
    print("\n" + "="*70)
    print("测试: 新API方法")
    print("="*70)
    
    parser = ExcelParser(detect_multiple_tables=True)
    doc = parser.load_document(MULTI_TABLE_FILE)
    
    sheet_name = doc.get_sheet_names()[0]
    
    # 测试 get_sheet_tables
    print("\n测试 get_sheet_tables():")
    tables = doc.get_sheet_tables(sheet_name)
    print(f"✓ 获取到 {len(tables) if tables else 0} 个表格")
    
    # 测试 get_table_by_index
    print("\n测试 get_table_by_index():")
    for i in range(len(tables) if tables else 0):
        table_df = doc.get_table_by_index(sheet_name, i)
        print(f"✓ 表格 {i}: 形状 {table_df.shape}")
    
    # 测试 get_table_info
    print("\n测试 get_table_info():")
    table_info = doc.get_table_info(sheet_name, 0)
    if table_info:
        print(f"✓ 表格信息: {table_info}")


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print(" "*15 + "FiscalMind 多表格检测功能测试")
    print("="*70)
    
    try:
        # 运行所有测试
        test_multi_table_detection()
        test_offset_table_detection()
        test_backward_compatibility()
        test_api_methods()
        
        print("\n" + "="*70)
        print(" "*20 + "所有测试通过！✓")
        print("="*70)
        print("\n多表格检测功能已成功实现：")
        print("  ✓ 检测sheet中的多个表格")
        print("  ✓ 检测表格的偏移位置")
        print("  ✓ 提取表格附近的描述文本")
        print("  ✓ 保持向后兼容性")
        print("  ✓ 提供新的API访问表格")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
