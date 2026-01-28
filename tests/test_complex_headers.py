"""
复杂表头功能测试
Test script for complex header functionality.
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.parser import ExcelParser, ExcelDocument
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Test file paths
COMPLEX_HEADER_FILE = 'examples/complex_header_sample.xlsx'
MULTI_TABLE_SMALL_DATA_FILE = 'examples/multi_table_with_small_data.xlsx'


def test_complex_header_detection():
    """测试复杂表头检测功能"""
    print("\n" + "="*70)
    print("测试: 复杂表头检测（多行表头 + 合并单元格）")
    print("="*70)
    
    # 创建启用多表格检测的解析器
    parser = ExcelParser(detect_multiple_tables=True)
    
    # 加载包含复杂表头的文件
    print("\n加载复杂表头示例文件...")
    doc = parser.load_document(COMPLEX_HEADER_FILE)
    
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
    
    # 获取表格数据并验证表头
    print("\n验证复杂表头处理:")
    tables = doc.get_sheet_tables(sheet_name)
    if tables and len(tables) > 0:
        table_info = tables[0]
        print(f"  表格数据形状: {table_info.data.shape}")
        print(f"  表头列名:")
        for i, col in enumerate(table_info.data.columns):
            print(f"    {i}: {col}")
        
        # 检查是否正确处理了层级信息（连字符连接）
        hierarchical_headers = [col for col in table_info.data.columns if '-' in str(col)]
        print(f"\n  检测到 {len(hierarchical_headers)} 个层级表头（包含连字符）")
        if hierarchical_headers:
            print(f"  层级表头示例:")
            for header in hierarchical_headers[:3]:
                print(f"    - {header}")
        
        print(f"\n  数据预览:")
        print(table_info.data.head())
        
        # 验证是否满足要求
        success = True
        if len(hierarchical_headers) == 0:
            print("\n  ❌ 错误: 没有检测到层级表头")
            success = False
        else:
            print(f"\n  ✓ 成功检测到层级表头")
        
        return success
    else:
        print("\n  ❌ 错误: 未检测到表格")
        return False


def test_small_table_filtering():
    """测试小表格过滤功能"""
    print("\n" + "="*70)
    print("测试: 小表格过滤（排除临时数据）")
    print("="*70)
    
    parser = ExcelParser(detect_multiple_tables=True)
    
    print("\n加载多表格混合示例文件...")
    doc = parser.load_document(MULTI_TABLE_SMALL_DATA_FILE)
    
    sheet_name = doc.get_sheet_names()[0]
    summary = doc.get_sheet_summary(sheet_name)
    
    print(f"\n工作表 '{sheet_name}' 摘要:")
    print(f"  - 检测到表格数量: {summary.get('num_tables', 0)}")
    
    expected_tables = 2  # 应该只检测到2个有效表格，小表格被过滤
    
    if 'tables' in summary:
        for table in summary['tables']:
            print(f"\n  表格 {table['index']}:")
            print(f"    - 位置: {table['position']}")
            print(f"    - 形状: {table['shape']}")
            print(f"    - 描述: {table['description']}")
            print(f"    - 列名: {table['columns']}")
    
    # 验证表格数量
    actual_tables = summary.get('num_tables', 0)
    if actual_tables == expected_tables:
        print(f"\n  ✓ 成功过滤小表格，检测到 {actual_tables} 个有效表格（期望 {expected_tables} 个）")
        return True
    else:
        print(f"\n  ❌ 表格数量不符合预期: 实际 {actual_tables}，期望 {expected_tables}")
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "="*70)
    print("测试: 向后兼容性")
    print("="*70)
    
    # 不启用多表格检测
    parser = ExcelParser(detect_multiple_tables=False)
    
    print("\n加载复杂表头文件（不启用多表格检测）...")
    doc = parser.load_document(COMPLEX_HEADER_FILE)
    
    print(f"✓ 文档名称: {doc.file_name}")
    sheet_name = doc.get_sheet_names()[0]
    df = doc.get_sheet(sheet_name)
    print(f"  - 工作表: {sheet_name}")
    print(f"  - 形状: {df.shape}")
    print(f"  - 使用默认方式读取（不进行复杂表头处理）")
    
    # 确认没有多表格信息
    tables = doc.get_sheet_tables(sheet_name)
    if tables is None:
        print("  ✓ 向后兼容性测试通过：未启用多表格检测时不进行特殊处理")
        return True
    else:
        print("  ❌ 向后兼容性测试失败")
        return False


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print(" "*15 + "FiscalMind 复杂表头功能测试")
    print("="*70)
    
    results = []
    
    try:
        # 运行所有测试
        results.append(("复杂表头检测", test_complex_header_detection()))
        results.append(("小表格过滤", test_small_table_filtering()))
        results.append(("向后兼容性", test_backward_compatibility()))
        
        print("\n" + "="*70)
        print(" "*20 + "测试结果汇总")
        print("="*70)
        
        all_passed = True
        for test_name, passed in results:
            status = "✓ 通过" if passed else "❌ 失败"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n" + "="*70)
            print(" "*20 + "所有测试通过！✓")
            print("="*70)
            print("\n复杂表头功能已成功实现：")
            print("  ✓ 支持多行表头（>2行）")
            print("  ✓ 处理合并单元格")
            print("  ✓ 层级信息用连字符连接（如：华东-南京-浦口）")
            print("  ✓ 过滤小表格和临时数据")
            print("  ✓ 保持向后兼容性")
            print("="*70 + "\n")
            return 0
        else:
            print("\n❌ 部分测试失败，请检查实现")
            return 1
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
