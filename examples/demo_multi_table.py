"""
多表格检测实际应用示例
Practical example of multi-table detection feature.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.parser import ExcelParser
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def demo_multi_table_detection():
    """演示多表格检测功能"""
    
    print("\n" + "="*80)
    print(" "*20 + "多表格检测功能演示")
    print("="*80)
    
    # 1. 创建启用多表格检测的解析器
    print("\n【步骤1】创建启用多表格检测的解析器")
    parser = ExcelParser(detect_multiple_tables=True)
    print("✓ 解析器已创建，多表格检测已启用")
    
    # 2. 加载包含多个表格的Excel文件
    print("\n【步骤2】加载包含多个表格的Excel文件")
    doc = parser.load_document('examples/multi_table_sheet.xlsx')
    print(f"✓ 文件加载成功: {doc.file_name}")
    
    # 3. 获取工作表信息
    print("\n【步骤3】分析工作表结构")
    sheet_name = doc.get_sheet_names()[0]
    summary = doc.get_sheet_summary(sheet_name)
    
    num_tables = summary.get('num_tables', 0)
    print(f"✓ 工作表 '{sheet_name}' 包含 {num_tables} 个表格")
    
    # 4. 显示每个表格的详细信息
    print("\n【步骤4】查看各表格详细信息")
    print("-" * 80)
    
    for table_info in summary.get('tables', []):
        print(f"\n表格 #{table_info['index']} - {table_info['description'] or '无描述'}")
        print(f"  位置: {table_info['position']}")
        print(f"  大小: {table_info['shape'][0]} 行 x {table_info['shape'][1]} 列")
        print(f"  列名: {', '.join(table_info['columns'])}")
    
    # 5. 访问和使用具体表格的数据
    print("\n【步骤5】访问和处理表格数据")
    print("-" * 80)
    
    for i in range(num_tables):
        table_data = doc.get_table_by_index(sheet_name, i)
        table_info = doc.get_table_info(sheet_name, i)
        
        print(f"\n处理表格 #{i}: {table_info.description or '无描述'}")
        print(f"数据预览（前3行）：")
        print(table_data.head(3).to_string(index=False))
        
        # 进行简单的统计分析
        numeric_cols = table_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\n数值列统计：")
            for col in numeric_cols:
                print(f"  {col}: 总和={table_data[col].sum():.0f}, "
                      f"平均={table_data[col].mean():.2f}, "
                      f"最大={table_data[col].max():.0f}")
    
    # 6. 演示偏移表格检测
    print("\n" + "="*80)
    print("\n【步骤6】演示偏移表格检测")
    doc2 = parser.load_document('examples/offset_table_sheet.xlsx')
    sheet2_name = doc2.get_sheet_names()[0]
    
    table_info = doc2.get_table_info(sheet2_name, 0)
    print(f"✓ 检测到偏移表格")
    print(f"  起始位置: 行{table_info.start_row}, 列{table_info.start_col} (非A1)")
    print(f"  描述: {table_info.description}")
    print(f"\n表格数据：")
    print(table_info.data.to_string(index=False))
    
    # 7. 总结
    print("\n" + "="*80)
    print(" "*25 + "演示完成")
    print("="*80)
    print("\n多表格检测功能优势：")
    print("  ✓ 自动识别单个sheet中的多个表格")
    print("  ✓ 正确处理表格的行列偏移")
    print("  ✓ 提取表格相关的描述信息")
    print("  ✓ 保持数据结构的完整性和准确性")
    print("  ✓ 便于分析和处理复杂的Excel文档")
    print("\n" + "="*80 + "\n")


def compare_with_without_detection():
    """对比启用和不启用多表格检测的差异"""
    
    print("\n" + "="*80)
    print(" "*15 + "对比：启用 vs 不启用多表格检测")
    print("="*80)
    
    # 不启用多表格检测（默认行为）
    print("\n【情况1】不启用多表格检测（默认）")
    parser_normal = ExcelParser(detect_multiple_tables=False)
    doc_normal = parser_normal.load_document('examples/multi_table_sheet.xlsx')
    sheet_name = doc_normal.get_sheet_names()[0]
    df_normal = doc_normal.get_sheet(sheet_name)
    
    print(f"  读取方式: 传统pandas.read_excel()")
    print(f"  结果形状: {df_normal.shape}")
    print(f"  列名: {df_normal.columns.tolist()}")
    print("\n  数据预览（前5行）：")
    print(df_normal.head().to_string(index=False))
    
    # 启用多表格检测
    print("\n" + "-"*80)
    print("\n【情况2】启用多表格检测")
    parser_multi = ExcelParser(detect_multiple_tables=True)
    doc_multi = parser_multi.load_document('examples/multi_table_sheet.xlsx')
    
    tables = doc_multi.get_sheet_tables(sheet_name)
    print(f"  读取方式: 智能表格检测")
    print(f"  检测结果: {len(tables)} 个独立表格")
    
    for i, table_info in enumerate(tables):
        print(f"\n  表格 {i}:")
        print(f"    形状: {table_info.data.shape}")
        print(f"    列名: {table_info.data.columns.tolist()}")
        print(f"    描述: {table_info.description}")
    
    print("\n" + "="*80)
    print("\n结论：")
    print("  • 默认方式适合: 标准格式的单表格sheet")
    print("  • 多表格检测适合: 包含多个表格或有偏移的复杂sheet")
    print("  • 建议: 根据实际Excel文件结构选择合适的方式")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    # 运行演示
    demo_multi_table_detection()
    
    # 运行对比
    compare_with_without_detection()
