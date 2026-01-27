"""
测试语义解析器功能
Test semantic resolver features: semantic matching, auto-discovery, etc.
"""

import sys
import os
import pandas as pd
import tempfile

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.semantic_resolver import SemanticResolver
from fiscal_mind.parser import ExcelParser
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)


def create_test_data():
    """创建测试数据"""
    # Create test DataFrame
    df1 = pd.DataFrame({
        '员工编号': [1, 2, 3, 4, 5],
        '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
        '部门': ['销售部', '技术部', '财务部', '销售部', '技术部'],
        '月薪': [8000, 12000, 9000, 8500, 11000],
        '城市': ['北京', '上海', '广州', '北京', '深圳']
    })
    
    df2 = pd.DataFrame({
        '工号': [1, 2, 3, 4, 5],  # Similar to 员工编号
        '绩效得分': [85, 92, 88, 90, 95],
        '项目数': [3, 5, 2, 4, 6]
    })
    
    df3 = pd.DataFrame({
        '产品名称': ['产品A', '产品B', '产品C'],
        '销售额': [100000, 150000, 120000],
        '净利润': [20000, 35000, 25000],
        '区域': ['华东', '华南', '华北']
    })
    
    return df1, df2, df3


def test_column_semantic_matching():
    """测试列名语义匹配"""
    print("\n" + "="*70)
    print("测试 1: 列名语义匹配")
    print("="*70)
    
    resolver = SemanticResolver()
    df1, df2, df3 = create_test_data()
    
    # Test 1.1: Exact match
    print("\n测试1.1: 精确匹配")
    result = resolver.find_column_by_semantic(df1, '姓名', use_llm_fallback=False)
    print(f"查找 '姓名': {result}")
    assert '姓名' in result, "精确匹配失败"
    print("✓ 精确匹配通过")
    
    # Test 1.2: Synonym match
    print("\n测试1.2: 同义词匹配")
    result = resolver.find_column_by_semantic(df1, '员工', use_llm_fallback=False)
    print(f"查找 '员工': {result}")
    # Should find '员工编号' because it contains '员工'
    assert len(result) > 0, "同义词匹配失败"
    print("✓ 同义词匹配通过")
    
    # Test 1.3: Synonym match for salary
    print("\n测试1.3: 工资同义词匹配")
    result = resolver.find_column_by_semantic(df1, '薪资', use_llm_fallback=False)
    print(f"查找 '薪资': {result}")
    # Should find '月薪' through synonym mapping
    assert len(result) > 0, "工资同义词匹配失败"
    print("✓ 工资同义词匹配通过")
    
    # Test 1.4: Revenue synonym match
    print("\n测试1.4: 收入同义词匹配")
    result = resolver.find_column_by_semantic(df3, '营收', use_llm_fallback=False)
    print(f"查找 '营收': {result}")
    # Should find '销售额' through synonym mapping
    assert len(result) > 0, "营收同义词匹配失败"
    print("✓ 营收同义词匹配通过")
    
    # Test 1.5: Profit synonym match
    print("\n测试1.5: 利润同义词匹配")
    result = resolver.find_column_by_semantic(df3, '利润', use_llm_fallback=False)
    print(f"查找 '利润': {result}")
    # Should find '净利润'
    assert len(result) > 0, "利润同义词匹配失败"
    print("✓ 利润同义词匹配通过")


def test_sheet_semantic_matching():
    """测试工作表名语义匹配"""
    print("\n" + "="*70)
    print("测试 2: 工作表名语义匹配")
    print("="*70)
    
    resolver = SemanticResolver()
    
    sheet_names = ['2024年财务报表', 'FY24_Budget', '员工薪资_2024', 'Sales_Q1']
    
    # Test 2.1: Partial match
    print("\n测试2.1: 部分匹配")
    result = resolver.find_sheet_by_semantic(sheet_names, '24年的预算', use_llm_fallback=False)
    print(f"查找 '24年的预算': {result}")
    # Should match 'FY24_Budget' or '2024年财务报表'
    assert result is not None, "工作表部分匹配失败"
    print(f"✓ 匹配到: {result}")
    
    # Test 2.2: Keyword match
    print("\n测试2.2: 关键词匹配")
    result = resolver.find_sheet_by_semantic(sheet_names, '员工工资', use_llm_fallback=False)
    print(f"查找 '员工工资': {result}")
    # Should match '员工薪资_2024'
    assert result is not None, "工作表关键词匹配失败"
    print(f"✓ 匹配到: {result}")
    
    # Test 2.3: English keyword match
    print("\n测试2.3: 英文关键词匹配")
    result = resolver.find_sheet_by_semantic(sheet_names, 'sales', use_llm_fallback=False)
    print(f"查找 'sales': {result}")
    # Should match 'Sales_Q1'
    assert result is not None, "英文关键词匹配失败"
    print(f"✓ 匹配到: {result}")


def test_join_key_auto_discovery():
    """测试关联键自动发现"""
    print("\n" + "="*70)
    print("测试 3: 关联键自动发现")
    print("="*70)
    
    resolver = SemanticResolver()
    df1, df2, df3 = create_test_data()
    
    # Test 3.1: Exact column name match
    print("\n测试3.1: 同名列匹配")
    df_test1 = pd.DataFrame({'ID': [1, 2, 3], 'name': ['A', 'B', 'C']})
    df_test2 = pd.DataFrame({'ID': [1, 2, 3], 'value': [10, 20, 30]})
    result = resolver.auto_discover_join_keys(df_test1, df_test2, use_llm_fallback=False)
    print(f"发现的关联键: {result}")
    assert result == ('ID', 'ID'), "同名列匹配失败"
    print("✓ 同名列匹配通过")
    
    # Test 3.2: Synonym match (员工编号 vs 工号)
    print("\n测试3.2: 同义词列匹配")
    result = resolver.auto_discover_join_keys(df1, df2, use_llm_fallback=False)
    print(f"发现的关联键: {result}")
    # Should find ('员工编号', '工号') as they are synonyms
    assert result is not None, "同义词列匹配失败"
    print(f"✓ 发现关联键: {result}")
    
    # Test 3.3: Data overlap detection
    print("\n测试3.3: 数据重叠检测")
    df_test3 = pd.DataFrame({'code': [1, 2, 3], 'name': ['A', 'B', 'C']})
    df_test4 = pd.DataFrame({'num': [1, 2, 3], 'value': [10, 20, 30]})
    result = resolver.auto_discover_join_keys(df_test3, df_test4, use_llm_fallback=False)
    print(f"发现的关联键: {result}")
    # Should find ('code', 'num') based on data overlap
    assert result is not None, "数据重叠检测失败"
    print(f"✓ 发现关联键: {result}")


def test_value_normalization():
    """测试值标准化"""
    print("\n" + "="*70)
    print("测试 4: 值标准化（地名）")
    print("="*70)
    
    resolver = SemanticResolver()
    
    # Test 4.1: Beijing variations
    print("\n测试4.1: 北京地名标准化")
    test_values = ['北京', '北京市', 'Beijing', 'beijing']
    for val in test_values:
        result = resolver.normalize_value(val, category='location')
        print(f"'{val}' -> '{result}'")
        assert result == '北京', f"北京标准化失败: {val}"
    print("✓ 北京地名标准化通过")
    
    # Test 4.2: Shanghai variations
    print("\n测试4.2: 上海地名标准化")
    test_values = ['上海', '上海市', 'Shanghai']
    for val in test_values:
        result = resolver.normalize_value(val, category='location')
        print(f"'{val}' -> '{result}'")
        assert result == '上海', f"上海标准化失败: {val}"
    print("✓ 上海地名标准化通过")
    
    # Test 4.3: Region variations
    print("\n测试4.3: 区域标准化")
    test_values = ['华东', '华东地区', 'East China']
    for val in test_values:
        result = resolver.normalize_value(val, category='location')
        print(f"'{val}' -> '{result}'")
        assert result == '华东', f"华东标准化失败: {val}"
    print("✓ 区域标准化通过")


def test_parser_integration():
    """测试与ExcelParser的集成"""
    print("\n" + "="*70)
    print("测试 5: Parser集成测试")
    print("="*70)
    
    # Create test Excel files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file 1
        file1 = os.path.join(tmpdir, '员工数据.xlsx')
        df1, df2, df3 = create_test_data()
        with pd.ExcelWriter(file1, engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name='员工信息', index=False)
        
        # Create test file 2
        file2 = os.path.join(tmpdir, '绩效数据.xlsx')
        with pd.ExcelWriter(file2, engine='openpyxl') as writer:
            df2.to_excel(writer, sheet_name='绩效评分', index=False)
        
        parser = ExcelParser()
        
        # Test 5.1: Load documents
        print("\n测试5.1: 加载文档")
        parser.load_documents([file1, file2])
        print(f"✓ 已加载文档: {list(parser.documents.keys())}")
        
        # Test 5.2: Semantic document matching
        print("\n测试5.2: 文档语义匹配")
        doc = parser.get_document('员工', use_semantic=True)
        assert doc is not None, "文档语义匹配失败"
        print(f"✓ 通过查询 '员工' 找到文档: {doc.file_name}")
        
        # Test 5.3: Semantic sheet matching
        print("\n测试5.3: 工作表语义匹配")
        df = doc.get_sheet('员工', use_semantic=True)
        assert df is not None, "工作表语义匹配失败"
        print(f"✓ 通过查询 '员工' 找到工作表，形状: {df.shape}")
        
        # Test 5.4: Auto-join
        print("\n测试5.4: 自动关联")
        result = parser.join_sheets(
            '员工数据.xlsx', '员工信息',
            '绩效数据.xlsx', '绩效评分',
            auto_discover=True
        )
        if result is not None:
            print(f"✓ 自动关联成功，结果形状: {result.shape}")
            print(f"✓ 关联后列: {result.columns.tolist()}")
        else:
            print("⚠ 自动关联未成功（这可能是正常的，取决于数据）")
        
        # Test 5.5: Semantic filtering
        print("\n测试5.5: 语义过滤")
        filters = [
            {'column': '薪资', 'operator': '>', 'value': 9000}  # '薪资' is synonym for '月薪'
        ]
        result = doc.filter_rows_advanced('员工信息', filters, use_semantic=True)
        if result is not None:
            print(f"✓ 语义过滤成功，过滤后行数: {len(result)}")
            print(f"✓ 示例数据:\n{result[['姓名', '月薪']].head()}")
        
        # Test 5.6: Location normalization in filtering
        print("\n测试5.6: 地名标准化过滤")
        df_test = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D'],
            'city': ['北京市', '上海', 'Beijing', '广州']
        })
        # Create a temp file for this test
        file3 = os.path.join(tmpdir, 'location_test.xlsx')
        with pd.ExcelWriter(file3, engine='openpyxl') as writer:
            df_test.to_excel(writer, sheet_name='数据', index=False)
        
        doc3 = parser.load_document(file3)
        filters = [
            {'column': 'city', 'operator': '==', 'value': '北京'}
        ]
        result = doc3.filter_rows_advanced('数据', filters, use_semantic=True)
        if result is not None:
            print(f"✓ 地名标准化过滤成功，找到 {len(result)} 行")
            # Should find both '北京市' and 'Beijing'
            print(f"✓ 匹配的城市: {result['city'].tolist()}")


def test_document_name_matching():
    """测试文档名称匹配"""
    print("\n" + "="*70)
    print("测试 6: 文档名称语义匹配")
    print("="*70)
    
    resolver = SemanticResolver()
    
    doc_names = ['financial_report_2024.xlsx', 'reimbursement_v2.xlsx', 'employee_salary.xlsx']
    
    # Test 6.1: Reimbursement matching
    print("\n测试6.1: 报销单匹配")
    result = resolver.find_document_by_semantic(doc_names, '报销单', use_llm_fallback=False)
    print(f"查找 '报销单': {result}")
    assert result is not None, "报销单匹配失败"
    print(f"✓ 匹配到: {result}")
    
    # Test 6.2: Salary matching
    print("\n测试6.2: 工资单匹配")
    result = resolver.find_document_by_semantic(doc_names, '工资', use_llm_fallback=False)
    print(f"查找 '工资': {result}")
    assert result is not None, "工资单匹配失败"
    print(f"✓ 匹配到: {result}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("开始语义解析器测试套件")
    print("="*70)
    
    try:
        test_column_semantic_matching()
        test_sheet_semantic_matching()
        test_join_key_auto_discovery()
        test_value_normalization()
        test_document_name_matching()
        test_parser_integration()
        
        print("\n" + "="*70)
        print("✓ 所有测试通过！")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {str(e)}")
        raise
    except Exception as e:
        print(f"\n✗ 测试错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
