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


class MockLLM:
    """模拟LLM用于测试"""
    
    def invoke(self, messages):
        """模拟LLM调用"""
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        # 获取prompt内容
        prompt = messages[0].content if messages else ""
        
        # 根据prompt类型返回相应的模拟响应
        # 1. 列名匹配 (数据表上下文)
        if "数据表上下文:" in prompt and "业务描述:" in prompt:
            if "业务描述: \"姓名\"" in prompt:
                return MockResponse("['姓名']")
            elif "业务描述: \"员工\"" in prompt:
                return MockResponse("['员工编号']")
            elif "业务描述: \"薪资\"" in prompt:
                return MockResponse("['月薪']")
            elif "业务描述: \"营收\"" in prompt:
                return MockResponse("['销售额']")
            elif "业务描述: \"利润\"" in prompt:
                return MockResponse("['净利润']")
            else:
                return MockResponse("[]")
        
        # 2. 工作表/文档名匹配 (可用的工作表)
        elif "可用的工作表:" in prompt and "用户查询:" in prompt:
            if "用户查询: \"24年的预算\"" in prompt or "用户查询: \"24年预算\"" in prompt:
                if "FY24_Budget" in prompt:
                    return MockResponse("FY24_Budget")
                else:
                    return MockResponse("2024年财务报表")
            elif "用户查询: \"员工工资\"" in prompt:
                return MockResponse("员工薪资_2024")
            elif "用户查询: \"sales\"" in prompt:
                return MockResponse("Sales_Q1")
            elif "用户查询: \"报销单\"" in prompt:
                return MockResponse("reimbursement_v2.xlsx")
            elif "用户查询: \"工资\"" in prompt:
                return MockResponse("employee_salary.xlsx")
            elif "用户查询: \"员工\"" in prompt:
                if "员工数据.xlsx" in prompt:
                    return MockResponse("员工数据.xlsx")
                elif "员工信息" in prompt:
                    return MockResponse("员工信息")
                else:
                    return MockResponse("null")
            else:
                return MockResponse("null")
        
        # 3. 关联键发现 (两个数据表)
        elif "请找出最适合用于关联" in prompt:
            # 检查列名是否在上下文中
            has_员工编号 = "列名: ['员工编号'" in prompt or '列名: ["员工编号"' in prompt
            has_工号 = "列名: ['工号'" in prompt or '列名: ["工号"' in prompt
            has_ID = "列名: ['ID'" in prompt or '列名: ["ID"' in prompt
            has_code = "列名: ['code'" in prompt or '列名: ["code"' in prompt
            has_num = "列名: ['num'" in prompt or '列名: ["num"' in prompt
            
            if has_员工编号 and has_工号:
                return MockResponse("('员工编号', '工号')")
            elif has_ID:
                return MockResponse("('ID', 'ID')")
            elif has_code and has_num:
                return MockResponse("('code', 'num')")
            else:
                return MockResponse("null")
        
        # 默认返回
        else:
            return MockResponse("[]")
    
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


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
    
    # 使用MockLLM进行测试
    resolver = SemanticResolver(llm=MockLLM())
    df1, df2, df3 = create_test_data()
    
    # Test 1.1: Exact match (通过LLM)
    print("\n测试1.1: 精确匹配")
    result = resolver.find_column_by_semantic(df1, '姓名')
    print(f"查找 '姓名': {result}")
    assert '姓名' in result, "精确匹配失败"
    print("✓ 精确匹配通过")
    
    # Test 1.2: Synonym match (通过LLM)
    print("\n测试1.2: 同义词匹配")
    result = resolver.find_column_by_semantic(df1, '员工')
    print(f"查找 '员工': {result}")
    # Should find '员工编号' because LLM understands the semantic relationship
    assert len(result) > 0, "同义词匹配失败"
    print("✓ 同义词匹配通过")
    
    # Test 1.3: Synonym match for salary (通过LLM)
    print("\n测试1.3: 工资同义词匹配")
    result = resolver.find_column_by_semantic(df1, '薪资')
    print(f"查找 '薪资': {result}")
    # Should find '月薪' through LLM semantic understanding
    assert len(result) > 0, "工资同义词匹配失败"
    print("✓ 工资同义词匹配通过")
    
    # Test 1.4: Revenue synonym match (通过LLM)
    print("\n测试1.4: 收入同义词匹配")
    result = resolver.find_column_by_semantic(df3, '营收')
    print(f"查找 '营收': {result}")
    # Should find '销售额' through LLM semantic understanding
    assert len(result) > 0, "营收同义词匹配失败"
    print("✓ 营收同义词匹配通过")
    
    # Test 1.5: Profit synonym match (通过LLM)
    print("\n测试1.5: 利润同义词匹配")
    result = resolver.find_column_by_semantic(df3, '利润')
    print(f"查找 '利润': {result}")
    # Should find '净利润' through LLM
    assert len(result) > 0, "利润同义词匹配失败"
    print("✓ 利润同义词匹配通过")
    
    # Test 1.6: Test without LLM (should only do exact match)
    print("\n测试1.6: 无LLM时的精确匹配")
    resolver_no_llm = SemanticResolver()  # No LLM
    result = resolver_no_llm.find_column_by_semantic(df1, '姓名')
    print(f"查找 '姓名' (无LLM): {result}")
    assert '姓名' in result, "无LLM精确匹配失败"
    print("✓ 无LLM精确匹配通过")


def test_sheet_semantic_matching():
    """测试工作表名语义匹配"""
    print("\n" + "="*70)
    print("测试 2: 工作表名语义匹配")
    print("="*70)
    
    resolver = SemanticResolver(llm=MockLLM())
    
    sheet_names = ['2024年财务报表', 'FY24_Budget', '员工薪资_2024', 'Sales_Q1']
    
    # Test 2.1: Partial match (通过LLM)
    print("\n测试2.1: 部分匹配")
    result = resolver.find_sheet_by_semantic(sheet_names, '24年的预算')
    print(f"查找 '24年的预算': {result}")
    # Should match 'FY24_Budget' or '2024年财务报表' via LLM
    assert result is not None, "工作表部分匹配失败"
    print(f"✓ 匹配到: {result}")
    
    # Test 2.2: Keyword match (通过LLM)
    print("\n测试2.2: 关键词匹配")
    result = resolver.find_sheet_by_semantic(sheet_names, '员工工资')
    print(f"查找 '员工工资': {result}")
    # Should match '员工薪资_2024' via LLM
    assert result is not None, "工作表关键词匹配失败"
    print(f"✓ 匹配到: {result}")
    
    # Test 2.3: English keyword match (通过LLM)
    print("\n测试2.3: 英文关键词匹配")
    result = resolver.find_sheet_by_semantic(sheet_names, 'sales')
    print(f"查找 'sales': {result}")
    # Should match 'Sales_Q1' via LLM
    assert result is not None, "英文关键词匹配失败"
    print(f"✓ 匹配到: {result}")


def test_join_key_auto_discovery():
    """测试关联键自动发现"""
    print("\n" + "="*70)
    print("测试 3: 关联键自动发现")
    print("="*70)
    
    resolver = SemanticResolver(llm=MockLLM())
    df1, df2, df3 = create_test_data()
    
    # Test 3.1: Exact column name match (无LLM也应该工作)
    print("\n测试3.1: 同名列匹配")
    df_test1 = pd.DataFrame({'ID': [1, 2, 3], 'name': ['A', 'B', 'C']})
    df_test2 = pd.DataFrame({'ID': [1, 2, 3], 'value': [10, 20, 30]})
    result = resolver.auto_discover_join_keys(df_test1, df_test2)
    print(f"发现的关联键: {result}")
    assert result == ('ID', 'ID'), "同名列匹配失败"
    print("✓ 同名列匹配通过")
    
    # Test 3.2: Synonym match via LLM (员工编号 vs 工号)
    print("\n测试3.2: 同义词列匹配 (通过LLM)")
    result = resolver.auto_discover_join_keys(df1, df2)
    print(f"发现的关联键: {result}")
    # Should find ('员工编号', '工号') via LLM
    assert result is not None, "同义词列匹配失败"
    assert result[0] == '员工编号' and result[1] == '工号', "关联键不正确"
    print(f"✓ 发现关联键: {result}")
    
    # Test 3.3: Data overlap detection via LLM
    print("\n测试3.3: 数据重叠检测 (通过LLM)")
    df_test3 = pd.DataFrame({'code': [1, 2, 3], 'name': ['A', 'B', 'C']})
    df_test4 = pd.DataFrame({'num': [1, 2, 3], 'value': [10, 20, 30]})
    result = resolver.auto_discover_join_keys(df_test3, df_test4)
    print(f"发现的关联键: {result}")
    # Should find ('code', 'num') via LLM analyzing the data
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
        
        # 使用MockLLM创建parser
        parser = ExcelParser(llm=MockLLM())
        
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
    
    resolver = SemanticResolver(llm=MockLLM())
    
    doc_names = ['financial_report_2024.xlsx', 'reimbursement_v2.xlsx', 'employee_salary.xlsx']
    
    # Test 6.1: Reimbursement matching (通过LLM)
    print("\n测试6.1: 报销单匹配")
    result = resolver.find_document_by_semantic(doc_names, '报销单')
    print(f"查找 '报销单': {result}")
    assert result is not None, "报销单匹配失败"
    print(f"✓ 匹配到: {result}")
    
    # Test 6.2: Salary matching (通过LLM)
    print("\n测试6.2: 工资单匹配")
    result = resolver.find_document_by_semantic(doc_names, '工资')
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
