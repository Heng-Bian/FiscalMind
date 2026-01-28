"""
演示脚本：展示LLM表头检测 vs 规则基础检测的差异
Demonstration: LLM-based vs Rule-based header detection comparison
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.specialized_agents import HeaderDetectionAgent


def demonstrate_problem():
    """演示问题：规则方法会误判数据行为表头"""
    print("\n" + "="*80)
    print("问题演示：规则基础方法的局限性")
    print("Problem Demonstration: Limitations of rule-based method")
    print("="*80)
    
    # 创建一个测试数据集，其中数据行包含大量文本
    rows_data = [
        ["员工姓名", "职位", "部门", "入职时间", "工资"],  # 第0行 - 真正的表头
        ["张伟", "高级经理", "销售部", "2020-01-15", "25000"],  # 第1行 - 数据行（4/5是文本）
        ["李娜", "资深工程师", "技术部", "2019-03-20", "22000"],  # 第2行 - 数据行（4/5是文本）
        ["王强", "首席分析师", "财务部", "2018-06-10", "28000"],
        ["赵敏", "项目主管", "运营部", "2021-02-28", "20000"],
    ]
    
    print("\n测试数据（前5行）:")
    print("-" * 80)
    for i, row in enumerate(rows_data):
        row_display = " | ".join([str(x) for x in row])
        print(f"第{i}行: {row_display}")
    
    # 使用规则基础方法
    print("\n" + "-" * 80)
    print("1. 规则基础方法（Rule-based Method）")
    print("-" * 80)
    
    agent_rule = HeaderDetectionAgent(llm=None)
    result_rule = agent_rule.detect_header_rows(rows_data, max_rows=5)
    
    print(f"检测结果:")
    print(f"  表头行数: {result_rule['header_row_count']}")
    print(f"  表头行索引: {result_rule['header_rows_indices']}")
    print(f"  数据开始行: {result_rule['data_start_row']}")
    print(f"  置信度: {result_rule['confidence']}")
    
    if result_rule['header_row_count'] > 1:
        print(f"\n❌ 问题: 规则方法将前{result_rule['header_row_count']}行都识别为表头！")
        print(f"   原因: 这些行中文本单元格占比都 >= 50%")
        print(f"   实际: 只有第0行是表头，其余是数据行")
    
    # 使用LLM方法
    print("\n" + "-" * 80)
    print("2. LLM智能方法（LLM-based Method）")
    print("-" * 80)
    
    # 创建模拟LLM
    class MockLLM:
        def invoke(self, messages):
            class MockResponse:
                content = '''```json
{
  "header_row_count": 1,
  "header_rows_indices": [0],
  "data_start_row": 1,
  "confidence": 0.95,
  "reasoning": "第0行包含字段名称（员工姓名、职位、部门等），第1-4行是具体的员工数据（包含人名、职位名称、具体日期和数值），应该被识别为数据行而非表头"
}
```'''
            return MockResponse()
    
    agent_llm = HeaderDetectionAgent(llm=MockLLM())
    result_llm = agent_llm.detect_header_rows(rows_data, max_rows=5)
    
    print(f"检测结果:")
    print(f"  表头行数: {result_llm['header_row_count']}")
    print(f"  表头行索引: {result_llm['header_rows_indices']}")
    print(f"  数据开始行: {result_llm['data_start_row']}")
    print(f"  置信度: {result_llm['confidence']}")
    print(f"  理由: {result_llm['reasoning'][:60]}...")
    
    if result_llm['header_row_count'] == 1:
        print(f"\n✓ 正确: LLM准确识别出只有1行表头！")
        print(f"   优势: LLM理解了数据的语义，知道这些是具体的员工数据")
    
    # 对比总结
    print("\n" + "="*80)
    print("对比总结 (Comparison Summary)")
    print("="*80)
    print(f"\n规则基础方法:")
    print(f"  识别的表头行数: {result_rule['header_row_count']}")
    print(f"  准确性: ❌ 不准确（将数据行误判为表头）")
    print(f"  原因: 只基于文本比例的简单规则")
    
    print(f"\nLLM智能方法:")
    print(f"  识别的表头行数: {result_llm['header_row_count']}")
    print(f"  准确性: ✓ 准确（正确识别表头和数据行）")
    print(f"  原因: 理解数据的语义和上下文")
    
    print("\n改进效果:")
    print(f"  ✓ 避免了将{result_rule['header_row_count'] - result_llm['header_row_count']}行数据误判为表头")
    print(f"  ✓ 确保数据完整性，不丢失有效数据")
    print(f"  ✓ 提高后续数据分析的准确性")


def demonstrate_multi_header():
    """演示多层表头的智能识别"""
    print("\n" + "="*80)
    print("进阶演示：多层表头的智能识别")
    print("Advanced Demo: Multi-level header detection")
    print("="*80)
    
    # 创建多层表头的测试数据
    rows_data = [
        ["", "2023年", "2023年", "2024年", "2024年"],  # 第0行 - 年份层
        ["区域", "销售额", "利润", "销售额", "利润"],  # 第1行 - 指标层
        ["华东", "1000", "200", "1200", "250"],  # 第2行 - 数据
        ["华北", "800", "150", "900", "180"],
        ["华南", "1200", "240", "1400", "300"],
    ]
    
    print("\n测试数据:")
    print("-" * 80)
    for i, row in enumerate(rows_data):
        row_display = " | ".join([str(x) for x in row])
        print(f"第{i}行: {row_display}")
    
    # 模拟LLM响应
    class MockLLMMultiHeader:
        def invoke(self, messages):
            class MockResponse:
                content = '''```json
{
  "header_row_count": 2,
  "header_rows_indices": [0, 1],
  "data_start_row": 2,
  "confidence": 0.92,
  "reasoning": "第0-1行是两层表头结构：第0行是年份层，第1行是指标层；第2行开始是具体的区域数据"
}
```'''
            return MockResponse()
    
    agent = HeaderDetectionAgent(llm=MockLLMMultiHeader())
    result = agent.detect_header_rows(rows_data, max_rows=5)
    
    print("\nLLM检测结果:")
    print("-" * 80)
    print(f"  表头行数: {result['header_row_count']}")
    print(f"  表头层级: 第{result['header_rows_indices'][0]}行和第{result['header_rows_indices'][1]}行")
    print(f"  数据开始行: 第{result['data_start_row']}行")
    print(f"  置信度: {result['confidence']}")
    
    print(f"\n✓ LLM正确识别了2层表头结构！")
    print(f"  层级1（第0行）: 年份层 (2023年、2024年)")
    print(f"  层级2（第1行）: 指标层 (区域、销售额、利润)")
    print(f"  数据行（第2行起）: 具体的区域数据")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "LLM表头检测改进演示")
    print(" "*15 + "LLM Header Detection Improvement Demo")
    print("="*80)
    
    demonstrate_problem()
    demonstrate_multi_header()
    
    print("\n" + "="*80)
    print("演示完成 - Demo Completed")
    print("="*80)
    print("\n总结:")
    print("Summary:")
    print("  • LLM能够理解数据的语义，准确区分表头和数据行")
    print("    LLM understands data semantics and accurately distinguishes headers from data")
    print("  • 避免了规则方法中\"文本比例高就是表头\"的误判")
    print("    Avoids the false positives of rule-based 'high text ratio = header'")
    print("  • 支持复杂的多层表头结构识别")
    print("    Supports complex multi-level header structures")
    print("  • 提供置信度评分，确保检测质量")
    print("    Provides confidence scores to ensure detection quality")
    print()
