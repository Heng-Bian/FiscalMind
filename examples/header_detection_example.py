"""
LLM-based表头检测功能示例
Example of LLM-based header detection functionality.

这个示例展示如何使用LLM来智能检测表头，避免将数据行误判为表头。
This example shows how to use LLM to intelligently detect headers and avoid 
mistaking data rows as headers.
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.parser import ExcelParser
from fiscal_mind.specialized_agents import HeaderDetectionAgent
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def example_1_basic_usage():
    """
    示例1: 基本使用 - 不使用LLM（规则基础方法）
    Example 1: Basic usage - without LLM (rule-based method)
    """
    print("\n" + "="*80)
    print("示例1: 不使用LLM的表头检测（规则基础方法）")
    print("Example 1: Header detection without LLM (rule-based method)")
    print("="*80)
    
    # 创建不带LLM的解析器
    parser = ExcelParser(detect_multiple_tables=True, llm=None)
    
    # 加载Excel文件
    doc = parser.load_document('examples/complex_header_sample.xlsx')
    
    print(f"\n文档: {doc.file_name}")
    print(f"工作表: {doc.get_sheet_names()}")
    
    df = doc.get_sheet(doc.get_sheet_names()[0])
    if df is not None:
        print(f"\n表格形状: {df.shape}")
        print(f"列名: {list(df.columns)[:5]}...")  # 只显示前5个列名
        print("\n前3行数据:")
        print(df.head(3))


def example_2_with_real_llm():
    """
    示例2: 使用真实的LLM进行智能表头检测
    Example 2: Using real LLM for intelligent header detection
    
    注意: 这需要配置LLM API密钥
    Note: This requires LLM API key configuration
    """
    print("\n" + "="*80)
    print("示例2: 使用LLM的智能表头检测")
    print("Example 2: Intelligent header detection with LLM")
    print("="*80)
    
    try:
        from langchain_openai import ChatOpenAI
        
        # 创建LLM实例（需要配置API密钥）
        # 这里使用环境变量或配置文件中的API密钥
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # 或使用其他模型
            temperature=0
        )
        
        # 创建带LLM的解析器
        parser = ExcelParser(detect_multiple_tables=True, llm=llm)
        
        # 加载Excel文件
        doc = parser.load_document('examples/complex_header_sample.xlsx')
        
        print(f"\n文档: {doc.file_name}")
        print(f"工作表: {doc.get_sheet_names()}")
        
        df = doc.get_sheet(doc.get_sheet_names()[0])
        if df is not None:
            print(f"\n表格形状: {df.shape}")
            print(f"列名: {list(df.columns)[:5]}...")
            print("\n前3行数据:")
            print(df.head(3))
            
    except ImportError:
        print("\n提示: 需要安装 langchain-openai 才能使用真实的LLM")
        print("运行: pip install langchain-openai")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("提示: 请确保已配置正确的API密钥")


def example_3_header_detection_agent():
    """
    示例3: 直接使用HeaderDetectionAgent
    Example 3: Using HeaderDetectionAgent directly
    """
    print("\n" + "="*80)
    print("示例3: 直接使用HeaderDetectionAgent")
    print("Example 3: Using HeaderDetectionAgent directly")
    print("="*80)
    
    # 模拟表格的前10行数据
    rows_data = [
        ["产品名称", "销售额", "成本", "利润", "利润率"],  # 表头
        ["产品A", 10000, 6000, 4000, "40%"],  # 数据行
        ["产品B", 15000, 9000, 6000, "40%"],
        ["产品C", 12000, 7000, 5000, "41.7%"],
        ["产品D", 8000, 5000, 3000, "37.5%"],
        ["产品E", 20000, 12000, 8000, "40%"],
    ]
    
    # 创建HeaderDetectionAgent（不使用LLM）
    agent = HeaderDetectionAgent(llm=None)
    result = agent.detect_header_rows(rows_data, max_rows=10)
    
    print("\n检测结果:")
    print(f"  表头行数: {result['header_row_count']}")
    print(f"  表头行索引: {result['header_rows_indices']}")
    print(f"  数据开始行: {result['data_start_row']}")
    print(f"  置信度: {result['confidence']}")
    print(f"  理由: {result['reasoning']}")
    
    # 根据检测结果提取表头和数据
    if result['header_row_count'] > 0:
        headers = rows_data[0]  # 假设只有一行表头
        data_rows = rows_data[result['data_start_row']:]
        
        print(f"\n表头: {headers}")
        print(f"数据行数: {len(data_rows)}")
        print("前3行数据:")
        for i, row in enumerate(data_rows[:3]):
            print(f"  行{i}: {row}")


def example_4_custom_llm():
    """
    示例4: 使用自定义LLM配置
    Example 4: Using custom LLM configuration
    """
    print("\n" + "="*80)
    print("示例4: 使用自定义LLM配置（模拟）")
    print("Example 4: Using custom LLM configuration (mock)")
    print("="*80)
    
    # 创建一个模拟的LLM
    class CustomMockLLM:
        def invoke(self, messages):
            class MockResponse:
                content = '''```json
{
  "header_row_count": 1,
  "header_rows_indices": [0],
  "data_start_row": 1,
  "confidence": 0.98,
  "reasoning": "使用自定义LLM检测: 第0行是表头，包含字段名称"
}
```'''
            return MockResponse()
    
    # 创建使用自定义LLM的agent
    agent = HeaderDetectionAgent(llm=CustomMockLLM())
    
    rows_data = [
        ["姓名", "年龄", "城市"],
        ["张三", 25, "北京"],
        ["李四", 30, "上海"],
    ]
    
    result = agent.detect_header_rows(rows_data, max_rows=10)
    
    print("\n检测结果:")
    print(f"  表头行数: {result['header_row_count']}")
    print(f"  置信度: {result['confidence']}")
    print(f"  理由: {result['reasoning']}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" "*20 + "LLM-Based 表头检测示例")
    print(" "*15 + "LLM-Based Header Detection Examples")
    print("="*80)
    
    print("\n本示例展示了以下功能:")
    print("This example demonstrates the following features:")
    print("  1. 基本使用（不使用LLM）")
    print("     Basic usage (without LLM)")
    print("  2. 使用真实LLM进行智能检测")
    print("     Using real LLM for intelligent detection")
    print("  3. 直接使用HeaderDetectionAgent")
    print("     Using HeaderDetectionAgent directly")
    print("  4. 自定义LLM配置")
    print("     Custom LLM configuration")
    
    # 运行示例
    example_1_basic_usage()
    # example_2_with_real_llm()  # 取消注释以使用真实LLM
    example_3_header_detection_agent()
    example_4_custom_llm()
    
    print("\n" + "="*80)
    print(" "*25 + "示例运行完成")
    print(" "*22 + "Examples completed")
    print("="*80)
    
    print("\n使用建议:")
    print("Usage recommendations:")
    print("  • 对于简单表格，可以不使用LLM（规则基础方法）")
    print("    For simple tables, you can use without LLM (rule-based method)")
    print("  • 对于复杂表格或容易误判的情况，建议使用LLM")
    print("    For complex tables or ambiguous cases, use LLM for better accuracy")
    print("  • LLM会自动fallback到规则方法（当置信度低或出错时）")
    print("    LLM automatically falls back to rule-based method (low confidence or errors)")
    print("\n")


if __name__ == "__main__":
    main()
