"""
PRR Agent 示例 - 演示Plan-ReAct-Reflect架构
Example demonstrating the Plan-ReAct-Reflect (PRR) Agent architecture.

这个示例展示了如何使用PRR Agent来回答财务BP的问题，例如:
"哪个大区今年表现更好?"
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.prr_agent import PRRAgent
from fiscal_mind.parser import ExcelParser
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_regional_sales_data():
    """创建包含区域数据的示例Excel文件"""
    examples_dir = Path(__file__).parent
    
    # 创建区域销售数据
    regional_sales = {
        '大区': ['华北', '华东', '华南', '华西', '华中'] * 6,
        '月份': ['2024-01'] * 5 + ['2024-02'] * 5 + ['2024-03'] * 5 + 
                ['2024-04'] * 5 + ['2024-05'] * 5 + ['2024-06'] * 5,
        '销售额': [
            # 1月
            1200000, 1500000, 980000, 850000, 1100000,
            # 2月
            1250000, 1600000, 1020000, 880000, 1150000,
            # 3月
            1300000, 1650000, 1100000, 920000, 1200000,
            # 4月
            1350000, 1700000, 1150000, 950000, 1250000,
            # 5月
            1400000, 1750000, 1200000, 980000, 1300000,
            # 6月
            1500000, 1850000, 1280000, 1050000, 1380000,
        ],
        '利润': [
            # 1月
            240000, 300000, 196000, 170000, 220000,
            # 2月
            250000, 320000, 204000, 176000, 230000,
            # 3月
            260000, 330000, 220000, 184000, 240000,
            # 4月
            270000, 340000, 230000, 190000, 250000,
            # 5月
            280000, 350000, 240000, 196000, 260000,
            # 6月
            300000, 370000, 256000, 210000, 276000,
        ],
        '订单数': [
            # 1月
            120, 150, 98, 85, 110,
            # 2月
            125, 160, 102, 88, 115,
            # 3月
            130, 165, 110, 92, 120,
            # 4月
            135, 170, 115, 95, 125,
            # 5月
            140, 175, 120, 98, 130,
            # 6月
            150, 185, 128, 105, 138,
        ]
    }
    
    df = pd.DataFrame(regional_sales)
    
    # 计算一些派生指标
    df['客单价'] = df['销售额'] / df['订单数']
    df['利润率'] = (df['利润'] / df['销售额'] * 100).round(2)
    
    # 保存到Excel
    output_path = examples_dir / 'regional_performance.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='区域业绩', index=False)
        
        # 添加一个汇总表
        summary = df.groupby('大区').agg({
            '销售额': 'sum',
            '利润': 'sum',
            '订单数': 'sum'
        }).reset_index()
        summary['平均客单价'] = summary['销售额'] / summary['订单数']
        summary['整体利润率'] = (summary['利润'] / summary['销售额'] * 100).round(2)
        summary = summary.sort_values('销售额', ascending=False)
        
        summary.to_excel(writer, sheet_name='区域汇总', index=False)
    
    logger.info(f"创建示例文件: {output_path}")
    return output_path


def create_product_performance_data():
    """创建产品业绩数据"""
    examples_dir = Path(__file__).parent
    
    product_data = {
        '产品': ['产品A', '产品B', '产品C', '产品D'] * 6,
        '季度': ['Q1'] * 4 + ['Q2'] * 4 + ['Q3'] * 4 + ['Q4'] * 4 + ['Q1'] * 4 + ['Q2'] * 4,
        '年份': [2023] * 16 + [2024] * 8,
        '销售数量': [
            # 2023 Q1
            1000, 1200, 800, 600,
            # 2023 Q2
            1100, 1250, 850, 650,
            # 2023 Q3
            1150, 1300, 900, 700,
            # 2023 Q4
            1200, 1350, 950, 750,
            # 2024 Q1
            1300, 1400, 1000, 800,
            # 2024 Q2
            1400, 1500, 1100, 850,
        ],
        '销售额': [
            # 2023 Q1
            500000, 720000, 360000, 180000,
            # 2023 Q2
            550000, 750000, 382500, 195000,
            # 2023 Q3
            575000, 780000, 405000, 210000,
            # 2023 Q4
            600000, 810000, 427500, 225000,
            # 2024 Q1
            650000, 840000, 450000, 240000,
            # 2024 Q2
            700000, 900000, 495000, 255000,
        ]
    }
    
    df = pd.DataFrame(product_data)
    df['单价'] = (df['销售额'] / df['销售数量']).round(2)
    
    output_path = examples_dir / 'product_performance.xlsx'
    df.to_excel(output_path, index=False, sheet_name='产品业绩')
    
    logger.info(f"创建示例文件: {output_path}")
    return output_path


def demo_basic_prr():
    """演示基本的PRR Agent功能"""
    print("\n" + "="*80)
    print("演示 1: 基本PRR Agent - 区域对比分析")
    print("="*80)
    
    # 创建示例数据
    regional_file = create_regional_sales_data()
    
    # 创建PRR Agent
    agent = PRRAgent()
    
    # 加载文档
    agent.load_documents([str(regional_file)])
    
    # 提问：哪个大区今年表现更好
    question = "哪个大区今年表现更好?"
    print(f"\n问题: {question}")
    print("-" * 80)
    
    # 执行查询
    answer = agent.query(question)
    
    print("\nPRR Agent 回答:")
    print(answer)


def demo_prr_with_multiple_docs():
    """演示处理多个文档的PRR Agent"""
    print("\n" + "="*80)
    print("演示 2: 多文档PRR Agent - 产品和区域综合分析")
    print("="*80)
    
    # 创建示例数据
    regional_file = create_regional_sales_data()
    product_file = create_product_performance_data()
    
    # 创建PRR Agent
    agent = PRRAgent()
    
    # 加载多个文档
    agent.load_documents([str(regional_file), str(product_file)])
    
    # 多个问题
    questions = [
        "哪个大区的利润率最高?",
        "销售额增长最快的是哪个区域?",
        "哪个产品的销售表现最好?"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 80)
        
        answer = agent.query(question)
        print("\nPRR Agent 回答:")
        print(answer)
        print("\n")


def demo_prr_workflow_details():
    """演示PRR工作流的详细过程"""
    print("\n" + "="*80)
    print("演示 3: PRR工作流详解")
    print("="*80)
    
    # 创建示例数据
    regional_file = create_regional_sales_data()
    
    # 创建PRR Agent（启用详细日志）
    logger.setLevel(logging.DEBUG)
    agent = PRRAgent(max_iterations=8)
    
    # 加载文档
    agent.load_documents([str(regional_file)])
    
    # 复杂问题
    question = "对比各大区的销售额和利润率，找出综合表现最好的大区"
    print(f"\n问题: {question}")
    print("-" * 80)
    print("\nPRR工作流执行中...")
    print("(查看日志了解Plan -> ReAct -> Reflect的循环过程)\n")
    
    # 执行查询
    answer = agent.query(question)
    
    print("\n最终答案:")
    print(answer)


def demo_comparison_with_basic_agent():
    """对比PRR Agent和基本Agent的差异"""
    print("\n" + "="*80)
    print("演示 4: PRR Agent vs 基本Agent对比")
    print("="*80)
    
    from fiscal_mind.agent import TableDocumentAgent
    
    # 创建示例数据
    regional_file = create_regional_sales_data()
    
    question = "哪个大区今年表现更好?"
    
    # 1. 使用基本Agent
    print("\n使用基本Agent:")
    print("-" * 80)
    basic_agent = TableDocumentAgent()
    basic_agent.load_documents([str(regional_file)])
    basic_answer = basic_agent.query(question)
    print(basic_answer)
    
    # 2. 使用PRR Agent
    print("\n使用PRR Agent:")
    print("-" * 80)
    prr_agent = PRRAgent()
    prr_agent.load_documents([str(regional_file)])
    prr_answer = prr_agent.query(question)
    print(prr_answer)
    
    print("\n" + "="*80)
    print("对比说明:")
    print("- 基本Agent: 简单的查询执行，适合直接的数据查询")
    print("- PRR Agent: 通过计划-执行-反思循环，能够处理更复杂的分析任务")
    print("="*80)


if __name__ == '__main__':
    print("PRR Agent 示例演示")
    print("="*80)
    print("Plan-ReAct-Reflect (PRR) 架构能够:")
    print("  1. Plan: 将复杂问题分解为可执行步骤")
    print("  2. ReAct: 执行推理和行动，调用工具获取数据")
    print("  3. Reflect: 评估进度，决定下一步行动")
    print("="*80)
    
    try:
        # 运行各个演示
        demo_basic_prr()
        
        demo_prr_with_multiple_docs()
        
        # 可选：详细工作流演示（会有很多日志输出）
        # demo_prr_workflow_details()
        
        # 可选：对比演示
        # demo_comparison_with_basic_agent()
        
        print("\n" + "="*80)
        print("所有演示完成！")
        print("="*80)
        
    except Exception as e:
        logger.error(f"演示过程中出错: {str(e)}", exc_info=True)
