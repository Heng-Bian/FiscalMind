"""
测试增强的PRR Agent与专业智能体集成
Test enhanced PRR Agent with specialized agents integration
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.prr_agent import PRRAgent
from fiscal_mind.specialized_agents import BusinessAnalysisAgent, CriticAgent, JudgmentAgent
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_data():
    """创建测试数据"""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # 创建医疗大区数据示例
    data = {
        '大区': ['华北', '华东', '华南', '华西', '华中'] * 6,
        '月份': ['2024-01'] * 5 + ['2024-02'] * 5 + ['2024-03'] * 5 + 
                ['2024-04'] * 5 + ['2024-05'] * 5 + ['2024-06'] * 5,
        '预算(万元)': [
            # 1月
            1000, 1000, 800, 700, 900,
            # 2月
            1000, 1000, 800, 700, 900,
            # 3月
            1000, 1000, 800, 700, 900,
            # 4月
            1000, 1000, 800, 700, 900,
            # 5月
            1000, 1000, 800, 700, 900,
            # 6月
            1000, 1000, 800, 700, 900,
        ],
        '实际(万元)': [
            # 1月 - 华东超支
            950, 1200, 750, 680, 870,
            # 2月
            980, 1250, 770, 690, 880,
            # 3月
            1000, 1300, 790, 700, 900,
            # 4月
            1020, 1350, 810, 710, 920,
            # 5月
            1050, 1400, 830, 720, 940,
            # 6月
            1100, 1500, 860, 750, 980,
        ],
        '医院准入数量': [
            # 1月
            12, 18, 10, 8, 11,
            # 2月
            12, 18, 10, 8, 11,
            # 3月
            13, 18, 11, 8, 12,
            # 4月
            13, 18, 11, 9, 12,
            # 5月
            14, 18, 12, 9, 13,
            # 6月
            14, 18, 12, 10, 13,
        ],
        '医保覆盖率(%)': [
            # 1月
            85, 95, 80, 70, 82,
            # 2月
            85, 95, 80, 70, 82,
            # 3月
            87, 95, 82, 72, 84,
            # 4月
            87, 95, 82, 72, 84,
            # 5月
            88, 95, 84, 74, 86,
            # 6月
            90, 95, 85, 75, 87,
        ],
    }
    
    df = pd.DataFrame(data)
    df['差额(万元)'] = df['实际(万元)'] - df['预算(万元)']
    df['预算执行率(%)'] = (df['实际(万元)'] / df['预算(万元)'] * 100).round(2)
    
    # 保存到Excel
    output_path = test_dir / 'medical_regional_performance.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='汇总', index=False)
        
        # 创建汇总表
        summary = df.groupby('大区').agg({
            '预算(万元)': 'sum',
            '实际(万元)': 'sum',
            '差额(万元)': 'sum',
            '医院准入数量': 'last',
            '医保覆盖率(%)': 'last'
        }).reset_index()
        summary['总预算执行率(%)'] = (summary['实际(万元)'] / summary['预算(万元)'] * 100).round(2)
        summary.to_excel(writer, sheet_name='大区汇总', index=False)
    
    logger.info(f"创建测试数据: {output_path}")
    return output_path


def test_specialized_agents():
    """测试专业智能体"""
    print("\n" + "="*80)
    print("测试 1: 专业智能体独立功能测试")
    print("="*80)
    
    # 模拟上下文
    context = """
    文档包含以下工作表:
    - 汇总: 包含大区、月份、预算(万元)、实际(万元)、差额(万元)、医院准入数量、医保覆盖率(%)
    - 大区汇总: 包含各大区的总计数据
    
    列名: 大区、月份、预算(万元)、实际(万元)、差额(万元)、医院准入数量、医保覆盖率(%)、预算执行率(%)
    """
    
    sample_data = [
        {"大区": "华东", "预算": 1000, "实际": 1200, "医院准入数量": 18, "医保覆盖率": 95},
        {"大区": "华北", "预算": 1000, "实际": 950, "医院准入数量": 12, "医保覆盖率": 85},
    ]
    
    # 测试业务分析智能体
    print("\n测试业务分析智能体:")
    print("-" * 80)
    business_agent = BusinessAnalysisAgent(llm=None)
    business_analysis = business_agent.analyze_business_context(context, sample_data)
    print(f"业务领域: {business_analysis['business_domain']}")
    print(f"关键维度: {business_analysis['key_dimensions']}")
    print(f"关键指标: {business_analysis['key_metrics']}")
    print(f"分析场景: {business_analysis['analysis_scenarios']}")
    
    # 测试批评者智能体
    print("\n测试批评者智能体:")
    print("-" * 80)
    critic_agent = CriticAgent(llm=None)
    user_query = "哪个大区今年表现更好？"
    critique = critic_agent.critique_analysis(user_query, business_analysis)
    print(f"匹配度评分: {critique['match_score']}/10")
    print(f"匹配点: {critique['matches']}")
    print(f"缺失点: {critique['gaps']}")
    print(f"建议: {critique['suggestions']}")
    
    # 测试评判智能体
    print("\n测试评判智能体:")
    print("-" * 80)
    judgment_agent = JudgmentAgent(llm=None)
    final_conclusion = """
    根据分析，华东大区表现最好。原因如下：
    - 预算执行率最高：实际支出为1.2亿元，预算为1.0亿元，差额+20%
    - 医院准入数量最多：共准入18家医院
    - 医保覆盖最广：医保覆盖率达95%
    """
    judgment = judgment_agent.judge_conclusion(user_query, business_analysis, final_conclusion, sample_data)
    print(f"总体质量: {judgment['overall_quality']}/10")
    print(f"问题相关性: {judgment['answer_relevance']}/10")
    print(f"数据支撑: {judgment['data_support']}/10")
    print(f"业务逻辑: {judgment['business_logic']}/10")
    print(f"是否可接受: {judgment['is_acceptable']}")
    if judgment['strengths']:
        print(f"优点: {', '.join(judgment['strengths'])}")
    if judgment['weaknesses']:
        print(f"不足: {', '.join(judgment['weaknesses'])}")


def test_enhanced_prr_agent():
    """测试增强的PRR Agent"""
    print("\n" + "="*80)
    print("测试 2: 增强的PRR Agent集成测试")
    print("="*80)
    
    # 创建测试数据
    data_file = create_test_data()
    
    # 创建增强的PRR Agent（不使用LLM，仅测试流程）
    print("\n创建PRR Agent（集成专业智能体）...")
    agent = PRRAgent(llm=None)  # 不使用LLM，使用规则方法
    
    # 加载文档
    print("加载测试文档...")
    agent.load_documents([str(data_file)])
    
    # 提问
    question = "哪个大区今年表现更好？"
    print(f"\n用户问题: {question}")
    print("-" * 80)
    print("正在分析...")
    print("注意观察日志中的智能体协作过程:")
    print("  1. 业务分析智能体 + 批评者智能体 协作")
    print("  2. 基于业务理解生成执行计划")
    print("  3. 执行计划并收集数据")
    print("  4. 生成结合业务分析的最终答案")
    print("  5. 评判智能体评估结论质量")
    print("-" * 80)
    
    # 执行查询
    answer = agent.query(question)
    
    print("\n最终答案:")
    print("="*80)
    print(answer)
    print("="*80)


def test_comparison_with_original():
    """对比原始PRR Agent和增强版的差异"""
    print("\n" + "="*80)
    print("测试 3: 对比原始版本和增强版本")
    print("="*80)
    
    data_file = create_test_data()
    question = "哪个大区今年表现更好？"
    
    print("\n说明:")
    print("原始PRR Agent: 简单的计划-执行-反思循环")
    print("增强PRR Agent: 集成业务分析、批评者、评判三个专业智能体")
    print("-" * 80)
    
    # 增强版（已集成专业智能体）
    print("\n使用增强的PRR Agent（集成专业智能体）:")
    print("-" * 80)
    enhanced_agent = PRRAgent(llm=None)
    enhanced_agent.load_documents([str(data_file)])
    enhanced_answer = enhanced_agent.query(question)
    print(enhanced_answer)
    
    print("\n" + "="*80)
    print("对比总结:")
    print("- 增强版在执行前先进行了深入的业务分析")
    print("- 批评者智能体确保分析方向与问题匹配")
    print("- 评判智能体在最后评估结论的合理性和质量")
    print("- 整体思考过程更加系统和专业")
    print("="*80)


if __name__ == '__main__':
    print("增强的PRR Agent测试套件")
    print("="*80)
    print("本测试展示了专业智能体（业务分析、批评者、评判）的集成")
    print("="*80)
    
    try:
        # 运行测试
        test_specialized_agents()
        
        test_enhanced_prr_agent()
        
        test_comparison_with_original()
        
        print("\n" + "="*80)
        print("所有测试完成！")
        print("="*80)
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}", exc_info=True)
