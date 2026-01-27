"""
PRR Agent 测试脚本
Test script for Plan-ReAct-Reflect Agent.
"""

import sys
import os
import unittest
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiscal_mind.prr_agent import PRRAgent
from fiscal_mind.parser import ExcelParser
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)


class TestPRRAgent(unittest.TestCase):
    """PRR Agent测试类"""
    
    @classmethod
    def setUpClass(cls):
        """准备测试数据"""
        cls.test_dir = Path(__file__).parent.parent / 'examples'
        cls.test_dir.mkdir(exist_ok=True)
        
        # 创建测试用的区域数据
        cls.regional_file = cls._create_test_regional_data()
        cls.product_file = cls._create_test_product_data()
    
    @classmethod
    def _create_test_regional_data(cls):
        """创建测试用的区域数据"""
        regional_sales = {
            '大区': ['华北', '华东', '华南'] * 4,
            '月份': ['2024-01'] * 3 + ['2024-02'] * 3 + ['2024-03'] * 3 + ['2024-04'] * 3,
            '销售额': [
                1200000, 1500000, 980000,  # 1月
                1250000, 1600000, 1020000,  # 2月
                1300000, 1650000, 1100000,  # 3月
                1350000, 1700000, 1150000,  # 4月
            ],
            '利润': [
                240000, 300000, 196000,  # 1月
                250000, 320000, 204000,  # 2月
                260000, 330000, 220000,  # 3月
                270000, 340000, 230000,  # 4月
            ],
            '订单数': [
                120, 150, 98,  # 1月
                125, 160, 102,  # 2月
                130, 165, 110,  # 3月
                135, 170, 115,  # 4月
            ]
        }
        
        df = pd.DataFrame(regional_sales)
        df['客单价'] = df['销售额'] / df['订单数']
        df['利润率'] = (df['利润'] / df['销售额'] * 100).round(2)
        
        output_path = cls.test_dir / 'test_regional.xlsx'
        df.to_excel(output_path, index=False, sheet_name='区域业绩')
        
        return output_path
    
    @classmethod
    def _create_test_product_data(cls):
        """创建测试用的产品数据"""
        product_data = {
            '产品': ['产品A', '产品B', '产品C'] * 4,
            '季度': ['Q1'] * 3 + ['Q2'] * 3 + ['Q3'] * 3 + ['Q4'] * 3,
            '销售数量': [
                1000, 1200, 800,  # Q1
                1100, 1250, 850,  # Q2
                1150, 1300, 900,  # Q3
                1200, 1350, 950,  # Q4
            ],
            '销售额': [
                500000, 720000, 360000,  # Q1
                550000, 750000, 382500,  # Q2
                575000, 780000, 405000,  # Q3
                600000, 810000, 427500,  # Q4
            ]
        }
        
        df = pd.DataFrame(product_data)
        df['单价'] = (df['销售额'] / df['销售数量']).round(2)
        
        output_path = cls.test_dir / 'test_product.xlsx'
        df.to_excel(output_path, index=False, sheet_name='产品业绩')
        
        return output_path
    
    def test_prr_agent_initialization(self):
        """测试PRR Agent初始化"""
        agent = PRRAgent()
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.parser)
        self.assertIsNotNone(agent.tool_executor)
        self.assertIsNotNone(agent.graph)
    
    def test_load_documents(self):
        """测试加载文档"""
        agent = PRRAgent()
        agent.load_documents([str(self.regional_file)])
        
        self.assertEqual(len(agent.parser.documents), 1)
        self.assertIn('test_regional.xlsx', agent.parser.documents)
    
    def test_regional_comparison_query(self):
        """测试区域对比查询 - 核心功能"""
        agent = PRRAgent()
        agent.load_documents([str(self.regional_file)])
        
        # 测试问题: "哪个大区今年表现更好"
        question = "哪个大区今年表现更好?"
        answer = agent.query(question)
        
        # 验证答案包含关键信息
        self.assertIsNotNone(answer)
        self.assertIn("大区", answer)
        # 答案应该包含某个区域的信息
        self.assertTrue(
            any(region in answer for region in ["华北", "华东", "华南"]),
            "答案应该包含具体的区域信息"
        )
    
    def test_profit_margin_query(self):
        """测试利润率查询"""
        agent = PRRAgent()
        agent.load_documents([str(self.regional_file)])
        
        question = "哪个大区的利润率最高?"
        answer = agent.query(question)
        
        self.assertIsNotNone(answer)
        self.assertIn("利润", answer)
    
    def test_product_query(self):
        """测试产品查询"""
        agent = PRRAgent()
        agent.load_documents([str(self.product_file)])
        
        question = "哪个产品的销售表现最好?"
        answer = agent.query(question)
        
        self.assertIsNotNone(answer)
        # 答案应该包含产品信息
        self.assertTrue(
            any(product in answer for product in ["产品A", "产品B", "产品C"]),
            "答案应该包含具体的产品信息"
        )
    
    def test_multi_document_query(self):
        """测试多文档查询"""
        agent = PRRAgent()
        agent.load_documents([str(self.regional_file), str(self.product_file)])
        
        # 验证加载了两个文档
        self.assertEqual(len(agent.parser.documents), 2)
        
        question = "有哪些数据?"
        answer = agent.query(question)
        
        self.assertIsNotNone(answer)
    
    def test_plan_generation(self):
        """测试计划生成"""
        agent = PRRAgent()
        agent.load_documents([str(self.regional_file)])
        
        # 测试基于规则的计划生成
        query = "哪个大区今年表现更好?"
        plan = agent._rule_based_generate_plan(query, "")
        
        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)
        # 计划应该包含识别、提取、计算、比较等步骤
        plan_text = " ".join(plan)
        self.assertTrue(
            any(keyword in plan_text for keyword in ["识别", "提取", "比较", "分析"]),
            "计划应该包含关键步骤"
        )
    
    def test_react_reasoning(self):
        """测试ReAct推理"""
        agent = PRRAgent()
        agent.load_documents([str(self.regional_file)])
        
        # 测试推理和行动决策
        step_desc = "识别包含区域数据的工作表"
        thought, action = agent._reason_and_act(
            step_description=step_desc,
            query="哪个大区表现更好?",
            context="",
            observations=[]
        )
        
        self.assertIsNotNone(thought)
        self.assertIsNotNone(action)
        self.assertIn("name", action)
        self.assertIn("parameters", action)
    
    def test_reflection_generation(self):
        """测试反思生成"""
        agent = PRRAgent()
        
        # 模拟观察结果
        observations = [{
            "step": 0,
            "step_description": "测试步骤",
            "result": {
                "success": True,
                "data": {"summary": "测试数据"}
            }
        }]
        
        reflection = agent._generate_reflection(
            query="测试问题",
            plan=["步骤1", "步骤2"],
            observations=observations,
            current_step=0
        )
        
        self.assertIsNotNone(reflection)
        self.assertIn("成功", reflection)
    
    def test_max_iterations(self):
        """测试最大迭代次数限制"""
        # 创建限制迭代次数的agent
        agent = PRRAgent(max_iterations=2)
        agent.load_documents([str(self.regional_file)])
        
        question = "哪个大区表现更好?"
        answer = agent.query(question)
        
        # 应该能够在有限迭代内返回答案
        self.assertIsNotNone(answer)
    
    def test_answer_format(self):
        """测试答案格式"""
        agent = PRRAgent()
        agent.load_documents([str(self.regional_file)])
        
        question = "哪个大区今年表现更好?"
        answer = agent.query(question)
        
        # 答案应该包含问题
        self.assertIn("问题", answer)
        # 答案应该是字符串
        self.assertIsInstance(answer, str)
        # 答案不应该为空
        self.assertGreater(len(answer), 0)


class TestPRRAgentWorkflow(unittest.TestCase):
    """测试PRR工作流的各个阶段"""
    
    def setUp(self):
        """每个测试前的准备"""
        self.test_dir = Path(__file__).parent.parent / 'examples'
        self.agent = PRRAgent()
    
    def test_workflow_states(self):
        """测试工作流状态转换"""
        # 验证graph已正确构建
        self.assertIsNotNone(self.agent.graph)
        
        # 验证graph包含所需的节点
        # 注意：LangGraph的节点在编译后不容易直接访问，
        # 这里我们通过执行来验证工作流
        pass
    
    def test_plan_node_execution(self):
        """测试计划节点执行"""
        # 这里测试计划生成逻辑
        query = "哪个大区销售额最高?"
        plan = self.agent._rule_based_generate_plan(query, "")
        
        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestPRRAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestPRRAgentWorkflow))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("="*80)
    print("PRR Agent 测试")
    print("="*80)
    
    result = run_tests()
    
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✓ 所有测试通过!")
    else:
        print(f"✗ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    print("="*80)
    
    sys.exit(0 if result.wasSuccessful() else 1)
