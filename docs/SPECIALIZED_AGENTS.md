# 专业智能体架构文档

## 概述

为了解决现有Agent编排方式无法深入理解文档背后业务逻辑的问题，FiscalMind v3.0 引入了三个专业智能体，它们协作提供更深入、更专业的数据分析。

## 问题背景

原有的PRR Agent虽然能够通过Plan-ReAct-Reflect循环执行复杂任务，但存在以下问题：

1. **缺乏业务理解**：AI无法理解数据背后的业务含义
2. **思考过程简单**：分析过程不够深入，缺少业务视角
3. **结论缺乏验证**：没有对最终结论进行质量评估

原始日志示例显示，AI的思考过程过于简单：
```
正在分析您的问题...
2026-01-28 00:05:36,536 - fiscal_mind.prr_agent - INFO - Generated plan with 5 steps
2026-01-28 00:05:36,536 - fiscal_mind.prr_agent - INFO -   Step 1: 识别包含区域信息的数据表
2026-01-28 00:05:36,536 - fiscal_mind.prr_agent - INFO -   Step 2: 从"汇总"表中提取各区域的预算、实际和差额数据
...
```

## 解决方案：三个专业智能体

### Agent A: 业务分析智能体 (BusinessAnalysisAgent)

**职责**：
- 获取所有表头和描述信息
- 分析少量示例数据
- 识别可能的业务场景
- 理解数据背后的业务逻辑

**核心能力**：
```python
{
    "business_domain": "财务/销售/人力资源/医疗等",
    "key_dimensions": ["区域", "产品", "时间"等],
    "key_metrics": ["销售额", "利润", "效率"等],
    "business_relationships": "业务关系描述",
    "analysis_scenarios": ["绩效评估", "预算对比", "趋势分析"等],
    "business_context": "业务背景和含义",
    "confidence_level": "high/medium/low"
}
```

**示例输出**：
```
业务领域: 财务
关键维度: ['大区', '月份']
关键指标: ['预算', '实际', '差额', '覆盖率']
分析场景: ['预算执行分析', '区域绩效对比']
```

### Agent B: 批评者智能体 (CriticAgent)

**职责**：
- 评估业务分析智能体的结论
- 检查分析是否与用户问题匹配
- 提供改进建议
- 与业务分析智能体多轮协作

**核心能力**：
```python
{
    "match_score": 0-10,  # 匹配度评分
    "matches": ["匹配的方面"],
    "gaps": ["缺失的方面"],
    "suggestions": ["改进建议"],
    "needs_refinement": true/false,
    "reasoning": "评估理由"
}
```

**多轮协作机制**：
- 最多3轮协作
- 每轮Business Agent分析 → Critic Agent评估 → 根据建议改进
- 当匹配度≥8分或达到最大轮数时结束

### Agent C: 评判智能体 (JudgmentAgent)

**职责**：
- 结合业务分析评估最终结论
- 验证结论与实际数据的一致性
- 判断分析的质量和可信度
- 提供改进建议

**核心能力**：
```python
{
    "overall_quality": 0-10,      # 总体质量
    "answer_relevance": 0-10,     # 问题相关性
    "data_support": 0-10,         # 数据支撑
    "business_logic": 0-10,       # 业务逻辑
    "comprehensiveness": 0-10,    # 全面性
    "strengths": ["优点列表"],
    "weaknesses": ["不足列表"],
    "is_acceptable": true/false,
    "improvement_suggestions": ["改进建议"],
    "final_verdict": "最终评语"
}
```

## 增强的工作流

### 原始PRR工作流
```
用户查询 → 加载上下文 → 计划 → 执行 → 反思 → 生成答案 → END
```

### 增强的PRR工作流（集成三个智能体）
```
用户查询
    ↓
加载数据上下文
    ↓
┌─────────────────────────────────────────┐
│  业务分析阶段                              │
│  - Business Agent: 分析业务场景            │
│  - Critic Agent: 评估分析质量              │
│  - 多轮协作直到满意 (最多3轮)               │
└─────────────────────────────────────────┘
    ↓
基于业务理解生成执行计划 (Plan)
    ↓
┌─────────────────────────────────────────┐
│  ReAct循环                                │
│  - 推理：思考当前步骤                       │
│  - 行动：调用工具获取数据                   │
│  - 观察：记录执行结果                       │
│  - 反思：评估进度                          │
└─────────────────────────────────────────┘
    ↓
生成结合业务分析的最终答案
    ↓
┌─────────────────────────────────────────┐
│  评判阶段                                  │
│  - Judgment Agent: 评估结论质量            │
│  - 验证数据支撑                            │
│  - 检查业务逻辑                            │
│  - 提供改进建议                            │
└─────────────────────────────────────────┘
    ↓
返回最终结果
```

## 使用方式

### 基本使用（不使用LLM）

```python
from fiscal_mind.prr_agent import PRRAgent

# 创建增强的PRR Agent（自动集成三个专业智能体）
agent = PRRAgent(llm=None)

# 加载文档
agent.load_documents(['data.xlsx'])

# 提问
answer = agent.query("哪个大区今年表现更好？")
print(answer)
```

### 使用LLM增强

```python
from fiscal_mind.prr_agent import PRRAgent
from langchain_openai import ChatOpenAI

# 使用LLM创建更智能的Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = PRRAgent(llm=llm)

# 加载文档
agent.load_documents(['data.xlsx'])

# 提问 - LLM会提供更深入的业务分析
answer = agent.query("哪个大区今年表现更好？")
print(answer)
```

### 独立使用专业智能体

```python
from fiscal_mind.specialized_agents import (
    BusinessAnalysisAgent,
    CriticAgent,
    JudgmentAgent,
    collaborate_business_and_critic
)

# 1. 业务分析
business_agent = BusinessAnalysisAgent(llm=None)
analysis = business_agent.analyze_business_context(context, sample_data)

# 2. 业务分析 + 批评者协作
business_agent = BusinessAnalysisAgent(llm=None)
critic_agent = CriticAgent(llm=None)
final_analysis, history = collaborate_business_and_critic(
    business_agent, 
    critic_agent,
    user_query="哪个大区表现更好？",
    context=data_context,
    sample_data=samples,
    max_rounds=3
)

# 3. 评判结论
judgment_agent = JudgmentAgent(llm=None)
judgment = judgment_agent.judge_conclusion(
    user_query=query,
    business_analysis=analysis,
    final_conclusion=answer,
    actual_data=data
)
```

## 实际效果对比

### 原始版本
```
日志显示思考过程简单:
- Step 1: 识别包含区域信息的数据表
- Step 2: 提取各区域的预算、实际和差额数据
- 直接生成答案，缺少业务理解
```

### 增强版本
```
日志显示完整的智能协作过程:

1. 业务分析阶段:
   - Business Analysis Agent: Analyzing business context...
   - Domain: 财务
   - Key Dimensions: 大区, 月份
   - Key Metrics: 预算, 实际, 差额, 覆盖率
   - Analysis Scenarios: 预算执行分析, 区域绩效对比
   - Collaboration Rounds: 3

2. 增强的计划生成:
   - Step 1: 识别包含区域(大区)数据的工作表，如"人员"、"医保"、"准入情况"和"费用效率分析"
   - Step 2: 从"汇总"表中提取各区域的预算、实际和差额数据作为核心表现指标
   - Step 3: 结合"医保"和"准入情况"表，分析各区域的医院准入数量和医保覆盖情况
   - Step 4: 利用"费用效率分析"和"人员"表计算各区域的费用投入与产出效率
   - Step 5: 综合各项指标对比分析，确定表现最好的大区并解释原因

3. 评判结果:
   - Overall Quality: 9.0/10
   - Answer Relevance: 8/10
   - Data Support: 8/10
   - Business Logic: 10/10
   - Strengths: 明确给出了对比结论, 结论包含具体数据, 提到了关键维度...
```

## 主要改进

1. **深入的业务理解**：通过业务分析智能体，系统能够理解数据的业务含义，而不仅仅是简单的数据提取

2. **质量保证机制**：批评者智能体确保分析方向正确，评判智能体验证结论质量

3. **多轮协作优化**：业务分析智能体和批评者智能体可以多轮协作，持续改进分析质量

4. **全面的评估体系**：从问题相关性、数据支撑、业务逻辑、全面性等多个维度评估结论

5. **透明的思考过程**：详细的日志输出展示了AI的完整思考过程，提高了可解释性

## 技术细节

### 状态管理

扩展了PRRAgentState以支持专业智能体：

```python
class PRRAgentState(TypedDict):
    # ... 原有字段 ...
    
    # 专业智能体相关
    business_analysis: Optional[Dict[str, Any]]  # 业务分析结果
    critique_history: List[Dict[str, Any]]       # 批评历史
    collaboration_rounds: int                     # 协作轮数
    judgment_result: Optional[Dict[str, Any]]    # 评判结果
```

### 工作流节点

新增两个关键节点：

1. **business_analysis_node**: 在加载上下文后、生成计划前执行
2. **judgment_node**: 在生成最终答案后执行

### LLM集成

所有三个智能体都支持LLM增强：
- 无LLM：使用基于规则的分析（适合快速测试）
- 有LLM：使用深度语义分析（适合生产环境）

## 测试

运行测试套件：

```bash
python tests/test_enhanced_prr.py
```

测试包括：
1. 专业智能体独立功能测试
2. 增强的PRR Agent集成测试
3. 原始版本与增强版本对比

## 总结

通过引入三个专业智能体（业务分析、批评者、评判），FiscalMind v3.0 显著提升了对复杂文档的理解和分析能力：

- ✅ **业务理解**：从数据中提取业务含义
- ✅ **质量保证**：多轮协作和评判机制
- ✅ **深度分析**：从业务视角生成更专业的计划
- ✅ **可解释性**：完整的思考过程日志
- ✅ **灵活性**：支持LLM增强或纯规则模式

这种智能体编排方式使得AI的分析过程更加系统化、专业化，能够更好地服务于财务BP的复杂分析需求。
