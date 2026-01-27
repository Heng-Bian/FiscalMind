# Plan-ReAct-Reflect (PRR) 架构文档

## 概述

Plan-ReAct-Reflect (PRR) 是一种先进的智能体架构，特别适合处理复杂的财务分析问题。该架构通过三个核心阶段的循环迭代，实现了智能化的问题求解。

## 核心架构

### 1. Plan (计划阶段)

**目标**: 将复杂查询分解为可执行的步骤序列

**工作原理**:
- 分析用户查询的意图
- 识别查询类型（比较、统计、趋势等）
- 生成3-5个具体执行步骤
- 考虑之前的观察和反思（如需重新规划）

**示例**:
```
用户问题: "哪个大区今年表现更好？"

生成的计划:
1. 识别包含区域(大区)数据的工作表
2. 提取各个区域的关键业绩指标(如销售额、利润等)
3. 计算各区域的总计或平均值
4. 比较各区域的表现，找出最优者
5. 生成详细的对比分析结果
```

### 2. ReAct (推理-行动阶段)

**目标**: 基于计划执行推理和具体行动

**工作原理**:
- **Reasoning (推理)**: 理解当前步骤的要求，思考需要什么工具
- **Acting (行动)**: 调用合适的工具执行具体操作
- **Observation (观察)**: 记录工具执行结果和关键信息

**示例**:
```
当前步骤: "提取各个区域的关键业绩指标"

Reasoning: 需要获取包含区域信息的数据
Acting: 调用 get_sheet_data 工具
Observation: 成功获取30条区域业绩数据
```

### 3. Reflect (反思阶段)

**目标**: 评估执行进度，决定下一步行动

**工作原理**:
- 评估最近执行的成功/失败
- 分析是否获得了有用信息
- 判断是否需要调整计划
- 决定继续/重新规划/完成

**示例**:
```
反思: 步骤 2/5 执行成功。获得了30条数据记录。
决策: 继续执行下一步
```

## PRR工作流程图

```
┌─────────────────┐
│  用户提问       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  加载数据上下文 │
└────────┬────────┘
         │
         ▼
    ┌────────┐
    │  Plan  │ ◄────────────┐
    └────┬───┘              │
         │                  │
         ▼                  │
    ┌────────┐              │
    │ ReAct  │              │ 需要重新规划
    └────┬───┘              │
         │                  │
         ▼                  │
    ┌─────────┐             │
    │ Reflect │─────────────┘
    └────┬────┘
         │
         ▼
    ┌──────────┐
    │ 决策节点 │
    └────┬─────┘
         │
    ┌────┴────┬────────┬─────────┐
    │         │        │         │
    ▼         ▼        ▼         ▼
 继续下一步  重新规划  完成    达到最大迭代
    │         │        │         │
    └────┬────┴────┬───┴────┬────┘
         │         │        │
         ▼         ▼        ▼
    返回ReAct  返回Plan  生成答案
                           │
                           ▼
                      ┌──────────┐
                      │ 返回结果 │
                      └──────────┘
```

## 使用示例

### 基础用法

```python
from fiscal_mind.prr_agent import PRRAgent

# 创建PRR Agent
agent = PRRAgent()

# 加载数据
agent.load_documents(['regional_performance.xlsx'])

# 提问
answer = agent.query("哪个大区今年表现更好?")
print(answer)
```

### 配置LLM增强

```python
from langchain_openai import ChatOpenAI
from fiscal_mind.prr_agent import PRRAgent

# 配置LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 创建带LLM的PRR Agent
agent = PRRAgent(llm=llm, max_iterations=10)

# LLM会增强以下能力:
# - 更智能的计划生成
# - 更自然的答案生成
# - 更准确的意图理解
```

### 自定义迭代次数

```python
# 创建限制迭代次数的agent
agent = PRRAgent(max_iterations=5)

# 对于简单问题，可能2-3次迭代就够了
# 对于复杂分析，可能需要更多迭代
```

## 适用场景

PRR架构特别适合以下类型的问题:

### 1. 对比分析类
- "哪个大区今年表现更好？"
- "哪个产品的利润率最高？"
- "销售额增长最快的是哪个区域？"

### 2. 多维分析类
- "对比各大区的销售额和利润率，找出综合表现最好的大区"
- "分析各产品线在不同区域的表现差异"

### 3. 趋势分析类
- "哪个部门的业绩增长趋势最好？"
- "过去6个月销售额变化最大的是哪个区域？"

### 4. 复杂聚合类
- "按区域和产品双维度统计销售数据"
- "计算各大区的月均增长率"

## 与其他Agent的对比

| 特性 | 基础Agent | Enhanced Agent | PRR Agent |
|------|-----------|----------------|-----------|
| 查询理解 | 关键词匹配 | LLM意图识别 | 智能规划 |
| 执行方式 | 单次执行 | 多步推理 | 迭代优化 |
| 错误处理 | 返回错误 | 重试机制 | 自动调整计划 |
| 复杂分析 | ❌ | ✅ | ✅✅ |
| 自我优化 | ❌ | ❌ | ✅ |
| 适用场景 | 简单查询 | 一般分析 | 复杂分析 |

## 技术细节

### 状态管理

PRR Agent使用以下状态来追踪执行过程:

```python
class PRRAgentState(TypedDict):
    # 基础信息
    messages: List[BaseMessage]
    query: str
    context: str
    
    # Plan相关
    plan: List[str]              # 执行计划
    current_step: int            # 当前步骤
    
    # ReAct相关
    current_thought: str         # 当前思考
    current_action: Dict         # 当前动作
    observations: List[Dict]     # 观察列表
    
    # Reflect相关
    reflections: List[str]       # 反思记录
    needs_replan: bool           # 是否需要重新规划
    
    # 控制
    iteration: int               # 迭代次数
    final_answer: str            # 最终答案
```

### 工具集成

PRR Agent复用了现有的工具执行器:
- `get_document_summary`: 获取文档摘要
- `get_sheet_data`: 获取工作表数据
- `aggregate_data`: 数据聚合
- `get_top_n`: 获取Top N记录
- `filter_data`: 数据过滤
- 等等...

### 决策逻辑

PRR Agent的决策基于以下因素:
1. **迭代次数**: 是否达到最大迭代限制
2. **计划完成度**: 是否完成所有计划步骤
3. **观察质量**: 是否获得足够的有效信息
4. **执行状态**: 最近的执行是否成功
5. **反思结果**: 是否需要调整策略

## 最佳实践

### 1. 选择合适的迭代次数
```python
# 简单查询
agent = PRRAgent(max_iterations=5)

# 复杂分析
agent = PRRAgent(max_iterations=10)
```

### 2. 提供清晰的数据
- 确保Excel文件有清晰的列名
- 使用一致的数据格式
- 包含必要的维度字段（如区域、产品等）

### 3. 明确的问题表述
```python
# 好的问题
"哪个大区今年的销售额最高？"
"对比各区域的利润率，找出表现最好的"

# 不太好的问题
"数据怎么样？"
"分析一下"
```

### 4. 配合LLM使用
对于复杂的自然语言查询，建议配置LLM:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = PRRAgent(llm=llm)
```

## 扩展开发

### 自定义计划生成器

```python
class CustomPRRAgent(PRRAgent):
    def _rule_based_generate_plan(self, query: str, context: str) -> List[str]:
        # 自定义计划生成逻辑
        if "特殊关键词" in query:
            return ["步骤1", "步骤2", "步骤3"]
        return super()._rule_based_generate_plan(query, context)
```

### 自定义推理逻辑

```python
class CustomPRRAgent(PRRAgent):
    def _reason_and_act(self, step_description: str, query: str, 
                       context: str, observations: List[Dict]) -> tuple:
        # 自定义推理和行动逻辑
        if "特定条件" in step_description:
            thought = "自定义思考"
            action = {"name": "custom_tool", "parameters": {}}
            return thought, action
        return super()._reason_and_act(step_description, query, context, observations)
```

### 自定义反思策略

```python
class CustomPRRAgent(PRRAgent):
    def _generate_reflection(self, query: str, plan: List[str],
                           observations: List[Dict], current_step: int) -> str:
        # 自定义反思逻辑
        reflection = super()._generate_reflection(query, plan, observations, current_step)
        
        # 添加自定义评估
        if self._custom_evaluation(observations):
            reflection += " -> 建议调整策略"
            
        return reflection
```

## 性能优化

### 1. 控制数据量
```python
# 限制每次查询的数据量
agent.query("查询前100条记录")
```

### 2. 缓存机制
```python
# 对于重复的文档加载，可以缓存parser
parser = ExcelParser()
parser.load_documents([...])

agent1 = PRRAgent(parser=parser)
agent2 = PRRAgent(parser=parser)  # 复用同一个parser
```

### 3. 并行处理
```python
# 对于独立的查询，可以并行处理
from concurrent.futures import ThreadPoolExecutor

questions = ["问题1", "问题2", "问题3"]
with ThreadPoolExecutor(max_workers=3) as executor:
    answers = list(executor.map(agent.query, questions))
```

## 故障排查

### 问题1: 答案不准确
**原因**: 可能是计划生成不合理
**解决**: 
- 配置LLM增强计划生成
- 检查数据质量
- 增加迭代次数

### 问题2: 执行超时
**原因**: 迭代次数过多
**解决**:
- 减少max_iterations
- 优化数据量
- 简化查询问题

### 问题3: 找不到相关数据
**原因**: 工作表或列名不匹配
**解决**:
- 使用标准化的列名
- 启用语义匹配
- 检查数据文件结构

## 参考资料

- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
- [ReAct论文](https://arxiv.org/abs/2210.03629)
- [FiscalMind主文档](../README.md)
