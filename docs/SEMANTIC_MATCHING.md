# 语义匹配功能文档 (Semantic Matching Features)

## 概述 (Overview)

FiscalMind v2.2 引入了全面的语义匹配功能，基于LLM的智能语义解析，提供高可信度的匹配结果。该系统使用静态同义词库作为基础，LLM作为保底机制，确保既快速又准确。

FiscalMind v2.2 introduces comprehensive semantic matching features with LLM-based intelligent semantic resolution, providing high-confidence matching results. The system uses static synonym dictionaries as the foundation, with LLM as a fallback mechanism, ensuring both speed and accuracy.

## 核心组件 (Core Components)

### SemanticResolver 类

语义解析器是整个语义匹配系统的核心，提供以下功能：

1. **列名语义匹配** (Column Semantic Matching)
2. **工作表名语义匹配** (Sheet Name Semantic Matching)
3. **关联键自动发现** (Auto Join Key Discovery)
4. **值标准化** (Value Normalization)
5. **文档名语义匹配** (Document Name Semantic Matching)

## 主要功能 (Key Features)

### 1. 工作表名称匹配 (Sheet Name Matching)

**问题场景**: 用户可能说"看下24年的预算"，但Excel中工作表叫"FY24_Budget"或"2024预算表"。

**解决方案**: 
- 精确匹配 → 包含匹配 → 模糊匹配 → 关键词匹配 → LLM保底
- 支持年份简写识别（如'24' → '2024'）
- 中文关键词智能提取

**示例**:
```python
from fiscal_mind.parser import ExcelParser

parser = ExcelParser()
doc = parser.load_document('budget.xlsx')

# 精确匹配不需要
df = doc.get_sheet('24年预算', use_semantic=True)  # 自动匹配到 '2024年预算表'
```

**测试结果**:
- ✓ '24年的预算' → '2024年财务报表' 或 'FY24_Budget'
- ✓ '员工工资' → '员工薪资_2024'
- ✓ 'sales' → 'Sales_Q1'

### 2. 过滤条件中的列名解析 (Column Name Resolution in Filters)

**问题场景**: LLM生成的过滤器可能包含"利润"，但实际列名是"净利润(万元)"。

**解决方案**:
- 自动进行语义列名匹配
- 支持同义词映射（如：薪资 → 月薪、营收 → 销售额）
- 支持值的模糊匹配（如：北京市 → 北京）

**示例**:
```python
from fiscal_mind.parser import ExcelDocument

doc = ExcelDocument('sales.xlsx')

# 使用语义列名（'营收'会自动匹配到'销售额'）
filters = [
    {'column': '营收', 'operator': '>', 'value': 100000},  # '营收' → '销售额'
    {'column': '区域', 'operator': '==', 'value': '北京'}  # 支持值标准化
]
result = doc.filter_rows_advanced('销售明细', filters, use_semantic=True)
```

**同义词映射示例**:
- 收入/营收/销售额 → revenue, income
- 利润/盈利/净利润 → profit, earnings  
- 工资/薪资/薪酬/月薪 → salary, wage, pay
- 报销/报销单 → reimbursement, expense claim

### 3. 多表关联的键匹配 (Join Key Discovery)

**问题场景**: 跨表关联通常需要匹配ID或名称。一张表叫"员工编号"，另一张叫"工号"。

**解决方案**:
- 自动发现关联键
- 支持同名列匹配
- 支持同义词匹配（员工编号 ↔ 工号）
- 支持数据值重叠检测
- LLM智能推理

**示例**:
```python
from fiscal_mind.parser import ExcelParser

parser = ExcelParser()

# 自动发现并关联
result = parser.join_sheets(
    'employees.xlsx', '员工信息',
    'performance.xlsx', '绩效评分',
    auto_discover=True  # 自动发现 '员工编号' ↔ '工号'
)
```

**发现策略**:
1. 完全相同的列名（优先ID/编号等关键字段）
2. 同义词匹配（如：员工编号 ↔ 工号）
3. 列名相似度（SequenceMatcher >= 0.7）
4. 数据值重叠度（>= 30%）
5. LLM智能分析（保底）

### 4. 查询意图分类 (Query Intent Classification)

**问题场景**: 用户说"帮我算一下平均..."或"对比一下..."，这些表达方式千变万化。

**解决方案**:
- 扩展的关键词库（支持丰富同义词）
- LLM语义分类（保底）
- 支持的意图类型：search, statistics, aggregation, filter, comparison, join, sort, general

**示例**:
```python
from fiscal_mind.agent import TableDocumentAgent
from langchain_openai import ChatOpenAI

# 带LLM的Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = TableDocumentAgent(llm=llm)

# 自动识别意图并执行
agent.query("帮我计算一下各部门的平均工资")  # 自动识别为 statistics
agent.query("对比一下华东和华南的销售额")      # 自动识别为 comparison
```

**关键词库扩展**:
- 统计类: 统计、汇总、总结、概览、计算、算一下、平均
- 对比类: 对比、比较、差异、区别
- 筛选类: 过滤、筛选、选择、条件、满足
- 排序类: 排序、排列、最大、最小、top

### 5. 单元格值的模糊匹配 (Categorical Value Matching)

**问题场景**: 用户查询"北京的销售额"，但数据列里存的是"北京市"或"Beijing"。

**解决方案**:
- 地名标准化映射
- 自动值规范化
- 在过滤时自动应用

**示例**:
```python
from fiscal_mind.semantic_resolver import SemanticResolver

resolver = SemanticResolver()

# 地名标准化
resolver.normalize_value('北京市', category='location')  # → '北京'
resolver.normalize_value('Beijing', category='location')  # → '北京'
resolver.normalize_value('华东地区', category='location') # → '华东'

# 在过滤中自动应用
filters = [
    {'column': '城市', 'operator': '==', 'value': '北京'}  # 匹配 '北京市', 'Beijing'
]
result = doc.filter_rows_advanced('销售', filters, use_semantic=True)
```

**支持的地名**:
- 北京/北京市/Beijing
- 上海/上海市/Shanghai
- 广州/广州市/Guangzhou
- 深圳/深圳市/Shenzhen
- 华东/华东地区/East China
- 华南/华南地区/South China
- 华北/华北地区/North China
- 华中/华中地区/Central China

### 6. 文档摘要与路径匹配 (File Name Mapping)

**问题场景**: 当用户提到"报销单"时，程序需要从多个已加载文档（如reimbursement_v2.xlsx）中定位文件。

**解决方案**:
- 文档名语义匹配
- 同义词支持（报销 ↔ reimbursement）

**示例**:
```python
from fiscal_mind.parser import ExcelParser

parser = ExcelParser()
parser.load_documents([
    'financial_report_2024.xlsx',
    'reimbursement_v2.xlsx',
    'employee_salary.xlsx'
])

# 语义匹配文档名
doc = parser.get_document('报销单', use_semantic=True)  # → reimbursement_v2.xlsx
doc = parser.get_document('工资', use_semantic=True)    # → employee_salary.xlsx
```

## 配置与使用 (Configuration and Usage)

### 基础使用（无LLM）

```python
from fiscal_mind.parser import ExcelParser
from fiscal_mind.semantic_resolver import SemanticResolver

# 创建解析器（使用默认语义解析器）
parser = ExcelParser()

# 所有语义功能自动可用
doc = parser.load_document('data.xlsx')
df = doc.get_sheet('工资', use_semantic=True)  # 语义匹配
```

### 高级使用（带LLM）

```python
from fiscal_mind.parser import ExcelParser
from langchain_openai import ChatOpenAI
import os

# 设置OpenAI API密钥
os.environ['OPENAI_API_KEY'] = 'your-api-key'

# 创建LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 创建解析器（带LLM支持）
parser = ExcelParser(llm=llm)

# 当静态匹配置信度不足时，自动使用LLM
doc = parser.load_document('data.xlsx')
df = doc.get_sheet('复杂的工作表名称', use_semantic=True)
```

### 自定义置信度阈值

```python
from fiscal_mind.semantic_resolver import SemanticResolver

# 自定义置信度阈值（默认0.7）
resolver = SemanticResolver(llm=llm, confidence_threshold=0.8)

# 手动使用
columns = resolver.find_column_by_semantic(df, '销售', use_llm_fallback=True)
```

## 性能与可信度 (Performance and Confidence)

### 匹配策略层次

所有语义匹配遵循以下优先级：

1. **精确匹配** (Exact Match) - 100% 可信度
2. **包含匹配** (Contains Match) - 90% 可信度
3. **同义词匹配** (Synonym Match) - 85% 可信度
4. **模糊匹配** (Fuzzy Match) - 60-100% 可信度
5. **LLM匹配** (LLM Fallback) - 使用GPT-4等大模型

### 置信度阈值

- 默认阈值：0.7
- 低于阈值时触发LLM保底
- 可自定义配置

### 性能考虑

- 静态匹配：毫秒级
- LLM匹配：秒级（仅在低置信度时触发）
- 建议：为常见业务场景扩展同义词库

## 同义词库 (Synonym Dictionary)

当前支持的财务/业务领域同义词：

| 中文概念 | 中文同义词 | English |
|---------|----------|---------|
| 收入 | 营收、销售额、营业收入、收益、进账 | revenue, income |
| 利润 | 盈利、净利润、毛利、利益 | profit, earnings |
| 成本 | 费用、开支、支出、花费 | cost, expense |
| 工资 | 薪资、薪酬、报酬、薪水、月薪、年薪 | salary, wage, pay |
| 员工 | 人员、职工、工作人员 | employee, staff |
| 编号 | ID、工号、员工编号、代码 | code, number |
| 报销 | 报销单、费用报销 | reimbursement, expense claim |
| 预算 | 预算表 | budget |
| 报表 | 报告、财务报表 | report, financial report |

（可扩展更多领域词汇）

## 最佳实践 (Best Practices)

### 1. 优先使用静态匹配

```python
# 好：快速且准确
filters = [{'column': '销售额', 'operator': '>', 'value': 100000}]

# 避免：不必要的LLM调用
# 如果列名已知，直接使用精确名称
```

### 2. 扩展同义词库

```python
# 为特定业务场景扩展同义词
resolver = SemanticResolver()
resolver.SYNONYM_MAP['自定义概念'] = ['同义词1', '同义词2', 'synonym']
```

### 3. 合理设置置信度阈值

```python
# 对于关键业务逻辑，提高阈值
resolver = SemanticResolver(confidence_threshold=0.85)

# 对于探索性分析，降低阈值
resolver = SemanticResolver(confidence_threshold=0.6)
```

### 4. 使用语义开关

```python
# 对于确定的场景，关闭语义匹配以提高性能
df = doc.get_sheet('精确表名', use_semantic=False)

# 对于用户输入或LLM生成的查询，启用语义匹配
df = doc.get_sheet(user_input, use_semantic=True)
```

## 测试覆盖 (Test Coverage)

详见 `tests/test_semantic_resolver.py`：

- ✓ 列名语义匹配（精确、同义词、模糊）
- ✓ 工作表名匹配（关键词、年份识别）
- ✓ 关联键自动发现（同名、同义词、数据重叠）
- ✓ 值标准化（地名、区域）
- ✓ Parser集成测试
- ✓ 文档名语义匹配

## 限制与未来改进 (Limitations and Future Improvements)

### 当前限制

1. 同义词库需手动维护
2. 中文分词依赖简单规则
3. 值标准化目前仅支持地名

### 未来改进方向

1. **词向量集成**: 使用词嵌入进行更智能的语义匹配
2. **领域自适应**: 自动学习业务领域的术语
3. **多语言支持**: 扩展到更多语言
4. **缓存机制**: 缓存LLM结果以提高性能
5. **用户反馈学习**: 从用户修正中学习

## 示例：完整工作流 (Complete Workflow Example)

```python
from fiscal_mind.parser import ExcelParser
from fiscal_mind.agent import TableDocumentAgent
from langchain_openai import ChatOpenAI
import os

# 1. 设置环境
os.environ['OPENAI_API_KEY'] = 'your-api-key'
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 2. 创建支持语义匹配的Parser
parser = ExcelParser(llm=llm)

# 3. 加载文档（语义匹配已启用）
parser.load_documents([
    'FY2024_Budget.xlsx',
    '员工薪资_2024Q1.xlsx',
    'reimbursement_data.xlsx'
])

# 4. 语义查询工作表
budget_doc = parser.get_document('24年预算', use_semantic=True)
df_budget = budget_doc.get_sheet('预算', use_semantic=True)

# 5. 语义过滤
filters = [
    {'column': '营收', 'operator': '>', 'value': 1000000},  # 自动匹配到'销售额'
    {'column': '区域', 'operator': 'in', 'value': ['北京', '上海']}  # 值标准化
]
result = budget_doc.filter_rows_advanced('数据', filters, use_semantic=True)

# 6. 自动关联
salary_doc = parser.get_document('工资', use_semantic=True)
joined = parser.join_sheets(
    budget_doc.file_name, '数据',
    salary_doc.file_name, '薪资',
    auto_discover=True  # 自动发现关联键
)

# 7. 使用Agent进行自然语言查询
agent = TableDocumentAgent(parser=parser, llm=llm)
response = agent.query("帮我对比一下各部门的平均工资和预算差异")
```

## 总结 (Summary)

FiscalMind v2.2的语义匹配功能提供了：

✅ **高可信度**: 静态+LLM双保险机制  
✅ **高性能**: 优先使用快速的静态匹配  
✅ **易用性**: 开箱即用，无需额外配置  
✅ **可扩展**: 支持自定义同义词和LLM  
✅ **全面覆盖**: 7大场景的语义匹配支持  

这使得FiscalMind能够更好地理解用户意图，处理复杂的Excel文档结构，提供更智能的财务数据分析体验。
