# 语义匹配功能文档 (Semantic Matching Features)

## 概述 (Overview)

FiscalMind v3.0 采用全新的基于大语言模型的智能语义匹配系统。系统利用LLM强大的语义理解能力，结合表头、数据类型、样本数据和文档描述等上下文信息，提供精准的语义匹配结果。

FiscalMind v3.0 introduces a completely LLM-based intelligent semantic matching system. The system leverages LLM's powerful semantic understanding capabilities, combined with contextual information including headers, data types, sample data, and document descriptions, to provide accurate semantic matching results.

## 核心组件 (Core Components)

### SemanticResolver 类

语义解析器是整个语义匹配系统的核心，基于大语言模型提供以下功能：

1. **列名语义匹配** (Column Semantic Matching) - 基于完整数据表上下文
2. **工作表名语义匹配** (Sheet Name Semantic Matching) - 支持工作表描述
3. **关联键自动发现** (Auto Join Key Discovery) - 分析两表完整上下文
4. **值标准化** (Value Normalization) - 地名等分类值标准化
5. **文档名语义匹配** (Document Name Semantic Matching) - 基于文档描述

## 主要功能 (Key Features)

### 1. 工作表名称匹配 (Sheet Name Matching)

**问题场景**: 用户可能说"看下24年的预算"，但Excel中工作表叫"FY24_Budget"或"2024预算表"。

**解决方案**: 
- 使用LLM分析用户查询意图和工作表名称
- 理解年份简写（如'24' → '2024'或'FY24'）
- 识别业务同义词（如"工资"可能匹配"薪资"）
- 支持工作表描述上下文增强匹配准确度

**示例**:
```python
from fiscal_mind.parser import ExcelParser
from langchain_openai import ChatOpenAI

# 创建带LLM的parser
llm = ChatOpenAI(model="gpt-4", temperature=0)
parser = ExcelParser(llm=llm)
doc = parser.load_document('budget.xlsx')

# LLM会理解"24年预算"的含义
df = doc.get_sheet('24年预算', use_semantic=True)  # 自动匹配到 '2024年预算表' 或 'FY24_Budget'
```

**测试结果**:
- ✓ '24年的预算' → '2024年财务报表' 或 'FY24_Budget' (由LLM智能判断)
- ✓ '员工工资' → '员工薪资_2024' (理解同义词)
- ✓ 'sales' → 'Sales_Q1' (跨语言匹配)

### 2. 过滤条件中的列名解析 (Column Name Resolution in Filters)

**问题场景**: LLM生成的过滤器可能包含"利润"，但实际列名是"净利润(万元)"。

**解决方案**:
- LLM基于完整数据表上下文进行语义匹配
- 上下文包含：列名、数据类型、样本数据（前5行）
- 理解业务同义词（如：薪资 ↔ 月薪、营收 ↔ 销售额）
- 分析数据内容辅助判断

**示例**:
```python
from fiscal_mind.parser import ExcelDocument
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
doc = ExcelDocument('sales.xlsx', llm=llm)

# LLM会分析数据表上下文，理解'营收'应该匹配到'销售额'
filters = [
    {'column': '营收', 'operator': '>', 'value': 100000},  # LLM匹配 '营收' → '销售额'
    {'column': '区域', 'operator': '==', 'value': '北京'}  # 支持值标准化
]
result = doc.filter_rows_advanced('销售明细', filters, use_semantic=True)
```

**LLM理解的业务语义**:
- 收入/营收/销售额 ↔ revenue/income (通过上下文分析数据内容)
- 利润/盈利/净利润 ↔ profit/earnings (理解财务术语)
- 工资/薪资/薪酬/月薪 ↔ salary/wage/pay (人力资源领域)
- 报销/报销单 ↔ reimbursement/expense claim (费用管理)

### 3. 多表关联的键匹配 (Join Key Discovery)

**问题场景**: 跨表关联通常需要匹配ID或名称。一张表叫"员工编号"，另一张叫"工号"。

**解决方案**:
- LLM分析两个表的完整上下文
- 上下文包含：列名、数据类型、样本数据
- 理解业务关联关系（如员工编号 ↔ 工号）
- 分析数据值域的重叠情况
- 优先使用精确列名匹配（性能优化）

**示例**:
```python
from fiscal_mind.parser import ExcelParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
parser = ExcelParser(llm=llm)

# LLM会分析两表的上下文，发现'员工编号'和'工号'是关联键
result = parser.join_sheets(
    'employees.xlsx', '员工信息',
    'performance.xlsx', '绩效评分',
    auto_discover=True  # LLM自动发现 '员工编号' ↔ '工号'
)
```

**发现策略**:
1. 完全相同的列名（优先ID/编号等关键字段）- 快速路径
2. LLM分析两表上下文：
   - 列名的业务含义（员工编号 ↔ 工号）
   - 数据类型兼容性
   - 样本数据值域重叠
   - 业务领域知识（财务、人力资源等）


### 4. 单元格值的模糊匹配 (Categorical Value Matching)

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

### 5. 文档摘要与路径匹配 (File Name Mapping)

**问题场景**: 当用户提到"报销单"时，程序需要从多个已加载文档（如reimbursement_v2.xlsx）中定位文件。

**解决方案**:
- LLM分析文档名称和用户查询
- 理解业务术语（报销 ↔ reimbursement）
- 支持文档描述上下文

**示例**:
```python
from fiscal_mind.parser import ExcelParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
parser = ExcelParser(llm=llm)
parser.load_documents([
    'financial_report_2024.xlsx',
    'reimbursement_v2.xlsx',
    'employee_salary.xlsx'
])

# LLM理解"报销单"应该匹配"reimbursement"
doc = parser.get_document('报销单', use_semantic=True)  # → reimbursement_v2.xlsx
doc = parser.get_document('工资', use_semantic=True)    # → employee_salary.xlsx
```

## 配置与使用 (Configuration and Usage)

### 基础使用（带LLM - 推荐）

```python
from fiscal_mind.parser import ExcelParser
from langchain_openai import ChatOpenAI
import os

# 设置OpenAI API密钥
os.environ['OPENAI_API_KEY'] = 'your-api-key'

# 创建LLM（推荐使用GPT-4以获得最佳效果）
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 创建解析器（带LLM支持）
parser = ExcelParser(llm=llm)

# LLM会分析完整上下文进行智能匹配
doc = parser.load_document('data.xlsx')
df = doc.get_sheet('工资', use_semantic=True)  # LLM语义匹配
```

### 无LLM使用（仅精确匹配）

```python
from fiscal_mind.parser import ExcelParser

# 创建解析器（无LLM）
parser = ExcelParser()

# 仅支持精确列名/工作表名匹配
doc = parser.load_document('data.xlsx')
df = doc.get_sheet('员工工资表', use_semantic=True)  # 需要精确匹配表名
```

### 自定义样本数据行数

```python
from fiscal_mind.semantic_resolver import SemanticResolver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# 自定义用于上下文的样本数据行数（默认5行）
resolver = SemanticResolver(llm=llm, sample_rows=10)

# 手动使用
columns = resolver.find_column_by_semantic(df, '销售额', description="产品销售数据表")
```

## 性能与可信度 (Performance and Confidence)

### 匹配策略

新的LLM-based语义匹配策略：

1. **精确匹配快速路径** - 对于某些场景（如同名关联键），直接返回，无需LLM
2. **LLM上下文匹配** - 提供完整上下文给LLM进行智能分析：
   - 表头信息（列名、数据类型）
   - 样本数据（默认前5行）
   - 可选的表格/文档描述
   - 用户的业务查询
3. **无LLM降级** - 当未提供LLM时，仅支持精确匹配

### 性能考虑

- **精确匹配**: 毫秒级（快速路径）
- **LLM匹配**: 1-3秒（取决于LLM提供商）
- **上下文大小**: 可通过`sample_rows`参数控制
- **成本优化**: 
  - 对于简单场景（如ID匹配），自动使用快速路径
  - 仅在需要语义理解时调用LLM

## LLM语义理解能力 (LLM Semantic Understanding)

LLM不依赖预定义的同义词库，而是通过分析上下文理解业务语义。以下是LLM可以理解的一些例子：

**财务领域**:
- 收入/营收/销售额 ↔ revenue/income (通过数据内容判断)
- 利润/盈利/净利润 ↔ profit/earnings (理解财务术语)
- 成本/费用/开支 ↔ cost/expense

**人力资源**:
- 工资/薪资/薪酬/月薪 ↔ salary/wage/pay
- 员工/人员/职工 ↔ employee/staff
- 编号/ID/工号 ↔ employee_id/code

**业务流程**:
- 报销/报销单 ↔ reimbursement/expense claim
- 预算/预算表 ↔ budget
- 报表/报告/财务报表 ↔ report/financial statement

LLM的优势在于：
1. 无需预先定义所有同义词
2. 可以理解上下文中的隐含关系
3. 支持跨语言匹配
4. 可以根据数据内容进行智能判断

## 最佳实践 (Best Practices)

### 1. 始终提供LLM以获得最佳体验

```python
from langchain_openai import ChatOpenAI

# 推荐：使用GPT-4获得最佳语义理解
llm = ChatOpenAI(model="gpt-4", temperature=0)
parser = ExcelParser(llm=llm)
```

### 2. 提供表格描述增强匹配准确度

```python
# 为表格提供业务描述
resolver = SemanticResolver(llm=llm)
columns = resolver.find_column_by_semantic(
    df, 
    '销售额',
    description="2024年第一季度产品销售明细表"
)
```

### 3. 控制样本数据量平衡性能和准确度

```python
# 对于大表，减少样本行数以提高性能
resolver = SemanticResolver(llm=llm, sample_rows=3)

# 对于复杂匹配，增加样本行数以提高准确度
resolver = SemanticResolver(llm=llm, sample_rows=10)
```

### 4. 使用语义开关控制行为

```python
# 对于已知的精确场景，可以关闭语义匹配
df = doc.get_sheet('员工信息表', use_semantic=False)

# 对于用户输入或模糊查询，启用语义匹配
df = doc.get_sheet(user_query, use_semantic=True)
```

## 测试覆盖 (Test Coverage)

详见 `tests/test_semantic_resolver.py`：

- ✓ 列名语义匹配（基于LLM上下文分析）
- ✓ 工作表名匹配（LLM理解业务含义）
- ✓ 关联键自动发现（分析数据类型和内容）
- ✓ 值标准化（地名、区域）
- ✓ Parser集成测试
- ✓ 文档名语义匹配
- ✓ MockLLM测试框架（无需真实LLM API）

## 限制与未来改进 (Limitations and Future Improvements)

### 当前限制

1. 依赖LLM API调用（有网络延迟和成本）
2. 样本数据量较大时可能影响性能
3. 值标准化目前仅支持地名
4. 无LLM时仅支持精确匹配

### 未来改进方向

1. **结果缓存**: 缓存LLM匹配结果以减少重复调用
2. **本地模型支持**: 支持本地部署的开源LLM
3. **增量上下文**: 对于大表，支持增量加载上下文
4. **用户反馈**: 允许用户修正匹配结果并学习
5. **多模态支持**: 支持图表、图像等多模态数据

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

FiscalMind v3.0的LLM-based语义匹配功能提供了：

✅ **智能理解**: 基于LLM的深度语义理解，无需预定义同义词库  
✅ **上下文感知**: 结合表头、数据类型、样本数据进行综合分析  
✅ **高准确度**: 利用GPT-4等先进模型的强大能力  
✅ **易用性**: 开箱即用，提供LLM即可使用全部功能  
✅ **灵活配置**: 支持自定义样本行数、表格描述等参数  
✅ **性能优化**: 对简单场景使用快速路径，避免不必要的LLM调用  

这使得FiscalMind能够更智能地理解用户意图，准确匹配复杂的业务场景，提供专业的财务数据分析体验。
