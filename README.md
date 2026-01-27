# FiscalMind

面向财务BP的表格分析Agent - A Table Analysis Agent for Financial Business Partners

## 简介 (Introduction)

FiscalMind 是一个基于 LangGraph 框架的智能表格文档分析系统，专为财务BP设计。它能够同时处理多个Excel文档，提供智能的数据分析和查询功能。

FiscalMind is an intelligent table document analysis system built on the LangGraph framework, designed for financial business partners. It can process multiple Excel documents simultaneously and provide intelligent data analysis and query capabilities.

## 主要功能 (Key Features)

- ✅ **多文档支持**: 同时加载和处理多个Excel文档
- ✅ **智能解析**: 自动解析Excel文件结构和内容
- ✅ **元功能**: 提供表格统计、搜索、聚合等元操作
- ✅ **LangGraph集成**: 基于LangGraph的智能工作流
- ✅ **交互式查询**: 支持自然语言式的交互查询
- ✅ **财务专用**: 针对财务场景优化的功能设计

## 技术栈 (Tech Stack)

- **Python 3.8+**
- **LangGraph**: 智能体工作流框架
- **Pandas**: 数据处理和分析
- **OpenPyXL**: Excel文件读写
- **LangChain**: LLM集成（可选）

## 安装 (Installation)

```bash
# 克隆仓库
git clone https://github.com/Heng-Bian/FiscalMind.git
cd FiscalMind

# 安装依赖
pip install -r requirements.txt
```

## 快速开始 (Quick Start)

### 1. 创建示例Excel文件

```bash
# 生成示例数据
python examples/create_samples.py
```

这将创建三个示例Excel文件：
- `financial_report.xlsx` - 财务报表
- `sales_data.xlsx` - 销售数据
- `employee_salary.xlsx` - 员工薪资

### 2. 基础使用

```python
from fiscal_mind.agent import TableDocumentAgent

# 创建Agent
agent = TableDocumentAgent()

# 加载Excel文档
agent.load_documents([
    'examples/financial_report.xlsx',
    'examples/sales_data.xlsx'
])

# 获取文档摘要
summary = agent.get_document_summary()
print(summary)

# 查询分析
response = agent.query("显示所有文档的统计信息")
print(response)
```

### 3. 交互模式

```bash
# 启动交互式界面
python -m fiscal_mind.main examples/*.xlsx -i
```

## 核心模块说明 (Core Modules)

### 1. Parser 模块 (`fiscal_mind/parser.py`)

Excel文档解析器，负责读取和解析Excel文件。

**主要类:**
- `ExcelDocument`: 表示单个Excel文档
- `ExcelParser`: 管理多个Excel文档

**主要功能:**
- 读取所有工作表
- 获取文档摘要
- 数据搜索
- 列提取
- 行过滤

### 2. Meta Functions 模块 (`fiscal_mind/meta_functions.py`)

提供表格数据的元操作和LLM交互工具。

**主要类:**
- `TableMetaFunctions`: 表格元功能（统计、摘要、格式化）
- `TableQueryHelper`: 查询辅助工具（聚合、过滤、排序）

**主要功能:**
- 表格结构分析
- 统计信息提取
- LLM上下文格式化
- 数据聚合和透视

### 3. Agent 模块 (`fiscal_mind/agent.py`)

基于LangGraph的智能Agent。

**主要类:**
- `TableDocumentAgent`: 表格文档分析Agent

**工作流节点:**
1. `load_context`: 加载文档上下文
2. `analyze_query`: 分析用户查询
3. `execute_query`: 执行查询操作
4. `format_response`: 格式化响应

## 使用示例 (Usage Examples)

### 示例 1: 文档加载和摘要

```python
from fiscal_mind.parser import ExcelParser

# 创建解析器
parser = ExcelParser()

# 加载文档
doc = parser.load_document('examples/financial_report.xlsx')

# 获取工作表列表
sheets = doc.get_sheet_names()
print(f"工作表: {sheets}")

# 获取文档摘要
summary = doc.get_document_summary()
print(summary)
```

### 示例 2: 数据搜索

```python
# 在文档中搜索值
results = doc.search_value("产品A")
for result in results:
    print(f"找到于: {result['sheet']}, 行: {result['row']}, 列: {result['column']}")
```

### 示例 3: 统计分析

```python
from fiscal_mind.meta_functions import TableMetaFunctions

# 获取工作表
df = doc.get_sheet('损益表')

# 获取数值列统计
stats = TableMetaFunctions.get_numeric_summary(df)
print(stats)

# 获取特定列统计
col_stats = TableMetaFunctions.get_column_statistics(df, '净利润')
print(col_stats)
```

### 示例 4: 数据查询和聚合

```python
from fiscal_mind.meta_functions import TableQueryHelper

# 按列聚合
result = TableQueryHelper.aggregate_by_column(
    df, 
    group_col='区域', 
    agg_col='销售额', 
    agg_func='sum'
)
print(result)

# 获取Top N
top_sales = TableQueryHelper.get_top_n_by_column(
    df, 
    column='销售额', 
    n=10
)
print(top_sales)
```

### 示例 5: 使用Agent进行智能查询

```python
from fiscal_mind.agent import TableDocumentAgent

agent = TableDocumentAgent()
agent.load_documents(['examples/sales_data.xlsx'])

# 智能查询
response = agent.query("显示销售数据的统计信息")
print(response)

# 分析特定工作表
analysis = agent.analyze_sheet('sales_data.xlsx', '销售明细')
print(analysis)
```

## 架构设计 (Architecture)

```
FiscalMind/
├── fiscal_mind/
│   ├── __init__.py          # 包初始化
│   ├── parser.py            # Excel解析器
│   ├── meta_functions.py    # 元功能和查询工具
│   ├── agent.py             # LangGraph Agent
│   ├── main.py              # 主程序入口
│   └── utils.py             # 工具函数
├── examples/
│   ├── create_samples.py    # 创建示例数据
│   └── *.xlsx               # 示例Excel文件
├── requirements.txt         # 依赖列表
└── README.md               # 项目文档
```

## Agent工作流 (Agent Workflow)

```
用户查询 (User Query)
    ↓
加载上下文 (Load Context)
    ↓
分析查询 (Analyze Query)
    ↓
执行查询 (Execute Query)
    ↓
格式化响应 (Format Response)
    ↓
返回结果 (Return Result)
```

## 扩展开发 (Extension Development)

### 添加新的查询类型

在 `agent.py` 中的 `_analyze_query_node` 方法中添加新的查询类型识别逻辑。

### 集成LLM

FiscalMind支持集成各种LLM提供商：

```python
from langchain_openai import ChatOpenAI

# 在Agent中集成LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
# 可在查询分析和响应生成中使用LLM
```

### 添加新的元功能

在 `meta_functions.py` 中添加新的静态方法到 `TableMetaFunctions` 或 `TableQueryHelper` 类。

## 后续开发计划 (Roadmap)

- [ ] 集成大语言模型进行自然语言查询理解
- [ ] 支持更多数据源（CSV, JSON等）
- [ ] 添加数据可视化功能
- [ ] 实现复杂的财务分析模型
- [ ] 添加数据导出功能
- [ ] 开发Web界面
- [ ] 支持多语言（中英文）
- [ ] 添加更多智能体协作功能

## 贡献 (Contributing)

欢迎提交Issue和Pull Request！

## 许可证 (License)

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式 (Contact)

- 项目主页: https://github.com/Heng-Bian/FiscalMind
- Issue反馈: https://github.com/Heng-Bian/FiscalMind/issues

---

**注意**: 本项目为初始版本，专注于表格解析和元功能实现。后续将添加更多智能体功能和LLM集成。
