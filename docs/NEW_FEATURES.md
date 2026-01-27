# FiscalMind 新功能文档

## 概述

本文档介绍FiscalMind系统的增强功能，包括高级数据处理、智能工具选择和Function Calling机制。

## 新增功能列表

### 1. 高级过滤功能 (Advanced Filtering)

#### 1.1 多操作符支持

新增的 `filter_rows_advanced()` 方法支持以下操作符：

- `==`: 等于
- `!=`: 不等于
- `>`: 大于
- `<`: 小于
- `>=`: 大于等于
- `<=`: 小于等于
- `between`: 范围过滤
- `in`: 包含于列表
- `contains`: 包含子串

#### 1.2 使用示例

```python
from fiscal_mind.parser import ExcelParser

parser = ExcelParser()
doc = parser.load_document('sales_data.xlsx')

# 单条件过滤：销售额大于50000
filters = [
    {'column': '销售额', 'operator': '>', 'value': 50000}
]
result = doc.filter_rows_advanced('销售明细', filters)

# 范围过滤：销售额在30000-60000之间
filters = [
    {'column': '销售额', 'operator': 'between', 'value': [30000, 60000]}
]
result = doc.filter_rows_advanced('销售明细', filters)

# 组合条件：销售额>50000 且 区域='华东'
filters = [
    {'column': '销售额', 'operator': '>', 'value': 50000},
    {'column': '区域', 'operator': '==', 'value': '华东'}
]
result = doc.filter_rows_advanced('销售明细', filters)
```

### 2. 排序功能 (Sorting)

```python
# 单列降序排序
sort_by = [{'column': '销售额', 'ascending': False}]
result = doc.sort_rows('销售明细', sort_by)

# 多列排序：先按区域升序，再按销售额降序
sort_by = [
    {'column': '区域', 'ascending': True},
    {'column': '销售额', 'ascending': False}
]
result = doc.sort_rows('销售明细', sort_by)
```

### 3. 表关联功能 (Table Joins)

```python
# 关联两个工作表
result = parser.join_sheets(
    doc1_name='sales_data.xlsx',
    sheet1_name='销售明细',
    doc2_name='product_info.xlsx',
    sheet2_name='产品信息',
    left_on='产品',
    right_on='产品名称',
    how='inner'  # 支持: inner, left, right, outer
)
```

### 4. 语义搜索 (Semantic Search)

```python
from fiscal_mind.meta_functions import TableQueryHelper

# 查找与"收入"相关的列
columns = TableQueryHelper.find_column_by_semantic(df, '收入')
# 可能返回: ['销售额', '营业收入']
```

支持的语义概念：收入、利润、成本、销售、日期、数量、价格、部门、员工、工资等

### 5. 自动分组聚合

```python
# 自动检测可分组的列
groupable_cols = TableQueryHelper.auto_detect_groupable_columns(df)

# 自动分组聚合
result = TableQueryHelper.group_and_aggregate(
    df,
    group_by=['区域', '产品'],
    agg_columns=['销售额', '数量'],
    agg_func='sum'
)
```

### 6. 数据质量分析

```python
from fiscal_mind.meta_functions import DataCleaningHelper

# 分析数据质量
report = DataCleaningHelper.analyze_data_quality(df)

# 获取填充策略建议
strategy = DataCleaningHelper.suggest_fill_strategy(df, '销售额')

# 执行数据清洗
operations = [
    {'type': 'fill_null', 'column': '销售额', 'method': 'median'},
    {'type': 'remove_duplicates'},
    {'type': 'remove_outliers', 'column': '销售额'}
]
cleaned_df = DataCleaningHelper.clean_data(df, operations)
```

### 7. Function Calling Agent

```python
from fiscal_mind.enhanced_agent import FunctionCallingAgent

# 创建Agent
agent = FunctionCallingAgent()
agent.load_documents(['sales_data.xlsx'])

# 智能查询
response = agent.query("销售额前5名的产品是哪些？")
```

## 可用工具列表

1. **get_document_summary**: 获取文档摘要
2. **get_sheet_data**: 获取工作表数据
3. **filter_data**: 高级数据过滤
4. **sort_data**: 数据排序
5. **aggregate_data**: 分组聚合
6. **get_statistics**: 统计分析
7. **search_value**: 值搜索
8. **find_columns**: 智能列查找
9. **join_tables**: 表关联
10. **analyze_data_quality**: 数据质量分析
11. **get_top_n**: TopN查询

## 架构升级

**旧架构 (关键词匹配)**:
```
用户查询 → 关键词匹配 → 固定操作 → 返回结果
```

**新架构 (Function Calling)**:
```
用户查询 → 工具选择(LLM/规则) → 工具执行 → 结果整合 → 返回答案
         ↑                                        ↓
         └──────────── 多步推理循环 ────────────────┘
```

## 总结

FiscalMind的新功能显著提升了系统的智能化水平，从"关键词检索器"进化为真正的"智能分析助手"。
