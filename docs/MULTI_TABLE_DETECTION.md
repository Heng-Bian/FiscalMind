# 多表格检测功能文档

## 概述

FiscalMind 现在支持在单个Excel工作表中检测和解析多个表格。这个功能可以处理以下场景：

1. **多表格sheet**：一个工作表包含多个独立的表格，每个表格有自己的表头和数据
2. **表格偏移**：表格不从A1单元格开始，而是有行列偏移
3. **表格描述**：自动提取表格附近的描述性文本

## 功能特性

### 智能表格检测

- 自动识别表头行（基于数据类型和内容分析）
- 检测表格边界（起始行、起始列、结束位置）
- 支持同一行的并排表格
- 提取表格上方的描述文本

### 向后兼容

- 默认关闭多表格检测，保持原有行为
- 通过参数启用新功能
- 所有现有API保持不变

## 使用方法

### 基础用法

```python
from fiscal_mind.parser import ExcelParser

# 创建启用多表格检测的解析器
parser = ExcelParser(detect_multiple_tables=True)

# 加载Excel文档
doc = parser.load_document('multi_table_file.xlsx')

# 获取工作表摘要（包含表格信息）
sheet_name = doc.get_sheet_names()[0]
summary = doc.get_sheet_summary(sheet_name)

print(f"检测到 {summary.get('num_tables', 0)} 个表格")

# 查看每个表格的信息
if 'tables' in summary:
    for table in summary['tables']:
        print(f"表格 {table['index']}:")
        print(f"  位置: {table['position']}")
        print(f"  形状: {table['shape']}")
        print(f"  描述: {table['description']}")
        print(f"  列名: {table['columns']}")
```

### 访问特定表格

```python
# 方法1: 获取所有表格信息
tables = doc.get_sheet_tables(sheet_name)
for i, table_info in enumerate(tables):
    print(f"表格 {i}: {table_info.description}")
    print(table_info.data.head())

# 方法2: 直接获取表格数据
table_0 = doc.get_table_by_index(sheet_name, 0)
table_1 = doc.get_table_by_index(sheet_name, 1)

# 方法3: 获取表格完整信息
table_info = doc.get_table_info(sheet_name, 0)
print(f"起始位置: 行{table_info.start_row}, 列{table_info.start_col}")
print(f"描述: {table_info.description}")
print(table_info.data)
```

### 在文档级别启用

```python
# 为单个文档启用
parser = ExcelParser()  # 默认不启用
doc = parser.load_document('file.xlsx', detect_multiple_tables=True)

# 为所有文档启用
parser = ExcelParser(detect_multiple_tables=True)
docs = parser.load_documents(['file1.xlsx', 'file2.xlsx'])
```

## API参考

### ExcelParser 类

#### `__init__(detect_multiple_tables: bool = False)`

初始化解析器。

**参数:**
- `detect_multiple_tables`: 是否启用多表格检测，默认False

#### `load_document(file_path: str, detect_multiple_tables: Optional[bool] = None)`

加载Excel文档。

**参数:**
- `file_path`: Excel文件路径
- `detect_multiple_tables`: 是否检测多表格，如果为None则使用解析器的默认设置

**返回:** `ExcelDocument` 对象

### ExcelDocument 类

#### `get_sheet_tables(sheet_name: str) -> Optional[List[TableInfo]]`

获取工作表中检测到的所有表格。

**参数:**
- `sheet_name`: 工作表名称

**返回:** 表格信息列表，如果未启用多表格检测或不存在则返回None

#### `get_table_by_index(sheet_name: str, table_index: int) -> Optional[pd.DataFrame]`

获取工作表中指定索引的表格数据。

**参数:**
- `sheet_name`: 工作表名称
- `table_index`: 表格索引（0-based）

**返回:** 表格DataFrame，如果不存在则返回None

#### `get_table_info(sheet_name: str, table_index: int) -> Optional[TableInfo]`

获取工作表中指定索引的表格完整信息。

**参数:**
- `sheet_name`: 工作表名称
- `table_index`: 表格索引（0-based）

**返回:** `TableInfo`对象，如果不存在则返回None

### TableInfo 类

表格元数据类，包含以下属性：

- `data`: 表格数据（pandas DataFrame）
- `start_row`: 表格起始行号（0-based）
- `start_col`: 表格起始列号（0-based）
- `description`: 表格描述文本（可能为None）
- `end_row`: 表格结束行号
- `end_col`: 表格结束列号

## 示例场景

### 场景1: 多个堆叠表格

一个工作表包含多个纵向排列的表格：

```
表1: 销售数据
产品   销售额   数量
...

表2: 费用明细
项目   金额   占比
...
```

### 场景2: 并排表格

一个工作表包含多个横向排列的表格：

```
表1: Q1数据        表2: Q2数据
月份 收入 ...     月份 收入 ...
...               ...
```

### 场景3: 偏移表格

表格不从A1开始，有标题和说明：

```
           财务报表汇总
           2024年度数据
           
           指标    2023  2024
           收入    1000  1200
           ...
```

## 注意事项

1. **性能考虑**: 多表格检测需要额外的计算，建议只在需要时启用
2. **表头识别**: 算法基于启发式规则识别表头，可能在某些特殊格式下不准确
3. **描述提取**: 只检查表头上方最多3行的描述文本
4. **空列处理**: 连续1个空列会被视为表格分界

## 测试示例

项目提供了完整的测试示例：

```bash
# 运行多表格检测测试
python tests/test_multi_table.py

# 创建示例文件
python examples/create_multi_table_sample.py
```

## 常见问题

**Q: 如何判断是否应该启用多表格检测？**

A: 如果您的Excel文件满足以下条件之一，建议启用：
- 一个sheet包含多个独立的报表
- 表格不是从A1开始
- 需要提取表格的描述信息

**Q: 启用多表格检测会影响现有代码吗？**

A: 不会。默认情况下多表格检测是关闭的，即使启用，`get_sheet()`方法仍返回第一个表格的数据，保持向后兼容。

**Q: 如何访问检测到的表格？**

A: 使用新增的API方法：
- `get_sheet_tables()` - 获取所有表格
- `get_table_by_index()` - 获取指定表格
- `get_table_info()` - 获取表格元数据

## 技术细节

### 表头检测算法

1. 扫描每一行，查找潜在的表头
2. 表头判断标准：
   - 至少包含2个非空单元格
   - 至少50%的单元格是文本而非数字
3. 连续空列视为表格边界

### 描述提取

- 检查表头上方最多3行
- 识别包含特定关键词的文本（如"表"、"数据"、":"等）
- 提取第一个匹配的描述

### 数据行验证

- 表头之后的行必须至少有一个非空值才被视为数据行
- 遇到完全空行时，表格结束
