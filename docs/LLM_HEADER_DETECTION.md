# LLM-Based 表头检测功能文档

## 概述

LLM-Based表头检测是FiscalMind v3.1新增的智能功能，用于解决多表头判断误差问题。该功能使用大语言模型（LLM）智能分析表格的前若干行数据，准确识别哪些行是表头，哪些行是数据行，避免将数据行误判为表头。

## 问题背景

在之前的版本中，FiscalMind使用规则基础的方法来检测表头：
- 判断一行中文本单元格的比例（50%以上为表头）
- 这导致包含大量文本的数据行也被误判为表头

**示例问题：**
```
第0行: 姓名 | 职位 | 部门 | 工资    <- 真正的表头
第1行: 张三 | 经理 | 销售部 | 20000  <- 数据行（但包含3个文本字段）
第2行: 李四 | 工程师 | 技术部 | 15000
```

规则方法会将第0行和第1行都识别为表头（因为第1行有75%的单元格是文本）。

## 解决方案

### 1. HeaderDetectionAgent 智能体

新增了专门的`HeaderDetectionAgent`智能体，职责包括：
- 接收表格的前N行数据（默认10行）
- 使用LLM分析这些数据的语义和结构
- 智能识别表头行和数据行
- 返回检测结果，包括表头行数、置信度等

### 2. LLM Prompt 设计

发送给LLM的prompt包含：
- 表格的前10行数据
- 明确的指导说明（区分表头和数据）
- 要求返回JSON格式的结构化结果

**Prompt示例：**
```
你是一位专业的数据分析专家。请分析以下表格的前若干行数据，识别哪些行是表头，哪些行是数据行。

重要提示:
- 表头通常包含列名、字段名等描述性文本
- 数据行通常包含具体的数值、名称、日期等实际数据
- 不要将数据行误判为表头
- 表头可能有多行（多层表头）
- 表头行必须是连续的，从第一行开始

表格数据（前N行）:
第0行: 姓名 | 职位 | 部门 | 工资
第1行: 张三 | 经理 | 销售部 | 20000
第2行: 李四 | 工程师 | 技术部 | 15000
...

请以JSON格式回答，包含以下字段:
- header_row_count: 表头行数（整数）
- header_rows_indices: 表头行的索引列表
- data_start_row: 数据开始的行索引
- confidence: 检测置信度（0.0-1.0）
- reasoning: 判断理由
```

**LLM返回示例：**
```json
{
  "header_row_count": 1,
  "header_rows_indices": [0],
  "data_start_row": 1,
  "confidence": 0.95,
  "reasoning": "第0行包含字段名称（姓名、职位、部门、工资），第1行开始是具体的人名和数据"
}
```

### 3. 集成到现有流程

LLM检测已集成到`TableDetector`的`_detect_multi_row_headers`方法中：

```python
# 解析器初始化时传入LLM
parser = ExcelParser(detect_multiple_tables=True, llm=llm)

# LLM会自动用于表头检测
doc = parser.load_document('data.xlsx')
```

**工作流程：**
1. `ExcelParser` 接收 `llm` 参数
2. 传递给 `ExcelDocument`
3. 传递给 `TableDetector.detect_tables()`
4. 在 `_detect_multi_row_headers()` 中调用 `HeaderDetectionAgent`
5. LLM分析并返回结果
6. 如果置信度 >= 0.7，使用LLM结果
7. 否则，fallback到规则基础方法

## 使用方法

### 方法1: 通过ExcelParser使用（推荐）

```python
from fiscal_mind.parser import ExcelParser
from langchain_openai import ChatOpenAI

# 创建LLM实例
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建带LLM的解析器
parser = ExcelParser(detect_multiple_tables=True, llm=llm)

# 加载文档（自动使用LLM检测表头）
doc = parser.load_document('data.xlsx')

# 获取数据
df = doc.get_sheet('Sheet1')
print(df.head())
```

### 方法2: 直接使用HeaderDetectionAgent

```python
from fiscal_mind.specialized_agents import HeaderDetectionAgent
from langchain_openai import ChatOpenAI

# 准备数据（表格的前N行）
rows_data = [
    ["姓名", "年龄", "城市"],
    ["张三", 25, "北京"],
    ["李四", 30, "上海"],
]

# 创建LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建Agent
agent = HeaderDetectionAgent(llm=llm)

# 检测表头
result = agent.detect_header_rows(rows_data, max_rows=10)

print(f"表头行数: {result['header_row_count']}")
print(f"数据开始行: {result['data_start_row']}")
print(f"置信度: {result['confidence']}")
print(f"理由: {result['reasoning']}")
```

### 方法3: 不使用LLM（向后兼容）

```python
from fiscal_mind.parser import ExcelParser

# 不传入LLM，使用规则基础方法
parser = ExcelParser(detect_multiple_tables=True, llm=None)

doc = parser.load_document('data.xlsx')
df = doc.get_sheet('Sheet1')
```

## 配置选项

### LLM配置

支持任何兼容LangChain的LLM：

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(...)

# 其他LLM提供商
from langchain_community.llms import ...
```

### 置信度阈值

当前置信度阈值为0.7，在代码中可以调整：

```python
# 在 parser.py 的 _detect_multi_row_headers 方法中
if confidence >= 0.7 and header_row_count > 0:
    # 使用LLM结果
```

## 优势

### 1. 智能识别
- LLM能理解数据的语义，准确区分表头和数据
- 不会被"文本比例高的数据行"误导

### 2. 处理复杂情况
- 多层表头
- 不规则表头
- 特殊格式的数据

### 3. 高置信度
- 提供置信度评分
- 低置信度时自动fallback到规则方法

### 4. 向后兼容
- 不传入LLM时，使用原有的规则方法
- 不影响现有代码

## 性能考虑

### 1. API调用成本
- 每次表头检测需要调用一次LLM API
- 建议只在复杂/模糊的情况下使用

### 2. 响应时间
- LLM调用需要网络请求，响应时间约1-3秒
- 如果对性能敏感，可以不使用LLM

### 3. 优化建议
- 对简单表格，使用规则方法
- 对复杂/重要表格，使用LLM
- 缓存LLM结果（如果同一文件重复加载）

## 测试

运行测试验证功能：

```bash
# 运行表头检测测试
python tests/test_header_detection.py

# 运行示例
python examples/header_detection_example.py
```

## 常见问题

### Q1: 是否必须使用LLM？
A: 不是。如果不传入`llm`参数，系统会使用规则基础方法，保持向后兼容。

### Q2: 支持哪些LLM？
A: 支持所有兼容LangChain的LLM，包括OpenAI、Azure OpenAI、Anthropic Claude等。

### Q3: LLM检测失败怎么办？
A: 系统会自动fallback到规则基础方法，确保始终能够检测表头。

### Q4: 如何提高检测准确率？
A: 
- 使用更强大的模型（如GPT-4）
- 确保提供的数据完整（前10行）
- 检查API配置是否正确

### Q5: 是否支持中文表头？
A: 完全支持中文、英文和混合语言的表头。

## 未来计划

- [ ] 支持自定义置信度阈值
- [ ] 添加缓存机制减少API调用
- [ ] 支持批量表头检测
- [ ] 提供更多的检测选项和配置

## 相关文档

- [专业智能体文档](SPECIALIZED_AGENTS.md)
- [多表格检测文档](MULTI_TABLE_DETECTION.md)
- [API文档](API.md)

## 贡献

欢迎提交Issue和Pull Request改进此功能！
