# FiscalMind

面向财务BP的智能表格分析系统 - AI-Powered Table Analysis System for Financial Business Partners

## 简介 (Introduction)

FiscalMind 是基于 LangGraph 框架的智能财务表格分析系统，通过深度集成大语言模型（LLM），实现从数据解析到业务理解的全流程智能化分析。

FiscalMind is an AI-powered financial table analysis system built on LangGraph, achieving end-to-end intelligent analysis from data parsing to business understanding through deep LLM integration.

## 核心特性 (Core Features)

### 🤖 LLM 驱动的三大核心能力

#### 1. 智能表头识别
**大模型自动识别表头结构**，区分表头行与数据行，避免误判
- 准确处理多层表头、合并单元格
- 自适应各种表格格式和偏移位置
- 提供置信度评分，低置信度时自动回退

#### 2. 语义字段匹配
**基于业务语义的智能字段匹配**，理解业务含义而非简单字符串比较
- 列名智能匹配："薪资" → "月薪"、"营收" → "销售额"
- 工作表匹配："24年预算" → "FY24_Budget"
- 关联键发现："员工编号" ↔ "工号"
- 值模糊匹配："北京" = "北京市" = "Beijing"

#### 3. 深度业务理解
**多智能体协作挖掘数据背后的业务逻辑**
- **业务分析智能体**: 识别业务领域、关键维度和核心指标
- **批评者智能体**: 评估分析质量，多轮协作优化
- **评判智能体**: 验证结论合理性和数据支撑

### 🎯 其他核心功能
- **多文档支持**: 同时处理多个Excel文档
- **多表格检测**: 自动识别单个sheet中的多个表格
- **PRR架构**: Plan-ReAct-Reflect智能规划执行流程
- **高级查询**: 过滤、排序、关联、聚合等操作
- **自然语言交互**: 支持复杂的自然语言查询

## 技术栈 (Tech Stack)

- **LangGraph**: 智能体工作流框架
- **LangChain**: LLM集成和工具调用
- **Pandas**: 数据处理和分析
- **OpenPyXL**: Excel文件读写

## 安装 (Installation)

```bash
git clone https://github.com/Heng-Bian/FiscalMind.git
cd FiscalMind
pip install -r requirements.txt
```

## 快速开始 (Quick Start)

### 方式一：命令行快速启动

```bash
# 将Excel文件放在项目根目录
cp your_data.xlsx ./

# 运行主程序（自动加载所有Excel文件并启动智能分析）
python main.py
```

程序会自动启动 PRR Agent 进行智能分析，进入交互模式：

```
FiscalMind> 哪个大区表现更好？
正在分析您的问题...

业务分析：
- 业务领域: 财务
- 关键维度: 大区, 月份
- 核心指标: 预算, 实际, 差额

分析结果: 根据销售额和利润率综合评估，华东大区表现最佳。
```

### 方式二：Python API

```python
from fiscal_mind.prr_agent import PRRAgent
from langchain_openai import ChatOpenAI

# 创建带LLM的PRR Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = PRRAgent(llm=llm)

# 加载文档并查询
agent.load_documents(['financial_report.xlsx'])
answer = agent.query("哪个大区今年表现更好？")
print(answer)
```

### LLM 配置说明

系统支持两种模式：

**有 LLM 模式**（推荐）：
- 启用智能表头识别、语义匹配、业务理解
- 提供专业级财务分析能力

**无 LLM 模式**：
- 使用规则方法处理
- 提供基础数据查询功能

```python
# 无LLM模式
agent = PRRAgent(llm=None)
```


## 核心模块 (Core Modules)

### Parser 模块 - 智能表格解析
**LLM 驱动的表头识别和多表格检测**

```python
from fiscal_mind.parser import ExcelParser

parser = ExcelParser()
doc = parser.load_document('report.xlsx')

# 自动检测表头、多表格、偏移位置
summary = doc.get_document_summary()
```

### Semantic Resolver - 语义匹配引擎
**基于 LLM 的智能字段匹配**

```python
from fiscal_mind.semantic_resolver import SemanticResolver

resolver = SemanticResolver(llm=llm)
# 自动匹配业务语义："薪资" → "月薪"
columns = resolver.find_column_by_semantic(df, "薪资")
```

### PRR Agent - 智能分析引擎
**Plan-ReAct-Reflect 架构 + 专业智能体**

```python
from fiscal_mind.prr_agent import PRRAgent

agent = PRRAgent(llm=llm)
agent.load_documents(['data.xlsx'])

# 自动规划、执行、反思、业务理解
answer = agent.query("哪个大区今年表现更好？")
```

## 使用示例 (Usage Examples)

### 示例 1: 智能表头识别

```python
from fiscal_mind.parser import ExcelParser

parser = ExcelParser()
doc = parser.load_document('complex_table.xlsx')

# LLM自动识别表头位置，处理多层表头和合并单元格
sheet = doc.get_sheet('数据表')
print(sheet.columns)  # 正确识别的表头
```

### 示例 2: 语义字段匹配

```python
from fiscal_mind.semantic_resolver import SemanticResolver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
resolver = SemanticResolver(llm=llm)

# 智能匹配列名
df = doc.get_sheet('员工表')
salary_cols = resolver.find_column_by_semantic(df, "薪资")
# 返回: ['月薪', '基本工资', '总收入'] 等相关列
```

### 示例 3: 业务理解与分析

```python
from fiscal_mind.prr_agent import PRRAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = PRRAgent(llm=llm)

agent.load_documents(['regional_data.xlsx'])

# PRR Agent会自动：
# 1. 业务理解 - 识别财务领域、关键维度、核心指标
# 2. 智能规划 - 分解查询为执行步骤
# 3. 推理执行 - 调用工具获取数据
# 4. 质量评判 - 验证结论合理性
answer = agent.query("对比各大区的销售额和利润率，找出综合表现最好的大区")
print(answer)
```

### 示例 4: 多表格检测

```python
# 自动检测单个sheet中的多个表格
doc = parser.load_document('multi_table.xlsx')
summary = doc.get_sheet_summary('Sheet1')

print(f"检测到 {summary['num_tables']} 个表格")
for i in range(summary['num_tables']):
    table = doc.get_table_info('Sheet1', i)
    print(f"表格{i}: {table.description}")
```


## 架构设计 (Architecture)

### 系统架构

```
用户查询
    ↓
业务理解（多智能体协作）
    ├── 识别业务领域和场景
    ├── 提取关键维度和指标
    └── 评估分析质量
    ↓
智能规划（PRR）
    ├── 分解执行步骤
    └── 融合业务上下文
    ↓
推理执行
    ├── LLM语义匹配（字段/工作表）
    ├── LLM表头识别（多层/偏移）
    └── 数据提取和聚合
    ↓
质量评判
    ├── 验证结论合理性
    ├── 评估数据支撑
    └── 业务逻辑检查
    ↓
生成答案
```

### 核心技术亮点

1. **LLM 表头识别**: 自动区分表头与数据，处理复杂格式
2. **LLM 语义匹配**: 理解业务含义，智能匹配字段
3. **LLM 业务理解**: 多智能体挖掘数据背后的业务逻辑
4. **PRR 自主规划**: 自动分解复杂查询，迭代优化

## 扩展开发 (Extension)

### 集成自定义 LLM

```python
from langchain_openai import ChatOpenAI

# 使用 OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 使用其他LLM提供商
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus")

# 创建Agent
agent = PRRAgent(llm=llm)
```

## 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！

## 许可证 (License)

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式 (Contact)

- 项目主页: https://github.com/Heng-Bian/FiscalMind
- Issue反馈: https://github.com/Heng-Bian/FiscalMind/issues
