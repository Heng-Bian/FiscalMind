"""
增强的Agent模块 - 使用LLM Function Calling的智能Agent
Enhanced Agent module using LLM Function Calling.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
import operator
import logging
import json

from fiscal_mind.parser import ExcelParser
from fiscal_mind.meta_functions import TableMetaFunctions
from fiscal_mind.tools import TOOL_SCHEMAS, get_tools_description
from fiscal_mind.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


class EnhancedAgentState(TypedDict):
    """增强的Agent状态定义"""
    messages: Annotated[List[BaseMessage], operator.add]
    parser: ExcelParser
    tool_executor: ToolExecutor
    query: str  # 用户查询
    context: str  # 为LLM准备的上下文
    selected_tools: List[Dict[str, Any]]  # LLM选择的工具列表
    tool_results: List[Dict[str, Any]]  # 工具执行结果
    final_answer: Optional[str]  # 最终答案
    error: Optional[str]  # 错误信息
    iteration: int  # 迭代次数，用于多步推理


class FunctionCallingAgent:
    """
    使用LLM Function Calling的增强Agent
    
    这个Agent使用工具内省（Tool Introspection）机制，让LLM自主选择合适的工具来回答用户问题。
    支持多步推理，可以链式调用多个工具。
    """
    
    def __init__(self, parser: Optional[ExcelParser] = None, llm=None, max_iterations: int = 5):
        """
        初始化增强Agent
        
        Args:
            parser: ExcelParser实例，如果为None则创建新实例
            llm: 语言模型实例（可选，如果提供则使用LLM进行工具选择）
            max_iterations: 最大迭代次数，用于多步推理
        """
        self.parser = parser or ExcelParser()
        self.tool_executor = ToolExecutor(self.parser)
        self.llm = llm
        self.max_iterations = max_iterations
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(EnhancedAgentState)
        
        # 添加节点
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("select_tools", self._select_tools_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("decide_next", self._decide_next_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # 设置入口点
        workflow.set_entry_point("load_context")
        
        # 添加边
        workflow.add_edge("load_context", "select_tools")
        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "decide_next")
        
        # 条件边：决定是继续迭代还是生成最终答案
        workflow.add_conditional_edges(
            "decide_next",
            self._should_continue,
            {
                "continue": "select_tools",
                "finish": "generate_answer"
            }
        )
        
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _load_context_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """加载文档上下文节点"""
        logger.info("Loading document context...")
        
        try:
            # 创建数据上下文
            context_parts = []
            context_parts.append(TableMetaFunctions.create_data_context(state["parser"], max_rows_per_sheet=3))
            context_parts.append("\n" + get_tools_description())
            context = "\n\n".join(context_parts)
            
            return {
                "context": context,
                "tool_executor": self.tool_executor,
                "selected_tools": [],
                "tool_results": [],
                "iteration": 0,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error loading context: {str(e)}")
            return {
                "context": "",
                "error": f"加载上下文失败: {str(e)}"
            }
    
    def _select_tools_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """选择工具节点 - 使用LLM或规则选择合适的工具"""
        logger.info(f"Selecting tools (iteration {state['iteration']})...")
        
        query = state["query"]
        context = state.get("context", "")
        previous_results = state.get("tool_results", [])
        
        # 如果配置了LLM，使用LLM进行工具选择
        if self.llm:
            selected_tools = self._llm_select_tools(query, context, previous_results)
        else:
            # 否则使用基于规则的工具选择（增强版）
            selected_tools = self._rule_based_select_tools(query, previous_results)
        
        return {
            "selected_tools": selected_tools
        }
    
    def _llm_select_tools(self, query: str, context: str, previous_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用LLM选择工具（Function Calling）
        
        注意：这需要配置支持Function Calling的LLM（如OpenAI GPT-4）
        TODO: 完整的LLM Function Calling集成示例：
        
        示例代码：
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        messages = [
            SystemMessage(content=f"你是一个数据分析助手。可用工具: {context}"),
            HumanMessage(content=query)
        ]
        
        response = llm.invoke(
            messages=messages,
            functions=TOOL_SCHEMAS
        )
        
        return response.tool_calls
        ```
        
        当前状态：等待配置LLM实例，使用规则兜底
        """
        # 如果LLM支持Function Calling，使用上述模式集成
        # 当前实现：回退到基于规则的工具选择
        
        logger.warning("LLM未配置，回退到基于规则的工具选择")
        return self._rule_based_select_tools(query, previous_results)
    
    def _rule_based_select_tools(self, query: str, previous_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于规则的工具选择（增强版）
        
        这是一个临时解决方案，实际应该使用LLM的Function Calling
        """
        query_lower = query.lower()
        selected_tools = []
        
        # 首次查询且没有结果时，先获取文档摘要
        if not previous_results:
            # 检查是否需要文档摘要
            if any(word in query_lower for word in ["文档", "工作表", "有哪些", "列表", "概览", "summary"]):
                selected_tools.append({
                    "name": "get_document_summary",
                    "parameters": {}
                })
                return selected_tools
        
        # 搜索相关
        if any(word in query_lower for word in ["搜索", "查找", "find", "search", "包含"]):
            # 提取搜索词（简化处理）
            search_term = query.split()[-1] if query.split() else ""
            selected_tools.append({
                "name": "search_value",
                "parameters": {
                    "doc_name": self._guess_doc_name(),
                    "value": search_term
                }
            })
        
        # 统计相关
        elif any(word in query_lower for word in ["统计", "汇总", "平均", "总和", "最大", "最小", "statistics", "sum", "average"]):
            selected_tools.append({
                "name": "get_statistics",
                "parameters": {
                    "doc_name": self._guess_doc_name(),
                    "sheet_name": self._guess_sheet_name()
                }
            })
        
        # 排名/排序相关
        elif any(word in query_lower for word in ["前", "top", "最高", "最低", "排名", "第一", "最后"]):
            # 提取数字
            n = 10  # 默认10
            for word in query.split():
                if word.isdigit():
                    n = int(word)
                    break
            
            selected_tools.append({
                "name": "get_top_n",
                "parameters": {
                    "doc_name": self._guess_doc_name(),
                    "sheet_name": self._guess_sheet_name(),
                    "column": self._guess_column_name(query),
                    "n": n,
                    "ascending": "最低" in query_lower or "最少" in query_lower
                }
            })
        
        # 聚合/分组相关
        elif any(word in query_lower for word in ["按", "分组", "group by", "聚合", "每个"]):
            selected_tools.append({
                "name": "aggregate_data",
                "parameters": {
                    "doc_name": self._guess_doc_name(),
                    "sheet_name": self._guess_sheet_name(),
                    "agg_func": "sum"
                }
            })
        
        # 过滤相关
        elif any(word in query_lower for word in ["大于", "小于", "等于", "超过", "低于", ">", "<"]):
            # 构建过滤条件（简化处理）
            filters = self._parse_filter_from_query(query)
            if filters:
                selected_tools.append({
                    "name": "filter_data",
                    "parameters": {
                        "doc_name": self._guess_doc_name(),
                        "sheet_name": self._guess_sheet_name(),
                        "filters": filters
                    }
                })
        
        # 数据质量相关
        elif any(word in query_lower for word in ["质量", "缺失", "异常", "清洗", "quality"]):
            selected_tools.append({
                "name": "analyze_data_quality",
                "parameters": {
                    "doc_name": self._guess_doc_name(),
                    "sheet_name": self._guess_sheet_name()
                }
            })
        
        # 默认：获取数据预览
        else:
            if not previous_results:
                selected_tools.append({
                    "name": "get_sheet_data",
                    "parameters": {
                        "doc_name": self._guess_doc_name(),
                        "sheet_name": self._guess_sheet_name(),
                        "max_rows": 10
                    }
                })
        
        return selected_tools
    
    def _execute_tools_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """执行工具节点"""
        logger.info("Executing tools...")
        
        selected_tools = state.get("selected_tools", [])
        tool_results = []
        
        for tool_call in selected_tools:
            tool_name = tool_call.get("name")
            parameters = tool_call.get("parameters", {})
            
            logger.info(f"Executing tool: {tool_name}")
            result = state["tool_executor"].execute_tool(tool_name, parameters)
            
            tool_results.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result
            })
        
        return {
            "tool_results": state.get("tool_results", []) + tool_results,
            "iteration": state.get("iteration", 0) + 1
        }
    
    def _decide_next_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """决定下一步节点 - 判断是否需要继续迭代"""
        # 这里可以添加更复杂的逻辑来决定是否需要继续
        # 目前简化为：执行一次工具后就生成答案
        return {}
    
    def _should_continue(self, state: EnhancedAgentState) -> str:
        """判断是否应该继续迭代"""
        iteration = state.get("iteration", 0)
        tool_results = state.get("tool_results", [])
        
        # 达到最大迭代次数
        if iteration >= self.max_iterations:
            return "finish"
        
        # 如果还没有任何结果，继续
        if not tool_results:
            return "continue"
        
        # 检查最后一个工具的结果
        last_result = tool_results[-1] if tool_results else None
        if last_result and not last_result["result"].get("success", False):
            # 如果失败，可能需要重试或结束
            return "finish"
        
        # 目前简化为执行一轮就结束
        return "finish"
    
    def _generate_answer_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """生成最终答案节点"""
        logger.info("Generating final answer...")
        
        tool_results = state.get("tool_results", [])
        query = state["query"]
        
        # 格式化工具结果
        answer_parts = []
        answer_parts.append(f"问题: {query}\n")
        
        if state.get("error"):
            answer_parts.append(f"错误: {state['error']}")
        elif tool_results:
            answer_parts.append("分析结果:\n")
            for i, tr in enumerate(tool_results, 1):
                tool_name = tr["tool"]
                result = tr["result"]
                
                if result.get("success"):
                    answer_parts.append(f"\n{i}. 使用工具: {tool_name}")
                    answer_parts.append(self._format_tool_result(tool_name, result["data"]))
                else:
                    answer_parts.append(f"\n{i}. 工具 {tool_name} 执行失败: {result.get('error', '未知错误')}")
        else:
            answer_parts.append("未找到相关信息")
        
        final_answer = "\n".join(answer_parts)
        
        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)]
        }
    
    def _format_tool_result(self, tool_name: str, data: Any) -> str:
        """格式化工具结果为可读文本"""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if key == "data" and isinstance(value, list) and len(value) > 0:
                    # 格式化表格数据
                    lines.append(f"  数据记录 ({len(value)} 条):")
                    for j, record in enumerate(value[:5], 1):  # 只显示前5条
                        lines.append(f"    {j}. {record}")
                    if len(value) > 5:
                        lines.append(f"    ... 还有 {len(value) - 5} 条记录")
                elif not isinstance(value, (dict, list)):
                    lines.append(f"  {key}: {value}")
            return "\n".join(lines)
        return str(data)
    
    # 辅助方法
    def _guess_doc_name(self) -> str:
        """猜测文档名称（简化实现）"""
        docs = list(self.parser.documents.keys())
        return docs[0] if docs else ""
    
    def _guess_sheet_name(self) -> str:
        """猜测工作表名称（简化实现）"""
        doc_name = self._guess_doc_name()
        if doc_name:
            doc = self.parser.get_document(doc_name)
            if doc:
                sheets = doc.get_sheet_names()
                return sheets[0] if sheets else ""
        return ""
    
    def _guess_column_name(self, query: str) -> str:
        """从查询中猜测列名（简化实现）"""
        # 常见的财务列名关键词
        keywords = {
            "销售": "销售额",
            "利润": "利润",
            "收入": "收入",
            "成本": "成本",
            "数量": "数量",
        }
        
        for keyword, col_name in keywords.items():
            if keyword in query:
                return col_name
        
        return "销售额"  # 默认
    
    def _parse_filter_from_query(self, query: str) -> List[Dict[str, Any]]:
        """从查询中解析过滤条件（简化实现）"""
        # 这是一个非常简化的实现
        # 实际应该使用NLP或LLM来解析
        filters = []
        
        if "大于" in query or "超过" in query or ">" in query:
            # 尝试提取数字
            words = query.split()
            for word in words:
                try:
                    value = float(word.replace(",", ""))
                    filters.append({
                        "column": self._guess_column_name(query),
                        "operator": ">",
                        "value": value
                    })
                    break
                except ValueError:
                    continue
        
        return filters
    
    # 公共接口
    def load_documents(self, file_paths: List[str]) -> None:
        """
        加载Excel文档
        
        Args:
            file_paths: Excel文件路径列表
        """
        self.parser.load_documents(file_paths)
        logger.info(f"Loaded {len(file_paths)} documents")
    
    def query(self, question: str) -> str:
        """
        处理用户查询
        
        Args:
            question: 用户问题
            
        Returns:
            查询结果文本
        """
        # 创建初始状态
        initial_state: EnhancedAgentState = {
            "messages": [HumanMessage(content=question)],
            "parser": self.parser,
            "tool_executor": self.tool_executor,
            "query": question,
            "context": "",
            "selected_tools": [],
            "tool_results": [],
            "final_answer": None,
            "error": None,
            "iteration": 0
        }
        
        # 执行工作流
        final_state = self.graph.invoke(initial_state)
        
        # 返回最终答案
        return final_state.get("final_answer", "无法生成响应")
