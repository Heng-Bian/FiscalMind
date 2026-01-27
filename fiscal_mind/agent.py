"""
LangGraph Agent模块 - 表格文档分析智能体
Agent module using LangGraph for table document analysis.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import logging

from fiscal_mind.parser import ExcelParser, ExcelDocument
from fiscal_mind.meta_functions import TableMetaFunctions, TableQueryHelper
from fiscal_mind.semantic_resolver import SemanticResolver

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Agent状态定义"""
    messages: Annotated[List[BaseMessage], operator.add]
    parser: ExcelParser
    current_documents: List[str]  # 当前正在处理的文档名称列表
    query: str  # 用户查询
    context: str  # 为LLM准备的上下文
    result: Optional[Dict[str, Any]]  # 查询结果
    error: Optional[str]  # 错误信息


class TableDocumentAgent:
    """表格文档分析Agent"""
    
    def __init__(self, parser: Optional[ExcelParser] = None, llm=None):
        """
        初始化Agent
        
        Args:
            parser: ExcelParser实例，如果为None则创建新实例
            llm: 语言模型实例（可选，用于增强查询分析）
        """
        self.parser = parser or ExcelParser(llm=llm)
        self.llm = llm
        self.semantic_resolver = SemanticResolver(llm=llm)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("execute_query", self._execute_query_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # 设置入口点
        workflow.set_entry_point("load_context")
        
        # 添加边
        workflow.add_edge("load_context", "analyze_query")
        workflow.add_edge("analyze_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def _load_context_node(self, state: AgentState) -> Dict[str, Any]:
        """加载文档上下文节点"""
        logger.info("Loading document context...")
        
        try:
            # 创建数据上下文
            context = TableMetaFunctions.create_data_context(state["parser"])
            
            return {
                "context": context,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error loading context: {str(e)}")
            return {
                "context": "",
                "error": f"加载上下文失败: {str(e)}"
            }
    
    def _analyze_query_node(self, state: AgentState) -> Dict[str, Any]:
        """分析查询节点（支持LLM增强的意图识别）"""
        logger.info(f"Analyzing query: {state['query']}")
        
        query_lower = state["query"].lower()
        
        query_info = {
            "type": "general",  # general, search, statistics, aggregation, filter, join
            "keywords": []
        }
        
        # 扩展的关键词库（支持更丰富的同义词和表达）
        intent_keywords = {
            "search": ["搜索", "查找", "找到", "寻找", "find", "search", "look for", "locate"],
            "statistics": ["统计", "汇总", "总结", "概览", "statistics", "summary", "summarize", "aggregate", "计算", "算一下", "平均"],
            "aggregation": ["聚合", "分组", "汇总", "group", "aggregate", "by", "按照"],
            "filter": ["过滤", "筛选", "选择", "filter", "where", "条件", "满足"],
            "comparison": ["对比", "比较", "compare", "差异", "区别"],
            "join": ["关联", "连接", "合并", "join", "merge", "组合"],
            "sort": ["排序", "排列", "sort", "order", "最大", "最小", "top"],
        }
        
        # 使用LLM进行意图分类（如果可用）
        if self.llm:
            try:
                query_type = self._classify_query_with_llm(state["query"])
                if query_type:
                    query_info["type"] = query_type
                    logger.info(f"LLM classified query type as: {query_type}")
                    return {"result": query_info}
            except Exception as e:
                logger.warning(f"LLM intent classification failed: {str(e)}, falling back to keyword matching")
        
        # 关键词匹配（保底机制）
        for intent_type, keywords in intent_keywords.items():
            if any(word in query_lower for word in keywords):
                query_info["type"] = intent_type
                break
        
        return {
            "result": query_info
        }
    
    def _classify_query_with_llm(self, query: str) -> Optional[str]:
        """使用LLM分类查询意图"""
        if not self.llm:
            return None
        
        try:
            prompt = f"""Classify the following user query into ONE of these categories:
- search: Finding specific data or records
- statistics: Computing statistics, averages, sums, counts
- aggregation: Grouping and aggregating data
- filter: Filtering data based on conditions
- comparison: Comparing different data points
- join: Joining or merging tables
- sort: Sorting or ranking data
- general: General questions or requests

User query: "{query}"

Return ONLY the category name (one word), nothing else."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip().lower()
            
            # Validate the result
            valid_types = ["search", "statistics", "aggregation", "filter", "comparison", "join", "sort", "general"]
            if result in valid_types:
                return result
                
        except Exception as e:
            logger.error(f"LLM classification error: {str(e)}")
        
        return None
    
    def _execute_query_node(self, state: AgentState) -> Dict[str, Any]:
        """执行查询节点"""
        logger.info("Executing query...")
        
        try:
            parser = state["parser"]
            query_type = state.get("result", {}).get("type", "general")
            
            # 根据查询类型执行不同操作
            if query_type == "statistics":
                # 获取统计信息
                result = self._get_statistics(parser)
            elif query_type == "search":
                # 执行搜索
                result = self._perform_search(parser, state["query"])
            else:
                # 一般查询，返回文档摘要
                result = parser.get_documents_summary()
            
            return {
                "result": result,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "result": None,
                "error": f"执行查询失败: {str(e)}"
            }
    
    def _format_response_node(self, state: AgentState) -> Dict[str, Any]:
        """格式化响应节点"""
        logger.info("Formatting response...")
        
        if state.get("error"):
            response = AIMessage(content=f"错误: {state['error']}")
        else:
            # 格式化结果为可读文本
            result = state.get("result", {})
            response_text = self._format_result(result)
            response = AIMessage(content=response_text)
        
        return {
            "messages": [response]
        }
    
    def _get_statistics(self, parser: ExcelParser) -> Dict[str, Any]:
        """获取所有文档的统计信息"""
        stats = {}
        for doc_name, doc in parser.documents.items():
            doc_stats = {}
            for sheet_name in doc.get_sheet_names():
                df = doc.get_sheet(sheet_name)
                doc_stats[sheet_name] = TableMetaFunctions.get_numeric_summary(df)
            stats[doc_name] = doc_stats
        return stats
    
    def _perform_search(self, parser: ExcelParser, query: str) -> Dict[str, Any]:
        """执行搜索操作"""
        # 提取搜索关键词（简单实现）
        words = query.split()
        search_term = words[-1] if words else ""
        
        results = parser.search_across_documents(search_term)
        return results
    
    def _format_result(self, result: Any) -> str:
        """格式化结果为文本"""
        if isinstance(result, dict):
            lines = []
            for key, value in result.items():
                if isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        lines.append(f"  {k}: {v}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
        return str(result)
    
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
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "parser": self.parser,
            "current_documents": list(self.parser.documents.keys()),
            "query": question,
            "context": "",
            "result": None,
            "error": None
        }
        
        # 执行工作流
        final_state = self.graph.invoke(initial_state)
        
        # 提取响应
        if final_state.get("messages"):
            last_message = final_state["messages"][-1]
            return last_message.content
        
        return "无法生成响应"
    
    def get_document_summary(self, doc_name: Optional[str] = None) -> str:
        """
        获取文档摘要
        
        Args:
            doc_name: 文档名称，如果为None则返回所有文档摘要
            
        Returns:
            摘要文本
        """
        if doc_name:
            doc = self.parser.get_document(doc_name)
            if doc:
                return TableMetaFunctions.summarize_document_for_llm(doc)
            return f"文档 {doc_name} 未找到"
        else:
            return TableMetaFunctions.create_data_context(self.parser)
    
    def analyze_sheet(self, doc_name: str, sheet_name: str) -> str:
        """
        分析特定工作表
        
        Args:
            doc_name: 文档名称
            sheet_name: 工作表名称
            
        Returns:
            分析结果文本
        """
        doc = self.parser.get_document(doc_name)
        if not doc:
            return f"文档 {doc_name} 未找到"
        
        df = doc.get_sheet(sheet_name)
        if df is None:
            return f"工作表 {sheet_name} 未找到"
        
        return TableMetaFunctions.format_for_llm_context(df)
