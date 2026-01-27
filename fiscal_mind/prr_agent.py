"""
Plan-ReAct-Reflect (PRR) Agent模块 - 面向财务BP的智能体编排
PRR Agent module for Financial Business Partners with intelligent orchestration.

PRR架构包含三个核心组件:
1. Plan (计划): 将复杂查询分解为可执行的步骤
2. ReAct (推理-行动): 基于计划执行推理和行动，调用工具获取数据
3. Reflect (反思): 评估执行结果，决定是否需要调整计划或继续执行

这种架构特别适合回答复杂的财务分析问题，如"哪个大区今年表现更好"。
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import logging
import json

from fiscal_mind.parser import ExcelParser
from fiscal_mind.meta_functions import TableMetaFunctions
from fiscal_mind.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


class PRRAgentState(TypedDict):
    """PRR Agent状态定义"""
    messages: Annotated[List[BaseMessage], operator.add]
    parser: ExcelParser
    tool_executor: ToolExecutor
    query: str  # 用户原始查询
    context: str  # 数据上下文
    
    # Plan相关
    plan: List[str]  # 执行计划（步骤列表）
    current_step: int  # 当前执行的步骤索引
    
    # ReAct相关
    current_thought: str  # 当前思考
    current_action: Optional[Dict[str, Any]]  # 当前要执行的动作
    observations: List[Dict[str, Any]]  # 观察结果列表
    
    # Reflect相关
    reflections: List[str]  # 反思记录
    needs_replan: bool  # 是否需要重新规划
    
    # 结果
    final_answer: Optional[str]  # 最终答案
    error: Optional[str]  # 错误信息
    iteration: int  # 当前迭代次数


class PRRAgent:
    """
    Plan-ReAct-Reflect Agent
    
    这是一个基于PRR架构的智能Agent，能够:
    1. 分析复杂的财务问题并制定执行计划
    2. 通过推理和行动来执行计划
    3. 反思执行结果并在需要时调整策略
    
    特别适合回答如下问题:
    - "哪个大区今年表现更好?"
    - "哪个产品的利润率最高?"
    - "销售额增长最快的部门是哪个?"
    """
    
    def __init__(self, parser: Optional[ExcelParser] = None, llm=None, max_iterations: int = 10):
        """
        初始化PRR Agent
        
        Args:
            parser: ExcelParser实例，如果为None则创建新实例
            llm: 语言模型实例（用于增强计划生成和反思）
            max_iterations: 最大迭代次数
        """
        self.parser = parser or ExcelParser()
        self.tool_executor = ToolExecutor(self.parser)
        self.llm = llm
        self.max_iterations = max_iterations
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建PRR工作流图"""
        workflow = StateGraph(PRRAgentState)
        
        # 添加节点
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("react", self._react_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # 设置入口点
        workflow.set_entry_point("load_context")
        
        # 添加边
        workflow.add_edge("load_context", "plan")
        workflow.add_edge("plan", "react")
        workflow.add_edge("react", "reflect")
        
        # 条件边：反思后决定下一步
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "replan": "plan",      # 需要重新规划
                "continue": "react",   # 继续执行下一步
                "finish": "generate_answer"  # 完成
            }
        )
        
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _load_context_node(self, state: PRRAgentState) -> Dict[str, Any]:
        """加载数据上下文"""
        logger.info("Loading context for PRR agent...")
        
        try:
            # 创建数据上下文
            context = TableMetaFunctions.create_data_context(state["parser"], max_rows_per_sheet=5)
            
            return {
                "context": context,
                "tool_executor": self.tool_executor,
                "plan": [],
                "current_step": 0,
                "current_thought": "",
                "current_action": None,
                "observations": [],
                "reflections": [],
                "needs_replan": False,
                "iteration": 0,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error loading context: {str(e)}")
            return {
                "context": "",
                "error": f"加载上下文失败: {str(e)}"
            }
    
    def _plan_node(self, state: PRRAgentState) -> Dict[str, Any]:
        """
        Plan节点: 生成或更新执行计划
        
        根据查询和当前上下文，生成一系列步骤来回答问题。
        如果是重新规划，会考虑之前的观察和反思。
        """
        logger.info("Planning...")
        
        query = state["query"]
        context = state.get("context", "")
        observations = state.get("observations", [])
        reflections = state.get("reflections", [])
        
        # 使用LLM生成计划（如果可用）
        if self.llm:
            plan = self._llm_generate_plan(query, context, observations, reflections)
        else:
            # 基于规则的计划生成
            plan = self._rule_based_generate_plan(query, context)
        
        logger.info(f"Generated plan with {len(plan)} steps")
        for i, step in enumerate(plan, 1):
            logger.info(f"  Step {i}: {step}")
        
        return {
            "plan": plan,
            "current_step": 0,
            "needs_replan": False
        }
    
    def _llm_generate_plan(self, query: str, context: str, 
                          observations: List[Dict], reflections: List[str]) -> List[str]:
        """使用LLM生成执行计划"""
        if not self.llm:
            return self._rule_based_generate_plan(query, context)
        
        try:
            # 构建提示
            prompt_parts = [
                "你是一个财务数据分析助手。请为以下查询生成一个执行计划。",
                f"\n用户查询: {query}",
                f"\n可用数据:\n{context[:1000]}...",  # 限制上下文长度
            ]
            
            if observations:
                prompt_parts.append("\n已有观察:")
                for obs in observations[-3:]:  # 只显示最近3个观察
                    prompt_parts.append(f"- {obs.get('summary', obs)}")
            
            if reflections:
                prompt_parts.append("\n之前的反思:")
                for ref in reflections[-2:]:  # 只显示最近2个反思
                    prompt_parts.append(f"- {ref}")
            
            prompt_parts.append("\n请生成3-5个步骤的执行计划，每步一行。只返回步骤列表，不要其他解释。")
            prompt_parts.append("示例格式:")
            prompt_parts.append("1. 识别包含区域信息的数据表")
            prompt_parts.append("2. 提取各区域的关键指标")
            prompt_parts.append("3. 比较各区域的表现")
            
            prompt = "\n".join(prompt_parts)
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            plan_text = response.content.strip()
            
            # 解析计划
            plan = []
            for line in plan_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # 移除编号和标记
                    step = line.lstrip('0123456789.-•) ').strip()
                    if step:
                        plan.append(step)
            
            if plan:
                return plan
                
        except Exception as e:
            logger.error(f"LLM plan generation failed: {str(e)}")
        
        # 回退到规则方法
        return self._rule_based_generate_plan(query, context)
    
    def _rule_based_generate_plan(self, query: str, context: str) -> List[str]:
        """基于规则生成执行计划"""
        query_lower = query.lower()
        plan = []
        
        # 检测查询类型并生成相应计划
        if any(word in query_lower for word in ["哪个", "which", "最好", "最差", "比较", "对比"]):
            # 比较类查询
            if any(word in query_lower for word in ["大区", "区域", "region"]):
                plan = [
                    "识别包含区域(大区)数据的工作表",
                    "提取各个区域的关键业绩指标(如销售额、利润等)",
                    "计算各区域的总计或平均值",
                    "比较各区域的表现，找出最优者",
                    "生成详细的对比分析结果"
                ]
            elif any(word in query_lower for word in ["产品", "product"]):
                plan = [
                    "识别包含产品数据的工作表",
                    "提取各产品的销售和利润数据",
                    "计算产品的关键指标",
                    "排序并找出表现最好的产品"
                ]
            elif any(word in query_lower for word in ["部门", "department"]):
                plan = [
                    "识别包含部门数据的工作表",
                    "汇总各部门的业绩数据",
                    "比较部门表现",
                    "生成排名结果"
                ]
            else:
                # 通用比较
                plan = [
                    "分析查询中的比较维度",
                    "获取相关数据",
                    "执行比较分析",
                    "总结结果"
                ]
        
        elif any(word in query_lower for word in ["统计", "汇总", "总计", "sum", "total"]):
            # 统计类查询
            plan = [
                "识别需要统计的数据表",
                "提取相关列的数据",
                "计算统计指标(总和、平均、最大、最小等)",
                "格式化统计结果"
            ]
        
        elif any(word in query_lower for word in ["趋势", "增长", "变化", "trend", "growth"]):
            # 趋势分析
            plan = [
                "识别时间序列数据",
                "提取时间维度和度量指标",
                "计算增长率或变化趋势",
                "分析趋势特征"
            ]
        
        else:
            # 默认通用计划
            plan = [
                "理解查询意图",
                "识别相关数据源",
                "提取和处理数据",
                "分析并生成答案"
            ]
        
        return plan
    
    def _react_node(self, state: PRRAgentState) -> Dict[str, Any]:
        """
        ReAct节点: 推理(Reasoning)和行动(Acting)
        
        基于当前计划步骤，进行推理并执行具体的工具调用。
        """
        logger.info(f"ReAct: Step {state['current_step'] + 1}/{len(state['plan'])}")
        
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        
        if current_step >= len(plan):
            # 所有步骤已完成
            return {
                "current_thought": "所有计划步骤已完成",
                "current_action": None,
                "current_step": current_step
            }
        
        current_step_desc = plan[current_step]
        logger.info(f"Executing: {current_step_desc}")
        
        # 推理: 决定执行什么动作
        thought, action = self._reason_and_act(
            step_description=current_step_desc,
            query=state["query"],
            context=state.get("context", ""),
            observations=state.get("observations", [])
        )
        
        # 执行动作
        observation = None
        if action:
            tool_name = action.get("name")
            parameters = action.get("parameters", {})
            
            logger.info(f"Executing tool: {tool_name}")
            result = state["tool_executor"].execute_tool(tool_name, parameters)
            
            observation = {
                "step": current_step,
                "step_description": current_step_desc,
                "thought": thought,
                "action": action,
                "result": result,
                "summary": self._summarize_observation(current_step_desc, result)
            }
        else:
            observation = {
                "step": current_step,
                "step_description": current_step_desc,
                "thought": thought,
                "action": None,
                "result": {"success": False, "error": "No action determined"},
                "summary": "未能确定具体行动"
            }
        
        # 添加到观察列表
        new_observations = state.get("observations", []) + [observation]
        
        return {
            "current_thought": thought,
            "current_action": action,
            "observations": new_observations,
            "iteration": state.get("iteration", 0) + 1
        }
    
    def _reason_and_act(self, step_description: str, query: str, 
                       context: str, observations: List[Dict]) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        推理并决定行动
        
        Returns:
            (thought, action): 思考过程和要执行的动作
        """
        step_lower = step_description.lower()
        
        # 思考: 分析当前步骤需要做什么
        thought = f"当前步骤: {step_description}"
        action = None
        
        # 根据步骤描述决定行动
        if "识别" in step_lower and ("工作表" in step_lower or "数据表" in step_lower):
            # 需要获取文档摘要
            thought += " -> 需要查看可用的工作表列表"
            action = {
                "name": "get_document_summary",
                "parameters": {}
            }
        
        elif "提取" in step_lower or "获取" in step_lower:
            # 需要获取数据
            if "区域" in step_lower or "大区" in step_lower:
                thought += " -> 需要获取包含区域信息的数据"
                # 尝试从观察中找到合适的工作表
                doc_name, sheet_name = self._find_sheet_with_keyword(observations, ["区域", "大区", "销售"])
                action = {
                    "name": "get_sheet_data",
                    "parameters": {
                        "doc_name": doc_name,
                        "sheet_name": sheet_name,
                        "max_rows": 100
                    }
                }
            elif "产品" in step_lower:
                thought += " -> 需要获取产品相关数据"
                doc_name, sheet_name = self._find_sheet_with_keyword(observations, ["产品"])
                action = {
                    "name": "get_sheet_data",
                    "parameters": {
                        "doc_name": doc_name,
                        "sheet_name": sheet_name,
                        "max_rows": 100
                    }
                }
            else:
                # 通用数据提取
                doc_name = self._guess_doc_name()
                sheet_name = self._guess_sheet_name()
                thought += f" -> 从 {doc_name}/{sheet_name} 获取数据"
                action = {
                    "name": "get_sheet_data",
                    "parameters": {
                        "doc_name": doc_name,
                        "sheet_name": sheet_name,
                        "max_rows": 100
                    }
                }
        
        elif "计算" in step_lower or "汇总" in step_lower or "聚合" in step_lower:
            # 需要聚合数据
            thought += " -> 执行数据聚合"
            # 尝试从之前的观察中识别分组列
            group_col = self._identify_group_column(observations, query)
            doc_name = self._guess_doc_name()
            sheet_name = self._guess_sheet_name()
            
            action = {
                "name": "aggregate_data",
                "parameters": {
                    "doc_name": doc_name,
                    "sheet_name": sheet_name,
                    "group_col": group_col,
                    "agg_func": "sum"
                }
            }
        
        elif "比较" in step_lower or "排序" in step_lower or "排名" in step_lower:
            # 需要排序或比较
            thought += " -> 执行排序找出最优"
            doc_name = self._guess_doc_name()
            sheet_name = self._guess_sheet_name()
            column = self._guess_metric_column(observations, query)
            
            action = {
                "name": "get_top_n",
                "parameters": {
                    "doc_name": doc_name,
                    "sheet_name": sheet_name,
                    "column": column,
                    "n": 5,
                    "ascending": False
                }
            }
        
        elif "统计" in step_lower:
            # 获取统计信息
            thought += " -> 获取统计数据"
            doc_name = self._guess_doc_name()
            sheet_name = self._guess_sheet_name()
            action = {
                "name": "get_statistics",
                "parameters": {
                    "doc_name": doc_name,
                    "sheet_name": sheet_name
                }
            }
        
        else:
            # 默认: 获取数据概览
            thought += " -> 获取数据概览"
            action = {
                "name": "get_document_summary",
                "parameters": {}
            }
        
        return thought, action
    
    def _reflect_node(self, state: PRRAgentState) -> Dict[str, Any]:
        """
        Reflect节点: 反思和评估
        
        评估当前执行进度和结果，决定是否需要:
        1. 继续执行下一步
        2. 重新规划
        3. 结束并生成答案
        """
        logger.info("Reflecting on progress...")
        
        observations = state.get("observations", [])
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        iteration = state.get("iteration", 0)
        
        # 生成反思
        reflection = self._generate_reflection(state["query"], plan, observations, current_step)
        logger.info(f"Reflection: {reflection}")
        
        new_reflections = state.get("reflections", []) + [reflection]
        
        # 决定下一步
        needs_replan = False
        next_step = current_step + 1
        
        # 检查是否需要重新规划
        if observations and not observations[-1]["result"].get("success", False):
            # 最后一次执行失败
            if "未找到" in reflection or "失败" in reflection:
                needs_replan = True
                reflection += " -> 需要调整计划"
        
        # 检查是否达到最大迭代次数
        if iteration >= self.max_iterations:
            reflection += " -> 已达到最大迭代次数，准备生成答案"
        
        return {
            "reflections": new_reflections,
            "needs_replan": needs_replan,
            "current_step": next_step
        }
    
    def _generate_reflection(self, query: str, plan: List[str], 
                           observations: List[Dict], current_step: int) -> str:
        """生成反思"""
        if not observations:
            return "还没有任何观察结果"
        
        last_obs = observations[-1]
        result = last_obs.get("result", {})
        
        if result.get("success"):
            # 成功执行
            completed_steps = current_step + 1
            total_steps = len(plan)
            reflection = f"步骤 {completed_steps}/{total_steps} 执行成功。"
            
            # 评估是否获得了有用信息
            data = result.get("data", {})
            if isinstance(data, dict):
                if "data" in data and len(data.get("data", [])) > 0:
                    reflection += f" 获得了{len(data['data'])}条数据记录。"
                elif any(key in data for key in ["summary", "statistics", "documents"]):
                    reflection += " 获得了有用的摘要信息。"
            
            # 检查是否接近完成
            if completed_steps == total_steps:
                reflection += " 所有计划步骤已完成，可以生成答案了。"
            elif completed_steps >= total_steps - 1:
                reflection += " 即将完成所有步骤。"
        else:
            # 执行失败
            error = result.get("error", "未知错误")
            reflection = f"步骤执行失败: {error}。"
            
            # 分析失败原因
            if "未找到" in error:
                reflection += " 可能需要调整数据源或参数。"
            elif "参数" in error:
                reflection += " 参数配置可能有误。"
        
        return reflection
    
    def _should_continue(self, state: PRRAgentState) -> str:
        """判断是否继续、重新规划或结束"""
        iteration = state.get("iteration", 0)
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        needs_replan = state.get("needs_replan", False)
        observations = state.get("observations", [])
        
        # 检查是否超过最大迭代次数
        if iteration >= self.max_iterations:
            logger.info("Reached maximum iterations, finishing...")
            return "finish"
        
        # 检查是否需要重新规划
        if needs_replan and iteration < self.max_iterations - 2:
            logger.info("Replanning...")
            return "replan"
        
        # 检查是否完成所有计划步骤
        if current_step >= len(plan):
            logger.info("All plan steps completed, finishing...")
            return "finish"
        
        # 检查是否有足够的信息回答问题
        if len(observations) >= 2:
            # 如果已经有多个成功的观察，可能可以回答了
            successful_obs = [obs for obs in observations if obs["result"].get("success", False)]
            if len(successful_obs) >= 2:
                logger.info("Have sufficient observations, finishing...")
                return "finish"
        
        # 继续执行下一步
        logger.info("Continuing to next step...")
        return "continue"
    
    def _generate_answer_node(self, state: PRRAgentState) -> Dict[str, Any]:
        """生成最终答案"""
        logger.info("Generating final answer...")
        
        query = state["query"]
        plan = state.get("plan", [])
        observations = state.get("observations", [])
        reflections = state.get("reflections", [])
        
        # 使用LLM生成答案（如果可用）
        if self.llm:
            answer = self._llm_generate_answer(query, plan, observations, reflections)
        else:
            answer = self._rule_based_generate_answer(query, plan, observations, reflections)
        
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)]
        }
    
    def _llm_generate_answer(self, query: str, plan: List[str], 
                           observations: List[Dict], reflections: List[str]) -> str:
        """使用LLM生成最终答案"""
        if not self.llm:
            return self._rule_based_generate_answer(query, plan, observations, reflections)
        
        try:
            # 构建提示
            prompt_parts = [
                "基于以下分析过程，请回答用户的问题。",
                f"\n用户问题: {query}",
                "\n执行计划:"
            ]
            
            for i, step in enumerate(plan, 1):
                prompt_parts.append(f"{i}. {step}")
            
            prompt_parts.append("\n执行结果:")
            for obs in observations:
                if obs["result"].get("success"):
                    prompt_parts.append(f"- {obs['summary']}")
            
            if reflections:
                prompt_parts.append("\n分析反思:")
                for ref in reflections[-3:]:
                    prompt_parts.append(f"- {ref}")
            
            prompt_parts.append("\n请基于以上信息，直接回答用户的问题。答案要简洁明了，包含具体数据。")
            
            prompt = "\n".join(prompt_parts)
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM answer generation failed: {str(e)}")
            return self._rule_based_generate_answer(query, plan, observations, reflections)
    
    def _rule_based_generate_answer(self, query: str, plan: List[str],
                                   observations: List[Dict], reflections: List[str]) -> str:
        """基于规则生成答案"""
        answer_parts = []
        answer_parts.append(f"问题: {query}\n")
        
        # 添加执行摘要
        successful_obs = [obs for obs in observations if obs["result"].get("success", False)]
        
        if successful_obs:
            answer_parts.append("分析结果:\n")
            
            # 格式化每个成功的观察结果
            for i, obs in enumerate(successful_obs, 1):
                answer_parts.append(f"{i}. {obs['step_description']}")
                answer_parts.append(f"   {obs['summary']}")
                
                # 如果有具体数据，展示关键信息
                result_data = obs["result"].get("data", {})
                if isinstance(result_data, dict) and "data" in result_data:
                    data_items = result_data["data"]
                    if isinstance(data_items, list) and len(data_items) > 0:
                        answer_parts.append(f"   获得 {len(data_items)} 条记录")
                        # 显示前3条
                        for j, item in enumerate(data_items[:3], 1):
                            answer_parts.append(f"     {j}. {item}")
                answer_parts.append("")
            
            # 尝试生成结论
            if "哪个" in query or "which" in query.lower():
                # 比较类问题，尝试找出最优结果
                answer_parts.append("结论:")
                answer_parts.append(self._extract_conclusion(query, successful_obs))
        else:
            answer_parts.append("抱歉，未能找到足够的数据来回答您的问题。")
            if observations:
                answer_parts.append("\n尝试的步骤:")
                for obs in observations:
                    answer_parts.append(f"- {obs['step_description']}: {obs['result'].get('error', '未知错误')}")
        
        return "\n".join(answer_parts)
    
    def _extract_conclusion(self, query: str, observations: List[Dict]) -> str:
        """从观察结果中提取结论"""
        # 查找包含排序或比较结果的观察
        for obs in reversed(observations):
            result_data = obs["result"].get("data", {})
            if isinstance(result_data, dict) and "data" in result_data:
                data_items = result_data["data"]
                if isinstance(data_items, list) and len(data_items) > 0:
                    first_item = data_items[0]
                    return f"根据分析，{first_item} 表现最佳。"
        
        return "请参考上述数据进行判断。"
    
    # 辅助方法
    def _summarize_observation(self, step_desc: str, result: Dict) -> str:
        """总结观察结果"""
        if not result.get("success", False):
            return f"执行失败: {result.get('error', '未知错误')}"
        
        data = result.get("data", {})
        
        if isinstance(data, dict):
            if "documents" in data:
                doc_count = len(data["documents"])
                return f"找到 {doc_count} 个文档"
            elif "data" in data:
                items = data["data"]
                if isinstance(items, list):
                    return f"获取 {len(items)} 条数据"
            elif "statistics" in data:
                return "获取统计信息"
        
        return "执行成功"
    
    def _find_sheet_with_keyword(self, observations: List[Dict], keywords: List[str]) -> tuple[str, str]:
        """从观察中找到包含关键词的工作表"""
        # 查找文档摘要观察
        for obs in observations:
            result = obs.get("result", {})
            if result.get("success") and "documents" in result.get("data", {}):
                docs = result["data"]["documents"]
                # 遍历文档和工作表
                for doc_name, doc_info in docs.items():
                    sheets = doc_info.get("sheets", {})
                    for sheet_name, sheet_info in sheets.items():
                        # 检查列名
                        columns = sheet_info.get("columns", [])
                        for keyword in keywords:
                            if any(keyword in str(col) for col in columns):
                                return doc_name, sheet_name
        
        # 如果没找到，返回默认值
        return self._guess_doc_name(), self._guess_sheet_name()
    
    def _identify_group_column(self, observations: List[Dict], query: str) -> str:
        """识别用于分组的列"""
        # 从查询中提取
        if "区域" in query or "大区" in query:
            return "区域"
        elif "产品" in query:
            return "产品"
        elif "部门" in query:
            return "部门"
        
        # 从观察的数据中推断
        for obs in observations:
            result = obs.get("result", {})
            if result.get("success"):
                data = result.get("data", {})
                if isinstance(data, dict) and "columns" in data:
                    columns = data["columns"]
                    for col in columns:
                        if any(keyword in str(col) for keyword in ["区域", "产品", "部门", "类别"]):
                            return col
        
        return "区域"  # 默认
    
    def _guess_metric_column(self, observations: List[Dict], query: str) -> str:
        """猜测度量列"""
        # 从查询中提取
        if "销售" in query:
            return "销售额"
        elif "利润" in query:
            return "利润"
        elif "收入" in query:
            return "收入"
        
        # 从数据中推断
        for obs in observations:
            result = obs.get("result", {})
            if result.get("success"):
                data = result.get("data", {})
                if isinstance(data, dict) and "columns" in data:
                    columns = data["columns"]
                    for col in columns:
                        if "销售" in str(col) or "金额" in str(col):
                            return col
        
        return "销售额"  # 默认
    
    def _guess_doc_name(self) -> str:
        """猜测文档名"""
        docs = list(self.parser.documents.keys())
        return docs[0] if docs else ""
    
    def _guess_sheet_name(self) -> str:
        """猜测工作表名"""
        doc_name = self._guess_doc_name()
        if doc_name:
            doc = self.parser.get_document(doc_name)
            if doc:
                sheets = doc.get_sheet_names()
                return sheets[0] if sheets else ""
        return ""
    
    # 公共接口
    def load_documents(self, file_paths: List[str]) -> None:
        """加载Excel文档"""
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
        initial_state: PRRAgentState = {
            "messages": [HumanMessage(content=question)],
            "parser": self.parser,
            "tool_executor": self.tool_executor,
            "query": question,
            "context": "",
            "plan": [],
            "current_step": 0,
            "current_thought": "",
            "current_action": None,
            "observations": [],
            "reflections": [],
            "needs_replan": False,
            "final_answer": None,
            "error": None,
            "iteration": 0
        }
        
        # 执行工作流
        final_state = self.graph.invoke(initial_state)
        
        # 返回最终答案
        return final_state.get("final_answer", "无法生成响应")
