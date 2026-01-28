"""
专业智能体模块 - 业务分析、批评、和评判智能体
Specialized Agents Module - Business Analysis, Critic, and Judgment Agents.

这个模块包含四个专业智能体:
1. BusinessAnalysisAgent (Agent A): 分析业务场景
2. CriticAgent (Agent B): 批评和建议
3. JudgmentAgent (Agent C): 评判结论的合理性
4. HeaderDetectionAgent: 使用LLM检测表头，避免将数据行误判为表头
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# 评分相关常量
MATCH_SCORE_PER_MATCH = 3  # 每个匹配点的分值
MIN_ACCEPTABLE_MATCH_SCORE = 6  # 可接受的最低匹配分数
HIGH_MATCH_SCORE_THRESHOLD = 8  # 高匹配度阈值

# 评判相关常量
HIGH_ANSWER_RELEVANCE_SCORE = 8  # 明确回答问题的相关性分数
DEFAULT_ANSWER_RELEVANCE_SCORE = 5  # 默认相关性分数
HIGH_DATA_SUPPORT_SCORE = 8  # 有数据支撑的分数
LOW_DATA_SUPPORT_SCORE = 4  # 缺少数据支撑的分数
BASE_BUSINESS_LOGIC_SCORE = 5  # 业务逻辑基础分数
METRIC_BONUS_SCORE = 2  # 每个提到的指标的奖励分数
ACCEPTABLE_QUALITY_THRESHOLD = 6  # 可接受的质量阈值
SHORT_CONCLUSION_LENGTH = 50  # 简短结论的字符数
DETAILED_CONCLUSION_LENGTH = 200  # 详细结论的字符数


class BusinessAnalysisAgent:
    """
    业务分析智能体 (Agent A)
    
    职责:
    - 获取所有表头和描述
    - 分析少量示例数据
    - 识别可能的业务场景
    - 理解数据背后的业务逻辑
    """
    
    def __init__(self, llm=None):
        """
        初始化业务分析智能体
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
    
    def analyze_business_context(self, context: str, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析业务上下文
        
        Args:
            context: 数据上下文（表头、描述等）
            sample_data: 示例数据
            
        Returns:
            业务分析结果
        """
        logger.info("BusinessAnalysisAgent: Analyzing business context...")
        
        if self.llm:
            return self._llm_analyze_business(context, sample_data)
        else:
            return self._rule_based_business_analysis(context, sample_data)
    
    def _llm_analyze_business(self, context: str, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用LLM进行业务分析"""
        try:
            prompt_parts = [
                "你是一位资深的财务业务分析专家。请基于以下数据的表头、描述和示例，深入分析可能的业务场景。",
                "\n数据结构和上下文:",
                context[:2000],  # 限制长度
                "\n示例数据（前几条）:"
            ]
            
            # 添加示例数据
            for i, data in enumerate(sample_data[:5], 1):
                prompt_parts.append(f"{i}. {data}")
            
            prompt_parts.extend([
                "\n请分析以下方面:",
                "1. 这些数据可能属于什么业务领域？（如：销售、财务、人力资源、医疗等）",
                "2. 数据中的关键业务维度是什么？（如：区域、产品、时间、部门等）",
                "3. 数据中的关键业务指标是什么？（如：销售额、利润、成本、效率等）",
                "4. 这些数据之间可能存在什么业务关系？",
                "5. 可能涉及哪些业务分析场景？（如：绩效评估、预算对比、趋势分析等）",
                "6. 数据的业务含义和背景是什么？",
                "\n请以结构化的JSON格式回答，包含以下字段:",
                "- business_domain: 业务领域",
                "- key_dimensions: 关键业务维度列表",
                "- key_metrics: 关键业务指标列表",
                "- business_relationships: 业务关系描述",
                "- analysis_scenarios: 可能的分析场景列表",
                "- business_context: 业务背景和含义",
                "- confidence_level: 分析的置信度 (high/medium/low)"
            ])
            
            prompt = "\n".join(prompt_parts)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # 尝试解析JSON
            try:
                # 提取JSON部分（如果被包裹在其他文本中）
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                analysis = json.loads(result_text)
            except json.JSONDecodeError:
                # 如果JSON解析失败，创建基于文本的结构
                analysis = {
                    "business_domain": "未知领域",
                    "key_dimensions": [],
                    "key_metrics": [],
                    "business_relationships": result_text,
                    "analysis_scenarios": [],
                    "business_context": result_text,
                    "confidence_level": "low"
                }
            
            logger.info(f"BusinessAnalysisAgent: Identified business domain: {analysis.get('business_domain', 'Unknown')}")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM business analysis failed: {str(e)}")
            return self._rule_based_business_analysis(context, sample_data)
    
    def _rule_based_business_analysis(self, context: str, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于规则的业务分析"""
        context_lower = context.lower()
        
        # 识别业务领域
        domain_keywords = {
            "销售": ["销售", "sales", "订单", "客户", "产品"],
            "财务": ["财务", "finance", "利润", "成本", "预算", "收入", "支出"],
            "人力资源": ["员工", "人员", "薪资", "工资", "部门"],
            "医疗": ["医保", "医院", "患者", "费用", "准入"],
            "库存": ["库存", "inventory", "仓库", "出入库"],
        }
        
        business_domain = "通用业务"
        for domain, keywords in domain_keywords.items():
            if any(kw in context_lower for kw in keywords):
                business_domain = domain
                break
        
        # 识别关键维度
        dimension_keywords = ["区域", "大区", "产品", "部门", "月份", "季度", "年份", "类别"]
        key_dimensions = [dim for dim in dimension_keywords if dim in context_lower]
        
        # 识别关键指标
        metric_keywords = ["销售额", "利润", "成本", "预算", "实际", "差额", "增长率", "覆盖率", "效率"]
        key_metrics = [metric for metric in metric_keywords if metric in context_lower]
        
        # 识别分析场景
        analysis_scenarios = []
        if any(kw in context_lower for kw in ["预算", "实际", "差额"]):
            analysis_scenarios.append("预算执行分析")
        if any(kw in context_lower for kw in ["区域", "大区"]):
            analysis_scenarios.append("区域绩效对比")
        if any(kw in context_lower for kw in ["增长", "趋势"]):
            analysis_scenarios.append("趋势分析")
        if any(kw in context_lower for kw in ["效率", "产出"]):
            analysis_scenarios.append("效率分析")
        
        return {
            "business_domain": business_domain,
            "key_dimensions": key_dimensions,
            "key_metrics": key_metrics,
            "business_relationships": f"这是{business_domain}领域的数据，主要维度包括{', '.join(key_dimensions)}",
            "analysis_scenarios": analysis_scenarios if analysis_scenarios else ["综合分析"],
            "business_context": f"数据涉及{business_domain}，可能需要从{', '.join(key_metrics)}等指标进行分析",
            "confidence_level": "medium"
        }


class CriticAgent:
    """
    批评者智能体 (Agent B)
    
    职责:
    - 评估业务分析智能体的结论
    - 检查分析是否与用户问题匹配
    - 提供改进建议
    - 与业务分析智能体多轮协作
    """
    
    def __init__(self, llm=None):
        """
        初始化批评者智能体
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
    
    def critique_analysis(self, user_query: str, business_analysis: Dict[str, Any],
                         previous_critiques: List[str] = None) -> Dict[str, Any]:
        """
        批评业务分析
        
        Args:
            user_query: 用户问题
            business_analysis: 业务分析结果
            previous_critiques: 之前的批评记录
            
        Returns:
            批评结果，包含匹配度和建议
        """
        logger.info("CriticAgent: Critiquing business analysis...")
        
        if self.llm:
            return self._llm_critique(user_query, business_analysis, previous_critiques)
        else:
            return self._rule_based_critique(user_query, business_analysis)
    
    def _llm_critique(self, user_query: str, business_analysis: Dict[str, Any],
                     previous_critiques: List[str] = None) -> Dict[str, Any]:
        """使用LLM进行批评"""
        try:
            prompt_parts = [
                "你是一位严谨的分析审查专家。请评估业务分析是否与用户问题匹配，并提供建议。",
                f"\n用户问题: {user_query}",
                "\n业务分析结果:",
                f"- 业务领域: {business_analysis.get('business_domain', '未知')}",
                f"- 关键维度: {', '.join(business_analysis.get('key_dimensions', []))}",
                f"- 关键指标: {', '.join(business_analysis.get('key_metrics', []))}",
                f"- 分析场景: {', '.join(business_analysis.get('analysis_scenarios', []))}",
                f"- 业务背景: {business_analysis.get('business_context', '')}",
            ]
            
            if previous_critiques:
                prompt_parts.append("\n之前的批评意见:")
                for i, critique in enumerate(previous_critiques, 1):
                    prompt_parts.append(f"{i}. {critique}")
            
            prompt_parts.extend([
                "\n请评估以下方面:",
                "1. 业务分析是否准确理解了数据的业务含义？",
                "2. 识别出的维度和指标是否与用户问题相关？",
                "3. 分析场景是否涵盖了用户问题的需求？",
                "4. 是否有遗漏的重要业务要素？",
                "5. 需要补充哪些分析才能更好地回答用户问题？",
                "\n请以JSON格式回答，包含:",
                "- match_score: 匹配度评分(0-10)",
                "- matches: 匹配的方面列表",
                "- gaps: 缺失的方面列表",
                "- suggestions: 改进建议列表",
                "- needs_refinement: 是否需要重新分析 (true/false)",
                "- reasoning: 评估理由"
            ])
            
            prompt = "\n".join(prompt_parts)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # 解析JSON
            try:
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                critique = json.loads(result_text)
            except json.JSONDecodeError:
                critique = {
                    "match_score": 5,
                    "matches": [],
                    "gaps": [],
                    "suggestions": [result_text],
                    "needs_refinement": False,
                    "reasoning": result_text
                }
            
            logger.info(f"CriticAgent: Match score: {critique.get('match_score', 0)}/10")
            return critique
            
        except Exception as e:
            logger.error(f"LLM critique failed: {str(e)}")
            return self._rule_based_critique(user_query, business_analysis)
    
    def _rule_based_critique(self, user_query: str, business_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """基于规则的批评"""
        query_lower = user_query.lower()
        
        # 检查关键词匹配
        matches = []
        gaps = []
        
        # 检查维度匹配
        key_dimensions = business_analysis.get('key_dimensions', [])
        if any(dim in query_lower for dim in ["区域", "大区"]):
            if any(dim in key_dimensions for dim in ["区域", "大区"]):
                matches.append("识别了区域维度")
            else:
                gaps.append("未识别区域维度")
        
        # 检查指标匹配
        key_metrics = business_analysis.get('key_metrics', [])
        metric_keywords = ["销售", "利润", "成本", "预算", "效率"]
        query_metrics = [kw for kw in metric_keywords if kw in query_lower]
        
        if query_metrics:
            matched_metrics = [m for m in key_metrics if any(qm in m for qm in query_metrics)]
            if matched_metrics:
                matches.append(f"识别了相关指标: {', '.join(matched_metrics)}")
            else:
                gaps.append(f"未识别查询中的指标: {', '.join(query_metrics)}")
        
        # 检查分析场景
        if "表现" in query_lower or "更好" in query_lower or "对比" in query_lower:
            if "绩效对比" in business_analysis.get('analysis_scenarios', []) or \
               "综合分析" in business_analysis.get('analysis_scenarios', []):
                matches.append("分析场景包含绩效对比")
            else:
                gaps.append("缺少绩效对比分析场景")
        
        # 计算匹配度 (使用模块级常量)
        match_score = min(10, len(matches) * MATCH_SCORE_PER_MATCH) if matches else MATCH_SCORE_PER_MATCH
        
        suggestions = []
        if gaps:
            suggestions.append(f"建议补充: {'; '.join(gaps)}")
        if match_score < 7:
            suggestions.append("建议重新审视数据结构，确保覆盖用户问题的关键要素")
        
        return {
            "match_score": match_score,
            "matches": matches,
            "gaps": gaps,
            "suggestions": suggestions if suggestions else ["分析基本符合要求"],
            "needs_refinement": match_score < MIN_ACCEPTABLE_MATCH_SCORE,
            "reasoning": f"匹配度评分 {match_score}/10, 基于 {len(matches)} 个匹配点"
        }


class JudgmentAgent:
    """
    评判智能体 (Agent C)
    
    职责:
    - 结合业务分析
    - 评估最终结论的合理性
    - 验证结论与实际数据的一致性
    - 判断分析的质量和可信度
    """
    
    def __init__(self, llm=None):
        """
        初始化评判智能体
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
    
    def judge_conclusion(self, user_query: str, business_analysis: Dict[str, Any],
                        final_conclusion: str, actual_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评判最终结论
        
        Args:
            user_query: 用户问题
            business_analysis: 业务分析结果
            final_conclusion: 最终结论
            actual_data: 实际数据
            
        Returns:
            评判结果
        """
        logger.info("JudgmentAgent: Judging final conclusion...")
        
        if self.llm:
            return self._llm_judge(user_query, business_analysis, final_conclusion, actual_data)
        else:
            return self._rule_based_judge(user_query, business_analysis, final_conclusion, actual_data)
    
    def _llm_judge(self, user_query: str, business_analysis: Dict[str, Any],
                  final_conclusion: str, actual_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用LLM进行评判"""
        try:
            prompt_parts = [
                "你是一位资深的分析质量评审专家。请评判最终分析结论的合理性和质量。",
                f"\n用户问题: {user_query}",
                "\n业务背景:",
                f"- 业务领域: {business_analysis.get('business_domain', '未知')}",
                f"- 关键维度: {', '.join(business_analysis.get('key_dimensions', []))}",
                f"- 关键指标: {', '.join(business_analysis.get('key_metrics', []))}",
                f"- 业务场景: {', '.join(business_analysis.get('analysis_scenarios', []))}",
                "\n最终结论:",
                final_conclusion[:1000],  # 限制长度
                "\n实际数据样本（前几条）:"
            ]
            
            for i, data in enumerate(actual_data[:5], 1):
                prompt_parts.append(f"{i}. {data}")
            
            prompt_parts.extend([
                "\n请评判以下方面:",
                "1. 结论是否直接回答了用户的问题？",
                "2. 结论是否有数据支撑？提到的数据是否准确？",
                "3. 结论是否符合业务逻辑和业务背景？",
                "4. 分析是否全面？是否考虑了多个维度？",
                "5. 结论的表达是否清晰、专业？",
                "6. 是否有明显的逻辑错误或矛盾？",
                "\n请以JSON格式回答，包含:",
                "- overall_quality: 总体质量评分(0-10)",
                "- answer_relevance: 问题相关性评分(0-10)",
                "- data_support: 数据支撑评分(0-10)",
                "- business_logic: 业务逻辑评分(0-10)",
                "- comprehensiveness: 全面性评分(0-10)",
                "- strengths: 优点列表",
                "- weaknesses: 不足列表",
                "- is_acceptable: 结论是否可接受 (true/false)",
                "- improvement_suggestions: 改进建议列表",
                "- final_verdict: 最终评语"
            ])
            
            prompt = "\n".join(prompt_parts)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # 解析JSON
            try:
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                judgment = json.loads(result_text)
            except json.JSONDecodeError:
                judgment = {
                    "overall_quality": 5,
                    "answer_relevance": 5,
                    "data_support": 5,
                    "business_logic": 5,
                    "comprehensiveness": 5,
                    "strengths": [],
                    "weaknesses": [],
                    "is_acceptable": True,
                    "improvement_suggestions": [result_text],
                    "final_verdict": result_text
                }
            
            logger.info(f"JudgmentAgent: Overall quality: {judgment.get('overall_quality', 0)}/10")
            return judgment
            
        except Exception as e:
            logger.error(f"LLM judgment failed: {str(e)}")
            return self._rule_based_judge(user_query, business_analysis, final_conclusion, actual_data)
    
    def _rule_based_judge(self, user_query: str, business_analysis: Dict[str, Any],
                         final_conclusion: str, actual_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于规则的评判"""
        strengths = []
        weaknesses = []
        
        conclusion_lower = final_conclusion.lower()
        query_lower = user_query.lower()
        
        # 检查是否回答了问题
        if "哪个" in query_lower:
            if any(word in conclusion_lower for word in ["根据分析", "表现最", "最好", "最佳", "领先"]):
                strengths.append("明确给出了对比结论")
            else:
                weaknesses.append("未明确指出哪个选项更好")
        
        # 检查是否有数据支撑
        has_numbers = any(char.isdigit() for char in final_conclusion)
        if has_numbers:
            strengths.append("结论包含具体数据")
        else:
            weaknesses.append("缺少具体数据支撑")
        
        # 检查是否提到关键维度
        key_dimensions = business_analysis.get('key_dimensions', [])
        mentioned_dims = [dim for dim in key_dimensions if dim in conclusion_lower]
        if mentioned_dims:
            strengths.append(f"提到了关键维度: {', '.join(mentioned_dims)}")
        
        # 检查是否提到关键指标
        key_metrics = business_analysis.get('key_metrics', [])
        mentioned_metrics = [metric for metric in key_metrics if metric in conclusion_lower]
        if mentioned_metrics:
            strengths.append(f"分析了关键指标: {', '.join(mentioned_metrics)}")
        else:
            weaknesses.append("未充分分析关键业务指标")
        
        # 检查结论长度 (使用模块级常量)
        if len(final_conclusion) < SHORT_CONCLUSION_LENGTH:
            weaknesses.append("结论过于简单，缺少详细分析")
        elif len(final_conclusion) > DETAILED_CONCLUSION_LENGTH:
            strengths.append("提供了详细的分析说明")
        
        # 计算评分 (使用模块级常量)
        answer_relevance = HIGH_ANSWER_RELEVANCE_SCORE if "明确给出了对比结论" in strengths else DEFAULT_ANSWER_RELEVANCE_SCORE
        data_support = HIGH_DATA_SUPPORT_SCORE if has_numbers else LOW_DATA_SUPPORT_SCORE
        business_logic = min(10, BASE_BUSINESS_LOGIC_SCORE + len(mentioned_metrics) * METRIC_BONUS_SCORE)
        comprehensiveness = min(10, BASE_BUSINESS_LOGIC_SCORE + len(mentioned_dims) + len(mentioned_metrics))
        overall_quality = (answer_relevance + data_support + business_logic + comprehensiveness) / 4
        
        is_acceptable = overall_quality >= ACCEPTABLE_QUALITY_THRESHOLD and len(weaknesses) <= len(strengths)
        
        improvement_suggestions = []
        if not has_numbers:
            improvement_suggestions.append("建议在结论中加入具体的数据和数字")
        if len(mentioned_metrics) < 2:
            improvement_suggestions.append("建议从多个指标维度进行综合分析")
        if len(final_conclusion) < SHORT_CONCLUSION_LENGTH:
            improvement_suggestions.append("建议提供更详细的分析过程和理由")
        
        return {
            "overall_quality": round(overall_quality, 1),
            "answer_relevance": answer_relevance,
            "data_support": data_support,
            "business_logic": business_logic,
            "comprehensiveness": comprehensiveness,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "is_acceptable": is_acceptable,
            "improvement_suggestions": improvement_suggestions if improvement_suggestions else ["分析质量良好"],
            "final_verdict": f"总体评分 {overall_quality:.1f}/10, {'可接受' if is_acceptable else '需要改进'}"
        }


def collaborate_business_and_critic(business_agent: BusinessAnalysisAgent,
                                    critic_agent: CriticAgent,
                                    user_query: str,
                                    context: str,
                                    sample_data: List[Dict[str, Any]],
                                    max_rounds: int = 3) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    业务分析智能体和批评者智能体的多轮协作
    
    Args:
        business_agent: 业务分析智能体
        critic_agent: 批评者智能体
        user_query: 用户问题
        context: 数据上下文
        sample_data: 示例数据
        max_rounds: 最大协作轮数
        
    Returns:
        (最终业务分析, 协作历史)
    """
    logger.info(f"Starting collaboration between BusinessAgent and CriticAgent (max {max_rounds} rounds)...")
    
    collaboration_history = []
    current_analysis = None
    previous_critiques = []
    enhanced_context = context  # 增强的上下文，包含之前的反馈
    
    for round_num in range(1, max_rounds + 1):
        logger.info(f"Collaboration round {round_num}/{max_rounds}")
        
        # 业务分析（使用增强的上下文）
        current_analysis = business_agent.analyze_business_context(enhanced_context, sample_data)
        
        # 批评评估
        critique = critic_agent.critique_analysis(user_query, current_analysis, previous_critiques)
        
        # 记录这一轮
        collaboration_history.append({
            "round": round_num,
            "analysis": current_analysis,
            "critique": critique
        })
        
        # 检查是否需要继续
        if not critique.get("needs_refinement", False) or critique.get("match_score", 0) >= HIGH_MATCH_SCORE_THRESHOLD:
            logger.info(f"Collaboration completed after {round_num} round(s)")
            break
        
        # 准备下一轮：增强上下文以包含批评建议
        previous_critiques.append(critique.get("reasoning", ""))
        
        # 将批评建议融入上下文，帮助下一轮分析改进
        if critique.get("suggestions"):
            logger.info(f"Applying suggestions: {critique['suggestions']}")
            suggestions_text = "\n重要提示（基于前轮反馈）:\n" + "\n".join(f"- {s}" for s in critique['suggestions'])
            enhanced_context = context + suggestions_text
    
    return current_analysis, collaboration_history


class HeaderDetectionAgent:
    """
    表头检测智能体
    
    职责:
    - 使用LLM分析表格的前N行数据
    - 智能识别哪些行是表头，哪些是数据行
    - 避免将数据行误判为表头
    - 支持多行表头的检测
    """
    
    def __init__(self, llm=None):
        """
        初始化表头检测智能体
        
        Args:
            llm: 语言模型实例，如果为None则使用规则基础方法
        """
        self.llm = llm
    
    def detect_header_rows(self, rows_data: List[List[Any]], max_rows: int = 10) -> Dict[str, Any]:
        """
        检测表头行
        
        Args:
            rows_data: 表格的前N行数据（二维列表）
            max_rows: 最多检查的行数，默认10行
            
        Returns:
            包含以下字段的字典:
            - header_row_count: 表头行数（从第0行开始算）
            - header_rows_indices: 表头行的索引列表
            - data_start_row: 数据开始的行索引
            - confidence: 检测置信度 (0.0-1.0)
            - reasoning: 检测理由
        """
        logger.info("HeaderDetectionAgent: Detecting header rows...")
        
        if self.llm:
            return self._llm_detect_headers(rows_data, max_rows)
        else:
            return self._rule_based_detect_headers(rows_data, max_rows)
    
    def _llm_detect_headers(self, rows_data: List[List[Any]], max_rows: int) -> Dict[str, Any]:
        """使用LLM检测表头"""
        try:
            # 限制行数
            rows_to_analyze = rows_data[:max_rows]
            
            # 构建prompt
            prompt_parts = [
                "你是一位专业的数据分析专家。请分析以下表格的前若干行数据，识别哪些行是表头，哪些行是数据行。",
                "\n重要提示:",
                "- 表头通常包含列名、字段名等描述性文本",
                "- 数据行通常包含具体的数值、名称、日期等实际数据",
                "- 不要将数据行误判为表头",
                "- 表头可能有多行（多层表头）",
                "- 表头行必须是连续的，从第一行开始",
                "\n表格数据（前{}行）:".format(len(rows_to_analyze))
            ]
            
            # 添加每一行的数据
            for i, row in enumerate(rows_to_analyze):
                # 过滤None值并转换为字符串
                row_str = [str(cell) if pd.notna(cell) else "" for cell in row]
                # 过滤空字符串
                non_empty = [cell for cell in row_str if cell.strip()]
                
                if non_empty:
                    row_display = " | ".join(non_empty[:10])  # 最多显示10列
                    if len(non_empty) > 10:
                        row_display += " ..."
                    prompt_parts.append(f"第{i}行: {row_display}")
                else:
                    prompt_parts.append(f"第{i}行: [空行]")
            
            prompt_parts.extend([
                "\n请分析并以JSON格式回答，包含以下字段:",
                "- header_row_count: 表头行数（整数，0表示没有表头）",
                "- header_rows_indices: 表头行的索引列表（如 [0, 1] 表示第0行和第1行是表头）",
                "- data_start_row: 数据开始的行索引（整数）",
                "- confidence: 检测置信度（0.0-1.0之间的浮点数）",
                "- reasoning: 你的判断理由（简短说明为什么这样判断）",
                "\n示例输出:",
                "{",
                '  "header_row_count": 2,',
                '  "header_rows_indices": [0, 1],',
                '  "data_start_row": 2,',
                '  "confidence": 0.9,',
                '  "reasoning": "第0-1行包含字段名称，第2行开始是具体数据"',
                "}"
            ])
            
            prompt = "\n".join(prompt_parts)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # 解析JSON响应
            try:
                # 提取JSON部分
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                result = json.loads(result_text)
                
                # 验证结果的有效性
                if not isinstance(result.get("header_row_count"), int):
                    raise ValueError("header_row_count must be an integer")
                if not isinstance(result.get("header_rows_indices"), list):
                    raise ValueError("header_rows_indices must be a list")
                if not isinstance(result.get("data_start_row"), int):
                    raise ValueError("data_start_row must be an integer")
                
                logger.info(f"HeaderDetectionAgent: Detected {result['header_row_count']} header rows with confidence {result.get('confidence', 0)}")
                logger.info(f"HeaderDetectionAgent: {result.get('reasoning', '')}")
                
                return result
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response: {str(e)}, falling back to rule-based")
                return self._rule_based_detect_headers(rows_data, max_rows)
                
        except Exception as e:
            logger.error(f"LLM header detection failed: {str(e)}, falling back to rule-based")
            return self._rule_based_detect_headers(rows_data, max_rows)
    
    def _rule_based_detect_headers(self, rows_data: List[List[Any]], max_rows: int) -> Dict[str, Any]:
        """基于规则的表头检测（作为后备方案）"""
        logger.info("Using rule-based header detection")
        
        header_rows_indices = []
        header_row_count = 0
        
        # 限制行数
        rows_to_check = min(len(rows_data), max_rows)
        
        # 从第一行开始检查
        for i in range(rows_to_check):
            row = rows_data[i]
            
            # 过滤空值
            non_empty = [v for v in row if pd.notna(v) and str(v).strip()]
            
            # 空行不是表头
            if len(non_empty) == 0:
                continue
            
            # 检查是否大部分是文本且不是数字
            text_count = 0
            for v in non_empty:
                try:
                    # 如果能转换为数字，可能不是表头
                    float(str(v).replace('%', '').replace(',', '').replace('，', ''))
                except (ValueError, AttributeError):
                    text_count += 1
            
            # 如果至少50%是文本，可能是表头
            text_ratio = text_count / len(non_empty) if non_empty else 0
            
            # 如果这行看起来像表头，且之前的行也是表头（或这是第一行）
            if text_ratio >= 0.5 and (i == 0 or i == len(header_rows_indices)):
                header_rows_indices.append(i)
                header_row_count += 1
            else:
                # 遇到数据行，停止
                break
        
        data_start_row = header_row_count
        confidence = 0.6  # 规则基础方法的置信度较低
        
        reasoning = f"基于规则检测: 前{header_row_count}行包含较多文本字段，判断为表头"
        
        return {
            "header_row_count": header_row_count,
            "header_rows_indices": header_rows_indices,
            "data_start_row": data_start_row,
            "confidence": confidence,
            "reasoning": reasoning
        }

