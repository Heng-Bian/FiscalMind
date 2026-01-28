"""
语义解析器模块 - 提供基于LLM的语义匹配功能
Semantic Resolver module - Provides LLM-based semantic matching capabilities.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
import logging
import ast

logger = logging.getLogger(__name__)


class SemanticResolver:
    """
    语义解析器 - 提供基于LLM的语义匹配功能
    
    使用大语言模型进行智能语义匹配，上下文包含表头、样本数据和文档描述。
    Semantic resolver using LLM for intelligent semantic matching with context including headers, sample data, and descriptions.
    """
    
    # 常见的地名标准化映射（保留用于值标准化）
    LOCATION_NORMALIZATIONS = {
        '北京': ['北京', '北京市', 'Beijing', 'beijing'],
        '上海': ['上海', '上海市', 'Shanghai', 'shanghai'],
        '广州': ['广州', '广州市', 'Guangzhou', 'guangzhou'],
        '深圳': ['深圳', '深圳市', 'Shenzhen', 'shenzhen'],
        '华东': ['华东', '华东地区', 'East China'],
        '华南': ['华南', '华南地区', 'South China'],
        '华北': ['华北', '华北地区', 'North China'],
        '华中': ['华中', '华中地区', 'Central China'],
    }
    
    def __init__(self, llm=None, sample_rows: int = 5):
        """
        初始化语义解析器
        
        Args:
            llm: 语言模型实例（必需，用于语义匹配）
            sample_rows: 用于上下文的样本数据行数（默认5行）
        """
        self.llm = llm
        self.sample_rows = sample_rows
        
    def _build_dataframe_context(self, df: pd.DataFrame, description: str = None) -> str:
        """
        构建DataFrame的上下文信息，包括表头、样本数据和描述
        
        Args:
            df: DataFrame对象
            description: 可选的表格描述
            
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 添加描述（如果有）
        if description:
            context_parts.append(f"表格描述: {description}")
        
        # 添加表头信息
        columns = list(df.columns)
        context_parts.append(f"列名: {columns}")
        
        # 添加数据类型信息
        dtypes_info = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        context_parts.append(f"数据类型: {dtypes_info}")
        
        # 添加样本数据
        sample_data = df.head(self.sample_rows).to_dict('records')
        context_parts.append(f"样本数据 (前{self.sample_rows}行):")
        for i, row in enumerate(sample_data, 1):
            context_parts.append(f"  行{i}: {row}")
        
        return "\n".join(context_parts)
        
    def find_column_by_semantic(self, df: pd.DataFrame, concept: str, 
                               description: str = None,
                               use_llm_fallback: bool = True) -> List[str]:
        """
        通过语义查找列名（基于LLM）
        
        Args:
            df: DataFrame对象
            concept: 概念/语义描述（业务需求）
            description: 可选的表格描述
            use_llm_fallback: 保留用于兼容性，实际始终使用LLM
            
        Returns:
            匹配的列名列表（按相关度排序）
        """
        if not self.llm:
            logger.warning("No LLM provided, falling back to exact match only")
            # 如果没有LLM，只进行精确匹配
            concept_lower = concept.lower().strip()
            for col in df.columns:
                col_str = str(col).strip()
                if col_str == concept or col_str.lower() == concept_lower:
                    logger.info(f"Exact match found: '{col}' for concept '{concept}'")
                    return [col]
            return []
        
        # 使用LLM进行语义匹配
        return self._find_column_by_llm(df, concept, description)
    
    def _find_column_by_llm(self, df: pd.DataFrame, concept: str, description: str = None) -> List[str]:
        """使用LLM查找列（基于完整上下文）"""
        try:
            from langchain_core.messages import HumanMessage
            
            # 构建上下文
            context = self._build_dataframe_context(df, description)
            
            prompt = f"""你是一个专业的数据分析助手。给定一个数据表的上下文信息，请根据业务描述匹配最相关的列名。

数据表上下文:
{context}

业务描述: "{concept}"

请分析上下文中的列名、数据类型和样本数据，找出与业务描述最相关的列。
考虑同义词、业务含义和数据内容的匹配。

返回格式要求:
1. 返回一个Python列表，包含最相关的列名
2. 按相关度从高到低排序
3. 如果没有相关列，返回空列表 []
4. 只返回列表，不要任何解释

示例: ['列名1', '列名2']"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # 解析响应
            try:
                # 尝试提取列表
                result = ast.literal_eval(result_text)
                if isinstance(result, list):
                    # 验证返回的列名存在于DataFrame中
                    columns_list = list(df.columns)
                    valid_cols = [col for col in result if col in columns_list]
                    if valid_cols:
                        logger.info(f"LLM matched columns for '{concept}': {valid_cols}")
                        return valid_cols
                    else:
                        logger.warning(f"LLM returned columns not in DataFrame: {result}")
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse LLM response: {result_text}, error: {e}")
                
        except Exception as e:
            logger.error(f"LLM column matching failed: {str(e)}")
        
        return []
    
    def find_sheet_by_semantic(self, sheet_names: List[str], query: str,
                              sheets_context: Dict[str, Any] = None,
                              use_llm_fallback: bool = True) -> Optional[str]:
        """
        通过语义查找工作表名称（基于LLM）
        
        Args:
            sheet_names: 工作表名称列表
            query: 用户查询（如"24年的预算", "2024预算表"）
            sheets_context: 可选的工作表上下文信息字典 {sheet_name: description}
            use_llm_fallback: 保留用于兼容性，实际始终使用LLM
            
        Returns:
            最匹配的工作表名称
        """
        if not sheet_names:
            return None
        
        if not self.llm:
            logger.warning("No LLM provided, falling back to exact match only")
            # 如果没有LLM，只进行精确匹配
            query_lower = query.lower().strip()
            for sheet in sheet_names:
                if sheet == query or sheet.lower() == query_lower:
                    logger.info(f"Exact sheet match: '{sheet}'")
                    return sheet
            return None
            
        # 使用LLM进行语义匹配
        return self._find_sheet_by_llm(sheet_names, query, sheets_context)
    
    def _find_sheet_by_llm(self, sheet_names: List[str], query: str, 
                          sheets_context: Dict[str, Any] = None) -> Optional[str]:
        """使用LLM查找工作表（基于完整上下文）"""
        try:
            from langchain_core.messages import HumanMessage
            
            # 构建上下文
            context_parts = ["可用的工作表:"]
            for sheet in sheet_names:
                context_parts.append(f"  - {sheet}")
                if sheets_context and sheet in sheets_context:
                    context_parts.append(f"    描述: {sheets_context[sheet]}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""你是一个专业的数据分析助手。给定一组工作表名称和用户查询，请找出最匹配的工作表。

{context}

用户查询: "{query}"

请分析用户查询的意图，考虑：
1. 关键词匹配（如"预算"、"24年"等）
2. 年份识别（如"24"可能指"2024"或"FY24"）
3. 业务含义（如"工资"可能匹配"薪资"）
4. 工作表描述（如果提供）

返回格式要求:
1. 只返回最匹配的一个工作表名称
2. 必须从上述工作表列表中选择
3. 如果没有匹配的，返回 null
4. 只返回工作表名称，不要任何解释

示例: 2024年财务报表"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            
            # 移除可能的引号
            result = result.strip('"\'')
            
            # 验证结果
            if result and result != 'null' and result in sheet_names:
                logger.info(f"LLM matched sheet '{result}' for query '{query}'")
                return result
            elif result and result not in sheet_names:
                logger.warning(f"LLM returned invalid sheet name: {result}")
                
        except Exception as e:
            logger.error(f"LLM sheet matching failed: {str(e)}")
        
        return None
    
    def auto_discover_join_keys(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               description1: str = None, description2: str = None,
                               use_llm_fallback: bool = True) -> Optional[Tuple[str, str]]:
        """
        自动发现两个表的关联键（基于LLM）
        
        Args:
            df1: 第一个DataFrame
            df2: 第二个DataFrame
            description1: 第一个表的描述
            description2: 第二个表的描述
            use_llm_fallback: 保留用于兼容性，实际始终使用LLM
            
        Returns:
            (left_on, right_on) 元组，如果未找到则返回None
        """
        # 首先尝试精确匹配（即使有LLM也优先使用这个简单的逻辑）
        common_cols = set(df1.columns) & set(df2.columns)
        if common_cols:
            # 优先选择包含"ID"、"编号"等关键词的列
            for col in common_cols:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in ['id', '编号', 'code', 'key']):
                    logger.info(f"Found exact join key: '{col}'")
                    return (col, col)
            # 如果没有ID类的，返回第一个公共列
            col = list(common_cols)[0]
            logger.info(f"Found common join key: '{col}'")
            return (col, col)
        
        # 如果没有公共列，使用LLM
        if not self.llm:
            logger.warning("No LLM provided and no common columns found")
            return None
        
        # 使用LLM进行关联键发现
        return self._discover_join_keys_by_llm(df1, df2, description1, description2)
    
    def _discover_join_keys_by_llm(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                   description1: str = None, description2: str = None) -> Optional[Tuple[str, str]]:
        """使用LLM发现关联键（基于完整上下文）"""
        try:
            from langchain_core.messages import HumanMessage
            
            # 构建两个表的上下文
            context1 = self._build_dataframe_context(df1, description1)
            context2 = self._build_dataframe_context(df2, description2)
            
            prompt = f"""你是一个专业的数据分析助手。给定两个数据表的上下文信息，请找出最适合用于关联（JOIN）这两个表的键。

表1上下文:
{context1}

表2上下文:
{context2}

请分析两个表的：
1. 列名及其含义
2. 数据类型是否兼容
3. 样本数据的内容和值域
4. 可能的业务关联关系（如员工编号、产品ID等）

返回格式要求:
1. 返回一个Python元组: ('表1的列名', '表2的列名')
2. 列名必须分别存在于对应的表中
3. 如果找不到合适的关联键，返回 null
4. 只返回元组或null，不要任何解释

示例: ('员工编号', '工号')"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # 解析响应
            try:
                if result_text == 'null':
                    return None
                    
                result = ast.literal_eval(result_text)
                if isinstance(result, tuple) and len(result) == 2:
                    left_key, right_key = result
                    # 验证列名存在
                    if left_key in df1.columns and right_key in df2.columns:
                        logger.info(f"LLM found join keys: {result}")
                        return (left_key, right_key)
                    else:
                        logger.warning(f"LLM returned invalid column names: {result}")
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse LLM response: {result_text}, error: {e}")
                
        except Exception as e:
            logger.error(f"LLM join key discovery failed: {str(e)}")
        
        return None
    
    def normalize_value(self, value: str, category: str = 'location') -> str:
        """
        标准化分类值（用于模糊匹配）
        
        Args:
            value: 原始值
            category: 值的类别（'location'等）
            
        Returns:
            标准化后的值
        """
        if pd.isna(value):
            return value
            
        value_str = str(value).strip()
        value_lower = value_str.lower()
        
        if category == 'location':
            # 地名标准化
            for standard, variants in self.LOCATION_NORMALIZATIONS.items():
                if value_str in variants or value_lower in [v.lower() for v in variants]:
                    return standard
        
        return value_str
    
    def find_document_by_semantic(self, document_names: List[str], query: str,
                                 documents_context: Dict[str, Any] = None,
                                 use_llm_fallback: bool = True) -> Optional[str]:
        """
        通过语义查找文档名称（基于LLM）
        
        Args:
            document_names: 文档名称列表
            query: 查询（如"报销单"）
            documents_context: 可选的文档上下文信息字典 {doc_name: description}
            use_llm_fallback: 保留用于兼容性，实际始终使用LLM
            
        Returns:
            最匹配的文档名称
        """
        # 文档名匹配与工作表匹配逻辑相似
        return self.find_sheet_by_semantic(document_names, query, documents_context, use_llm_fallback)
