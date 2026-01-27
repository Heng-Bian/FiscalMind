"""
语义解析器模块 - 提供基于LLM的语义匹配功能
Semantic Resolver module - Provides LLM-based semantic matching capabilities.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from difflib import SequenceMatcher
import logging
import os

logger = logging.getLogger(__name__)


class SemanticResolver:
    """
    语义解析器 - 提供高可信度的语义匹配功能
    
    使用静态同义词库作为基础，LLM作为保底机制，确保高可信度的匹配结果。
    Semantic resolver using static synonym dictionaries as base, with LLM as fallback for high-confidence matching.
    """
    
    # 财务/业务领域同义词映射
    SYNONYM_MAP = {
        '收入': ['收入', '营收', '销售额', '营业收入', '收益', '进账', 'revenue', 'income'],
        '利润': ['利润', '盈利', '净利润', '毛利', '利益', '赚钱', 'profit', 'earnings'],
        '成本': ['成本', '费用', '开支', '支出', '花费', 'cost', 'expense'],
        '销售': ['销售', '售出', '出售', '营业', 'sales'],
        '日期': ['日期', '时间', '年月', '期间', 'date', 'time'],
        '数量': ['数量', '件数', '个数', '总数', 'quantity', 'amount', 'count'],
        '价格': ['价格', '单价', '金额', '售价', 'price'],
        '部门': ['部门', '科室', '组织', '单位', 'department', 'division'],
        '员工': ['员工', '人员', '职工', '工作人员', 'employee', 'staff'],
        '工资': ['工资', '薪资', '薪酬', '报酬', '薪水', '月薪', '年薪', '薪', 'salary', 'wage', 'pay'],
        '编号': ['编号', 'ID', 'id', '工号', '员工编号', '代码', 'code', 'number'],
        '名称': ['名称', '名字', '姓名', 'name'],
        '地址': ['地址', '地点', '位置', 'address', 'location'],
        '区域': ['区域', '地区', '区', 'region', 'area'],
        '城市': ['城市', '市', 'city'],
        '省份': ['省份', '省', 'province', 'state'],
        '国家': ['国家', '国', 'country'],
        '产品': ['产品', '商品', '货物', 'product', 'goods'],
        '类别': ['类别', '类型', '种类', '分类', 'category', 'type'],
        '状态': ['状态', '情况', 'status', 'state'],
        '报销': ['报销', '报销单', '费用报销', 'reimbursement', 'expense claim'],
        '报表': ['报表', '报告', '财务报表', 'report', 'financial report', 'statement'],
        '预算': ['预算', '预算表', 'budget'],
    }
    
    # 常见的地名标准化映射
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
    
    def __init__(self, llm=None, confidence_threshold: float = 0.7):
        """
        初始化语义解析器
        
        Args:
            llm: 语言模型实例（可选，用于高级语义匹配）
            confidence_threshold: 置信度阈值，低于此值时使用LLM
        """
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        
    def find_column_by_semantic(self, df: pd.DataFrame, concept: str, 
                               use_llm_fallback: bool = True) -> List[str]:
        """
        通过语义查找列名
        
        Args:
            df: DataFrame对象
            concept: 概念/语义描述
            use_llm_fallback: 是否使用LLM作为保底（当静态匹配置信度低时）
            
        Returns:
            匹配的列名列表（按置信度排序）
        """
        concept_lower = concept.lower().strip()
        matching_cols = []
        
        # Step 1: 精确匹配（最高优先级）
        for col in df.columns:
            col_str = str(col).strip()
            if col_str == concept or col_str.lower() == concept_lower:
                logger.info(f"Exact match found: '{col}' for concept '{concept}'")
                return [col]
        
        # Step 2: 直接包含匹配
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if concept_lower in col_lower or col_lower in concept_lower:
                matching_cols.append((col, 0.9))  # High confidence
        
        # Step 3: 同义词匹配
        synonym_matches = self._find_by_synonyms(df.columns, concept_lower)
        for col, score in synonym_matches:
            if col not in [c for c, _ in matching_cols]:
                matching_cols.append((col, score))
        
        # Step 4: 模糊匹配（使用SequenceMatcher）
        fuzzy_matches = self._find_by_fuzzy_match(df.columns, concept_lower)
        for col, score in fuzzy_matches:
            if col not in [c for c, _ in matching_cols] and score >= 0.6:
                matching_cols.append((col, score))
        
        # Sort by confidence score
        matching_cols.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: 如果置信度不够，使用LLM
        if use_llm_fallback and self.llm and (
            not matching_cols or matching_cols[0][1] < self.confidence_threshold
        ):
            logger.info(f"Low confidence matches, using LLM fallback for concept '{concept}'")
            llm_result = self._find_by_llm(df, concept)
            if llm_result:
                return llm_result
        
        # Return column names only (without scores)
        return [col for col, _ in matching_cols]
    
    def find_sheet_by_semantic(self, sheet_names: List[str], query: str,
                              use_llm_fallback: bool = True) -> Optional[str]:
        """
        通过语义查找工作表名称
        
        Args:
            sheet_names: 工作表名称列表
            query: 用户查询（如"24年的预算", "2024预算表"）
            use_llm_fallback: 是否使用LLM作为保底
            
        Returns:
            最匹配的工作表名称
        """
        if not sheet_names:
            return None
            
        query_lower = query.lower().strip()
        
        # Step 1: 精确匹配
        for sheet in sheet_names:
            if sheet == query or sheet.lower() == query_lower:
                logger.info(f"Exact sheet match: '{sheet}'")
                return sheet
        
        # Step 2: 包含匹配
        candidates = []
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            if query_lower in sheet_lower or sheet_lower in query_lower:
                score = len(query_lower) / max(len(sheet_lower), len(query_lower))
                candidates.append((sheet, score))
        
        # Step 3: 模糊匹配
        if not candidates:
            for sheet in sheet_names:
                ratio = SequenceMatcher(None, query_lower, sheet.lower()).ratio()
                if ratio >= 0.5:  # Lowered threshold
                    candidates.append((sheet, ratio))
        
        # Step 4: 关键词匹配（财务场景）+ 同义词匹配
        if not candidates:
            keywords = self._extract_keywords(query_lower)
            for sheet in sheet_names:
                sheet_lower = sheet.lower()
                match_count = 0
                for kw in keywords:
                    # Check direct match or if keyword appears in sheet
                    if kw in sheet_lower:
                        match_count += 1
                    # Also check for partial year matches (e.g., '24' in '2024' or 'FY24')
                    elif kw.isdigit() and len(kw) == 2:
                        # Check if this 2-digit year appears in a 4-digit year
                        full_year = '20' + kw
                        if full_year in sheet_lower:
                            match_count += 1
                        # Also check for patterns like 'FY24', 'fy24'
                        if 'fy' + kw in sheet_lower or 'FY' + kw in sheet:
                            match_count += 1
                    else:
                        # Check if keyword is a synonym of something in sheet
                        if self._check_synonym_in_text(kw, sheet_lower):
                            match_count += 1
                
                if match_count > 0:
                    score = match_count / len(keywords) if keywords else 0
                    candidates.append((sheet, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: LLM fallback
        if use_llm_fallback and self.llm and (
            not candidates or candidates[0][1] < self.confidence_threshold
        ):
            logger.info(f"Low confidence, using LLM for sheet matching: '{query}'")
            llm_result = self._find_sheet_by_llm(sheet_names, query)
            if llm_result:
                return llm_result
        
        return candidates[0][0] if candidates else None
    
    def auto_discover_join_keys(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               use_llm_fallback: bool = True) -> Optional[Tuple[str, str]]:
        """
        自动发现两个表的关联键
        
        Args:
            df1: 第一个DataFrame
            df2: 第二个DataFrame
            use_llm_fallback: 是否使用LLM作为保底
            
        Returns:
            (left_on, right_on) 元组，如果未找到则返回None
        """
        candidates = []
        
        # Step 1: 列名完全相同
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
            return (col, col)
        
        # Step 2: 同义词匹配（如"员工编号" vs "工号"）
        for col1 in df1.columns:
            for col2 in df2.columns:
                # Check if they are synonyms
                if self._are_synonyms(str(col1), str(col2)):
                    # Check data types match
                    if df1[col1].dtype == df2[col2].dtype or self._compatible_types(df1[col1].dtype, df2[col2].dtype):
                        score = 0.8
                        candidates.append((col1, col2, score))
        
        # Step 3: 列名相似度匹配
        for col1 in df1.columns:
            for col2 in df2.columns:
                ratio = SequenceMatcher(None, str(col1).lower(), str(col2).lower()).ratio()
                if ratio >= 0.7:
                    # Check data types
                    if df1[col1].dtype == df2[col2].dtype or self._compatible_types(df1[col1].dtype, df2[col2].dtype):
                        candidates.append((col1, col2, ratio))
        
        # Step 4: 数据预览匹配（检查值的重叠）
        if not candidates:
            for col1 in df1.columns:
                for col2 in df2.columns:
                    if self._compatible_types(df1[col1].dtype, df2[col2].dtype):
                        overlap = self._calculate_value_overlap(df1[col1], df2[col2])
                        if overlap >= 0.3:  # 至少30%的值重叠
                            candidates.append((col1, col2, overlap * 0.7))  # Lower confidence
        
        # Sort by score
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Step 5: LLM fallback
        if use_llm_fallback and self.llm and (
            not candidates or candidates[0][2] < self.confidence_threshold
        ):
            logger.info("Using LLM to discover join keys")
            llm_result = self._discover_join_keys_by_llm(df1, df2)
            if llm_result:
                return llm_result
        
        return (candidates[0][0], candidates[0][1]) if candidates else None
    
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
                                 use_llm_fallback: bool = True) -> Optional[str]:
        """
        通过语义查找文档名称
        
        Args:
            document_names: 文档名称列表
            query: 查询（如"报销单"）
            use_llm_fallback: 是否使用LLM作为保底
            
        Returns:
            最匹配的文档名称
        """
        # This is similar to find_sheet_by_semantic
        return self.find_sheet_by_semantic(document_names, query, use_llm_fallback)
    
    # ========== Private Helper Methods ==========
    
    def _find_by_synonyms(self, columns: pd.Index, concept: str) -> List[Tuple[str, float]]:
        """使用同义词映射查找列"""
        matches = []
        
        for key, synonyms in self.SYNONYM_MAP.items():
            if concept in [s.lower() for s in synonyms]:
                # This concept is a synonym, find columns matching any synonym
                for col in columns:
                    col_lower = str(col).lower()
                    for syn in synonyms:
                        if syn.lower() in col_lower:
                            matches.append((col, 0.85))
                            break
        
        return matches
    
    def _find_by_fuzzy_match(self, columns: pd.Index, concept: str) -> List[Tuple[str, float]]:
        """使用模糊匹配查找列"""
        matches = []
        
        for col in columns:
            col_lower = str(col).lower()
            ratio = SequenceMatcher(None, concept, col_lower).ratio()
            if ratio >= 0.6:
                matches.append((col, ratio))
        
        return matches
    
    def _find_by_llm(self, df: pd.DataFrame, concept: str) -> Optional[List[str]]:
        """使用LLM查找列（保底机制）"""
        if not self.llm:
            return None
            
        try:
            from langchain_core.messages import HumanMessage
            
            columns_list = list(df.columns)
            prompt = f"""Given a DataFrame with the following columns:
{columns_list}

The user is looking for columns related to the concept: "{concept}"

Please return ONLY the most relevant column names as a Python list (e.g., ['column1', 'column2']).
If no relevant columns are found, return an empty list [].
Return ONLY the list, no explanation."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # Parse the response
            import ast
            try:
                result = ast.literal_eval(result_text)
                if isinstance(result, list):
                    # Validate that returned columns exist
                    valid_cols = [col for col in result if col in columns_list]
                    if valid_cols:
                        logger.info(f"LLM found columns: {valid_cols}")
                        return valid_cols
            except:
                logger.warning(f"Failed to parse LLM response: {result_text}")
                
        except Exception as e:
            logger.error(f"LLM column matching failed: {str(e)}")
        
        return None
    
    def _find_sheet_by_llm(self, sheet_names: List[str], query: str) -> Optional[str]:
        """使用LLM查找工作表"""
        if not self.llm:
            return None
            
        try:
            from langchain_core.messages import HumanMessage
            
            prompt = f"""Given the following sheet names:
{sheet_names}

The user query is: "{query}"

Please return ONLY the most relevant sheet name from the list above.
Return ONLY the sheet name, no explanation."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            
            # Validate
            if result in sheet_names:
                logger.info(f"LLM matched sheet: {result}")
                return result
                
        except Exception as e:
            logger.error(f"LLM sheet matching failed: {str(e)}")
        
        return None
    
    def _discover_join_keys_by_llm(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """使用LLM发现关联键"""
        if not self.llm:
            return None
            
        try:
            from langchain_core.messages import HumanMessage
            
            cols1 = list(df1.columns)
            cols2 = list(df2.columns)
            
            # Sample data preview
            preview1 = df1.head(3).to_dict('records')
            preview2 = df2.head(3).to_dict('records')
            
            prompt = f"""Given two DataFrames:

DataFrame 1 columns: {cols1}
Sample data: {preview1}

DataFrame 2 columns: {cols2}
Sample data: {preview2}

Please identify the best join keys for these two tables.
Return ONLY a tuple in the format: ('left_key', 'right_key')
For example: ('employee_id', 'emp_id')

Return ONLY the tuple, no explanation."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # Parse the response
            import ast
            try:
                result = ast.literal_eval(result_text)
                if isinstance(result, tuple) and len(result) == 2:
                    left_key, right_key = result
                    # Validate
                    if left_key in cols1 and right_key in cols2:
                        logger.info(f"LLM found join keys: {result}")
                        return (left_key, right_key)
            except:
                logger.warning(f"Failed to parse LLM response: {result_text}")
                
        except Exception as e:
            logger.error(f"LLM join key discovery failed: {str(e)}")
        
        return None
    
    def _are_synonyms(self, term1: str, term2: str) -> bool:
        """检查两个术语是否是同义词"""
        term1_lower = term1.lower().strip()
        term2_lower = term2.lower().strip()
        
        for key, synonyms in self.SYNONYM_MAP.items():
            synonyms_lower = [s.lower() for s in synonyms]
            if term1_lower in synonyms_lower and term2_lower in synonyms_lower:
                return True
        
        return False
    
    def _check_synonym_in_text(self, keyword: str, text: str) -> bool:
        """检查关键词的同义词是否出现在文本中"""
        keyword_lower = keyword.lower().strip()
        text_lower = text.lower().strip()
        
        # Check if the keyword itself is in any synonym group
        for key, synonyms in self.SYNONYM_MAP.items():
            synonyms_lower = [s.lower() for s in synonyms]
            if keyword_lower in synonyms_lower:
                # Check if any synonym appears in the text
                for syn in synonyms_lower:
                    if syn in text_lower:
                        return True
        
        return False
    
    def _compatible_types(self, dtype1, dtype2) -> bool:
        """检查两个数据类型是否兼容（可以关联）"""
        # Numeric types are compatible
        numeric_types = ['int64', 'int32', 'float64', 'float32']
        if str(dtype1) in numeric_types and str(dtype2) in numeric_types:
            return True
        
        # String/object types are compatible
        string_types = ['object', 'string']
        if str(dtype1) in string_types and str(dtype2) in string_types:
            return True
        
        return dtype1 == dtype2
    
    def _calculate_value_overlap(self, series1: pd.Series, series2: pd.Series, sample_size: int = 100) -> float:
        """计算两个Series值的重叠度"""
        # Sample for performance
        s1_values = set(series1.dropna().head(sample_size).astype(str))
        s2_values = set(series2.dropna().head(sample_size).astype(str))
        
        if not s1_values or not s2_values:
            return 0.0
        
        intersection = s1_values & s2_values
        union = s1_values | s2_values
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        import re
        
        # Remove common words
        common_words = {'的', '是', '在', '和', '与', '了', '这', '那', '有', '一', '个', '年',
                       'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'on', 'at'}
        
        keywords = []
        
        # Extract numbers (including years like '24', '2024')
        numbers = re.findall(r'\d+', text)
        keywords.extend(numbers)
        
        # Extract Chinese characters (split by common words)
        # First remove common words
        text_cleaned = text
        for word in ['的', '是', '在', '和', '与', '了', '这', '那', '年']:
            text_cleaned = text_cleaned.replace(word, ' ')
        
        # Extract Chinese word sequences
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', text_cleaned)
        keywords.extend(chinese_words)
        
        # For Chinese words, also try to extract known domain terms
        # This helps match things like '员工工资' to '员工' and '工资'
        for chinese_word in chinese_words:
            # Check if this word contains known keywords from our synonym map
            for key, synonyms in self.SYNONYM_MAP.items():
                for syn in synonyms:
                    if syn in chinese_word and syn != chinese_word and len(syn) >= 2:
                        keywords.append(syn)
        
        # Extract English words
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        keywords.extend(english_words)
        
        # Filter out empty strings and duplicates, but keep order
        seen = set()
        filtered_keywords = []
        for kw in keywords:
            kw = kw.strip()
            if kw and kw not in seen and kw not in common_words:
                seen.add(kw)
                filtered_keywords.append(kw)
        
        return filtered_keywords
