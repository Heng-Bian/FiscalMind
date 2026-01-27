"""
元功能模块 - 提供表格数据与LLM交互的工具函数
Meta-functions module for table data and LLM interaction.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import logging
from fiscal_mind.parser import ExcelDocument, ExcelParser

logger = logging.getLogger(__name__)


class TableMetaFunctions:
    """表格元功能类，提供各种表格操作和分析功能"""
    
    @staticmethod
    def get_table_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取表格结构信息（适合传给LLM）
        
        Args:
            df: DataFrame对象
            
        Returns:
            表格结构描述
        """
        schema = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "sample_data": df.head(3).to_dict(orient="records")
        }
        return schema
    
    @staticmethod
    def get_column_statistics(df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """
        获取列的统计信息
        
        Args:
            df: DataFrame对象
            column_name: 列名
            
        Returns:
            列的统计信息
        """
        if column_name not in df.columns:
            return {"error": f"Column '{column_name}' not found"}
        
        col = df[column_name]
        stats = {
            "column_name": column_name,
            "dtype": str(col.dtype),
            "count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "unique_count": int(col.nunique())
        }
        
        # 数值列的统计
        if pd.api.types.is_numeric_dtype(col):
            stats.update({
                "mean": float(col.mean()) if not col.empty else None,
                "median": float(col.median()) if not col.empty else None,
                "min": float(col.min()) if not col.empty else None,
                "max": float(col.max()) if not col.empty else None,
                "std": float(col.std()) if not col.empty else None
            })
        
        # 分类列的统计
        if col.nunique() < 20:  # 如果唯一值少于20个，显示值分布
            stats["value_counts"] = col.value_counts().head(10).to_dict()
        
        return stats
    
    @staticmethod
    def get_numeric_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取所有数值列的汇总统计
        
        Args:
            df: DataFrame对象
            
        Returns:
            数值列的统计摘要
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return {"message": "No numeric columns found"}
        
        summary = {}
        for col in numeric_cols:
            summary[col] = TableMetaFunctions.get_column_statistics(df, col)
        
        return summary
    
    @staticmethod
    def format_for_llm_context(df: pd.DataFrame, max_rows: int = 10) -> str:
        """
        将DataFrame格式化为适合LLM上下文的文本
        
        Args:
            df: DataFrame对象
            max_rows: 最大行数
            
        Returns:
            格式化的文本
        """
        lines = []
        lines.append(f"表格形状: {len(df)} 行 x {len(df.columns)} 列")
        lines.append(f"列名: {', '.join(df.columns.tolist())}")
        lines.append("\n数据预览:")
        lines.append(df.head(max_rows).to_string())
        
        # 添加数值列的统计信息
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            lines.append("\n数值列统计:")
            lines.append(df[numeric_cols].describe().to_string())
        
        return "\n".join(lines)
    
    @staticmethod
    def extract_key_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        提取表格的关键信息（适合传给LLM做决策）
        
        Args:
            df: DataFrame对象
            
        Returns:
            关键信息字典
        """
        key_info = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes_summary": df.dtypes.value_counts().to_dict(),
            "missing_data": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "text_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        return key_info
    
    @staticmethod
    def summarize_document_for_llm(doc: ExcelDocument) -> str:
        """
        将Excel文档摘要格式化为LLM可读的文本
        
        Args:
            doc: ExcelDocument对象
            
        Returns:
            格式化的摘要文本
        """
        lines = []
        lines.append(f"文档名称: {doc.file_name}")
        lines.append(f"工作表数量: {len(doc.sheets)}")
        lines.append(f"工作表名称: {', '.join(doc.get_sheet_names())}")
        lines.append("")
        
        for sheet_name in doc.get_sheet_names():
            df = doc.get_sheet(sheet_name)
            lines.append(f"工作表: {sheet_name}")
            lines.append(f"  - 形状: {len(df)} 行 x {len(df.columns)} 列")
            lines.append(f"  - 列名: {', '.join(df.columns.tolist())}")
            
            # 显示数值列
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                lines.append(f"  - 数值列: {', '.join(numeric_cols)}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def create_data_context(parser: ExcelParser, max_rows_per_sheet: int = 5) -> str:
        """
        创建所有已加载文档的数据上下文（用于LLM）
        
        Args:
            parser: ExcelParser对象
            max_rows_per_sheet: 每个工作表的最大行数
            
        Returns:
            格式化的数据上下文
        """
        lines = []
        lines.append(f"已加载 {len(parser.documents)} 个Excel文档:")
        lines.append("")
        
        for doc_name, doc in parser.documents.items():
            lines.append(TableMetaFunctions.summarize_document_for_llm(doc))
            lines.append("="*50)
            lines.append("")
        
        return "\n".join(lines)


class TableQueryHelper:
    """表格查询辅助类，提供常见的查询操作"""
    
    @staticmethod
    def find_column_by_keyword(df: pd.DataFrame, keyword: str) -> List[str]:
        """
        根据关键词查找列名
        
        Args:
            df: DataFrame对象
            keyword: 关键词
            
        Returns:
            匹配的列名列表
        """
        keyword_lower = keyword.lower()
        matching_cols = [
            col for col in df.columns 
            if keyword_lower in str(col).lower()
        ]
        return matching_cols
    
    @staticmethod
    def find_column_by_semantic(df: pd.DataFrame, concept: str) -> List[str]:
        """
        通过语义关联查找列名（使用同义词映射）
        
        Args:
            df: DataFrame对象
            concept: 概念/语义描述
            
        Returns:
            匹配的列名列表
            
        Note:
            这是一个简化的语义搜索实现，使用预定义的同义词映射。
            在实际应用中，可以集成词向量或LLM进行更智能的匹配。
        """
        # 预定义的财务/业务领域同义词映射
        synonym_map = {
            '收入': ['收入', '营收', '销售额', '营业收入', '收益', '进账'],
            '利润': ['利润', '盈利', '净利润', '毛利', '利益', '赚钱'],
            '成本': ['成本', '费用', '开支', '支出', '花费'],
            '销售': ['销售', '售出', '出售', '营业'],
            '日期': ['日期', '时间', '年月', '期间'],
            '数量': ['数量', '件数', '个数', '总数'],
            '价格': ['价格', '单价', '金额', '售价'],
            '部门': ['部门', '科室', '组织', '单位'],
            '员工': ['员工', '人员', '职工', '工作人员'],
            '工资': ['工资', '薪资', '薪酬', '报酬', '薪水'],
        }
        
        concept_lower = concept.lower()
        matching_cols = []
        
        # 直接匹配
        for col in df.columns:
            col_lower = str(col).lower()
            if concept_lower in col_lower:
                matching_cols.append(col)
        
        # 同义词匹配
        for key, synonyms in synonym_map.items():
            if concept_lower in [s.lower() for s in synonyms]:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(syn.lower() in col_lower for syn in synonyms):
                        if col not in matching_cols:
                            matching_cols.append(col)
        
        return matching_cols
    
    @staticmethod
    def auto_detect_groupable_columns(df: pd.DataFrame, max_unique_ratio: float = 0.5) -> List[str]:
        """
        自动检测适合分组的列（维度列）
        
        Args:
            df: DataFrame对象
            max_unique_ratio: 唯一值比例阈值（默认0.5，即唯一值少于总行数的50%）
            
        Returns:
            适合分组的列名列表
        """
        groupable_cols = []
        total_rows = len(df)
        
        if total_rows == 0:
            return groupable_cols
        
        for col in df.columns:
            # 排除完全唯一的列（如ID）
            unique_count = df[col].nunique()
            unique_ratio = unique_count / total_rows
            
            # 适合分组的条件：
            # 1. 唯一值比例小于阈值
            # 2. 非数值类型或虽然是数值但唯一值很少
            if unique_ratio < max_unique_ratio:
                if not pd.api.types.is_numeric_dtype(df[col]) or unique_count < 20:
                    groupable_cols.append(col)
        
        return groupable_cols
    
    @staticmethod
    def auto_detect_measure_columns(df: pd.DataFrame) -> List[str]:
        """
        自动检测度量列（数值列）
        
        Args:
            df: DataFrame对象
            
        Returns:
            度量列名列表
        """
        return df.select_dtypes(include=['number']).columns.tolist()
    
    @staticmethod
    def group_and_aggregate(df: pd.DataFrame, 
                           group_by: Optional[List[str]] = None,
                           agg_columns: Optional[List[str]] = None,
                           agg_func: str = 'sum') -> Optional[pd.DataFrame]:
        """
        自动分组聚合（支持自动检测维度和度量）
        
        Args:
            df: DataFrame对象
            group_by: 分组列列表，如果为None则自动检测
            agg_columns: 聚合列列表，如果为None则自动检测所有数值列
            agg_func: 聚合函数 ('sum', 'mean', 'count', 'min', 'max', 'std')
            
        Returns:
            聚合后的DataFrame
        """
        if group_by is None:
            group_by = TableQueryHelper.auto_detect_groupable_columns(df)
            if not group_by:
                logger.warning("无法自动检测分组列")
                return None
        
        if agg_columns is None:
            agg_columns = TableQueryHelper.auto_detect_measure_columns(df)
            if not agg_columns:
                logger.warning("无法找到数值列进行聚合")
                return None
        
        # 验证列存在
        missing_cols = set(group_by + agg_columns) - set(df.columns)
        if missing_cols:
            logger.warning(f"列不存在: {missing_cols}")
            return None
        
        try:
            result = df.groupby(group_by)[agg_columns].agg(agg_func).reset_index()
            return result
        except Exception as e:
            logger.error(f"分组聚合失败: {str(e)}")
            return None
    
    @staticmethod
    def aggregate_by_column(df: pd.DataFrame, group_col: str, agg_col: str, 
                           agg_func: str = 'sum') -> Optional[pd.DataFrame]:
        """
        按列聚合数据
        
        Args:
            df: DataFrame对象
            group_col: 分组列
            agg_col: 聚合列
            agg_func: 聚合函数 ('sum', 'mean', 'count', 'min', 'max')
            
        Returns:
            聚合后的DataFrame
        """
        if group_col not in df.columns or agg_col not in df.columns:
            return None
        
        try:
            result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
            return result
        except Exception as e:
            return None
    
    @staticmethod
    def filter_by_value_range(df: pd.DataFrame, column: str, 
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None) -> Optional[pd.DataFrame]:
        """
        按值范围过滤数据
        
        Args:
            df: DataFrame对象
            column: 列名
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            过滤后的DataFrame
        """
        if column not in df.columns:
            return None
        
        result = df.copy()
        if min_val is not None:
            result = result[result[column] >= min_val]
        if max_val is not None:
            result = result[result[column] <= max_val]
        
        return result
    
    @staticmethod
    def get_top_n_by_column(df: pd.DataFrame, column: str, n: int = 10, 
                           ascending: bool = False) -> pd.DataFrame:
        """
        获取按列排序的前N行
        
        Args:
            df: DataFrame对象
            column: 排序列
            n: 行数
            ascending: 是否升序
            
        Returns:
            排序后的前N行
        """
        if column not in df.columns:
            return df.head(n)
        
        return df.nlargest(n, column) if not ascending else df.nsmallest(n, column)
    
    @staticmethod
    def pivot_table(df: pd.DataFrame, index_col: str, column_col: str, 
                   value_col: str, aggfunc: str = 'sum') -> Optional[pd.DataFrame]:
        """
        创建数据透视表
        
        Args:
            df: DataFrame对象
            index_col: 索引列
            column_col: 列字段
            value_col: 值字段
            aggfunc: 聚合函数
            
        Returns:
            透视表DataFrame
        """
        try:
            result = pd.pivot_table(
                df, 
                values=value_col, 
                index=index_col, 
                columns=column_col, 
                aggfunc=aggfunc
            )
            return result
        except Exception as e:
            return None


class DataCleaningHelper:
    """数据清洗辅助类"""
    
    @staticmethod
    def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据质量并提供清洗建议
        
        Args:
            df: DataFrame对象
            
        Returns:
            数据质量报告和清洗建议
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_analysis': {},
            'suggestions': []
        }
        
        for col in df.columns:
            col_data = df[col]
            null_count = col_data.isnull().sum()
            null_ratio = null_count / len(df) if len(df) > 0 else 0
            
            col_analysis = {
                'null_count': int(null_count),
                'null_ratio': float(null_ratio),
                'dtype': str(col_data.dtype),
                'unique_count': int(col_data.nunique())
            }
            
            # 检测异常值（针对数值列）
            if pd.api.types.is_numeric_dtype(col_data):
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                col_analysis['outlier_count'] = len(outliers)
                col_analysis['outlier_ratio'] = len(outliers) / len(df) if len(df) > 0 else 0
            
            report['columns_analysis'][col] = col_analysis
            
            # 生成建议
            if null_ratio > 0.5:
                report['suggestions'].append({
                    'column': col,
                    'type': 'high_missing',
                    'severity': 'high',
                    'message': f"列 '{col}' 缺失值比例高达 {null_ratio:.1%}，建议考虑删除此列或使用特殊处理"
                })
            elif null_ratio > 0.1:
                report['suggestions'].append({
                    'column': col,
                    'type': 'moderate_missing',
                    'severity': 'medium',
                    'message': f"列 '{col}' 有 {null_ratio:.1%} 的缺失值，建议进行填充（均值/中位数/众数）"
                })
            
            if pd.api.types.is_numeric_dtype(col_data) and col_analysis.get('outlier_ratio', 0) > 0.05:
                report['suggestions'].append({
                    'column': col,
                    'type': 'outliers',
                    'severity': 'medium',
                    'message': f"列 '{col}' 有 {col_analysis['outlier_ratio']:.1%} 的异常值，建议检查并处理"
                })
        
        return report
    
    @staticmethod
    def suggest_fill_strategy(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        为特定列建议缺失值填充策略
        
        Args:
            df: DataFrame对象
            column: 列名
            
        Returns:
            填充策略建议
        """
        if column not in df.columns:
            return {'error': f"列 '{column}' 不存在"}
        
        col_data = df[column]
        null_count = col_data.isnull().sum()
        
        if null_count == 0:
            return {'strategy': 'none', 'message': '该列没有缺失值'}
        
        suggestion = {
            'null_count': int(null_count),
            'null_ratio': float(null_count / len(df)) if len(df) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            # 数值列建议
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            suggestion['strategy'] = 'mean_or_median'
            suggestion['options'] = {
                'mean': float(mean_val) if not pd.isna(mean_val) else None,
                'median': float(median_val) if not pd.isna(median_val) else None,
                'zero': 0
            }
            suggestion['recommended'] = 'median'
            suggestion['message'] = '数值列建议使用中位数填充，更不易受异常值影响'
        else:
            # 非数值列建议
            mode_val = col_data.mode()
            
            suggestion['strategy'] = 'mode_or_constant'
            suggestion['options'] = {
                'mode': mode_val[0] if len(mode_val) > 0 else None,
                'constant': '未知',
                'forward_fill': 'ffill',
                'backward_fill': 'bfill'
            }
            suggestion['recommended'] = 'mode'
            suggestion['message'] = '分类列建议使用众数或指定常量填充'
        
        return suggestion
    
    @staticmethod
    def clean_data(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        执行数据清洗操作
        
        Args:
            df: DataFrame对象
            operations: 清洗操作列表，每个操作是一个字典:
                {
                    'type': 'fill_null' | 'remove_duplicates' | 'remove_outliers' | 'drop_column',
                    'column': '列名',  # 部分操作需要
                    'value': 填充值,    # fill_null需要
                    'method': '方法'   # 可选
                }
                
        Returns:
            清洗后的DataFrame
        """
        cleaned_df = df.copy()
        
        for op in operations:
            op_type = op.get('type')
            
            try:
                if op_type == 'fill_null':
                    column = op.get('column')
                    value = op.get('value')
                    method = op.get('method', 'constant')
                    
                    if column and column in cleaned_df.columns:
                        if method == 'constant':
                            cleaned_df[column].fillna(value, inplace=True)
                        elif method == 'mean':
                            cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                        elif method == 'median':
                            cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                        elif method == 'mode':
                            mode_val = cleaned_df[column].mode()
                            if len(mode_val) > 0:
                                cleaned_df[column].fillna(mode_val[0], inplace=True)
                        elif method == 'ffill':
                            cleaned_df[column].fillna(method='ffill', inplace=True)
                        elif method == 'bfill':
                            cleaned_df[column].fillna(method='bfill', inplace=True)
                
                elif op_type == 'remove_duplicates':
                    subset = op.get('columns')  # 可选，指定基于哪些列去重
                    cleaned_df = cleaned_df.drop_duplicates(subset=subset)
                
                elif op_type == 'remove_outliers':
                    column = op.get('column')
                    if column and column in cleaned_df.columns:
                        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                            q1 = cleaned_df[column].quantile(0.25)
                            q3 = cleaned_df[column].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            cleaned_df = cleaned_df[
                                (cleaned_df[column] >= lower_bound) & 
                                (cleaned_df[column] <= upper_bound)
                            ]
                
                elif op_type == 'drop_column':
                    column = op.get('column')
                    if column and column in cleaned_df.columns:
                        cleaned_df = cleaned_df.drop(columns=[column])
                
            except Exception as e:
                logger.error(f"清洗操作失败 {op}: {str(e)}")
                continue
        
        return cleaned_df
