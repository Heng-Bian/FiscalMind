"""
元功能模块 - 提供表格数据与LLM交互的工具函数
Meta-functions module for table data and LLM interaction.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from fiscal_mind.parser import ExcelDocument, ExcelParser


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
