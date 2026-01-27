"""
Excel文档解析器模块
Excel document parser module for extracting and processing table data.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExcelDocument:
    """表示单个Excel文档及其内容"""
    
    def __init__(self, file_path: str):
        """
        初始化Excel文档
        
        Args:
            file_path: Excel文件路径
        """
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.sheets: Dict[str, pd.DataFrame] = {}
        self._load_document()
    
    def _load_document(self):
        """加载Excel文档的所有工作表"""
        try:
            # 读取所有工作表
            excel_file = pd.ExcelFile(self.file_path)
            for sheet_name in excel_file.sheet_names:
                self.sheets[sheet_name] = pd.read_excel(
                    excel_file, 
                    sheet_name=sheet_name
                )
            logger.info(f"成功加载文档: {self.file_name}, 包含 {len(self.sheets)} 个工作表")
        except Exception as e:
            logger.error(f"加载文档失败 {self.file_path}: {str(e)}")
            raise
    
    def get_sheet_names(self) -> List[str]:
        """获取所有工作表名称"""
        return list(self.sheets.keys())
    
    def get_sheet(self, sheet_name: str) -> Optional[pd.DataFrame]:
        """获取指定工作表的数据"""
        return self.sheets.get(sheet_name)
    
    def get_sheet_summary(self, sheet_name: str) -> Dict[str, Any]:
        """
        获取工作表摘要信息
        
        Args:
            sheet_name: 工作表名称
            
        Returns:
            包含工作表统计信息的字典
        """
        df = self.get_sheet(sheet_name)
        if df is None:
            return {}
        
        summary = {
            "sheet_name": sheet_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "has_null": df.isnull().any().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        return summary
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        获取整个文档的摘要信息
        
        Returns:
            包含文档统计信息的字典
        """
        total_rows = sum(len(df) for df in self.sheets.values())
        total_cols = sum(len(df.columns) for df in self.sheets.values())
        
        summary = {
            "file_name": self.file_name,
            "file_path": str(self.file_path),
            "num_sheets": len(self.sheets),
            "sheet_names": self.get_sheet_names(),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "sheets_summary": {
                name: self.get_sheet_summary(name) 
                for name in self.get_sheet_names()
            }
        }
        
        return summary
    
    def search_value(self, value: Any, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        在工作表中搜索特定值
        
        Args:
            value: 要搜索的值
            sheet_name: 指定工作表名称，如果为None则搜索所有工作表
            
        Returns:
            包含搜索结果的列表
        """
        results = []
        sheets_to_search = {sheet_name: self.sheets[sheet_name]} if sheet_name else self.sheets
        
        for name, df in sheets_to_search.items():
            # 搜索包含该值的单元格
            for col in df.columns:
                mask = df[col].astype(str).str.contains(str(value), case=False, na=False)
                if mask.any():
                    matching_rows = df[mask]
                    for idx, row in matching_rows.iterrows():
                        results.append({
                            "sheet": name,
                            "row": idx,
                            "column": col,
                            "value": row[col],
                            "row_data": row.to_dict()
                        })
        
        return results
    
    def get_column_data(self, column_name: str, sheet_name: str) -> Optional[pd.Series]:
        """
        获取指定列的数据
        
        Args:
            column_name: 列名
            sheet_name: 工作表名称
            
        Returns:
            列数据Series，如果不存在则返回None
        """
        df = self.get_sheet(sheet_name)
        if df is not None and column_name in df.columns:
            return df[column_name]
        return None
    
    def filter_rows(self, sheet_name: str, conditions: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        根据条件过滤行（保留向后兼容，等于操作）
        
        Args:
            sheet_name: 工作表名称
            conditions: 过滤条件字典，键为列名，值为过滤值
            
        Returns:
            过滤后的DataFrame
        """
        df = self.get_sheet(sheet_name)
        if df is None:
            return None
        
        filtered_df = df.copy()
        for col, value in conditions.items():
            if col in df.columns:
                filtered_df = filtered_df[filtered_df[col] == value]
        
        return filtered_df
    
    def filter_rows_advanced(self, sheet_name: str, filters: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        高级行过滤，支持多种比较操作符
        
        Args:
            sheet_name: 工作表名称
            filters: 过滤条件列表，每个条件是一个字典:
                    {
                        'column': '列名',
                        'operator': '操作符',  # ==, !=, >, <, >=, <=, between, in, contains
                        'value': 值 或 [min, max] (for between) 或 [值列表] (for in)
                    }
        
        Returns:
            过滤后的DataFrame
            
        Example:
            filters = [
                {'column': '销售额', 'operator': '>', 'value': 1000000},
                {'column': '区域', 'operator': 'in', 'value': ['华东', '华南']},
                {'column': '日期', 'operator': 'between', 'value': ['2023-01-01', '2023-12-31']}
            ]
        """
        df = self.get_sheet(sheet_name)
        if df is None:
            return None
        
        filtered_df = df.copy()
        
        for filter_cond in filters:
            col = filter_cond.get('column')
            operator = filter_cond.get('operator', '==')
            value = filter_cond.get('value')
            
            if col not in df.columns:
                logger.warning(f"列 '{col}' 不存在，跳过此过滤条件")
                continue
            
            try:
                if operator == '==':
                    filtered_df = filtered_df[filtered_df[col] == value]
                elif operator == '!=':
                    filtered_df = filtered_df[filtered_df[col] != value]
                elif operator == '>':
                    filtered_df = filtered_df[filtered_df[col] > value]
                elif operator == '<':
                    filtered_df = filtered_df[filtered_df[col] < value]
                elif operator == '>=':
                    filtered_df = filtered_df[filtered_df[col] >= value]
                elif operator == '<=':
                    filtered_df = filtered_df[filtered_df[col] <= value]
                elif operator == 'between':
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        filtered_df = filtered_df[
                            (filtered_df[col] >= value[0]) & (filtered_df[col] <= value[1])
                        ]
                    else:
                        logger.warning(f"between操作符需要[min, max]格式的值")
                elif operator == 'in':
                    if isinstance(value, (list, tuple)):
                        filtered_df = filtered_df[filtered_df[col].isin(value)]
                    else:
                        logger.warning(f"in操作符需要列表格式的值")
                elif operator == 'contains':
                    filtered_df = filtered_df[
                        filtered_df[col].astype(str).str.contains(str(value), case=False, na=False)
                    ]
                else:
                    logger.warning(f"不支持的操作符: {operator}")
            except Exception as e:
                logger.error(f"过滤条件应用失败 {filter_cond}: {str(e)}")
                continue
        
        return filtered_df
    
    def sort_rows(self, sheet_name: str, sort_by: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        对工作表行进行排序
        
        Args:
            sheet_name: 工作表名称
            sort_by: 排序条件列表，每个条件是一个字典:
                    {
                        'column': '列名',
                        'ascending': True/False  # True为升序，False为降序
                    }
        
        Returns:
            排序后的DataFrame
            
        Example:
            sort_by = [
                {'column': '销售额', 'ascending': False},  # 按销售额降序
                {'column': '日期', 'ascending': True}       # 再按日期升序
            ]
        """
        df = self.get_sheet(sheet_name)
        if df is None:
            return None
        
        # 提取列名和排序方向
        columns = []
        ascending = []
        
        for sort_cond in sort_by:
            col = sort_cond.get('column')
            asc = sort_cond.get('ascending', True)
            
            if col in df.columns:
                columns.append(col)
                ascending.append(asc)
            else:
                logger.warning(f"列 '{col}' 不存在，跳过此排序条件")
        
        if not columns:
            return df.copy()
        
        try:
            sorted_df = df.sort_values(by=columns, ascending=ascending)
            return sorted_df
        except Exception as e:
            logger.error(f"排序失败: {str(e)}")
            return df.copy()


class TableJoiner:
    """表格关联工具类，支持多表join操作"""
    
    @staticmethod
    def join_tables(left_df: pd.DataFrame, right_df: pd.DataFrame, 
                   left_on: str, right_on: str, 
                   how: str = 'inner') -> Optional[pd.DataFrame]:
        """
        关联两个表格
        
        Args:
            left_df: 左表DataFrame
            right_df: 右表DataFrame
            left_on: 左表关联键
            right_on: 右表关联键
            how: 关联方式 ('inner', 'left', 'right', 'outer')
            
        Returns:
            关联后的DataFrame
        """
        try:
            result = pd.merge(
                left_df, 
                right_df, 
                left_on=left_on, 
                right_on=right_on, 
                how=how,
                suffixes=('_left', '_right')
            )
            return result
        except Exception as e:
            logger.error(f"表格关联失败: {str(e)}")
            return None
    
    @staticmethod
    def vlookup(lookup_df: pd.DataFrame, table_df: pd.DataFrame, 
               lookup_column: str, return_column: str,
               match_column: str) -> pd.Series:
        """
        模拟Excel的VLOOKUP功能
        
        Args:
            lookup_df: 需要查找的DataFrame
            table_df: 查找表DataFrame
            lookup_column: lookup_df中用于匹配的列
            match_column: table_df中用于匹配的列
            return_column: table_df中要返回的列
            
        Returns:
            查找结果Series
        """
        try:
            # 创建查找字典
            lookup_dict = table_df.set_index(match_column)[return_column].to_dict()
            # 应用查找
            result = lookup_df[lookup_column].map(lookup_dict)
            return result
        except Exception as e:
            logger.error(f"VLOOKUP失败: {str(e)}")
            return pd.Series()


class ExcelParser:
    """Excel文档解析器，支持处理多个文档"""
    
    def __init__(self):
        """初始化解析器"""
        self.documents: Dict[str, ExcelDocument] = {}
        self.joiner = TableJoiner()
    
    def load_document(self, file_path: str) -> ExcelDocument:
        """
        加载Excel文档
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            ExcelDocument对象
        """
        doc = ExcelDocument(file_path)
        self.documents[doc.file_name] = doc
        return doc
    
    def load_documents(self, file_paths: List[str]) -> List[ExcelDocument]:
        """
        批量加载多个Excel文档
        
        Args:
            file_paths: Excel文件路径列表
            
        Returns:
            ExcelDocument对象列表
        """
        return [self.load_document(path) for path in file_paths]
    
    def get_document(self, file_name: str) -> Optional[ExcelDocument]:
        """
        获取已加载的文档
        
        Args:
            file_name: 文件名
            
        Returns:
            ExcelDocument对象，如果不存在则返回None
        """
        return self.documents.get(file_name)
    
    def get_all_documents(self) -> Dict[str, ExcelDocument]:
        """获取所有已加载的文档"""
        return self.documents
    
    def get_documents_summary(self) -> Dict[str, Any]:
        """
        获取所有文档的摘要信息
        
        Returns:
            包含所有文档统计信息的字典
        """
        return {
            "total_documents": len(self.documents),
            "documents": {
                name: doc.get_document_summary() 
                for name, doc in self.documents.items()
            }
        }
    
    def search_across_documents(self, value: Any) -> Dict[str, List[Dict[str, Any]]]:
        """
        在所有文档中搜索值
        
        Args:
            value: 要搜索的值
            
        Returns:
            按文档名分组的搜索结果
        """
        results = {}
        for name, doc in self.documents.items():
            doc_results = doc.search_value(value)
            if doc_results:
                results[name] = doc_results
        return results
    
    def join_sheets(self, doc1_name: str, sheet1_name: str, 
                   doc2_name: str, sheet2_name: str,
                   left_on: str, right_on: str, 
                   how: str = 'inner') -> Optional[pd.DataFrame]:
        """
        跨文档关联工作表
        
        Args:
            doc1_name: 第一个文档名
            sheet1_name: 第一个工作表名
            doc2_name: 第二个文档名
            sheet2_name: 第二个工作表名
            left_on: 第一个表的关联键
            right_on: 第二个表的关联键
            how: 关联方式 ('inner', 'left', 'right', 'outer')
            
        Returns:
            关联后的DataFrame
        """
        doc1 = self.get_document(doc1_name)
        doc2 = self.get_document(doc2_name)
        
        if not doc1 or not doc2:
            logger.error("文档未找到")
            return None
        
        df1 = doc1.get_sheet(sheet1_name)
        df2 = doc2.get_sheet(sheet2_name)
        
        if df1 is None or df2 is None:
            logger.error("工作表未找到")
            return None
        
        return self.joiner.join_tables(df1, df2, left_on, right_on, how)
