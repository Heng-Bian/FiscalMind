"""
工具执行器模块 - 执行LLM选择的工具
Tool executor module for executing tools selected by LLM.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import logging
from fiscal_mind.parser import ExcelParser
from fiscal_mind.meta_functions import TableMetaFunctions, TableQueryHelper, DataCleaningHelper

logger = logging.getLogger(__name__)


class ToolExecutor:
    """工具执行器，负责执行各种分析工具"""
    
    def __init__(self, parser: ExcelParser):
        """
        初始化工具执行器
        
        Args:
            parser: ExcelParser实例
        """
        self.parser = parser
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定的工具
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            执行结果
        """
        try:
            # 路由到具体的工具方法
            if tool_name == "get_document_summary":
                return self._get_document_summary(parameters)
            elif tool_name == "get_sheet_data":
                return self._get_sheet_data(parameters)
            elif tool_name == "filter_data":
                return self._filter_data(parameters)
            elif tool_name == "sort_data":
                return self._sort_data(parameters)
            elif tool_name == "aggregate_data":
                return self._aggregate_data(parameters)
            elif tool_name == "get_statistics":
                return self._get_statistics(parameters)
            elif tool_name == "search_value":
                return self._search_value(parameters)
            elif tool_name == "find_columns":
                return self._find_columns(parameters)
            elif tool_name == "join_tables":
                return self._join_tables(parameters)
            elif tool_name == "analyze_data_quality":
                return self._analyze_data_quality(parameters)
            elif tool_name == "get_top_n":
                return self._get_top_n(parameters)
            else:
                return {
                    "success": False,
                    "error": f"未知的工具: {tool_name}"
                }
        except Exception as e:
            logger.error(f"工具执行失败 {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": f"工具执行失败: {str(e)}"
            }
    
    def _get_document_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取文档摘要"""
        doc_name = params.get('doc_name')
        
        if doc_name:
            doc = self.parser.get_document(doc_name)
            if not doc:
                return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
            result = doc.get_document_summary()
        else:
            result = self.parser.get_documents_summary()
        
        return {"success": True, "data": result}
    
    def _get_sheet_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取工作表数据"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        max_rows = params.get('max_rows', 10)
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        df = doc.get_sheet(sheet_name)
        if df is None:
            return {"success": False, "error": f"工作表 '{sheet_name}' 未找到"}
        
        # 返回前N行的数据
        preview = df.head(max_rows)
        result = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "data": preview.to_dict(orient='records'),
            "preview_rows": len(preview)
        }
        
        return {"success": True, "data": result}
    
    def _filter_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """过滤数据"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        filters = params['filters']
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        # 使用高级过滤
        filtered_df = doc.filter_rows_advanced(sheet_name, filters)
        
        if filtered_df is None:
            return {"success": False, "error": "过滤失败"}
        
        result = {
            "filtered_rows": len(filtered_df),
            "columns": filtered_df.columns.tolist(),
            "data": filtered_df.head(20).to_dict(orient='records')
        }
        
        return {"success": True, "data": result}
    
    def _sort_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """排序数据"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        sort_by = params['sort_by']
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        sorted_df = doc.sort_rows(sheet_name, sort_by)
        
        if sorted_df is None:
            return {"success": False, "error": "排序失败"}
        
        result = {
            "total_rows": len(sorted_df),
            "columns": sorted_df.columns.tolist(),
            "data": sorted_df.head(20).to_dict(orient='records')
        }
        
        return {"success": True, "data": result}
    
    def _aggregate_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """聚合数据"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        group_by = params.get('group_by')
        agg_columns = params.get('agg_columns')
        agg_func = params.get('agg_func', 'sum')
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        df = doc.get_sheet(sheet_name)
        if df is None:
            return {"success": False, "error": f"工作表 '{sheet_name}' 未找到"}
        
        agg_result = TableQueryHelper.group_and_aggregate(
            df, group_by, agg_columns, agg_func
        )
        
        if agg_result is None:
            return {"success": False, "error": "聚合失败"}
        
        result = {
            "aggregated_rows": len(agg_result),
            "columns": agg_result.columns.tolist(),
            "data": agg_result.to_dict(orient='records')
        }
        
        return {"success": True, "data": result}
    
    def _get_statistics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取统计信息"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        column = params.get('column')
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        df = doc.get_sheet(sheet_name)
        if df is None:
            return {"success": False, "error": f"工作表 '{sheet_name}' 未找到"}
        
        if column:
            stats = TableMetaFunctions.get_column_statistics(df, column)
        else:
            stats = TableMetaFunctions.get_numeric_summary(df)
        
        return {"success": True, "data": stats}
    
    def _search_value(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """搜索值"""
        doc_name = params['doc_name']
        sheet_name = params.get('sheet_name')
        value = params['value']
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        results = doc.search_value(value, sheet_name)
        
        return {
            "success": True,
            "data": {
                "matches_found": len(results),
                "results": results[:50]  # 限制返回前50个结果
            }
        }
    
    def _find_columns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """查找列"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        keyword = params['keyword']
        use_semantic = params.get('use_semantic', True)
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        df = doc.get_sheet(sheet_name)
        if df is None:
            return {"success": False, "error": f"工作表 '{sheet_name}' 未找到"}
        
        if use_semantic:
            matching_cols = TableQueryHelper.find_column_by_semantic(df, keyword)
        else:
            matching_cols = TableQueryHelper.find_column_by_keyword(df, keyword)
        
        return {
            "success": True,
            "data": {
                "keyword": keyword,
                "matching_columns": matching_cols,
                "all_columns": df.columns.tolist()
            }
        }
    
    def _join_tables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """关联表格"""
        result_df = self.parser.join_sheets(
            params['doc1_name'],
            params['sheet1_name'],
            params['doc2_name'],
            params['sheet2_name'],
            params['left_on'],
            params['right_on'],
            params.get('how', 'inner')
        )
        
        if result_df is None:
            return {"success": False, "error": "表格关联失败"}
        
        result = {
            "joined_rows": len(result_df),
            "columns": result_df.columns.tolist(),
            "data": result_df.head(20).to_dict(orient='records')
        }
        
        return {"success": True, "data": result}
    
    def _analyze_data_quality(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """分析数据质量"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        df = doc.get_sheet(sheet_name)
        if df is None:
            return {"success": False, "error": f"工作表 '{sheet_name}' 未找到"}
        
        quality_report = DataCleaningHelper.analyze_data_quality(df)
        
        return {"success": True, "data": quality_report}
    
    def _get_top_n(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取Top N"""
        doc_name = params['doc_name']
        sheet_name = params['sheet_name']
        column = params['column']
        n = params.get('n', 10)
        ascending = params.get('ascending', False)
        
        doc = self.parser.get_document(doc_name)
        if not doc:
            return {"success": False, "error": f"文档 '{doc_name}' 未找到"}
        
        df = doc.get_sheet(sheet_name)
        if df is None:
            return {"success": False, "error": f"工作表 '{sheet_name}' 未找到"}
        
        top_df = TableQueryHelper.get_top_n_by_column(df, column, n, ascending)
        
        result = {
            "column": column,
            "n": n,
            "ascending": ascending,
            "data": top_df.to_dict(orient='records')
        }
        
        return {"success": True, "data": result}
