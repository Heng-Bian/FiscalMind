"""
工具模块 - 提供辅助功能
Utilities module providing helper functions.
"""

import pandas as pd
from typing import Dict, Any, List
import json


def dataframe_to_json(df: pd.DataFrame) -> str:
    """
    将DataFrame转换为JSON字符串
    
    Args:
        df: DataFrame对象
        
    Returns:
        JSON字符串
    """
    return df.to_json(orient='records', force_ascii=False, indent=2)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    将DataFrame转换为Markdown表格
    
    Args:
        df: DataFrame对象
        max_rows: 最大显示行数
        
    Returns:
        Markdown格式的表格
    """
    return df.head(max_rows).to_markdown()


def export_to_excel(data: Dict[str, pd.DataFrame], output_path: str) -> None:
    """
    将多个DataFrame导出到Excel文件
    
    Args:
        data: 字典，键为工作表名，值为DataFrame
        output_path: 输出文件路径
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def validate_excel_file(file_path: str) -> bool:
    """
    验证文件是否为有效的Excel文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否有效
    """
    try:
        pd.ExcelFile(file_path)
        return True
    except Exception:
        return False


def merge_dataframes(dfs: List[pd.DataFrame], how: str = 'outer') -> pd.DataFrame:
    """
    合并多个DataFrame
    
    Args:
        dfs: DataFrame列表
        how: 合并方式 ('inner', 'outer', 'left', 'right')
        
    Returns:
        合并后的DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.concat([result, df], ignore_index=True)
    
    return result


def create_summary_dict(summary: Dict[str, Any]) -> str:
    """
    将摘要字典格式化为可读文本
    
    Args:
        summary: 摘要字典
        
    Returns:
        格式化的文本
    """
    return json.dumps(summary, ensure_ascii=False, indent=2)
