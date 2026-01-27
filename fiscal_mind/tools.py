"""
工具定义模块 - LLM Function Calling工具定义
Tools definition module for LLM Function Calling.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ==================== Pydantic Models for Tool Parameters ====================

class FilterCondition(BaseModel):
    """过滤条件"""
    column: str = Field(..., description="要过滤的列名")
    operator: Literal['==', '!=', '>', '<', '>=', '<=', 'between', 'in', 'contains'] = Field(
        ..., description="比较操作符"
    )
    value: Any = Field(..., description="过滤值，对于between是[min, max]，对于in是值列表")


class SortCondition(BaseModel):
    """排序条件"""
    column: str = Field(..., description="要排序的列名")
    ascending: bool = Field(True, description="True为升序，False为降序")


class AggregationParams(BaseModel):
    """聚合参数"""
    group_by: Optional[List[str]] = Field(None, description="分组列，None则自动检测")
    agg_columns: Optional[List[str]] = Field(None, description="聚合列，None则使用所有数值列")
    agg_func: Literal['sum', 'mean', 'count', 'min', 'max', 'std'] = Field(
        'sum', description="聚合函数"
    )


class JoinParams(BaseModel):
    """表关联参数"""
    doc1_name: str = Field(..., description="第一个文档名")
    sheet1_name: str = Field(..., description="第一个工作表名")
    doc2_name: str = Field(..., description="第二个文档名")
    sheet2_name: str = Field(..., description="第二个工作表名")
    left_on: str = Field(..., description="第一个表的关联键列")
    right_on: str = Field(..., description="第二个表的关联键列")
    how: Literal['inner', 'left', 'right', 'outer'] = Field('inner', description="关联方式")


# ==================== Tool Function Schemas ====================

TOOL_SCHEMAS = [
    {
        "name": "get_document_summary",
        "description": "获取Excel文档的摘要信息，包括工作表列表、行列数、列名等基本信息",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称，如果为空则返回所有文档摘要"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_sheet_data",
        "description": "获取指定工作表的数据内容（前N行）",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                },
                "max_rows": {
                    "type": "integer",
                    "description": "返回的最大行数，默认10",
                    "default": 10
                }
            },
            "required": ["doc_name", "sheet_name"]
        }
    },
    {
        "name": "filter_data",
        "description": "根据条件过滤数据，支持多种比较操作符（==, !=, >, <, >=, <=, between, in, contains）",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                },
                "filters": {
                    "type": "array",
                    "description": "过滤条件列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "operator": {
                                "type": "string",
                                "enum": ["==", "!=", ">", "<", ">=", "<=", "between", "in", "contains"]
                            },
                            "value": {}
                        },
                        "required": ["column", "operator", "value"]
                    }
                }
            },
            "required": ["doc_name", "sheet_name", "filters"]
        }
    },
    {
        "name": "sort_data",
        "description": "对数据进行排序，支持多列排序",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                },
                "sort_by": {
                    "type": "array",
                    "description": "排序条件列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "ascending": {"type": "boolean", "default": True}
                        },
                        "required": ["column"]
                    }
                }
            },
            "required": ["doc_name", "sheet_name", "sort_by"]
        }
    },
    {
        "name": "aggregate_data",
        "description": "对数据进行分组聚合，可以自动检测维度列和度量列",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                },
                "group_by": {
                    "type": "array",
                    "description": "分组列列表，如果为空则自动检测",
                    "items": {"type": "string"}
                },
                "agg_columns": {
                    "type": "array",
                    "description": "要聚合的列列表，如果为空则使用所有数值列",
                    "items": {"type": "string"}
                },
                "agg_func": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "min", "max", "std"],
                    "default": "sum",
                    "description": "聚合函数"
                }
            },
            "required": ["doc_name", "sheet_name"]
        }
    },
    {
        "name": "get_statistics",
        "description": "获取数值列的统计信息（均值、中位数、最大值、最小值等）",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                },
                "column": {
                    "type": "string",
                    "description": "列名，如果为空则返回所有数值列的统计"
                }
            },
            "required": ["doc_name", "sheet_name"]
        }
    },
    {
        "name": "search_value",
        "description": "在工作表中搜索包含特定值的单元格",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称，如果为空则搜索所有工作表"
                },
                "value": {
                    "type": "string",
                    "description": "要搜索的值"
                }
            },
            "required": ["doc_name", "value"]
        }
    },
    {
        "name": "find_columns",
        "description": "根据关键词或语义查找列名",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                },
                "keyword": {
                    "type": "string",
                    "description": "关键词或语义描述，如'销售'、'利润'、'收入'等"
                },
                "use_semantic": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否使用语义搜索"
                }
            },
            "required": ["doc_name", "sheet_name", "keyword"]
        }
    },
    {
        "name": "join_tables",
        "description": "关联两个工作表，支持inner/left/right/outer join",
        "parameters": {
            "type": "object",
            "properties": {
                "doc1_name": {"type": "string", "description": "第一个文档名"},
                "sheet1_name": {"type": "string", "description": "第一个工作表名"},
                "doc2_name": {"type": "string", "description": "第二个文档名"},
                "sheet2_name": {"type": "string", "description": "第二个工作表名"},
                "left_on": {"type": "string", "description": "第一个表的关联键列"},
                "right_on": {"type": "string", "description": "第二个表的关联键列"},
                "how": {
                    "type": "string",
                    "enum": ["inner", "left", "right", "outer"],
                    "default": "inner",
                    "description": "关联方式"
                }
            },
            "required": ["doc1_name", "sheet1_name", "doc2_name", "sheet2_name", "left_on", "right_on"]
        }
    },
    {
        "name": "analyze_data_quality",
        "description": "分析数据质量，包括缺失值、异常值等，并提供清洗建议",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                }
            },
            "required": ["doc_name", "sheet_name"]
        }
    },
    {
        "name": "get_top_n",
        "description": "获取按指定列排序的前N条或后N条记录",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "文档名称"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "工作表名称"
                },
                "column": {
                    "type": "string",
                    "description": "排序的列名"
                },
                "n": {
                    "type": "integer",
                    "default": 10,
                    "description": "返回的记录数"
                },
                "ascending": {
                    "type": "boolean",
                    "default": False,
                    "description": "False获取前N大的，True获取前N小的"
                }
            },
            "required": ["doc_name", "sheet_name", "column"]
        }
    }
]


# ==================== Tool Descriptions for LLM ====================

def get_tools_description() -> str:
    """
    获取所有工具的描述文本，用于提供给LLM
    
    Returns:
        工具描述文本
    """
    descriptions = ["可用的工具函数：\n"]
    
    for i, tool in enumerate(TOOL_SCHEMAS, 1):
        descriptions.append(f"{i}. {tool['name']}: {tool['description']}")
    
    return "\n".join(descriptions)


def get_tool_by_name(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    根据工具名称获取工具schema
    
    Args:
        tool_name: 工具名称
        
    Returns:
        工具schema字典，如果不存在则返回None
    """
    for tool in TOOL_SCHEMAS:
        if tool['name'] == tool_name:
            return tool
    return None
