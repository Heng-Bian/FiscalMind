"""
FiscalMind - 面向财务BP的表格分析Agent
A table analysis agent for financial business partners.
"""

__version__ = "3.0.0"

# 导出主要类
from fiscal_mind.parser import ExcelParser, ExcelDocument
from fiscal_mind.agent import TableDocumentAgent
from fiscal_mind.enhanced_agent import FunctionCallingAgent
from fiscal_mind.prr_agent import PRRAgent
from fiscal_mind.specialized_agents import (
    BusinessAnalysisAgent,
    CriticAgent,
    JudgmentAgent,
    collaborate_business_and_critic
)

__all__ = [
    'ExcelParser',
    'ExcelDocument', 
    'TableDocumentAgent',
    'FunctionCallingAgent',
    'PRRAgent',
    'BusinessAnalysisAgent',
    'CriticAgent',
    'JudgmentAgent',
    'collaborate_business_and_critic',
]
