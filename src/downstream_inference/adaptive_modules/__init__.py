"""
Adaptive Planning Modules for T-RAG
Version 1: Query Decomposition
"""

from .query_decomposer import analyze_and_decompose_query, is_multi_hop_query
from .prompts import (
    SYSTEM_PROMPT_QUERY_ANALYSIS,
    USER_PROMPT_QUERY_ANALYSIS
)

__all__ = [
    'analyze_and_decompose_query',
    'is_multi_hop_query',
    'SYSTEM_PROMPT_QUERY_ANALYSIS',
    'USER_PROMPT_QUERY_ANALYSIS'
]

__version__ = '1.0.0'
