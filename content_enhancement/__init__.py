"""
Content Enhancement Package
知识图谱内容增强模块包

包含:
- global_analysis: 全局分析模块
- entity_detail_analyzer: 实体细节分析模块  
- analysis_pipeline: 分析流程控制模块
"""

from .global_analysis import GlobalAnalyzer
from .entity_detail_analyzer import EntityDetailAnalyzer
from .analysis_pipeline import AnalysisPipeline, analyze_knowledge_graph, AnalysisConfig

__version__ = "1.0.0"
__author__ = "Knowledge Graph Enhancement Team"

# 导出主要类和函数
__all__ = [
    'GlobalAnalyzer',
    'EntityDetailAnalyzer', 
    'AnalysisPipeline',
    'analyze_knowledge_graph',
    'AnalysisConfig'
]

# 包级别配置
DEFAULT_CONFIG = {
    'enable_global_analysis': True,
    'enable_detail_analysis': True,
    'similarity_threshold': 0.3,
    'max_recommendations': 15,
    'parallel_processing': True
}

def get_version():
    """返回包版本"""
    return __version__

def get_default_config():
    """返回默认配置"""
    return DEFAULT_CONFIG.copy() 