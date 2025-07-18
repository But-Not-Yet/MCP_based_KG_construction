"""
分析流程控制器 - 协调全局分析和细节分析模块
Analysis Pipeline Controller

此控制器负责：
1. 统一分析入口
2. 数据预处理和格式转换
3. 模块间协调调用
4. 结果整合和输出格式化
"""

import asyncio
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# 导入分析模块
from .global_analysis import GlobalAnalyzer, AnalysisResult
from .entity_detail_analyzer import EntityDetailAnalyzer, AttributeGap
from .llm_client import LLMClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """分析配置参数"""
    enable_global_analysis: bool = True
    enable_detail_analysis: bool = True
    similarity_threshold: float = 0.3
    causal_threshold: float = 0.3
    confidence_threshold: float = 0.5
    max_recommendations: int = 20
    parallel_execution: bool = True





@dataclass
class InputData:
    """输入数据结构"""
    original_text: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisOutput:
    """分析输出结果"""
    timestamp: str
    input_summary: Dict[str, Any]
    global_analysis_results: Optional[Dict[str, Any]] = None
    detail_analysis_results: Optional[Dict[str, Any]] = None
    integrated_recommendations: List[Dict[str, Any]] = None
    quality_metrics: Dict[str, float] = None
    llm_status: str = "UNKNOWN"


class AnalysisPipeline:
    """分析流程控制器"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()

        # 初始化 LLMClient，可根据配置/环境变量决定是否启用
        self.llm_client = LLMClient()

        self.global_analyzer = GlobalAnalyzer(llm_client=self.llm_client)
        self.detail_analyzer = EntityDetailAnalyzer(llm_client=self.llm_client)
        
        logger.info(f"分析流程控制器初始化完成，配置: {self.config}")
    
    async def run_analysis(self, input_data: InputData) -> AnalysisOutput:
        """
        执行完整的分析流程
        
        Args:
            input_data: 输入数据
            
        Returns:
            完整的分析结果
        """
        logger.info("开始执行分析流程...")
        
        # 1. 数据预处理
        processed_data = self._preprocess_data(input_data)
        
        # 2. 并行或串行执行分析
        if self.config.parallel_execution:
            results = await self._run_parallel_analysis(processed_data)
        else:
            results = await self._run_sequential_analysis(processed_data)
        
        # 3. 结果整合
        integrated_results = self._integrate_results(results)
        
        # 4. 生成输出
        output = self._generate_output(input_data, results, integrated_results)
        
        logger.info("分析流程执行完成")
        return output
    
    def _preprocess_data(self, input_data: InputData) -> Dict[str, Any]:
        """数据预处理"""
        logger.info("开始数据预处理...")
        
        # 转换实体格式
        entities_for_global = self._convert_entities_for_global(input_data.entities)
        relations_for_global = self._convert_relations_for_global(input_data.relations)
        
        # 转换关系格式
        relations_for_detail = self._convert_relations_for_detail(input_data.relations)
        
        processed_data = {
            'original_text': input_data.original_text,
            'entities': input_data.entities,
            'relations': input_data.relations,
            'entities_for_global': entities_for_global,
            'relations_for_global': relations_for_global,
            'relations_for_detail': relations_for_detail,
            'metadata': input_data.metadata or {}
        }
        
        logger.info(f"预处理完成: {len(input_data.entities)}个实体, {len(input_data.relations)}个关系")
        return processed_data
    
    async def _run_parallel_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """并行执行分析"""
        logger.info("开始并行分析...")
        
        tasks = []
        
        # 创建分析任务
        if self.config.enable_global_analysis:
            tasks.append(self._run_global_analysis(processed_data))
        
        if self.config.enable_detail_analysis:
            tasks.append(self._run_detail_analysis(processed_data))
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        analysis_results = {}
        if self.config.enable_global_analysis:
            analysis_results['global'] = results[0] if not isinstance(results[0], Exception) else None
            if isinstance(results[0], Exception):
                logger.error(f"全局分析出错: {results[0]}")
        
        if self.config.enable_detail_analysis:
            detail_index = 1 if self.config.enable_global_analysis else 0
            analysis_results['detail'] = results[detail_index] if not isinstance(results[detail_index], Exception) else None
            if isinstance(results[detail_index], Exception):
                logger.error(f"细节分析出错: {results[detail_index]}")
        
        return analysis_results
    
    async def _run_sequential_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """串行执行分析"""
        logger.info("开始串行分析...")
        
        results = {}
        
        # 全局分析
        if self.config.enable_global_analysis:
            try:
                results['global'] = await self._run_global_analysis(processed_data)
            except Exception as e:
                logger.error(f"全局分析出错: {e}")
                results['global'] = None
        
        # 细节分析
        if self.config.enable_detail_analysis:
            try:
                results['detail'] = await self._run_detail_analysis(processed_data)
            except Exception as e:
                logger.error(f"细节分析出错: {e}")
                results['detail'] = None
        
        return results
    
    async def _run_global_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, AnalysisResult]:
        """执行全局分析"""
        logger.info("执行全局分析...")
        
        # 构建知识图谱数据
        kg_data = {
            'entities': processed_data['entities_for_global'],
            'relations': processed_data['relations_for_global']
        }
        
        # 加载数据到全局分析器
        self.global_analyzer.load_knowledge_graph(kg_data)
        
        # 执行分析
        results = self.global_analyzer.analyze_all_modules()
        
        logger.info(f"全局分析完成，共{len(results)}个模块结果")
        return results
    
    async def _run_detail_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行细节分析"""
        logger.info("执行细节分析...")
        
        # 执行分析
        results = await self.detail_analyzer.analyze_entity_details(
            processed_data['entities'],
            processed_data['relations_for_detail'],
            processed_data['original_text']
        )
        
        logger.info(f"细节分析完成，发现{len(results.get('attribute_gaps', []))}个属性缺失")
        return results
    
    def _integrate_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """整合分析结果"""
        logger.info("整合分析结果...")
        
        integrated = {
            'priority_recommendations': [],
            'all_recommendations': [],
            'issue_summary': {
                'critical_issues': 0,
                'major_issues': 0,
                'minor_issues': 0,
                'total_issues': 0
            },
            'enhancement_opportunities': [],
            'quality_score': 0.0
        }
        
        # 整合全局分析结果
        if analysis_results.get('global'):
            global_recommendations = self._extract_global_recommendations(analysis_results['global'])
            integrated['all_recommendations'].extend(global_recommendations)
        
        # 整合细节分析结果
        if analysis_results.get('detail'):
            detail_recommendations = self._extract_detail_recommendations(analysis_results['detail'])
            integrated['all_recommendations'].extend(detail_recommendations)
        
        # 按优先级排序和筛选
        integrated['priority_recommendations'] = self._prioritize_recommendations(
            integrated['all_recommendations']
        )
        
        # 计算质量评分
        integrated['quality_score'] = self._calculate_quality_score(analysis_results)
        
        # 生成问题摘要
        integrated['issue_summary'] = self._generate_issue_summary(integrated['all_recommendations'])
        
        logger.info(f"结果整合完成，共{len(integrated['all_recommendations'])}个建议")
        return integrated
    
    def _extract_global_recommendations(self, global_results: Dict[str, AnalysisResult]) -> List[Dict[str, Any]]:
        """提取全局分析建议"""
        recommendations = []
        
        for module_name, result in global_results.items():
            for rec in result.recommendations:
                recommendations.append({
                    'source': 'global_analysis',
                    'module': module_name,
                    'type': rec.get('type', 'unknown'),
                    'description': self._format_recommendation_description(rec),
                    'priority': self._calculate_priority(rec, result.confidence_score),
                    'confidence': result.confidence_score,
                    'implementation': rec,
                    'category': self._categorize_recommendation(rec)
                })
        
        return recommendations
    
    def _extract_detail_recommendations(self, detail_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取细节分析建议"""
        recommendations = []
        
        for suggestion in detail_results.get('enhancement_suggestions', []):
            recommendations.append({
                'source': 'detail_analysis',
                'module': 'entity_detail',
                'type': suggestion.get('type', 'unknown'),
                'description': suggestion.get('action', ''),
                'priority': suggestion.get('priority', '中'),
                'confidence': 0.8,  # 细节分析置信度
                'implementation': suggestion,
                'category': self._categorize_recommendation(suggestion)
            })
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按优先级排序建议"""
        # 优先级排序规则
        priority_order = {'高': 3, '中': 2, '低': 1}
        
        # 按优先级和置信度排序
        sorted_recommendations = sorted(
            recommendations,
            key=lambda x: (priority_order.get(x['priority'], 0), x['confidence']),
            reverse=True
        )
        
        # 返回前N个高优先级建议
        return sorted_recommendations[:self.config.max_recommendations]
    
    def _calculate_quality_score(self, analysis_results: Dict[str, Any]) -> float:
        """计算整体质量评分"""
        score = 100.0  # 基础分数
        
        # 全局分析扣分
        if analysis_results.get('global'):
            for module_name, result in analysis_results['global'].items():
                # 根据发现的问题数量扣分
                issue_count = len(result.findings)
                score -= min(issue_count * 2, 20)  # 每个问题扣2分，最多扣20分
        
        # 细节分析扣分
        if analysis_results.get('detail'):
            detail_result = analysis_results['detail']
            # 属性缺失扣分
            attribute_gaps = len(detail_result.get('attribute_gaps', []))
            score -= min(attribute_gaps * 3, 30)  # 每个属性缺失扣3分，最多扣30分
            
            # 逻辑错误扣分
            logical_errors = len(detail_result.get('logical_errors', []))
            score -= min(logical_errors * 5, 25)  # 每个逻辑错误扣5分，最多扣25分
        
        return max(0.0, min(100.0, score))
    
    def _generate_issue_summary(self, recommendations: List[Dict[str, Any]]) -> Dict[str, int]:
        """生成问题摘要"""
        summary = {
            'critical_issues': 0,
            'major_issues': 0,
            'minor_issues': 0,
            'total_issues': 0
        }
        
        for rec in recommendations:
            priority = rec.get('priority', '低')
            if priority == '高':
                summary['critical_issues'] += 1
            elif priority == '中':
                summary['major_issues'] += 1
            else:
                summary['minor_issues'] += 1
        
        summary['total_issues'] = len(recommendations)
        return summary
    
    def _generate_output(self, input_data: InputData, analysis_results: Dict[str, Any], 
                        integrated_results: Dict[str, Any]) -> AnalysisOutput:
        """生成输出结果"""
        output = AnalysisOutput(
            timestamp=datetime.now().isoformat(),
            input_summary={
                'text_length': len(input_data.original_text),
                'entity_count': len(input_data.entities),
                'relation_count': len(input_data.relations),
                'metadata': input_data.metadata
            },
            global_analysis_results=analysis_results.get('global'),
            detail_analysis_results=analysis_results.get('detail'),
            integrated_recommendations=integrated_results.get('priority_recommendations', []),
            quality_metrics={
                'overall_score': integrated_results.get('quality_score', 0.0),
                'issue_count': integrated_results.get('issue_summary', {}).get('total_issues', 0),
                'critical_issues': integrated_results.get('issue_summary', {}).get('critical_issues', 0)
            }
        )
        
        # Add LLM status to the output
        if self.llm_client:
            output.llm_status = "OPERATIONAL" if self.llm_client.is_operational else "DEGRADED (check API key/environment variables)"

        return output
    
    # 辅助方法
    def _convert_entities_for_global(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为全局分析转换实体格式"""
        converted = []
        for entity in entities:
            converted.append({
                'name': entity.get('name', ''),
                'type': entity.get('type', 'unknown'),
                'attributes': entity.get('attributes', {}),
                'relations': entity.get('relations', [])
            })
        return converted
    
    def _convert_relations_for_global(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为全局分析转换关系格式"""
        converted = []
        for relation in relations:
            converted.append({
                'name': relation.get('name', relation.get('predicate', '')),
                'source': relation.get('source', relation.get('subject', '')),
                'target': relation.get('target', relation.get('object', '')),
                'type': relation.get('type', 'unknown'),
                'attributes': relation.get('attributes', {})
            })
        return converted
    
    def _convert_relations_for_detail(self, relations: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """为细节分析转换关系格式"""
        converted = []
        for relation in relations:
            subject = relation.get('source', relation.get('subject', ''))
            predicate = relation.get('name', relation.get('predicate', ''))
            object_val = relation.get('target', relation.get('object', ''))
            converted.append((subject, predicate, object_val))
        return converted
    
    def _format_recommendation_description(self, recommendation: Dict[str, Any]) -> str:
        """格式化建议描述"""
        rec_type = recommendation.get('type', '')
        if rec_type == 'similar_entity_relation':
            entities = recommendation.get('entities', [])
            return f"为实体 {entities[0]} 和 {entities[1]} 添加相似关系"
        elif rec_type == 'add_causal_relation':
            cause = recommendation.get('cause', '')
            effect = recommendation.get('effect', '')
            return f"添加因果关系: {cause} → {effect}"
        else:
            return recommendation.get('description', str(recommendation))
    
    def _categorize_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """分类建议"""
        rec_type = recommendation.get('type', '')
        if 'relation' in rec_type:
            return '关系补全'
        elif 'attribute' in rec_type:
            return '属性补全'
        elif 'logical' in rec_type:
            return '逻辑修正'
        else:
            return '其他增强'
    
    def _calculate_priority(self, recommendation: Dict[str, Any], confidence: float) -> str:
        """计算建议优先级"""
        if confidence >= 0.8:
            return '高'
        elif confidence >= 0.6:
            return '中'
        else:
            return '低'


# 便捷函数
async def analyze_knowledge_graph(
    text: str,
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    config: AnalysisConfig = None
) -> AnalysisOutput:
    """
    便捷的知识图谱分析函数
    
    Args:
        text: 原始文本
        entities: 实体列表
        relations: 关系列表
        config: 分析配置
    
    Returns:
        分析结果
    """
    pipeline = AnalysisPipeline(config)
    input_data = InputData(
        original_text=text,
        entities=entities,
        relations=relations
    )
    
    return await pipeline.run_analysis(input_data)


# 使用示例
async def main():
    """使用示例"""
    # 示例数据
    sample_text = """
    张三是阿里巴巴的高级工程师，负责云计算平台的开发。
    他毕业于清华大学计算机科学专业，有10年的工作经验。
    阿里巴巴成立于1999年，总部位于杭州，是中国最大的电商公司之一。
    """
    
    sample_entities = [
        {
            'name': '张三',
            'type': 'Person',
            'attributes': {'职业': '高级工程师', '工作年限': '10年'},
            'relations': ['工作于', '毕业于']
        },
        {
            'name': '阿里巴巴',
            'type': 'Organization',
            'attributes': {'成立时间': '1999年', '总部': '杭州'},
            'relations': ['雇佣', '位于']
        },
        {
            'name': '清华大学',
            'type': 'Organization',
            'attributes': {'类型': '高等院校'},
            'relations': ['培养']
        }
    ]
    
    sample_relations = [
        {
            'name': '工作于',
            'source': '张三',
            'target': '阿里巴巴',
            'type': '雇佣关系'
        },
        {
            'name': '毕业于',
            'source': '张三',
            'target': '清华大学',
            'type': '教育关系'
        }
    ]
    
    # 执行分析
    config = AnalysisConfig(
        similarity_threshold=0.3,
        max_recommendations=10,
        parallel_execution=True
    )
    
    result = await analyze_knowledge_graph(
        sample_text,
        sample_entities,
        sample_relations,
        config
    )
    
    # 输出结果
    print(f"分析完成！质量评分: {result.quality_metrics['overall_score']:.1f}")
    print(f"发现问题: {result.quality_metrics['issue_count']} 个")
    print("\n优先级建议:")
    for i, rec in enumerate(result.integrated_recommendations[:5], 1):
        print(f"{i}. [{rec['priority']}] {rec['description']}")


if __name__ == "__main__":
    asyncio.run(main()) 