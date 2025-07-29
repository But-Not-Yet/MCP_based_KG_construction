#!/usr/bin/env python3
"""
增强执行器 - 自动应用分析结果，增强知识图谱
Enhancement Executor - Automatically apply analysis results to enhance knowledge graph
"""

import json
import re
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import asyncio
from .llm_client import LLMClient

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class EnhancementResult:
    """增强结果数据结构"""
    original_entities: List[str]
    original_relations: List[str]
    enhanced_entities: List[Dict[str, Any]]
    enhanced_relations: List[Dict[str, Any]]
    enhanced_triples: List[Dict[str, Any]]
    enhancement_summary: Dict[str, Any]
    applied_enhancements: List[Dict[str, Any]]
    timestamp: str

class EnhancementExecutor:
    """增强执行器"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        # 移除硬编码的关键词和模板，这些现在由其他模块（或LLM）动态提供
        
    async def execute_enhancements(self, 
                                 original_text: str,
                                 entities: List[Dict[str, Any]], 
                                 relations: List[Dict[str, Any]],
                                 analysis_result: Any) -> EnhancementResult:
        """
        执行增强操作
        - 此版本接收结构化输入，并应用所有类型的建议
        """
        logger.info("开始执行增强操作...")
        
        # 直接使用传入的结构化数据进行操作
        enhanced_entities = [e.copy() for e in entities]
        enhanced_triples = [
            {"head": r['source'], "relation": r['name'], "tail": r['target'], "confidence": r.get('confidence', 0.8)}
            for r in relations
        ]
        
        applied_enhancements = []
        
        if hasattr(analysis_result, 'integrated_recommendations'):
            recommendations = analysis_result.integrated_recommendations
            
            for rec in recommendations:
                enhancement = await self._apply_recommendation(
                    rec, enhanced_entities, enhanced_triples, original_text
                )
                if enhancement:
                    applied_enhancements.append(enhancement)
        
        # **关键修复**：从最终的三元组列表重建实体和关系
        final_entities, final_relations = self._rebuild_from_triples(enhanced_triples, enhanced_entities)

        enhancement_summary = self._generate_enhancement_summary(
            [e['name'] for e in entities], 
            [r['name'] for r in relations], 
            final_entities, 
            final_relations, 
            applied_enhancements
        )
        
        result = EnhancementResult(
            original_entities=[e['name'] for e in entities],
            original_relations=[r['name'] for r in relations],
            enhanced_entities=final_entities,
            enhanced_relations=final_relations, # 返回结构化关系
            enhanced_triples=enhanced_triples,
            enhancement_summary=enhancement_summary,
            applied_enhancements=applied_enhancements,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"增强操作完成，应用了 {len(applied_enhancements)} 个增强建议。")
        return result

    def _rebuild_from_triples(self, triples: List[Dict], original_entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """从增强后的三元组列表重新构建实体和关系列表"""
        entity_map = {e['name']: e for e in original_entities}
        relation_set = set()

        for t in triples:
            for entity_name in [t['head'], t['tail']]:
                if entity_name not in entity_map:
                    entity_map[entity_name] = {'name': entity_name, 'type': 'unknown', 'enhanced': True}
            relation_set.add(t['relation'])

        final_entities = list(entity_map.values())
        final_relations = [{'name': r, 'type': 'inferred'} for r in sorted(list(relation_set))]
        return final_entities, final_relations

    async def _apply_recommendation(self, 
                                  recommendation: Dict[str, Any],
                                  enhanced_entities: List[Dict[str, Any]],
                                  enhanced_triples: List[Dict[str, Any]],
                                  original_text: str) -> Optional[Dict[str, Any]]:
        """应用单个建议（简化版）"""
        # (此处的具体实现可以根据需要进一步细化，目前主要依赖逻辑分析)
        category = recommendation.get('category')
        if category == '逻辑推理':
            return await self._apply_logic_suggestion(recommendation, enhanced_triples)
        
        # 移除了过于激进的 'similar_entity_relation' 关系补全逻辑
        # 其他更精确的建议类型可以在此添加处理分支
        
        return None

    async def _apply_logic_suggestion(self,
                                    recommendation: Dict[str, Any],
                                    enhanced_triples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """应用逻辑分析建议，调用LLM生成新三元组"""
        if not self.llm_client or not self.llm_client.is_operational:
            return None

        suggestion_text = recommendation.get('description', '')
        # (构建prompt和调用LLM的逻辑与之前相同...)
        prompt = f"""
你是一个知识图谱构建助手。请将下面的增强建议转化为一个或多个具体的三元组。
增强建议: {suggestion_text}
严格以JSON格式返回一个列表，例如: `[{{"head": "A", "relation": "导致", "tail": "B"}}]`
如果无法生成，返回空列表 `[]`。
"""
        response_text = self.llm_client.custom_query(prompt)
        new_triples_data = self.llm_client._clean_and_parse_json(response_text, "apply_logic_suggestion")

        if new_triples_data and isinstance(new_triples_data, list):
            added_count = 0
            for triple_data in new_triples_data:
                if all(k in triple_data for k in ["head", "relation", "tail"]):
                    new_triple = {**triple_data, 'confidence': recommendation.get('confidence', 0.8), 'enhanced': True, 'source': 'logic_analysis'}
                    if new_triple not in enhanced_triples:
                        enhanced_triples.append(new_triple)
                        added_count += 1
            if added_count > 0:
                return recommendation
        return None

    def _generate_enhancement_summary(self, 
                                    original_entities: List[str],
                                    original_relations: List[str],
                                    enhanced_entities: List[Dict[str, Any]],
                                    enhanced_relations: List[Dict[str, Any]],
                                    applied_enhancements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成增强摘要"""
        return {
            'status': 'Completed',
            'original_entity_count': len(original_entities),
            'enhanced_entity_count': len(enhanced_entities),
            'original_relation_count': len(original_relations),
            'enhanced_relation_count': len(enhanced_relations),
            'applied_enhancements_count': len(applied_enhancements)
        }

# 使用示例
async def main():
    """使用示例"""
    executor = EnhancementExecutor()
    
    # 示例数据
    sample_text = "张三是阿里巴巴的高级工程师，负责人工智能项目。他毕业于清华大学。"
    sample_entities = [{"name": "张三", "type": "Person"}, {"name": "阿里巴巴", "type": "Organization"}, {"name": "清华大学", "type": "Location"}]
    sample_relations = [{"source": "张三", "name": "工作于", "target": "阿里巴巴", "confidence": 0.9}, {"source": "张三", "name": "毕业于", "target": "清华大学", "confidence": 0.8}]
    
    # 模拟分析结果
    class MockAnalysisResult:
        def __init__(self):
            self.integrated_recommendations = [
                {
                    'type': 'similar_entity_relation',
                    'entities': ['张三', '阿里巴巴'],
                    'confidence': 0.8
                }
            ]
    
    analysis_result = MockAnalysisResult()
    
    # 执行增强
    result = await executor.execute_enhancements(
        sample_text, sample_entities, sample_relations, analysis_result
    )
    
    print("增强结果:")
    print(f"原始实体数: {len(result.original_entities)}")
    print(f"增强后实体数: {len(result.enhanced_entities)}")
    print(f"原始关系数: {len(result.original_relations)}")
    print(f"增强后关系数: {len(result.enhanced_relations)}")
    print(f"应用的增强数: {len(result.applied_enhancements)}")

if __name__ == "__main__":
    asyncio.run(main()) 