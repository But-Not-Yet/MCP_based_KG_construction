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
    
    def __init__(self):
        self.entity_type_keywords = {
            'Person': ['先生', '女士', '教授', '博士', '老师', '工程师', '经理', '总监', 'CEO', '董事长'],
            'Organization': ['公司', '企业', '集团', '大学', '学院', '研究所', '部门', '组织', '机构'],
            'Location': ['市', '省', '国家', '地区', '区域', '城市', '县', '镇', '村'],
            'Technology': ['系统', '平台', '技术', '算法', '模型', '框架', '工具', '软件'],
            'Concept': ['理论', '概念', '方法', '思想', '观点', '原理', '策略', '方案'],
            'Event': ['会议', '事件', '活动', '项目', '计划', '过程', '流程'],
            'Product': ['产品', '服务', '解决方案', '应用', '工具']
        }
        
        self.relation_templates = {
            'Person-Organization': ['工作于', '就职于', '任职于', '隶属于', '服务于'],
            'Person-Person': ['合作', '指导', '同事', '上级', '下属', '朋友'],
            'Organization-Location': ['位于', '总部在', '设立在', '分布在'],
            'Person-Technology': ['研发', '使用', '掌握', '开发', '应用'],
            'Person-Concept': ['提出', '研究', '探讨', '分析', '发现'],
            'Organization-Product': ['开发', '生产', '提供', '发布', '销售']
        }
        
    async def execute_enhancements(self, 
                                 original_text: str,
                                 entities: List[str], 
                                 relations: List[str],
                                 triples: List[Any],
                                 analysis_result: Any,
                                 enable_auto_relations: bool = False) -> EnhancementResult:
        """
        执行增强操作
        
        Args:
            original_text: 原始文本
            entities: 原始实体列表
            relations: 原始关系列表
            triples: 原始三元组列表
            analysis_result: 分析结果
            
        Returns:
            增强结果
        """
        logger.info("开始执行增强操作...")
        
        # 初始化增强数据
        enhanced_entities = self._initialize_enhanced_entities(entities)
        enhanced_relations = self._initialize_enhanced_relations(relations)
        enhanced_triples = self._initialize_enhanced_triples(triples)
        
        applied_enhancements = []
        
        # 应用分析建议
        if hasattr(analysis_result, 'integrated_recommendations'):
            recommendations = analysis_result.integrated_recommendations
            
            for rec in recommendations:
                enhancement = await self._apply_recommendation(
                    rec, enhanced_entities, enhanced_relations, enhanced_triples, original_text
                )
                if enhancement:
                    applied_enhancements.append(enhancement)
        
        # 自动实体类型推断和属性补全
        enhanced_entities = self._enhance_entity_types_and_attributes(enhanced_entities, original_text)
        
        # 自动关系补全
        new_relations, new_triples = self._enhance_relations(enhanced_entities, original_text)
        enhanced_relations.extend(new_relations)
        enhanced_triples.extend(new_triples)
        
        # 生成增强摘要
        enhancement_summary = self._generate_enhancement_summary(
            entities, relations, enhanced_entities, enhanced_relations, applied_enhancements
        )
        
        result = EnhancementResult(
            original_entities=entities,
            original_relations=relations,
            enhanced_entities=enhanced_entities,
            enhanced_relations=enhanced_relations,
            enhanced_triples=enhanced_triples,
            enhancement_summary=enhancement_summary,
            applied_enhancements=applied_enhancements,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"增强操作完成，应用了{len(applied_enhancements)}个增强")
        return result
    
    def _initialize_enhanced_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        """初始化增强实体"""
        enhanced = []
        for entity in entities:
            enhanced.append({
                'name': entity,
                'type': 'unknown',
                'attributes': {},
                'confidence': 0.8,
                'enhanced': False
            })
        return enhanced
    
    def _initialize_enhanced_relations(self, relations: List[str]) -> List[Dict[str, Any]]:
        """初始化增强关系"""
        enhanced = []
        for relation in relations:
            enhanced.append({
                'name': relation,
                'type': 'unknown',
                'attributes': {},
                'confidence': 0.8,
                'enhanced': False
            })
        return enhanced
    
    def _initialize_enhanced_triples(self, triples: List[Any]) -> List[Dict[str, Any]]:
        """初始化增强三元组"""
        enhanced = []
        for triple in triples:
            enhanced.append({
                'head': triple.head if hasattr(triple, 'head') else str(triple[0]),
                'relation': triple.relation if hasattr(triple, 'relation') else str(triple[1]),
                'tail': triple.tail if hasattr(triple, 'tail') else str(triple[2]),
                'confidence': getattr(triple, 'confidence', 0.8),
                'enhanced': False
            })
        return enhanced
    
    async def _apply_recommendation(self, 
                                  recommendation: Dict[str, Any],
                                  enhanced_entities: List[Dict[str, Any]],
                                  enhanced_relations: List[Dict[str, Any]],
                                  enhanced_triples: List[Dict[str, Any]],
                                  original_text: str) -> Optional[Dict[str, Any]]:
        """应用单个建议"""
        rec_type = recommendation.get('type', '')
        
        if rec_type == 'similar_entity_relation':
            return self._apply_similar_entity_relation(
                recommendation, enhanced_entities, enhanced_relations, enhanced_triples
            )
        elif rec_type == 'add_causal_relation':
            return self._apply_causal_relation(
                recommendation, enhanced_entities, enhanced_relations, enhanced_triples
            )
        elif rec_type == '属性补全':
            return self._apply_attribute_completion(
                recommendation, enhanced_entities, original_text
            )
        elif rec_type == '关系补全':
            return self._apply_relation_completion(
                recommendation, enhanced_entities, enhanced_relations, enhanced_triples
            )
        else:
            logger.warning(f"未知的建议类型: {rec_type}")
            return None
    
    def _apply_similar_entity_relation(self, 
                                     recommendation: Dict[str, Any],
                                     enhanced_entities: List[Dict[str, Any]],
                                     enhanced_relations: List[Dict[str, Any]],
                                     enhanced_triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """应用相似实体关系建议"""
        entities = recommendation.get('entities', [])
        if len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            
            # 添加相似关系
            relation_name = '相似关系'
            if not any(r['name'] == relation_name for r in enhanced_relations):
                enhanced_relations.append({
                    'name': relation_name,
                    'type': 'similarity',
                    'attributes': {'auto_generated': True},
                    'confidence': 0.7,
                    'enhanced': True
                })
            
            # 添加三元组
            enhanced_triples.append({
                'head': entity1,
                'relation': relation_name,
                'tail': entity2,
                'confidence': 0.7,
                'enhanced': True
            })
            
            return {
                'type': 'similar_entity_relation',
                'description': f'为{entity1}和{entity2}添加相似关系',
                'entities': entities,
                'confidence': 0.7
            }
        return None
    
    def _apply_causal_relation(self, 
                             recommendation: Dict[str, Any],
                             enhanced_entities: List[Dict[str, Any]],
                             enhanced_relations: List[Dict[str, Any]],
                             enhanced_triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """应用因果关系建议"""
        cause = recommendation.get('cause', '')
        effect = recommendation.get('effect', '')
        
        if cause and effect:
            # 添加因果关系
            relation_name = '导致'
            if not any(r['name'] == relation_name for r in enhanced_relations):
                enhanced_relations.append({
                    'name': relation_name,
                    'type': 'causality',
                    'attributes': {'auto_generated': True},
                    'confidence': 0.6,
                    'enhanced': True
                })
            
            # 添加三元组
            enhanced_triples.append({
                'head': cause,
                'relation': relation_name,
                'tail': effect,
                'confidence': 0.6,
                'enhanced': True
            })
            
            return {
                'type': 'causal_relation',
                'description': f'添加因果关系: {cause} → {effect}',
                'cause': cause,
                'effect': effect,
                'confidence': 0.6
            }
        return None
    
    def _apply_attribute_completion(self, 
                                  recommendation: Dict[str, Any],
                                  enhanced_entities: List[Dict[str, Any]],
                                  original_text: str) -> Dict[str, Any]:
        """应用属性补全建议"""
        target_entity = recommendation.get('target_entity', '')
        action = recommendation.get('action', '')
        
        # 找到目标实体
        for entity in enhanced_entities:
            if entity['name'] == target_entity:
                # 根据文本内容推断属性
                inferred_attributes = self._infer_attributes_from_text(target_entity, original_text)
                entity['attributes'].update(inferred_attributes)
                entity['enhanced'] = True
                
                return {
                    'type': 'attribute_completion',
                    'description': f'为{target_entity}补全属性',
                    'entity': target_entity,
                    'added_attributes': list(inferred_attributes.keys()),
                    'confidence': 0.6
                }
        return None
    
    def _apply_relation_completion(self, 
                                 recommendation: Dict[str, Any],
                                 enhanced_entities: List[Dict[str, Any]],
                                 enhanced_relations: List[Dict[str, Any]],
                                 enhanced_triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """应用关系补全建议"""
        target_entities = recommendation.get('target_entities', [])
        
        if len(target_entities) >= 2:
            entity1, entity2 = target_entities[0], target_entities[1]
            
            # 根据实体类型推断关系
            type1 = self._get_entity_type(entity1, enhanced_entities)
            type2 = self._get_entity_type(entity2, enhanced_entities)
            
            suggested_relation = self._suggest_relation_by_types(type1, type2)
            
            if suggested_relation:
                # 添加关系
                if not any(r['name'] == suggested_relation for r in enhanced_relations):
                    enhanced_relations.append({
                        'name': suggested_relation,
                        'type': 'inferred',
                        'attributes': {'auto_generated': True},
                        'confidence': 0.5,
                        'enhanced': True
                    })
                
                # 添加三元组
                enhanced_triples.append({
                    'head': entity1,
                    'relation': suggested_relation,
                    'tail': entity2,
                    'confidence': 0.5,
                    'enhanced': True
                })
                
                return {
                    'type': 'relation_completion',
                    'description': f'为{entity1}和{entity2}添加{suggested_relation}关系',
                    'entities': target_entities,
                    'relation': suggested_relation,
                    'confidence': 0.5
                }
        return None
    
    def _enhance_entity_types_and_attributes(self, 
                                           enhanced_entities: List[Dict[str, Any]], 
                                           original_text: str) -> List[Dict[str, Any]]:
        """增强实体类型和属性"""
        for entity in enhanced_entities:
            if entity['type'] == 'unknown':
                # 推断类型
                inferred_type = self._infer_entity_type(entity['name'], original_text)
                entity['type'] = inferred_type
                entity['enhanced'] = True
                
                # 推断属性
                inferred_attributes = self._infer_attributes_from_text(entity['name'], original_text)
                entity['attributes'].update(inferred_attributes)
        
        return enhanced_entities
    
    def _enhance_relations(self, 
                         enhanced_entities: List[Dict[str, Any]], 
                         original_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """增强关系 - 修复版，避免过度生成关系"""
        new_relations = []
        new_triples = []
        
        # 限制自动关系生成，只在特定条件下添加关系
        for i, entity1 in enumerate(enhanced_entities):
            for entity2 in enhanced_entities[i+1:]:
                # 更严格的相关性检查
                if self._are_entities_strongly_related(entity1['name'], entity2['name'], original_text):
                    # 根据实体类型推断关系，但不使用通用的"相关"
                    suggested_relation = self._suggest_specific_relation_by_types(entity1['type'], entity2['type'])
                    
                    # 只有找到明确的关系类型才添加
                    if suggested_relation and suggested_relation != '相关':
                        # 检查是否已存在类似关系
                        if not self._relation_already_exists(entity1['name'], entity2['name'], suggested_relation, new_triples):
                            relation_dict = {
                                'name': suggested_relation,
                                'type': 'inferred',
                                'attributes': {'auto_generated': True, 'source': 'type_inference'},
                                'confidence': 0.6,  # 提高置信度要求
                                'enhanced': True
                            }
                            
                            triple_dict = {
                                'head': entity1['name'],
                                'relation': suggested_relation,
                                'tail': entity2['name'],
                                'confidence': 0.6,
                                'enhanced': True
                            }
                            
                            # 检查是否已存在相同关系
                            if not any(r['name'] == suggested_relation for r in new_relations):
                                new_relations.append(relation_dict)
                            
                            new_triples.append(triple_dict)
        
        return new_relations, new_triples
    
    def _are_entities_strongly_related(self, entity1: str, entity2: str, text: str) -> bool:
        """检查两个实体在文本中是否强相关 - 更严格的条件"""
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return False
        
        # 更严格的距离要求：必须在同一个句子内（距离小于50个字符）
        distance = abs(pos1 - pos2)
        if distance > 50:
            return False
        
        # 检查是否在同一个语义单元内（通过标点符号判断）
        start = min(pos1, pos2)
        end = max(pos1 + len(entity1), pos2 + len(entity2))
        context = text[max(0, start-20):min(len(text), end+20)]
        
        # 如果上下文中有明确的连接词，才认为强相关
        connection_words = ['的', '和', '与', '在', '于', '是', '为', '由', '从', '到', '向', '对']
        return any(word in context for word in connection_words)
    
    def _suggest_specific_relation_by_types(self, type1: str, type2: str) -> Optional[str]:
        """根据实体类型建议具体关系 - 不返回通用的"相关"关系"""
        key1 = f"{type1}-{type2}"
        key2 = f"{type2}-{type1}"
        
        if key1 in self.relation_templates:
            return self.relation_templates[key1][0]  # 返回第一个模板
        elif key2 in self.relation_templates:
            return self.relation_templates[key2][0]
        
        # 不返回通用关系，只返回None
        return None
    
    def _relation_already_exists(self, entity1: str, entity2: str, relation: str, existing_triples: List[Dict[str, Any]]) -> bool:
        """检查关系是否已存在"""
        for triple in existing_triples:
            if ((triple['head'] == entity1 and triple['tail'] == entity2) or 
                (triple['head'] == entity2 and triple['tail'] == entity1)):
                return True
        return False
    
    def _infer_entity_type(self, entity_name: str, text: str) -> str:
        """推断实体类型"""
        # 基于关键词匹配
        for entity_type, keywords in self.entity_type_keywords.items():
            if any(keyword in entity_name or keyword in text for keyword in keywords):
                return entity_type
        
        # 基于上下文分析
        context = self._get_entity_context(entity_name, text)
        
        # 人物特征
        if any(keyword in context for keyword in ['先生', '女士', '老师', '教授', '博士', '工程师']):
            return 'Person'
        
        # 组织特征
        if any(keyword in context for keyword in ['公司', '大学', '学院', '研究所', '部门']):
            return 'Organization'
        
        # 地点特征
        if any(keyword in context for keyword in ['位于', '总部', '地址', '市', '省']):
            return 'Location'
        
        # 概念特征
        if any(keyword in context for keyword in ['理论', '方法', '概念', '思想', '原理']):
            return 'Concept'
        
        return 'Entity'  # 默认类型
    
    def _infer_attributes_from_text(self, entity_name: str, text: str) -> Dict[str, Any]:
        """从文本中推断属性"""
        attributes = {}
        context = self._get_entity_context(entity_name, text, window=100)
        
        # 时间属性
        time_pattern = r'(\d{4})年'
        time_matches = re.findall(time_pattern, context)
        if time_matches:
            attributes['相关年份'] = time_matches[0]
        
        # 地点属性
        location_pattern = r'([一-龥]{2,}[市县区镇村省州])'
        location_matches = re.findall(location_pattern, context)
        if location_matches:
            attributes['相关地点'] = location_matches[0]
        
        # 职业/角色属性
        role_pattern = r'(教授|博士|工程师|经理|总监|CEO|董事长|老师|研究员)'
        role_matches = re.findall(role_pattern, context)
        if role_matches:
            attributes['角色'] = role_matches[0]
        
        # 领域属性
        domain_pattern = r'(计算机|人工智能|机器学习|数据科学|软件工程|管理|金融|医疗|教育)'
        domain_matches = re.findall(domain_pattern, context)
        if domain_matches:
            attributes['领域'] = domain_matches[0]
        
        return attributes
    
    def _get_entity_context(self, entity_name: str, text: str, window: int = 50) -> str:
        """获取实体的上下文"""
        start = max(0, text.find(entity_name) - window)
        end = min(len(text), text.find(entity_name) + len(entity_name) + window)
        return text[start:end]
    
    def _are_entities_related_in_text(self, entity1: str, entity2: str, text: str) -> bool:
        """检查两个实体在文本中是否相关"""
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return False
        
        # 如果距离小于100个字符，认为相关
        return abs(pos1 - pos2) < 100
    
    def _suggest_relation_by_types(self, type1: str, type2: str) -> Optional[str]:
        """根据实体类型建议关系"""
        key1 = f"{type1}-{type2}"
        key2 = f"{type2}-{type1}"
        
        if key1 in self.relation_templates:
            return self.relation_templates[key1][0]  # 返回第一个模板
        elif key2 in self.relation_templates:
            return self.relation_templates[key2][0]
        
        # 通用关系
        return '相关'
    
    def _get_entity_type(self, entity_name: str, enhanced_entities: List[Dict[str, Any]]) -> str:
        """获取实体类型"""
        for entity in enhanced_entities:
            if entity['name'] == entity_name:
                return entity['type']
        return 'unknown'
    
    def _generate_enhancement_summary(self, 
                                    original_entities: List[str],
                                    original_relations: List[str],
                                    enhanced_entities: List[Dict[str, Any]],
                                    enhanced_relations: List[Dict[str, Any]],
                                    applied_enhancements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成增强摘要"""
        enhanced_entity_count = len([e for e in enhanced_entities if e.get('enhanced', False)])
        enhanced_relation_count = len([r for r in enhanced_relations if r.get('enhanced', False)])
        
        return {
            'original_stats': {
                'entities': len(original_entities),
                'relations': len(original_relations)
            },
            'enhanced_stats': {
                'entities': len(enhanced_entities),
                'relations': len(enhanced_relations),
                'enhanced_entities': enhanced_entity_count,
                'enhanced_relations': enhanced_relation_count
            },
            'improvements': {
                'new_entities': len(enhanced_entities) - len(original_entities),
                'new_relations': len(enhanced_relations) - len(original_relations),
                'enhanced_entities': enhanced_entity_count,
                'enhanced_relations': enhanced_relation_count
            },
            'applied_enhancements': len(applied_enhancements),
            'enhancement_types': list(set(e['type'] for e in applied_enhancements))
        }

# 使用示例
async def main():
    """使用示例"""
    executor = EnhancementExecutor()
    
    # 示例数据
    sample_text = "张三是阿里巴巴的高级工程师，负责人工智能项目。他毕业于清华大学。"
    sample_entities = ["张三", "阿里巴巴", "清华大学", "人工智能"]
    sample_relations = ["工作于", "毕业于", "负责"]
    
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
        sample_text, sample_entities, sample_relations, [], analysis_result
    )
    
    print("增强结果:")
    print(f"原始实体数: {len(result.original_entities)}")
    print(f"增强后实体数: {len(result.enhanced_entities)}")
    print(f"原始关系数: {len(result.original_relations)}")
    print(f"增强后关系数: {len(result.enhanced_relations)}")
    print(f"应用的增强数: {len(result.applied_enhancements)}")

if __name__ == "__main__":
    asyncio.run(main()) 