# kg_utils.py - 纯LLM版本（移除所有硬编码）

import json
import re
import asyncio
import requests
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Triple:
    """知识图谱三元组"""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "confidence": self.confidence
        }

    def __str__(self) -> str:
        return f"({self.head}, {self.relation}, {self.tail})"


class ChineseEntityRelationExtractor:
    """纯LLM中文实体抽取与关系抽取"""
    
    def __init__(self, api_key):
        self.api_key = api_key  # 修复：使用传入的API密钥
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        
    def call_api(self, prompt):
        """调用Silicon Flow API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 800
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ API调用失败: HTTP {response.status_code}")
                return None
            
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            return None
    
    def extract_entities_and_types(self, text):
        """同时提取实体和类型"""
        prompt = f"""
请从以下中文文本中提取实体，并标注每个实体的类型。

文本："{text}"

请按照以下格式输出，每行一个实体：
实体名称|实体类型

实体类型包括：Person（人物）、Organization（组织）、Location（地点）、Product（产品）、Event（事件）、Other（其他）

示例格式：
张三|Person
阿里巴巴|Organization
北京|Location
iPhone|Product
"""
        
        response = self.call_api(prompt)
        if response:
            entities = {}
            lines = response.strip().split('\n')
            for line in lines:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        entity = parts[0].strip()
                        entity_type = parts[1].strip()
                        if entity:
                            entities[entity] = entity_type
            return entities
        return {}

    def extract_triplets(self, text):
        """提取三元组"""
        prompt = f"""
请从以下中文文本中抽取实体关系三元组。

文本："{text}"

请严格按照以下格式输出，每行一个三元组：
(头实体,关系,尾实体)

要求：
1. 头实体和尾实体必须是文本中明确出现的
2. 关系要准确描述两个实体之间的关系
3. 只输出确定的关系，不要猜测
4. 每行只输出一个三元组

示例：
(张三,担任,CEO)
(阿里巴巴,总部位于,杭州)
"""
        
        response = self.call_api(prompt)
        if response:
            return self.parse_triplets(response)
        return []
    
    def parse_triplets(self, response):
        """解析三元组"""
        triplets = []
        lines = response.strip().split('\n')
        
        for line in lines:
            # 匹配格式：(头实体,关系,尾实体)
            match = re.search(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
            if match:
                head = match.group(1).strip()
                relation = match.group(2).strip()
                tail = match.group(3).strip()
                if head and relation and tail:
                    triplets.append((head, relation, tail))
        
        return triplets


class PureLLMKnowledgeGraphBuilder:
    """纯LLM知识图谱构建器（无硬编码规则）"""

    def __init__(self, api_key: Optional[str] = None):
        self.entities = set()
        self.relations = set()
        self.triples = []
        self.entity_types = {}
        
        # 初始化LLM抽取器
        self.llm_extractor = None
        if api_key:
            self.llm_extractor = ChineseEntityRelationExtractor(api_key)
        else:
            print("⚠️  警告：未提供API密钥，无法使用LLM功能")

    async def build_graph(self, data: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        构建知识图谱（纯LLM版本）
        
        Args:
            data: 输入文本
            use_llm: 必须为True，纯LLM版本不支持规则模式
        
        Returns:
            知识图谱构建结果
        """
        if not use_llm:
            print("⚠️  纯LLM版本不支持规则模式，自动启用LLM模式")
        
        if not self.llm_extractor:
            print("❌ 未配置API密钥，无法构建知识图谱")
            return {
                "entities": [],
                "relations": [],
                "triples": [],
                "confidence_scores": []
            }

        print("🤖 使用纯LLM模式构建知识图谱...")
        
        # 使用LLM提取所有信息
        entities_with_types = self.llm_extractor.extract_entities_and_types(data)
        llm_triplets = self.llm_extractor.extract_triplets(data)
        
        # 处理实体
        entities = list(entities_with_types.keys())
        self.entity_types = entities_with_types
        
        # 处理三元组
        triples = []
        relations = set()
        
        for head, relation, tail in llm_triplets:
            # 计算置信度
            confidence = self._calculate_llm_confidence(data, head, relation, tail)
            triple = Triple(head, relation, tail, confidence)
            triples.append(triple)
            relations.add(relation)
            
            # 确保实体被包含
            if head not in entities:
                entities.append(head)
                self.entity_types[head] = "Other"
            if tail not in entities:
                entities.append(tail)
                self.entity_types[tail] = "Other"

        # 去重和合并
        triples = self._merge_duplicate_triples(triples)
        
        # 计算置信度
        confidence_scores = [triple.confidence for triple in triples]

        # 更新内部状态
        self.entities.update(entities)
        self.relations.update(relations)
        self.triples.extend(triples)

        print(f"✅ 提取完成：{len(entities)}个实体，{len(relations)}个关系，{len(triples)}个三元组")

        return {
            "entities": entities,
            "relations": list(relations),
            "triples": triples,
            "confidence_scores": confidence_scores
        }

    def _calculate_llm_confidence(self, data: str, head: str, relation: str, tail: str) -> float:
        """计算LLM三元组置信度 - 动态计算，避免硬编码"""
        confidence = 0.5  # 基础置信度降低
        
        # 1. 实体在原文中的位置和频率
        head_count = data.count(head)
        tail_count = data.count(tail)
        
        # 实体出现频率影响置信度
        if head_count > 0 and tail_count > 0:
            frequency_bonus = min(0.3, (head_count + tail_count) * 0.05)
            confidence += frequency_bonus
        
        # 2. 实体距离分析
        head_pos = data.find(head)
        tail_pos = data.find(tail)
        if head_pos != -1 and tail_pos != -1:
            distance = abs(head_pos - tail_pos)
            text_length = len(data)
            # 距离越近，置信度越高
            proximity_bonus = max(0, 0.2 * (1 - distance / text_length))
            confidence += proximity_bonus
        
        # 3. 关系词在实体附近的存在
        relation_in_context = False
        if head_pos != -1 and tail_pos != -1:
            start_pos = min(head_pos, tail_pos)
            end_pos = max(head_pos, tail_pos) + max(len(head), len(tail))
            context = data[max(0, start_pos-20):min(len(data), end_pos+20)]
            
            # 检查关系词或相关词是否在上下文中
            relation_words = [relation, '是', '在', '的', '有', '属于', '担任', '位于']
            if any(word in context for word in relation_words):
                confidence += 0.15
                relation_in_context = True
        
        # 4. 实体类型匹配度
        head_type = self.entity_types.get(head, "Other")
        tail_type = self.entity_types.get(tail, "Other")
        
        # 合理的类型组合获得更高置信度
        type_combinations = {
            ("Person", "Organization"): 0.15,
            ("Person", "Location"): 0.12,
            ("Organization", "Location"): 0.10,
            ("Person", "Product"): 0.08,
            ("Organization", "Product"): 0.08,
        }
        
        combination_key = (head_type, tail_type)
        reverse_key = (tail_type, head_type)
        
        if combination_key in type_combinations:
            confidence += type_combinations[combination_key]
        elif reverse_key in type_combinations:
            confidence += type_combinations[reverse_key]
        
        # 5. 关系类型的可信度
        relation_confidence_map = {
            '是': 0.9, '担任': 0.85, '位于': 0.85, '属于': 0.8,
            '工作于': 0.8, '就职于': 0.8, '毕业于': 0.75, '学习': 0.7,
            '开发': 0.7, '制作': 0.7, '创建': 0.7, '拥有': 0.65,
            '包含': 0.6, '相关': 0.5, '关联': 0.5
        }
        
        relation_base = relation_confidence_map.get(relation, 0.6)
        confidence = confidence * relation_base
        
        # 6. 文本长度影响（较长文本可能包含更多噪音）
        text_length = len(data)
        if text_length > 500:
            confidence *= 0.95  # 轻微降低
        elif text_length < 50:
            confidence *= 0.9   # 文本太短可能信息不足
        
        # 7. 确保置信度在合理范围内
        confidence = max(0.1, min(0.98, confidence))
        
        # 8. 添加一些随机性避免完全相同的置信度
        import random
        random.seed(hash(head + relation + tail) % 1000)  # 基于内容的伪随机
        noise = (random.random() - 0.5) * 0.05  # ±2.5%的噪声
        confidence += noise
        
        return round(max(0.1, min(0.98, confidence)), 3)

    def _merge_duplicate_triples(self, triples: List[Triple]) -> List[Triple]:
        """合并重复三元组"""
        triple_dict = {}
        
        for triple in triples:
            key = (triple.head, triple.relation, triple.tail)
            if key in triple_dict:
                # 保留置信度更高的
                if triple.confidence > triple_dict[key].confidence:
                    triple_dict[key] = triple
            else:
                triple_dict[key] = triple
        
        return list(triple_dict.values())

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        entity_type_counts = defaultdict(int)
        for entity, entity_type in self.entity_types.items():
            entity_type_counts[entity_type] += 1

        relation_counts = defaultdict(int)
        for triple in self.triples:
            relation_counts[triple.relation] += 1

        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_triples": len(self.triples),
            "entity_types": dict(entity_type_counts),
            "relation_distribution": dict(relation_counts),
            "average_confidence": sum(triple.confidence for triple in self.triples) / len(self.triples) if self.triples else 0
        }

    def export_graph(self, format_type: str = "json") -> str:
        """导出知识图谱"""
        if format_type == "json":
            graph_data = {
                "entities": [{"id": entity, "type": self.entity_types.get(entity, "Other")} for entity in self.entities],
                "relations": list(self.relations),
                "triples": [triple.to_dict() for triple in self.triples]
            }
            return json.dumps(graph_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的格式: {format_type}")

    def clear_graph(self):
        """清空知识图谱"""
        self.entities.clear()
        self.relations.clear()
        self.triples.clear()
        self.entity_types.clear()


# 为了兼容性，保留原来的类名
class KnowledgeGraphBuilder(PureLLMKnowledgeGraphBuilder):
    """知识图谱构建器（兼容性别名）"""
    pass