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
        """计算LLM三元组置信度"""
        base_confidence = 0.85
        
        # 检查实体是否在原文中
        head_in_text = head in data
        tail_in_text = tail in data
        
        if head_in_text and tail_in_text:
            base_confidence += 0.1
        elif head_in_text or tail_in_text:
            base_confidence += 0.05
        
        # 根据实体类型调整
        head_type = self.entity_types.get(head, "Other")
        tail_type = self.entity_types.get(tail, "Other")
        
        # 合理的类型组合
        good_combinations = [
            ("Person", "Organization"),
            ("Person", "Location"),
            ("Organization", "Location"),
            ("Person", "Product")
        ]
        
        if (head_type, tail_type) in good_combinations or (tail_type, head_type) in good_combinations:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)

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