# kg_utils.py

import json
import re
import asyncio
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


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self):
        self.entities = set()
        self.relations = set()
        self.triples = []
        self.entity_types = {}  # 实体类型映射

    async def build_graph(self, data: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        构建知识图谱

        Args:
            data: 输入数据
            use_llm: 是否使用LLM进行图谱生成

        Returns:
            知识图谱构建结果
        """
        # 提取实体
        entities = self._extract_entities(data)

        # 提取关系
        relations = self._extract_relations(data)

        # 生成三元组
        if use_llm:
            triples = await self._generate_triples_with_llm(data, entities, relations)
        else:
            triples = self._generate_triples_rule_based(data, entities, relations)

        # 计算置信度
        confidence_scores = [triple.confidence for triple in triples]

        # 更新内部状态
        self.entities.update(entities)
        self.relations.update(relations)
        self.triples.extend(triples)

        return {
            "entities": list(entities),
            "relations": list(relations),
            "triples": triples,
            "confidence_scores": confidence_scores
        }

    def _extract_entities(self, data: str) -> List[str]:
        """提取实体"""
        entities = []

        # 人名 - 更精确的匹配，避免过长匹配
        person_pattern = r'[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文陶欧阳司马上官诸葛][一-龥]{1,2}'
        persons = re.findall(person_pattern, data)
        for person in persons:
            # 过滤掉明显不是人名的匹配，并去除关系词
            clean_person = person
            # 移除常见的关系词后缀
            for suffix in ['在', '是', '去', '来', '到']:
                if clean_person.endswith(suffix):
                    clean_person = clean_person[:-1]

            if len(clean_person) >= 2 and not any(word in clean_person for word in ['大学', '学院', '公司', '企业']):
                entities.append(clean_person)
                self.entity_types[clean_person] = "Person"

        # 地名
        place_pattern = r'[一-龥]{2,}[市县区镇村]|[一-龥]{2,}[省州]|北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆|天津|青岛|大连|厦门|苏州|无锡|宁波|合肥|福州|济南|长沙|郑州|石家庄|太原|呼和浩特|沈阳|长春|哈尔滨|南昌|贵阳|昆明|拉萨|西宁|银川|乌鲁木齐|香港|澳门|台北|巴黎|伦敦|纽约|东京|首尔|新加坡|悉尼|多伦多|柏林|罗马|马德里|阿姆斯特丹|布鲁塞尔|维也纳|苏黎世|斯德哥尔摩|哥本哈根|赫尔辛基|奥斯陆|华盛顿|洛杉矶|芝加哥|休斯顿|费城|凤凰城|圣安东尼奥|圣地亚哥|达拉斯|圣何塞'
        places = re.findall(place_pattern, data)
        for place in places:
            entities.append(place)
            self.entity_types[place] = "Place"

        # 组织机构 - 更精确的匹配
        org_pattern = r'[一-龥]{2,}(?:大学|学院|公司|企业|集团|组织|机构|医院|银行)'
        orgs = re.findall(org_pattern, data)
        for org in orgs:
            entities.append(org)
            self.entity_types[org] = "Organization"

        # 职位/角色
        role_pattern = r'CEO|总裁|总经理|经理|主任|主管|董事长|副总|部长|科长|教授|博士|硕士|学士|工程师|设计师|分析师|顾问|学生|老师|医生|护士|律师|会计师'
        roles = re.findall(role_pattern, data)
        for role in roles:
            entities.append(role)
            self.entity_types[role] = "Role"

        # 去重并过滤
        unique_entities = []
        seen = set()
        for entity in entities:
            if entity not in seen and len(entity.strip()) > 0:
                unique_entities.append(entity)
                seen.add(entity)

        return unique_entities

    def _extract_relations(self, data: str) -> List[str]:
        """提取关系"""
        relations = []

        # 定义关系模式
        relation_patterns = {
            "位于": r"位于|在.*[市县区省州]",
            "在": r"在(?![一-龥]*[工作学习])",  # "在"但不是"在...工作"或"在...学习"
            "担任": r"担任|是.*[CEO总裁经理主任]",
            "工作于": r"工作于|在.*[公司企业].*工作",
            "属于": r"属于|隶属于",
            "来自": r"来自|出生于",
            "前往": r"去|前往|到达",
            "拥有": r"拥有|具有|有",
            "管理": r"管理|负责|主管",
            "学习于": r"毕业于|就读于|学习于",
            "总部位于": r"总部.*位于|总部.*在",
            "是": r"是.*[学生老师医生护士律师会计师]|是.*[大学学院].*[学生]",
            "就读于": r"是.*[大学学院].*学生|在.*[大学学院].*学习|[大学学院].*学生"
        }

        for relation, pattern in relation_patterns.items():
            if re.search(pattern, data):
                relations.append(relation)

        return relations

    def _generate_triples_rule_based(self, data: str, entities: List[str], relations: List[str]) -> List[Triple]:
        """基于规则生成三元组"""
        triples = []

        # 简化的规则匹配
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # 检查两个实体之间是否存在关系
                    relation = self._find_relation_between_entities(data, entity1, entity2)
                    if relation:
                        confidence = self._calculate_triple_confidence(data, entity1, relation, entity2)
                        triple = Triple(entity1, relation, entity2, confidence)
                        triples.append(triple)

        return triples

    async def _generate_triples_with_llm(self, data: str, entities: List[str], relations: List[str]) -> List[Triple]:
        """使用LLM生成三元组 - 模拟实现"""
        triples = []

        # 首先使用规则方法作为基础
        rule_based_triples = self._generate_triples_rule_based(data, entities, relations)
        triples.extend(rule_based_triples)

        # 模拟LLM增强的三元组生成
        # 在实际实现中，这里会调用真正的LLM API
        enhanced_triples = await self._simulate_llm_enhancement(data, entities, relations)
        triples.extend(enhanced_triples)

        # 去重和合并
        triples = self._merge_duplicate_triples(triples)

        return triples

    async def _simulate_llm_enhancement(self, data: str, entities: List[str], relations: List[str]) -> List[Triple]:
        """模拟LLM增强的三元组生成"""
        enhanced_triples = []

        # 模拟LLM能够识别的更复杂的关系
        if "CEO" in data and len(entities) >= 2:
            # 假设LLM能够推断出CEO与公司的关系
            for entity in entities:
                if self.entity_types.get(entity) == "Person":
                    for other_entity in entities:
                        if self.entity_types.get(other_entity) == "Organization":
                            triple = Triple(entity, "担任CEO", other_entity, 0.85)
                            enhanced_triples.append(triple)

        # 移除自动地理关系推理，避免添加不相关的信息
        # 只基于输入文本中明确存在的关系进行提取

        return enhanced_triples

    def _find_relation_between_entities(self, data: str, entity1: str, entity2: str) -> Optional[str]:
        """查找两个实体之间的关系"""
        # 在文本中查找实体之间的关系
        entity1_pos = data.find(entity1)
        entity2_pos = data.find(entity2)

        if entity1_pos == -1 or entity2_pos == -1:
            return None

        # 获取两个实体之间的文本
        start_pos = min(entity1_pos, entity2_pos)
        end_pos = max(entity1_pos + len(entity1), entity2_pos + len(entity2))
        between_text = data[start_pos:end_pos]

        # 检查关系词
        relation_keywords = {
            "位于": ["位于"],
            "在": ["在"],
            "担任": ["担任"],
            "工作于": ["工作于", "在.*工作"],
            "前往": ["去", "前往"],
            "来自": ["来自", "出生于"],
            "是": ["是"],
            "就读于": ["就读于", "学习于"]
        }

        for relation, keywords in relation_keywords.items():
            for keyword in keywords:
                if re.search(keyword, between_text):
                    return relation

        return None

    def _calculate_triple_confidence(self, data: str, head: str, relation: str, tail: str) -> float:
        """计算三元组的置信度"""
        base_confidence = 0.8

        # 根据实体类型调整置信度
        head_type = self.entity_types.get(head, "Unknown")
        tail_type = self.entity_types.get(tail, "Unknown")

        # 合理的实体类型组合会提高置信度
        reasonable_combinations = {
            ("Person", "Role"): 0.1,
            ("Person", "Organization"): 0.1,
            ("Person", "Place"): 0.05,
            ("Organization", "Place"): 0.1,
        }

        type_pair = (head_type, tail_type)
        if type_pair in reasonable_combinations:
            base_confidence += reasonable_combinations[type_pair]

        # 根据关系词在文本中的明确程度调整
        if relation in ["位于", "担任"]:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _merge_duplicate_triples(self, triples: List[Triple]) -> List[Triple]:
        """合并重复的三元组"""
        triple_dict = {}

        for triple in triples:
            key = (triple.head, triple.relation, triple.tail)
            if key in triple_dict:
                # 保留置信度更高的三元组
                if triple.confidence > triple_dict[key].confidence:
                    triple_dict[key] = triple
            else:
                triple_dict[key] = triple

        return list(triple_dict.values())

    def get_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
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
                "entities": [{"id": entity, "type": self.entity_types.get(entity, "Unknown")} for entity in self.entities],
                "relations": list(self.relations),
                "triples": [triple.to_dict() for triple in self.triples]
            }
            return json.dumps(graph_data, ensure_ascii=False, indent=2)

        elif format_type == "turtle":
            # 简化的Turtle格式导出
            turtle_lines = ["@prefix : <http://example.org/> ."]
            for triple in self.triples:
                turtle_lines.append(f":{triple.head} :{triple.relation} :{triple.tail} .")
            return "\n".join(turtle_lines)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def clear_graph(self):
        """清空知识图谱"""
        self.entities.clear()
        self.relations.clear()
        self.triples.clear()
        self.entity_types.clear()
