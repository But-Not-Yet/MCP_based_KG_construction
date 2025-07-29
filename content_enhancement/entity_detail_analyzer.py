# entity_detail_analyzer.py



#todo 周末任务：1. 实现逻辑分析（紫色模块）；2. 预留质量分析部分方便将调研结果插入；3. 数据寻找；4. 图谱质量分析到内容增强的循环迭代的实现


import re
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# 引入 LLMClient（若可用）
try:
    from .llm_client import LLMClient
except ImportError:
    LLMClient = None  # type: ignore


class AttributeType(Enum):
    """属性类型枚举"""
    BASIC = "基础属性"        # 姓名、类型等
    TEMPORAL = "时间属性"     # 出生日期、成立时间等
    SPATIAL = "空间属性"      # 地址、位置等
    RELATIONAL = "关系属性"   # 职务、从属关系等
    DESCRIPTIVE = "描述属性"  # 特征、规模等

@dataclass
class AttributeTemplate:
    """属性模板"""
    entity_type: str
    required_attributes: List[str]
    optional_attributes: List[str]
    attribute_patterns: Dict[str, str]

@dataclass
class AttributeGap:
    """属性缺失记录"""
    entity: str
    entity_type: str
    missing_attribute: str
    attribute_type: AttributeType
    confidence: float
    suggested_sources: List[str]

# ---------------------------------------------
# 偏细节分析器主类
# ---------------------------------------------


class EntityDetailAnalyzer:
    """偏细节分析器 - 专注于单实体和小规模组合分析"""

    def __init__(self, llm_client: Optional["LLMClient"] = None):
        """初始化

        Args:
            llm_client: 可选的大语言模型客户端，若提供则优先从 LLM 获取属性模板。
        """
        self.llm_client = llm_client

        # 尝试从 LLM 获取属性模板，失败则回退到硬编码模板
        if self.llm_client is not None:
            llm_templates = self._load_attribute_templates_from_llm()
            self.attribute_templates = (
                llm_templates if llm_templates else self._init_attribute_templates()
            )
        else:
            self.attribute_templates = self._init_attribute_templates()

        # TODO: 后续可将逻辑规则也迁移至 LLM
        self.logical_rules = self._init_logical_rules()

    # ------------------------------------------------------------------
    # LLM 动态加载
    # ------------------------------------------------------------------

    def _load_attribute_templates_from_llm(self) -> Dict[str, "AttributeTemplate"]:
        """尝试从 LLM 获取属性模板，如失败返回空 dict"""
        templates: Dict[str, AttributeTemplate] = {}

        if self.llm_client is None:
            return templates

        # 可根据场景动态决定需要哪些实体类型；此处采用默认列表
        default_entity_types = ["Person", "Organization", "Location", "Product"]
        for etype in default_entity_types:
            try:
                tmpl_dict = self.llm_client.get_attribute_template(etype)
                if tmpl_dict:
                    templates[etype] = AttributeTemplate(
                        entity_type=etype,
                        required_attributes=tmpl_dict.get("required_attributes", []),
                        optional_attributes=tmpl_dict.get("optional_attributes", []),
                        attribute_patterns=tmpl_dict.get("attribute_patterns", {}),
                    )
            except Exception as exc:  # pylint: disable=broad-except
                # 若某一实体类型获取失败，记录日志并跳过
                print(f"⚠️  从 LLM 获取 {etype} 属性模板失败: {exc}")

        return templates

    def _init_attribute_templates(self) -> Dict[str, AttributeTemplate]:
        """初始化实体属性模板"""
        return {
            "Person": AttributeTemplate(
                entity_type="Person",
                required_attributes=["姓名", "性别"],
                optional_attributes=["年龄", "职业", "出生地", "教育背景", "工作单位"],

                attribute_patterns={
                    "年龄": r"(\d{1,3})[岁|周岁]",
                    "职业": r"(CEO|总裁|经理|主任|教授|医生|工程师)",
                    "出生地": r"[一-龥]{2,}[市县区镇村省州]",
                    "教育背景": r"[一-龥]{2,}[大学学院]"
                }
            ),
            "Organization": AttributeTemplate(
                entity_type="Organization", 
                required_attributes=["名称", "类型"],
                optional_attributes=["成立时间", "总部地址", "规模", "行业", "法人代表"],
                attribute_patterns={
                    "成立时间": r"(\d{4})年",
                    "总部地址": r"[一-龥]{2,}[市县区镇村省州]",
                    "规模": r"(\d+)[人|员工]",
                    "行业": r"(科技|金融|教育|医疗|制造)[行业|领域]"
                }
            ),
            "Location": AttributeTemplate(
                entity_type="Location",
                required_attributes=["名称", "地理类型"],
                optional_attributes=["所属区域", "人口", "面积", "特色产业"],
                attribute_patterns={
                    "人口": r"(\d+)[万|千]人",
                    "面积": r"(\d+)[平方公里|公顷]",
                    "所属区域": r"[一-龥]{2,}[省州市县区]"
                }
            ),
            "Product": AttributeTemplate(
                entity_type="Product",
                required_attributes=["名称", "类别"],
                optional_attributes=["价格", "生产商", "发布时间", "功能特性"],
                attribute_patterns={
                    "价格": r"(\d+)[元|美元|USD]",
                    "发布时间": r"(\d{4})年(\d{1,2})月",
                    "生产商": r"[一-龥]{2,}[公司企业集团]"
                }
            )
        }
    
    def _init_logical_rules(self) -> List[Dict[str, Any]]:
        """初始化逻辑检查规则"""
        return [
            {
                "rule_name": "人物职业一致性",
                "condition": lambda entity, attrs: entity["type"] == "Person" and "职业" in attrs and "工作单位" in attrs,
                "check": lambda attrs: self._check_profession_consistency(attrs["职业"], attrs["工作单位"]),
                "error_message": "职业与工作单位不匹配"
            },
            {
                "rule_name": "地理位置层次性",
                "condition": lambda entity, attrs: entity["type"] == "Location" and "所属区域" in attrs,
                "check": lambda attrs: self._check_geographic_hierarchy(entity["name"], attrs["所属区域"]),
                "error_message": "地理位置层次关系错误"
            },
            {
                "rule_name": "组织成立时间合理性",
                "condition": lambda entity, attrs: entity["type"] == "Organization" and "成立时间" in attrs,
                "check": lambda attrs: self._check_founding_time_validity(attrs["成立时间"]),
                "error_message": "成立时间不合理"
            }
        ]

    async def analyze_entity_details(self, entities: List[Dict], relations: List[Tuple], original_text: str) -> Dict[str, Any]:
        """
        执行偏细节分析
        
        Args:
            entities: 已抽取的实体列表 [{"name": "张三", "type": "Person"}, ...]
            relations: 已抽取的关系列表 [("张三", "工作于", "阿里巴巴"), ...]
            original_text: 原始文本
            
        Returns:
            分析结果
        """
        analysis_results = {
            "attribute_gaps": [],
            "logical_errors": [],
            "local_associations": [],
            "enhancement_suggestions": []
        }
        
        # 1. 逐个实体进行属性完整性分析
        for entity in entities:
            entity_analysis = await self._analyze_single_entity(entity, original_text, relations)
            analysis_results["attribute_gaps"].extend(entity_analysis["gaps"])
            analysis_results["logical_errors"].extend(entity_analysis["errors"])
        
        # 2. 小规模实体组合关联分析
        local_associations = await self._analyze_local_associations(entities, relations, original_text)
        analysis_results["local_associations"] = local_associations
        
        # 3. 生成增强建议
        enhancement_suggestions = self._generate_enhancement_suggestions(analysis_results)
        analysis_results["enhancement_suggestions"] = enhancement_suggestions
        
        return analysis_results

    async def _analyze_single_entity(self, entity: Dict, text: str, relations: List[Tuple]) -> Dict[str, Any]:
        """分析单个实体的详细信息"""
        entity_name = entity["name"]
        entity_type = entity["type"]
        
        result = {
            "gaps": [],
            "errors": [],
            "current_attributes": {}
        }
        
        # 1. 提取当前实体的所有属性
        current_attrs = self._extract_entity_attributes(entity_name, text, relations)
        result["current_attributes"] = current_attrs
        
        # 2. 检查属性完整性
        if entity_type in self.attribute_templates:
            template = self.attribute_templates[entity_type]
            gaps = self._check_attribute_completeness(entity_name, entity_type, current_attrs, template)
            result["gaps"] = gaps
        
        # 3. 逻辑一致性检查
        errors = self._check_logical_consistency(entity, current_attrs)
        result["errors"] = errors
        
        return result

    def _extract_entity_attributes(self, entity_name: str, text: str, relations: List[Tuple]) -> Dict[str, str]:
        """从文本和关系中提取实体属性"""
        attributes = {}
        
        # 1. 从关系中提取属性
        for subj, pred, obj in relations:
            if subj == entity_name:
                # 直接关系：张三 -> 工作于 -> 阿里巴巴
                if pred in ["工作于", "就职于", "任职于"]:
                    attributes["工作单位"] = obj
                elif pred in ["是", "担任"]:
                    attributes["职业"] = obj
                elif pred in ["位于", "在"]:
                    attributes["地址"] = obj
                elif pred in ["出生于", "来自"]:
                    attributes["出生地"] = obj
            elif obj == entity_name:
                # 反向关系：阿里巴巴 -> 雇佣 -> 张三
                if pred in ["雇佣", "聘用"]:
                    attributes["雇主"] = subj
        
        # 2. 从文本中用模式匹配提取属性
        entity_type = self._infer_entity_type(entity_name)
        if entity_type in self.attribute_templates:
            template = self.attribute_templates[entity_type]
            for attr_name, pattern in template.attribute_patterns.items():
                # 在实体附近查找属性
                entity_context = self._get_entity_context(entity_name, text, window=50)
                matches = re.findall(pattern, entity_context)
                if matches:
                    attributes[attr_name] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return attributes

    def _check_attribute_completeness(self, entity_name: str, entity_type: str, 
                                    current_attrs: Dict[str, str], 
                                    template: AttributeTemplate) -> List[AttributeGap]:
        """检查属性完整性"""
        gaps = []
        
        # 检查必需属性
        for required_attr in template.required_attributes:
            if required_attr not in current_attrs:
                gap = AttributeGap(
                    entity=entity_name,
                    entity_type=entity_type,
                    missing_attribute=required_attr,
                    attribute_type=self._classify_attribute_type(required_attr),
                    confidence=0.9,  # 必需属性缺失的置信度很高
                    suggested_sources=self._suggest_attribute_sources(entity_type, required_attr)
                )
                gaps.append(gap)
        
        # 检查重要的可选属性
        important_optional = self._get_important_optional_attributes(entity_type, current_attrs)
        for optional_attr in important_optional:
            if optional_attr not in current_attrs:
                gap = AttributeGap(
                    entity=entity_name,
                    entity_type=entity_type,
                    missing_attribute=optional_attr,
                    attribute_type=self._classify_attribute_type(optional_attr),
                    confidence=0.6,  # 可选属性缺失的置信度中等
                    suggested_sources=self._suggest_attribute_sources(entity_type, optional_attr)
                )
                gaps.append(gap)
        
        return gaps

    async def _analyze_local_associations(self, entities: List[Dict], 
                                        relations: List[Tuple], 
                                        text: str) -> List[Dict[str, Any]]:
        """分析小规模实体组合的局部关联"""
        associations = []
        
        # 1. 分析实体对之间的关联强度
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                association = self._analyze_entity_pair_association(
                    entity1, entity2, relations, text
                )
                if association["strength"] > 0.3:  # 只保留有意义的关联
                    associations.append(association)
        
        # 2. 分析三元实体组合
        if len(entities) >= 3:
            triangular_associations = self._analyze_triangular_associations(entities, relations)
            associations.extend(triangular_associations)
        
        return associations

    def _analyze_entity_pair_association(self, entity1: Dict, entity2: Dict, 
                                       relations: List[Tuple], text: str) -> Dict[str, Any]:
        """分析两个实体之间的关联强度"""
        name1, name2 = entity1["name"], entity2["name"]
        
        association = {
            "entities": [name1, name2],
            "strength": 0.0,
            "association_types": [],
            "evidence": []
        }
        
        # 1. 直接关系证据
        direct_relations = [(s, p, o) for s, p, o in relations 
                          if (s == name1 and o == name2) or (s == name2 and o == name1)]
        if direct_relations:
            association["strength"] += 0.8
            association["association_types"].append("直接关系")
            association["evidence"].extend([f"{r[0]}-{r[1]}-{r[2]}" for r in direct_relations])
        
        # 2. 共现频率证据
        cooccurrence_score = self._calculate_cooccurrence_score(name1, name2, text)
        association["strength"] += cooccurrence_score * 0.3
        if cooccurrence_score > 0.5:
            association["association_types"].append("高频共现")
        
        # 3. 类型相关性证据
        type_similarity = self._calculate_type_similarity(entity1["type"], entity2["type"])
        association["strength"] += type_similarity * 0.2
        if type_similarity > 0.7:
            association["association_types"].append("类型相关")
        
        return association

    def _generate_enhancement_suggestions(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成增强建议"""
        suggestions = []
        
        # 1. 基于属性缺失的建议
        for gap in analysis_results["attribute_gaps"]:
            suggestion = {
                "type": "属性补全",
                "target_entity": gap.entity,
                "action": f"补全{gap.missing_attribute}属性",
                "priority": "高" if gap.confidence > 0.8 else "中",
                "suggested_sources": gap.suggested_sources,
                "implementation": f"从{', '.join(gap.suggested_sources)}中查询{gap.entity}的{gap.missing_attribute}"
            }
            suggestions.append(suggestion)
        
        # 2. 基于逻辑错误的建议
        for error in analysis_results["logical_errors"]:
            suggestion = {
                "type": "逻辑修正",
                "target_entity": error["entity"],
                "action": f"修正{error['error_type']}",
                "priority": "高",
                "implementation": error["suggested_fix"]
            }
            suggestions.append(suggestion)
        
        # 3. 基于局部关联的建议
        for association in analysis_results["local_associations"]:
            if association["strength"] > 0.7 and "直接关系" not in association["association_types"]:
                suggestion = {
                    "type": "关系补全",
                    "target_entities": association["entities"],
                    "action": "添加潜在关系",
                    "priority": "中",
                    "implementation": f"基于{', '.join(association['association_types'])}添加{association['entities'][0]}与{association['entities'][1]}的关系"
                }
                suggestions.append(suggestion)
        
        return suggestions

    # 辅助方法
    def _get_entity_context(self, entity_name: str, text: str, window: int = 50) -> str:
        """获取实体的上下文文本"""
        start = max(0, text.find(entity_name) - window)
        end = min(len(text), text.find(entity_name) + len(entity_name) + window)
        return text[start:end]


    def _get_entity_score(self, entity_name: str,text: str)->str:
        """获取实体分数"""
        start = max(0, text.find(entity_name))
        end = min(len(text), text.find(entity_name) + len(entity_name))
        return text[start:end]



    def _classify_attribute_type(self, attribute_name: str) -> AttributeType:
        """分类属性类型"""
        if attribute_name in ["姓名", "名称", "类型"]:
            return AttributeType.BASIC
        elif attribute_name in ["出生日期", "成立时间", "发布时间"]:
            return AttributeType.TEMPORAL
        elif attribute_name in ["地址", "位置", "总部地址"]:
            return AttributeType.SPATIAL
        elif attribute_name in ["职业", "职务", "工作单位"]:
            return AttributeType.RELATIONAL
        else:
            return AttributeType.DESCRIPTIVE

    def _suggest_attribute_sources(self, entity_type: str, attribute_name: str) -> List[str]:
        """建议属性来源"""
        sources = ["知识库查询", "文本深度分析"]
        
        if entity_type == "Person":
            if attribute_name in ["年龄", "出生日期"]:
                sources.append("人物百科")
            elif attribute_name == "职业":
                sources.append("职业数据库")
        elif entity_type == "Organization":
            if attribute_name in ["成立时间", "总部地址"]:
                sources.append("企业信息库")
        
        return sources

    def _calculate_cooccurrence_score(self, entity1: str, entity2: str, text: str) -> float:
        """计算共现分数"""
        # 简化实现：计算两个实体在文本中的距离
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)




        if pos1 == -1 or pos2 == -1:
            return 0.0
        
        distance = abs(pos1 - pos2)
        max_distance = len(text)
        
        # 距离越近，共现分数越高
        score = max(0, 1 - (distance / max_distance))
        return score

    def _calculate_type_similarity(self, type1: str, type2: str) -> float:
        """计算类型相似度"""
        similarity_matrix = {
            ("Person", "Organization"): 0.6,  # 人和组织有较强关联
            ("Organization", "Location"): 0.5,  # 组织和地点有关联
            ("Person", "Product"): 0.4,
            ("Location", "Product"): 0.3
        }
        
        key = tuple(sorted([type1, type2]))
        return similarity_matrix.get(key, 0.1)  # 默认低相似度

    def _infer_entity_type(self, entity_name: str) -> str:
        """推断实体类型"""
        # 基于实体名称特征推断类型
        if any(keyword in entity_name for keyword in ["公司", "企业", "集团", "有限公司", "股份", "Co.", "Ltd.", "Inc."]):
            return "Organization"
        elif any(keyword in entity_name for keyword in ["市", "县", "区", "镇", "村", "省", "州", "路", "街", "广场"]):
            return "Location"
        elif any(keyword in entity_name for keyword in ["系统", "软件", "平台", "产品", "服务", "工具", "应用"]):
            return "Product"
        else:
            return "Person"  # 默认为人物

    def _get_important_optional_attributes(self, entity_type: str, current_attrs: Dict[str, str]) -> List[str]:
        """获取重要的可选属性"""
        if entity_type == "Person":
            if "工作单位" in current_attrs:
                return ["职业", "年龄"]
            else:
                return ["年龄", "出生地"]
        elif entity_type == "Organization":
            return ["成立时间", "总部地址", "行业"]
        elif entity_type == "Location":
            return ["所属区域", "人口"]
        elif entity_type == "Product":
            return ["价格", "生产商", "发布时间"]
        else:
            return []

    def _check_logical_consistency(self, entity: Dict, attributes: Dict[str, str]) -> List[Dict[str, Any]]:
        """检查逻辑一致性"""
        errors = []
        
        for rule in self.logical_rules:
            try:
                if rule["condition"](entity, attributes):
                    if not rule["check"](attributes):
                        errors.append({
                            "entity": entity["name"],
                            "error_type": rule["rule_name"],
                            "message": rule["error_message"],
                            "suggested_fix": f"检查并修正{entity['name']}的{rule['rule_name']}"
                        })
            except Exception as e:
                # 如果规则检查出错，记录但不中断
                continue
        
        return errors

    def _check_profession_consistency(self, profession: str, workplace: str) -> bool:
        """检查职业与工作单位的一致性"""
        # 简化实现，实际应用可以更复杂
        profession_workplace_map = {
            "教授": ["大学", "学院", "研究所"],
            "医生": ["医院", "诊所", "卫生院"],
            "工程师": ["公司", "企业", "研究院"],
            "CEO": ["公司", "企业", "集团"]
        }
        
        if profession in profession_workplace_map:
            expected_workplaces = profession_workplace_map[profession]
            return any(workplace_keyword in workplace for workplace_keyword in expected_workplaces)
        
        return True  # 未知职业默认一致

    def _check_geographic_hierarchy(self, location: str, parent_location: str) -> bool:
        """检查地理位置层次关系"""
        # 简化实现，实际应用需要地理数据库
        return len(parent_location) >= len(location)

    def _check_founding_time_validity(self, founding_time: str) -> bool:
        """检查成立时间合理性"""
        import datetime
        try:
            year = int(founding_time.replace("年", ""))
            current_year = datetime.datetime.now().year
            return 1800 <= year <= current_year
        except:
            return False

    def _analyze_triangular_associations(self, entities: List[Dict], relations: List[Tuple]) -> List[Dict[str, Any]]:
        """分析三角关系"""
        triangular_associations = []
        
        # 简化实现：查找三个实体间的关系
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                for k in range(j + 1, len(entities)):
                    entity1, entity2, entity3 = entities[i], entities[j], entities[k]
                    
                    # 检查是否存在三角关系
                    relations_count = 0
                    for rel in relations:
                        if (rel[0] in [entity1["name"], entity2["name"], entity3["name"]] and 
                            rel[2] in [entity1["name"], entity2["name"], entity3["name"]]):
                            relations_count += 1
                    
                    if relations_count >= 2:  # 至少有两个关系
                        triangular_associations.append({
                            "entities": [entity1["name"], entity2["name"], entity3["name"]],
                            "strength": min(1.0, relations_count / 3.0),
                            "association_types": ["三角关系"],
                            "evidence": [f"存在{relations_count}个相关关系"]
                        })
        
        return triangular_associations

# 使用示例
"""
# 在 knowledge_completion.py 中集成
from entity_detail_analyzer import EntityDetailAnalyzer

class KnowledgeCompletor:
    def __init__(self):
        self.detail_analyzer = EntityDetailAnalyzer()
    
    async def complete_knowledge(self, raw_data: str, quality_result: Dict[str, Any], 
                               entities: List[Dict], relations: List[Tuple]) -> Dict[str, Any]:
        # ... 现有代码 ...
        
        # 新增：偏细节分析
        detail_analysis = await self.detail_analyzer.analyze_entity_details(
            entities, relations, raw_data
        )
        
        # 基于细节分析结果进行补全
        enhanced_data = await self._apply_detail_enhancements(
            raw_data, detail_analysis
        )
        
        return {
            "enhanced_data": enhanced_data,
            "detail_analysis": detail_analysis,
            # ... 其他返回值 ...
        }
"""