# data_quality.py

import re
import asyncio
from typing import Dict, List, Any
import math


class DataQualityAssessor:
    """数据质量评估器"""
    
    def __init__(self, quality_threshold: float = 0.5):
        self.quality_threshold = quality_threshold
        
    async def assess_quality(self, data: str) -> Dict[str, Any]:
        """
        评估数据质量
        
        Args:
            data: 输入数据
            
        Returns:
            质量评估结果
        """
        # 计算各个维度的分数
        completeness = self._assess_completeness(data)
        consistency = self._assess_consistency(data)
        relevance = self._assess_relevance(data)
        
        # 计算综合质量分数 (基于论文公式3的简化版本)
        quality_score = self._calculate_quality_score(completeness, consistency, relevance)
        
        # 检测具体问题
        issues = self._detect_issues(data)
        
        # 生成建议
        recommendation = self._generate_recommendation(quality_score, issues)
        
        return {
            "quality_score": quality_score,
            "is_high_quality": quality_score >= self.quality_threshold,
            "completeness": completeness,
            "consistency": consistency,
            "relevance": relevance,
            "issues": issues,
            "recommendation": recommendation
        }
    
    def _assess_completeness(self, data: str) -> float:
        """评估数据完整性"""
        score = 0.0
        
        # 基本长度检查
        if len(data.strip()) > 10:
            score += 0.3
        
        # 实体识别 (简化版本)
        entities = self._extract_entities(data)
        if len(entities) >= 2:
            score += 0.3
        
        # 关系识别
        relations = self._extract_relations(data)
        if len(relations) >= 1:
            score += 0.4
        
        return min(score, 1.0)
    
    def _assess_consistency(self, data: str) -> float:
        """评估数据一致性"""
        score = 1.0
        
        # 检查明显的语义冲突
        conflicts = self._detect_semantic_conflicts(data)
        score -= len(conflicts) * 0.2
        
        # 检查格式一致性
        format_issues = self._detect_format_issues(data)
        score -= len(format_issues) * 0.1
        
        return max(score, 0.0)
    
    def _assess_relevance(self, data: str) -> float:
        """评估数据相关性"""
        # 简化的相关性评估
        score = 0.8  # 默认假设输入数据是相关的
        
        # 检查是否包含无意义内容
        if self._contains_meaningless_content(data):
            score -= 0.3
            
        return max(score, 0.0)
    
    def _calculate_quality_score(self, completeness: float, consistency: float, relevance: float) -> float:
        """计算综合质量分数"""
        # 加权平均 (可以根据需要调整权重)
        weights = {"completeness": 0.4, "consistency": 0.4, "relevance": 0.2}
        
        quality_score = (
            weights["completeness"] * completeness +
            weights["consistency"] * consistency +
            weights["relevance"] * relevance
        )
        
        return round(quality_score, 3)
    
    def _extract_entities(self, data: str) -> List[str]:
        """简化的实体提取"""
        # 使用正则表达式提取可能的实体
        entities = []
        
        # 人名模式 (中文)
        person_pattern = r'[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文][一-龥]{1,3}'
        persons = re.findall(person_pattern, data)
        entities.extend(persons)
        
        # 地名模式
        place_pattern = r'[一-龥]{2,}[市县区镇村]|[一-龥]{2,}[省州]'
        places = re.findall(place_pattern, data)
        entities.extend(places)
        
        # 公司/组织名称
        org_pattern = r'[一-龥]{2,}[公司企业集团组织机构]'
        orgs = re.findall(org_pattern, data)
        entities.extend(orgs)
        
        return list(set(entities))
    
    def _extract_relations(self, data: str) -> List[str]:
        """简化的关系提取"""
        relations = []
        
        # 常见关系词
        relation_patterns = [
            r'是', r'在', r'去', r'来自', r'属于', r'位于', r'担任', r'工作于',
            r'CEO', r'总裁', r'经理', r'主任', r'总部', r'分公司'
        ]
        
        for pattern in relation_patterns:
            if re.search(pattern, data):
                relations.append(pattern)
        
        return relations
    
    def _detect_semantic_conflicts(self, data: str) -> List[str]:
        """检测语义冲突"""
        conflicts = []
        
        # 地理冲突检测
        if '巴黎' in data and '德国' in data:
            conflicts.append("地理冲突：巴黎不在德国")
        
        if '北京' in data and '日本' in data:
            conflicts.append("地理冲突：北京不在日本")
        
        # 可以添加更多冲突检测规则
        
        return conflicts
    
    def _detect_format_issues(self, data: str) -> List[str]:
        """检测格式问题"""
        issues = []
        
        # 检查标点符号
        if not re.search(r'[。！？.]', data):
            issues.append("缺少标点符号")
        
        # 检查是否全是大写或小写
        if data.isupper():
            issues.append("全大写格式")
        elif data.islower():
            issues.append("全小写格式")
        
        return issues
    
    def _contains_meaningless_content(self, data: str) -> bool:
        """检查是否包含无意义内容"""
        meaningless_patterns = [
            r'啊{3,}', r'哈{3,}', r'呵{3,}',  # 重复感叹词
            r'[a-zA-Z]{20,}',  # 过长的英文字符串
            r'\d{10,}'  # 过长的数字串
        ]
        
        for pattern in meaningless_patterns:
            if re.search(pattern, data):
                return True
        
        return False
    
    def _detect_issues(self, data: str) -> List[str]:
        """检测数据中的具体问题"""
        issues = []
        
        # 长度问题
        if len(data.strip()) < 5:
            issues.append("数据过短")
        
        # 实体缺失
        entities = self._extract_entities(data)
        if len(entities) < 2:
            issues.append("实体信息不足")
        
        # 关系缺失
        relations = self._extract_relations(data)
        if len(relations) == 0:
            issues.append("缺少关系信息")
        
        # 语义冲突
        conflicts = self._detect_semantic_conflicts(data)
        issues.extend(conflicts)
        
        # 格式问题
        format_issues = self._detect_format_issues(data)
        issues.extend(format_issues)
        
        return issues
    
    def _generate_recommendation(self, quality_score: float, issues: List[str]) -> str:
        """生成改进建议"""
        if quality_score >= self.quality_threshold:
            return "数据质量良好，可直接用于知识图谱构建"
        
        recommendations = []
        
        if "数据过短" in issues:
            recommendations.append("建议补充更多详细信息")
        
        if "实体信息不足" in issues:
            recommendations.append("建议添加更多实体信息（人名、地名、组织名等）")
        
        if "缺少关系信息" in issues:
            recommendations.append("建议明确实体间的关系")
        
        if any("冲突" in issue for issue in issues):
            recommendations.append("建议修正语义冲突")
        
        if not recommendations:
            recommendations.append("建议进行知识补全以提高数据质量")
        
        return "；".join(recommendations)
