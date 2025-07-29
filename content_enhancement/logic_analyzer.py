from __future__ import annotations
import json
import logging
from typing import Dict, List, Any, Optional

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class LogicAnalyzer:
    """
    逻辑分析器 (紫色模块) - 使用AI智能体进行深层逻辑推理
    """

    def __init__(self, llm_client: Optional[LLMClient]):
        if not llm_client or not llm_client.is_operational:
            logger.warning("LogicAnalyzer 未接收到可用的 LLMClient，将无法执行。")
        self.llm_client = llm_client

    def _build_agent_prompt(self, original_text: str, entities: List[Dict], relations: List[Dict]) -> str:
        """
        构建用于AI智能体推理的提示。
        """
        # 为了提示的简洁性，对实体和关系进行格式化
        formatted_entities = [f"- {e.get('name')} (类型: {e.get('type', '未知')})" for e in entities]
        formatted_relations = [f"- ({r.get('source')}, {r.get('name')}, {r.get('target')})" for r in relations]

        prompt = f"""
你是一个高度智能的“知识图谱推理智能体”。你的任务是分析下面提供的原始文本和已抽取的知识图谱片段，
进行深度的逻辑推理，以发现其中隐藏的、缺失的或可以推断出的信息。

不要重复已有的事实，你的目标是找出那些需要“思考”才能发现的知识。

---
### 原始文本:
```
{original_text}
```

---
### 已抽取的知识图谱:
#### 实体:
{chr(10).join(formatted_entities)}

#### 关系:
{chr(10).join(formatted_relations)}

---
### 你的推理任务:
1.  **隐含关系推断**: 基于上下文和你的世界知识，如果两个实体在没有明确动词的情况下共同出现，它们之间最可能存在什么关系？(例如: "微软, 比尔·盖茨" 暗示 `(比尔·盖茨, 创立, 微软)`; "北京, 中国" 暗示 `(北京, 位于, 中国)`)。这是最高优先级的任务。
2.  **因果链推理**: 如果 A 导致 B，B 导致 C，那么 A 和 C 之间是否存在间接的因果关系？
3.  **传递关系推断**: 如果 A 是 B 的一部分，B 是 C 的一部分，那么 A 和 C 是什么关系？
4.  **缺失角色分析**: 一个事件发生了，但它的“执行者”或“影响对象”在图中缺失了吗？
5.  **逻辑矛盾检测**: 是否存在逻辑上相互矛盾的关系？（例如：A 是 B 的父公司，同时 B 又是 A 的父公司）
6.  **反事实推断**: 如果某个条件不成立，可能会发生什么？（例如：如果“安全系统”未部署，可能会导致什么后果？）

---
### 输出格式:
请严格以JSON格式返回你的发现。返回一个根对象，包含一个 `findings` 键，其值为一个列表。
列表中的每个对象都应包含以下字段:
- `type`: 字符串，你的发现类型 (例如: "Inferred_Transitive_Relation", "Missing_Event_Actor", "Logical_Inconsistency")。
- `description`: 字符串，对你的发现进行详细的自然语言描述。
- `evidence`: 列表，支撑你推理的证据，引用已抽取的实体或关系。
- `suggestion`: 字符串，一个具体的增强建议 (例如: "添加关系 (A, '间接导致', C)")。
- `confidence`: 浮点数，你对这个发现的置信度 (0.0到1.0)。

如果没有任何发现，请返回 `{"findings": []}`。

现在，请开始你的推理。
"""
        return prompt.strip()

    async def analyze_with_agent(self, original_text: str, entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """
        使用AI智能体进行分析，找出缺失的信息和关系。

        Args:
            original_text: 原始文本.
            entities: 已抽取的实体列表.
            relations: 已抽取的关系列表.

        Returns:
            一个包含推理结果的字典。
        """
        if not self.llm_client:
            return {"findings": []}

        prompt = self._build_agent_prompt(original_text, entities, relations)
        
        # 这里我们直接使用封装好的LLMClient的custom_query
        # 因为我们需要它返回一个可以被解析为JSON的字符串
        response_text = self.llm_client.custom_query(prompt, temperature=0.3)

        if not response_text:
            logger.warning("逻辑分析智能体未能从LLM返回任何内容。")
            return {"findings": []}

        # 使用已有的清理和解析函数
        analysis_result = self.llm_client._clean_and_parse_json(response_text, context="LogicAnalyzer")

        if analysis_result and "findings" in analysis_result and isinstance(analysis_result["findings"], list):
            return analysis_result
        else:
            logger.warning(f"逻辑分析智能体返回的格式不正确或为空。返回内容: {response_text[:200]}...")
            return {"findings": []}

"""
# 使用示例
async def main():
    # 假设我们有一个LLMClient实例
    # from .llm_client import LLMClient
    # llm_client = LLMClient()
    
    analyzer = LogicAnalyzer(llm_client)
    
    text = "特斯拉发布了Cybertruck，其独特的造型引发了广泛讨论。该车的生产基地位于德州超级工厂，而该工厂同时还生产Model Y。马斯克是特斯拉的CEO。"
    entities = [{"name": "特斯拉", "type": "Organization"}, {"name": "Cybertruck", "type": "Product"}, {"name": "德州超级工厂", "type": "Location"}, {"name": "Model Y", "type": "Product"}, {"name": "马斯克", "type": "Person"}]
    relations = [{"source": "特斯拉", "name": "发布", "target": "Cybertruck"}, {"source": "德州超级工厂", "name": "生产", "target": "Cybertruck"}, {"source": "德州超级工厂", "name": "生产", "target": "Model Y"}, {"source": "马斯克", "name": "担任CEO", "target": "特斯拉"}]
    
    results = await analyzer.analyze_with_agent(text, entities, relations)
    
    import json
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    import asyncio
    # 要运行此示例，需要提供一个有效的 LLMClient
    # asyncio.run(main())
""" 