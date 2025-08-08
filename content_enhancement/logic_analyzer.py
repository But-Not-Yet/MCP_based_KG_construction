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
        # --- 最终修复: 使用最标准的 \n.join() 确保健壮性 ---
        formatted_entities = "\n".join([f"- {e.get('name')} (类型: {e.get('type', '未知')})" for e in entities]) if entities else "(无实体被抽取)"
        formatted_relations = "\n".join([f"- ({r.get('source')}, {r.get('name')}, {r.get('target')})" for r in relations]) if relations else "(无关系被抽取)"

        prompt = f"""
你是一个严谨的知识图谱事实核查与优化专家。你的任务是基于你的世界知识和逻辑推理能力，审查一个从文本中抽取的知识图谱片段，然后提供一个可直接执行的“增强方案”。

---
### 原始文本:
```
{original_text}
```

---
### 已抽取的知识图谱 (待审查):
#### 实体:
{formatted_entities}

#### 关系:
{formatted_relations}

---
### 你的核心任务:
1.  **事实核查 (Fact-Checking)**: 这是最高优先级的任务。请用你的世界知识，逐一审查上面“关系”列表中的每一条三元组，判断其是否事实正确。例如，`(陕西省, 位于, 西安中部)` 是事实错误的。
2.  **隐含关系推断 (Implicit Inference)**: 根据文本和常识，补全实体间缺失的关键关系。例如，文本 "微软的比尔盖茨" 暗示了 `(比尔·盖茨, 创立, 微软)`。
3.  **逻辑一致性检查**: 检查是否存在自相矛盾的关系，例如 `(A, 包含, B)` 和 `(B, 包含, A)` 不能同时成立。

---
### 输出格式:
请严格以JSON格式返回一个根对象，包含一个 `enhancements` 键，其值为一个列表。
列表中的每个对象都是一个“增强指令”，包含以下字段:
- `type`: 字符串，指令类型 (例如: "FACTUAL_CORRECTION", "IMPLICIT_RELATION_INFERENCE")。
- `description`: 字符串，简要说明你做出此判断的理由。
- `confidence`: 浮点数，你对这个指令的置信度 (0.0到1.0)。
- `actions`: 一个操作列表，每个操作都是一个包含 "action" ("add" 或 "remove") 和 "triple" (一个包含 "head", "relation", "tail" 的对象) 的字典。

**示例:**
对于输入文本 "陕西省位于西安中部"，你的输出应该是：
```json
{{
  "enhancements": [
    {{
      "type": "FACTUAL_CORRECTION",
      "description": "“陕西省位于西安中部”是错误的。西安是陕西省的省会，陕西省位于中国。",
      "confidence": 0.99,
      "actions": [
        {{
          "action": "remove",
          "triple": {{"head": "陕西省", "relation": "位于", "tail": "西安中部"}}
        }},
        {{
          "action": "add",
          "triple": {{"head": "西安", "relation": "是省会", "tail": "陕西省"}}
        }},
        {{
          "action": "add",
          "triple": {{"head": "陕西省", "relation": "位于", "tail": "中国"}}
        }}
      ]
    }}
  ]
}}
```

如果没有任何需要修改或补充的，请返回 `{{"enhancements": []}}`。现在，请开始你的分析和优化。
"""
        return prompt.strip()

    async def analyze_with_agent(self, original_text: str, entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """
        使用AI智能体进行分析，直接生成增强方案。
        """
        if not self.llm_client:
            return {"enhancements": []}

        prompt = self._build_agent_prompt(original_text, entities, relations)
        
        response_text = await self.llm_client.acustom_query(prompt, temperature=0.1)

        if not response_text:
            logger.warning("逻辑分析智能体未能从LLM返回任何内容。")
            return {"enhancements": []}

        analysis_result = self.llm_client._clean_and_parse_json(response_text, context="LogicAnalyzer")

        if analysis_result and "enhancements" in analysis_result and isinstance(analysis_result["enhancements"], list):
            return analysis_result
        else:
            logger.warning(f"逻辑分析智能体返回的格式不正确或为空。返回内容: {response_text[:200]}...")
            return {"enhancements": []}

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
