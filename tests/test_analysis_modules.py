import unittest
from types import SimpleNamespace

from content_enhancement.entity_detail_analyzer import EntityDetailAnalyzer
from content_enhancement.global_analysis import GlobalAnalyzer


class MockLLMClient:
    """用于单元测试的模拟 LLMClient"""

    def __init__(self, succeed: bool = True):
        self._succeed = succeed

    # ---- 动态关键词 ----
    def get_keywords(self, category: str):
        if not self._succeed:
            return []
        if category == "verb":
            return ["测试动词1", "测试动词2"]
        if category == "causal":
            return ["测试因", "测试果"]
        return []

    # ---- 属性模板 ----
    def get_attribute_template(self, entity_type: str):
        if not self._succeed:
            return {}
        return {
            "required_attributes": ["测试必需"],
            "optional_attributes": ["测试可选"],
            "attribute_patterns": {"测试必需": r"测试"}
        }


class AnalysisModuleIntegrationTest(unittest.TestCase):
    """测试 EntityDetailAnalyzer 与 GlobalAnalyzer 的 LLM 集成与回退"""

    def test_entity_detail_analyzer_fallback(self):
        """在无 LLM 场景下应使用默认模板"""
        analyzer = EntityDetailAnalyzer(llm_client=None)
        self.assertIn("Person", analyzer.attribute_templates)
        self.assertIn("姓名", analyzer.attribute_templates["Person"].required_attributes)

    def test_entity_detail_analyzer_with_llm(self):
        """当 LLM 可用时应覆盖模板"""
        mock_client = MockLLMClient(succeed=True)
        analyzer = EntityDetailAnalyzer(llm_client=mock_client)
        tmpl = analyzer.attribute_templates.get("Person")
        self.assertIsNotNone(tmpl)
        self.assertIn("测试必需", tmpl.required_attributes)

    def test_global_analyzer_keyword_fallback(self):
        """无 LLM 时应使用默认关键词"""
        g = GlobalAnalyzer(llm_client=None)
        self.assertIn("进行", g.verb_keywords)
        self.assertIn("因为", g.causal_keywords)

    def test_global_analyzer_keyword_llm(self):
        """LLM 可用时应使用返回关键词"""
        mock_client = MockLLMClient(succeed=True)
        g = GlobalAnalyzer(llm_client=mock_client)
        self.assertIn("测试动词1", g.verb_keywords)
        self.assertIn("测试因", g.causal_keywords)


if __name__ == "__main__":
    unittest.main() 