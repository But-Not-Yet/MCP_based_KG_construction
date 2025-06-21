#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from kg_utils import KnowledgeGraphBuilder

async def quick_test():
    """快速测试LLM功能是否正常"""
    
    print("🚀 快速测试 - 检查LLM功能")
    print("=" * 50)
    
    # 使用示例API密钥
    api_key = "sk-igsxqudwjumptrovmyuuxemhjvhwqxnhegsuuswqpipnxfre"
    
    # 创建知识图谱构建器
    kg_builder = KnowledgeGraphBuilder(api_key=api_key)
    
    # 简单的测试文本
    test_text = "张三是阿里巴巴的CEO"
    
    print(f"📝 测试文本: {test_text}")
    print("\n🔄 正在测试LLM模式...")
    
    try:
        # 测试LLM模式
        result = await kg_builder.build_graph(test_text, use_llm=True)
        
        if result["triples"]:
            print("✅ LLM模式测试成功！")
            print(f"   提取到 {len(result['entities'])} 个实体")
            print(f"   提取到 {len(result['relations'])} 个关系")
            print(f"   生成了 {len(result['triples'])} 个三元组")
            
            print("\n📊 提取结果:")
            print(f"   实体: {result['entities']}")
            print(f"   关系: {result['relations']}")
            print("   三元组:")
            for i, triple in enumerate(result['triples'], 1):
                print(f"     {i}. {triple}")
        else:
            print("⚠️  LLM模式测试：未提取到三元组")
            print("   这可能表示API调用失败或响应异常")
            
    except Exception as e:
        print(f"❌ LLM模式测试失败: {e}")
    
    print("\n🔄 正在测试规则模式（作为对比）...")
    
    try:
        # 测试规则模式作为对比
        rule_builder = KnowledgeGraphBuilder()  # 不提供API密钥
        rule_result = await rule_builder.build_graph(test_text, use_llm=False)
        
        print("✅ 规则模式测试成功！")
        print(f"   提取到 {len(rule_result['entities'])} 个实体")
        print(f"   提取到 {len(rule_result['relations'])} 个关系")
        print(f"   生成了 {len(rule_result['triples'])} 个三元组")
        
        if rule_result['triples']:
            print("   规则模式三元组:")
            for i, triple in enumerate(rule_result['triples'], 1):
                print(f"     {i}. {triple}")
                
    except Exception as e:
        print(f"❌ 规则模式测试失败: {e}")
    
    print("\n" + "=" * 50)
    print("💡 测试建议:")
    print("   - 如果LLM模式成功：说明API工作正常")
    print("   - 如果LLM模式失败：请运行 'python api_diagnostics.py' 进行详细诊断")
    print("   - 如果两种模式都失败：请检查项目依赖和代码")

if __name__ == "__main__":
    asyncio.run(quick_test()) 