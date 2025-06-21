#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from kg_utils import KnowledgeGraphBuilder

async def main():
    """知识图谱构建示例"""
    
    # 您的Silicon Flow API密钥
    api_key = "sk-igsxqudwjumptrovmyuuxemhjvhwqxnhegsuuswqpipnxfre"
    
    # 创建知识图谱构建器（使用LLM）
    kg_builder_llm = KnowledgeGraphBuilder(api_key=api_key)
    
    # 创建知识图谱构建器（不使用LLM，仅规则）
    kg_builder_rule = KnowledgeGraphBuilder()
    
    # 测试文本
    test_texts = [
        "张三是阿里巴巴公司的CEO，阿里巴巴总部位于杭州。",
        "李四在北京大学学习计算机科学，北京大学位于北京市海淀区。",
        "王五担任腾讯公司的产品经理，腾讯公司在深圳有分公司。"
    ]
    
    print("=" * 60)
    print("知识图谱构建对比测试")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n【测试文本 {i}】: {text}")
        print("-" * 50)
        
        # 使用LLM构建知识图谱
        print("🤖 使用LLM构建知识图谱:")
        try:
            llm_result = await kg_builder_llm.build_graph(text, use_llm=True)
            print(f"实体数量: {len(llm_result['entities'])}")
            print(f"关系数量: {len(llm_result['relations'])}")
            print(f"三元组数量: {len(llm_result['triples'])}")
            
            print("实体:", llm_result['entities'])
            print("关系:", llm_result['relations'])
            print("三元组:")
            for triple in llm_result['triples']:
                print(f"  - {triple}")
            
            if llm_result['confidence_scores']:
                avg_confidence = sum(llm_result['confidence_scores']) / len(llm_result['confidence_scores'])
                print(f"平均置信度: {avg_confidence:.3f}")
                
        except Exception as e:
            print(f"LLM构建失败: {e}")
        
        print()
        
        # 使用规则构建知识图谱
        print("📋 使用规则构建知识图谱:")
        try:
            rule_result = await kg_builder_rule.build_graph(text, use_llm=False)
            print(f"实体数量: {len(rule_result['entities'])}")
            print(f"关系数量: {len(rule_result['relations'])}")
            print(f"三元组数量: {len(rule_result['triples'])}")
            
            print("实体:", rule_result['entities'])
            print("关系:", rule_result['relations'])
            print("三元组:")
            for triple in rule_result['triples']:
                print(f"  - {triple}")
                
            if rule_result['confidence_scores']:
                avg_confidence = sum(rule_result['confidence_scores']) / len(rule_result['confidence_scores'])
                print(f"平均置信度: {avg_confidence:.3f}")
                
        except Exception as e:
            print(f"规则构建失败: {e}")
        
        print("=" * 60)
    
    # 获取统计信息
    print("\n📊 LLM版本统计信息:")
    llm_stats = kg_builder_llm.get_statistics()
    for key, value in llm_stats.items():
        print(f"{key}: {value}")
    
    print("\n📊 规则版本统计信息:")
    rule_stats = kg_builder_rule.get_statistics()
    for key, value in rule_stats.items():
        print(f"{key}: {value}")
    
    # 导出知识图谱
    print("\n💾 导出知识图谱 (JSON格式):")
    json_export = kg_builder_llm.export_graph("json")
    print(json_export[:500] + "..." if len(json_export) > 500 else json_export)


if __name__ == "__main__":
    asyncio.run(main()) 