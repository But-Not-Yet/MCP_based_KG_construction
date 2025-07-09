#!/usr/bin/env python3
"""
测试分析模块是否正常工作
"""
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_analysis_modules():
    """测试分析模块"""
    print("🔍 测试分析模块...")
    
    try:
        # 测试导入
        from content_enhancement.analysis_pipeline import analyze_knowledge_graph, AnalysisConfig
        print("✅ 导入成功")
        
        # 准备测试数据
        sample_text = "黄超可以看一下这个文章：Prompt Engineering Through the Lens of Optimal Control。北大董彬老师的工作，把agent设计作为一个最优控制问题，从最优控制的视角，把构建智能体中的提示工程（Prompt Engineering）问题，抽象成一个数学框架，研究怎么最大程度地榨取llm的能力。"
        
        sample_entities = [
            {'name': '黄超', 'type': 'Person', 'attributes': {}, 'relations': []},
            {'name': 'Prompt Engineering Through the Lens of Optimal Control', 'type': 'Article', 'attributes': {}, 'relations': []},
            {'name': '北大', 'type': 'Organization', 'attributes': {}, 'relations': []},
            {'name': '董彬', 'type': 'Person', 'attributes': {}, 'relations': []},
            {'name': '最优控制', 'type': 'Concept', 'attributes': {}, 'relations': []},
            {'name': 'llm', 'type': 'Technology', 'attributes': {}, 'relations': []}
        ]
        
        sample_relations = [
            {'name': '可以查看', 'source': '黄超', 'target': 'Prompt Engineering Through the Lens of Optimal Control', 'type': 'Action'},
            {'name': '是', 'source': 'Prompt Engineering Through the Lens of Optimal Control', 'target': '董彬', 'type': 'Attribution'},
            {'name': '工作内容', 'source': '董彬', 'target': '最优控制', 'type': 'Research'},
            {'name': '被抽象成', 'source': '提示工程', 'target': '数学框架', 'type': 'Abstraction'},
            {'name': '研究对象', 'source': '数学框架', 'target': 'llm', 'type': 'Research'}
        ]
        
        # 创建配置
        config = AnalysisConfig(
            enable_global_analysis=True,
            enable_detail_analysis=True,
            similarity_threshold=0.3,
            max_recommendations=10
        )
        
        print("📊 开始分析...")
        
        # 执行分析
        result = await analyze_knowledge_graph(sample_text, sample_entities, sample_relations, config)
        
        print(f"✅ 分析完成！")
        print(f"   - 时间戳: {result.timestamp}")
        print(f"   - 质量分数: {result.quality_metrics.get('overall_score', 0):.1f}")
        print(f"   - 总问题数: {result.quality_metrics.get('issue_count', 0)}")
        print(f"   - 建议数量: {len(result.integrated_recommendations)}")
        
        if result.integrated_recommendations:
            print("\n📝 前3个建议:")
            for i, rec in enumerate(result.integrated_recommendations[:3]):
                print(f"   {i+1}. {rec.get('description', '无描述')} (优先级: {rec.get('priority', '未知')})")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_analysis_modules())
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败！")
        sys.exit(1) 