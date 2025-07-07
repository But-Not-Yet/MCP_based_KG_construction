#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证知识图谱增强功能
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "content_enhancement"))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试原有模块
        from data_quality import DataQualityAssessor
        from knowledge_completion import KnowledgeCompletor
        from kg_utils import KnowledgeGraphBuilder
        print("✅ 原有模块导入成功")
        
        # 测试新增模块
        from content_enhancement.global_analysis import GlobalAnalyzer
        from content_enhancement.entity_detail_analyzer import EntityDetailAnalyzer
        from content_enhancement.analysis_pipeline import analyze_knowledge_graph, AnalysisConfig
        print("✅ 新增分析模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


async def test_analysis_pipeline():
    """测试分析流程"""
    print("\n📊 测试分析流程...")
    
    try:
        from content_enhancement.analysis_pipeline import analyze_knowledge_graph, AnalysisConfig
        
        # 测试数据
        test_text = """
        张三是阿里巴巴的高级工程师，负责云计算平台的开发。
        他毕业于清华大学计算机科学专业，有10年的工作经验。
        阿里巴巴成立于1999年，总部位于杭州，是中国最大的电商公司之一。
        """
        
        test_entities = [
            {
                'name': '张三',
                'type': 'Person',
                'attributes': {'职业': '高级工程师', '工作年限': '10年'},
                'relations': ['工作于', '毕业于']
            },
            {
                'name': '阿里巴巴',
                'type': 'Organization',
                'attributes': {'成立时间': '1999年', '总部': '杭州'},
                'relations': ['雇佣', '位于']
            },
            {
                'name': '清华大学',
                'type': 'Organization',
                'attributes': {'类型': '高等院校'},
                'relations': ['培养']
            }
        ]
        
        test_relations = [
            {
                'name': '工作于',
                'source': '张三',
                'target': '阿里巴巴',
                'type': '雇佣关系'
            },
            {
                'name': '毕业于',
                'source': '张三',
                'target': '清华大学',
                'type': '教育关系'
            }
        ]
        
        # 执行分析
        config = AnalysisConfig(
            enable_global_analysis=True,
            enable_detail_analysis=True,
            similarity_threshold=0.3,
            max_recommendations=10
        )
        
        result = await analyze_knowledge_graph(test_text, test_entities, test_relations, config)
        
        print(f"✅ 分析完成!")
        print(f"   - 质量评分: {result.quality_metrics['overall_score']:.1f}")
        print(f"   - 发现问题: {result.quality_metrics['issue_count']} 个")
        print(f"   - 建议数量: {len(result.integrated_recommendations)} 个")
        
        if result.integrated_recommendations:
            print("   - 前3个建议:")
            for i, rec in enumerate(result.integrated_recommendations[:3], 1):
                print(f"     {i}. [{rec['priority']}] {rec['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_original_functions():
    """测试原有功能"""
    print("\n🔧 测试原有功能...")
    
    try:
        from data_quality import DataQualityAssessor
        from kg_utils import KnowledgeGraphBuilder
        
        # 测试数据质量评估
        assessor = DataQualityAssessor()
        quality_result = await assessor.assess_quality("张三是阿里巴巴的工程师")
        print(f"✅ 数据质量评估: {quality_result['quality_score']:.2f}")
        
        # 测试知识图谱构建
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            builder = KnowledgeGraphBuilder(api_key=api_key)
            kg_result = await builder.build_graph("张三是阿里巴巴的工程师", use_llm=True)
            print(f"✅ 知识图谱构建: {len(kg_result['entities'])} 个实体, {len(kg_result['triples'])} 个三元组")
        else:
            print("⚠️  未设置OPENAI_API_KEY，跳过LLM相关测试")
        
        return True
        
    except Exception as e:
        print(f"❌ 原有功能测试失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n📁 检查文件结构...")
    
    required_files = [
        "kg_server.py",
        "kg_server_enhanced.py", 
        "content_enhancement/global_analysis.py",
        "content_enhancement/entity_detail_analyzer.py",
        "content_enhancement/analysis_pipeline.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """检查依赖"""
    print("\n📦 检查依赖...")
    
    required_packages = [
        "networkx",
        "numpy", 
        "scipy",
        "jieba",
        "asyncio"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖: {', '.join(missing_packages)}")
        print("请运行: uv sync")
        return False
    
    return True


async def main():
    """主测试函数"""
    print("🚀 开始测试知识图谱增强功能\n")
    
    tests = [
        ("文件结构", test_file_structure),
        ("依赖检查", check_dependencies),
        ("模块导入", test_imports),
        ("原有功能", test_original_functions),
        ("分析流程", test_analysis_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"测试: {test_name}")
        print(f"{'='*50}")
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        
        if result:
            passed += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    print(f"\n{'='*50}")
    print(f"🎯 测试结果: {passed}/{total} 通过")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 所有测试通过！您可以运行以下命令:")
        print("   原版: uv run kg_server.py")
        print("   增强版: uv run kg_server_enhanced.py")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 