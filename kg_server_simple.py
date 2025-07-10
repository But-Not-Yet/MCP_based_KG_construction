#!/usr/bin/env python3
"""
简化版知识图谱构建MCP服务器
用于测试和解决初始化问题
"""
import asyncio
import json
import os
import sys
import time
from typing import Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置控制台编码
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 原有模块
from kg_utils import KnowledgeGraphBuilder
from kg_visualizer import KnowledgeGraphVisualizer

# 新增分析模块
try:
    from content_enhancement.analysis_pipeline import analyze_knowledge_graph, AnalysisConfig
    from content_enhancement.enhancement_executor import EnhancementExecutor
    ANALYSIS_AVAILABLE = True
    print("✅ 分析模块加载成功")
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    print(f"❌ 分析模块加载失败: {e}")
    print("高级分析功能将不可用")
except Exception as e:
    ANALYSIS_AVAILABLE = False
    print(f"❌ 分析模块初始化错误: {e}")
    print("高级分析功能将不可用")

# 全局组件
try:
    kg_builder = KnowledgeGraphBuilder(api_key=os.getenv("OPENAI_API_KEY"))
    kg_visualizer = KnowledgeGraphVisualizer()
    if ANALYSIS_AVAILABLE:
        enhancement_executor = EnhancementExecutor()
    print("✅ 核心组件初始化成功")
except Exception as e:
    print(f"❌ 核心组件初始化失败: {e}")
    raise

# 创建服务器实例
server = Server("knowledge-graph-builder-simple")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """列出可用的工具"""
    tools = []
    
    # 如果分析模块可用，添加高级分析工具
    if ANALYSIS_AVAILABLE:
        tools.append(
            Tool(
                name="build_and_analyze_kg",
                description="构建知识图谱并进行高级分析，自动增强并生成可视化",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "要处理的文本数据"
                        },
                        "auto_enhance": {
                            "type": "boolean",
                            "description": "是否自动增强知识图谱",
                            "default": True
                        },
                        "generate_visualization": {
                            "type": "boolean",
                            "description": "是否生成可视化文件",
                            "default": True
                        },
                        "output_file": {
                            "type": "string",
                            "description": "可视化输出文件名",
                            "default": "enhanced_knowledge_graph.html"
                        },
                        "auto_enhance_relations": {
                            "type": "boolean",
                            "description": "是否自动增强关系（可能会添加很多关系）",
                            "default": False
                        }
                    },
                    "required": ["text"]
                }
            )
        )
    
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """处理工具调用"""
    if name == "build_and_analyze_kg" and ANALYSIS_AVAILABLE:
        return await build_and_analyze_kg_simple(arguments)
    else:
        raise ValueError(f"未知工具: {name}")

async def build_and_analyze_kg_simple(arguments: dict[str, Any]) -> list[TextContent]:
    """简化版一体化工具 - 分析、增强、可视化"""
    try:
        text = arguments.get("text", "")
        auto_enhance = arguments.get("auto_enhance", True)
        generate_visualization = arguments.get("generate_visualization", True)
        output_file = arguments.get("output_file", "enhanced_knowledge_graph.html")
        auto_enhance_relations = arguments.get("auto_enhance_relations", False)
        
        if not text.strip():
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "输入文本不能为空"
                }, ensure_ascii=False, indent=2)
            )]
        
        start_time = time.time()
        
        # 阶段1：构建基础知识图谱
        kg_result = await kg_builder.build_graph(text, use_llm=True)
        
        if not kg_result["entities"] and not kg_result["triples"]:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "无法从输入文本中提取到有效的实体或关系"
                }, ensure_ascii=False, indent=2)
            )]
        
        # 阶段2：转换数据格式用于分析
        entities = [
            {
                'name': entity,
                'type': 'unknown',
                'attributes': {},
                'relations': []
            }
            for entity in kg_result["entities"]
        ]
        
        relations = [
            {
                'name': triple.relation,
                'source': triple.head,
                'target': triple.tail,
                'type': 'unknown'
            }
            for triple in kg_result["triples"]
        ]
        
        # 阶段3：配置分析参数并执行高级分析
        config = AnalysisConfig(
            enable_global_analysis=True,
            enable_detail_analysis=True,
            similarity_threshold=0.3,
            max_recommendations=10
        )
        
        analysis_result = await analyze_knowledge_graph(
            text, entities, relations, config
        )
        
        # 阶段4：自动增强（如果启用）
        enhancement_result = None
        final_entities = kg_result["entities"]
        final_relations = kg_result["relations"]
        final_triples = kg_result["triples"]
        
        if auto_enhance:
            enhancement_result = await enhancement_executor.execute_enhancements(
                text, kg_result["entities"], kg_result["relations"], kg_result["triples"], analysis_result, auto_enhance_relations
            )
            
            # 使用增强后的数据
            final_entities = [e['name'] for e in enhancement_result.enhanced_entities]
            final_relations = [r['name'] for r in enhancement_result.enhanced_relations]
            
            # 构建增强后的三元组用于可视化
            enhanced_triples = []
            for triple_dict in enhancement_result.enhanced_triples:
                from kg_utils import Triple
                enhanced_triple = Triple(
                    head=triple_dict['head'],
                    relation=triple_dict['relation'],
                    tail=triple_dict['tail'],
                    confidence=triple_dict.get('confidence', 0.8)
                )
                enhanced_triples.append(enhanced_triple)
            
            final_triples = enhanced_triples
        
        # 阶段5：生成可视化（如果启用）
        visualization_info = None
        if generate_visualization:
            visualization_file = kg_visualizer.save_simple_visualization(
                final_triples,
                final_entities,
                final_relations,
                output_file
            )
            
            abs_path = os.path.abspath(visualization_file)
            visualization_url = f"file:///{abs_path.replace(os.sep, '/')}"
            http_url = f"http://localhost:8000/{visualization_file}"
            
            visualization_info = {
                "file_path": visualization_file,
                "file_url": visualization_url,
                "http_url": http_url,
                "server_info": f"可手动启动HTTP服务器访问：在项目目录运行 'python -m http.server 8000'，然后访问 {http_url}"
            }
        
        processing_time = time.time() - start_time
        
        # 构建结果
        result = {
            "success": True,
            "input_text": text,
            "processing_time": round(processing_time, 3),
            "original_knowledge_graph": {
                "entities_count": len(kg_result["entities"]),
                "relations_count": len(kg_result["relations"]),
                "triples_count": len(kg_result["triples"]),
                "entities": kg_result["entities"],
                "relations": kg_result["relations"]
            },
            "analysis_results": {
                "timestamp": analysis_result.timestamp,
                "quality_score": analysis_result.quality_metrics.get('overall_score', 0),
                "total_issues": analysis_result.quality_metrics.get('issue_count', 0),
                "critical_issues": analysis_result.quality_metrics.get('critical_issues', 0),
                "recommendations_count": len(analysis_result.integrated_recommendations),
                "top_recommendations": analysis_result.integrated_recommendations[:5]
            },
            "enhancement_results": {
                "auto_enhance_enabled": auto_enhance,
                "enhancement_applied": enhancement_result is not None,
                "enhancement_summary": enhancement_result.enhancement_summary if enhancement_result else None,
                "applied_enhancements": enhancement_result.applied_enhancements if enhancement_result else [],
                "final_entities_count": len(final_entities),
                "final_relations_count": len(final_relations),
                "final_triples_count": len(final_triples)
            },
            "visualization": {
                "visualization_enabled": generate_visualization,
                "visualization_generated": visualization_info is not None,
                "visualization_info": visualization_info
            },
            "detailed_analysis": {
                "global_analysis": analysis_result.global_analysis_results is not None,
                "detail_analysis": analysis_result.detail_analysis_results is not None,
                "all_recommendations": analysis_result.integrated_recommendations
            }
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e),
                "error_details": error_details
            }, ensure_ascii=False, indent=2)
        )]

async def main():
    """运行服务器"""
    print("🚀 启动简化版知识图谱构建服务器")
    print(f"🔧 高级分析功能: {'✅ 可用' if ANALYSIS_AVAILABLE else '❌ 不可用'}")
    
    try:
        # 验证组件状态
        print("🔧 验证组件状态...")
        if not hasattr(kg_builder, 'build_graph'):
            raise RuntimeError("知识图谱构建器未正确初始化")
        if not hasattr(kg_visualizer, 'save_simple_visualization'):
            raise RuntimeError("可视化组件未正确初始化")
        if ANALYSIS_AVAILABLE and not hasattr(enhancement_executor, 'execute_enhancements'):
            raise RuntimeError("增强执行器未正确初始化")
        print("✅ 所有组件验证通过")
        
        # 使用 stdio 传输运行服务器
        print("🔗 启动MCP服务器...")
        async with stdio_server() as (read_stream, write_stream):
            # 添加初始化延迟确保所有组件就绪
            await asyncio.sleep(0.1)
            
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="knowledge-graph-builder-simple",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    ),
                ),
            )
            
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main()) 