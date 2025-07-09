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

# 新增分析模块
try:
    from content_enhancement.analysis_pipeline import analyze_knowledge_graph, AnalysisConfig
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
                description="构建知识图谱并进行高级分析",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "要处理的文本数据"
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
    """简化版一体化工具"""
    try:
        text = arguments.get("text", "")
        
        if not text.strip():
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "输入文本不能为空"
                }, ensure_ascii=False, indent=2)
            )]
        
        start_time = time.time()
        
        # 构建基础知识图谱
        kg_result = await kg_builder.build_graph(text, use_llm=True)
        
        if not kg_result["entities"] and not kg_result["triples"]:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "无法从输入文本中提取到有效的实体或关系"
                }, ensure_ascii=False, indent=2)
            )]
        
        # 转换数据格式用于分析
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
        
        # 配置分析参数
        config = AnalysisConfig(
            enable_global_analysis=True,
            enable_detail_analysis=True,
            similarity_threshold=0.3,
            max_recommendations=10
        )
        
        # 执行高级分析
        analysis_result = await analyze_knowledge_graph(
            text, entities, relations, config
        )
        
        processing_time = time.time() - start_time
        
        # 构建结果
        result = {
            "success": True,
            "input_text": text,
            "processing_time": round(processing_time, 3),
            "knowledge_graph": {
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