#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱构建 MCP 服务器
提供全自动化的知识图谱构建服务
"""

import asyncio
import json
import time
import sys
import os
from typing import Any, Sequence
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
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from data_quality import DataQualityAssessor
from knowledge_completion import KnowledgeCompletor
from kg_utils import KnowledgeGraphBuilder
from kg_visualizer import KnowledgeGraphVisualizer
# 移除了simple_file_server依赖，直接生成文件路径

# 全局组件
quality_assessor = DataQualityAssessor()
knowledge_completor = KnowledgeCompletor()
kg_builder = KnowledgeGraphBuilder(api_key=os.getenv("OPENAI_API_KEY"))
kg_visualizer = KnowledgeGraphVisualizer()

# 创建服务器实例
server = Server("knowledge-graph-builder")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """
    列出可用的工具
    """
    return [
        Tool(
            name="build_knowledge_graph",
            description="全自动构建知识图谱：自动评估数据质量、补全知识、构建图谱并生成可视化",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要处理的文本数据"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "可视化输出文件名（可选）",
                        "default": "knowledge_graph.html"
                    }
                },
                "required": ["text"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    处理工具调用
    """
    if name == "build_knowledge_graph":
        return await build_knowledge_graph_tool(arguments)
    else:
        raise ValueError(f"未知工具: {name}")


async def build_knowledge_graph_tool(arguments: dict[str, Any]) -> list[TextContent]:
    """
    全自动知识图谱构建工具
    """
    try:
        text = arguments.get("text", "")
        output_file = arguments.get("output_file", "knowledge_graph.html")

        if not text.strip():
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "输入文本不能为空"
                }, ensure_ascii=False, indent=2)
            )]

        start_time = time.time()

        # 阶段1：数据质量评估
        quality_result = await quality_assessor.assess_quality(text)

        # 阶段2：知识补全（如果需要）
        processed_text = text
        completion_info = {"skipped": True, "reason": "数据质量良好"}

        if not quality_result["is_high_quality"]:
            completion_result = await knowledge_completor.complete_knowledge(text, quality_result)
            processed_text = completion_result["enhanced_data"]
            completion_info = {
                "skipped": False,
                "completions": completion_result["completions"],
                "corrections": completion_result["corrections"],
                "confidence": completion_result["confidence"]
            }

        # 阶段3：知识图谱构建
        kg_result = await kg_builder.build_graph(processed_text, use_llm=True)

        # 检查是否成功提取到实体和三元组
        if not kg_result["entities"] and not kg_result["triples"]:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "无法从输入文本中提取到有效的实体或关系",
                    "suggestion": "请尝试输入包含明确实体和关系的文本，例如：'张三担任阿里巴巴公司的CEO'"
                }, ensure_ascii=False, indent=2)
            )]

        # 阶段4：生成可视化（简洁版本，只包含知识图谱网络图）
        visualization_file = kg_visualizer.save_simple_visualization(
            kg_result["triples"],
            kg_result["entities"],
            kg_result["relations"],
            output_file
        )

        # 生成文件访问URL（简化版本，避免超时）
        abs_path = os.path.abspath(visualization_file)
        visualization_url = f"file:///{abs_path.replace(os.sep, '/')}"

        # 提供HTTP服务器启动说明
        http_url = f"http://localhost:8000/{visualization_file}"
        server_info = f"可手动启动HTTP服务器访问：在项目目录运行 'python -m http.server 8000'，然后访问 {http_url}"

        processing_time = time.time() - start_time

        # 构建结果
        result = {
            "success": True,
            "input_text": text,
            "processing_time": round(processing_time, 3),
            "stages": {
                "quality_assessment": {
                    "quality_score": quality_result["quality_score"],
                    "is_high_quality": quality_result["is_high_quality"],
                    "completeness": quality_result["completeness"],
                    "consistency": quality_result["consistency"],
                    "relevance": quality_result["relevance"],
                    "issues": quality_result["issues"],
                    "recommendation": quality_result["recommendation"]
                },
                "knowledge_completion": completion_info,
                "knowledge_graph": {
                    "entities_count": len(kg_result["entities"]),
                    "relations_count": len(kg_result["relations"]),
                    "triples_count": len(kg_result["triples"]),
                    "entities": kg_result["entities"],
                    "relations": kg_result["relations"],
                    "average_confidence": sum(kg_result["confidence_scores"]) / len(kg_result["confidence_scores"]) if kg_result["confidence_scores"] else 0
                },
                "visualization": {
                    "file_path": visualization_file,
                    "file_url": visualization_url,
                    "http_url": http_url,
                    "server_info": server_info,
                    "file_size": len(kg_visualizer.generate_simple_html(
                        kg_result["triples"],
                        kg_result["entities"],
                        kg_result["relations"]
                    ))
                }
            },
            "summary": {
                "original_text": text,
                "processed_text": processed_text,
                "quality_improved": not quality_result["is_high_quality"],
                "final_entities": len(kg_result["entities"]),
                "final_triples": len(kg_result["triples"]),
                "visualization_ready": True,
                "visualization_file": visualization_file,
                "visualization_url": visualization_url
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
    """
    运行服务器
    """
    # 使用 stdio 传输运行服务器
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="knowledge-graph-builder",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
