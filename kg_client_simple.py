#!/usr/bin/env python3
"""
简化版知识图谱构建MCP客户端
用于测试简化版服务器
"""
import asyncio
import json
import sys
import subprocess
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_simple_server():
    """测试简化版服务器"""
    print("🚀 启动简化版知识图谱构建客户端")
    
    # 启动服务器
    server_params = StdioServerParameters(
        command="python",
        args=["kg_server_simple.py"],
        cwd="."
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化
                await session.initialize()
                
                # 列出工具
                tools = await session.list_tools()
                print(f"📋 可用工具: {[tool.name for tool in tools.tools]}")
                
                if not tools.tools:
                    print("❌ 没有可用工具")
                    return
                
                # 测试数据
                test_text = "黄超可以看一下这个文章：Prompt Engineering Through the Lens of Optimal Control。北大董彬老师的工作，把agent设计作为一个最优控制问题，从最优控制的视角，把构建智能体中的提示工程（Prompt Engineering）问题，抽象成一个数学框架，研究怎么最大程度地榨取llm的能力。"
                
                print("\n🔍 测试知识图谱构建和分析...")
                
                # 调用工具
                result = await session.call_tool(
                    name="build_and_analyze_kg",
                    arguments={"text": test_text}
                )
                
                # 解析结果
                if result.content:
                    content = result.content[0].text
                    data = json.loads(content)
                    
                    print(f"✅ 处理成功！")
                    print(f"   - 处理时间: {data.get('processing_time', 0):.2f}秒")
                    print(f"   - 实体数量: {data.get('knowledge_graph', {}).get('entities_count', 0)}")
                    print(f"   - 关系数量: {data.get('knowledge_graph', {}).get('relations_count', 0)}")
                    print(f"   - 质量评分: {data.get('analysis_results', {}).get('quality_score', 0):.1f}")
                    print(f"   - 建议数量: {data.get('analysis_results', {}).get('recommendations_count', 0)}")
                    
                    # 显示前几个建议
                    recommendations = data.get('analysis_results', {}).get('top_recommendations', [])
                    if recommendations:
                        print("\n📝 主要建议:")
                        for i, rec in enumerate(recommendations[:3]):
                            print(f"   {i+1}. {rec.get('description', '无描述')} (优先级: {rec.get('priority', '未知')})")
                    
                    print("\n🎉 测试完成！")
                else:
                    print("❌ 没有收到结果")
                    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_server()) 