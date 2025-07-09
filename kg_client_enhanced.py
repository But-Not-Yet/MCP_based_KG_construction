import json
import asyncio
import os
import webbrowser
from typing import Optional
from contextlib import AsyncExitStack

from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


class EnhancedKnowledgeGraphClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    async def connect_to_server(self):
        """连接到增强版知识图谱服务器"""
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'kg_server_enhanced.py'],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await self.session.initialize()

    async def build_knowledge_graph(self, text: str, output_file: str = "knowledge_graph.html") -> dict:
        """
        构建知识图谱（原有功能）

        Args:
            text: 要处理的文本数据
            output_file: 可视化输出文件名

        Returns:
            包含知识图谱构建结果的字典
        """
        try:
            result = await self.session.call_tool("build_knowledge_graph", {
                "text": text,
                "output_file": output_file
            })

            result_text = result.content[0].text
            result_data = json.loads(result_text)

            return result_data

        except Exception as e:
            return {
                "success": False,
                "error": f"构建知识图谱时发生错误: {str(e)}"
            }

    async def analyze_knowledge_graph(self, text: str, **kwargs) -> dict:
        """
        分析知识图谱（新增功能）

        Args:
            text: 要分析的文本数据
            **kwargs: 分析配置参数

        Returns:
            包含分析结果的字典
        """
        try:
            params = {"text": text}
            params.update(kwargs)
            
            result = await self.session.call_tool("analyze_knowledge_graph", params)

            result_text = result.content[0].text
            result_data = json.loads(result_text)

            return result_data

        except Exception as e:
            return {
                "success": False,
                "error": f"分析知识图谱时发生错误: {str(e)}"
            }

    async def build_and_analyze(self, text: str, output_file: str = "knowledge_graph.html", **kwargs) -> dict:
        """
        构建并分析知识图谱（一体化功能）

        Args:
            text: 要处理的文本数据
            output_file: 可视化输出文件名
            **kwargs: 分析配置参数

        Returns:
            包含构建和分析结果的字典
        """
        try:
            params = {"text": text, "output_file": output_file}
            params.update(kwargs)
            
            result = await self.session.call_tool("build_and_analyze_kg", params)

            result_text = result.content[0].text
            result_data = json.loads(result_text)

            return result_data

        except Exception as e:
            return {
                "success": False,
                "error": f"构建和分析知识图谱时发生错误: {str(e)}"
            }

    def display_result(self, result: dict):
        """显示知识图谱构建结果"""
        if not result.get("success", False):
            print(f"\n❌ 错误: {result.get('error', '未知错误')}")
            return

        print("\n✅ 知识图谱构建成功!")
        print(f"⏱️  处理时间: {result.get('processing_time', 0):.3f} 秒")

        # 显示阶段信息
        stages = result.get("stages", {})

        # 数据质量评估
        quality = stages.get("quality_assessment", {})
        print(f"\n📊 数据质量评估:")
        print(f"   质量分数: {quality.get('quality_score', 0):.3f}")
        print(f"   高质量数据: {'是' if quality.get('is_high_quality', False) else '否'}")
        print(f"   完整性: {quality.get('completeness', 0):.3f}")
        print(f"   一致性: {quality.get('consistency', 0):.3f}")
        print(f"   相关性: {quality.get('relevance', 0):.3f}")

        # 知识补全
        completion = stages.get("knowledge_completion", {})
        if not completion.get("skipped", True):
            print(f"\n🔧 知识补全:")
            print(f"   置信度: {completion.get('confidence', 0):.3f}")
            print(f"   补全数量: {len(completion.get('completions', []))}")
            print(f"   修正数量: {len(completion.get('corrections', []))}")
        else:
            print(f"\n🔧 知识补全: 跳过 (数据质量良好)")

        # 知识图谱
        kg = stages.get("knowledge_graph", {})
        print(f"\n🕸️  知识图谱:")
        print(f"   实体数量: {kg.get('entities_count', 0)}")
        print(f"   关系类型: {kg.get('relations_count', 0)}")
        print(f"   三元组数量: {kg.get('triples_count', 0)}")
        print(f"   平均置信度: {kg.get('average_confidence', 0):.3f}")

        # 可视化
        viz = stages.get("visualization", {})
        print(f"\n🎨 可视化:")
        print(f"   文件路径: {viz.get('file_path', 'N/A')}")
        print(f"   文件大小: {viz.get('file_size', 0)} 字节")

        # 显示分析结果（如果有）
        analysis = result.get("analysis_results")
        if analysis:
            print(f"\n🔍 高级分析结果:")
            print(f"   质量评分: {analysis.get('quality_score', 0):.1f}")
            print(f"   发现问题: {analysis.get('total_issues', 0)} 个")
            print(f"   关键问题: {analysis.get('critical_issues', 0)} 个")
            print(f"   建议数量: {analysis.get('recommendations_count', 0)} 个")
            
            top_recommendations = analysis.get('top_recommendations', [])
            if top_recommendations:
                print(f"\n💡 主要建议:")
                for i, rec in enumerate(top_recommendations[:3], 1):
                    print(f"   {i}. [{rec.get('priority', '中')}] {rec.get('description', '')}")

        # 尝试打开可视化文件
        file_url = viz.get("file_url", "")
        if file_url and os.path.exists(viz.get("file_path", "")):
            try:
                print(f"\n🌐 正在打开可视化文件...")
                webbrowser.open(file_url)
                print(f"   已在浏览器中打开: {file_url}")
            except Exception as e:
                print(f"   无法自动打开浏览器: {e}")
                print(f"   请手动打开: {file_url}")

        print(f"\n💡 提示: {viz.get('server_info', '')}")

    def display_analysis_result(self, result: dict):
        """显示纯分析结果"""
        if not result.get("success", False):
            print(f"\n❌ 错误: {result.get('error', '未知错误')}")
            return

        print("\n✅ 知识图谱分析完成!")
        print(f"⏱️  处理时间: {result.get('processing_time', 0):.3f} 秒")

        # 知识图谱信息
        kg = result.get("knowledge_graph", {})
        print(f"\n🕸️  知识图谱:")
        print(f"   实体数量: {kg.get('entities_count', 0)}")
        print(f"   关系类型: {kg.get('relations_count', 0)}")
        print(f"   三元组数量: {kg.get('triples_count', 0)}")

        # 分析结果
        analysis = result.get("analysis_results", {})
        print(f"\n🔍 分析结果:")
        print(f"   质量评分: {analysis.get('quality_score', 0):.1f}")
        print(f"   发现问题: {analysis.get('total_issues', 0)} 个")
        print(f"   关键问题: {analysis.get('critical_issues', 0)} 个")
        print(f"   建议数量: {analysis.get('recommendations_count', 0)} 个")
        
        # 显示详细建议
        top_recommendations = analysis.get('top_recommendations', [])
        if top_recommendations:
            print(f"\n💡 详细建议:")
            for i, rec in enumerate(top_recommendations, 1):
                priority = rec.get('priority', '中')
                description = rec.get('description', '')
                confidence = rec.get('confidence', 0)
                print(f"   {i}. [{priority}] {description} (置信度: {confidence:.2f})")

    async def interactive_mode(self):
        """交互式模式"""
        print("🎯 增强版知识图谱客户端")
        print("功能选项:")
        print("  1. 输入文本 -> 构建知识图谱")
        print("  2. 输入 'analyze:文本' -> 纯分析模式")
        print("  3. 输入 'build+analyze:文本' -> 构建+分析模式")
        print("  4. 输入 'quit' -> 退出")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n📝 请输入: ").strip()
                if user_input.lower() == 'quit':
                    break

                if not user_input:
                    print("❌ 请输入有效的内容")
                    continue

                # 解析输入
                if user_input.startswith('analyze:'):
                    text = user_input[8:].strip()
                    print("\n🔍 正在进行知识图谱分析...")
                    result = await self.analyze_knowledge_graph(text)
                    self.display_analysis_result(result)
                    
                elif user_input.startswith('build+analyze:'):
                    text = user_input[14:].strip()
                    print("\n🔄 正在构建并分析知识图谱...")
                    result = await self.build_and_analyze(text)
                    self.display_result(result)
                    
                else:
                    # 默认构建模式
                    print("\n🔄 正在构建知识图谱...")
                    result = await self.build_knowledge_graph(user_input)
                    self.display_result(result)

            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    """主函数"""
    client = EnhancedKnowledgeGraphClient()
    try:
        print("🔗 正在连接到增强版知识图谱服务器...")
        await client.connect_to_server()
        print("✅ 连接成功!")

        # 检查可用工具
        tools_response = await client.session.list_tools()
        available_tools = [tool.name for tool in tools_response.tools]
        print(f"🔧 可用工具: {', '.join(available_tools)}")

        # 交互式模式
        await client.interactive_mode()

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 