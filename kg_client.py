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


class KnowledgeGraphClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    async def connect_to_server(self):
        """连接到知识图谱服务器"""
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'kg_server.py'],
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
        构建知识图谱

        Args:
            text: 要处理的文本数据
            output_file: 可视化输出文件名

        Returns:
            包含知识图谱构建结果的字典
        """
        try:
            # 直接调用知识图谱构建工具
            result = await self.session.call_tool("build_knowledge_graph", {
                "text": text,
                "output_file": output_file
            })

            # 解析结果
            result_text = result.content[0].text
            result_data = json.loads(result_text)

            return result_data

        except Exception as e:
            return {
                "success": False,
                "error": f"构建知识图谱时发生错误: {str(e)}"
            }

    async def process_text_with_ai(self, text: str) -> str:
        """
        使用AI处理文本并构建知识图谱

        Args:
            text: 用户输入的文本

        Returns:
            AI生成的回答
        """
        system_prompt = (
            "你是一个知识图谱构建助手。"
            "你可以帮助用户从文本中构建知识图谱。"
            "当用户提供文本时，你必须调用 build_knowledge_graph 工具来构建知识图谱。"
            "然后基于构建结果为用户提供详细的分析和说明。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        # 获取工具信息
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # 请求大模型
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages,
            tools=available_tools
        )

        content = response.choices[0]
        if content.finish_reason == "tool_calls":
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            result = await self.session.call_tool(tool_name, tool_args)

            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id,
            })

            # 继续向大模型发送请求以生成最终回答
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=messages
            )
            return response.choices[0].message.content

        return content.message.content

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

    async def interactive_mode(self):
        """交互式模式"""
        print("🎯 知识图谱构建客户端")
        print("输入文本来构建知识图谱，输入 'quit' 退出")
        print("=" * 50)

        while True:
            try:
                text = input("\n📝 请输入文本: ").strip()
                if text.lower() == 'quit':
                    break

                if not text:
                    print("❌ 请输入有效的文本")
                    continue

                print("\n🔄 正在构建知识图谱...")
                result = await self.build_knowledge_graph(text)
                self.display_result(result)

            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()

    async def batch_mode(self, texts: list, output_prefix: str = "kg"):
        """批量处理模式"""
        print(f"🔄 批量处理 {len(texts)} 个文本...")

        for i, text in enumerate(texts, 1):
            print(f"\n处理第 {i}/{len(texts)} 个文本...")
            output_file = f"{output_prefix}_{i}.html"
            result = await self.build_knowledge_graph(text, output_file)

            if result.get("success"):
                print(f"✅ 第 {i} 个文本处理完成: {output_file}")
            else:
                print(f"❌ 第 {i} 个文本处理失败: {result.get('error')}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    """主函数"""
    client = KnowledgeGraphClient()
    try:
        print("🔗 正在连接到知识图谱服务器...")
        await client.connect_to_server()
        print("✅ 连接成功!")

        # 检查命令行参数
        import sys
        if len(sys.argv) > 1:
            # 批量模式
            texts = sys.argv[1:]
            await client.batch_mode(texts)
        else:
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
