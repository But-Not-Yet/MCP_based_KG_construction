import json
import asyncio
import os
import webbrowser
from typing import Optional
from contextlib import AsyncExitStack
import logging

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
logging.basicConfig(level=logging.INFO)

class EnhancedKGClient:
    """
    增强版知识图谱客户端
    - 专注于“构建 + 自动内容增强”的核心功能
    """
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self):
        """连接到增强版知识图谱服务器"""
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'kg_server_enhanced.py'],
            env=os.environ
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await self.session.initialize()
        # 验证工具是否存在
        tools_response = await self.session.list_tools()
        tool_names = [t.name for t in tools_response.tools]
        if "build_and_analyze_kg" not in tool_names:
            raise RuntimeError("错误：服务器未提供 'build_and_analyze_kg' 工具。")

    async def build_and_enhance_kg(self, text: str) -> dict:
        """
        调用服务器的 build_and_analyze_kg 工具来构建并自动增强知识图谱。
        """
        try:
            result = await self.session.call_tool("build_and_analyze_kg", {
                "text": text,
                "auto_enhance": True  # 始终开启自动增强
            })
            result_text = result.content[0].text
            return json.loads(result_text)
        except Exception as e:
            logging.error(f"调用 build_and_analyze_kg 工具时出错: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def display_result(self, result: dict):
        """显示增强后的知识图谱构建结果"""
        if not result.get("success"):
            print(f"\n❌ 处理失败: {result.get('error', '未知错误')}")
            if 'error_details' in result:
                print("\n--- 错误详情 ---")
                print(result['error_details'])
                print("-----------------")
            return

        print("\n✅ 知识图谱构建与增强成功!")
        print(f"⏱️  处理时间: {result.get('processing_time', 0):.3f} 秒")

        summary = result.get("summary", {})
        enhancement_summary = result.get("stages", {}).get("enhancement_results", {}).get("enhancement_summary", {})
        
        print("\n--- 增强摘要 ---")
        print(f"  原始实体数: {enhancement_summary.get('original_entity_count', 'N/A')}")
        print(f"  增强后实体数: {enhancement_summary.get('enhanced_entity_count', 'N/A')}")
        print(f"  原始关系数: {enhancement_summary.get('original_relation_count', 'N/A')}")
        print(f"  增强后关系数: {enhancement_summary.get('enhanced_relation_count', 'N/A')}")

        viz = result.get("stages", {}).get("visualization", {})
        file_path = viz.get("file_path")
        if file_path and os.path.exists(file_path):
            print(f"\n🎨 可视化文件已生成: {file_path}")
            try:
                webbrowser.open(f"file:///{os.path.abspath(file_path)}")
                print("   已在默认浏览器中打开。")
            except Exception as e:
                print(f"   无法自动打开浏览器: {e}")
        else:
            print("\n🎨 未生成可视化文件。")

    async def interactive_mode(self):
        """交互式模式"""
        print("\n🎯 增强版知识图谱客户端 (构建并自动增强模式)")
        print("   - 输入任意文本以构建和增强知识图谱。")
        print("   - 输入 'quit' 退出。")
        print("=" * 50)

        while True:
            try:
                text = input("\n📝 请输入文本: ").strip()
                if text.lower() == 'quit':
                    break
                if not text:
                    continue

                print("\n🔄 正在构建并增强知识图谱，请稍候...")
                result = await self.build_and_enhance_kg(text)
                self.display_result(result)

            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 再见!")
                break
            except Exception as e:
                logging.error(f"交互模式中发生未知错误: {e}", exc_info=True)

    async def cleanup(self):
        if self.exit_stack:
            await self.exit_stack.aclose()


async def main():
    """主函数"""
    client = EnhancedKGClient()
    try:
        print("🔗 正在连接到增强版知识图谱服务器...")
        await client.connect_to_server()
        print("✅ 连接成功!")
        await client.interactive_mode()
    except Exception as e:
        logging.error(f"启动或连接服务器时发生致命错误: {e}", exc_info=True)
    finally:
        print("\n shutting down...")
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已中断。") 