#!/usr/bin/env python3
"""
简化版知识图谱构建MCP客户端
用于测试简化版服务器
"""
import asyncio
import json
import sys
import subprocess
import argparse
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def get_user_input():
    """获取用户输入的文本"""
    parser = argparse.ArgumentParser(description='知识图谱构建客户端')
    parser.add_argument('--text', '-t', type=str, help='要分析的文本内容')
    parser.add_argument('--file', '-f', type=str, help='包含文本的文件路径')
    parser.add_argument('--output', '-o', type=str, default='enhanced_kg.html', help='输出文件名（默认：enhanced_kg.html）')
    parser.add_argument('--no-enhance', action='store_true', help='跳过自动增强')
    parser.add_argument('--no-visualization', action='store_true', help='跳过生成可视化')
    parser.add_argument('--enhance-relations', action='store_true', help='启用自动关系增强（可能产生全连接图）')
    
    args = parser.parse_args()
    
    # 从命令行参数获取文本
    if args.text:
        return args.text, args
    
    # 从文件读取文本
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    print(f"📖 从文件读取: {args.file}")
                    return text, args
                else:
                    print(f"❌ 文件 {args.file} 是空的")
        except FileNotFoundError:
            print(f"❌ 文件 {args.file} 不存在")
        except Exception as e:
            print(f"❌ 读取文件 {args.file} 时出错: {e}")
    
    # 交互式输入
    print("📝 请输入要分析的文本内容:")
    print("   (输入 'quit' 或 'exit' 退出，输入 'demo' 使用示例文本)")
    print("   (支持多行输入，完成后按 Ctrl+D (Linux/Mac) 或 Ctrl+Z 然后回车 (Windows))")
    print("-" * 50)
    
    try:
        lines = []
        while True:
            try:
                line = input()
                if line.lower() in ['quit', 'exit']:
                    print("👋 再见！")
                    sys.exit(0)
                elif line.lower() == 'demo':
                    demo_text = "本项目实现了基于大型语言模型（LLM）的文本隐写术，使用GPT-2模型在生成的文本中隐藏信息。"
                    print("🎯 使用示例文本")
                    return demo_text, args
                lines.append(line)
            except EOFError:
                break
        
        text = '\n'.join(lines).strip()
        if not text:
            print("❌ 没有输入任何文本")
            sys.exit(1)
        
        return text, args
        
    except KeyboardInterrupt:
        print("\n👋 再见！")
        sys.exit(0)

async def test_simple_server():
    """测试简化版服务器"""
    print("🚀 启动简化版知识图谱构建客户端")
    
    # 获取用户输入
    text_input, args = get_user_input()
    
    print(f"\n📄 输入文本预览 ({len(text_input)} 字符):")
    preview = text_input[:200] + "..." if len(text_input) > 200 else text_input
    print(f"   {preview}")
    print("-" * 50)
    
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
                
                print("\n🔍 开始知识图谱构建、分析和增强...")
                
                # 调用工具
                result = await session.call_tool(
                    name="build_and_analyze_kg",
                    arguments={
                        "text": text_input,
                        "auto_enhance": not args.no_enhance,
                        "generate_visualization": not args.no_visualization,
                        "output_file": args.output,
                        "auto_enhance_relations": args.enhance_relations
                    }
                )
                
                # 解析结果
                if result.content:
                    content = result.content[0].text
                    data = json.loads(content)
                    
                    print(f"✅ 处理成功！")
                    print(f"   - 处理时间: {data.get('processing_time', 0):.2f}秒")
                    
                    # 原始知识图谱信息
                    original_kg = data.get('original_knowledge_graph', {})
                    print(f"   - 原始实体数量: {original_kg.get('entities_count', 0)}")
                    print(f"   - 原始关系数量: {original_kg.get('relations_count', 0)}")
                    print(f"   - 原始三元组数量: {original_kg.get('triples_count', 0)}")
                    
                    # 分析结果
                    analysis = data.get('analysis_results', {})
                    print(f"   - 质量评分: {analysis.get('quality_score', 0):.1f}")
                    print(f"   - 发现问题数: {analysis.get('total_issues', 0)}")
                    print(f"   - 建议数量: {analysis.get('recommendations_count', 0)}")
                    
                    # 增强结果
                    enhancement = data.get('enhancement_results', {})
                    if enhancement.get('enhancement_applied'):
                        print(f"   - 增强后实体数量: {enhancement.get('final_entities_count', 0)}")
                        print(f"   - 增强后关系数量: {enhancement.get('final_relations_count', 0)}")
                        print(f"   - 增强后三元组数量: {enhancement.get('final_triples_count', 0)}")
                        print(f"   - 应用的增强数量: {len(enhancement.get('applied_enhancements', []))}")
                    
                    # 可视化结果
                    visualization = data.get('visualization', {})
                    if visualization.get('visualization_generated'):
                        viz_info = visualization.get('visualization_info', {})
                        print(f"   - 可视化文件: {viz_info.get('file_path', 'N/A')}")
                        print(f"   - 访问URL: {viz_info.get('file_url', 'N/A')}")
                    
                    # 显示前几个建议
                    recommendations = analysis.get('top_recommendations', [])
                    if recommendations:
                        print("\n📝 主要建议:")
                        for i, rec in enumerate(recommendations[:3]):
                            print(f"   {i+1}. {rec.get('description', '无描述')} (优先级: {rec.get('priority', '未知')})")
                    
                    # 显示应用的增强
                    applied_enhancements = enhancement.get('applied_enhancements', [])
                    if applied_enhancements:
                        print(f"\n🔧 应用的增强 ({len(applied_enhancements)}个):")
                        for i, enh in enumerate(applied_enhancements[:3]):
                            print(f"   {i+1}. {enh.get('description', '无描述')} (置信度: {enh.get('confidence', 0):.2f})")
                    
                    print(f"\n🎉 处理完成！结果已保存到: {args.output}")
                else:
                    print("❌ 没有收到结果")
                    
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

def show_usage_examples():
    """显示使用示例"""
    print("\n💡 使用示例:")
    print("   # 交互式输入")
    print("   python kg_client_simple.py")
    print()
    print("   # 命令行直接指定文本")
    print("   python kg_client_simple.py --text \"人工智能是计算机科学的一个分支\"")
    print()
    print("   # 从文件读取")
    print("   python kg_client_simple.py --file input.txt")
    print()
    print("   # 自定义输出文件")
    print("   python kg_client_simple.py --text \"示例文本\" --output my_kg.html")
    print()
    print("   # 跳过增强和可视化")
    print("   python kg_client_simple.py --text \"示例文本\" --no-enhance --no-visualization")
    print()
    print("   # 启用关系增强")
    print("   python kg_client_simple.py --text \"示例文本\" --enhance-relations")

if __name__ == "__main__":
    # 如果用户只输入 --help 或 -h，显示使用示例
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_usage_examples()
        sys.exit(0)
    
    asyncio.run(test_simple_server())
