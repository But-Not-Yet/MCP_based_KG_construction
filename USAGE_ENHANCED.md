# 🚀 知识图谱系统使用指南

## 📋 现在您有两种运行方式

### 方式1：原有系统（推荐新手）
```bash
# 原来的方式仍然完全可用
uv run kg_server.py
```

### 方式2：增强版系统（新功能）
```bash
# 使用增强版，包含高级分析功能
uv run kg_server_enhanced.py
```

## 🔧 **增强版新功能**

### 1. 原有功能保持不变
- `build_knowledge_graph` - 原有的知识图谱构建功能

### 2. 新增高级分析功能
- `analyze_knowledge_graph` - 纯分析功能
- `build_and_analyze_kg` - 构建+分析一体化

## 🎯 **使用示例**

### 在MCP Inspector中使用
```bash
# 启动增强版服务器
npx -y @modelcontextprotocol/inspector uv run kg_server_enhanced.py
```

### 在Cherry Studio中使用
配置JSON（注意修改路径）：
```json
{
  "mcpServers": {
    "kg_server_enhanced": {
      "command": "uv",
      "args": [
        "--directory",
        "您的项目路径",
        "run",
        "kg_server_enhanced.py"
      ]
    }
  }
}
```

## 📊 **新功能对比**

| 功能 | 原版 | 增强版 |
|------|------|--------|
| 知识图谱构建 | ✅ | ✅ |
| 数据质量评估 | ✅ | ✅ |
| 知识补全 | ✅ | ✅ |
| 可视化 | ✅ | ✅ |
| 全局分析 | ❌ | ✅ |
| 细节分析 | ❌ | ✅ |
| 智能建议 | ❌ | ✅ |
| 质量评分 | ❌ | ✅ |

## 🚀 **快速开始**

### 1. 测试原有功能
```bash
# 启动原版服务器
uv run kg_server.py

# 在MCP Inspector中测试
# 输入文本："张三是阿里巴巴的工程师"
# 选择工具：build_knowledge_graph
```

### 2. 测试新增功能
```bash
# 启动增强版服务器
uv run kg_server_enhanced.py

# 在MCP Inspector中测试
# 输入文本："张三是阿里巴巴的工程师"
# 选择工具：build_and_analyze_kg
```

## 💡 **建议**

1. **首次使用**：建议先用原版熟悉系统
2. **需要深度分析**：使用增强版的分析功能
3. **生产环境**：根据需求选择合适的版本
4. **开发测试**：两个版本都可以并存使用

## 🔍 **故障排除**

### 如果分析功能不可用
```bash
# 检查文件是否存在
ls content_enhancement/
# 应该包含：
# - global_analysis.py
# - entity_detail_analyzer.py
# - analysis_pipeline.py
```

### 依赖问题
```bash
# 重新安装依赖
uv sync
```

## 📈 **性能建议**

- 小数据量：使用原版即可
- 大数据量：使用增强版并启用并行分析
- 需要详细报告：使用 `build_and_analyze_kg` 工具

您现在可以：
1. 继续使用原来的方式 `uv run kg_server.py`
2. 或者尝试新的增强版 `uv run kg_server_enhanced.py`

两种方式都完全可用！🎉 