# LLM 知识图谱构建集成指南

## 概述

已成功将您的大语言模型实体和关系抽取功能集成到 `kg_utils.py` 中。现在您的知识图谱构建系统支持两种模式：

- **LLM模式**: 使用Silicon Flow API和Qwen2.5-7B-Instruct模型进行高质量的实体和关系抽取
- **规则模式**: 使用硬编码规则进行传统的实体和关系抽取

## 主要改进

### 1. 新增 `ChineseEntityRelationExtractor` 类
- 集成了您原有的LLM实体抽取功能
- 支持实体抽取、关系抽取和三元组生成
- 具备错误处理和API调用功能

### 2. 增强的 `KnowledgeGraphBuilder` 类
- **构造函数**: 现在接受可选的 `api_key` 参数
- **LLM方法**: 
  - `_extract_entities_with_llm()`: 使用LLM提取实体
  - `_extract_relations_with_llm()`: 使用LLM提取关系
  - `_generate_triples_with_llm()`: 使用LLM生成三元组
- **智能回退**: LLM失败时自动回退到规则方法
- **置信度计算**: 专门针对LLM结果的置信度评估

## 使用方法

### 基本用法

```python
import asyncio
from kg_utils import KnowledgeGraphBuilder

async def main():
    # 使用LLM模式（需要API密钥）
    api_key = "your-silicon-flow-api-key"
    kg_builder = KnowledgeGraphBuilder(api_key=api_key)
    
    text = "张三是阿里巴巴公司的CEO，阿里巴巴总部位于杭州。"
    
    # 构建知识图谱（使用LLM）
    result = await kg_builder.build_graph(text, use_llm=True)
    
    print("实体:", result['entities'])
    print("关系:", result['relations'])
    print("三元组:", result['triples'])

asyncio.run(main())
```

### 对比测试

```python
# 创建两个构建器实例
kg_builder_llm = KnowledgeGraphBuilder(api_key="your-api-key")  # LLM模式
kg_builder_rule = KnowledgeGraphBuilder()  # 规则模式

# 对比结果
llm_result = await kg_builder_llm.build_graph(text, use_llm=True)
rule_result = await kg_builder_rule.build_graph(text, use_llm=False)
```

## 配置选项

### 构造参数
- `api_key` (可选): Silicon Flow API密钥，提供则启用LLM功能

### build_graph 参数
- `data`: 输入文本
- `use_llm`: 是否使用LLM（默认True，需要提供api_key）

## 错误处理

系统具备完善的错误处理机制：

1. **API调用失败**: 自动回退到规则方法
2. **网络错误**: 显示错误信息并使用备用方案
3. **解析错误**: 容错处理，确保系统稳定运行

## 性能对比

### LLM模式优势
- ✅ 更准确的实体识别
- ✅ 更丰富的关系类型
- ✅ 更好的上下文理解
- ✅ 更高的置信度

### 规则模式优势
- ✅ 响应速度快
- ✅ 无需网络连接
- ✅ 成本低
- ✅ 可控性强

## 运行示例

执行以下命令运行完整示例：

```bash
python kg_example.py
```

这将展示LLM模式和规则模式的对比结果。

## 注意事项

1. **API密钥**: 确保您的Silicon Flow API密钥有效且有足够的调用次数
2. **网络连接**: LLM模式需要稳定的网络连接
3. **响应时间**: LLM模式响应时间较长，适合对准确性要求高的场景
4. **成本考虑**: LLM调用会产生API费用，根据需要选择合适的模式

## 扩展建议

1. **缓存机制**: 对相同文本的结果进行缓存，提高效率
2. **批量处理**: 支持批量文本处理，减少API调用次数
3. **模型选择**: 支持不同的LLM模型选择
4. **结果融合**: 结合LLM和规则方法的结果，提高准确性

## 技术支持

如有问题，请检查：
1. API密钥是否正确
2. 网络连接是否正常
3. 依赖包是否完整安装
4. 输入文本格式是否正确 