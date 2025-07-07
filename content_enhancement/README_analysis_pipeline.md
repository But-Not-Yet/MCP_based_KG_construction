# 知识图谱分析流程使用指南

## 📖 概述

这个分析流程系统包含三个核心模块：

1. **`global_analysis.py`** - 全局分析模块
2. **`entity_detail_analyzer.py`** - 实体细节分析模块  
3. **`analysis_pipeline.py`** - 分析流程控制器

## 🏗️ 系统架构

```
输入数据 → 分析流程控制器 → 并行/串行执行 → 结果整合 → 输出报告
    ↓              ↓                    ↓
原始文本        预处理&格式转换          全局分析
实体列表        →                    细节分析
关系列表        数据分发               ↓
                                    结果整合
```

## 🚀 快速开始

### 1. 基础用法

```python
import asyncio
from analysis_pipeline import analyze_knowledge_graph

async def main():
    # 准备数据
    text = "您的原始文本..."
    entities = [
        {
            'name': '张三',
            'type': 'Person',
            'attributes': {'职业': '工程师'},
            'relations': ['工作于']
        }
    ]
    relations = [
        {
            'name': '工作于',
            'source': '张三',
            'target': '阿里巴巴',
            'type': '雇佣关系'
        }
    ]
    
    # 执行分析
    result = await analyze_knowledge_graph(text, entities, relations)
    
    # 查看结果
    print(f"质量评分: {result.quality_metrics['overall_score']}")
    print(f"建议数量: {len(result.integrated_recommendations)}")

asyncio.run(main())
```

### 2. 高级配置

```python
from analysis_pipeline import AnalysisPipeline, AnalysisConfig, InputData

# 自定义配置
config = AnalysisConfig(
    enable_global_analysis=True,      # 启用全局分析
    enable_detail_analysis=True,      # 启用细节分析
    similarity_threshold=0.4,         # 相似度阈值
    causal_threshold=0.3,            # 因果关系阈值
    confidence_threshold=0.6,         # 置信度阈值
    max_recommendations=15,           # 最大建议数
    parallel_execution=True           # 并行执行
)

# 创建分析器
pipeline = AnalysisPipeline(config)

# 准备输入数据
input_data = InputData(
    original_text="您的文本...",
    entities=[...],
    relations=[...],
    metadata={"source": "用户输入", "version": "1.0"}
)

# 执行分析
result = await pipeline.run_analysis(input_data)
```

## 📊 输出结果解析

### 分析结果结构

```python
class AnalysisOutput:
    timestamp: str                    # 分析时间戳
    input_summary: Dict               # 输入数据摘要
    global_analysis_results: Dict     # 全局分析结果
    detail_analysis_results: Dict     # 细节分析结果
    integrated_recommendations: List  # 整合建议
    quality_metrics: Dict            # 质量指标
```

### 质量指标说明

```python
quality_metrics = {
    'overall_score': 85.6,           # 整体质量评分 (0-100)
    'issue_count': 12,               # 发现问题总数
    'critical_issues': 2             # 关键问题数量
}
```

### 建议结构

```python
recommendation = {
    'source': 'global_analysis',     # 来源模块
    'module': 'verb_predicate_analysis',  # 具体分析器
    'type': 'add_verb',             # 建议类型
    'description': '为实体添加动词描述',  # 建议描述
    'priority': '高',               # 优先级
    'confidence': 0.85,             # 置信度
    'implementation': {...},         # 实现细节
    'category': '属性补全'           # 建议分类
}
```

## 🔧 详细功能介绍

### 全局分析模块功能

1. **动词/谓语缺失分析**
   - 检测实体描述中缺失的动词
   - 识别流程性关系
   - 推荐相关动词

2. **关键词频率分析**
   - 统计高频关键词
   - 计算实体相似度
   - 推荐相似关系

3. **因果关系分析**
   - 识别现有因果关系
   - 分析层次结构
   - 推断缺失因果关系

4. **直接逻辑关系分析**
   - 分析类别关系
   - 检测循环依赖
   - 发现传递关系

### 细节分析模块功能

1. **属性完整性检查**
   - 基于实体类型模板
   - 检测必需属性缺失
   - 推荐属性补全

2. **逻辑一致性验证**
   - 职业与单位匹配
   - 地理位置层次
   - 时间合理性

3. **局部关联分析**
   - 实体对关联强度
   - 三元组关系分析
   - 共现模式识别

## 🛠️ 自定义扩展

### 添加新的分析规则

```python
# 在 entity_detail_analyzer.py 中添加新规则
new_rule = {
    "rule_name": "自定义规则",
    "condition": lambda entity, attrs: your_condition,
    "check": lambda attrs: your_check_logic,
    "error_message": "错误描述"
}
```

### 扩展实体类型模板

```python
# 添加新的实体类型模板
new_template = AttributeTemplate(
    entity_type="CustomType",
    required_attributes=["必需属性1", "必需属性2"],
    optional_attributes=["可选属性1", "可选属性2"],
    attribute_patterns={
        "属性1": r"正则表达式",
        "属性2": r"正则表达式"
    }
)
```

## 🎯 使用场景

### 1. 知识图谱质量评估
```python
# 评估现有知识图谱质量
result = await analyze_knowledge_graph(text, entities, relations)
quality_score = result.quality_metrics['overall_score']
if quality_score < 70:
    print("知识图谱质量需要改进")
```

### 2. 增量更新建议
```python
# 获取高优先级改进建议
high_priority_recommendations = [
    rec for rec in result.integrated_recommendations 
    if rec['priority'] == '高'
]
```

### 3. 批量分析
```python
async def batch_analysis(documents):
    results = []
    for doc in documents:
        result = await analyze_knowledge_graph(
            doc['text'], doc['entities'], doc['relations']
        )
        results.append(result)
    return results
```

## 📈 性能优化建议

1. **并行执行**: 设置 `parallel_execution=True`
2. **阈值调整**: 根据需求调整各类阈值
3. **结果限制**: 通过 `max_recommendations` 控制输出
4. **选择性分析**: 根据需要启用/禁用特定模块

## 🔍 故障排除

### 常见问题

1. **模块导入错误**
   - 确保所有文件在同一目录或正确的Python路径中

2. **数据格式错误**
   - 检查实体和关系的数据结构是否符合要求

3. **分析结果为空**
   - 检查输入数据是否有效
   - 调整置信度阈值

4. **性能问题**
   - 对于大规模数据，考虑分批处理
   - 调整并行执行配置

## 📝 日志配置

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.INFO)

# 自定义日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🔗 集成示例

### 与现有系统集成

```python
class KnowledgeGraphEnhancer:
    def __init__(self):
        self.pipeline = AnalysisPipeline()
    
    async def enhance_knowledge_graph(self, kg_data):
        # 执行分析
        result = await self.pipeline.run_analysis(kg_data)
        
        # 应用建议
        enhanced_kg = self.apply_recommendations(
            kg_data, result.integrated_recommendations
        )
        
        return enhanced_kg, result
```

通过这个完整的分析流程，您可以：
- 🔍 **全面分析**知识图谱的质量问题
- 🎯 **精准识别**需要改进的具体内容
- 📊 **量化评估**知识图谱的完整性和准确性
- 🚀 **自动生成**具体的改进建议

这个系统为您的知识图谱质量增强提供了完整的技术解决方案！ 