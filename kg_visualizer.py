# kg_visualizer.py

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import math
import random
from typing import List, Dict, Any, Tuple
from kg_utils import Triple
import colorsys


class KnowledgeGraphVisualizer:
    """知识图谱可视化器"""

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_colors = {}
        self.relation_colors = {}
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]

    def create_graph_from_triples(self, triples: List[Triple]) -> nx.Graph:
        """从三元组创建网络图"""
        self.graph.clear()

        # 添加节点和边
        for triple in triples:
            # 添加节点
            if not self.graph.has_node(triple.head):
                self.graph.add_node(triple.head, type='entity')
            if not self.graph.has_node(triple.tail):
                self.graph.add_node(triple.tail, type='entity')

            # 添加边
            self.graph.add_edge(
                triple.head,
                triple.tail,
                relation=triple.relation,
                confidence=triple.confidence,
                weight=triple.confidence
            )

        return self.graph

    def assign_colors(self, entities: List[str], relations: List[str]):
        """为实体和关系分配颜色"""
        # 为实体类型分配颜色
        entity_types = set()
        for entity in entities:
            # 简单的实体类型判断
            if any(keyword in entity for keyword in ['大学', '学院', '机构']):
                entity_types.add('Organization')
            elif any(keyword in entity for keyword in ['市', '省', '区', '县', '国']):
                entity_types.add('Place')
            elif len(entity) <= 4 and any(char in entity for char in '张王李赵刘陈杨黄周吴'):
                entity_types.add('Person')
            else:
                entity_types.add('Other')

        # 分配颜色
        type_colors = {
            'Person': '#FF6B6B',      # 红色 - 人物
            'Place': '#4ECDC4',       # 青色 - 地点
            'Organization': '#45B7D1', # 蓝色 - 组织
            'Other': '#96CEB4'        # 绿色 - 其他
        }

        for entity in entities:
            if any(keyword in entity for keyword in ['大学', '学院', '机构']):
                self.entity_colors[entity] = type_colors['Organization']
            elif any(keyword in entity for keyword in ['市', '省', '区', '县', '国']):
                self.entity_colors[entity] = type_colors['Place']
            elif len(entity) <= 4 and any(char in entity for char in '张王李赵刘陈杨黄周吴'):
                self.entity_colors[entity] = type_colors['Person']
            else:
                self.entity_colors[entity] = type_colors['Other']

        # 为关系分配颜色
        for i, relation in enumerate(relations):
            self.relation_colors[relation] = self.color_palette[i % len(self.color_palette)]

    def create_interactive_plot(self, triples: List[Triple], title: str = "知识图谱可视化") -> go.Figure:
        """创建交互式图谱"""
        # 创建图
        graph = self.create_graph_from_triples(triples)

        if len(graph.nodes()) == 0:
            # 创建空图
            fig = go.Figure()
            fig.add_annotation(
                text="暂无数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig

        # 获取实体和关系
        entities = list(graph.nodes())
        relations = list(set([data['relation'] for _, _, data in graph.edges(data=True)]))

        # 分配颜色
        self.assign_colors(entities, relations)

        # 使用spring布局
        try:
            pos = nx.spring_layout(graph, k=3, iterations=50)
        except:
            # 如果spring布局失败，使用随机布局
            pos = {node: (random.random(), random.random()) for node in graph.nodes()}

        # 创建边的轨迹
        edge_traces = []
        edge_info = []

        for relation in relations:
            edge_x = []
            edge_y = []
            edge_text = []

            for edge in graph.edges(data=True):
                if edge[2]['relation'] == relation:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]

                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                    # 计算边的中点用于显示关系标签
                    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                    edge_info.append({
                        'x': mid_x, 'y': mid_y,
                        'text': f"{relation}<br>置信度: {edge[2]['confidence']:.3f}",
                        'relation': relation
                    })

            if edge_x:  # 只有当有边时才添加轨迹
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color=self.relation_colors.get(relation, '#888')),
                    hoverinfo='none',
                    mode='lines',
                    name=f'关系: {relation}',
                    showlegend=True
                )
                edge_traces.append(edge_trace)

        # 创建节点轨迹
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # 计算节点大小（基于连接数）
            degree = graph.degree(node)
            size = max(20, min(50, 20 + degree * 5))
            node_sizes.append(size)

            # 节点颜色
            node_colors.append(self.entity_colors.get(node, '#96CEB4'))

            # 节点信息
            connections = list(graph.neighbors(node))
            node_info = f"实体: {node}<br>连接数: {degree}"
            if connections:
                node_info += f"<br>连接到: {', '.join(connections[:3])}"
                if len(connections) > 3:
                    node_info += f"<br>等 {len(connections)} 个实体"

            node_text.append(node_info)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_text,
            text=[node[:10] + '...' if len(node) > 10 else node for node in graph.nodes()],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name='实体',
            showlegend=False
        )

        # 创建关系标签轨迹
        if edge_info:
            relation_trace = go.Scatter(
                x=[info['x'] for info in edge_info],
                y=[info['y'] for info in edge_info],
                mode='text',
                text=[info['relation'] for info in edge_info],
                textfont=dict(size=8, color='black'),
                hoverinfo='text',
                hovertext=[info['text'] for info in edge_info],
                showlegend=False,
                name='关系标签'
            )
        else:
            relation_trace = None

        # 创建图形
        fig = go.Figure()

        # 添加边
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)

        # 添加节点
        fig.add_trace(node_trace)

        # 添加关系标签
        if relation_trace:
            fig.add_trace(relation_trace)

        # 更新布局
        layout_config = {
            'showlegend': True,
            'hovermode': 'closest',
            'xaxis': dict(showgrid=False, zeroline=False, showticklabels=False),
            'yaxis': dict(showgrid=False, zeroline=False, showticklabels=False),
            'plot_bgcolor': 'white',
            'height': 600
        }

        # 如果有标题则显示标题和说明，否则使用最小边距
        if title and title.strip():
            layout_config.update({
                'title': dict(
                    text=title,
                    x=0.5,
                    font=dict(size=20)
                ),
                'margin': dict(b=20, l=5, r=5, t=40),
                'annotations': [
                    dict(
                        text="拖拽节点可以移动 | 鼠标悬停查看详情 | 点击图例隐藏/显示关系",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12, color='gray')
                    )
                ]
            })
        else:
            # 无标题模式，使用最小边距
            layout_config.update({
                'margin': dict(b=5, l=5, r=5, t=5)
            })

        fig.update_layout(**layout_config)

        return fig

    def create_statistics_plot(self, triples: List[Triple]) -> go.Figure:
        """创建统计图表"""
        if not triples:
            fig = go.Figure()
            fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
            return fig

        # 统计数据
        relations = [triple.relation for triple in triples]
        confidences = [triple.confidence for triple in triples]

        # 关系分布
        relation_counts = {}
        for relation in relations:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        # 置信度分布
        confidence_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        confidence_counts = [0] * 5

        for conf in confidences:
            if conf <= 0.2:
                confidence_counts[0] += 1
            elif conf <= 0.4:
                confidence_counts[1] += 1
            elif conf <= 0.6:
                confidence_counts[2] += 1
            elif conf <= 0.8:
                confidence_counts[3] += 1
            else:
                confidence_counts[4] += 1

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('关系类型分布', '置信度分布', '三元组置信度', '统计摘要'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )

        # 关系分布饼图
        fig.add_trace(
            go.Pie(
                labels=list(relation_counts.keys()),
                values=list(relation_counts.values()),
                name="关系分布"
            ),
            row=1, col=1
        )

        # 置信度分布柱状图
        fig.add_trace(
            go.Bar(
                x=confidence_ranges,
                y=confidence_counts,
                name="置信度分布",
                marker_color='lightblue'
            ),
            row=1, col=2
        )

        # 三元组置信度散点图
        fig.add_trace(
            go.Scatter(
                x=list(range(len(confidences))),
                y=confidences,
                mode='markers+lines',
                name="置信度趋势",
                marker=dict(size=8, color=confidences, colorscale='Viridis', showscale=True)
            ),
            row=2, col=1
        )

        # 统计摘要表格
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        stats_data = [
            ['总三元组数', len(triples)],
            ['关系类型数', len(relation_counts)],
            ['平均置信度', f'{avg_confidence:.3f}'],
            ['最高置信度', f'{max(confidences):.3f}' if confidences else '0.000'],
            ['最低置信度', f'{min(confidences):.3f}' if confidences else '0.000']
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=['指标', '数值'], fill_color='lightgray'),
                cells=dict(values=[[row[0] for row in stats_data],
                                 [row[1] for row in stats_data]],
                          fill_color='white')
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text="知识图谱统计分析",
            showlegend=False,
            height=800
        )

        return fig

    def generate_html_report(self, triples: List[Triple], entities: List[str],
                           relations: List[str], title: str = "知识图谱分析报告") -> str:
        """生成完整的HTML报告"""

        # 创建图谱可视化
        graph_fig = self.create_interactive_plot(triples, "知识图谱网络图")

        # 创建统计图表
        stats_fig = self.create_statistics_plot(triples)

        # 转换为HTML
        graph_html = graph_fig.to_html(include_plotlyjs='cdn', div_id="graph-plot")
        stats_html = stats_fig.to_html(include_plotlyjs='cdn', div_id="stats-plot")

        # 创建完整的HTML报告
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .summary-card p {{
            margin: 0;
            opacity: 0.9;
        }}
        .plot-container {{
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .entity-list, .relation-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }}
        .tag {{
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            border: 1px solid #bbdefb;
        }}
        .relation-tag {{
            background-color: #f3e5f5;
            color: #7b1fa2;
            border-color: #ce93d8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>基于三阶段MCP协议的知识图谱构建与可视化</p>
        </div>

        <div class="content">
            <div class="section">
                <h2>📊 概览统计</h2>
                <div class="summary">
                    <div class="summary-card">
                        <h3>{len(entities)}</h3>
                        <p>实体数量</p>
                    </div>
                    <div class="summary-card">
                        <h3>{len(relations)}</h3>
                        <p>关系类型</p>
                    </div>
                    <div class="summary-card">
                        <h3>{len(triples)}</h3>
                        <p>三元组数量</p>
                    </div>
                    <div class="summary-card">
                        <h3>{"0.000" if len(triples) == 0 else f"{sum(t.confidence for t in triples)/len(triples):.3f}"}</h3>
                        <p>平均置信度</p>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🕸️ 知识图谱网络图</h2>
                <div class="plot-container">
                    {graph_html.split('<body>')[1].split('</body>')[0] if '<body>' in graph_html else graph_html}
                </div>
            </div>

            <div class="section">
                <h2>统计分析</h2>
                <div class="plot-container">
                    {stats_html.split('<body>')[1].split('</body>')[0] if '<body>' in stats_html else stats_html}
                </div>
            </div>

            <div class="section">
                <h2>实体列表</h2>
                <div class="entity-list">
                    {''.join([f'<span class="tag">{entity}</span>' for entity in entities[:20]])}
                    {f'<span class="tag">... 还有 {len(entities)-20} 个实体</span>' if len(entities) > 20 else ''}
                </div>
            </div>

            <div class="section">
                <h2>🔗 关系类型</h2>
                <div class="relation-list">
                    {''.join([f'<span class="tag relation-tag">{relation}</span>' for relation in relations])}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        """

        return html_content

    def generate_simple_html(self, triples: List[Triple], entities: List[str],
                            relations: List[str], title: str = "知识图谱可视化") -> str:
        """生成简洁的HTML可视化，只包含知识图谱网络图，无标题"""
        if not triples and not entities:
            # 如果没有数据，生成一个空的图表
            graph_html = self.create_empty_graph()
        else:
            # 生成网络图，不显示标题
            graph_fig = self.create_interactive_plot(triples, "")
            graph_html = graph_fig.to_html(include_plotlyjs='cdn', div_id="graph-plot")

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }}
        .graph-container {{
            width: 100vw;
            height: 100vh;
            margin: 0;
            padding: 0;
        }}
    </style>
</head>
<body>
    <div class="graph-container">
        {graph_html.split('<body>')[1].split('</body>')[0] if '<body>' in graph_html else graph_html}
    </div>
</body>
</html>
        """

        return html_content

    def create_empty_graph(self) -> str:
        """创建空的图表显示"""
        return """
        <div style="display: flex; align-items: center; justify-content: center; height: 600px; color: #666; font-size: 18px;">
            <div style="text-align: center;">
                <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
                <div>暂无数据可视化</div>
                <div style="font-size: 14px; margin-top: 10px; color: #999;">
                    请输入包含实体和关系的文本数据
                </div>
            </div>
        </div>
        """

    def save_visualization(self, triples: List[Triple], entities: List[str],
                          relations: List[str], filename: str = "knowledge_graph_visualization.html"):
        """保存可视化结果到HTML文件"""
        html_content = self.generate_html_report(triples, entities, relations)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filename

    def save_simple_visualization(self, triples: List[Triple], entities: List[str],
                                 relations: List[str], filename: str = "knowledge_graph.html"):
        """保存简洁的可视化结果到HTML文件，只包含知识图谱网络图"""
        html_content = self.generate_simple_html(triples, entities, relations)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filename
