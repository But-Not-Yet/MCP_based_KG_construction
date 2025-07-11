# enhanced_kg_visualizer.py
# 增强版知识图谱可视化器 - 提供更美观的图谱显示

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import math
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from kg_utils import Triple
import colorsys

class EnhancedKnowledgeGraphVisualizer:
    """增强版知识图谱可视化器"""

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_colors = {}
        self.relation_colors = {}
        
        # 现代化配色方案
        self.modern_palette = {
            'Person': {
                'color': '#FF6B6B',
                'gradient': 'linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%)',
                'icon': '👤'
            },
            'Organization': {
                'color': '#4ECDC4', 
                'gradient': 'linear-gradient(135deg, #4ECDC4 0%, #7FDDDD 100%)',
                'icon': '🏢'
            },
            'Location': {
                'color': '#45B7D1',
                'gradient': 'linear-gradient(135deg, #45B7D1 0%, #74C7DD 100%)',
                'icon': '📍'
            },
            'Product': {
                'color': '#96CEB4',
                'gradient': 'linear-gradient(135deg, #96CEB4 0%, #B5D8C7 100%)',
                'icon': '📦'
            },
            'Event': {
                'color': '#FFEAA7',
                'gradient': 'linear-gradient(135deg, #FFEAA7 0%, #FFECB8 100%)',
                'icon': '📅'
            },
            'Date': {
                'color': '#DDA0DD',
                'gradient': 'linear-gradient(135deg, #DDA0DD 0%, #E6B8E6 100%)',
                'icon': '🕒'
            },
            'Number': {
                'color': '#F7DC6F',
                'gradient': 'linear-gradient(135deg, #F7DC6F 0%, #F9E58A 100%)',
                'icon': '🔢'
            },
            'Other': {
                'color': '#BB8FCE',
                'gradient': 'linear-gradient(135deg, #BB8FCE 0%, #C9A4D4 100%)',
                'icon': '❓'
            }
        }
        
        # 关系颜色方案
        self.relation_palette = [
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
            '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#FF5722'
        ]

    def classify_entity_type(self, entity: str, entity_types: Dict[str, str] = None) -> str:
        """智能分类实体类型"""
        if entity_types and entity in entity_types:
            return entity_types[entity]
        
        # 智能分类规则
        if any(keyword in entity for keyword in ['大学', '学院', '机构', '公司', '企业', '集团', '组织', '政府', '部门']):
            return 'Organization'
        elif any(keyword in entity for keyword in ['市', '省', '区', '县', '国', '街', '路', '广场', '大厦', '中心']):
            return 'Location'
        elif len(entity) <= 4 and any(char in entity for char in '张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文'):
            return 'Person'
        elif any(keyword in entity for keyword in ['年', '月', '日', '时', '点', '期间', '时候']):
            return 'Date'
        elif entity.replace('.', '').replace(',', '').isdigit() or any(keyword in entity for keyword in ['万', '亿', '千', '百', '元', '美元', '%', '％']):
            return 'Number'
        elif any(keyword in entity for keyword in ['会议', '活动', '比赛', '项目', '计划', '工程']):
            return 'Event'
        elif any(keyword in entity for keyword in ['产品', '软件', '系统', '平台', '应用', 'App', 'iPhone', 'iPad']):
            return 'Product'
        else:
            return 'Other'

    def create_enhanced_graph(self, triples: List[Triple], entity_types: Dict[str, str] = None) -> nx.Graph:
        """创建增强的网络图"""
        self.graph.clear()

        # 添加节点和边
        for triple in triples:
            # 添加节点
            if not self.graph.has_node(triple.head):
                entity_type = self.classify_entity_type(triple.head, entity_types)
                self.graph.add_node(triple.head, 
                                  type=entity_type,
                                  color=self.modern_palette[entity_type]['color'],
                                  icon=self.modern_palette[entity_type]['icon'])
            
            if not self.graph.has_node(triple.tail):
                entity_type = self.classify_entity_type(triple.tail, entity_types)
                self.graph.add_node(triple.tail, 
                                  type=entity_type,
                                  color=self.modern_palette[entity_type]['color'],
                                  icon=self.modern_palette[entity_type]['icon'])

            # 添加边
            self.graph.add_edge(
                triple.head,
                triple.tail,
                relation=triple.relation,
                confidence=triple.confidence,
                weight=triple.confidence
            )

        return self.graph

    def get_optimal_layout(self, graph: nx.Graph) -> Dict:
        """获取最优布局"""
        num_nodes = len(graph.nodes())
        
        if num_nodes <= 5:
            # 小图：使用圆形布局
            return nx.circular_layout(graph, scale=2)
        elif num_nodes <= 15:
            # 中图：使用spring布局（高迭代数）
            return nx.spring_layout(graph, k=3, iterations=100, seed=42)
        elif num_nodes <= 50:
            # 大图：使用Kamada-Kawai布局
            try:
                return nx.kamada_kawai_layout(graph, scale=2)
            except:
                return nx.spring_layout(graph, k=2, iterations=50, seed=42)
        else:
            # 超大图：使用层次布局
            try:
                return nx.multipartite_layout(graph, align='horizontal')
            except:
                return nx.spring_layout(graph, k=1.5, iterations=30, seed=42)

    def create_enhanced_plot(self, triples: List[Triple], entity_types: Dict[str, str] = None, 
                           title: str = "知识图谱可视化") -> go.Figure:
        """创建增强的交互式图谱"""
        # 创建图
        graph = self.create_enhanced_graph(triples, entity_types)

        if len(graph.nodes()) == 0:
            return self._create_empty_plot()

        # 获取最优布局
        pos = self.get_optimal_layout(graph)

        # 创建图形
        fig = go.Figure()

        # 添加背景网格（可选）
        self._add_background_grid(fig, pos)

        # 添加边
        self._add_enhanced_edges(fig, graph, pos)

        # 添加节点
        self._add_enhanced_nodes(fig, graph, pos)

        # 添加关系标签
        self._add_relation_labels(fig, graph, pos)

        # 设置增强布局
        self._set_enhanced_layout(fig, title)

        return fig

    def _create_empty_plot(self) -> go.Figure:
        """创建空图显示"""
        fig = go.Figure()
        fig.add_annotation(
            text="🕸️<br><br>暂无知识图谱数据<br><span style='font-size:14px; color:#666;'>请输入包含实体和关系的文本</span>",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=20, color='#999')
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=600
        )
        return fig

    def _add_background_grid(self, fig: go.Figure, pos: Dict):
        """添加背景网格（微妙的视觉增强）"""
        x_coords = [coord[0] for coord in pos.values()]
        y_coords = [coord[1] for coord in pos.values()]
        
        if x_coords and y_coords:
            x_range = [min(x_coords) - 0.5, max(x_coords) + 0.5]
            y_range = [min(y_coords) - 0.5, max(y_coords) + 0.5]
            
            # 添加微妙的网格线
            for i in range(int(x_range[0]), int(x_range[1]) + 1):
                fig.add_shape(
                    type="line",
                    x0=i, y0=y_range[0], x1=i, y1=y_range[1],
                    line=dict(color="rgba(200, 200, 200, 0.1)", width=1)
                )
            for i in range(int(y_range[0]), int(y_range[1]) + 1):
                fig.add_shape(
                    type="line",
                    x0=x_range[0], y0=i, x1=x_range[1], y1=i,
                    line=dict(color="rgba(200, 200, 200, 0.1)", width=1)
                )

    def _add_enhanced_edges(self, fig: go.Figure, graph: nx.Graph, pos: Dict):
        """添加增强的边"""
        relations = list(set([data['relation'] for _, _, data in graph.edges(data=True)]))
        
        for i, relation in enumerate(relations):
            edge_x = []
            edge_y = []
            confidences = []

            for edge in graph.edges(data=True):
                if edge[2]['relation'] == relation:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]

                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    confidences.append(edge[2]['confidence'])

            if edge_x:
                # 根据置信度调整边的样式
                avg_confidence = np.mean(confidences)
                line_width = 2 + (avg_confidence * 3)  # 2-5px
                opacity = 0.5 + (avg_confidence * 0.4)  # 0.5-0.9
                
                color = self.relation_palette[i % len(self.relation_palette)]
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(
                        width=line_width, 
                        color=color,
                        # 添加阴影效果
                        dash='solid'
                    ),
                    opacity=opacity,
                    hoverinfo='skip',
                    mode='lines',
                    name=f'关系: {relation}',
                    showlegend=True,
                    legendgroup='relations'
                )
                fig.add_trace(edge_trace)

    def _add_enhanced_nodes(self, fig: go.Figure, graph: nx.Graph, pos: Dict):
        """添加增强的节点"""
        # 按类型分组节点
        node_groups = {}
        for node in graph.nodes(data=True):
            node_type = node[1]['type']
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append(node)

        for node_type, nodes in node_groups.items():
            node_x = []
            node_y = []
            node_text = []
            node_sizes = []
            hover_text = []

            for node_name, node_data in nodes:
                x, y = pos[node_name]
                node_x.append(x)
                node_y.append(y)

                # 计算节点大小（基于连接数和重要性）
                degree = graph.degree(node_name)
                # 节点大小：20-60px，基于度数
                size = max(25, min(60, 25 + degree * 8))
                node_sizes.append(size)

                # 节点标签（智能截断）
                display_name = node_name
                if len(display_name) > 8:
                    display_name = display_name[:8] + '...'
                node_text.append(display_name)

                # 悬停信息
                connections = list(graph.neighbors(node_name))
                icon = node_data['icon']
                hover_info = f"{icon} <b>{node_name}</b><br>"
                hover_info += f"🔗 连接数: {degree}<br>"
                hover_info += f"📋 类型: {node_type}<br>"
                
                if connections:
                    hover_info += f"🔗 连接到:<br>"
                    for conn in connections[:5]:  # 最多显示5个连接
                        hover_info += f"  • {conn}<br>"
                    if len(connections) > 5:
                        hover_info += f"  ... 还有 {len(connections)-5} 个连接"

                hover_text.append(hover_info)

            # 添加节点轨迹
            type_config = self.modern_palette[node_type]
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=hover_text,
                text=node_text,
                textposition="middle center",
                textfont=dict(
                    size=10, 
                    color='white',
                    family='Arial Black'
                ),
                marker=dict(
                    size=node_sizes,
                    color=type_config['color'],
                    line=dict(width=3, color='white'),
                    opacity=0.9,
                    # 添加渐变效果（通过颜色变化模拟）
                    colorscale=[[0, type_config['color']], [1, type_config['color']]],
                ),
                name=f"{type_config['icon']} {node_type}",
                showlegend=True,
                legendgroup='entities'
            )
            fig.add_trace(node_trace)

    def _add_relation_labels(self, fig: go.Figure, graph: nx.Graph, pos: Dict):
        """添加关系标签"""
        edge_info = []
        
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            # 计算边的中点
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            
            # 添加一些偏移避免重叠
            offset_x = random.uniform(-0.1, 0.1)
            offset_y = random.uniform(-0.1, 0.1)
            
            edge_info.append({
                'x': mid_x + offset_x,
                'y': mid_y + offset_y,
                'text': edge[2]['relation'],
                'confidence': edge[2]['confidence']
            })

        if edge_info:
            # 根据置信度设置标签颜色
            colors = []
            for info in edge_info:
                if info['confidence'] > 0.8:
                    colors.append('darkgreen')
                elif info['confidence'] > 0.6:
                    colors.append('orange')
                else:
                    colors.append('red')

            relation_trace = go.Scatter(
                x=[info['x'] for info in edge_info],
                y=[info['y'] for info in edge_info],
                mode='text',
                text=[info['text'] for info in edge_info],
                textfont=dict(
                    size=9, 
                    color=colors,
                    family='Arial'
                ),
                hoverinfo='text',
                hovertext=[f"关系: {info['text']}<br>置信度: {info['confidence']:.3f}" for info in edge_info],
                showlegend=False,
                name='关系标签'
            )
            fig.add_trace(relation_trace)

    def _set_enhanced_layout(self, fig: go.Figure, title: str):
        """设置增强的布局"""
        layout_config = {
            'showlegend': True,
            'hovermode': 'closest',
            'hoverlabel': dict(
                bgcolor="white",
                bordercolor="gray",
                font_size=12,
                font_family="Arial"
            ),
            'xaxis': dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                showline=False
            ),
            'yaxis': dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                showline=False
            ),
            'plot_bgcolor': '#FAFAFA',
            'paper_bgcolor': 'white',
            'height': 700,
            'font': dict(family="Arial, sans-serif"),
            'legend': dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        }

        # 根据是否有标题调整边距
        if title and title.strip():
            layout_config.update({
                'title': dict(
                    text=f"<b>{title}</b>",
                    x=0.5,
                    font=dict(size=24, color='#2C3E50'),
                    pad=dict(t=20)
                ),
                'margin': dict(b=50, l=20, r=20, t=80),
                'annotations': [
                    dict(
                        text="💡 <i>拖拽节点移动 • 悬停查看详情 • 点击图例筛选</i>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.08,
                        xanchor='center', yanchor='top',
                        font=dict(size=11, color='#7F8C8D')
                    )
                ]
            })
        else:
            layout_config.update({
                'margin': dict(b=20, l=20, r=20, t=20)
            })

        fig.update_layout(**layout_config)

    def create_statistics_dashboard(self, triples: List[Triple], entity_types: Dict[str, str] = None) -> go.Figure:
        """创建增强的统计仪表板"""
        if not triples:
            fig = go.Figure()
            fig.add_annotation(text="暂无统计数据", x=0.5, y=0.5, showarrow=False)
            return fig

        # 统计数据准备
        relations = [triple.relation for triple in triples]
        confidences = [triple.confidence for triple in triples]
        
        # 实体类型统计
        all_entities = set()
        for triple in triples:
            all_entities.add(triple.head)
            all_entities.add(triple.tail)
        
        entity_type_counts = {}
        for entity in all_entities:
            entity_type = self.classify_entity_type(entity, entity_types)
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        # 关系分布
        relation_counts = {}
        for relation in relations:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        # 置信度分布
        confidence_ranges = ['很低(0-0.2)', '较低(0.2-0.4)', '中等(0.4-0.6)', '较高(0.6-0.8)', '很高(0.8-1.0)']
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
            rows=2, cols=3,
            subplot_titles=(
                '📊 实体类型分布', '🔗 关系类型分布', '📈 置信度分布',
                '📉 置信度趋势', '📋 统计摘要', '🎯 质量指标'
            ),
            specs=[
                [{"type": "pie"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}, {"type": "indicator"}]
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )

        # 1. 实体类型分布饼图
        type_colors = [self.modern_palette[t]['color'] for t in entity_type_counts.keys()]
        fig.add_trace(
            go.Pie(
                labels=[f"{self.modern_palette[t]['icon']} {t}" for t in entity_type_counts.keys()],
                values=list(entity_type_counts.values()),
                marker_colors=type_colors,
                hole=0.3,
                textinfo="label+percent",
                textfont_size=10
            ),
            row=1, col=1
        )

        # 2. 关系分布饼图
        fig.add_trace(
            go.Pie(
                labels=list(relation_counts.keys()),
                values=list(relation_counts.values()),
                marker_colors=self.relation_palette[:len(relation_counts)],
                hole=0.3,
                textinfo="label+value",
                textfont_size=10
            ),
            row=1, col=2
        )

        # 3. 置信度分布柱状图
        colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#27AE60']
        fig.add_trace(
            go.Bar(
                x=confidence_ranges,
                y=confidence_counts,
                marker_color=colors,
                text=confidence_counts,
                textposition='auto',
                name="置信度分布"
            ),
            row=1, col=3
        )

        # 4. 置信度趋势
        fig.add_trace(
            go.Scatter(
                x=list(range(len(confidences))),
                y=confidences,
                mode='markers+lines',
                name="置信度趋势",
                line=dict(color='#3498DB', width=2),
                marker=dict(
                    size=8, 
                    color=confidences, 
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="置信度")
                )
            ),
            row=2, col=1
        )

        # 5. 统计摘要表格
        avg_confidence = np.mean(confidences) if confidences else 0
        stats_data = [
            ['🔢 总实体数', len(all_entities)],
            ['🔗 总三元组数', len(triples)],
            ['📊 关系类型数', len(relation_counts)],
            ['⭐ 平均置信度', f'{avg_confidence:.3f}'],
            ['📈 最高置信度', f'{max(confidences):.3f}' if confidences else '0'],
            ['📉 最低置信度', f'{min(confidences):.3f}' if confidences else '0']
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['指标', '数值'],
                    fill_color='#3498DB',
                    font=dict(color='white', size=12),
                    align='center'
                ),
                cells=dict(
                    values=[[row[0] for row in stats_data], [row[1] for row in stats_data]],
                    fill_color=[['#ECF0F1'] * len(stats_data), ['white'] * len(stats_data)],
                    font=dict(size=11),
                    align=['left', 'center']
                )
            ),
            row=2, col=2
        )

        # 6. 质量指标仪表
        quality_score = avg_confidence * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "图谱质量分数"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2ECC71"},
                    'steps': [
                        {'range': [0, 50], 'color': "#F8D7DA"},
                        {'range': [50, 80], 'color': "#FFF3CD"},
                        {'range': [80, 100], 'color': "#D4EDDA"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=3
        )

        # 更新布局
        fig.update_layout(
            title=dict(
                text="<b>📊 知识图谱统计分析仪表板</b>",
                x=0.5,
                font=dict(size=20, color='#2C3E50')
            ),
            showlegend=False,
            height=900,
            plot_bgcolor='white',
            paper_bgcolor='#FAFAFA'
        )

        return fig

    def generate_premium_html(self, triples: List[Triple], entities: List[str],
                            relations: List[str], entity_types: Dict[str, str] = None,
                            title: str = "知识图谱分析报告") -> str:
        """生成高级版HTML报告"""
        
        # 创建图谱可视化
        graph_fig = self.create_enhanced_plot(triples, entity_types, "")
        
        # 创建统计仪表板
        stats_fig = self.create_statistics_dashboard(triples, entity_types)
        
        # 转换为HTML
        graph_html = graph_fig.to_html(include_plotlyjs='cdn', div_id="graph-plot")
        stats_html = stats_fig.to_html(include_plotlyjs='cdn', div_id="stats-plot")

        # 统计数据
        entity_type_stats = {}
        for entity in entities:
            entity_type = self.classify_entity_type(entity, entity_types)
            entity_type_stats[entity_type] = entity_type_stats.get(entity_type, 0) + 1

        avg_confidence = sum(t.confidence for t in triples) / len(triples) if triples else 0

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            font-size: 3em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.2em;
        }}
        
        .content {{
            display: grid;
            gap: 30px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2C3E50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 1em;
            opacity: 0.9;
        }}
        
        .plot-container {{
            margin: 20px 0;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .entity-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .entity-type-card {{
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 30px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🕸️ {title}</h1>
            <p>基于AI的智能知识图谱构建与分析平台</p>
        </div>

        <div class="content">
            <div class="card">
                <h2 class="section-title">📊 数据概览</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{len(entities)}</div>
                        <div class="stat-label">实体总数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(relations)}</div>
                        <div class="stat-label">关系类型</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(triples)}</div>
                        <div class="stat-label">知识三元组</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{avg_confidence:.2f}</div>
                        <div class="stat-label">平均置信度</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2 class="section-title">🕸️ 知识图谱网络</h2>
                <div class="plot-container">
                    {graph_html.split('<body>')[1].split('</body>')[0] if '<body>' in graph_html else graph_html}
                </div>
            </div>

            <div class="card">
                <h2 class="section-title">📈 统计分析仪表板</h2>
                <div class="plot-container">
                    {stats_html.split('<body>')[1].split('</body>')[0] if '<body>' in stats_html else stats_html}
                </div>
            </div>

            <div class="card">
                <h2 class="section-title">🏷️ 实体类型分布</h2>
                <div class="entity-grid">
                    {''.join([
                        f'<div class="entity-type-card" style="background: {self.modern_palette[entity_type]["color"]};">'
                        f'{self.modern_palette[entity_type]["icon"]} {entity_type}<br>'
                        f'<small>{count} 个实体</small></div>'
                        for entity_type, count in entity_type_stats.items()
                    ])}
                </div>
            </div>
        </div>

        <div class="footer">
            <p>🚀 Powered by Enhanced Knowledge Graph Visualizer | Built with ❤️ and AI</p>
        </div>
    </div>
</body>
</html>
        """

        return html_content

    def save_enhanced_visualization(self, triples: List[Triple], entities: List[str],
                                  relations: List[str], entity_types: Dict[str, str] = None,
                                  filename: str = "enhanced_knowledge_graph.html"):
        """保存增强的可视化结果"""
        html_content = self.generate_premium_html(triples, entities, relations, entity_types)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename 