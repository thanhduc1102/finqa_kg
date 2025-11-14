"""
HƯỚNG DẪN: Tạo Knowledge Graph và Visualization cho 1 Sample Bất Kỳ

Sử dụng:
    python single_sample_visualizer.py [sample_index]
    
    Ví dụ:
    python single_sample_visualizer.py 0      # Sample đầu tiên
    python single_sample_visualizer.py 42     # Sample thứ 42
"""

import json
import sys
import asyncio
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from finqa_kg.src.pipeline.structured_kg_builder import StructuredKGBuilder


def print_section(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_source_data(sample):
    """In ra dữ liệu nguồn để dễ đối chiếu"""
    print_section("📥 DỮ LIỆU NGUỒN")
    
    # Question & Answer
    qa = sample.get('qa', {})
    question = qa.get('question', '')
    answer = qa.get('answer', '')
    program = qa.get('program', '')
    
    print(f"❓ Question: {question}")
    print(f"✅ Answer: {answer}")
    print(f"🔧 Program: {program}")
    
    # Table
    table = sample.get('table', [])
    if table:
        print(f"\n📊 TABLE ({len(table)} rows × {len(table[0]) if table else 0} columns):")
        print(f"{'─'*80}")
        
        # Print header
        if table:
            header = table[0]
            print(f"│ {'Row':<5} │ ", end="")
            for i, col in enumerate(header):
                print(f"Col {i}: {str(col):<20} │ ", end="")
            print()
            print(f"{'─'*80}")
        
        # Print data rows
        for i, row in enumerate(table):
            print(f"│ {i:<5} │ ", end="")
            for cell in row:
                print(f"{str(cell):<25} │ ", end="")
            print()
        print(f"{'─'*80}")
    
    # Text
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    
    if pre_text:
        print(f"\n📝 PRE-TEXT ({len(pre_text)} sentences):")
        for i, text in enumerate(pre_text[:3]):  # First 3
            print(f"   {i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
        if len(pre_text) > 3:
            print(f"   ... and {len(pre_text)-3} more sentences")
    
    if post_text:
        print(f"\n📝 POST-TEXT ({len(post_text)} sentences):")
        for i, text in enumerate(post_text[:3]):  # First 3
            print(f"   {i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
        if len(post_text) > 3:
            print(f"   ... and {len(post_text)-3} more sentences")


async def build_and_analyze_kg(sample):
    """Xây dựng KG và phân tích"""
    print_section("🔨 XÂY DỰNG KNOWLEDGE GRAPH")
    
    # Extract data
    table = sample.get('table', [])
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    
    # Build KG
    print("⏳ Building graph...")
    builder = StructuredKGBuilder()
    kg = await builder.build_graph(
        pre_text=pre_text,
        post_text=post_text,
        table=table
    )
    
    # Statistics
    stats = builder.get_statistics()
    print(f"\n✓ Graph built successfully!")
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Value index: {stats['indexed_values']} unique values")
    
    # Node type breakdown
    print(f"\n📊 Node Types:")
    for ntype, count in stats['node_types'].items():
        print(f"   - {ntype}: {count}")
    
    # Show sample cell nodes
    print(f"\n🔬 SAMPLE CELL NODES (with metadata):")
    print(f"{'─'*80}")
    
    cell_count = 0
    for node_id in builder.cell_nodes[:5]:  # First 5 cells
        node_data = kg.nodes[node_id]
        print(f"\n   Node ID: {node_id}")
        print(f"   ├─ Location: [{node_data.get('row_index')}, {node_data.get('col_index')}]")
        print(f"   ├─ Column: {node_data.get('column_name')}")
        print(f"   ├─ Raw value: {node_data.get('raw_value')}")
        print(f"   ├─ Parsed value: {node_data.get('value')}")
        print(f"   ├─ Label: {node_data.get('label')}")
        print(f"   └─ Format: percent={node_data.get('is_percent')}, currency={node_data.get('is_currency')}")
        cell_count += 1
    
    # Value index examples
    print(f"\n🗂️ VALUE INDEX (examples):")
    print(f"{'─'*80}")
    
    shown = 0
    for value, node_ids in list(builder.value_index.items())[:10]:  # First 10
        print(f"   {value} → {node_ids}")
        shown += 1
    
    if len(builder.value_index) > 10:
        print(f"   ... and {len(builder.value_index)-10} more values")
    
    return builder, kg


def create_html_visualization(builder, kg, sample, sample_idx, output_path):
    """Tạo HTML visualization với đối chiếu dữ liệu"""
    
    print_section("🎨 TẠO VISUALIZATION")
    
    # Prepare nodes for vis.js
    nodes = []
    edges = []
    
    # Color scheme
    colors = {
        'table': '#4A90E2',
        'row': '#7B68EE',
        'cell': '#50C878',
        'text': '#FFB347',
        'number': '#FF6B6B'
    }
    
    # Add nodes
    for node_id, node_data in kg.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        
        # Create label
        if node_type == 'cell':
            value = node_data.get('raw_value', '')
            row = node_data.get('row_index', '?')
            col = node_data.get('col_index', '?')
            label = f"[{row},{col}]\\n{value}"
        elif node_type == 'row':
            label = f"Row {node_data.get('row_index', '?')}"
        elif node_type == 'table':
            label = "TABLE"
        elif node_type == 'text':
            content = node_data.get('content', '')[:20]
            label = f"{content}..."
        else:
            label = node_id
        
        # Tooltip with full metadata
        tooltip_lines = []
        for key, val in node_data.items():
            if key not in ['type']:
                tooltip_lines.append(f"{key}: {val}")
        tooltip = "\\n".join(tooltip_lines)
        
        nodes.append({
            'id': node_id,
            'label': label,
            'title': tooltip,
            'color': colors.get(node_type, '#CCCCCC'),
            'shape': 'box' if node_type == 'cell' else 'ellipse'
        })
    
    # Add edges
    for u, v, edge_data in kg.edges(data=True):
        relation = edge_data.get('relation', 'RELATED')
        edges.append({
            'from': u,
            'to': v,
            'label': relation,
            'arrows': 'to'
        })
    
    # Prepare table HTML
    table = sample.get('table', [])
    table_html = '<table class="source-table">'
    for i, row in enumerate(table):
        row_class = 'header-row' if i == 0 else 'data-row'
        table_html += f'<tr class="{row_class}">'
        for j, cell in enumerate(row):
            cell_class = 'header-cell' if i == 0 else 'data-cell'
            table_html += f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell}</td>'
        table_html += '</tr>'
    table_html += '</table>'
    
    # Prepare text HTML
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    
    text_html = '<div class="text-section">'
    if pre_text:
        text_html += '<h4>Pre-text:</h4>'
        for text in pre_text:
            text_html += f'<p class="text-sentence">{text}</p>'
    if post_text:
        text_html += '<h4>Post-text:</h4>'
        for text in post_text:
            text_html += f'<p class="text-sentence">{text}</p>'
    text_html += '</div>'
    
    # Question/Answer
    qa = sample.get('qa', {})
    
    # HTML template
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Sample {sample_idx} - KG Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
        }}
        
        .container {{
            display: grid;
            grid-template-columns: 400px 1fr;
            grid-template-rows: auto 1fr;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }}
        
        .header {{
            grid-column: 1 / -1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 24px;
        }}
        
        .qa-info {{
            background: #f0f7ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        
        .qa-info strong {{
            color: #1976d2;
        }}
        
        .sidebar {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-y: auto;
            padding: 20px;
        }}
        
        .sidebar h3 {{
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .source-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 12px;
        }}
        
        .source-table td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        
        .header-cell {{
            background: #4A90E2;
            color: white;
            font-weight: bold;
        }}
        
        .data-cell {{
            background: #f9f9f9;
        }}
        
        .data-cell:hover {{
            background: #fff9c4;
            cursor: pointer;
        }}
        
        .text-section {{
            margin-top: 20px;
        }}
        
        .text-section h4 {{
            color: #666;
            margin: 10px 0;
        }}
        
        .text-sentence {{
            background: #f9f9f9;
            padding: 10px;
            margin: 5px 0;
            border-left: 3px solid #FFB347;
            font-size: 12px;
            line-height: 1.4;
        }}
        
        .graph-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
        }}
        
        #network {{
            width: 100%;
            height: 100%;
        }}
        
        .legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 100;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 12px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
        }}
        
        .stats {{
            background: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 12px;
        }}
        
        .stats strong {{
            color: #2e7d32;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Sample {sample_idx} - Knowledge Graph Visualization</h1>
            
            <div class="qa-info">
                <p><strong>Question:</strong> {qa.get('question', '')}</p>
                <p><strong>Answer:</strong> {qa.get('answer', '')}</p>
                <p><strong>Program:</strong> <code>{qa.get('program', '')}</code></p>
            </div>
            
            <div class="stats">
                <strong>Graph Statistics:</strong>
                Nodes: {builder.get_statistics()['total_nodes']} |
                Edges: {builder.get_statistics()['total_edges']} |
                Indexed Values: {builder.get_statistics()['indexed_values']}
            </div>
        </div>
        
        <div class="sidebar">
            <h3>📊 Source Table</h3>
            {table_html}
            
            <h3>📝 Source Text</h3>
            {text_html}
        </div>
        
        <div class="graph-container">
            <div id="network"></div>
            
            <div class="legend">
                <strong>Node Types:</strong>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4A90E2"></div>
                    <span>Table</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #7B68EE"></div>
                    <span>Row</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #50C878"></div>
                    <span>Cell</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FFB347"></div>
                    <span>Text</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        
        var container = document.getElementById('network');
        var data = {{nodes: nodes, edges: edges}};
        
        var options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150,
                    nodeSpacing: 100
                }}
            }},
            physics: {{
                enabled: false
            }},
            nodes: {{
                font: {{size: 12, face: 'monospace'}},
                borderWidth: 2
            }},
            edges: {{
                smooth: {{type: 'cubicBezier', forceDirection: 'vertical'}},
                font: {{size: 10}}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Click on cell in table to highlight in graph
        document.querySelectorAll('.data-cell').forEach(cell => {{
            cell.addEventListener('click', function() {{
                var row = this.dataset.row;
                var col = this.dataset.col;
                
                // Find corresponding node
                var cellNodes = nodes.get({{
                    filter: function(item) {{
                        return item.label.includes('[' + row + ',' + col + ']');
                    }}
                }});
                
                if (cellNodes.length > 0) {{
                    network.selectNodes([cellNodes[0].id]);
                    network.focus(cellNodes[0].id, {{
                        scale: 1.5,
                        animation: true
                    }});
                }}
            }});
        }});
        
        // Click on node to log details
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                console.log('Node details:', node);
            }}
        }});
    </script>
</body>
</html>"""
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Visualization saved!")
    print(f"   File: {output_path}")
    print(f"   Open in browser to view interactive graph")


async def main():
    # Get sample index from command line
    if len(sys.argv) > 1:
        sample_idx = int(sys.argv[1])
    else:
        sample_idx = 0  # Default to first sample
    
    print_section(f"🎯 SINGLE SAMPLE KNOWLEDGE GRAPH VISUALIZER")
    print(f"Processing sample index: {sample_idx}")
    
    # Load dataset
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if sample_idx >= len(data):
        print(f"❌ Error: Sample index {sample_idx} out of range (max: {len(data)-1})")
        return
    
    sample = data[sample_idx]
    
    # Step 1: Show source data
    print_source_data(sample)
    
    # Step 2: Build and analyze KG
    builder, kg = await build_and_analyze_kg(sample)
    
    # Step 3: Create visualization
    output_dir = Path(__file__).parent / "output" / "single_sample_viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"sample_{sample_idx}_visualization.html"
    create_html_visualization(builder, kg, sample, sample_idx, output_path)
    
    print_section("✅ HOÀN THÀNH")
    print(f"Bạn có thể:")
    print(f"  1. Mở file HTML trong browser: {output_path}")
    print(f"  2. Click vào cells trong table (bên trái) để highlight trong graph (bên phải)")
    print(f"  3. Hover vào nodes để xem metadata đầy đủ")
    print(f"  4. Zoom/pan graph để explore")
    print(f"\nĐể visualize sample khác, chạy:")
    print(f"  python {Path(__file__).name} [sample_index]")


if __name__ == "__main__":
    asyncio.run(main())
