"""
Interactive KG Visualization
Tạo visualization HTML interactive để xem cấu trúc KG chi tiết
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

def create_html_visualization(kg, sample_data, output_path):
    """Create interactive HTML visualization using vis.js"""
    
    # Prepare nodes and edges for vis.js
    nodes = []
    edges = []
    
    # Color scheme by node type
    colors = {
        'table': '#4A90E2',      # Blue
        'row': '#7B68EE',        # Purple  
        'cell': '#50C878',       # Green
        'text': '#FFB347',       # Orange
        'number': '#FF6B6B',     # Red
        'temporal': '#95E1D3'    # Mint
    }
    
    # Add nodes
    for node_id, node_data in kg.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        
        # Create label
        if node_type == 'cell':
            value = node_data.get('raw_value', '')
            row = node_data.get('row_index', '?')
            col = node_data.get('col_index', '?')
            label = f"[{row},{col}]\n{value}"
        elif node_type == 'row':
            label = f"Row {node_data.get('row_index', '?')}"
        elif node_type == 'table':
            label = "TABLE"
        elif node_type == 'text':
            content = node_data.get('content', '')[:30]
            label = f"TEXT\n{content}..."
        else:
            label = node_data.get('label', node_id)
        
        # Add node
        nodes.append({
            'id': node_id,
            'label': label,
            'title': json.dumps(node_data, indent=2, ensure_ascii=False),  # Tooltip
            'color': colors.get(node_type, '#CCCCCC'),
            'shape': 'box' if node_type == 'cell' else 'ellipse',
            'font': {'size': 10}
        })
    
    # Add edges
    for u, v, edge_data in kg.edges(data=True):
        relation = edge_data.get('relation', 'RELATED')
        edges.append({
            'from': u,
            'to': v,
            'label': relation,
            'arrows': 'to',
            'font': {'size': 8},
            'color': {'color': '#848484'}
        })
    
    # Generate HTML
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        #info {
            position: fixed;
            top: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-width: 400px;
            z-index: 1000;
        }
        #network {
            width: 100%;
            height: 100vh;
            border: 1px solid lightgray;
        }
        .stat {
            margin: 5px 0;
            font-size: 14px;
        }
        .question {
            background: #f0f0f0;
            padding: 10px;
            margin: 10px 0;
            border-radius: 3px;
            font-size: 13px;
        }
        .legend {
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #ccc;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 12px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3 style="margin-top:0">Sample {SAMPLE_IDX}</h3>
        <div class="question">
            <strong>Question:</strong><br>
            {QUESTION}
        </div>
        <div class="stat"><strong>Answer:</strong> {ANSWER}</div>
        <div class="stat"><strong>Program:</strong> {PROGRAM}</div>
        <hr>
        <div class="stat"><strong>Nodes:</strong> {NUM_NODES}</div>
        <div class="stat"><strong>Edges:</strong> {NUM_EDGES}</div>
        
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
            <div class="legend-item">
                <div class="legend-color" style="background: #FF6B6B"></div>
                <span>Number</span>
            </div>
        </div>
    </div>
    
    <div id="network"></div>
    
    <script type="text/javascript">
        var nodes = new vis.DataSet({NODES_JSON});
        var edges = new vis.DataSet({EDGES_JSON});
        
        var container = document.getElementById('network');
        var data = {
            nodes: nodes,
            edges: edges
        };
        var options = {
            layout: {
                hierarchical: {
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150,
                    nodeSpacing: 200
                }
            },
            physics: {
                enabled: false
            },
            nodes: {
                font: {
                    size: 12,
                    face: 'monospace'
                },
                borderWidth: 2
            },
            edges: {
                smooth: {
                    type: 'cubicBezier',
                    forceDirection: 'vertical'
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }
        };
        
        var network = new vis.Network(container, data, options);
        
        // Click handler
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                console.log('Clicked node:', node);
            }
        });
    </script>
</body>
</html>
    """
    
    # Fill template
    html = html_template.replace('{SAMPLE_IDX}', str(sample_data.get('idx', '?')))
    html = html.replace('{QUESTION}', sample_data.get('question', ''))
    html = html.replace('{ANSWER}', str(sample_data.get('answer', '')))
    html = html.replace('{PROGRAM}', sample_data.get('program', ''))
    html = html.replace('{NUM_NODES}', str(kg.number_of_nodes()))
    html = html.replace('{NUM_EDGES}', str(kg.number_of_edges()))
    html = html.replace('{NODES_JSON}', json.dumps(nodes, ensure_ascii=False))
    html = html.replace('{EDGES_JSON}', json.dumps(edges, ensure_ascii=False))
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Visualization saved to: {output_path}")

async def visualize_sample(sample, sample_idx, output_dir):
    """Build KG and create visualization for one sample"""
    print(f"\n{'='*80}")
    print(f"Visualizing Sample {sample_idx}")
    print(f"{'='*80}")
    
    # Extract data
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    table = sample.get('table', [])
    question = sample.get('qa', {}).get('question', '')
    answer = sample.get('qa', {}).get('answer', '')
    program = sample.get('qa', {}).get('program', '')
    
    print(f"Question: {question[:80]}...")
    
    # Build KG
    builder = StructuredKGBuilder()
    kg = await builder.build_graph(
        pre_text=pre_text,
        post_text=post_text,
        table=table
    )
    
    print(f"KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    
    # Create visualization
    sample_data = {
        'idx': sample_idx,
        'question': question,
        'answer': answer,
        'program': program
    }
    
    output_path = output_dir / f"kg_sample_{sample_idx}.html"
    create_html_visualization(kg, sample_data, output_path)
    
    return output_path

async def main():
    print("="*80)
    print("INTERACTIVE KNOWLEDGE GRAPH VISUALIZATION")
    print("="*80)
    
    # Load dataset
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "kg_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize interesting samples
    sample_indices = [0, 1, 2, 3, 4]  # First 5 samples
    
    print(f"\n📊 Creating visualizations for {len(sample_indices)} samples...")
    
    for idx in sample_indices:
        if idx < len(data):
            try:
                output_path = await visualize_sample(data[idx], idx, output_dir)
                print(f"   → Open in browser: {output_path}")
            except Exception as e:
                print(f"❌ Error visualizing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOpen the HTML files in your browser to explore the knowledge graphs interactively!")
    print(f"Location: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
