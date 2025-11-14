"""
Visualize Knowledge Graph Structure
Trực quan hóa cấu trúc đồ thị tri thức cho một số mẫu điển hình
"""

import json
import sys
import asyncio
from pathlib import Path
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from finqa_kg.src.pipeline.structured_kg_builder import StructuredKGBuilder

async def visualize_sample(sample, sample_idx, output_dir):
    """Visualize KG for one sample"""
    print(f"\n{'='*80}")
    print(f"Visualizing Sample {sample_idx}")
    print(f"{'='*80}")
    
    # Extract data
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    table = sample.get('table', [])
    question = sample.get('qa', {}).get('question', '')
    
    print(f"Question: {question}")
    
    # Build KG
    builder = StructuredKGBuilder()
    kg = await builder.build_graph(
        pre_text=pre_text,
        post_text=post_text,
        table=table
    )
    
    # Create detailed text visualization
    viz_text = []
    viz_text.append(f"SAMPLE {sample_idx} - KG STRUCTURE")
    viz_text.append("="*80)
    viz_text.append(f"\nQuestion: {question}")
    viz_text.append(f"Answer: {sample.get('qa', {}).get('answer', '')}")
    viz_text.append(f"Program: {sample.get('qa', {}).get('program', '')}")
    
    viz_text.append(f"\n{'='*80}")
    viz_text.append(f"KG STATISTICS")
    viz_text.append(f"{'='*80}")
    viz_text.append(f"Total Nodes: {kg.number_of_nodes()}")
    viz_text.append(f"Total Edges: {kg.number_of_edges()}")
    
    # Count by type
    from collections import defaultdict
    node_types = defaultdict(list)
    for node_id, node_data in kg.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        node_types[node_type].append((node_id, node_data))
    
    viz_text.append(f"\nNode Type Distribution:")
    for ntype, nodes in sorted(node_types.items()):
        viz_text.append(f"  {ntype}: {len(nodes)}")
    
    # Detailed nodes by type
    viz_text.append(f"\n{'='*80}")
    viz_text.append(f"DETAILED NODES BY TYPE")
    viz_text.append(f"{'='*80}")
    
    # Numbers
    if 'number' in node_types:
        viz_text.append(f"\n📊 NUMBER ENTITIES ({len(node_types['number'])}):")
        for node_id, node_data in sorted(node_types['number'][:20]):  # Top 20
            value = node_data.get('value', 'N/A')
            text = node_data.get('text', 'N/A')
            context = node_data.get('context', '')[:60]
            viz_text.append(f"  {node_id}: value={value}, text='{text}'")
            if context:
                viz_text.append(f"    context: {context}...")
    
    # Table structure
    if 'table' in node_types:
        viz_text.append(f"\n📋 TABLE ENTITIES ({len(node_types['table'])}):")
        for node_id, node_data in node_types['table']:
            rows = node_data.get('rows', 0)
            cols = node_data.get('cols', 0)
            viz_text.append(f"  {node_id}: {rows} rows × {cols} cols")
    
    if 'row' in node_types:
        viz_text.append(f"\n📏 ROW ENTITIES ({len(node_types['row'])}):")
        for node_id, node_data in sorted(node_types['row'][:10]):
            row_idx = node_data.get('row_index', -1)
            viz_text.append(f"  {node_id}: row_index={row_idx}")
    
    if 'cell' in node_types:
        viz_text.append(f"\n📦 CELL ENTITIES ({len(node_types['cell'])}):")
        for node_id, node_data in sorted(node_types['cell'][:20]):
            row = node_data.get('row', -1)
            col = node_data.get('col', -1)
            value = str(node_data.get('value', ''))[:30]
            viz_text.append(f"  {node_id}: [{row},{col}] = '{value}'")
    
    # Temporal entities
    if 'temporal' in node_types:
        viz_text.append(f"\n⏰ TEMPORAL ENTITIES ({len(node_types['temporal'])}):")
        for node_id, node_data in node_types['temporal']:
            value = node_data.get('value', 'N/A')
            text = node_data.get('text', 'N/A')
            viz_text.append(f"  {node_id}: {text} ({value})")
    
    # Relationships
    viz_text.append(f"\n{'='*80}")
    viz_text.append(f"RELATIONSHIPS")
    viz_text.append(f"{'='*80}")
    
    edge_types = defaultdict(int)
    for u, v, edge_data in kg.edges(data=True):
        rel_type = edge_data.get('relation', 'unknown')
        edge_types[rel_type] += 1
    
    viz_text.append(f"\nRelationship Type Distribution:")
    for rel_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        viz_text.append(f"  {rel_type}: {count}")
    
    # Sample relationships
    viz_text.append(f"\nSample Relationships (first 20):")
    for i, (u, v, edge_data) in enumerate(kg.edges(data=True)):
        if i >= 20:
            break
        rel_type = edge_data.get('relation', 'unknown')
        u_type = kg.nodes[u].get('type', '?')
        v_type = kg.nodes[v].get('type', '?')
        viz_text.append(f"  {u}({u_type}) --[{rel_type}]--> {v}({v_type})")
    
    # Save visualization
    output_file = output_dir / f"kg_visualization_sample_{sample_idx}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(viz_text))
    
    print(f"✓ Visualization saved to: {output_file}")
    
    return '\n'.join(viz_text)

async def main():
    print("="*80)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("Trực quan hóa cấu trúc đồ thị tri thức")
    print("="*80)
    
    # Load dataset
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Select interesting samples to visualize
    sample_indices = [0, 1, 5, 10, 20]  # Various samples
    
    output_dir = Path(__file__).parent / "output" / "kg_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Visualizing {len(sample_indices)} samples...")
    
    for idx in sample_indices:
        if idx < len(data):
            try:
                await visualize_sample(data[idx], idx, output_dir)
            except Exception as e:
                print(f"❌ Error visualizing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
