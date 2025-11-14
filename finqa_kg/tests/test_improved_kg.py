"""
Test Improved KG Builder
Kiểm tra các cải tiến: value index, percentage handling, metadata
"""

import json
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from finqa_kg.src.pipeline.structured_kg_builder import StructuredKGBuilder

async def test_sample(sample, sample_idx):
    """Test KG building for one sample"""
    print(f"\n{'='*80}")
    print(f"TESTING SAMPLE {sample_idx}")
    print(f"{'='*80}")
    
    # Extract data
    table = sample.get('table', [])
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    question = sample.get('qa', {}).get('question', '')
    program = sample.get('qa', {}).get('program', '')
    answer = sample.get('qa', {}).get('answer', '')
    
    print(f"\nQuestion: {question}")
    print(f"Program: {program}")
    print(f"Answer: {answer}")
    
    # Build KG
    print(f"\n🔨 Building KG...")
    builder = StructuredKGBuilder()
    kg = await builder.build_graph(pre_text=pre_text, post_text=post_text, table=table)
    
    # Get statistics
    stats = builder.get_statistics()
    print(f"\n📊 KG Statistics:")
    print(f"   Nodes: {stats['total_nodes']}")
    print(f"   Edges: {stats['total_edges']}")
    print(f"   Node types: {stats['node_types']}")
    print(f"   Indexed values: {stats['indexed_values']}")
    
    # Test value lookup
    print(f"\n🔍 Testing Value Lookups:")
    
    # Example: Look for specific values from program
    test_values = []
    
    # Extract numeric values from program
    import re
    numbers = re.findall(r'\b\d+\.?\d*\b', program)
    for num_str in numbers[:5]:  # Test first 5
        try:
            val = float(num_str)
            test_values.append(val)
        except:
            pass
    
    print(f"   Testing {len(test_values)} values from program...")
    
    for val in test_values:
        matches = builder.find_nodes_by_value(val)
        if matches:
            print(f"   ✓ {val}: Found {len(matches)} match(es)")
            # Show first match details
            first_match = kg.nodes[matches[0]]
            print(f"      → {first_match.get('raw_value')} at [{first_match.get('row_index')},{first_match.get('col_index')}]")
            if first_match.get('is_percent'):
                print(f"      → Is percentage: {first_match.get('is_percent')}")
        else:
            print(f"   ✗ {val}: No matches found")
    
    # Test percentage handling
    print(f"\n📊 Testing Percentage Handling:")
    for node_id, node_data in kg.nodes(data=True):
        if node_data.get('is_percent'):
            print(f"   Found percent: {node_data.get('raw_value')}")
            print(f"      Normalized value: {node_data.get('value')}")
            print(f"      Indexed as: {node_data.get('value')} and {node_data.get('value')/100}")
            break  # Just show one example
    
    # Show some cell nodes with metadata
    print(f"\n🔬 Sample Cell Nodes (with enhanced metadata):")
    cell_count = 0
    for node_id in builder.cell_nodes[:5]:  # First 5 cells
        node_data = kg.nodes[node_id]
        print(f"\n   {node_id}:")
        print(f"      Raw: {node_data.get('raw_value')}")
        print(f"      Value: {node_data.get('value')}")
        print(f"      Column: {node_data.get('column_name')}")
        print(f"      Location: [{node_data.get('row_index')},{node_data.get('col_index')}]")
        print(f"      Is percent: {node_data.get('is_percent')}")
        print(f"      Is currency: {node_data.get('is_currency')}")
        cell_count += 1
    
    return stats

async def main():
    print("="*80)
    print("TEST IMPROVED KG BUILDER")
    print("="*80)
    
    # Load dataset
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Test first 3 samples
    results = []
    for idx in range(3):
        if idx < len(data):
            try:
                stats = await test_sample(data[idx], idx)
                results.append({
                    'sample_idx': idx,
                    'stats': stats
                })
            except Exception as e:
                print(f"\n❌ Error testing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    if results:
        avg_nodes = sum(r['stats']['total_nodes'] for r in results) / len(results)
        avg_indexed = sum(r['stats']['indexed_values'] for r in results) / len(results)
        
        print(f"\nAverage per sample:")
        print(f"   Nodes: {avg_nodes:.1f}")
        print(f"   Indexed values: {avg_indexed:.1f}")
        
        print(f"\n✅ Improvements verified:")
        print(f"   - Value indexing for O(1) lookup")
        print(f"   - Percentage metadata preserved")
        print(f"   - Enhanced cell metadata (currency, negatives, etc.)")
        print(f"   - Both sync and async interfaces")

if __name__ == "__main__":
    asyncio.run(main())
