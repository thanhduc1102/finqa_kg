"""Debug table cells for sample 5"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder

# Load sample 5
with open('../FinQA/dataset/train.json') as f:
    data = json.load(f)
sample = data[4]

print("Sample ID:", sample['id'])
print("Question:", sample['qa']['question'])
print("\nBuilding KG...")

builder = IntelligentKGBuilder()
kg = builder.build_kg(sample)

print(f"\nTotal nodes: {kg.number_of_nodes()}")

# Find all cell nodes
print("\nTable cells with values:")
cell_count = 0
for node_id, node_data in kg.nodes(data=True):
    if node_data.get('type') == 'cell' and node_data.get('value'):
        cell_count += 1
        if cell_count <= 10:
            print(f"  Cell: value={node_data['value']}, "
                  f"text={node_data.get('text', '')[:40]}, "
                  f"context={node_data.get('context', '')[:100]}")

print(f"\nTotal cells with values: {cell_count}")

# Check if 991.1 exists
print("\n" + "="*60)
print("Looking for value 991.1 (2007 net revenue):")
found = False
for node_id, node_data in kg.nodes(data=True):
    if node_data.get('value') == 991.1:
        found = True
        print(f"  Found! Node: {node_id}")
        print(f"  Type: {node_data.get('type')}")
        print(f"  Text: {node_data.get('text')}")
        print(f"  Context: {node_data.get('context', '')[:200]}")

if not found:
    print("  NOT FOUND!")

# Check for any values near 991
print("\nValues between 900-1000:")
for node_id, node_data in kg.nodes(data=True):
    val = node_data.get('value')
    if val and 900 <= val <= 1000:
        print(f"  value={val}, type={node_data.get('type')}, "
              f"text={node_data.get('text', '')[:50]}")
