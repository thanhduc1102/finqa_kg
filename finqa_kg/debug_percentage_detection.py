"""Debug percentage detection for Sample 3"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder
from src.pipeline.question_analyzer import QuestionAnalyzer
import networkx as nx

# Load Sample 3
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[2]

# Build KG
builder = IntelligentKGBuilder()
kg = builder.build_kg(sample)

print("="*80)
print("CHECKING PERCENTAGE DETECTION")
print("="*80)

# Find the 9896 node
print("\nLooking for 9896 node...")
node_9896 = None
for node_id, node_data in kg.nodes(data=True):
    if node_data.get('value') == 9896.0:
        node_9896 = (node_id, node_data)
        print(f"\nFound 9896 at node {node_id}:")
        print(f"  Type: {node_data.get('type')}")
        print(f"  Label: {node_data.get('label')}")
        print(f"  Value: {node_data.get('value')}")
        print(f"  Context: {node_data.get('context')}")
        break

# Find PERCENT nodes
print("\n" + "="*80)
print("ALL PERCENT NODES")
print("="*80)

for node_id, node_data in kg.nodes(data=True):
    if node_data.get('label') == 'PERCENT':
        print(f"\nNode {node_id}:")
        print(f"  Value: {node_data.get('value')}")
        print(f"  Context: {node_data.get('context')}")
        
        # Check if contexts overlap
        if node_9896:
            context1 = node_9896[1].get('context', '').lower()
            context2 = node_data.get('context', '').lower()
            
            # Check if value appears in context
            if '9896' in context2:
                print(f"  ✓ Contains 9896 in context!")
            
            # Check word overlap
            words1 = set(context1.split())
            words2 = set(context2.split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            overlap = len(intersection) / len(union) if union else 0
            
            print(f"  Context overlap: {overlap:.2%}")
            if overlap >= 0.3:
                print(f"  ✓ Significant overlap!")
