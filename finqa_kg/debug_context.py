"""Check context of specific entities for Sample 5"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder

# Load Sample 5
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[4]

print("Sample:", sample['id'])
print("Question:", sample['qa']['question'])
print()

# Check table structure
print("Table:")
for i, row in enumerate(sample['table']):
    print(f"  Row {i}: {row}")
print()

# Build KG
builder = IntelligentKGBuilder()
graph = builder.build_kg(sample)
entity_index = builder.get_entity_index(graph)

print("="*60)
print("Entities with values 991.1 or 959.2:")
for value in [991.1, 959.2]:
    if value in entity_index['by_value']:
        for item in entity_index['by_value'][value]:
            data = item['data']
            print(f"\n  Value: {value}")
            print(f"    Node: {item['id']}")
            print(f"    Text: '{data.get('text', '')}'")
            print(f"    Context: '{data.get('context', '')}'")
            print(f"    Label: {data.get('label', '')}")

print("\n" + "="*60)
print("Entities with text 'revenue' (checking first 5):")
if 'revenue' in entity_index['by_text']:
    for item in entity_index['by_text']['revenue'][:5]:
        data = item['data']
        print(f"\n  Node: {item['id']}")
        print(f"    Text: '{data.get('text', '')}'")
        print(f"    Value: {data.get('value', '')}")
        print(f"    Context: '{data.get('context', '')[:100]}...'")

print("\n" + "="*60)
print("Search test: 'revenue' in context of entities with numeric values:")
count = 0
for value, items in entity_index['by_value'].items():
    try:
        val = float(value)
        for item in items:
            context = item['data'].get('context', '').lower()
            if 'revenue' in context and 'table[' in context:
                print(f"\n  Value: {val}")
                print(f"    Context: '{context}'")
                count += 1
                if count >= 10:
                    break
        if count >= 10:
            break
    except:
        pass
