"""
Debug entity indexing for Sample 5
"""
"""
Debug entity indexing for Sample 5
"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder

# Load Sample 5
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[4]  # Sample 5 (index 4)

print(f"Sample: {sample['id']}")
print(f"Question: {sample['qa']['question']}")
print()

# Build KG
print("Building KG...")
builder = IntelligentKGBuilder()
graph = builder.build_kg(sample)
print(f"  ✓ KG built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")

# Get entity index
entity_index = builder.get_entity_index(graph)

print("Entity Index Contents:")
print(f"  by_text entries: {len(entity_index['by_text'])}")
print(f"  by_value entries: {len(entity_index['by_value'])}")
print(f"  by_label entries: {len(entity_index['by_label'])}")

# Look for revenue-related entries
print("\nRevenue-related entities:")
for key, items in entity_index['by_text'].items():
    if 'revenue' in key:
        for item in items:
            print(f"  Text: '{key}'")
            print(f"    -> Node: {item['id']}, value={item['data'].get('value', '')}, label={item['data'].get('label', '')}")

# Look for specific value 991.1
print("\n" + "="*60)
print("Looking for value 991.1:")
if 991.1 in entity_index['by_value']:
    for item in entity_index['by_value'][991.1]:
        data = item['data']
        print(f"  Found! Node: {item['id']}")
        print(f"    Type: {data.get('type', '')}")
        print(f"    Text: {data.get('text', '')}")
        print(f"    Value: {data.get('value', '')}")
        print(f"    Context: {data.get('context', '')}")
else:
    print("  NOT found by value 991.1!")

# Check by text '991.1'
if '991.1' in entity_index['by_text']:
    for item in entity_index['by_text']['991.1']:
        print(f"  Found by text '991.1'! Node: {item['id']}, value={item['data'].get('value', '')}")
else:
    print("  NOT found by text '991.1'!")

# Show all numeric values between 900-1000
print("\n" + "="*60)
print("All values between 900-1000:")
for value, items in entity_index['by_value'].items():
    try:
        val = float(value)
        if 900 <= val <= 1000:
            for item in items:
                data = item['data']
                print(f"  Value: {val}, Node: {item['id']}, text={data.get('text', '')[:30]}, context={data.get('context', '')[:60]}...")
    except:
        pass
