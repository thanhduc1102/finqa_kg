"""Debug Sample 3 - compound term extraction"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder

# Load Sample 3
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[2]

print("="*80)
print("SAMPLE 3 DEBUG")
print("="*80)
print(f"ID: {sample['id']}")
print(f"Question: {sample['qa']['question']}")
print(f"Expected Answer: {sample['qa']['exe_ans']}")
print()

# Check text content
print("Pre-text (first 500 chars):")
for text in sample['pre_text'][:2]:
    print(f"  {text[:500]}...")
print()

# Check table
print("Table:")
for i, row in enumerate(sample['table'][:5]):
    print(f"  Row {i}: {row}")
print()

# Build KG and check entities
print("Building KG...")
builder = IntelligentKGBuilder()
kg = builder.build_kg(sample)
entity_index = builder.get_entity_index(kg)

print(f"✓ KG: {kg.number_of_nodes()} nodes")
print()

# Look for "expense" entities
print("="*80)
print("ENTITIES containing 'expense':")
print("="*80)
if 'expense' in entity_index['by_text']:
    for item in entity_index['by_text']['expense']:
        data = item['data']
        print(f"  Node: {item['id']}")
        print(f"    Text: '{data.get('text', '')}'")
        print(f"    Value: {data.get('value', '')}")
        print(f"    Label: {data.get('label', '')}")
        print(f"    Context: {data.get('context', '')[:100]}...")
        print()
else:
    print("  NOT FOUND in by_text!")

# Look for "operating" entities
print("="*80)
print("ENTITIES containing 'operating':")
print("="*80)
if 'operating' in entity_index['by_text']:
    for item in entity_index['by_text']['operating']:
        data = item['data']
        print(f"  Node: {item['id']}")
        print(f"    Text: '{data.get('text', '')}'")
        print(f"    Value: {data.get('value', '')}")
        print(f"    Context: {data.get('context', '')[:100]}...")
        print()
else:
    print("  NOT FOUND!")

# Look for compound terms
print("="*80)
print("ENTITIES containing 'operating expense' (compound):")
print("="*80)
if 'operating expense' in entity_index['by_text']:
    print("  FOUND!")
else:
    print("  NOT FOUND - compound term not extracted!")

# Look for numeric values in expected range (40000-45000)
print()
print("="*80)
print("NUMERIC VALUES between 40000-45000:")
print("="*80)
found_any = False
for value, items in entity_index['by_value'].items():
    try:
        val = float(value)
        if 40000 <= val <= 45000:
            found_any = True
            for item in items:
                data = item['data']
                print(f"  Value: {val}")
                print(f"    Node: {item['id']}")
                print(f"    Text: '{data.get('text', '')}'")
                print(f"    Context: {data.get('context', '')[:100]}...")
                print()
    except:
        pass

if not found_any:
    print("  NOT FOUND - expected value ~41932 not in KG!")
    
# Check what the highest MONEY values are
print()
print("="*80)
print("TOP 10 MONEY values:")
print("="*80)
money_values = []
for value, items in entity_index['by_value'].items():
    for item in items:
        if item['data'].get('label') == 'MONEY':
            try:
                money_values.append((float(value), item))
            except:
                pass

money_values.sort(reverse=True)
for val, item in money_values[:10]:
    data = item['data']
    print(f"  {val}: text='{data.get('text', '')}', context={data.get('context', '')[:80]}...")
