"""Debug Sample 4 - percentage_of temporal matching"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder
from src.pipeline.question_analyzer import QuestionAnalyzer

# Load Sample 4
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[3]

print("="*80)
print("SAMPLE 4 DEBUG - percentage_of")
print("="*80)
print(f"ID: {sample['id']}")
print(f"Question: {sample['qa']['question']}")
print(f"Expected: {sample['qa']['exe_ans']}")
print()

# Show ground truth program
print(f"Ground Truth Program: {sample['qa'].get('program', 'N/A')}")
print()

# Check text
print("Pre-text (first 300 chars):")
print(sample['pre_text'][0][:300] if sample['pre_text'] else "N/A")
print()

# Check table
print("Table (first 10 rows):")
for i, row in enumerate(sample['table'][:10]):
    print(f"  Row {i}: {row}")
print()

# Build KG
print("Building KG...")
builder = IntelligentKGBuilder()
kg = builder.build_kg(sample)
entity_index = builder.get_entity_index(kg)
print(f"✓ KG: {kg.number_of_nodes()} nodes\n")

# Analyze question
analyzer = QuestionAnalyzer()
qa = analyzer.analyze(sample['qa']['question'])
print("Question Analysis:")
print(f"  Type: {qa.question_type}")
print(f"  Entities: {qa.entities_mentioned}")
print(f"  Temporal: {qa.temporal_entities}")
print()

# Look for "available-for-sale" entities
print("="*80)
print("ENTITIES containing 'available':")
print("="*80)
count = 0
for text_key, items in entity_index['by_text'].items():
    if 'available' in text_key:
        for item in items:
            data = item['data']
            print(f"  Text: '{text_key}'")
            print(f"    Node: {item['id']}, Value: {data.get('value', '')}")
            print(f"    Context: {data.get('context', '')[:120]}...")
            print()
            count += 1
if count == 0:
    print("  NOT FOUND\n")

# Look for year 2012 values
print("="*80)
print("ENTITIES with '2012' in context:")
print("="*80)
count = 0
for value, items in entity_index['by_value'].items():
    for item in items:
        context = item['data'].get('context', '')
        if '2012' in context:
            data = item['data']
            print(f"  Value: {value}")
            print(f"    Context: {context[:120]}...")
            print()
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break

# Look for specific values mentioned in program
print()
print("="*80)
print("Looking for expected values (14001, 26302):")
print("="*80)
for target_val in [14001, 26302, 14001.0, 26302.0]:
    if target_val in entity_index['by_value']:
        for item in entity_index['by_value'][target_val]:
            data = item['data']
            print(f"  ✓ Found {target_val}:")
            print(f"    Node: {item['id']}, Label: {data.get('label', '')}")
            print(f"    Context: {data.get('context', '')[:120]}...")
            print()
