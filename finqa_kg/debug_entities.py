"""Debug script: Xem các entities được extract"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder

# Load sample
with open('/mnt/e/AI/FinQA_research/FinQA/dataset/train.json') as f:
    data = json.load(f)
sample = data[0]

# Build KG
print("Building KG...")
builder = IntelligentKGBuilder()
kg = builder.build_kg(sample)
entity_index = builder.get_entity_index(kg)

print(f"\nTotal entities/cells with values: {len(entity_index['by_value'])}")
print(f"Entity index by_value keys (first 20): {list(entity_index['by_value'].keys())[:20]}")

# Find entities with "interest expense" in context
print('\n' + '='*60)
print('Entities with "interest" AND "expense" in context:')
print('='*60)

candidates = []
for node_id, node_data in kg.nodes(data=True):
    if node_data.get('type') in ['entity', 'cell'] and node_data.get('value'):
        context = node_data.get('context', '').lower()
        if 'interest' in context and 'expense' in context:
            candidates.append({
                'value': node_data['value'],
                'text': node_data.get('text', ''),
                'type': node_data.get('type'),
                'label': node_data.get('label', ''),
                'context': context
            })

print(f"Found {len(candidates)} candidates")
for i, cand in enumerate(candidates[:10]):
    print(f"\n{i+1}. VALUE={cand['value']}, type={cand['type']}, label={cand['label']}")
    print(f"   text: {cand['text']}")
    print(f"   context: {cand['context'][:200]}...")
    
    # Check if 3.8 is mentioned
    if '3.8' in cand['context'] or '$ 3.8' in cand['context']:
        print(f"   >>> HAS 3.8! <<<")

# Check if 3.8 exists anywhere
print('\n' + '='*60)
print('Does value 3.8 exist in index?')
print('='*60)
if 3.8 in entity_index['by_value']:
    print(f"YES! Found {len(entity_index['by_value'][3.8])} entities with value=3.8")
    for match in entity_index['by_value'][3.8]:
        print(f"  Node: {match['id']}")
        print(f"  Type: {match['data'].get('type')}")
        print(f"  Text: {match['data'].get('text')}")
        print(f"  Context: {match['data'].get('context', '')[:200]}...")
else:
    print("NO! 3.8 not found in entity index")
    
print('\n' + '='*60)
print('Check for "$3.8 million" or similar:')
print('='*60)
for node_id, node_data in kg.nodes(data=True):
    if node_data.get('type') in ['entity', 'cell']:
        text = node_data.get('text', '')
        if '3.8' in text or (node_data.get('value') and abs(node_data['value'] - 3.8) < 0.01):
            print(f"Found: value={node_data.get('value')}, text={text}, type={node_data.get('type')}, label={node_data.get('label')}")
