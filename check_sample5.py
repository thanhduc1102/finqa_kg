import json

with open('FinQA/dataset/train.json') as f:
    data = json.load(f)

# Sample 5 (index 4)
s = data[4]
print('Sample ID:', s['id'])
print('Question:', s['qa']['question'])
print('Answer:', s['qa']['exe_ans'])
print('Program:', s['qa']['program'])
print('\nTable (first 5 rows):')
for row in s['table'][:5]:
    print(row)

print('\nPre-text (first 3):')
for text in s['pre_text'][:3]:
    print(text[:100], '...')
