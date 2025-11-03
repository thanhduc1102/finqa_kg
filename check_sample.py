import json

with open('FinQA/dataset/train.json') as f:
    data = json.load(f)

sample = data[0]
print('ID:', sample['id'])
print('\nTABLE (first 3 rows):')
for row in sample['table'][:3]:
    print(row)

print('\nQA:')
print('Question:', sample['qa']['question'])
print('Answer:', sample['qa']['answer'])
print('Program:', sample['qa']['program'])
print('Exe_ans:', sample['qa'].get('exe_ans', 'N/A'))
