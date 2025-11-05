import json
s = json.load(open('../FinQA/dataset/train.json'))[3]
print('Program:', s['qa'].get('program', 'N/A'))
print('Answer:', s['qa']['exe_ans'])
