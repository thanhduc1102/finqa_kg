import json
data = json.load(open('../FinQA/dataset/train.json'))
s = data[4]
print(f"Question: {s['qa']['question']}")
print(f"Program: {s['qa'].get('program', 'N/A')}")
print(f"Answer: {s['qa']['exe_ans']}")
