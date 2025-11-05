"""Check if 'total operating expenses' appears in Sample 3"""
import json
import re

# Load Sample 3
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[2]

print("Question:", sample['qa']['question'])
print()

# Check pre_text
print("="*80)
print("Searching in pre_text:")
print("="*80)
for i, text in enumerate(sample['pre_text']):
    if 'operating' in text.lower() and 'expense' in text.lower():
        print(f"\nPre-text[{i}]:")
        print(text)
        print()
        
        # Check regex match
        pattern = r'(?:total\s+)?operating\s+expenses?'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            print(f"  ✓ REGEX MATCHES: {len(matches)}")
            for match in matches:
                print(f"    - '{match.group(0)}' at position {match.start()}-{match.end()}")
        else:
            print("  ✗ No regex matches")

# Check post_text
print()
print("="*80)
print("Searching in post_text:")
print("="*80)
for i, text in enumerate(sample['post_text']):
    if 'operating' in text.lower() and 'expense' in text.lower():
        print(f"\nPost-text[{i}]:")
        print(text)

# Check table headers/cells
print()
print("="*80)
print("Searching in table:")
print("="*80)
for i, row in enumerate(sample['table']):
    for j, cell in enumerate(row):
        if 'operating' in str(cell).lower() and 'expense' in str(cell).lower():
            print(f"  Row {i}, Col {j}: '{cell}'")
            
            # Check regex
            pattern = r'(?:total\s+)?operating\s+expenses?'
            if re.search(pattern, str(cell), re.IGNORECASE):
                print(f"    ✓ Matches regex!")
            else:
                print(f"    ✗ Doesn't match regex (missing space?)")
