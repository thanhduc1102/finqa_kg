"""Analyze Sample 3 program requirements"""
import json

# Load data
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)

sample = data[2]  # Sample 3 (0-indexed)

print("=" * 80)
print("SAMPLE 3 DEEP DIVE")
print("=" * 80)

# Question
qa = sample['qa']
print(f"\n📝 Question: {qa['question']}")
print(f"✅ Expected Answer: {qa['exe_ans']}")
print(f"💻 Ground Truth Program: {qa['program']}")

# Parse program
print("\n" + "=" * 80)
print("PROGRAM BREAKDOWN")
print("=" * 80)

program = qa['program']
print(f"\nFull program: {program}")
print("\nStep-by-step:")

# The program is a comma-separated list
steps = [s.strip() for s in program.split(',')]
for i, step in enumerate(steps, 1):
    print(f"  {i}. {step}")

# Identify operation
if 'divide(' in program:
    print("\n🔍 Program Type: DIVISION")
    print("   → This is NOT a simple lookup!")
    print("   → Need to extract TWO values and divide them")

# Look at table
print("\n" + "=" * 80)
print("TABLE STRUCTURE")
print("=" * 80)

table = sample['table']
print(f"\nDimensions: {len(table)} rows × {len(table[0]) if table else 0} columns\n")

for i, row in enumerate(table):
    print(f"Row {i}: {row}")

# Look for key values
print("\n" + "=" * 80)
print("FINDING THE VALUES")
print("=" * 80)

# Look for 9896 and 0.236
print("\nSearching for 9896 (numerator)...")
for i, row in enumerate(table):
    for j, cell in enumerate(row):
        if '9896' in str(cell):
            print(f"  ✓ Found at Row {i}, Col {j}: {cell}")
            print(f"    Row label: {row[0] if j > 0 else 'N/A'}")

print("\nSearching for 23.6% or 0.236 (denominator)...")
for i, row in enumerate(table):
    for j, cell in enumerate(row):
        cell_str = str(cell).lower()
        if '23.6' in cell_str or '0.236' in cell_str or 'percent' in cell_str:
            print(f"  ✓ Found at Row {i}, Col {j}: {cell}")
            print(f"    Row label: {row[0] if j > 0 else 'N/A'}")

# Look at pre_text
print("\n" + "=" * 80)
print("CONTEXT CLUES")
print("=" * 80)

for i, text in enumerate(sample.get('pre_text', [])[:10]):
    text_lower = text.lower()
    if any(word in text_lower for word in ['fuel', 'million', 'percent', '23.6', '9896']):
        print(f"\n[{i}] {text}")

# Key insight
print("\n" + "=" * 80)
print("💡 KEY INSIGHTS")
print("=" * 80)

print("""
1. Question asks: "what was the fuel expense in 2018 in millions?"
   
2. Expected answer: 41932 million
   
3. Our current output: 9896 (just extracted the number from table)
   
4. Calculation needed: 9896 / 0.236 = 41932.2...
   
5. This means:
   - 9896 is fuel expense as a PERCENTAGE (23.6%) of something
   - 0.236 is the percentage value
   - We need to divide to get the actual millions: 9896 / 0.236 = ~41932

6. Detection strategy:
   - Question contains "in millions" → may need unit conversion
   - If we find a percentage value nearby, likely need division
   - Look for cells in same row/column that contain percentages
""")
