"""
Deep analysis of Sample 3 to understand multi-step reasoning requirement
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'finqa_kg/src')))

def analyze_sample3():
    """Analyze Sample 3 in detail"""
    
    # Load data
    with open('../FinQA/dataset/train.json', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    sample = data[2]  # Sample 3 (0-indexed)
    
    print("=" * 80)
    print("SAMPLE 3 ANALYSIS")
    print("=" * 80)
    
    # Question analysis
    qa = sample['qa']
    print(f"\n📝 Question: {qa['question']}")
    print(f"❓ Question Type: {qa.get('question_type', 'unknown')}")
    print(f"✅ Expected Answer: {qa['exe_ans']}")
    print(f"💻 Ground Truth Program: {qa['program']}")
    
    # Parse the program
    print("\n" + "=" * 80)
    print("PROGRAM BREAKDOWN")
    print("=" * 80)
    
    program = qa['program']
    steps = program.split(', ')
    
    print(f"\nTotal Steps: {len(steps)}")
    for i, step in enumerate(steps, 1):
        print(f"  Step {i}: {step}")
    
    # Analyze what the program does
    print("\n" + "=" * 80)
    print("PROGRAM LOGIC")
    print("=" * 80)
    
    print("\n1️⃣ table_select_row('fuel', 3)")
    print("   → Select row containing 'fuel' at position 3")
    print("   → Returns: Row with fuel expense values")
    
    print("\n2️⃣ table_sum(#0)")
    print("   → Sum all values in the selected row")
    print("   → This gives total fuel expense across all years")
    
    print("\n3️⃣ table_select_row('total operating expenses', 2)")
    print("   → Select row containing 'total operating expenses' at position 2")
    
    print("\n4️⃣ table_select_col(#2, 3)")
    print("   → Select column 3 from the selected row")
    print("   → Gets total operating expenses for a specific year/column")
    
    print("\n5️⃣ divide(#1, #3)")
    print("   → Divide sum of fuel expenses by total operating expenses")
    print("   → This gives the percentage")
    
    print("\n6️⃣ multiply(#4, const_1000)")
    print("   → Multiply by 1000 to convert from billions to millions")
    print("   → Final answer: fuel expense in millions")
    
    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print("\n🔍 Why it requires multi-step reasoning:")
    print("   1. Question asks for 'fuel expense in 2018 in millions'")
    print("   2. Table likely shows values as percentages or in different units")
    print("   3. Need to:")
    print("      - Find fuel expense value")
    print("      - Find total operating expenses")
    print("      - Calculate: (fuel_expense / total_expenses) * 1000")
    
    print("\n🎯 What we're currently doing:")
    print("   - Extracting '9896' from table")
    print("   - Returning it directly as const_9896")
    print("   - Missing: The division by 0.236 (23.6%)")
    
    print("\n💡 What we need to detect:")
    print("   - Phrase 'in millions' indicates unit conversion needed")
    print("   - If value is a percentage of something, need division")
    print("   - Look for related values that could be denominators")
    
    # Look at the actual table
    print("\n" + "=" * 80)
    print("TABLE STRUCTURE")
    print("=" * 80)
    
    table = sample['table']
    print(f"\nTable dimensions: {len(table)} rows × {len(table[0]) if table else 0} columns")
    
    print("\nTable contents:")
    for i, row in enumerate(table[:10]):  # First 10 rows
        print(f"  Row {i}: {row}")
    
    # Look at pre_text for context
    print("\n" + "=" * 80)
    print("CONTEXT (pre_text)")
    print("=" * 80)
    
    for i, text in enumerate(sample.get('pre_text', [])[:5]):
        print(f"\n[{i}] {text}")

if __name__ == '__main__':
    analyze_sample3()
