"""
Simple Sync KG Analysis
Phân tích đơn giản và nhanh về KG structure
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_sample(idx=0):
    """Load one sample from dataset"""
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[idx]

def analyze_sample_structure(sample, idx):
    """Analyze what information is available in a sample"""
    print(f"\n{'='*80}")
    print(f"SAMPLE {idx} STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    # Question & Answer
    qa = sample.get('qa', {})
    question = qa.get('question', '')
    answer = qa.get('answer', '')
    program = qa.get('program', '')
    
    print(f"\n📝 Question: {question}")
    print(f"✓ Answer: {answer}")
    print(f"🔧 Program: {program}")
    
    # Table
    table = sample.get('table', [])
    if table:
        print(f"\n📊 TABLE:")
        print(f"   Dimensions: {len(table)} rows × {len(table[0]) if table else 0} columns")
        print(f"\n   Header (Row 0):")
        if table:
            for i, cell in enumerate(table[0]):
                print(f"      Col {i}: {cell}")
        
        print(f"\n   First data row (Row 1):")
        if len(table) > 1:
            for i, cell in enumerate(table[1]):
                print(f"      Col {i}: {cell}")
        
        # Show full table
        print(f"\n   Full Table:")
        for i, row in enumerate(table):
            print(f"      Row {i}: {row}")
    
    # Text
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    
    print(f"\n📄 TEXT:")
    print(f"   Pre-text: {len(pre_text)} sentences")
    for i, text in enumerate(pre_text[:3]):
        print(f"      {i}: {text[:80]}...")
    
    if post_text:
        print(f"   Post-text: {len(post_text)} sentences")
        for i, text in enumerate(post_text[:3]):
            print(f"      {i}: {text[:80]}...")
    
    return {
        'question': question,
        'answer': answer,
        'program': program,
        'table': table,
        'pre_text': pre_text,
        'post_text': post_text
    }

def trace_program_execution(program, sample_data):
    """Trace what data the program needs"""
    print(f"\n{'='*80}")
    print(f"PROGRAM EXECUTION TRACE")
    print(f"{'='*80}")
    
    print(f"\nProgram: {program}")
    
    # Extract operations
    import re
    operations = re.findall(r'(\w+)\((.*?)\)', program)
    
    print(f"\n🔍 Operations breakdown:")
    for i, (op, args) in enumerate(operations, 1):
        print(f"\n   Step {i}: {op}({args})")
        
        # Analyze arguments
        arg_list = [a.strip() for a in args.split(',')]
        for arg in arg_list:
            if arg.startswith('const_'):
                # Constant value
                value = arg.replace('const_', '')
                print(f"      ├─ Constant: {value}")
            
            elif arg.startswith('#'):
                # Table reference
                ref = arg.replace('#', '')
                print(f"      ├─ Table reference: #{ref}")
                # Try to locate in table
                table = sample_data.get('table', [])
                if table:
                    try:
                        ref_int = int(ref)
                        # Find in table
                        found = False
                        for row_idx, row in enumerate(table):
                            for col_idx, cell in enumerate(row):
                                if str(cell).strip() == ref or ref in str(cell):
                                    print(f"      │  → Found in table[{row_idx}][{col_idx}]: {cell}")
                                    found = True
                                    break
                        if not found:
                            print(f"      │  ⚠️ Reference #{ref} not easily found in table")
                    except:
                        print(f"      │  (complex reference)")
            
            elif arg.startswith('table_'):
                # Table column operation
                print(f"      ├─ Table operation: {arg}")
    
    return operations

def main():
    print("="*80)
    print("SIMPLE KNOWLEDGE GRAPH ANALYSIS")
    print("="*80)
    
    # Analyze a few samples
    samples_to_analyze = [0, 1, 2]
    
    for idx in samples_to_analyze:
        sample = load_sample(idx)
        sample_data = analyze_sample_structure(sample, idx)
        
        # Trace program
        program = sample_data['program']
        if program:
            trace_program_execution(program, sample_data)
        
        # Ask: Can we extract all needed info?
        print(f"\n{'='*80}")
        print(f"COVERAGE ASSESSMENT")
        print(f"{'='*80}")
        
        table = sample_data['table']
        pre_text = sample_data['pre_text']
        post_text = sample_data['post_text']
        
        print(f"\n✓ Data Sources Available:")
        print(f"   - Table: {len(table)} rows with {len(table[0]) if table else 0} columns")
        print(f"   - Text: {len(pre_text) + len(post_text)} sentences")
        
        # Count numbers in table
        import re
        all_numbers = []
        for row in table:
            for cell in row:
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', str(cell))
                all_numbers.extend(numbers)
        
        print(f"   - Numbers in table: {len(all_numbers)} (unique: {len(set(all_numbers))})")
        
        # Count numbers in text
        text_numbers = []
        for text in pre_text + post_text:
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
            text_numbers.extend(numbers)
        
        print(f"   - Numbers in text: {len(text_numbers)} (unique: {len(set(text_numbers))})")
        
        total_unique = len(set(all_numbers + text_numbers))
        print(f"\n   📊 Total unique numbers: {total_unique}")
        
        # Extract needed arguments
        program = sample_data['program']
        needed_args = len(re.findall(r'const_\w+|#\d+', program))
        print(f"   🎯 Arguments needed by program: {needed_args}")
        
        if total_unique >= needed_args:
            print(f"   ✅ Sufficient data available")
        else:
            print(f"   ⚠️ May need more data: have {total_unique}, need {needed_args}")
        
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
