"""
DEEP DIVE ANALYSIS - Investigate problem samples
=================================================

Xem xét chi tiết các samples có coverage thấp để hiểu nguyên nhân.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import re

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_sample_details(sample: Dict[str, Any]):
    """Print all details of a sample"""
    
    print("="*100)
    print(f"SAMPLE ID: {sample.get('id', 'N/A')}")
    print("="*100)
    
    # Question and Answer
    qa = sample.get('qa', {})
    print(f"\nQUESTION:")
    print(f"  {qa.get('question', 'N/A')}")
    print(f"\nEXPECTED ANSWER: {qa.get('exe_ans', 'N/A')}")
    print(f"\nPROGRAM:")
    print(f"  {qa.get('program', 'N/A')}")
    
    # Extract required numbers
    program = qa.get('program', '')
    const_pattern = r'const_(-?\d+(?:\.\d+)?)'
    required_numbers = [float(m) for m in re.findall(const_pattern, program)]
    print(f"\nREQUIRED NUMBERS: {required_numbers}")
    
    # Table
    table = sample.get('table', [])
    print(f"\nTABLE ({len(table)} rows):")
    for i, row in enumerate(table):
        print(f"  Row {i}: {row}")
    
    # Extract all numbers from table
    table_numbers = []
    for row in table:
        for cell in row:
            cell_str = str(cell).replace(',', '').replace('$', '').replace('%', '').strip()
            
            # Try to parse as float
            try:
                num = float(cell_str)
                table_numbers.append(num)
            except:
                # Try to find numbers in string
                num_matches = re.findall(r'-?\d+\.?\d*', cell_str)
                for match in num_matches:
                    try:
                        num = float(match)
                        if abs(num) > 0.001:
                            table_numbers.append(num)
                    except:
                        pass
    
    print(f"\nNUMBERS EXTRACTED FROM TABLE: {sorted(set(table_numbers))}")
    
    # Check what's missing
    print(f"\nCOVERAGE ANALYSIS:")
    for req_num in required_numbers:
        found = any(abs(tn - req_num) < 0.01 for tn in table_numbers)
        status = "✓ FOUND" if found else "✗ MISSING"
        print(f"  {req_num}: {status}")
    
    # Text
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    
    print(f"\nPRE-TEXT ({len(pre_text)} sentences):")
    for i, sent in enumerate(pre_text):
        print(f"  [{i}] {sent}")
    
    print(f"\nPOST-TEXT ({len(post_text)} sentences):")
    for i, sent in enumerate(post_text):
        print(f"  [{i}] {sent}")
    
    # Try to find required numbers in text
    all_text = ' '.join(pre_text + post_text)
    print(f"\nNUMBERS IN TEXT:")
    for req_num in required_numbers:
        # Try different formats
        found_in_text = False
        patterns = [
            str(int(req_num)) if req_num == int(req_num) else str(req_num),
            f"{req_num:,.0f}",
            f"{req_num:.2f}",
        ]
        for pattern in patterns:
            if pattern in all_text:
                found_in_text = True
                print(f"  {req_num}: Found as '{pattern}' in text")
                break
        if not found_in_text:
            print(f"  {req_num}: NOT found in text")
    
    print("\n" + "="*100 + "\n")


def main():
    """Main entry point"""
    
    print("DEEP DIVE ANALYSIS - Problem Samples")
    print("="*100)
    
    # Load data
    train_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "train.json"
    
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Problem sample IDs from quick analysis
    problem_ids = [
        'ABMD/2012/page_75.pdf-1',
        'IPG/2009/page_89.pdf-3',
        'CDNS/2018/page_32.pdf-2',
        'KHC/2017/page_21.pdf-4',
        'CDW/2013/page_106.pdf-2',
    ]
    
    # Find and analyze these samples
    for sample in data:
        sample_id = sample.get('id', '')
        if sample_id in problem_ids:
            print_sample_details(sample)


if __name__ == "__main__":
    main()
