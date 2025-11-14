"""
KG Constant Coverage Check
Kiểm tra xem KG có chứa đầy đủ các constant values mà program cần không
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def extract_constants_from_program(program):
    """Extract all constant values needed by program"""
    if not program:
        return []
    
    constants = []
    
    # Parse program
    # Pattern to find all numeric arguments
    # Match: plain numbers, percentages, const_xxx
    patterns = [
        r'(\d+\.?\d*%?)',  # Numbers like 100, 3.8, 23.6%
        r'const_(\d+)',     # const_1000
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, program)
        for match in matches:
            # Clean and parse
            value_str = match.replace('%', '').replace('const_', '')
            try:
                value = float(value_str)
                constants.append({
                    'value': value,
                    'original': match
                })
            except:
                pass
    
    return constants

def extract_numbers_from_table(table):
    """Extract all numbers from table"""
    numbers = []
    
    for row_idx, row in enumerate(table):
        for col_idx, cell in enumerate(row):
            cell_str = str(cell).strip()
            
            # Skip empty or text-only cells
            if not cell_str or cell_str in ['-', 'n/a', 'N/A']:
                continue
            
            # Extract numbers from cell
            # Remove $, commas, handle percentages, handle parentheses (negative)
            cleaned = cell_str.replace(',', '').replace('$', '').strip()
            
            # Handle percentage
            is_percent = '%' in cleaned
            cleaned = cleaned.replace('%', '').strip()
            
            # Handle parentheses (negative number)
            is_negative = cleaned.startswith('(') and cleaned.endswith(')')
            if is_negative:
                cleaned = cleaned[1:-1].strip()
            
            # Try to parse
            try:
                value = float(cleaned)
                if is_negative:
                    value = -value
                
                numbers.append({
                    'value': value,
                    'original': cell_str,
                    'location': f'[{row_idx},{col_idx}]',
                    'is_percent': is_percent
                })
            except:
                # Not a number, skip
                pass
    
    return numbers

def extract_numbers_from_text(text_list):
    """Extract numbers from text sentences"""
    numbers = []
    
    for text in text_list:
        # Pattern for numbers
        pattern = r'-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?%?'
        matches = re.findall(pattern, text)
        
        for match in matches:
            cleaned = match.replace(',', '').replace('$', '').replace('%', '').strip()
            try:
                value = float(cleaned)
                numbers.append({
                    'value': value,
                    'original': match,
                    'context': text[:50] + '...'
                })
            except:
                pass
    
    return numbers

def check_constant_coverage(sample, sample_idx):
    """Check if data sources contain all constants needed by program"""
    
    # Extract required constants from program
    program = sample.get('qa', {}).get('program', '')
    required_constants = extract_constants_from_program(program)
    
    # Extract available numbers from table and text
    table = sample.get('table', [])
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    
    table_numbers = extract_numbers_from_table(table)
    text_numbers = extract_numbers_from_text(pre_text + post_text)
    
    # Check coverage
    result = {
        'sample_idx': sample_idx,
        'sample_id': sample.get('id', 'unknown'),
        'question': sample.get('qa', {}).get('question', ''),
        'program': program,
        'required_constants': required_constants,
        'available_table_numbers': table_numbers,
        'available_text_numbers': text_numbers,
        'coverage': {
            'found': [],
            'missing': [],
            'percentage': 0.0
        }
    }
    
    # Match each required constant
    for const in required_constants:
        found = False
        const_value = const['value']
        
        # Check in table numbers
        for num in table_numbers:
            if abs(num['value'] - const_value) < 0.01:  # Tolerance for floating point
                result['coverage']['found'].append({
                    'constant': const,
                    'found_in': 'table',
                    'location': num['location'],
                    'original': num['original']
                })
                found = True
                break
        
        # If not found in table, check in text
        if not found:
            for num in text_numbers:
                if abs(num['value'] - const_value) < 0.01:
                    result['coverage']['found'].append({
                        'constant': const,
                        'found_in': 'text',
                        'context': num['context']
                    })
                    found = True
                    break
        
        if not found:
            result['coverage']['missing'].append(const)
    
    # Calculate coverage percentage
    if required_constants:
        result['coverage']['percentage'] = (len(result['coverage']['found']) / len(required_constants)) * 100
    else:
        result['coverage']['percentage'] = 100.0
    
    return result

def print_coverage_result(result):
    """Print detailed coverage result"""
    print(f"\n{'='*80}")
    print(f"SAMPLE {result['sample_idx']} - CONSTANT COVERAGE")
    print(f"{'='*80}")
    
    print(f"\nQuestion: {result['question'][:70]}...")
    print(f"Program: {result['program']}")
    
    print(f"\n📊 Required Constants: {len(result['required_constants'])}")
    for const in result['required_constants']:
        print(f"   - {const['value']} (from {const['original']})")
    
    print(f"\n📦 Available Numbers:")
    print(f"   - Table: {len(result['available_table_numbers'])} numbers")
    print(f"   - Text: {len(result['available_text_numbers'])} numbers")
    
    coverage = result['coverage']
    print(f"\n✓ Coverage: {coverage['percentage']:.1f}%")
    print(f"   Found: {len(coverage['found'])}/{len(result['required_constants'])}")
    
    if coverage['found']:
        print(f"\n   ✅ Found constants:")
        for item in coverage['found']:
            const = item['constant']
            if item['found_in'] == 'table':
                print(f"      {const['value']} → table{item['location']} = {item['original']}")
            else:
                print(f"      {const['value']} → text: {item.get('context', '')[:40]}...")
    
    if coverage['missing']:
        print(f"\n   ❌ Missing constants:")
        for const in coverage['missing']:
            print(f"      {const['value']} (from {const['original']})")

def main():
    print("="*80)
    print("KG CONSTANT COVERAGE CHECK")
    print("Kiểm tra xem data sources có chứa đủ constants cho program")
    print("="*80)
    
    # Load dataset
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check coverage for first 50 samples
    num_samples = 50
    results = []
    
    print(f"\n📂 Checking {num_samples} samples...")
    
    for idx in range(num_samples):
        if idx < len(data):
            result = check_constant_coverage(data[idx], idx)
            results.append(result)
    
    # Print detailed results for first 10
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS - FIRST 10 SAMPLES")
    print(f"{'='*80}")
    
    for result in results[:10]:
        print_coverage_result(result)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    total_samples = len(results)
    perfect_coverage = sum(1 for r in results if r['coverage']['percentage'] >= 100)
    good_coverage = sum(1 for r in results if 80 <= r['coverage']['percentage'] < 100)
    poor_coverage = sum(1 for r in results if r['coverage']['percentage'] < 80)
    
    avg_coverage = sum(r['coverage']['percentage'] for r in results) / total_samples if results else 0
    
    total_required = sum(len(r['required_constants']) for r in results)
    total_found = sum(len(r['coverage']['found']) for r in results)
    total_missing = sum(len(r['coverage']['missing']) for r in results)
    
    print(f"\n📊 Coverage Distribution:")
    print(f"   Perfect (100%): {perfect_coverage}/{total_samples} ({perfect_coverage/total_samples*100:.1f}%)")
    print(f"   Good (80-99%): {good_coverage}/{total_samples} ({good_coverage/total_samples*100:.1f}%)")
    print(f"   Poor (<80%): {poor_coverage}/{total_samples} ({poor_coverage/total_samples*100:.1f}%)")
    print(f"   Average coverage: {avg_coverage:.1f}%")
    
    print(f"\n🎯 Constants Summary:")
    print(f"   Total required: {total_required}")
    coverage_pct = (total_found/total_required*100) if total_required > 0 else 0
    print(f"   Total found: {total_found} ({coverage_pct:.1f}%)")
    print(f"   Total missing: {total_missing}")
    
    # Worst performers
    if poor_coverage > 0:
        worst = sorted(results, key=lambda r: r['coverage']['percentage'])[:5]
        print(f"\n❌ Samples with Poorest Coverage:")
        for r in worst:
            if r['coverage']['percentage'] < 100:
                print(f"\n   Sample {r['sample_idx']} - {r['coverage']['percentage']:.1f}%")
                print(f"      Question: {r['question'][:60]}...")
                print(f"      Missing {len(r['coverage']['missing'])} constants")
    
    # Save results
    output_dir = Path(__file__).parent / "output" / "kg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "constant_coverage_check.json"
    
    # Simplify for JSON
    json_results = []
    for r in results:
        json_results.append({
            'sample_idx': r['sample_idx'],
            'sample_id': r['sample_id'],
            'question': r['question'],
            'coverage_percentage': r['coverage']['percentage'],
            'required_count': len(r['required_constants']),
            'found_count': len(r['coverage']['found']),
            'missing_count': len(r['coverage']['missing'])
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_samples': total_samples,
                'perfect_coverage': perfect_coverage,
                'good_coverage': good_coverage,
                'poor_coverage': poor_coverage,
                'avg_coverage': avg_coverage,
                'total_required': total_required,
                'total_found': total_found,
                'total_missing': total_missing
            },
            'results': json_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    # Key findings
    print(f"\n💡 KEY FINDINGS:")
    if avg_coverage >= 95:
        print(f"   ✅ Excellent! Data sources contain {avg_coverage:.1f}% of required constants on average")
        print(f"   → KG builder should capture these effectively")
    elif avg_coverage >= 80:
        print(f"   ⚠️ Good but improvable: {avg_coverage:.1f}% coverage")
        print(f"   → Some constants may be missing or in unexpected formats")
    else:
        print(f"   ❌ Poor coverage: only {avg_coverage:.1f}%")
        print(f"   → Need to improve number extraction or data preprocessing")
    
    if total_missing > 0:
        print(f"\n   Missing {total_missing} constants across all samples")
        print(f"   → Investigate: Are they implicit? In unexpected format? Actually missing?")

if __name__ == "__main__":
    main()
