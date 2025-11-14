"""
QUICK KG QUALITY ANALYSIS (Lightweight Version)
================================================

Phân tích nhanh KG quality mà không cần load heavy models.
Chỉ kiểm tra structure và coverage.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import re

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_numbers_from_program(program: str) -> List[float]:
    """Extract numbers from program string"""
    if not program:
        return []
    
    numbers = []
    # Pattern: const_X or const_X.X
    const_pattern = r'const_(-?\d+(?:\.\d+)?)'
    matches = re.findall(const_pattern, program)
    
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers


def extract_numbers_from_table(table: List[List[str]]) -> List[float]:
    """Extract all numbers from table"""
    numbers = []
    
    for row in table:
        for cell in row:
            # Try multiple patterns
            cell_str = str(cell).replace(',', '').replace('$', '').replace('%', '').strip()
            
            # Pattern 1: Simple number
            try:
                num = float(cell_str)
                numbers.append(num)
                continue
            except:
                pass
            
            # Pattern 2: Number with parentheses (negative)
            if '(' in cell_str and ')' in cell_str:
                cell_str = cell_str.replace('(', '-').replace(')', '')
                try:
                    num = float(cell_str)
                    numbers.append(num)
                    continue
                except:
                    pass
            
            # Pattern 3: Extract any number from string
            num_matches = re.findall(r'-?\d+\.?\d*', cell_str)
            for match in num_matches:
                try:
                    num = float(match)
                    if abs(num) > 0.001:  # Skip very small numbers (likely decimals)
                        numbers.append(num)
                except:
                    pass
    
    return numbers


def analyze_sample_basic(sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    """
    Phân tích cơ bản một sample mà không cần build KG
    
    Kiểm tra:
    1. Table có chứa đủ numbers cần thiết không?
    2. Text có chứa context cần thiết không?
    3. Structure của table và text như thế nào?
    """
    
    analysis = {
        'sample_id': sample.get('id', f'sample_{sample_idx}'),
        'sample_idx': sample_idx,
    }
    
    # Get ground truth
    qa = sample.get('qa', {})
    question = qa.get('question', '')
    program = qa.get('program', '')
    answer = qa.get('exe_ans', '')
    
    # Extract required numbers from program
    required_numbers = extract_numbers_from_program(program)
    
    # Extract available numbers from table
    table = sample.get('table', [])
    table_numbers = extract_numbers_from_table(table)
    
    # Check coverage
    found_count = 0
    missing_numbers = []
    
    for req_num in required_numbers:
        found = False
        for table_num in table_numbers:
            if abs(table_num - req_num) < 0.01:
                found = True
                break
        if found:
            found_count += 1
        else:
            missing_numbers.append(req_num)
    
    number_coverage = found_count / len(required_numbers) if required_numbers else 1.0
    
    # Analyze table structure
    table_rows = len(table)
    table_cols = len(table[0]) if table else 0
    table_cells = sum(len(row) for row in table)
    
    # Analyze text
    pre_text = sample.get('pre_text', [])
    post_text = sample.get('post_text', [])
    total_text_sentences = len(pre_text) + len(post_text)
    
    # Combine all text
    all_text = ' '.join(pre_text + post_text).lower()
    
    # Extract keywords from question
    question_lower = question.lower()
    question_words = [w.strip('.,!?;:') for w in question_lower.split()]
    stop_words = {'what', 'is', 'the', 'was', 'were', 'are', 'in', 'of', 'for', 'to', 'a', 'an', 
                 'and', 'or', 'but', 'on', 'at', 'by', 'from', 'with', 'as', 'be', 'been', 'being',
                 'how', 'many', 'much', 'did', 'do', 'does', 'would', 'could', 'should', '?'}
    keywords = [w for w in question_words if w not in stop_words and len(w) > 2]
    
    # Check if keywords appear in text
    found_keywords = [k for k in keywords if k in all_text]
    keyword_coverage = len(found_keywords) / len(keywords) if keywords else 1.0
    
    # Overall assessment
    analysis['data_structure'] = {
        'table_rows': table_rows,
        'table_cols': table_cols,
        'table_cells': table_cells,
        'text_sentences': total_text_sentences,
    }
    
    analysis['number_analysis'] = {
        'required_count': len(required_numbers),
        'found_count': found_count,
        'missing_count': len(missing_numbers),
        'coverage': number_coverage,
        'missing_numbers': missing_numbers,
        'required_numbers': required_numbers,
        'table_numbers': table_numbers,
    }
    
    analysis['text_analysis'] = {
        'keywords': keywords,
        'found_keywords': found_keywords,
        'keyword_coverage': keyword_coverage,
    }
    
    analysis['question_info'] = {
        'question': question,
        'program': program,
        'answer': answer,
    }
    
    # Overall score
    analysis['overall_score'] = number_coverage * 0.7 + keyword_coverage * 0.3
    
    return analysis


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick KG Quality Analysis')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to analyze (default: 50)')
    
    args = parser.parse_args()
    
    print("="*100)
    print("QUICK KG QUALITY ANALYSIS (No Model Loading)")
    print("="*100)
    print(f"Analyzing {args.num_samples} samples...")
    print()
    
    # Load data
    train_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "train.json"
    
    if not train_path.exists():
        print(f"✗ Error: train.json not found at {train_path}")
        return
    
    print(f"Loading data from: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples\n")
    
    # Analyze samples
    all_analyses = []
    
    for i, sample in enumerate(data[:args.num_samples]):
        print(f"Analyzing sample {i+1}/{args.num_samples}...", end='\r')
        
        try:
            analysis = analyze_sample_basic(sample, i)
            all_analyses.append(analysis)
        except Exception as e:
            print(f"\n✗ Error analyzing sample {i}: {e}")
            continue
    
    print(f"\n\n{'='*100}")
    print("ANALYSIS RESULTS")
    print("="*100)
    
    # 1. Number Coverage Summary
    print("\n1. NUMBER COVERAGE (Most Important!)")
    print("-" * 100)
    
    avg_number_coverage = sum(a['number_analysis']['coverage'] for a in all_analyses) / len(all_analyses)
    perfect_coverage = sum(1 for a in all_analyses if a['number_analysis']['coverage'] >= 1.0)
    low_coverage = sum(1 for a in all_analyses if a['number_analysis']['coverage'] < 0.5)
    
    print(f"Average number coverage: {avg_number_coverage:.2%}")
    print(f"Samples with 100% coverage: {perfect_coverage}/{len(all_analyses)} ({perfect_coverage/len(all_analyses):.1%})")
    print(f"Samples with <50% coverage: {low_coverage}/{len(all_analyses)} ({low_coverage/len(all_analyses):.1%})")
    
    # Show samples with issues
    print(f"\nSamples with coverage issues:")
    problem_samples = [a for a in all_analyses if a['number_analysis']['coverage'] < 1.0]
    problem_samples.sort(key=lambda x: x['number_analysis']['coverage'])
    
    for a in problem_samples[:10]:  # Show worst 10
        print(f"\n  Sample: {a['sample_id']}")
        print(f"  Coverage: {a['number_analysis']['coverage']:.1%}")
        print(f"  Required: {a['number_analysis']['required_numbers']}")
        print(f"  Missing: {a['number_analysis']['missing_numbers']}")
        print(f"  Question: {a['question_info']['question'][:80]}...")
    
    # 2. Text Coverage Summary
    print(f"\n\n2. TEXT/KEYWORD COVERAGE")
    print("-" * 100)
    
    avg_keyword_coverage = sum(a['text_analysis']['keyword_coverage'] for a in all_analyses) / len(all_analyses)
    perfect_keyword = sum(1 for a in all_analyses if a['text_analysis']['keyword_coverage'] >= 1.0)
    
    print(f"Average keyword coverage: {avg_keyword_coverage:.2%}")
    print(f"Samples with 100% keyword coverage: {perfect_keyword}/{len(all_analyses)} ({perfect_keyword/len(all_analyses):.1%})")
    
    # 3. Data Structure Statistics
    print(f"\n\n3. DATA STRUCTURE STATISTICS")
    print("-" * 100)
    
    avg_table_rows = sum(a['data_structure']['table_rows'] for a in all_analyses) / len(all_analyses)
    avg_table_cols = sum(a['data_structure']['table_cols'] for a in all_analyses) / len(all_analyses)
    avg_table_cells = sum(a['data_structure']['table_cells'] for a in all_analyses) / len(all_analyses)
    avg_text_sentences = sum(a['data_structure']['text_sentences'] for a in all_analyses) / len(all_analyses)
    
    print(f"Average table rows: {avg_table_rows:.1f}")
    print(f"Average table columns: {avg_table_cols:.1f}")
    print(f"Average table cells: {avg_table_cells:.1f}")
    print(f"Average text sentences: {avg_text_sentences:.1f}")
    
    # 4. Overall Assessment
    print(f"\n\n4. OVERALL ASSESSMENT")
    print("-" * 100)
    
    avg_overall = sum(a['overall_score'] for a in all_analyses) / len(all_analyses)
    print(f"Average overall score: {avg_overall:.2%}")
    
    if avg_number_coverage < 0.8:
        print("\n❌ CRITICAL ISSUE: Number coverage is LOW!")
        print("   Many samples are missing required numbers from tables.")
        print("   Possible causes:")
        print("   - Number extraction regex is not comprehensive enough")
        print("   - Table parsing is missing cells")
        print("   - Numbers are in text but not being extracted from there")
    elif avg_number_coverage < 0.95:
        print("\n⚠ WARNING: Number coverage could be better")
        print("   Some edge cases are not being handled")
    else:
        print("\n✓ GOOD: Number coverage is excellent!")
    
    if avg_keyword_coverage < 0.7:
        print("\n⚠ WARNING: Keyword coverage is low")
        print("   Text might not contain enough context for questions")
    else:
        print("\n✓ GOOD: Text contains relevant keywords")
    
    # 5. Recommendations
    print(f"\n\n5. RECOMMENDATIONS FOR KG BUILDER")
    print("-" * 100)
    
    print("\nBased on analysis, the KG builder should:")
    print("\n✓ MUST DO:")
    print("  1. Extract ALL numbers from table cells (with various formats)")
    print("  2. Handle numbers with: $, %, commas, parentheses (negative)")
    print("  3. Create entity nodes for each number with its value")
    print("  4. Store cell position (row, column) for each number")
    print("  5. Link numbers to their row/column headers for context")
    
    print("\n✓ SHOULD DO:")
    print("  6. Extract entities from text (dates, organizations, etc.)")
    print("  7. Link text entities to table entities when semantically related")
    print("  8. Preserve table structure (TABLE → ROW → CELL hierarchy)")
    print("  9. Add relations between related entities")
    
    print("\n✓ GOOD TO HAVE:")
    print("  10. Use embeddings for semantic similarity")
    print("  11. Add temporal relations for dates")
    print("  12. Group related information (e.g., same company, same year)")
    
    # Save detailed report
    output_dir = Path(__file__).parent / "output" / "kg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_path = output_dir / f"quick_analysis_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n✓ Detailed results saved to: {json_path}")
    
    # Save text report
    report_path = output_dir / f"quick_analysis_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("QUICK KG QUALITY ANALYSIS REPORT\n")
        f.write("="*100 + "\n\n")
        f.write(f"Analyzed {len(all_analyses)} samples\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Average number coverage: {avg_number_coverage:.2%}\n")
        f.write(f"Average keyword coverage: {avg_keyword_coverage:.2%}\n")
        f.write(f"Average overall score: {avg_overall:.2%}\n\n")
        
        f.write("\nSamples with coverage < 100%:\n")
        for a in problem_samples[:20]:
            f.write(f"\n{a['sample_id']}: {a['number_analysis']['coverage']:.1%}\n")
            f.write(f"  Required: {a['number_analysis']['required_numbers']}\n")
            f.write(f"  Missing: {a['number_analysis']['missing_numbers']}\n")
            f.write(f"  Question: {a['question_info']['question']}\n")
    
    print(f"✓ Text report saved to: {report_path}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    main()
