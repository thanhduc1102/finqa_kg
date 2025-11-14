"""
Deep Analysis Script for Knowledge Graph Quality
Phân tích sâu về chất lượng đồ thị tri thức - từng khía cạnh cụ thể
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_dataset(max_samples=50):
    """Load FinQA dataset"""
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:max_samples]

def extract_numbers_from_text(text):
    """Extract all numbers from text"""
    # Pattern for numbers with commas, decimals, percentages, etc.
    patterns = [
        r'-?\$?\s*\d+(?:,\d{3})*(?:\.\d+)?%?',  # Numbers with $ and %
        r'-?\d+(?:,\d{3})*(?:\.\d+)?',  # Regular numbers
    ]
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)
    return list(set(numbers))

def extract_numbers_from_table(table):
    """Extract all numbers from table"""
    numbers = set()
    for row in table:
        for cell in row:
            cell_str = str(cell)
            extracted = extract_numbers_from_text(cell_str)
            numbers.update(extracted)
    return list(numbers)

def analyze_single_sample(sample, sample_idx):
    """Phân tích chi tiết một mẫu duy nhất"""
    analysis = {
        'sample_id': sample.get('id', f'sample_{sample_idx}'),
        'question': sample.get('qa', {}).get('question', ''),
        'answer': sample.get('qa', {}).get('answer', ''),
        'program': sample.get('qa', {}).get('program', ''),
        'pre_text': sample.get('pre_text', []),
        'post_text': sample.get('post_text', []),
        'table': sample.get('table', []),
        'issues': [],
        'metrics': {}
    }
    
    # 1. Phân tích số liệu trong câu hỏi
    question = analysis['question']
    program = analysis['program']
    answer = analysis['answer']
    
    # Extract numbers mentioned in question
    numbers_in_question = extract_numbers_from_text(question)
    analysis['metrics']['numbers_in_question'] = len(numbers_in_question)
    
    # 2. Phân tích program - extract arguments needed
    program_args = []
    if program:
        # Extract const_xxx and table_xxx references
        const_matches = re.findall(r'const_\d+|const_m?\d+', program)
        table_matches = re.findall(r'table_\d+', program)
        program_args = const_matches + table_matches
    
    analysis['metrics']['program_args_count'] = len(program_args)
    analysis['program_args'] = program_args
    
    # 3. Phân tích bảng
    table = analysis['table']
    if table:
        analysis['metrics']['table_rows'] = len(table)
        analysis['metrics']['table_cols'] = len(table[0]) if table else 0
        
        # Extract all numbers from table
        table_numbers = extract_numbers_from_table(table)
        analysis['metrics']['table_numbers_count'] = len(table_numbers)
        analysis['table_numbers'] = table_numbers
        
        # Check for header row
        if table:
            first_row = table[0]
            has_numeric_header = any(re.search(r'\d', str(cell)) for cell in first_row)
            analysis['metrics']['has_numeric_header'] = has_numeric_header
    else:
        analysis['issues'].append("NO_TABLE")
    
    # 4. Phân tích text
    pre_text = analysis['pre_text']
    post_text = analysis['post_text']
    
    all_text = ' '.join(pre_text + post_text)
    text_numbers = extract_numbers_from_text(all_text)
    analysis['metrics']['text_numbers_count'] = len(text_numbers)
    analysis['text_numbers'] = text_numbers
    
    analysis['metrics']['pre_text_sentences'] = len(pre_text)
    analysis['metrics']['post_text_sentences'] = len(post_text)
    
    # 5. Critical checks
    total_numbers = len(set(table_numbers + text_numbers))
    analysis['metrics']['total_unique_numbers'] = total_numbers
    
    if total_numbers < len(program_args):
        analysis['issues'].append(f"INSUFFICIENT_NUMBERS: need {len(program_args)}, have {total_numbers}")
    
    # Check if question references specific entities
    if 'million' in question.lower() or 'billion' in question.lower():
        analysis['issues'].append("SCALE_CONVERSION_NEEDED")
    
    if 'percent' in question.lower() or '%' in question:
        analysis['issues'].append("PERCENTAGE_INVOLVED")
    
    # Temporal keywords
    temporal_keywords = ['year', 'quarter', 'month', 'period', 'fiscal', 'ended']
    if any(kw in question.lower() for kw in temporal_keywords):
        analysis['issues'].append("TEMPORAL_CONTEXT_NEEDED")
    
    # Check for comparison
    comparison_keywords = ['increase', 'decrease', 'change', 'difference', 'more', 'less', 'higher', 'lower']
    if any(kw in question.lower() for kw in comparison_keywords):
        analysis['issues'].append("COMPARISON_OPERATION")
    
    return analysis

def analyze_entity_extraction_quality(samples):
    """Đánh giá chất lượng entity extraction"""
    print("\n" + "="*80)
    print("ENTITY EXTRACTION QUALITY ANALYSIS")
    print("="*80)
    
    stats = {
        'total_samples': len(samples),
        'numbers_extracted': [],
        'temporal_entities': [],
        'missing_entities': [],
        'table_coverage': [],
        'text_coverage': []
    }
    
    for idx, sample in enumerate(samples):
        analysis = analyze_single_sample(sample, idx)
        
        # Number extraction coverage
        total_nums = analysis['metrics'].get('total_unique_numbers', 0)
        needed_nums = analysis['metrics'].get('program_args_count', 0)
        
        if needed_nums > 0:
            coverage = min(100, (total_nums / needed_nums) * 100)
            stats['numbers_extracted'].append(coverage)
        
        # Count issues
        if 'TEMPORAL_CONTEXT_NEEDED' in analysis['issues']:
            stats['temporal_entities'].append(idx)
        
        if 'INSUFFICIENT_NUMBERS' in str(analysis['issues']):
            stats['missing_entities'].append(idx)
    
    # Report
    print(f"\n📊 Entity Extraction Coverage:")
    if stats['numbers_extracted']:
        avg_coverage = sum(stats['numbers_extracted']) / len(stats['numbers_extracted'])
        print(f"   Average number coverage: {avg_coverage:.1f}%")
        print(f"   Samples with 100% coverage: {sum(1 for c in stats['numbers_extracted'] if c >= 100)}/{len(stats['numbers_extracted'])}")
        print(f"   Samples with <50% coverage: {sum(1 for c in stats['numbers_extracted'] if c < 50)}/{len(stats['numbers_extracted'])}")
    
    print(f"\n⏰ Temporal Entity Requirements:")
    print(f"   Samples needing temporal context: {len(stats['temporal_entities'])}/{len(samples)}")
    
    print(f"\n⚠️ Missing Entities:")
    print(f"   Samples with insufficient numbers: {len(stats['missing_entities'])}/{len(samples)}")
    
    return stats

def analyze_relation_quality(samples):
    """Đánh giá chất lượng relation extraction"""
    print("\n" + "="*80)
    print("RELATION EXTRACTION QUALITY ANALYSIS")
    print("="*80)
    
    relation_types = Counter()
    complex_relations = []
    
    for idx, sample in enumerate(samples):
        analysis = analyze_single_sample(sample, idx)
        
        # Analyze required relations from program
        program = analysis.get('program', '')
        if not program:
            continue
        
        # Count operations - each operation implies a relation
        operations = re.findall(r'(add|subtract|multiply|divide|exp|greater|table_sum|table_average|table_max|table_min)', program)
        relation_types.update(operations)
        
        # Complex relations = multiple operations
        if len(operations) > 2:
            complex_relations.append({
                'idx': idx,
                'id': analysis['sample_id'],
                'operations': len(operations),
                'types': operations
            })
    
    print(f"\n📈 Operation Distribution (indicates relation types):")
    for op, count in relation_types.most_common():
        print(f"   {op}: {count}")
    
    print(f"\n🔗 Complex Relations (multi-step):")
    print(f"   Samples with >2 operations: {len(complex_relations)}/{len(samples)}")
    if complex_relations:
        print(f"\n   Examples:")
        for item in complex_relations[:5]:
            print(f"   - Sample {item['idx']}: {item['operations']} ops = {item['types']}")
    
    return relation_types, complex_relations

def analyze_table_topology(samples):
    """Đánh giá chất lượng table topology"""
    print("\n" + "="*80)
    print("TABLE TOPOLOGY ANALYSIS")
    print("="*80)
    
    table_stats = {
        'has_table': 0,
        'table_sizes': [],
        'header_types': Counter(),
        'multi_row_tables': 0,
        'table_operations': Counter()
    }
    
    for idx, sample in enumerate(samples):
        table = sample.get('table', [])
        program = sample.get('qa', {}).get('program', '')
        
        if not table:
            continue
        
        table_stats['has_table'] += 1
        rows = len(table)
        cols = len(table[0]) if table else 0
        table_stats['table_sizes'].append((rows, cols))
        
        if rows > 2:
            table_stats['multi_row_tables'] += 1
        
        # Analyze header
        if table:
            first_row = table[0]
            has_numbers = any(re.search(r'\d', str(cell)) for cell in first_row)
            table_stats['header_types']['numeric' if has_numbers else 'text'] += 1
        
        # Table operations in program
        table_ops = re.findall(r'(table_sum|table_average|table_max|table_min)', program)
        table_stats['table_operations'].update(table_ops)
    
    print(f"\n📊 Table Structure:")
    print(f"   Samples with tables: {table_stats['has_table']}/{len(samples)}")
    print(f"   Multi-row tables (>2 rows): {table_stats['multi_row_tables']}")
    
    if table_stats['table_sizes']:
        avg_rows = sum(r for r, c in table_stats['table_sizes']) / len(table_stats['table_sizes'])
        avg_cols = sum(c for r, c in table_stats['table_sizes']) / len(table_stats['table_sizes'])
        print(f"   Average size: {avg_rows:.1f} rows × {avg_cols:.1f} cols")
    
    print(f"\n📋 Header Types:")
    for htype, count in table_stats['header_types'].items():
        print(f"   {htype}: {count}")
    
    print(f"\n🔢 Table Operations:")
    for op, count in table_stats['table_operations'].items():
        print(f"   {op}: {count}")
    
    return table_stats

def analyze_critical_samples(samples):
    """Phân tích chi tiết các mẫu quan trọng"""
    print("\n" + "="*80)
    print("CRITICAL SAMPLES DETAILED ANALYSIS")
    print("="*80)
    
    critical_samples = []
    
    for idx, sample in enumerate(samples):
        analysis = analyze_single_sample(sample, idx)
        
        # Critical criteria
        is_critical = False
        criticality_score = 0
        
        if 'INSUFFICIENT_NUMBERS' in str(analysis['issues']):
            criticality_score += 3
            is_critical = True
        
        if 'TEMPORAL_CONTEXT_NEEDED' in analysis['issues']:
            criticality_score += 2
        
        if 'COMPARISON_OPERATION' in analysis['issues']:
            criticality_score += 2
        
        if analysis['metrics'].get('program_args_count', 0) > 3:
            criticality_score += 2
            is_critical = True
        
        if is_critical or criticality_score >= 4:
            critical_samples.append({
                'idx': idx,
                'id': analysis['sample_id'],
                'score': criticality_score,
                'analysis': analysis
            })
    
    # Sort by criticality
    critical_samples.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🎯 Found {len(critical_samples)} critical samples")
    print(f"\n📝 Top 10 Most Critical:")
    
    for i, item in enumerate(critical_samples[:10], 1):
        analysis = item['analysis']
        print(f"\n{i}. Sample {item['idx']} (Score: {item['score']})")
        print(f"   Question: {analysis['question'][:80]}...")
        print(f"   Issues: {', '.join(analysis['issues'])}")
        print(f"   Metrics:")
        print(f"     - Program args needed: {analysis['metrics'].get('program_args_count', 0)}")
        print(f"     - Total numbers available: {analysis['metrics'].get('total_unique_numbers', 0)}")
        print(f"     - Table: {analysis['metrics'].get('table_rows', 0)}×{analysis['metrics'].get('table_cols', 0)}")
    
    return critical_samples

def generate_improvement_recommendations(all_analyses):
    """Tạo đề xuất cải tiến dựa trên phân tích"""
    print("\n" + "="*80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Check entity extraction issues
    entity_stats = all_analyses.get('entity_stats', {})
    if entity_stats.get('missing_entities'):
        recommendations.append({
            'priority': 'HIGH',
            'area': 'Entity Extraction',
            'issue': f"{len(entity_stats['missing_entities'])} samples have insufficient numbers",
            'recommendation': 'Improve number extraction to handle: commas, percentages, units (million/billion), negative numbers, ranges'
        })
    
    # Check temporal issues
    if entity_stats.get('temporal_entities'):
        recommendations.append({
            'priority': 'HIGH',
            'area': 'Temporal Context',
            'issue': f"{len(entity_stats['temporal_entities'])} samples need temporal context",
            'recommendation': 'Add temporal entity extraction: years, quarters, fiscal periods. Link numbers to time periods.'
        })
    
    # Check table operations
    table_stats = all_analyses.get('table_stats', {})
    if table_stats.get('table_operations'):
        recommendations.append({
            'priority': 'MEDIUM',
            'area': 'Table Topology',
            'issue': 'Table operations detected but topology may be incomplete',
            'recommendation': 'Ensure row-column-cell relationships are explicit. Add column headers as semantic labels.'
        })
    
    # Check complex relations
    relation_stats = all_analyses.get('relation_stats', {})
    if relation_stats and relation_stats[1]:  # complex_relations list
        recommendations.append({
            'priority': 'MEDIUM',
            'area': 'Relation Extraction',
            'issue': f"{len(relation_stats[1])} samples have multi-step operations",
            'recommendation': 'Create intermediate result nodes. Chain operations explicitly in KG.'
        })
    
    print("\n🎯 Priority Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['area']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   → {rec['recommendation']}\n")
    
    return recommendations

def main():
    print("="*80)
    print("KNOWLEDGE GRAPH DEEP ANALYSIS")
    print("Phân tích sâu về chất lượng đồ thị tri thức")
    print("="*80)
    
    # Load data
    print("\n📂 Loading dataset...")
    samples = load_dataset(max_samples=50)
    print(f"✓ Loaded {len(samples)} samples")
    
    all_analyses = {}
    
    # 1. Entity extraction analysis
    entity_stats = analyze_entity_extraction_quality(samples)
    all_analyses['entity_stats'] = entity_stats
    
    # 2. Relation extraction analysis
    relation_stats = analyze_relation_quality(samples)
    all_analyses['relation_stats'] = relation_stats
    
    # 3. Table topology analysis
    table_stats = analyze_table_topology(samples)
    all_analyses['table_stats'] = table_stats
    
    # 4. Critical samples analysis
    critical_samples = analyze_critical_samples(samples)
    all_analyses['critical_samples'] = critical_samples
    
    # 5. Generate recommendations
    recommendations = generate_improvement_recommendations(all_analyses)
    all_analyses['recommendations'] = recommendations
    
    # Save detailed report
    output_dir = Path(__file__).parent / "output" / "kg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "deep_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        # Convert to JSON-serializable format
        json_data = {
            'entity_stats': {
                'total_samples': entity_stats['total_samples'],
                'avg_coverage': sum(entity_stats['numbers_extracted']) / len(entity_stats['numbers_extracted']) if entity_stats['numbers_extracted'] else 0,
                'temporal_count': len(entity_stats['temporal_entities']),
                'missing_count': len(entity_stats['missing_entities'])
            },
            'relation_stats': {
                'operation_counts': dict(relation_stats[0]),
                'complex_relations_count': len(relation_stats[1])
            },
            'table_stats': table_stats,
            'critical_samples': [
                {
                    'idx': c['idx'],
                    'id': c['id'],
                    'score': c['score'],
                    'question': c['analysis']['question'][:100],
                    'issues': c['analysis']['issues']
                }
                for c in critical_samples[:20]
            ],
            'recommendations': recommendations
        }
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
