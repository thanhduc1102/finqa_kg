"""
Validate Knowledge Graph Coverage
Kiểm tra xem KG có chứa đầy đủ thông tin cần thiết để trả lời câu hỏi không
"""

import json
import sys
import asyncio
from pathlib import Path
from collections import defaultdict
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from finqa_kg.src.pipeline.structured_kg_builder import StructuredKGBuilder
from finqa_kg.src.pipeline.semantic_retriever import SemanticRetriever

class KGCoverageValidator:
    """Validate if KG contains all necessary information"""
    
    def __init__(self):
        self.kg_builder = StructuredKGBuilder()
        self.results = []
    
    async def build_and_validate_sample(self, sample, sample_idx):
        """Build KG for one sample and validate coverage"""
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx}: {sample.get('id', 'unknown')}")
        print(f"{'='*80}")
        
        # Extract data
        pre_text = sample.get('pre_text', [])
        post_text = sample.get('post_text', [])
        table = sample.get('table', [])
        question = sample.get('qa', {}).get('question', '')
        answer = sample.get('qa', {}).get('answer', '')
        program = sample.get('qa', {}).get('program', '')
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Program: {program}")
        
        # Build KG
        print(f"\n🔨 Building Knowledge Graph...")
        kg = await self.kg_builder.build_graph(
            pre_text=pre_text,
            post_text=post_text,
            table=table
        )
        
        # Analyze KG structure
        node_types = defaultdict(int)
        for node_id, node_data in kg.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            node_types[node_type] += 1
        
        print(f"\n📊 KG Structure:")
        print(f"   Nodes: {kg.number_of_nodes()}")
        print(f"   Edges: {kg.number_of_edges()}")
        print(f"   Node types: {dict(node_types)}")
        
        # Extract required arguments from program
        required_args = self.extract_program_arguments(program)
        print(f"\n🎯 Required Arguments from Program: {len(required_args)}")
        for arg in required_args:
            print(f"   - {arg}")
        
        # Check if KG contains required numbers
        validation_result = await self.validate_coverage(
            kg=kg,
            required_args=required_args,
            question=question,
            program=program
        )
        
        # Store result
        result = {
            'sample_idx': sample_idx,
            'sample_id': sample.get('id', 'unknown'),
            'question': question,
            'answer': answer,
            'program': program,
            'kg_stats': {
                'nodes': kg.number_of_nodes(),
                'edges': kg.number_of_edges(),
                'node_types': dict(node_types)
            },
            'required_args': required_args,
            'validation': validation_result
        }
        
        self.results.append(result)
        
        # Print validation summary
        print(f"\n✓ Validation Result:")
        print(f"   Coverage: {validation_result['coverage_percentage']:.1f}%")
        print(f"   Found: {validation_result['found_count']}/{validation_result['required_count']}")
        if validation_result['missing_args']:
            print(f"   ❌ Missing: {validation_result['missing_args']}")
        if validation_result['issues']:
            print(f"   ⚠️ Issues: {', '.join(validation_result['issues'])}")
        
        return result
    
    def extract_program_arguments(self, program):
        """Extract argument references from program"""
        if not program:
            return []
        
        # Extract const_xxx and table_xxx
        const_matches = re.findall(r'const_(-?\d+(?:\.\d+)?)', program)
        table_matches = re.findall(r'#(\d+)', program)  # Table references like #0, #1
        
        args = []
        for const in const_matches:
            try:
                args.append(('const', float(const)))
            except:
                args.append(('const', const))
        
        for table_ref in table_matches:
            args.append(('table', int(table_ref)))
        
        return args
    
    async def validate_coverage(self, kg, required_args, question, program):
        """Check if KG contains all required arguments"""
        result = {
            'required_count': len(required_args),
            'found_count': 0,
            'missing_args': [],
            'found_args': [],
            'issues': [],
            'coverage_percentage': 0.0
        }
        
        if not required_args:
            result['coverage_percentage'] = 100.0
            return result
        
        # Get all numeric entities from KG
        kg_numbers = []
        kg_table_cells = []
        
        for node_id, node_data in kg.nodes(data=True):
            node_type = node_data.get('type', '')
            
            if node_type == 'number':
                try:
                    value = float(node_data.get('value', 0))
                    kg_numbers.append({
                        'node_id': node_id,
                        'value': value,
                        'text': node_data.get('text', ''),
                        'context': node_data.get('context', '')
                    })
                except:
                    pass
            
            elif node_type == 'cell':
                kg_table_cells.append({
                    'node_id': node_id,
                    'value': node_data.get('value', ''),
                    'row': node_data.get('row', -1),
                    'col': node_data.get('col', -1)
                })
        
        print(f"\n📦 KG Content:")
        print(f"   Numbers extracted: {len(kg_numbers)}")
        print(f"   Table cells: {len(kg_table_cells)}")
        
        # Check each required argument
        for arg_type, arg_value in required_args:
            found = False
            
            if arg_type == 'const':
                # Look for this constant in KG numbers
                for kg_num in kg_numbers:
                    if abs(kg_num['value'] - float(arg_value)) < 0.01:
                        found = True
                        result['found_args'].append({
                            'type': 'const',
                            'value': arg_value,
                            'found_in': kg_num['node_id'],
                            'context': kg_num['context'][:50]
                        })
                        break
            
            elif arg_type == 'table':
                # Table reference - check if we have table structure
                if kg_table_cells:
                    found = True
                    result['found_args'].append({
                        'type': 'table',
                        'value': arg_value,
                        'found_in': f"table with {len(kg_table_cells)} cells"
                    })
            
            if found:
                result['found_count'] += 1
            else:
                result['missing_args'].append({
                    'type': arg_type,
                    'value': arg_value
                })
        
        # Calculate coverage
        if result['required_count'] > 0:
            result['coverage_percentage'] = (result['found_count'] / result['required_count']) * 100
        
        # Check for potential issues
        if len(kg_numbers) < result['required_count']:
            result['issues'].append(f"Few numbers: {len(kg_numbers)} < {result['required_count']}")
        
        if not kg_table_cells and 'table' in program:
            result['issues'].append("No table cells found but program uses table")
        
        return result
    
    def generate_summary_report(self):
        """Generate summary statistics"""
        print(f"\n{'='*80}")
        print(f"OVERALL COVERAGE SUMMARY")
        print(f"{'='*80}")
        
        if not self.results:
            print("No results to summarize")
            return
        
        # Calculate statistics
        total_samples = len(self.results)
        perfect_coverage = sum(1 for r in self.results if r['validation']['coverage_percentage'] >= 100)
        good_coverage = sum(1 for r in self.results if 80 <= r['validation']['coverage_percentage'] < 100)
        poor_coverage = sum(1 for r in self.results if r['validation']['coverage_percentage'] < 80)
        
        avg_coverage = sum(r['validation']['coverage_percentage'] for r in self.results) / total_samples
        avg_nodes = sum(r['kg_stats']['nodes'] for r in self.results) / total_samples
        avg_edges = sum(r['kg_stats']['edges'] for r in self.results) / total_samples
        
        print(f"\n📊 Coverage Statistics:")
        print(f"   Total samples: {total_samples}")
        print(f"   Perfect coverage (100%): {perfect_coverage} ({perfect_coverage/total_samples*100:.1f}%)")
        print(f"   Good coverage (80-99%): {good_coverage} ({good_coverage/total_samples*100:.1f}%)")
        print(f"   Poor coverage (<80%): {poor_coverage} ({poor_coverage/total_samples*100:.1f}%)")
        print(f"   Average coverage: {avg_coverage:.1f}%")
        
        print(f"\n🔍 KG Size Statistics:")
        print(f"   Average nodes: {avg_nodes:.1f}")
        print(f"   Average edges: {avg_edges:.1f}")
        
        # Most common issues
        all_issues = []
        for r in self.results:
            all_issues.extend(r['validation']['issues'])
        
        if all_issues:
            from collections import Counter
            issue_counts = Counter(all_issues)
            print(f"\n⚠️ Most Common Issues:")
            for issue, count in issue_counts.most_common(5):
                print(f"   {issue}: {count}")
        
        # Worst performers
        worst = sorted(self.results, key=lambda r: r['validation']['coverage_percentage'])[:5]
        print(f"\n❌ Samples with Lowest Coverage:")
        for i, r in enumerate(worst, 1):
            print(f"\n   {i}. Sample {r['sample_idx']} - {r['validation']['coverage_percentage']:.1f}%")
            print(f"      Question: {r['question'][:60]}...")
            print(f"      Missing: {len(r['validation']['missing_args'])} args")
            if r['validation']['missing_args']:
                for missing in r['validation']['missing_args'][:3]:
                    print(f"         - {missing['type']}: {missing['value']}")
        
        return {
            'total_samples': total_samples,
            'perfect_coverage': perfect_coverage,
            'good_coverage': good_coverage,
            'poor_coverage': poor_coverage,
            'avg_coverage': avg_coverage,
            'avg_nodes': avg_nodes,
            'avg_edges': avg_edges
        }

async def main():
    print("="*80)
    print("KNOWLEDGE GRAPH COVERAGE VALIDATION")
    print("Kiểm tra xem KG có chứa đầy đủ thông tin cần thiết")
    print("="*80)
    
    # Load dataset
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Test with first 10 samples
    num_samples = 10
    samples = data[:num_samples]
    
    print(f"\n📂 Testing with {num_samples} samples")
    
    validator = KGCoverageValidator()
    
    # Validate each sample
    for idx, sample in enumerate(samples):
        try:
            await validator.build_and_validate_sample(sample, idx)
        except Exception as e:
            print(f"\n❌ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary
    summary = validator.generate_summary_report()
    
    # Save results
    output_dir = Path(__file__).parent / "output" / "kg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "coverage_validation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'results': validator.results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
