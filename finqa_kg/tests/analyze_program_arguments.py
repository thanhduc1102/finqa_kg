"""
Program Argument Analysis
Phân tích chi tiết về cách program sử dụng arguments và làm thế nào để retrieve chúng từ KG
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class ProgramArgumentAnalyzer:
    """Analyze how programs use arguments"""
    
    def __init__(self):
        self.results = []
    
    def analyze_program(self, program, sample_idx):
        """
        Phân tích program để hiểu:
        1. Operations được sử dụng
        2. Argument types (const, table ref, intermediate result)
        3. Data flow giữa các operations
        """
        if not program:
            return None
        
        # Parse program into operations
        operations = self._parse_program(program)
        
        # Analyze argument types
        arg_analysis = self._analyze_arguments(operations)
        
        # Build execution graph
        exec_graph = self._build_execution_graph(operations)
        
        result = {
            'sample_idx': sample_idx,
            'program': program,
            'operations': operations,
            'argument_analysis': arg_analysis,
            'execution_graph': exec_graph
        }
        
        self.results.append(result)
        return result
    
    def _parse_program(self, program):
        """
        Parse program string into structured operations
        
        Example:
            "divide(100, 100), divide(3.8, #0)"
            →
            [
                {'op': 'divide', 'args': ['100', '100'], 'output_ref': '#0'},
                {'op': 'divide', 'args': ['3.8', '#0'], 'output_ref': '#1'}
            ]
        """
        operations = []
        
        # Split by comma outside of parentheses
        op_strings = self._split_program(program)
        
        for idx, op_str in enumerate(op_strings):
            op_str = op_str.strip()
            
            # Parse: operation_name(arg1, arg2, ...)
            match = re.match(r'(\w+)\((.*)\)', op_str)
            if match:
                op_name = match.group(1)
                args_str = match.group(2)
                
                # Split arguments carefully (handle nested parens)
                args = self._split_arguments(args_str)
                
                operations.append({
                    'index': idx,
                    'op': op_name,
                    'args': args,
                    'output_ref': f'#{idx}',  # Output is referenced as #idx
                    'raw': op_str
                })
        
        return operations
    
    def _split_program(self, program):
        """Split program by commas, respecting parentheses"""
        parts = []
        current = []
        depth = 0
        
        for char in program:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
                continue
            
            current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def _split_arguments(self, args_str):
        """Split arguments by commas, respecting parentheses"""
        if not args_str.strip():
            return []
        
        args = []
        current = []
        depth = 0
        
        for char in args_str:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
                continue
            
            current.append(char)
        
        if current:
            args.append(''.join(current).strip())
        
        return args
    
    def _analyze_arguments(self, operations):
        """Phân tích các loại argument được sử dụng"""
        analysis = {
            'total_args': 0,
            'const_args': [],
            'table_refs': [],
            'intermediate_refs': [],
            'other_args': []
        }
        
        for op in operations:
            for arg in op['args']:
                analysis['total_args'] += 1
                
                if arg.startswith('const_'):
                    # Constant value
                    value = arg.replace('const_', '')
                    analysis['const_args'].append({
                        'arg': arg,
                        'value': value,
                        'op_index': op['index'],
                        'op': op['op']
                    })
                
                elif arg.startswith('#'):
                    # Reference to previous operation result
                    ref_idx = int(arg[1:])
                    analysis['intermediate_refs'].append({
                        'arg': arg,
                        'ref_index': ref_idx,
                        'op_index': op['index'],
                        'op': op['op']
                    })
                
                elif arg.startswith('table_'):
                    # Table operation
                    analysis['table_refs'].append({
                        'arg': arg,
                        'op_index': op['index'],
                        'op': op['op']
                    })
                
                else:
                    # Direct value (number, percentage, etc.)
                    try:
                        # Try to parse as number
                        value = arg.replace('%', '').replace('$', '').replace(',', '')
                        float(value)
                        analysis['const_args'].append({
                            'arg': arg,
                            'value': value,
                            'op_index': op['index'],
                            'op': op['op']
                        })
                    except:
                        analysis['other_args'].append({
                            'arg': arg,
                            'op_index': op['index'],
                            'op': op['op']
                        })
        
        return analysis
    
    def _build_execution_graph(self, operations):
        """
        Build dependency graph showing data flow
        
        Returns dict mapping operation index to its dependencies
        """
        graph = {}
        
        for op in operations:
            deps = []
            for arg in op['args']:
                if arg.startswith('#'):
                    ref_idx = int(arg[1:])
                    deps.append(ref_idx)
            
            graph[op['index']] = {
                'op': op['op'],
                'depends_on': deps,
                'has_dependencies': len(deps) > 0
            }
        
        return graph
    
    def print_analysis(self, result):
        """Print detailed analysis for one program"""
        print(f"\n{'='*80}")
        print(f"SAMPLE {result['sample_idx']} - PROGRAM ANALYSIS")
        print(f"{'='*80}")
        
        print(f"\nProgram: {result['program']}")
        
        # Operations
        print(f"\n📋 Operations ({len(result['operations'])}):")
        for op in result['operations']:
            print(f"   #{op['index']}: {op['op']}({', '.join(op['args'])})")
        
        # Arguments
        analysis = result['argument_analysis']
        print(f"\n🎯 Argument Analysis:")
        print(f"   Total arguments: {analysis['total_args']}")
        print(f"   - Constants: {len(analysis['const_args'])}")
        print(f"   - Intermediate refs: {len(analysis['intermediate_refs'])}")
        print(f"   - Table refs: {len(analysis['table_refs'])}")
        print(f"   - Other: {len(analysis['other_args'])}")
        
        if analysis['const_args']:
            print(f"\n   📊 Constants need to be retrieved from KG:")
            for const in analysis['const_args']:
                print(f"      - {const['arg']} (value: {const['value']}) in operation #{const['op_index']} ({const['op']})")
        
        if analysis['intermediate_refs']:
            print(f"\n   🔗 Intermediate results (computed from previous ops):")
            for ref in analysis['intermediate_refs']:
                print(f"      - {ref['arg']} (from operation #{ref['ref_index']}) used in #{ref['op_index']} ({ref['op']})")
        
        if analysis['table_refs']:
            print(f"\n   📋 Table operations:")
            for ref in analysis['table_refs']:
                print(f"      - {ref['arg']} in operation #{ref['op_index']} ({ref['op']})")
        
        # Execution graph
        exec_graph = result['execution_graph']
        print(f"\n🔄 Execution Dependencies:")
        for idx, info in sorted(exec_graph.items()):
            if info['has_dependencies']:
                dep_str = ', '.join(f"#{d}" for d in info['depends_on'])
                print(f"   #{idx} ({info['op']}) depends on: {dep_str}")
            else:
                print(f"   #{idx} ({info['op']}) - independent (can execute first)")
    
    def generate_summary_statistics(self):
        """Generate summary statistics across all programs"""
        print(f"\n{'='*80}")
        print(f"OVERALL PROGRAM STATISTICS")
        print(f"{'='*80}")
        
        if not self.results:
            print("No results to analyze")
            return
        
        total_programs = len(self.results)
        
        # Operation statistics
        all_operations = Counter()
        for result in self.results:
            for op in result['operations']:
                all_operations[op['op']] += 1
        
        print(f"\n📊 Operation Frequency ({total_programs} programs):")
        for op, count in all_operations.most_common():
            print(f"   {op}: {count} ({count/total_programs*100:.1f}% of programs)")
        
        # Argument statistics
        total_const_args = sum(len(r['argument_analysis']['const_args']) for r in self.results)
        total_intermediate_refs = sum(len(r['argument_analysis']['intermediate_refs']) for r in self.results)
        total_table_refs = sum(len(r['argument_analysis']['table_refs']) for r in self.results)
        
        print(f"\n🎯 Argument Type Distribution:")
        print(f"   Constants (need KG retrieval): {total_const_args}")
        print(f"   Intermediate refs (computed): {total_intermediate_refs}")
        print(f"   Table operations: {total_table_refs}")
        
        # Program complexity
        program_lengths = [len(r['operations']) for r in self.results]
        avg_length = sum(program_lengths) / len(program_lengths) if program_lengths else 0
        max_length = max(program_lengths) if program_lengths else 0
        
        print(f"\n🔧 Program Complexity:")
        print(f"   Average operations per program: {avg_length:.1f}")
        print(f"   Max operations in a program: {max_length}")
        
        multi_step = sum(1 for length in program_lengths if length > 1)
        print(f"   Multi-step programs: {multi_step}/{total_programs} ({multi_step/total_programs*100:.1f}%)")
        
        return {
            'total_programs': total_programs,
            'operation_counts': dict(all_operations),
            'total_const_args': total_const_args,
            'total_intermediate_refs': total_intermediate_refs,
            'total_table_refs': total_table_refs,
            'avg_program_length': avg_length,
            'max_program_length': max_length
        }

def main():
    print("="*80)
    print("PROGRAM ARGUMENT ANALYSIS")
    print("Phân tích chi tiết về program arguments và data flow")
    print("="*80)
    
    # Load dataset
    dataset_path = project_root / "FinQA" / "dataset" / "train.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Analyze programs
    num_samples = 50
    analyzer = ProgramArgumentAnalyzer()
    
    print(f"\n📂 Analyzing {num_samples} programs...")
    
    for idx, sample in enumerate(data[:num_samples]):
        program = sample.get('qa', {}).get('program', '')
        if program:
            analyzer.analyze_program(program, idx)
    
    # Print detailed analysis for first few
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS - FIRST 5 SAMPLES")
    print(f"{'='*80}")
    
    for result in analyzer.results[:5]:
        analyzer.print_analysis(result)
    
    # Summary statistics
    summary = analyzer.generate_summary_statistics()
    
    # Save results
    output_dir = Path(__file__).parent / "output" / "kg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "program_argument_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'detailed_results': analyzer.results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   1. Average {summary['avg_program_length']:.1f} operations per program")
    print(f"   2. {summary['total_const_args']} constant arguments need KG retrieval")
    print(f"   3. {summary['total_intermediate_refs']} intermediate results (chained operations)")
    print(f"   4. Most common operations: {', '.join(list(summary['operation_counts'].keys())[:5])}")
    
    print(f"\n🎯 IMPLICATIONS FOR KG:")
    print(f"   - KG must store all {summary['total_const_args']} constant values accurately")
    print(f"   - Need to handle {summary['total_intermediate_refs']} intermediate result references")
    print(f"   - Support for table aggregation operations ({summary['total_table_refs']} instances)")

if __name__ == "__main__":
    main()
