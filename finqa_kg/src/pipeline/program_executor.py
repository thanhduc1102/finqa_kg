"""
Program Executor
Execute program với provenance tracking:
- Parse program string
- Execute operations step by step
- Track which KG nodes were used
- Generate computation graph
"""

import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class ComputationStep:
    """Một bước tính toán"""
    step_num: int
    operation: str
    arg1: Any
    arg2: Optional[Any]
    result: float
    source_nodes: List[str]  # KG nodes involved

@dataclass
class ExecutionResult:
    """Kết quả execution"""
    final_answer: float
    steps: List[ComputationStep]
    computation_graph: nx.DiGraph
    is_correct: bool
    ground_truth: Optional[float]
    error: Optional[str] = None

class ProgramExecutor:
    """
    Execute FinQA program với full provenance tracking
    
    Program format examples:
    - "divide(637, 5)"
    - "divide(subtract(115, 100), 100)"
    - "multiply(divide(subtract(#0, #1), #1), const_100)"
    """
    
    def __init__(self):
        """Initialize executor"""
        self.step_counter = 0
    
    def execute(self,
               program: str,
               placeholders: Dict[str, Any],
               ground_truth: Optional[float] = None) -> ExecutionResult:
        """
        Main execution function
        
        Args:
            program: Program string
            placeholders: Mapping #0 -> value/node_id
            ground_truth: Expected answer (optional)
            
        Returns:
            ExecutionResult
        """
        try:
            # Reset counter
            self.step_counter = 0
            
            # Handle empty program
            if not program or program.strip() == '':
                return ExecutionResult(
                    final_answer=0.0,
                    steps=[],
                    computation_graph=nx.DiGraph(),
                    is_correct=False,
                    ground_truth=ground_truth,
                    error="Empty program"
                )
            
            # Parse program
            parsed = self._parse_program(program)
            
            # Execute
            result, steps, comp_graph = self._execute_parsed(parsed, placeholders)
            
            # Check correctness
            is_correct = False
            if ground_truth is not None:
                is_correct = abs(result - ground_truth) < 0.01
            
            return ExecutionResult(
                final_answer=result,
                steps=steps,
                computation_graph=comp_graph,
                is_correct=is_correct,
                ground_truth=ground_truth,
                error=None
            )
        
        except Exception as e:
            return ExecutionResult(
                final_answer=0.0,
                steps=[],
                computation_graph=nx.DiGraph(),
                is_correct=False,
                ground_truth=ground_truth,
                error=str(e)
            )
    
    def _parse_program(self, program: str) -> Dict[str, Any]:
        """
        Parse program string thành tree structure
        
        Example:
        "divide(subtract(115, 100), 100)"
        ->
        {
            'type': 'op',
            'op': 'divide',
            'args': [
                {'type': 'op', 'op': 'subtract', 'args': [...]},
                {'type': 'number', 'value': 100}
            ]
        }
        """
        def parse_expr(expr: str) -> Dict[str, Any]:
            expr = expr.strip()
            
            # Check if it's a function call
            if '(' in expr and expr.endswith(')'):
                # Extract operator name
                op_end = expr.index('(')
                op_name = expr[:op_end]
                
                # Extract arguments
                args_str = expr[op_end+1:-1]
                
                # Split arguments by comma (handle nested)
                args = []
                depth = 0
                current = ""
                
                for char in args_str:
                    if char == ',' and depth == 0:
                        if current.strip():
                            args.append(parse_expr(current))
                        current = ""
                    else:
                        if char == '(':
                            depth += 1
                        elif char == ')':
                            depth -= 1
                        current += char
                
                if current.strip():
                    args.append(parse_expr(current))
                
                return {
                    'type': 'op',
                    'op': op_name,
                    'args': args
                }
            
            # Check if it's a constant
            elif expr.startswith('const_'):
                value = float(expr.replace('const_', ''))
                return {'type': 'const', 'value': value}
            
            # Check if it's a placeholder
            elif expr.startswith('#'):
                return {'type': 'placeholder', 'id': expr}
            
            # Check if it's a number
            elif expr.replace('.', '').replace('-', '').isdigit():
                return {'type': 'number', 'value': float(expr)}
            
            else:
                # Unknown, treat as variable
                return {'type': 'variable', 'name': expr}
        
        return parse_expr(program)
    
    def _execute_parsed(self,
                       parsed: Dict[str, Any],
                       placeholders: Dict[str, Any]) -> Tuple[float, List[ComputationStep], nx.DiGraph]:
        """
        Execute parsed program tree
        """
        steps = []
        comp_graph = nx.DiGraph()
        
        def execute_node(node: Dict[str, Any]) -> Tuple[float, List[str]]:
            """
            Execute một node, return (result, source_nodes)
            """
            node_type = node['type']
            
            if node_type == 'number':
                return node['value'], []
            
            elif node_type == 'const':
                return node['value'], []
            
            elif node_type == 'placeholder':
                placeholder_id = node['id']
                if placeholder_id in placeholders:
                    placeholder_data = placeholders[placeholder_id]
                    value = placeholder_data.get('value', 0.0)
                    source_node = placeholder_data.get('node_id', '')
                    return value, [source_node] if source_node else []
                else:
                    return 0.0, []
            
            elif node_type == 'variable':
                # Try to resolve from placeholders by name
                var_name = node['name']
                for ph_id, ph_data in placeholders.items():
                    if ph_data.get('text', '').lower() == var_name.lower():
                        return ph_data.get('value', 0.0), [ph_data.get('node_id', '')]
                return 0.0, []
            
            elif node_type == 'op':
                op = node['op']
                args = node['args']
                
                # Execute arguments recursively
                arg_values = []
                all_sources = []
                
                for arg in args:
                    val, sources = execute_node(arg)
                    arg_values.append(val)
                    all_sources.extend(sources)
                
                # Execute operation
                result = self._execute_operation(op, arg_values)
                
                # Record step
                self.step_counter += 1
                step = ComputationStep(
                    step_num=self.step_counter,
                    operation=op,
                    arg1=arg_values[0] if len(arg_values) > 0 else None,
                    arg2=arg_values[1] if len(arg_values) > 1 else None,
                    result=result,
                    source_nodes=all_sources
                )
                steps.append(step)
                
                # Add to computation graph
                step_node_id = f"step_{self.step_counter}"
                comp_graph.add_node(step_node_id,
                                   type='computation',
                                   operation=op,
                                   result=result,
                                   step_num=self.step_counter)
                
                # Add edges from sources
                for source in all_sources:
                    if source:
                        comp_graph.add_edge(source, step_node_id,
                                           relation='used_in')
                
                return result, [step_node_id]
            
            return 0.0, []
        
        final_result, final_sources = execute_node(parsed)
        
        return final_result, steps, comp_graph
    
    def _execute_operation(self, op: str, args: List[float]) -> float:
        """
        Execute một operation
        """
        if op == 'add':
            return sum(args)
        
        elif op == 'subtract':
            if len(args) >= 2:
                return args[0] - args[1]
            return args[0] if args else 0.0
        
        elif op == 'multiply':
            result = 1.0
            for arg in args:
                result *= arg
            return result
        
        elif op == 'divide':
            if len(args) >= 2 and args[1] != 0:
                return args[0] / args[1]
            return 0.0
        
        elif op == 'exp' or op == 'power':
            if len(args) >= 2:
                return args[0] ** args[1]
            return 0.0
        
        elif op == 'greater':
            if len(args) >= 2:
                return 1.0 if args[0] > args[1] else 0.0
            return 0.0
        
        # NEW: Additional boolean operations
        elif op == 'less':
            if len(args) >= 2:
                return 1.0 if args[0] < args[1] else 0.0
            return 0.0
        
        elif op == 'equal':
            if len(args) >= 2:
                return 1.0 if abs(args[0] - args[1]) < 0.01 else 0.0
            return 0.0
        
        elif op == 'not_equal':
            if len(args) >= 2:
                return 1.0 if abs(args[0] - args[1]) >= 0.01 else 0.0
            return 0.0
        
        elif op == 'greater_equal':
            if len(args) >= 2:
                return 1.0 if args[0] >= args[1] else 0.0
            return 0.0
        
        elif op == 'less_equal':
            if len(args) >= 2:
                return 1.0 if args[0] <= args[1] else 0.0
            return 0.0
        
        elif op == 'table_sum':
            return sum(args)
        
        elif op == 'table_average':
            return sum(args) / len(args) if args else 0.0
        
        elif op == 'table_max':
            return max(args) if args else 0.0
        
        elif op == 'table_min':
            return min(args) if args else 0.0
        
        else:
            print(f"Warning: Unknown operation '{op}'")
            return 0.0
    
    def generate_explanation(self, result: ExecutionResult, question: str = "") -> str:
        """
        Generate human-readable explanation
        """
        if result.error:
            return f"Error during execution: {result.error}"
        
        explanation = []
        
        if question:
            explanation.append(f"Question: {question}\n")
        
        explanation.append("Computation Steps:")
        
        for step in result.steps:
            if step.arg2 is not None:
                step_str = f"{step.step_num}. {step.operation.upper()}({step.arg1}, {step.arg2}) = {step.result}"
            else:
                step_str = f"{step.step_num}. {step.operation.upper()}({step.arg1}) = {step.result}"
            
            explanation.append(step_str)
            
            # Add source information
            if step.source_nodes:
                sources_str = "   Sources: " + ", ".join([s for s in step.source_nodes if s])
                explanation.append(sources_str)
        
        explanation.append(f"\nFinal Answer: {result.final_answer}")
        
        if result.ground_truth is not None:
            explanation.append(f"Ground Truth: {result.ground_truth}")
            explanation.append(f"Correct: {'OK YES' if result.is_correct else 'X NO'}")
            if not result.is_correct:
                error_pct = abs(result.final_answer - result.ground_truth) / result.ground_truth * 100 if result.ground_truth != 0 else 100
                explanation.append(f"Error: {error_pct:.2f}%")
        
        return "\n".join(explanation)
