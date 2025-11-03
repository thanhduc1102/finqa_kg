"""
Single Sample Processing Pipeline for FinQA
Xử lý từng sample riêng lẻ: Build KG → Extract Info → Execute Program → Explain
"""

from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import numpy as np
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OperatorType(Enum):
    """Các operator trong FinQA program"""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    EXP = "exp"
    GREATER = "greater"
    TABLE_SUM = "table_sum"
    TABLE_AVERAGE = "table_average"
    TABLE_MAX = "table_max"
    TABLE_MIN = "table_min"

@dataclass
class ProgramStep:
    """Một bước trong program execution"""
    operator: str
    arg1: Any
    arg2: Optional[Any]
    result: float
    source_nodes: List[str]  # KG nodes involved
    
@dataclass
class ExecutionResult:
    """Kết quả thực thi program"""
    final_answer: float
    steps: List[ProgramStep]
    is_correct: bool
    ground_truth: Optional[float]
    explanation: str
    computation_graph: nx.DiGraph

class SingleSampleProcessor:
    """
    Xử lý một sample FinQA duy nhất:
    1. Build mini Knowledge Graph từ sample
    2. Extract entities và numbers cần thiết
    3. Parse và execute program
    4. Generate explanation với KG visualization
    """
    
    def __init__(self):
        self.kg = None
        self.entity_index = {}
        self.number_index = {}
        
    async def process_sample(self, sample: Dict[str, Any]) -> ExecutionResult:
        """
        Xử lý một sample hoàn chỉnh
        
        Args:
            sample: Dict với keys: pre_text, post_text, table, qa
            
        Returns:
            ExecutionResult với answer, steps, explanation
        """
        logger.info(f"Processing sample: {sample.get('id', 'unknown')}")
        
        # Step 1: Build Knowledge Graph cho sample này
        self.kg = await self._build_sample_kg(sample)
        logger.info(f"KG built: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
        
        # Step 2: Index entities và numbers
        self._index_entities_and_numbers()
        logger.info(f"Indexed {len(self.number_index)} numbers")
        
        # Step 3: Parse program từ QA
        qa = sample.get('qa', {})
        program_str = qa.get('program', '')
        ground_truth = qa.get('exe_ans', None)
        
        if not program_str:
            logger.warning("No program found, attempting synthesis...")
            program_str = await self._synthesize_program(qa.get('question', ''))
        
        # Step 4: Execute program với KG guidance
        result = await self._execute_program(program_str, ground_truth)
        
        # Step 5: Generate explanation
        result.explanation = self._generate_explanation(result, qa.get('question', ''))
        
        return result
    
    async def _build_sample_kg(self, sample: Dict[str, Any]) -> nx.MultiDiGraph:
        """
        Build một mini KG chỉ cho sample này
        Tối ưu: Không dùng heavy NLP models, chỉ cần structure information
        """
        kg = nx.MultiDiGraph()
        
        # Root document node
        doc_id = sample.get('id', 'doc_0')
        kg.add_node(doc_id, 
                   type='document',
                   filename=sample.get('filename', ''))
        
        # Add text nodes
        text_nodes = []
        for idx, text in enumerate(sample.get('pre_text', [])):
            node_id = f"text_{idx}"
            kg.add_node(node_id,
                       type='text',
                       content=text,
                       position='pre')
            kg.add_edge(doc_id, node_id, relation='has_pre_text')
            text_nodes.append(node_id)
            
        for idx, text in enumerate(sample.get('post_text', [])):
            node_id = f"text_{len(sample.get('pre_text', [])) + idx}"
            kg.add_node(node_id,
                       type='text',
                       content=text,
                       position='post')
            kg.add_edge(doc_id, node_id, relation='has_post_text')
            text_nodes.append(node_id)
        
        # Add table structure
        table = sample.get('table', [])
        if table:
            table_node = 'table_0'
            kg.add_node(table_node,
                       type='table',
                       headers=table[0] if len(table) > 0 else [],
                       num_rows=len(table)-1,
                       num_cols=len(table[0]) if len(table) > 0 else 0)
            kg.add_edge(doc_id, table_node, relation='has_table')
            
            # Add cells with row/col info
            headers = table[0] if len(table) > 0 else []
            for row_idx, row in enumerate(table[1:], start=1):
                for col_idx, cell_value in enumerate(row):
                    cell_id = f"cell_{row_idx}_{col_idx}"
                    
                    # Extract number if exists
                    number_val = self._extract_number(cell_value)
                    
                    kg.add_node(cell_id,
                               type='cell',
                               value=cell_value,
                               number=number_val,
                               row=row_idx,
                               col=col_idx,
                               header=headers[col_idx] if col_idx < len(headers) else '',
                               row_label=row[0] if col_idx > 0 else cell_value)
                    kg.add_edge(table_node, cell_id, relation='has_cell')
        
        # Extract numbers from all text
        for node_id in text_nodes:
            text = kg.nodes[node_id].get('content', '')
            numbers = self._extract_numbers_with_context(text)
            
            for num_idx, (num_str, num_val, context) in enumerate(numbers):
                num_node_id = f"{node_id}_num_{num_idx}"
                kg.add_node(num_node_id,
                           type='number',
                           value=num_val,
                           text=num_str,
                           context=context)
                kg.add_edge(node_id, num_node_id, relation='contains_number')
        
        # Add QA node
        qa = sample.get('qa', {})
        if qa:
            qa_node = 'qa_0'
            kg.add_node(qa_node,
                       type='qa',
                       question=qa.get('question', ''),
                       answer=qa.get('answer', ''),
                       program=qa.get('program', ''))
            kg.add_edge(doc_id, qa_node, relation='has_qa')
        
        return kg
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract single number from text"""
        if not isinstance(text, str):
            return None
            
        # Remove common symbols
        clean = text.replace('$', '').replace(',', '').replace('%', '').strip()
        
        try:
            return float(clean)
        except:
            return None
    
    def _extract_numbers_with_context(self, text: str) -> List[Tuple[str, float, str]]:
        """Extract numbers với context xung quanh"""
        results = []
        
        # Patterns: $1.2M, 15%, 1,234.56, etc.
        patterns = [
            r'\$?\s?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s?%?',
            r'([0-9]+\.?[0-9]*)\s?(?:million|billion|thousand|M|B|K)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                num_str = match.group(0)
                try:
                    num_val = float(match.group(1).replace(',', ''))
                    
                    # Get context (20 chars before and after)
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]
                    
                    results.append((num_str, num_val, context))
                except:
                    continue
        
        return results
    
    def _index_entities_and_numbers(self):
        """Build index của tất cả numbers trong KG"""
        self.number_index = {}
        
        for node_id, data in self.kg.nodes(data=True):
            if data.get('type') == 'number':
                num_val = data.get('value')
                if num_val is not None:
                    # Index by value (để lookup nhanh)
                    if num_val not in self.number_index:
                        self.number_index[num_val] = []
                    self.number_index[num_val].append({
                        'node_id': node_id,
                        'text': data.get('text', ''),
                        'context': data.get('context', '')
                    })
            
            elif data.get('type') == 'cell':
                num_val = data.get('number')
                if num_val is not None:
                    if num_val not in self.number_index:
                        self.number_index[num_val] = []
                    self.number_index[num_val].append({
                        'node_id': node_id,
                        'text': data.get('value', ''),
                        'context': f"Row {data.get('row')}, Col {data.get('col')}: {data.get('header', '')}",
                        'row': data.get('row'),
                        'col': data.get('col'),
                        'header': data.get('header', ''),
                        'row_label': data.get('row_label', '')
                    })
    
    async def _synthesize_program(self, question: str) -> str:
        """
        Synthesize program từ question nếu không có sẵn
        Simple heuristic-based approach
        """
        question_lower = question.lower()
        
        # Pattern matching cho common question types
        if 'average' in question_lower or 'mean' in question_lower:
            return "divide(table_sum, table_count)"
        elif 'percentage' in question_lower or 'percent' in question_lower:
            if 'growth' in question_lower or 'increase' in question_lower:
                return "divide(subtract(#0, #1), #1)"
        elif 'total' in question_lower or 'sum' in question_lower:
            return "table_sum"
        elif 'difference' in question_lower:
            return "subtract(#0, #1)"
        
        return ""
    
    async def _execute_program(self, program_str: str, ground_truth: Optional[float]) -> ExecutionResult:
        """
        Execute program với KG guidance
        Program format: "divide(637, const_5)" hoặc "divide(subtract(#0, #1), #1)"
        """
        logger.info(f"Executing program: {program_str}")
        
        steps = []
        computation_graph = nx.DiGraph()
        
        # Parse program
        tokens = self._parse_program(program_str)
        
        # Execute với stack-based approach
        result, steps, computation_graph = await self._execute_tokens(tokens)
        
        is_correct = False
        if ground_truth is not None:
            is_correct = abs(result - ground_truth) < 0.01
        
        return ExecutionResult(
            final_answer=result,
            steps=steps,
            is_correct=is_correct,
            ground_truth=ground_truth,
            explanation="",
            computation_graph=computation_graph
        )
    
    def _parse_program(self, program_str: str) -> List[Dict[str, Any]]:
        """
        Parse program string thành tokens
        Ví dụ: "divide(637, const_5)" → [{'op': 'divide', 'args': [637, 5]}]
        """
        tokens = []
        
        # Simple recursive parser
        def parse_expression(expr: str) -> Any:
            expr = expr.strip()
            
            # Check if it's a function call
            if '(' in expr:
                op_name = expr[:expr.index('(')]
                args_str = expr[expr.index('(')+1:expr.rindex(')')]
                
                # Split arguments (handle nested calls)
                args = []
                depth = 0
                current_arg = ""
                for char in args_str:
                    if char == ',' and depth == 0:
                        args.append(parse_expression(current_arg))
                        current_arg = ""
                    else:
                        if char == '(':
                            depth += 1
                        elif char == ')':
                            depth -= 1
                        current_arg += char
                
                if current_arg:
                    args.append(parse_expression(current_arg))
                
                return {'type': 'op', 'op': op_name, 'args': args}
            
            # Check if it's a constant
            elif expr.startswith('const_'):
                return {'type': 'const', 'value': float(expr.replace('const_', ''))}
            
            # Check if it's a number
            elif expr.replace('.', '').replace('-', '').isdigit():
                return {'type': 'number', 'value': float(expr)}
            
            # Otherwise it's a placeholder (#0, #1, etc)
            elif expr.startswith('#'):
                return {'type': 'placeholder', 'index': int(expr[1:])}
            
            else:
                return {'type': 'unknown', 'value': expr}
        
        return [parse_expression(program_str)]
    
    async def _execute_tokens(self, tokens: List[Dict[str, Any]]) -> Tuple[float, List[ProgramStep], nx.DiGraph]:
        """Execute parsed tokens"""
        steps = []
        computation_graph = nx.DiGraph()
        
        def execute_token(token: Dict[str, Any], step_idx: int = 0) -> Tuple[float, List[str]]:
            """Execute một token, return (result, source_nodes)"""
            
            if token['type'] == 'number':
                value = token['value']
                # Find corresponding KG node
                source_nodes = []
                if value in self.number_index:
                    source_nodes = [item['node_id'] for item in self.number_index[value][:1]]
                return value, source_nodes
            
            elif token['type'] == 'const':
                return token['value'], []
            
            elif token['type'] == 'placeholder':
                # For simplicity, use first available number
                idx = token['index']
                available_numbers = sorted(self.number_index.keys())
                if idx < len(available_numbers):
                    value = available_numbers[idx]
                    source_nodes = [item['node_id'] for item in self.number_index[value][:1]]
                    return value, source_nodes
                return 0.0, []
            
            elif token['type'] == 'op':
                op = token['op']
                args = token['args']
                
                # Execute arguments recursively
                arg_values = []
                all_source_nodes = []
                
                for arg in args:
                    val, sources = execute_token(arg, step_idx + len(arg_values))
                    arg_values.append(val)
                    all_source_nodes.extend(sources)
                
                # Execute operator
                result = self._execute_operator(op, arg_values)
                
                # Record step
                step = ProgramStep(
                    operator=op,
                    arg1=arg_values[0] if len(arg_values) > 0 else None,
                    arg2=arg_values[1] if len(arg_values) > 1 else None,
                    result=result,
                    source_nodes=all_source_nodes
                )
                steps.append(step)
                
                # Add to computation graph
                step_node = f"step_{len(steps)}"
                computation_graph.add_node(step_node,
                                          type='computation',
                                          operator=op,
                                          result=result)
                for source in all_source_nodes:
                    computation_graph.add_edge(source, step_node)
                
                return result, [step_node]
            
            return 0.0, []
        
        final_result, final_sources = execute_token(tokens[0])
        
        return final_result, steps, computation_graph
    
    def _execute_operator(self, op: str, args: List[float]) -> float:
        """Execute một operator"""
        if op == 'add':
            return args[0] + args[1]
        elif op == 'subtract':
            return args[0] - args[1]
        elif op == 'multiply':
            return args[0] * args[1]
        elif op == 'divide':
            return args[0] / args[1] if args[1] != 0 else 0
        elif op == 'exp':
            return args[0] ** args[1]
        elif op == 'greater':
            return 1.0 if args[0] > args[1] else 0.0
        elif op in ['table_sum', 'table_average', 'table_max', 'table_min']:
            # For table operations, need to aggregate
            return sum(args) if args else 0.0
        else:
            logger.warning(f"Unknown operator: {op}")
            return 0.0
    
    def _generate_explanation(self, result: ExecutionResult, question: str) -> str:
        """Generate human-readable explanation"""
        explanation = f"Question: {question}\n\n"
        explanation += "Computation Steps:\n"
        
        for idx, step in enumerate(result.steps, 1):
            explanation += f"{idx}. {step.operator.upper()}({step.arg1}, {step.arg2}) = {step.result}\n"
            
            # Add source information
            if step.source_nodes:
                explanation += f"   Sources: {', '.join(step.source_nodes)}\n"
                for node_id in step.source_nodes[:2]:  # Show first 2 sources
                    if node_id in self.kg.nodes:
                        node_data = self.kg.nodes[node_id]
                        if node_data.get('type') == 'cell':
                            explanation += f"   - Table: {node_data.get('row_label')} | {node_data.get('header')} = {node_data.get('value')}\n"
                        elif node_data.get('type') == 'number':
                            explanation += f"   - Text: {node_data.get('context', '')}\n"
        
        explanation += f"\nFinal Answer: {result.final_answer}\n"
        
        if result.ground_truth is not None:
            explanation += f"Ground Truth: {result.ground_truth}\n"
            explanation += f"Correct: {'✓' if result.is_correct else '✗'}\n"
        
        return explanation
    
    def visualize_computation(self, result: ExecutionResult, output_path: str = None):
        """Visualize computation graph"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Full KG
        pos1 = nx.spring_layout(self.kg)
        nx.draw(self.kg, pos1, ax=ax1, with_labels=True, node_size=500, 
                node_color='lightblue', font_size=8, arrows=True)
        ax1.set_title(f"Knowledge Graph ({self.kg.number_of_nodes()} nodes)")
        
        # Right: Computation graph
        pos2 = nx.spring_layout(result.computation_graph)
        nx.draw(result.computation_graph, pos2, ax=ax2, with_labels=True,
                node_size=800, node_color='lightgreen', font_size=10, arrows=True)
        ax2.set_title(f"Computation Flow ({len(result.steps)} steps)")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
