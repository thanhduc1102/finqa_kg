"""
Program Synthesizer
Tự động sinh program từ:
- Question analysis
- KG retrieval
- Argument ordering logic
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re

from .question_analyzer import QuestionAnalysis

@dataclass
class ProgramSynthesisResult:
    """Kết quả synthesis"""
    program: str  # Program string như "divide(subtract(#0, #1), #1)"
    placeholders: Dict[str, Any]  # Mapping #0 -> actual value/entity
    explanation: str  # Giải thích logic
    confidence: float  # Confidence score

class ProgramSynthesizer:
    """
    Synthesize program từ question analysis và KG
    
    Process:
    1. Dựa vào question type để chọn template
    2. Query KG để tìm entities/numbers cần thiết
    3. Resolve argument order
    4. Generate program string
    """
    
    def __init__(self):
        """Initialize với program templates"""
        self.program_templates = self._load_program_templates()
    
    def _load_program_templates(self) -> Dict[str, Dict]:
        """
        Load templates cho từng question type
        """
        return {
            'percentage_change': {
                'template': 'multiply(divide(subtract(#new, #old), #old), const_100)',
                'description': 'Percentage change = ((new - old) / old) * 100',
                'required_args': ['new', 'old'],
                'arg_types': ['number', 'number']
            },
            'ratio': {
                'template': 'divide(#numerator, #denominator)',
                'description': 'Ratio = numerator / denominator',
                'required_args': ['numerator', 'denominator'],
                'arg_types': ['number', 'number']
            },
            'average': {
                'template': 'divide(#sum, #count)',
                'description': 'Average = sum / count',
                'required_args': ['sum', 'count'],
                'arg_types': ['number', 'number']
            },
            'sum': {
                'template': 'add(#values)',
                'description': 'Sum of all values',
                'required_args': ['values'],
                'arg_types': ['list']
            },
            'difference': {
                'template': 'subtract(#larger, #smaller)',
                'description': 'Difference = larger - smaller',
                'required_args': ['larger', 'smaller'],
                'arg_types': ['number', 'number']
            },
            'product': {
                'template': 'multiply(#factor1, #factor2)',
                'description': 'Product = factor1 * factor2',
                'required_args': ['factor1', 'factor2'],
                'arg_types': ['number', 'number']
            },
            'percentage_of': {
                'template': 'multiply(divide(#part, #whole), const_100)',
                'description': 'Percentage = (part / whole) * 100',
                'required_args': ['part', 'whole'],
                'arg_types': ['number', 'number']
            },
            'direct_lookup': {
                'template': '#value',
                'description': 'Direct extraction from text/table',
                'required_args': ['value'],
                'arg_types': ['number']
            },
            'conversion': {
                'template': 'divide(#value, #conversion_factor)',
                'description': 'Convert value using factor',
                'required_args': ['value', 'conversion_factor'],
                'arg_types': ['number', 'number']
            }
        }
    
    def synthesize(self, 
                  question_analysis: QuestionAnalysis,
                  kg: nx.MultiDiGraph,
                  entity_index: Dict) -> ProgramSynthesisResult:
        """
        Main synthesis function
        
        Args:
            question_analysis: Phân tích từ QuestionAnalyzer
            kg: Knowledge Graph
            entity_index: Index của entities trong KG
            
        Returns:
            ProgramSynthesisResult
        """
        q_type = question_analysis.question_type
        
        # 1. Get template
        if q_type not in self.program_templates:
            return self._fallback_synthesis(question_analysis, kg, entity_index)
        
        template_info = self.program_templates[q_type]
        template = template_info['template']
        required_args = template_info['required_args']
        
        # 2. Retrieve arguments từ KG
        arguments = self._retrieve_arguments(
            question_analysis,
            kg,
            entity_index,
            required_args
        )
        
        if not arguments:
            return self._fallback_synthesis(question_analysis, kg, entity_index)
        
        # 3. Generate program string
        program, placeholders = self._generate_program_string(
            template,
            arguments,
            question_analysis
        )
        
        # 4. Generate explanation
        explanation = self._generate_explanation(
            q_type,
            template_info['description'],
            arguments,
            placeholders
        )
        
        confidence = self._calculate_confidence(arguments, required_args)
        
        return ProgramSynthesisResult(
            program=program,
            placeholders=placeholders,
            explanation=explanation,
            confidence=confidence
        )
    
    def _retrieve_arguments(self,
                           qa: QuestionAnalysis,
                           kg: nx.MultiDiGraph,
                           entity_index: Dict,
                           required_args: List[str]) -> Dict[str, Any]:
        """
        Retrieve arguments từ KG dựa vào:
        - Entities mentioned in question
        - Temporal information
        - Numbers in question
        - Argument order logic
        - Context matching
        """
        arguments = {}
        
        # Special handling for direct_lookup: tìm trong context
        if qa.question_type == 'direct_lookup':
            # Extract key entities từ question (không phải year)
            key_entities = [e for e in qa.entities_mentioned 
                          if not e.isdigit() and len(e) > 3]
            
            # Search trong KG cho nodes có context chứa các entities này
            best_candidates = []
            for node_id, node_data in kg.nodes(data=True):
                if node_data.get('type') == 'entity' and node_data.get('value') is not None:
                    context = node_data.get('context', '').lower()
                    
                    # Count how many key entities appear in context
                    match_score = sum(1 for entity in key_entities 
                                    if entity.lower() in context)
                    
                    # Also check for temporal match
                    for temporal in qa.temporal_entities:
                        if temporal in context:
                            match_score += 2  # Bonus for temporal match
                    
                    if match_score > 0:
                        best_candidates.append({
                            'value': node_data['value'],
                            'text': node_data.get('text', str(node_data['value'])),
                            'node_id': node_id,
                            'context': node_data.get('context', ''),
                            'score': match_score
                        })
            
            # Sort by score and take best
            if best_candidates:
                best_candidates.sort(key=lambda x: x['score'], reverse=True)
                if 'value' in required_args:
                    arguments['value'] = best_candidates[0]
                elif required_args:
                    arguments[required_args[0]] = best_candidates[0]
            
            return arguments
        
        # 1. Search by entities mentioned
        for entity_text in qa.entities_mentioned:
            # Search in entity_index
            matches = entity_index['by_text'].get(entity_text.lower(), [])
            if matches:
                for match in matches:
                    node_data = match['data']
                    if node_data.get('value') is not None:
                        # Map to argument name
                        arg_name = self._map_to_argument_name(
                            entity_text, required_args, qa
                        )
                        if arg_name:
                            if arg_name not in arguments:
                                arguments[arg_name] = []
                            arguments[arg_name].append({
                                'value': node_data['value'],
                                'text': entity_text,
                                'node_id': match['id'],
                                'context': node_data.get('context', '')
                            })
        
        # 2. Search by numbers mentioned in question
        for num in qa.numbers_mentioned:
            matches = entity_index['by_value'].get(num, [])
            if matches:
                # These are explicitly mentioned, treat with priority
                for match in matches[:2]:  # Limit to 2
                    node_data = match['data']
                    # Add to all possible args nếu chưa có
                    for arg_name in required_args:
                        if arg_name not in arguments:
                            arguments[arg_name] = []
                        arguments[arg_name].append({
                            'value': num,
                            'text': str(num),
                            'node_id': match['id'],
                            'context': node_data.get('context', ''),
                            'priority': 'high'
                        })
        
        # 3. Use temporal information để resolve order
        if qa.temporal_entities and len(qa.temporal_entities) >= 2:
            # Sort temporal
            sorted_temporal = sorted(qa.temporal_entities)
            
            # Find corresponding values
            for idx, temporal in enumerate(sorted_temporal):
                # Search for numbers associated with this temporal entity
                for node_id, node_data in kg.nodes(data=True):
                    if node_data.get('type') == 'entity':
                        context = node_data.get('context', '')
                        if temporal in context and node_data.get('value') is not None:
                            # Map based on order
                            if 'old' in required_args and 'new' in required_args:
                                arg_name = 'old' if idx == 0 else 'new'
                            elif 'smaller' in required_args and 'larger' in required_args:
                                arg_name = 'smaller' if idx == 0 else 'larger'
                            else:
                                continue
                            
                            if arg_name not in arguments:
                                arguments[arg_name] = []
                            arguments[arg_name].append({
                                'value': node_data['value'],
                                'text': f"{temporal}={node_data['value']}",
                                'node_id': node_id,
                                'context': context,
                                'temporal': temporal
                            })
        
        # 4. If still missing args, search all numerical entities
        for arg_name in required_args:
            if arg_name not in arguments or not arguments[arg_name]:
                # Get all numerical entities
                money_entities = entity_index['by_label'].get('MONEY', [])
                for match in money_entities[:3]:  # Limit
                    node_data = match['data']
                    if node_data.get('value') is not None:
                        if arg_name not in arguments:
                            arguments[arg_name] = []
                        arguments[arg_name].append({
                            'value': node_data['value'],
                            'text': node_data.get('text', str(node_data['value'])),
                            'node_id': match['id'],
                            'context': node_data.get('context', '')
                        })
        
        # 5. Deduplicate and select best
        for arg_name in arguments:
            if arguments[arg_name]:
                # Prioritize by: priority tag > explicit mention > context relevance
                sorted_candidates = sorted(
                    arguments[arg_name],
                    key=lambda x: (x.get('priority', '') == 'high', 
                                  x.get('temporal') is not None),
                    reverse=True
                )
                arguments[arg_name] = sorted_candidates[0]  # Take best
        
        return arguments
    
    def _map_to_argument_name(self, entity_text: str, 
                             required_args: List[str],
                             qa: QuestionAnalysis) -> Optional[str]:
        """
        Map entity text to argument name
        """
        # Simple heuristic mapping
        entity_lower = entity_text.lower()
        
        # Temporal-based mapping
        if qa.temporal_entities:
            sorted_temp = sorted(qa.temporal_entities)
            if entity_text in sorted_temp:
                idx = sorted_temp.index(entity_text)
                if 'old' in required_args and 'new' in required_args:
                    return 'old' if idx == 0 else 'new'
        
        # Keyword-based mapping
        if 'revenue' in entity_lower or 'income' in entity_lower:
            if 'numerator' in required_args:
                return 'numerator'
        
        # Default: return first available
        for arg in required_args:
            return arg
        
        return None
    
    def _generate_program_string(self,
                                template: str,
                                arguments: Dict[str, Any],
                                qa: QuestionAnalysis) -> Tuple[str, Dict]:
        """
        Generate program string từ template và arguments
        """
        program = template
        placeholders = {}
        placeholder_idx = 0
        
        # Replace placeholders trong template
        for arg_name, arg_data in arguments.items():
            placeholder_pattern = f"#{arg_name}"
            if placeholder_pattern in program:
                placeholder = f"#{placeholder_idx}"
                program = program.replace(placeholder_pattern, str(arg_data['value']))
                placeholders[placeholder] = {
                    'value': arg_data['value'],
                    'text': arg_data['text'],
                    'node_id': arg_data.get('node_id', ''),
                    'context': arg_data.get('context', '')
                }
                placeholder_idx += 1
        
        return program, placeholders
    
    def _generate_explanation(self,
                             q_type: str,
                             formula_desc: str,
                             arguments: Dict[str, Any],
                             placeholders: Dict) -> str:
        """Generate human-readable explanation"""
        explanation = f"Question Type: {q_type}\n"
        explanation += f"Formula: {formula_desc}\n\n"
        explanation += "Arguments Retrieved from KG:\n"
        
        for arg_name, arg_data in arguments.items():
            explanation += f"  {arg_name}: {arg_data['value']} "
            explanation += f"(from: {arg_data['context'][:100]}...)\n"
        
        return explanation
    
    def _calculate_confidence(self, arguments: Dict, required_args: List[str]) -> float:
        """Calculate confidence score"""
        found_args = len(arguments)
        total_args = len(required_args)
        
        if total_args == 0:
            return 0.0
        
        return found_args / total_args
    
    def _fallback_synthesis(self,
                           qa: QuestionAnalysis,
                           kg: nx.MultiDiGraph,
                           entity_index: Dict) -> ProgramSynthesisResult:
        """
        Fallback khi không match template nào
        """
        # Simple heuristic: nếu có 2 numbers, thử divide
        if len(qa.numbers_mentioned) >= 2:
            return ProgramSynthesisResult(
                program=f"divide({qa.numbers_mentioned[0]}, {qa.numbers_mentioned[1]})",
                placeholders={
                    '#0': {'value': qa.numbers_mentioned[0]},
                    '#1': {'value': qa.numbers_mentioned[1]}
                },
                explanation="Fallback: Using simple division",
                confidence=0.3
            )
        
        return ProgramSynthesisResult(
            program="",
            placeholders={},
            explanation="Could not synthesize program",
            confidence=0.0
        )
