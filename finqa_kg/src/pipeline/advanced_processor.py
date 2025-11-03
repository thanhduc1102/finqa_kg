"""
Advanced Single Sample Processor with KG-Guided Program Synthesis
Sử dụng Knowledge Graph để guide việc synthesis program từ question
"""

from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import re
import logging
from dataclasses import dataclass

from .single_sample_processor import SingleSampleProcessor, ExecutionResult, ProgramStep

logger = logging.getLogger(__name__)

@dataclass
class QuestionIntent:
    """Phân tích intent của question"""
    intent_type: str  # 'calculation', 'comparison', 'aggregation', etc.
    entities_mentioned: List[str]
    numbers_mentioned: List[float]
    temporal_info: List[str]  # years, quarters, etc.
    operators_needed: List[str]
    
class AdvancedSampleProcessor(SingleSampleProcessor):
    """
    Extended processor với khả năng:
    1. Phân tích question để identify intent
    2. Tìm relevant entities/numbers trong KG
    3. Synthesize program từ KG evidence
    4. Execute với detailed explanation
    """
    
    def __init__(self):
        super().__init__()
        self.question_patterns = self._load_question_patterns()
    
    def _load_question_patterns(self) -> Dict[str, Dict]:
        """Load các patterns để nhận diện question type"""
        return {
            'average': {
                'keywords': ['average', 'mean', 'per'],
                'operators': ['divide', 'sum'],
                'example': 'What is the average X per Y?'
            },
            'percentage_change': {
                'keywords': ['percentage', 'percent', 'growth', 'change', 'increase', 'decrease'],
                'operators': ['subtract', 'divide', 'multiply'],
                'example': 'What is the percentage change from X to Y?'
            },
            'total': {
                'keywords': ['total', 'sum', 'combined'],
                'operators': ['add', 'sum'],
                'example': 'What is the total of X and Y?'
            },
            'difference': {
                'keywords': ['difference', 'more than', 'less than'],
                'operators': ['subtract'],
                'example': 'What is the difference between X and Y?'
            },
            'ratio': {
                'keywords': ['ratio', 'proportion', 'per'],
                'operators': ['divide'],
                'example': 'What is the ratio of X to Y?'
            },
            'compound': {
                'keywords': ['calculate', 'compute', 'find'],
                'operators': ['multiple'],
                'example': 'Calculate X based on Y and Z'
            }
        }
    
    async def process_sample(self, sample: Dict[str, Any]) -> ExecutionResult:
        """
        Enhanced processing với program synthesis
        """
        logger.info(f"Processing sample with advanced synthesis: {sample.get('id', 'unknown')}")
        
        # Step 1: Build KG
        self.kg = await self._build_sample_kg(sample)
        self._index_entities_and_numbers()
        
        # Step 2: Analyze question
        question = sample.get('qa', {}).get('question', '')
        intent = self._analyze_question(question)
        logger.info(f"Question intent: {intent.intent_type}")
        logger.info(f"Entities mentioned: {intent.entities_mentioned}")
        logger.info(f"Numbers mentioned: {intent.numbers_mentioned}")
        
        # Step 3: Find relevant KG nodes
        relevant_nodes = self._find_relevant_nodes(intent, question)
        logger.info(f"Found {len(relevant_nodes)} relevant nodes")
        
        # Step 4: Get program (use provided or synthesize)
        qa = sample.get('qa', {})
        program_str = qa.get('program', '')
        
        if not program_str:
            logger.info("Synthesizing program from KG evidence...")
            program_str = self._synthesize_program_advanced(intent, relevant_nodes)
            logger.info(f"Synthesized program: {program_str}")
        else:
            logger.info(f"Using provided program: {program_str}")
        
        # Step 5: Execute
        ground_truth = qa.get('exe_ans', None)
        result = await self._execute_program(program_str, ground_truth)
        
        # Step 6: Enhanced explanation
        result.explanation = self._generate_enhanced_explanation(
            result, question, intent, relevant_nodes
        )
        
        return result
    
    def _analyze_question(self, question: str) -> QuestionIntent:
        """
        Phân tích question để xác định intent và extract thông tin
        """
        question_lower = question.lower()
        
        # Identify intent type
        intent_type = 'unknown'
        for pattern_name, pattern_info in self.question_patterns.items():
            if any(kw in question_lower for kw in pattern_info['keywords']):
                intent_type = pattern_name
                break
        
        # Extract entities mentioned (simple NER)
        entities = self._extract_entities_from_question(question)
        
        # Extract numbers mentioned
        numbers = self._extract_numbers_from_question(question)
        
        # Extract temporal information
        temporal = self._extract_temporal_info(question)
        
        # Determine needed operators
        operators = self.question_patterns.get(intent_type, {}).get('operators', [])
        
        return QuestionIntent(
            intent_type=intent_type,
            entities_mentioned=entities,
            numbers_mentioned=numbers,
            temporal_info=temporal,
            operators_needed=operators
        )
    
    def _extract_entities_from_question(self, question: str) -> List[str]:
        """Extract potential entity names from question"""
        entities = []
        
        # Common financial entities
        patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Proper nouns
            r'\b(revenue|profit|cost|income|expense|asset|liability|equity)\b',
            r'\b(company|corporation|firm|business)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_numbers_from_question(self, question: str) -> List[float]:
        """Extract numbers from question"""
        numbers = []
        pattern = r'\b(\d+(?:\.\d+)?)\b'
        
        for match in re.finditer(pattern, question):
            try:
                numbers.append(float(match.group(1)))
            except:
                continue
        
        return numbers
    
    def _extract_temporal_info(self, question: str) -> List[str]:
        """Extract years, quarters, etc."""
        temporal = []
        
        # Years
        years = re.findall(r'\b(19|20)\d{2}\b', question)
        temporal.extend(years)
        
        # Quarters
        quarters = re.findall(r'\bQ[1-4]\b', question, re.IGNORECASE)
        temporal.extend(quarters)
        
        # Months
        months = re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', 
                          question, re.IGNORECASE)
        temporal.extend(months)
        
        return temporal
    
    def _find_relevant_nodes(self, intent: QuestionIntent, question: str) -> List[Tuple[str, float]]:
        """
        Tìm các nodes trong KG relevant với question
        Return: List of (node_id, relevance_score)
        """
        relevant = []
        
        question_lower = question.lower()
        
        # Search through all nodes
        for node_id, data in self.kg.nodes(data=True):
            score = 0.0
            
            # Check node type
            node_type = data.get('type', '')
            
            # Numbers are highly relevant for calculations
            if node_type in ['number', 'cell']:
                score += 1.0
                
                # Check if number value matches mentioned numbers
                node_value = data.get('value') or data.get('number')
                if node_value in intent.numbers_mentioned:
                    score += 2.0
            
            # Text nodes - check content
            if node_type == 'text':
                content = data.get('content', '').lower()
                
                # Check for entity mentions
                for entity in intent.entities_mentioned:
                    if entity.lower() in content:
                        score += 1.5
                
                # Check for temporal info
                for temporal in intent.temporal_info:
                    if temporal.lower() in content:
                        score += 1.0
            
            # Cell nodes - check headers and row labels
            if node_type == 'cell':
                header = data.get('header', '').lower()
                row_label = data.get('row_label', '').lower()
                
                # Match with entities
                for entity in intent.entities_mentioned:
                    if entity.lower() in header or entity.lower() in row_label:
                        score += 2.0
                
                # Match with question keywords
                for word in question_lower.split():
                    if len(word) > 3:  # Skip short words
                        if word in header or word in row_label:
                            score += 0.5
            
            if score > 0:
                relevant.append((node_id, score))
        
        # Sort by score
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        return relevant[:10]  # Top 10
    
    def _synthesize_program_advanced(self, 
                                    intent: QuestionIntent, 
                                    relevant_nodes: List[Tuple[str, float]]) -> str:
        """
        Synthesize program dựa trên intent và relevant nodes
        """
        # Get top numbers from relevant nodes
        numbers = []
        for node_id, score in relevant_nodes:
            node_data = self.kg.nodes[node_id]
            
            if node_data.get('type') == 'number':
                num_val = node_data.get('value')
                if num_val is not None:
                    numbers.append(num_val)
            elif node_data.get('type') == 'cell':
                num_val = node_data.get('number')
                if num_val is not None:
                    numbers.append(num_val)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_numbers = []
        for num in numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)
        
        # Generate program based on intent
        if intent.intent_type == 'average':
            if len(unique_numbers) >= 2:
                return f"divide({unique_numbers[0]}, {unique_numbers[1]})"
        
        elif intent.intent_type == 'percentage_change':
            if len(unique_numbers) >= 2:
                # (new - old) / old * 100
                return f"divide(subtract({unique_numbers[0]}, {unique_numbers[1]}), {unique_numbers[1]})"
        
        elif intent.intent_type == 'total':
            if len(unique_numbers) >= 2:
                return f"add({unique_numbers[0]}, {unique_numbers[1]})"
        
        elif intent.intent_type == 'difference':
            if len(unique_numbers) >= 2:
                return f"subtract({unique_numbers[0]}, {unique_numbers[1]})"
        
        elif intent.intent_type == 'ratio':
            if len(unique_numbers) >= 2:
                return f"divide({unique_numbers[0]}, {unique_numbers[1]})"
        
        # Default: try first operation that makes sense
        if len(unique_numbers) >= 2:
            return f"divide({unique_numbers[0]}, {unique_numbers[1]})"
        
        return ""
    
    def _generate_enhanced_explanation(self,
                                      result: ExecutionResult,
                                      question: str,
                                      intent: QuestionIntent,
                                      relevant_nodes: List[Tuple[str, float]]) -> str:
        """Generate detailed explanation with KG context"""
        
        explanation = f"""
{'='*70}
QUESTION ANALYSIS
{'='*70}
Question: {question}
Intent Type: {intent.intent_type}
Entities Mentioned: {', '.join(intent.entities_mentioned) or 'None'}
Numbers Mentioned: {', '.join(map(str, intent.numbers_mentioned)) or 'None'}
Temporal Info: {', '.join(intent.temporal_info) or 'None'}

{'='*70}
KNOWLEDGE GRAPH EVIDENCE
{'='*70}
Top Relevant Nodes:
"""
        
        for idx, (node_id, score) in enumerate(relevant_nodes[:5], 1):
            node_data = self.kg.nodes[node_id]
            explanation += f"\n{idx}. [{node_data.get('type', 'unknown').upper()}] {node_id} (score: {score:.2f})"
            
            if node_data.get('type') == 'cell':
                explanation += f"\n   Value: {node_data.get('value')}"
                explanation += f"\n   Location: Row {node_data.get('row')}, Col {node_data.get('col')}"
                explanation += f"\n   Header: {node_data.get('header')}"
                explanation += f"\n   Row Label: {node_data.get('row_label')}"
            elif node_data.get('type') == 'number':
                explanation += f"\n   Value: {node_data.get('value')}"
                explanation += f"\n   Context: {node_data.get('context', '')[:100]}..."
            elif node_data.get('type') == 'text':
                explanation += f"\n   Content: {node_data.get('content', '')[:100]}..."
        
        explanation += f"""

{'='*70}
COMPUTATION STEPS
{'='*70}
"""
        
        for idx, step in enumerate(result.steps, 1):
            explanation += f"\nStep {idx}: {step.operator.upper()}({step.arg1}, {step.arg2})"
            explanation += f"\n  → Result: {step.result}"
            
            if step.source_nodes:
                explanation += f"\n  → Evidence from KG nodes: {', '.join(step.source_nodes[:3])}"
        
        explanation += f"""

{'='*70}
FINAL RESULT
{'='*70}
Computed Answer: {result.final_answer}
"""
        
        if result.ground_truth is not None:
            explanation += f"Ground Truth:    {result.ground_truth}\n"
            explanation += f"Match:           {'✓ CORRECT' if result.is_correct else '✗ INCORRECT'}\n"
            
            if not result.is_correct:
                diff = abs(result.final_answer - result.ground_truth)
                explanation += f"Difference:      {diff:.4f}\n"
        
        explanation += f"{'='*70}\n"
        
        return explanation
