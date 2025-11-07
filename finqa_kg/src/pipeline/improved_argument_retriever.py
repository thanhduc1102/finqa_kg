"""
Improved Program Synthesizer với better argument retrieval
Thêm semantic matching và robust logic
"""
import networkx as nx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
import numpy as np

from .question_analyzer import QuestionAnalysis


@dataclass
class ArgumentCandidate:
    """Enhanced candidate with richer metadata"""
    value: float
    text: str
    context: str
    score: float
    label: str = ''
    node_id: str = ''
    source_type: str = ''  # 'table', 'text', 'computed'
    confidence: float = 0.0


class ImprovedArgumentRetriever:
    """
    Improved argument retrieval với:
    - Better context matching
    - Multi-strategy retrieval
    - Confidence scoring
    """
    
    def __init__(self):
        self.min_confidence = 0.3
    
    def retrieve_arguments(self,
                          qa: QuestionAnalysis,
                          kg: nx.MultiDiGraph,
                          entity_index: Dict,
                          required_args: List[str]) -> Dict[str, ArgumentCandidate]:
        """
        Multi-strategy argument retrieval
        """
        # Strategy 1: Direct entity match
        direct_matches = self._direct_entity_match(qa, kg, entity_index)
        
        # Strategy 2: Context-based retrieval
        context_matches = self._context_based_retrieval(qa, kg)
        
        # Strategy 3: Table-aware retrieval
        table_matches = self._table_aware_retrieval(qa, kg)
        
        # Merge and score all candidates
        all_candidates = self._merge_candidates(
            direct_matches, context_matches, table_matches
        )
        
        # Assign to required arguments
        arguments = self._assign_to_args(
            all_candidates, required_args, qa
        )
        
        return arguments
    
    def _direct_entity_match(self, qa, kg, entity_index) -> List[ArgumentCandidate]:
        """Match entities directly mentioned in question"""
        candidates = []
        
        for entity_text in qa.entities_mentioned:
            matches = entity_index['by_text'].get(entity_text.lower(), [])
            for match in matches:
                node_data = match['data']
                if node_data.get('value') is not None:
                    candidates.append(ArgumentCandidate(
                        value=node_data['value'],
                        text=node_data.get('text', str(node_data['value'])),
                        context=node_data.get('context', ''),
                        score=50.0,  # Base score for direct match
                        label=node_data.get('label', ''),
                        node_id=match.get('node_id', ''),
                        source_type='direct',
                        confidence=0.7
                    ))
        
        return candidates
    
    def _context_based_retrieval(self, qa, kg) -> List[ArgumentCandidate]:
        """Enhanced context matching"""
        candidates = []
        key_entities = [e for e in qa.entities_mentioned if len(e) > 2]
        
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') not in ['entity', 'cell']:
                continue
            if node_data.get('value') is None:
                continue
            
            context = node_data.get('context', '').lower()
            label = node_data.get('label', '')
            
            # Skip unwanted labels based on question type
            if qa.question_type != 'percentage_change' and label == 'DATE':
                continue
            
            # Calculate match score
            match_score = 0
            
            # Entity keyword matching
            for entity in key_entities:
                if entity.lower() in context:
                    match_score += 15
                    
                    # Proximity bonus
                    value_str = str(node_data['value'])
                    if value_str in context:
                        entity_pos = context.find(entity.lower())
                        value_pos = context.find(value_str)
                        if abs(entity_pos - value_pos) < 50:
                            match_score += 10
            
            # Temporal matching
            for temporal in qa.temporal_entities:
                # Normalize temporal for better matching
                norm_temporal = re.sub(r'[\s\.\,\-]', '', temporal.lower())
                norm_context = re.sub(r'[\s\.\,\-]', '', context)
                if norm_temporal in norm_context:
                    match_score += 5
            
            # Label-based scoring
            if label in ['MONEY', 'REVENUE', 'EXPENSE', 'INCOME', 'EQUITY', 'ASSET']:
                match_score += 8
            elif label == 'PERCENT' and qa.question_type not in ['percentage_of', 'percentage_change']:
                match_score -= 15
            
            if match_score > 0:
                candidates.append(ArgumentCandidate(
                    value=node_data['value'],
                    text=node_data.get('text', str(node_data['value'])),
                    context=context,
                    score=match_score,
                    label=label,
                    node_id=node_id,
                    source_type='context',
                    confidence=min(match_score / 50.0, 1.0)
                ))
        
        return candidates
    
    def _table_aware_retrieval(self, qa, kg) -> List[ArgumentCandidate]:
        """Table-specific retrieval logic"""
        candidates = []
        
        # Find table nodes
        table_cells = [
            (nid, ndata) for nid, ndata in kg.nodes(data=True)
            if ndata.get('type') == 'cell' and ndata.get('value') is not None
        ]
        
        key_entities = [e for e in qa.entities_mentioned if len(e) > 2]
        
        for node_id, node_data in table_cells:
            context = node_data.get('context', '').lower()
            
            # Check if context matches question entities
            match_count = sum(1 for e in key_entities if e.lower() in context)
            
            if match_count > 0:
                candidates.append(ArgumentCandidate(
                    value=node_data['value'],
                    text=node_data.get('text', str(node_data['value'])),
                    context=context,
                    score=match_count * 20,  # Boost table matches
                    label=node_data.get('label', ''),
                    node_id=node_id,
                    source_type='table',
                    confidence=min(match_count / 3.0, 0.9)
                ))
        
        return candidates
    
    def _merge_candidates(self, *candidate_lists) -> List[ArgumentCandidate]:
        """Merge and deduplicate candidates"""
        all_candidates = []
        seen_values = set()
        
        for cand_list in candidate_lists:
            for cand in cand_list:
                # Deduplicate by value (approximately)
                value_key = round(cand.value, 6)
                if value_key not in seen_values:
                    all_candidates.append(cand)
                    seen_values.add(value_key)
                else:
                    # Update score if this is a better match
                    for existing in all_candidates:
                        if round(existing.value, 6) == value_key:
                            if cand.score > existing.score:
                                existing.score = cand.score
                                existing.confidence = max(existing.confidence, cand.confidence)
                            break
        
        # Sort by score
        all_candidates.sort(key=lambda x: (x.score, x.confidence), reverse=True)
        
        return all_candidates
    
    def _assign_to_args(self, candidates: List[ArgumentCandidate],
                       required_args: List[str],
                       qa: QuestionAnalysis) -> Dict[str, ArgumentCandidate]:
        """Intelligently assign candidates to required arguments"""
        arguments = {}
        
        if not candidates:
            return arguments
        
        # Type-specific assignment logic
        if qa.question_type == 'percentage_change':
            if len(candidates) >= 2 and len(qa.temporal_entities) >= 2:
                # Temporal ordering
                sorted_temporal = sorted(qa.temporal_entities)
                old_val = sorted_temporal[0]
                new_val = sorted_temporal[-1]
                
                # Find candidates matching temporal
                old_cands = [c for c in candidates if old_val in c.context]
                new_cands = [c for c in candidates if new_val in c.context]
                
                if old_cands and new_cands:
                    arguments['old'] = old_cands[0]
                    arguments['new'] = new_cands[0]
                else:
                    # Fallback
                    arguments['old'] = candidates[1] if len(candidates) > 1 else candidates[0]
                    arguments['new'] = candidates[0]
        
        elif qa.question_type in ['ratio', 'percentage_of']:
            if len(candidates) >= 2:
                # Try to identify part/whole or numerator/denominator
                # Heuristic: larger value is usually 'whole' or 'denominator'
                sorted_by_value = sorted(candidates[:5], key=lambda x: x.value)
                
                if 'part' in required_args:
                    arguments['part'] = sorted_by_value[0]
                    arguments['whole'] = sorted_by_value[-1]
                elif 'numerator' in required_args:
                    arguments['numerator'] = candidates[0]
                    arguments['denominator'] = candidates[1]
        
        elif qa.question_type == 'comparison':
            if len(candidates) >= 2:
                arguments['value1'] = candidates[0]
                arguments['value2'] = candidates[1]
        
        elif qa.question_type == 'direct_lookup':
            if 'value' in required_args and candidates:
                arguments['value'] = candidates[0]
        
        else:
            # Generic assignment
            for i, arg_name in enumerate(required_args):
                if i < len(candidates):
                    arguments[arg_name] = candidates[i]
        
        return arguments
