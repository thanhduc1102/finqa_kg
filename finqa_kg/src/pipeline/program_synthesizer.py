"""
Program Synthesizer
Tự động sinh program từ:
- Question analysis
- KG retrieval
- Argument ordering logic
- ENHANCED: Semantic retrieval with embeddings + reranking
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re

from .question_analyzer import QuestionAnalysis
from .semantic_retriever import get_semantic_retriever, SemanticMatch
from .semantic_matcher import get_semantic_matcher  # PHASE 1.1: Semantic matching!

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
        
        # DISABLE semantic retriever - TOO SLOW and not helping!
        self.semantic_retriever = None  # DISABLED
        self.use_semantic_retrieval = False
        
        # PHASE 1.1: Initialize semantic matcher for entity matching
        print("[ProgramSynthesizer] Initializing semantic matcher...")
        self.semantic_matcher = get_semantic_matcher('all-MiniLM-L6-v2')  # Fast model
        print("[ProgramSynthesizer] Semantic matcher ready!")
    
    def _normalize_argument(self, arg: Any) -> Dict[str, Any]:
        """
        CRITICAL FIX: Normalize argument to standard format
        
        Ensures ALL arguments have consistent structure:
        {'value': float, 'text': str, 'context': str, 'score': int, ...}
        
        Args:
            arg: Can be:
                - float/int (raw number)
                - str (text)
                - dict (already structured)
                
        Returns:
            Standardized dict
        """
        if isinstance(arg, dict):
            # Already a dict - ensure it has required keys
            return {
                'value': arg.get('value', 0.0),
                'text': arg.get('text', str(arg.get('value', ''))),
                'context': arg.get('context', ''),
                'score': arg.get('score', 0),
                'node_id': arg.get('node_id', ''),
                'label': arg.get('label', ''),
                'matched': arg.get('matched', [])
            }
        elif isinstance(arg, (int, float)):
            # Raw number
            return {
                'value': float(arg),
                'text': str(arg),
                'context': f'Raw number: {arg}',
                'score': 50,
                'node_id': '',
                'label': 'NUMBER',
                'matched': []
            }
        elif isinstance(arg, str):
            # Try to parse as number
            try:
                value = float(arg.replace(',', ''))
                return {
                    'value': value,
                    'text': arg,
                    'context': f'Parsed from string: {arg}',
                    'score': 50,
                    'node_id': '',
                    'label': 'NUMBER',
                    'matched': []
                }
            except:
                # Not a number
                return {
                    'value': 0.0,
                    'text': arg,
                    'context': f'Text entity: {arg}',
                    'score': 0,
                    'node_id': '',
                    'label': 'TEXT',
                    'matched': []
                }
    
    def _load_program_templates(self) -> Dict[str, Dict]:
        """
        Load templates cho từng question type
        """
        return {
            'percentage_change': {
                'template': 'divide(subtract(#new, #old), #old)',
                'description': 'Percentage change = (new - old) / old',
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
                'template': 'divide(#part, #whole)',
                'description': 'Percentage = part / whole',
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
            },
            'comparison': {
                'template': 'greater(#value1, #value2)',
                'description': 'Compare two values (returns 1 if value1 > value2, else 0)',
                'required_args': ['value1', 'value2'],
                'arg_types': ['number', 'number']
            }
        }
    
    def _semantic_retrieval_strategy(self,
                                    qa: QuestionAnalysis,
                                    kg: nx.MultiDiGraph,
                                    required_args: List[str],
                                    temporal_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Semantic retrieval strategy based on question type
        
        Args:
            qa: Question analysis
            kg: Knowledge graph
            required_args: List of required argument names
            temporal_filter: Optional temporal constraint
        
        Returns:
            Dict mapping arg names to retrieved values (normalized format)
        """
        try:
            q_type = qa.question_type
            question = qa.question
            
            # Strategy 1: percentage_of, ratio - need part and whole
            if q_type in ['percentage_of', 'ratio'] and set(required_args) == {'part', 'whole'}:
                return self._semantic_retrieve_part_whole(qa, kg, temporal_filter)
            
            # Strategy 2: percentage_change - need old and new
            elif q_type == 'percentage_change' and set(required_args) == {'new', 'old'}:
                return self._semantic_retrieve_old_new(qa, kg, temporal_filter)
            
            # Strategy 3: difference - need value1 and value2
            elif q_type == 'difference' and 'value1' in required_args and 'value2' in required_args:
                return self._semantic_retrieve_two_values(qa, kg, temporal_filter)
            
            # Strategy 4: sum - need multiple values
            elif q_type == 'sum':
                return self._semantic_retrieve_multiple_values(qa, kg, temporal_filter)
            
            # Strategy 5: direct_lookup - need single value
            elif q_type == 'direct_lookup' and 'value' in required_args:
                return self._semantic_retrieve_single_value(qa, kg, temporal_filter)
            
            # Fallback: generic retrieval
            else:
                return self._semantic_retrieve_generic(qa, kg, required_args, temporal_filter)
        
        except Exception as e:
            print(f"  [SEMANTIC] Error in semantic retrieval: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _semantic_retrieve_part_whole(self,
                                     qa: QuestionAnalysis,
                                     kg: nx.MultiDiGraph,
                                     temporal_filter: Optional[str]) -> Dict[str, Any]:
        """
        Retrieve part and whole for percentage_of questions
        Example: "what percentage of X were Y?"
        - part: Y (the specific item)
        - whole: X (the total)
        
        CRITICAL: Queries MUST be different enough to retrieve different values!
        """
        question = qa.question.lower()
        
        # Pattern analysis for percentage_of questions:
        # "what percentage of [WHOLE] were [PART]"
        # "what percent of [WHOLE] was [PART]"
        
        # Strategy: Extract specific keywords that differentiate part vs whole
        key_entities = [e for e in qa.entities_mentioned if len(e) > 2 and not e.isdigit()]
        
        # ENHANCED: Parse question structure to identify part vs whole
        # Common patterns:
        # - "percentage of X consisting/comprised of Y" -> whole=X, part=Y
        # - "percentage of X due to Y" -> whole=X, part=Y
        # - "percent of X associated with Y" -> whole=X, part=Y
        
        part_keywords = []
        whole_keywords = []
        
        # Identify part keywords (more specific)
        for marker in ['due to', 'consisting of', 'comprised of', 'associated with', 
                      'related to', 'from', 'for']:
            if marker in question:
                # Text after marker is usually the part
                after_marker = question.split(marker, 1)[1]
                # Extract entities from this part
                part_keywords = [e for e in key_entities if e.lower() in after_marker]
                break
        
        # Identify whole keywords (less specific, usually mentions "total")
        if 'total' in question or 'all' in question:
            # Text around "total" is usually the whole
            for e in key_entities:
                if e.lower() in question.split('total')[0] if 'total' in question else question:
                    if e not in part_keywords:
                        whole_keywords.append(e)
        
        # If parsing failed, use heuristics
        if not part_keywords and not whole_keywords:
            if len(key_entities) >= 2:
                # Last entity is usually more specific (part)
                part_keywords = [key_entities[-1]]
                # First entities are usually general (whole)
                whole_keywords = key_entities[:-1]
            elif len(key_entities) == 1:
                # Single entity - might be in part, whole is "total"
                part_keywords = [key_entities[0]]
                whole_keywords = ['total']
        
        # Build queries
        temporal_str = f" {temporal_filter}" if temporal_filter else ""
        
        # PART query: specific item
        if part_keywords:
            part_query = f"{' '.join(part_keywords)}{temporal_str}"
        else:
            part_query = f"specific item {question}"
        
        # WHOLE query: total/broader category
        if whole_keywords:
            whole_query = f"total {' '.join(whole_keywords)}{temporal_str}"
        else:
            whole_query = f"total all{temporal_str}"
        
        print(f"  [Semantic] Part query: '{part_query}'")
        print(f"  [Semantic] Whole query: '{whole_query}'")
        
        # Retrieve separately with different queries
        part_matches = self.semantic_retriever.retrieve_for_question(
            part_query,
            kg,
            top_k=5,
            temporal_filter=temporal_filter
        )
        
        whole_matches = self.semantic_retriever.retrieve_for_question(
            whole_query,
            kg,
            top_k=5,
            temporal_filter=temporal_filter
        )
        
        results = {}
        
        if part_matches:
            results['part'] = self._semantic_match_to_dict(part_matches[0])
            print(f"  [Semantic] part: {part_matches[0].value} (score={part_matches[0].score:.3f})")
        
        if whole_matches:
            results['whole'] = self._semantic_match_to_dict(whole_matches[0])
            print(f"  [Semantic] whole: {whole_matches[0].value} (score={whole_matches[0].score:.3f})")
        
        # Sanity check: part and whole should be different!
        if 'part' in results and 'whole' in results:
            if results['part']['value'] == results['whole']['value']:
                print(f"  [Semantic] WARNING️  WARNING: Part and whole are identical! Trying alternative...")
                
                # Try next candidates
                if len(part_matches) > 1:
                    results['part'] = self._semantic_match_to_dict(part_matches[1])
                    print(f"  [Semantic] Using part candidate #2: {part_matches[1].value}")
                elif len(whole_matches) > 1:
                    results['whole'] = self._semantic_match_to_dict(whole_matches[1])
                    print(f"  [Semantic] Using whole candidate #2: {whole_matches[1].value}")
        
        return results
    
    def _semantic_retrieve_old_new(self,
                                  qa: QuestionAnalysis,
                                  kg: nx.MultiDiGraph,
                                  temporal_filter: Optional[str]) -> Dict[str, Any]:
        """
        Retrieve old and new values for percentage_change questions
        Example: "what was the change from 2007 to 2008?"
        """
        temporal_entities = sorted(qa.temporal_entities, key=len, reverse=True)
        
        if len(temporal_entities) >= 2:
            # Have specific years/dates
            old_temporal = temporal_entities[-1]  # Earlier
            new_temporal = temporal_entities[0]   # Later (or most specific)
            
            key_entities = [e for e in qa.entities_mentioned if len(e) > 2]
            entity_desc = ' '.join(key_entities) if key_entities else 'value'
            
            arg_descriptions = {
                'old': f"{entity_desc} {old_temporal}",
                'new': f"{entity_desc} {new_temporal}"
            }
        else:
            # No specific temporals, use generic
            arg_descriptions = {
                'old': f"earlier value {qa.question}",
                'new': f"later value {qa.question}"
            }
        
        results = self.semantic_retriever.retrieve_multi_args(
            qa.question,
            kg,
            arg_descriptions,
            temporal_filter=None  # Don't use single temporal filter for old/new
        )
        
        converted = {}
        for arg_name, match in results.items():
            converted[arg_name] = self._semantic_match_to_dict(match)
        
        return converted
    
    def _semantic_retrieve_two_values(self,
                                    qa: QuestionAnalysis,
                                    kg: nx.MultiDiGraph,
                                    temporal_filter: Optional[str]) -> Dict[str, Any]:
        """Generic retrieval for two values"""
        matches = self.semantic_retriever.retrieve_for_question(
            qa.question,
            kg,
            top_k=5,
            temporal_filter=temporal_filter
        )
        
        if len(matches) >= 2:
            return {
                'value1': self._semantic_match_to_dict(matches[0]),
                'value2': self._semantic_match_to_dict(matches[1])
            }
        
        return {}
    
    def _semantic_retrieve_multiple_values(self,
                                         qa: QuestionAnalysis,
                                         kg: nx.MultiDiGraph,
                                         temporal_filter: Optional[str]) -> Dict[str, Any]:
        """Retrieve multiple values for sum"""
        matches = self.semantic_retriever.retrieve_for_question(
            qa.question,
            kg,
            top_k=10,
            temporal_filter=temporal_filter
        )
        
        if matches:
            # Return as numbered arguments
            result = {}
            for i, match in enumerate(matches[:5]):  # Max 5 values
                result[f'value{i}'] = self._semantic_match_to_dict(match)
            return result
        
        return {}
    
    def _semantic_retrieve_single_value(self,
                                       qa: QuestionAnalysis,
                                       kg: nx.MultiDiGraph,
                                       temporal_filter: Optional[str]) -> Dict[str, Any]:
        """
        Retrieve single value for direct lookup
        ENHANCED: Build more specific query from question entities
        """
        # Build specific query from question
        key_entities = [e for e in qa.entities_mentioned if len(e) > 2 and not e.isdigit()]
        
        # Create targeted query
        if key_entities:
            # Combine all entities into query
            query_parts = []
            for entity in key_entities:
                query_parts.append(entity)
            
            if temporal_filter:
                query_parts.append(temporal_filter)
            
            # Example: "interest expense 2009"
            query = ' '.join(query_parts)
        else:
            # Fallback to full question
            query = qa.question
        
        print(f"  [Semantic] Direct lookup query: '{query}'")
        
        matches = self.semantic_retriever.retrieve_for_question(
            query,
            kg,
            top_k=5,  # Get top 5 to examine
            temporal_filter=temporal_filter
        )
        
        if matches:
            # Show top matches for debugging
            print(f"  [Semantic] Top 3 matches:")
            for i, match in enumerate(matches[:3]):
                print(f"    {i+1}. value={match.value}, score={match.score:.3f}, text='{match.text}'")
                print(f"       context: {match.context[:100]}")
            
            return {
                'value': self._semantic_match_to_dict(matches[0])
            }
        
        return {}
    
    def _semantic_retrieve_generic(self,
                                  qa: QuestionAnalysis,
                                  kg: nx.MultiDiGraph,
                                  required_args: List[str],
                                  temporal_filter: Optional[str]) -> Dict[str, Any]:
        """Generic semantic retrieval for any question type"""
        matches = self.semantic_retriever.retrieve_for_question(
            qa.question,
            kg,
            top_k=len(required_args) + 2,
            temporal_filter=temporal_filter
        )
        
        if len(matches) >= len(required_args):
            result = {}
            for i, arg_name in enumerate(required_args):
                result[arg_name] = self._semantic_match_to_dict(matches[i])
            return result
        
        return {}
    
    def _semantic_match_to_dict(self, match: SemanticMatch) -> Dict[str, Any]:
        """Convert SemanticMatch to standard argument format"""
        return {
            'value': match.value,
            'text': match.text,
            'context': match.context,
            'score': int(match.score * 100),  # Convert to 0-100 scale
            'node_id': match.node_id,
            'label': match.label,
            'matched': ['semantic'],
            'semantic_score': match.semantic_score,
            'rerank_score': match.rerank_score
        }
    
    def _extract_table_row_id(self, context: str) -> str:
        """
        Extract table row identifier from context string
        
        Context format: "table[row_label]: col1=val1, col2=val2"
        Returns: row_label or empty string
        
        Examples:
            "table[total cash and investments]: dec 29 2012=26302" → "total cash and investments"
            "table[available-for-sale]: dec 29 2012=14001" → "available-for-sale"
        """
        import re
        match = re.search(r'table\[([^\]]+)\]', context)
        if match:
            return match.group(1).strip().lower()
        return ""
    
    def _retrieve_from_kg(self, kg: nx.MultiDiGraph, 
                         keywords: List[str], 
                         temporal_filter: str = None,
                         exclude_labels: List[str] = None,
                         prefer_table: bool = False,
                         require_all_keywords: bool = False) -> List[Dict]:
        """
        ENHANCED: Retrieve values with improved scoring mechanism
        
        Args:
            kg: Knowledge Graph
            keywords: Keywords to match (e.g., ['interest', 'expense', '3.8'])
            temporal_filter: Temporal constraint (e.g., '2009') 
            exclude_labels: Entity labels to exclude (e.g., ['DATE'])
            prefer_table: If True, boost table entity scores
            require_all_keywords: If True, only return entities matching ALL keywords
            
        Returns:
            List of matching entities with scores
        """
        print(f"\n  [KG Retrieval] keywords={keywords}, temporal={temporal_filter}, prefer_table={prefer_table}")
        
        results = []
        exclude_labels = exclude_labels or []
        
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') != 'entity':
                continue
            
            # Filter by label
            entity_label = node_data.get('label', '')
            if entity_label in exclude_labels:
                continue
            
            entity_text = node_data.get('text', '').lower()
            entity_value = node_data.get('value')
            context = node_data.get('context', '').lower()
            
            if entity_value is None:
                continue
            
            # VALUE MAGNITUDE VALIDATION
            # Skip suspiciously small values unless question is about percentages
            if isinstance(entity_value, (int, float)):
                # Very small values (< 0.001) are likely wrong unless it's a percentage/rate
                if abs(entity_value) < 0.001 and entity_label not in ['PERCENT', 'RATE']:
                    # Check if question mentions "percent" or "rate"
                    # (these questions can have small decimal values)
                    continue
                
                # Very large values (> 1e12) are suspicious
                if abs(entity_value) > 1e12:
                    continue
            
            # ENHANCED SCORING MECHANISM
            score = 0
            matched_keywords = []
            exact_matches = 0
            partial_matches = 0
            
            # Combined search text (entity text + context)
            search_text = f"{entity_text} {context}"
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Exact match in entity text (highest score)
                if keyword_lower == entity_text:
                    score += 20
                    exact_matches += 1
                    matched_keywords.append(f"EXACT:{keyword}")
                # Exact phrase match in context
                elif keyword_lower in context:
                    score += 15
                    exact_matches += 1
                    matched_keywords.append(keyword)
                # Partial word match
                elif any(keyword_lower in word for word in search_text.split()):
                    score += 5
                    partial_matches += 1
                    matched_keywords.append(f"partial:{keyword}")
            
            # CRITICAL: If require_all_keywords is set, skip if not all matched
            if require_all_keywords and len(matched_keywords) < len(keywords):
                continue
            
            # Keyword density bonus: more keywords matched = better
            keyword_coverage = len(matched_keywords) / len(keywords) if keywords else 0
            score += int(keyword_coverage * 10)
            
            # Temporal boost
            if temporal_filter and temporal_filter.lower() in context:
                score += 5
                matched_keywords.append(f"temporal:{temporal_filter}")
            
            # Table preference boost - STRONG for financial data
            if prefer_table and 'table[' in context:
                score += 15
            
            # Prefer 'total' for whole, avoid 'total' for part
            if 'total' in keywords and 'total' in search_text:
                score += 10  # Bonus for matching 'total'
            
            if score > 0:
                is_from_table = 'table[' in context
                results.append({
                    'value': entity_value,
                    'text': entity_text,
                    'score': score,
                    'matched': matched_keywords,
                    'context': context,
                    'node_id': node_id,
                    'label': entity_label,
                    'source': 'table' if is_from_table else 'text',
                    'exact_matches': exact_matches,
                    'partial_matches': partial_matches
                })
        
        # Sort by score (desc), then by exact matches (desc), then by absolute value (desc)
        results.sort(key=lambda x: (
            x['score'], 
            x.get('exact_matches', 0),
            abs(x['value']) if isinstance(x['value'], (int, float)) else 0
        ), reverse=True)
        
        if results:
            print(f"  Found {len(results)} entities ({sum(1 for r in results if r['source']=='table')} table, {sum(1 for r in results if r['source']=='text')} text)")
            for i, r in enumerate(results[:3]):
                print(f"    {i+1}. [{r['source']}] value={r['value']}, score={r['score']}, "
                      f"exact={r.get('exact_matches', 0)}, matched={r['matched']}")
        else:
            print(f"  No entities found!")
        
        return results
    
    def _retrieve_from_table(self, kg: nx.MultiDiGraph, 
                            row_keywords: List[str], 
                            col_keywords: List[str] = None,
                            exclude_labels: List[str] = None) -> List[Dict]:
        """
        DEPRECATED: Use _retrieve_from_kg() instead
        
        Kept for backward compatibility with structured retrieval
        """
        # Combine row and column keywords
        all_keywords = row_keywords.copy()
        temporal_filter = None
        
        if col_keywords:
            # Col keywords usually contain temporal info
            all_keywords.extend(col_keywords)
            # Try to extract temporal
            for kw in col_keywords:
                if kw.isdigit() and len(kw) == 4:  # Year
                    temporal_filter = kw
                    break
        
        # Call new method with table preference
        return self._retrieve_from_kg(
            kg, 
            keywords=all_keywords,
            temporal_filter=temporal_filter,
            exclude_labels=exclude_labels,
            prefer_table=True  # Prefer table for structured retrieval
        )
    
    def _retrieve_from_table_with_hint(self, kg: nx.MultiDiGraph, 
                                      row_keywords: List[str], 
                                      col_keywords: List[str] = None,
                                      exclude_labels: List[str] = None,
                                      row_type_hint: str = None,
                                      exclude_node_ids: set = None) -> List[Dict]:
        """
        ENHANCED: Retrieve with context-aware scoring
        
        Args:
            row_type_hint: 'prefer_total' (for whole/denominator) or 'avoid_total' (for part/numerator)
            exclude_node_ids: Set of node IDs to exclude (prevent duplicate retrieval)
        """
        print(f"\n  [_retrieve_from_table_with_hint] keywords={row_keywords}, hint={row_type_hint}")
        
        # Combine row and column keywords
        all_keywords = row_keywords.copy()
        temporal_filter = None
        
        if col_keywords:
            all_keywords.extend(col_keywords)
            for kw in col_keywords:
                if kw.isdigit() and len(kw) == 4:
                    temporal_filter = kw
                    break
        
        exclude_labels = exclude_labels or []
        exclude_node_ids = exclude_node_ids or set()
        
        results = []
        
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') != 'entity':
                continue
            
            # Skip excluded nodes
            if node_id in exclude_node_ids:
                continue
            
            # Filter by label
            entity_label = node_data.get('label', '')
            if entity_label in exclude_labels:
                continue
            
            entity_text = node_data.get('text', '').lower()
            entity_value = node_data.get('value')
            context = node_data.get('context', '').lower()
            
            if entity_value is None:
                continue
            
            # VALUE MAGNITUDE VALIDATION
            if isinstance(entity_value, (int, float)):
                if abs(entity_value) < 0.001 and entity_label not in ['PERCENT', 'RATE']:
                    continue
                if abs(entity_value) > 1e12:
                    continue
            
            # ENHANCED SCORING with row_type_hint
            score = 0
            matched_keywords = []
            exact_matches = 0
            
            search_text = f"{entity_text} {context}"
            
            for keyword in all_keywords:
                keyword_lower = keyword.lower()
                
                if keyword_lower == entity_text:
                    score += 20
                    exact_matches += 1
                    matched_keywords.append(f"EXACT:{keyword}")
                elif keyword_lower in context:
                    score += 15
                    exact_matches += 1
                    matched_keywords.append(keyword)
                elif any(keyword_lower in word for word in search_text.split()):
                    score += 5
                    matched_keywords.append(f"partial:{keyword}")
            
            keyword_coverage = len(matched_keywords) / len(all_keywords) if all_keywords else 0
            score += int(keyword_coverage * 10)
            
            # CRITICAL FIX #13: ENFORCE temporal match - MUST match if temporal_filter provided!
            temporal_matched = False
            if temporal_filter:
                # Normalize temporal for matching (remove spaces, dots)
                import re
                normalized_temporal = re.sub(r'[\s\.\,\-]', '', temporal_filter.lower())
                normalized_context = re.sub(r'[\s\.\,\-]', '', context)
                
                if normalized_temporal in normalized_context:
                    score += 10  # Boost temporal match
                    matched_keywords.append(f"temporal:{temporal_filter}")
                    temporal_matched = True
                else:
                    # CRITICAL: If temporal_filter provided but NOT matched, SKIP this entity!
                    continue  # Skip to next entity
            
            # Table preference
            if 'table[' in context:
                score += 15
            
            # CRITICAL FIX: Context-aware scoring based on row_type_hint
            if row_type_hint == 'prefer_total':
                # Boost if context contains 'total', 'sum', 'all'
                if any(word in context for word in ['total', ' sum ', ' all ']):
                    score += 20
                    matched_keywords.append('HINT:total_context')
                # Penalize if context suggests specific item (not total)
                if any(word in context for word in ['proceeds', 'gain', 'loss', 'specific']):
                    score -= 10
            
            elif row_type_hint == 'avoid_total':
                # Penalize if context contains 'total', 'sum', 'all'
                if any(word in context for word in ['total', ' sum ', ' all ']):
                    score -= 20
                    matched_keywords.append('HINT:avoid_total')
                # Boost if context suggests specific item
                if any(word in context for word in ['proceeds', 'gain', 'loss', 'specific', 'due to']):
                    score += 15
                    matched_keywords.append('HINT:specific_item')
            
            if score > 0:
                is_from_table = 'table[' in context
                results.append({
                    'value': entity_value,
                    'text': entity_text,
                    'score': score,
                    'matched': matched_keywords,
                    'context': context,
                    'node_id': node_id,
                    'label': entity_label,
                    'source': 'table' if is_from_table else 'text',
                    'exact_matches': exact_matches
                })
        
        # Sort by score, exact matches, abs value
        results.sort(key=lambda x: (
            x['score'], 
            x.get('exact_matches', 0),
            abs(x['value']) if isinstance(x['value'], (int, float)) else 0
        ), reverse=True)
        
        if results:
            print(f"  Found {len(results)} entities")
            for i, r in enumerate(results[:3]):
                print(f"    {i+1}. [{r['source']}] value={r['value']}, score={r['score']}, matched={r['matched'][:3]}")
        
        return results
    
    def _retrieve_part_whole_semantic(self,
                                     qa: QuestionAnalysis,
                                     kg: nx.MultiDiGraph,
                                     entity_index: Dict) -> Dict[str, Any]:
        """
        NEW SEMANTIC APPROACH: Retrieve part and whole using semantic similarity
        
        Instead of keyword matching, compute semantic similarity between:
        - part_text vs all table row contexts
        - whole_text vs all table row contexts
        
        Select the entity whose context is MOST SEMANTICALLY SIMILAR to part_text/whole_text
        
        Args:
            qa: QuestionAnalysis with argument_structure
            kg: Knowledge graph
            entity_index: Entity index
            
        Returns:
            Dict with 'part' and 'whole' arguments
        """
        print(f"\n  [SEMANTIC PART/WHOLE RETRIEVAL]")
        
        # Get part and whole text from argument_structure
        part_info = qa.argument_structure.get('part', {})
        whole_info = qa.argument_structure.get('whole', {})
        
        part_text = part_info.get('raw_text', '')
        whole_text = whole_info.get('raw_text', '')
        
        if not part_text or not whole_text:
            print(f"    ERROR: Missing part_text or whole_text!")
            return {}
        
        print(f"    Part text: '{part_text}'")
        print(f"    Whole text: '{whole_text}'")
        
        # Use semantic retriever if available
        if self.semantic_retriever:
            print(f"    Using semantic retriever...")
            
            # Retrieve for part
            part_matches = self.semantic_retriever.retrieve_for_question(
                part_text,
                kg,
                top_k=5,
                temporal_filter=qa.temporal_constraint
            )
            
            # Retrieve for whole
            whole_matches = self.semantic_retriever.retrieve_for_question(
                whole_text,
                kg,
                top_k=5,
                temporal_filter=qa.temporal_constraint
            )
            
            if not part_matches or not whole_matches:
                print(f"    ERROR: No semantic matches found!")
                return {}
            
            # Get top match for each, ensuring they're DIFFERENT
            part_match = part_matches[0]
            whole_match = whole_matches[0]
            
            # If they're the same entity, try next candidate
            if part_match.node_id == whole_match.node_id:
                print(f"    WARNING: Part and whole matched same entity! Trying alternatives...")
                if len(whole_matches) > 1:
                    whole_match = whole_matches[1]
                elif len(part_matches) > 1:
                    part_match = part_matches[1]
            
            # Ensure part < whole (swap if needed)
            if part_match.value >= whole_match.value:
                print(f"    Swapping: part={part_match.value} >= whole={whole_match.value}")
                part_match, whole_match = whole_match, part_match
            
            print(f"    OK Part: {part_match.value} (semantic_score={part_match.semantic_score:.3f})")
            print(f"    OK Whole: {whole_match.value} (semantic_score={whole_match.semantic_score:.3f})")
            
            return {
                'part': self._semantic_match_to_dict(part_match),
                'whole': self._semantic_match_to_dict(whole_match)
            }
        
        # Fallback: Use keyword-based if no semantic retriever
        print(f"    WARNING: No semantic retriever available, using keyword fallback...")
        return {}
    
    def _retrieve_part_whole_different_rows(self,
                                           qa: QuestionAnalysis,
                                           kg: nx.MultiDiGraph,
                                           entity_index: Dict) -> Dict[str, Any]:
        """
        PHASE 1.1: SEMANTIC ENTITY MATCHER - Replace keyword matching with embeddings!
        
        This is the ROOT CAUSE fix for 44% failures (19/43)!
        
        OLD APPROACH (FAILED):
        - Keyword matching: "investments" matches both "available-for-sale" and "total"
        - Result: Wrong entities retrieved
        
        NEW APPROACH (SEMANTIC):
        - Use sentence-transformers to compute semantic similarity
        - "available-for-sale investments" ≈ "available-for-sale investments" (0.9+)
        - "available-for-sale investments" ≉ "total investments" (0.4-)
        
        Strategy:
        1. Get entity candidates from KG (all numeric entities)
        2. Use semantic matcher to find part (most similar to part phrase)
        3. Use semantic matcher to find whole (most similar to whole phrase)
        4. Validate: different rows, part < whole, semantically distinct
        
        Args:
            qa: QuestionAnalysis with argument_structure
            kg: Knowledge graph
            entity_index: Entity index
            
        Returns:
            Dict with 'part' and 'whole' from DIFFERENT rows
        """
        print(f"\n  [SEMANTIC MATCHER] Using embedding-based retrieval...")
        
        part_info = qa.argument_structure.get('part', {})
        whole_info = qa.argument_structure.get('whole', {})
        
        part_keywords = part_info.get('keywords', [])
        whole_keywords = whole_info.get('keywords', [])
        
        if not part_keywords or not whole_keywords:
            print(f"    ERROR: Missing keywords!")
            return {}
        
        # Build query phrases from keywords
        part_phrase = ' '.join(part_keywords)
        whole_phrase = ' '.join(whole_keywords)
        
        print(f"    Part phrase: '{part_phrase}'")
        print(f"    Whole phrase: '{whole_phrase}'")
        
        # Step 1: Collect all numeric entity candidates from KG
        candidates = []
        for node_id, data in kg.nodes(data=True):
            if data.get('label') in ['NUMBER', 'PERCENT', 'MONEY']:
                # Skip DATE entities
                if data.get('label') == 'DATE':
                    continue
                    
                try:
                    value = float(data.get('value', 0))
                    text = data.get('text', '')
                    context = data.get('context', '')
                    
                    candidates.append({
                        'node_id': node_id,
                        'value': value,
                        'text': text,
                        'context': context,
                        'label': data.get('label', ''),
                    })
                except (ValueError, TypeError):
                    continue
        
        if not candidates:
            print(f"    ERROR: No numeric entities in KG!")
            return {}
        
        print(f"    Found {len(candidates)} numeric entity candidates")
        
        # Step 2: Apply temporal filter if specified (FIXED: exact year matching)
        if qa.temporal_constraint:
            import re
            
            # Extract year from temporal constraint (e.g., "2012" from "december 2012")
            # PROBLEM: "dec 292012" has day+year without space!
            # SOLUTION: Find ALL 4-digit sequences matching 19xx/20xx pattern
            year_pattern = r'(19\d{2}|20\d{2})'
            constraint_years = re.findall(year_pattern, qa.temporal_constraint)
            # Note: This will match "2012" in both "2012" and "292012"
            
            if constraint_years:
                temporal_filtered = []
                for cand in candidates:
                    context = cand['context']
                    # Extract years from context (works for "dec292012" too!)
                    context_years = re.findall(year_pattern, context)
                    
                    # Check if ANY constraint year matches ANY context year EXACTLY
                    if any(cy == cony for cy in constraint_years for cony in context_years):
                        temporal_filtered.append(cand)
                
                if temporal_filtered:
                    print(f"    Filtered to {len(temporal_filtered)} candidates with exact year match: {constraint_years}")
                    candidates = temporal_filtered
                else:
                    print(f"    WARNING: No exact year match for {constraint_years}, using all candidates")
            else:
                # Fallback to substring match if no year found
                temporal_filtered = []
                for cand in candidates:
                    if qa.temporal_constraint.lower() in cand['context'].lower():
                        temporal_filtered.append(cand)
                
                if temporal_filtered:
                    print(f"    Filtered to {len(temporal_filtered)} candidates matching '{qa.temporal_constraint}'")
                    candidates = temporal_filtered
        
        # Step 3: Use semantic matcher to find part and whole
        part_match, whole_match = self.semantic_matcher.match_part_and_whole(
            part_text=part_phrase,
            whole_text=whole_phrase,
            all_entities=candidates,
            temporal_constraint=qa.temporal_constraint
        )
        
        if not part_match or not whole_match:
            print(f"    ERROR: Semantic matcher returned no results!")
            return {}
        
        # Step 4: Validate and format results
        print(f"    Part match: value={part_match.get('value')}")
        print(f"      Context: {part_match.get('context', '')[:80]}...")
        print(f"    Whole match: value={whole_match.get('value')}")
        print(f"      Context: {whole_match.get('context', '')[:80]}...")
        
        # Convert to standard format
        return {
            'part': {
                'value': part_match.get('value', 0),
                'text': part_match.get('text', ''),
                'context': part_match.get('context', ''),
                'score': 90,  # High score for semantic match
                'node_id': part_match.get('node_id', ''),
                'label': part_match.get('label', ''),
                'matched': [part_phrase]
            },
            'whole': {
                'value': whole_match.get('value', 0),
                'text': whole_match.get('text', ''),
                'context': whole_match.get('context', ''),
                'score': 90,
                'node_id': whole_match.get('node_id', ''),
                'label': whole_match.get('label', ''),
                'matched': [whole_phrase]
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
        # CRITICAL FIX: Store kg and entity_index for use in helper methods
        self.kg = kg
        self.entity_index = entity_index
        
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
        
        # CRITICAL FIX: Normalize ALL arguments to consistent format
        normalized_arguments = {}
        for arg_name, arg_value in arguments.items():
            normalized_arguments[arg_name] = self._normalize_argument(arg_value)
        arguments = normalized_arguments
        
        # CRITICAL FIX: Check if we have enough arguments
        print(f"  Retrieved arguments: {list(arguments.keys())}")
        print(f"  Required arguments: {required_args}")
        
        if not arguments:
            print(f"  ERROR No arguments retrieved - using fallback")
            return self._fallback_synthesis(question_analysis, kg, entity_index)
        
        # Check if we have all required arguments
        missing_args = [arg for arg in required_args if arg not in arguments]
        if missing_args:
            print(f"  WARNING️  Missing required arguments: {missing_args}")
            print(f"  Available arguments: {[(k, v['value']) for k, v in arguments.items()]}")
            # Don't fail completely - try to use what we have
            # But reduce confidence
        
        # Validate argument values
        for arg_name, arg_data in arguments.items():
            if arg_data['value'] is None or (isinstance(arg_data['value'], float) and 
                                            (arg_data['value'] == 0.0 or abs(arg_data['value']) > 1e10)):
                print(f"  WARNING️  Suspicious value for {arg_name}: {arg_data['value']}")
        
        # PHASE 4 FIX: If direct_lookup found a percentage, switch to division
        if q_type == 'direct_lookup' and 'numerator' in arguments and 'denominator' in arguments:
            print(f"\n  ⚙️  Switching from direct_lookup to division (percentage-based calculation)")
            template = 'divide(#numerator, #denominator)'
            required_args = ['numerator', 'denominator']
        
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
    
    def _retrieve_with_structure(self,
                                qa: QuestionAnalysis,
                                kg: nx.MultiDiGraph,
                                entity_index: Dict,
                                required_args: List[str]) -> Dict[str, Any]:
        """
        Retrieve arguments using structured question parsing.
        
        Uses qa.argument_structure to identify which keywords map to which argument role.
        Example:
            Question: "what percentage of total revenues were due to debt securities?"
            argument_structure = {
                'whole': {'raw_text': 'total revenues', 'keywords': ['total', 'revenues']},
                'part': {'raw_text': 'debt securities', 'keywords': ['debt', 'securities']}
            }
        
        Args:
            qa: QuestionAnalysis with argument_structure field
            kg: Knowledge graph
            entity_index: Entity index
            required_args: List of required argument names (e.g., ['part', 'whole'])
            
        Returns:
            Dict mapping argument names to retrieved values (normalized format)
        """
        print(f"\n  [_retrieve_with_structure] Processing {len(required_args)} arguments...")
        
        arguments = {}
    
    def _auto_parse_percentage_of(self, question: str) -> Dict:
        """
        CRITICAL FIX: Auto-parse percentage_of questions to extract part/whole
        
        Patterns:
        - "what percentage of X was/were Y" -> whole=X, part=Y
        - "what percent of X due to Y" -> whole=X, part=Y
        - "what percentage of X consisting of Y" -> whole=X, part=Y
        
        Args:
            question: Question text
            
        Returns:
            Dict with 'part' and 'whole' keys containing keywords
        """
        q_lower = question.lower()
        
        # Common split markers for part vs whole (in priority order - more specific first!)
        markers = [
            'was comprised of',
            'were comprised of',
            'was composed of',
            'were composed of',
            'consisting of',
            'due to',
            'associated with',
            'related to',
            'attributable to',
            'from',
        ]
        
        # Find which marker is present
        part_text = ""
        whole_text = ""
        
        for marker in markers:
            if marker in q_lower:
                # Split by marker
                parts = q_lower.split(marker, 1)
                if len(parts) == 2:
                    before_marker = parts[0]
                    after_marker = parts[1]
                    
                    # Text BEFORE marker contains whole (after "percentage of" or "percent of")
                    if ' of ' in before_marker:
                        # Get everything after the LAST "of"
                        whole_part = before_marker.rsplit(' of ', 1)[1]
                        # Clean up (remove "as of", temporal info, etc.)
                        whole_text = whole_part.strip()
                        # Remove temporal markers
                        for temporal_marker in [' as of ', ' in ', ' for ', ' during ', ' at ']:
                            if temporal_marker in whole_text:
                                whole_text = whole_text.split(temporal_marker)[0].strip()
                    
                    # Text AFTER marker contains part
                    part_text = after_marker.strip(' ?.')
                    # Remove trailing question marks, periods
                    part_text = part_text.rstrip('?.')
                    
                    break
        
        # If no marker found, try simpler pattern: "what percent of X" where X is whole
        if not whole_text:
            if ' of ' in q_lower:
                after_of = q_lower.rsplit(' of ', 1)[1]
                # Remove temporal and question words
                for stop in [' in ', ' for ', ' during ', ' at ', ' as of ', '?', ' was ', ' were ']:
                    if stop in after_of:
                        after_of = after_of.split(stop)[0]
                whole_text = after_of.strip()
        
        if not whole_text or not part_text:
            return {}
        
        # Extract keywords from each
        import re
        def extract_keywords(text):
            # Remove common stop words
            stopwords = {'a', 'an', 'the', 'as', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'was', 'were', 'is', 'are'}
            words = re.findall(r'\b[a-z]+\b', text)
            keywords = [w for w in words if w not in stopwords and len(w) > 2]
            # Limit to avoid noise
            return keywords[:6]  # Max 6 keywords
        
        whole_keywords = extract_keywords(whole_text)
        part_keywords = extract_keywords(part_text)
        
        if not whole_keywords or not part_keywords:
            return {}
        
        return {
            'whole': {
                'raw_text': whole_text,
                'keywords': whole_keywords
            },
            'part': {
                'raw_text': part_text,
                'keywords': part_keywords
            }
        }
    
    def _retrieve_with_structure(self,
                                qa: QuestionAnalysis,
                                kg: nx.MultiDiGraph,
                                entity_index: Dict,
                                required_args: List[str]) -> Dict[str, Any]:
        """
        Retrieve arguments using structured question parsing.
        
        Uses qa.argument_structure to identify which keywords map to which argument role.
        Example:
            Question: "what percentage of total revenues were due to debt securities?"
            argument_structure = {
                'whole': {'raw_text': 'total revenues', 'keywords': ['total', 'revenues']},
                'part': {'raw_text': 'debt securities', 'keywords': ['debt', 'securities']}
            }
        
        Args:
            qa: QuestionAnalysis with argument_structure field
            kg: Knowledge graph
            entity_index: Entity index
            required_args: List of required argument names (e.g., ['part', 'whole'])
            
        Returns:
            Dict mapping argument names to retrieved values (normalized format)
        """
        print(f"\n  [_retrieve_with_structure] Processing {len(required_args)} arguments...")
        
        # CRITICAL NEW APPROACH: For percentage_of/ratio, ensure part and whole from DIFFERENT ROWS
        if qa.question_type in ['percentage_of', 'ratio'] and set(required_args) == {'part', 'whole'}:
            return self._retrieve_part_whole_different_rows(qa, kg, entity_index)
        
        arguments = {}
        used_node_ids = set()  # Track used nodes to AVOID DUPLICATES
        
        # CRITICAL OPTIMIZATION: Remove common keywords from both sets
        # If "investments" appears in both part and whole, it won't help distinguish them!
        if len(required_args) >= 2:
            all_arg_keywords = {}
            for arg_name in required_args:
                if arg_name in qa.argument_structure:
                    arg_info = qa.argument_structure[arg_name]
                    if isinstance(arg_info, dict):
                        all_arg_keywords[arg_name] = set(arg_info.get('keywords', []))
            
            # Find common keywords across ALL arguments
            if len(all_arg_keywords) >= 2:
                common = set.intersection(*all_arg_keywords.values())
                if common:
                    print(f"\n  [DEDUPE] Removing common keywords from all: {common}")
                    # Remove common keywords from argument_structure
                    for arg_name, keywords_set in all_arg_keywords.items():
                        unique_keywords = keywords_set - common
                        if unique_keywords:  # Only if there are unique keywords left!
                            qa.argument_structure[arg_name]['keywords'] = list(unique_keywords)
                            print(f"    {arg_name}: {qa.argument_structure[arg_name]['keywords']}")
        
        # Process each required argument
        for arg_name in required_args:
            print(f"\n    Processing argument: {arg_name}")
            
            # Get keywords for this argument from structure
            if arg_name not in qa.argument_structure:
                print(f"      X No structure info for '{arg_name}'")
                continue
            
            arg_info = qa.argument_structure[arg_name]
            if not isinstance(arg_info, dict):
                print(f"      X Invalid structure format for '{arg_name}': {type(arg_info)}")
                continue
            
            keywords = arg_info.get('keywords', [])
            raw_text = arg_info.get('raw_text', '')
            
            if not keywords:
                print(f"      X No keywords for '{arg_name}'")
                continue
            
            print(f"      Keywords: {keywords}")
            print(f"      Raw text: '{raw_text}'")
            
            # Add temporal constraint to column keywords if available
            col_keywords = []
            if qa.temporal_constraint:
                temporal = qa.temporal_constraint
                print(f"      Temporal constraint: {temporal}")
                col_keywords.append(temporal)
            
            # CRITICAL FIX: Pass arg_name to scoring so 'total' vs 'part' can be distinguished
            # Add row_type hint to retrieval
            row_type_hint = None
            if arg_name in ['whole', 'denominator']:
                row_type_hint = 'prefer_total'  # Look for 'total', 'sum', 'all'
            elif arg_name in ['part', 'numerator']:
                row_type_hint = 'avoid_total'  # Avoid 'total', prefer specific items
            
            # Retrieve from table using row keywords + optional column keywords
            results = self._retrieve_from_table_with_hint(
                kg, 
                row_keywords=keywords,
                col_keywords=col_keywords if col_keywords else None,
                exclude_labels=['DATE'],  # Don't match year entities as values
                row_type_hint=row_type_hint,
                exclude_node_ids=used_node_ids  # CRITICAL: Avoid re-using same node
            )
            
            if not results:
                print(f"      X No results found for '{arg_name}'")
                continue
            
            # SPECIAL HANDLING FOR SUM/AVERAGE: Retrieve MULTIPLE values
            if arg_name == 'values' and qa.question_type in ['sum', 'average']:
                print(f"      SUM/AVERAGE detected - retrieving multiple values...")
                
                # Take top N results (up to 10) that match the keywords
                multiple_values = []
                for i, result in enumerate(results[:10]):
                    # Skip if value is too similar to already selected (within 1%)
                    if any(abs(result['value'] - v['value']) / max(abs(v['value']), 1) < 0.01 
                          for v in multiple_values):
                        continue
                    
                    # Skip if already used
                    if result.get('node_id') in used_node_ids:
                        continue
                    
                    multiple_values.append({
                        'value': result['value'],
                        'text': result.get('text', ''),
                        'context': result.get('context', ''),
                        'score': result['score'],
                        'node_id': result.get('node_id', ''),
                        'label': result.get('label', ''),
                        'matched': result.get('matched', keywords)
                    })
                    used_node_ids.add(result.get('node_id', ''))
                    
                    if len(multiple_values) >= 5:  # Limit to 5 values max
                        break
                
                if len(multiple_values) >= 2:
                    print(f"      OK Retrieved {len(multiple_values)} values: {[v['value'] for v in multiple_values]}")
                    arguments[arg_name] = multiple_values  # Store as list!
                    continue
                else:
                    print(f"      WARNING: Only found {len(multiple_values)} values for SUM")
            
            # Take best result (single value case)
            best_match = results[0]
            
            # Normalize to standard format
            normalized = {
                'value': best_match['value'],
                'text': best_match.get('text', ''),
                'context': best_match.get('context', ''),
                'score': best_match['score'],
                'node_id': best_match.get('node_id', ''),
                'label': best_match.get('label', ''),
                'matched': best_match.get('matched', keywords)
            }
            
            arguments[arg_name] = normalized
            used_node_ids.add(best_match.get('node_id', ''))
            print(f"      OK Retrieved {arg_name} = {normalized['value']} (score={normalized['score']})")
            print(f"        Matched keywords: {normalized['matched']}")
            print(f"        Context: {normalized['context'][:80]}...")
        
        print(f"\n  [_retrieve_with_structure] Retrieved {len(arguments)}/{len(required_args)} arguments")
        
        # CRITICAL VALIDATION: For percentage_of, part MUST be < whole
        if qa.question_type == 'percentage_of' and 'part' in arguments and 'whole' in arguments:
            part_val = arguments['part']['value']
            whole_val = arguments['whole']['value']
            
            print(f"\n  [VALIDATION] Checking part/whole relationship...")
            print(f"    part={part_val}, whole={whole_val}")
            
            # If part >= whole, something is wrong - swap them!
            if part_val >= whole_val:
                print(f"    WARNING: part >= whole! Swapping...")
                arguments['part'], arguments['whole'] = arguments['whole'], arguments['part']
                print(f"    After swap: part={arguments['part']['value']}, whole={arguments['whole']['value']}")
            else:
                print(f"    OK: part < whole")
        
        return arguments
    
    def _retrieve_arguments(self,
                           qa: QuestionAnalysis,
                           kg: nx.MultiDiGraph,
                           entity_index: Dict,
                           required_args: List[str]) -> Dict[str, Any]:
        """
        Retrieve arguments từ KG dựa vào:
        - ENHANCED: Semantic retrieval (primary method)
        - Entities mentioned in question
        - Temporal information
        - Numbers in question
        - Argument order logic
        - Context matching (fallback)
        """
        # DEBUG LOGGING
        print(f"\n[DEBUG] Argument Retrieval:")
        print(f"  Question: {qa.question}")
        print(f"  Type: {qa.question_type}")
        print(f"  Required Args: {required_args}")
        print(f"  Entities mentioned: {qa.entities_mentioned}")
        print(f"  Temporal: {qa.temporal_entities}")
        print(f"  Numbers: {qa.numbers_mentioned}")
        
        # NEW: Show structured argument info
        if qa.argument_structure:
            print(f"  Argument Structure:")
            for arg_name, arg_info in qa.argument_structure.items():
                if isinstance(arg_info, dict):
                    print(f"    {arg_name}: {arg_info.get('raw_text', '')} -> keywords: {arg_info.get('keywords', [])}")
        if qa.temporal_constraint:
            print(f"  Temporal Constraint: {qa.temporal_constraint}")
        
        print(f"  Entity Index sizes: by_text={len(entity_index['by_text'])}, "
              f"by_value={len(entity_index['by_value'])}, by_label={len(entity_index['by_label'])}")
        
        arguments = {}
        
        # ========================================================================
        # PHASE -0.5: AUTO-GENERATE STRUCTURE FOR PERCENTAGE_OF (CRITICAL FIX!)
        # ========================================================================
        # Many percentage_of questions don't have argument_structure populated
        # Let's parse it ourselves!
        if qa.question_type == 'percentage_of' and not qa.argument_structure:
            print(f"\n  [AUTO-STRUCTURE] Parsing percentage_of question structure...")
            qa.argument_structure = self._auto_parse_percentage_of(qa.question)
            if qa.argument_structure:
                print(f"  [AUTO-STRUCTURE] Generated: {list(qa.argument_structure.keys())}")
        
        # ========================================================================
        # PHASE -1: STRUCTURED ARGUMENT RETRIEVAL (NEW - Highest Priority)
        # ========================================================================
        if qa.argument_structure and qa.argument_structure:
            print(f"\n  [STRUCTURED RETRIEVAL] Using parsed argument structure...")
            
            structured_args = self._retrieve_with_structure(
                qa, kg, entity_index, required_args
            )
            
            if structured_args and len(structured_args) == len(required_args):
                print(f"  [STRUCTURED] OKOK Retrieved all {len(required_args)} arguments!")
                return structured_args
            elif structured_args:
                print(f"  [STRUCTURED] WARNING️  Partial success. Got {len(structured_args)}/{len(required_args)}")
                arguments.update(structured_args)
            else:
                print(f"  [STRUCTURED] X Failed. Falling back...")
        
        # ========================================================================
        # PHASE 0: SEMANTIC RETRIEVAL (Disabled by default)
        # ========================================================================
        if self.use_semantic_retrieval:
            print(f"\n  [SEMANTIC RETRIEVAL] Starting semantic search...")
            
            # Determine temporal filter
            temporal_filter = None
            if qa.temporal_entities:
                # Use most specific temporal entity
                temporal_filter = max(qa.temporal_entities, key=len)
                print(f"  [SEMANTIC] Using temporal filter: {temporal_filter}")
            
            # Strategy depends on question type
            semantic_args = self._semantic_retrieval_strategy(
                qa, kg, required_args, temporal_filter
            )
            
            if semantic_args:
                print(f"  [SEMANTIC] OK Retrieved {len(semantic_args)}/{len(required_args)} arguments semantically")
                
                # Check if we got all required arguments
                if len(semantic_args) == len(required_args):
                    print(f"  [SEMANTIC] OKOK Complete! Using semantic results")
                    return semantic_args
                else:
                    print(f"  [SEMANTIC] WARNING️  Incomplete. Missing: {set(required_args) - set(semantic_args.keys())}")
                    # Keep partial results and try fallback for missing args
                    arguments.update(semantic_args)
            else:
                print(f"  [SEMANTIC] X No semantic results. Falling back to keyword matching...")
        
        # ========================================================================
        # FALLBACK: Original keyword-based retrieval
        # ========================================================================
        
        # PHASE 2.1: TRY TABLE-AWARE RETRIEVAL FIRST
        # For questions mentioning specific years/dates with financial concepts
        key_entities = [e for e in qa.entities_mentioned 
                      if not e.isdigit() and len(e) > 2]
        
        if qa.temporal_entities and key_entities:
            # Try table query with keywords
            table_keywords = key_entities + qa.temporal_entities
            table_values = self._retrieve_from_table(kg, table_keywords)
            
            if table_values:
                print(f"  OK Table query found {len(table_values)} values")
                # CRITICAL FIX: Use full table_values dict (includes context),
                # not just values
                # table_values is already list of dicts with 'value', 'text', 'score', 'context', 'matched'
                # Just boost their scores
                for tv in table_values[:10]:
                    tv['score'] = 100  # Boost table values
                    tv['type'] = 'table'
                all_candidates = table_values[:10]
                
                # Assign arguments based on question type
                if qa.question_type == 'percentage_change' and len(table_values) >= 2:
                    # For percentage_change: try to find old and new values
                    # Use temporal ordering
                    if len(qa.temporal_entities) >= 2:
                        sorted_temporal = sorted(qa.temporal_entities)
                        # Query specifically for each year
                        # CRITICAL FIX: Exclude DATE labels to filter out year values (2007, 2008)
                        # when looking for financial amounts (revenues, expenses, etc.)
                        old_values = self._retrieve_from_table(kg, key_entities + [sorted_temporal[0]], 
                                                              exclude_labels=['DATE'])
                        new_values = self._retrieve_from_table(kg, key_entities + [sorted_temporal[-1]],
                                                              exclude_labels=['DATE'])
                        
                        if old_values and new_values:
                            # CRITICAL FIX: old_values and new_values are dicts with 'value', 'context', etc.
                            # Not just values!
                            arguments['old'] = old_values[0] if isinstance(old_values[0], dict) else {
                                'value': old_values[0],
                                'text': f'table_{sorted_temporal[0]}',
                                'score': 100,
                                'context': f'Table query: {sorted_temporal[0]}'
                            }
                            arguments['new'] = new_values[0] if isinstance(new_values[0], dict) else {
                                'value': new_values[0],
                                'text': f'table_{sorted_temporal[-1]}',
                                'score': 100,
                                'context': f'Table query: {sorted_temporal[-1]}'
                            }
                            print(f"  OK Table query assigned: old={arguments['old']['value']} ({sorted_temporal[0]}), "
                                  f"new={arguments['new']['value']} ({sorted_temporal[-1]})")
                            return arguments
                
                # For other types, continue with context-based
                print(f"  -> Continuing with context-based retrieval...")
        
        # PHASE 2.2: UNIVERSAL CONTEXT-BASED RETRIEVAL
        # Works for all question types by scoring candidates based on context match
        
        print(f"\n  [Context-Based Retrieval]")
        print(f"  Key entities: {key_entities}")
        
        # Collect all candidates with scoring
        all_candidates = []
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') in ['entity', 'cell'] and node_data.get('value') is not None:
                entity_label = node_data.get('label', '')
                
                # PHASE 3 FIX: Filter out DATE entities for most question types
                # (except percentage_change which has special handling)
                if qa.question_type != 'percentage_change' and entity_label == 'DATE':
                    continue
                
                context = node_data.get('context', '').lower()
                
                # IMPROVED SCORING LOGIC
                match_score = 0
                
                # Base score: key entity matches
                for entity in key_entities:
                    if entity.lower() in context:
                        match_score += 10
                        
                        # Proximity bonus
                        value_str = str(node_data['value'])
                        if value_str in context:
                            entity_pos = context.find(entity.lower())
                            value_pos = context.find(value_str)
                            if abs(entity_pos - value_pos) < 50:
                                match_score += 5
                
                # Temporal match bonus
                for temporal in qa.temporal_entities:
                    if temporal in context:
                        match_score += 2
                
                # PHASE 3 FIX: Enhanced label-based scoring
                # Financial term label bonus
                if entity_label in ['EXPENSE', 'REVENUE', 'INCOME', 'MONEY', 'EQUITY', 'ASSET']:
                    match_score += 5  # Increased from 3 to 5
                
                # Penalize PERCENT for non-percentage questions
                if entity_label == 'PERCENT' and qa.question_type not in ['percentage_of', 'percentage_change']:
                    match_score -= 10
                
                if match_score > 0:
                    all_candidates.append({
                        'value': node_data['value'],
                        'text': node_data.get('text', str(node_data['value'])),
                        'node_id': node_id,
                        'context': context,
                        'score': match_score,
                        'type': node_data.get('type'),
                        'label': node_data.get('label', '')
                    })
        
        print(f"  Found {len(all_candidates)} candidates")
        
        # Sort candidates by score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if all_candidates:
            print(f"  Top 5 candidates:")
            for i, cand in enumerate(all_candidates[:5]):
                print(f"    {i+1}. value={cand['value']}, score={cand['score']}, "
                      f"label={cand.get('label')}, context={cand['context'][:80]}...")
        
        # ASSIGN ARGUMENTS BASED ON QUESTION TYPE
        if qa.question_type == 'direct_lookup':
            # Single value needed
            if all_candidates and 'value' in required_args:
                # PHASE 4 FIX: Check if the value has an associated percentage
                # If question asks for "total X" but table shows "Y (Z% of total X)",
                # we need to calculate: Y / Z% = total X
                candidate = all_candidates[0]
                
                # Look for percentage in same context (same row in table)
                percentage_value = self._find_associated_percentage(
                    kg, candidate, qa
                )
                
                if percentage_value:
                    # Found percentage - need division instead of direct lookup
                    print(f"  WARNING️  Found associated percentage: {percentage_value}%")
                    print(f"  Converting direct_lookup to division: {candidate['value']} / {percentage_value}%")
                    
                    # Change to division template
                    arguments['numerator'] = candidate
                    arguments['denominator'] = {
                        'value': percentage_value / 100.0,  # Convert percentage to decimal
                        'text': f'{percentage_value}%',
                        'score': 100,
                        'context': f'Percentage from same row as {candidate["value"]}'
                    }
                    print(f"  Calculated denominator: {arguments['denominator']['value']}")
                else:
                    # Normal direct lookup
                    arguments['value'] = candidate
                    print(f"  Selected for 'value': {arguments['value']['value']}")
        
        elif qa.question_type in ['percentage_of', 'ratio']:
            # Need 2 values: part/whole or numerator/denominator
            if len(all_candidates) >= 2:
                # PHASE 3 FIX: Enhanced temporal filtering
                # Use qa.temporal_constraint (e.g., "december 2012") instead of temporal_entities
                # because temporal_entities can be incomplete (e.g., ['dec .'])
                temporal_text = None
                if qa.temporal_constraint:
                    temporal_text = qa.temporal_constraint
                elif qa.temporal_entities:
                    # Fallback: use the longest temporal entity
                    sorted_temporal = sorted(qa.temporal_entities, key=len, reverse=True)
                    temporal_text = sorted_temporal[0]
                
                if temporal_text:
                    # CRITICAL FIX: EXACT YEAR MATCHING (not substring!)
                    # Extract year from temporal: "dec 29 2012" -> "2012"
                    # PROBLEM: "dec 292012" has day+year without space!
                    # SOLUTION: Find ALL 4-digit sequences matching 19xx/20xx pattern
                    import re
                    # Match any 4-digit year (19xx or 20xx) - will match in "292012"
                    year_pattern = r'(19\d{2}|20\d{2})'
                    temporal_years = re.findall(year_pattern, temporal_text)
                    # Note: This will match "2012" in both "2012" and "292012"
                    
                    if temporal_years:
                        # Filter candidates by EXACT year match
                        relevant = []
                        for c in all_candidates:
                            context = c['context']
                            # Extract years from context (works for "dec292012" too!)
                            context_years = re.findall(year_pattern, context)
                            
                            # Check if ANY temporal year matches ANY context year EXACTLY
                            if any(ty == cy for ty in temporal_years for cy in context_years):
                                relevant.append(c)
                        
                        if len(relevant) >= 2:
                            all_candidates = relevant
                            print(f"  ✓ Filtered to {len(relevant)} candidates with EXACT year match: {temporal_years}")
                        else:
                            print(f"  ⚠ No exact year match for {temporal_years}, using all {len(all_candidates)} candidates")
                    else:
                        # Fallback: normalized substring matching if no year found
                        normalized_temporal = re.sub(r'[\s\.\,\-]', '', temporal_text.lower())
                        
                        if len(normalized_temporal) > 2:  # Avoid matching '20'
                            relevant = []
                            for c in all_candidates:
                                normalized_context = re.sub(r'[\s\.\,\-]', '', c['context'].lower())
                                if normalized_temporal in normalized_context:
                                    relevant.append(c)
                            
                            if len(relevant) >= 2:
                                all_candidates = relevant
                                print(f"  Filtered to {len(relevant)} candidates matching '{temporal_text}'")
                
                # PHASE 3 FIX: Enhanced assignment with context-based matching
                if 'part' in required_args and 'whole' in required_args:
                    # IMPROVED: Use key_entities to match 'part' candidate
                    # E.g., question "percent of net earnings to net cash"
                    #   -> key_entities=['earnings'] -> match context with "earnings" = part
                    #   -> remaining candidates -> find one with "total/cash/operating" = whole
                    
                    # CRITICAL: Filter out temporal entities from key_entities
                    # E.g., key_entities=['dec .', 'earnings'] -> use only ['earnings']
                    # because 'dec .' matches ALL candidates!
                    non_temporal_keys = []
                    if key_entities:
                        temporal_words = set()
                        if qa.temporal_entities:
                            temporal_words = set(t.lower() for t in qa.temporal_entities)
                        if qa.temporal_constraint:
                            # Add words from constraint too
                            temporal_words.update(qa.temporal_constraint.lower().split())
                        
                        non_temporal_keys = [
                            k for k in key_entities
                            if k.lower() not in temporal_words and
                               k.lower() not in ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                  'jul', 'aug', 'sep', 'oct', 'nov', '20', '2012', '2013']
                        ]
                    
                    part_candidates = []
                    whole_candidates = []
                    
                    for c in all_candidates:
                        context_lower = c['context'].lower()
                        
                        # STRATEGY 1: Match NON-TEMPORAL key entities to find 'part'
                        # If context contains any non-temporal key entity, it's likely the 'part'
                        has_key_entity = False
                        if non_temporal_keys:
                            has_key_entity = any(
                                key.lower() in context_lower 
                                for key in non_temporal_keys
                            )
                        
                        # STRATEGY 2: Look for 'whole' indicators
                        has_whole_indicator = any(
                            word in context_lower 
                            for word in ['total', 'sum', 'all', 'cash', 'operating']
                        )
                        
                        # Classify candidate
                        if has_key_entity and not has_whole_indicator:
                            # Clear 'part' match (has key entity, no total indicator)
                            part_candidates.append(c)
                        elif has_whole_indicator:
                            # Likely 'whole' (has total/cash/operating indicator)
                            whole_candidates.append(c)
                        elif has_key_entity:
                            # Has key but also has whole indicator - could be either
                            part_candidates.append(c)
                            whole_candidates.append(c)
                        elif non_temporal_keys:
                            # Has no key entity and no whole indicator
                            # But we HAVE non-temporal keys to match
                            # This is likely NOT the part we want
                            pass  # Don't add
                        else:
                            # No non-temporal keys available (e.g., only 'dec .')
                            # Fallback: distinguish by whole indicator ONLY
                            if has_whole_indicator:
                                whole_candidates.append(c)
                            else:
                                part_candidates.append(c)
                    
                    # If we found clear part/whole distinction, use it
                    if part_candidates and whole_candidates:
                        # Take highest-scoring from each group
                        arguments['part'] = part_candidates[0]
                        arguments['whole'] = whole_candidates[0]
                        print(f"  Keyword-based: 'part': {arguments['part']['value']}, 'whole': {arguments['whole']['value']}")
                    else:
                        # Fallback: Size-based heuristic
                        # But filter out extreme outliers first
                        values = [c['value'] for c in all_candidates[:10]]
                        median_val = sorted(values)[len(values)//2]
                        # Remove values < 10% of median (likely erroneous)
                        filtered = [c for c in all_candidates[:10] if c['value'] >= median_val * 0.1]
                        
                        sorted_by_value = sorted(filtered, key=lambda x: x['value'])
                        arguments['part'] = sorted_by_value[0]
                        arguments['whole'] = sorted_by_value[-1]
                        print(f"  Size-based: 'part': {arguments['part']['value']}, 'whole': {arguments['whole']['value']}")
                
                elif 'numerator' in required_args and 'denominator' in required_args:
                    # Similar logic
                    arguments['numerator'] = all_candidates[0]
                    arguments['denominator'] = all_candidates[1] if len(all_candidates) > 1 else all_candidates[0]
                    print(f"  Selected 'numerator': {arguments['numerator']['value']}, 'denominator': {arguments['denominator']['value']}")
        
        elif qa.question_type == 'comparison':
            # Need 2 values to compare
            if len(all_candidates) >= 2:
                # Try to identify which values to compare based on question context
                # For "did X exceed Y", find X and Y
                arguments['value1'] = all_candidates[0]
                arguments['value2'] = all_candidates[1]
                print(f"  Comparison: value1={arguments['value1']['value']}, value2={arguments['value2']['value']}")
        
        elif qa.question_type == 'difference':
            # ENHANCED: Strict temporal matching for difference questions
            # Need 2 DIFFERENT values, preferably from different time periods
            if len(qa.temporal_entities) >= 2 and len(all_candidates) >= 2:
                sorted_temporal = sorted(qa.temporal_entities)
                print(f"  Temporal ordering: {sorted_temporal[0]} (earlier) -> {sorted_temporal[-1]} (later)")
                
                # Find candidates matching each temporal period
                earlier_candidates = [c for c in all_candidates if sorted_temporal[0] in c['context']]
                later_candidates = [c for c in all_candidates if sorted_temporal[-1] in c['context']]
                
                if earlier_candidates and later_candidates:
                    # Use values from different periods
                    val1 = later_candidates[0]
                    val2 = earlier_candidates[0]
                    
                    # Validation: Must be DIFFERENT values!
                    if abs(val1['value'] - val2['value']) < 0.01:
                        print(f"  WARNING: Values too similar ({val1['value']} vs {val2['value']}), trying alternatives...")
                        # Try next candidates
                        if len(later_candidates) > 1:
                            val1 = later_candidates[1]
                        elif len(earlier_candidates) > 1:
                            val2 = earlier_candidates[1]
                    
                    # Determine larger and smaller
                    if val1['value'] > val2['value']:
                        arguments['larger'] = val1
                        arguments['smaller'] = val2
                    else:
                        arguments['larger'] = val2
                        arguments['smaller'] = val1
                    
                    print(f"  Selected 'larger': {arguments['larger']['value']}, 'smaller': {arguments['smaller']['value']}")
                else:
                    print(f"  WARNING: Could not match temporal periods, using top 2 candidates")
                    if len(all_candidates) >= 2:
                        # Size-based fallback
                        sorted_by_val = sorted(all_candidates[:5], key=lambda x: x['value'], reverse=True)
                        arguments['larger'] = sorted_by_val[0]
                        arguments['smaller'] = sorted_by_val[1]
            elif len(all_candidates) >= 2:
                # No temporal info - use largest and smallest
                sorted_by_val = sorted(all_candidates[:5], key=lambda x: x['value'], reverse=True)
                arguments['larger'] = sorted_by_val[0]
                arguments['smaller'] = sorted_by_val[-1]
                print(f"  Size-based difference: {arguments['larger']['value']} - {arguments['smaller']['value']}")
        
        elif qa.question_type == 'percentage_change':
            # Need 2 values with temporal ordering: old and new
            if len(qa.temporal_entities) >= 2 and len(all_candidates) >= 2:
                sorted_temporal = sorted(qa.temporal_entities)
                print(f"  Temporal ordering: {sorted_temporal[0]} (old) -> {sorted_temporal[-1]} (new)")
                
                # Find candidates matching each temporal
                old_candidates = [c for c in all_candidates if sorted_temporal[0] in c['context']]
                new_candidates = [c for c in all_candidates if sorted_temporal[-1] in c['context']]
                
                if old_candidates and new_candidates:
                    arguments['old'] = old_candidates[0]
                    arguments['new'] = new_candidates[0]
                    print(f"  Selected 'old': {arguments['old']['value']} ({sorted_temporal[0]}), "
                          f"'new': {arguments['new']['value']} ({sorted_temporal[-1]})")
                elif len(all_candidates) >= 2:
                    # Fallback: use top 2
                    arguments['old'] = all_candidates[1]
                    arguments['new'] = all_candidates[0]
                    print(f"  Fallback - 'old': {arguments['old']['value']}, 'new': {arguments['new']['value']}")
        
        # If we got arguments, return them
        if arguments:
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
        
        CRITICAL FIX: Must replace #arg_name with #0, #1, #2... (NOT with actual values)
        The executor needs placeholders like #0, #1 to look up values in the placeholder dict
        
        ENHANCEMENT: Handle list of values for SUM/AVERAGE
        """
        program = template
        placeholders = {}
        placeholder_idx = 0
        
        # SPECIAL HANDLING FOR SUM/AVERAGE WITH MULTIPLE VALUES
        if qa.question_type in ['sum', 'average'] and 'values' in arguments:
            arg_data = arguments['values']
            
            # Check if it's a list of values
            if isinstance(arg_data, list) and len(arg_data) > 1:
                print(f"  [Program Gen] Handling {len(arg_data)} values for {qa.question_type}")
                
                # Generate add(#0, #1, #2, ...) for sum
                # or add(#0, #1, #2, ...), divide(#0, count) for average
                value_placeholders = []
                for i, val_dict in enumerate(arg_data):
                    ph = f"#{placeholder_idx}"
                    value_placeholders.append(ph)
                    placeholders[ph] = {
                        'value': val_dict['value'],
                        'text': val_dict.get('text', str(val_dict['value'])),
                        'node_id': val_dict.get('node_id', ''),
                        'context': val_dict.get('context', '')
                    }
                    placeholder_idx += 1
                
                # Build program
                if qa.question_type == 'sum':
                    # add(#0, #1, #2, ...)
                    program = f"add({', '.join(value_placeholders)})"
                elif qa.question_type == 'average':
                    # add(#0, #1, #2, ...), divide(#0, count)
                    add_step = f"add({', '.join(value_placeholders)})"
                    count = len(arg_data)
                    program = f"{add_step}, divide(#{placeholder_idx}, {count})"
                    # Note: #placeholder_idx will refer to result of add (first step)
                
                return program, placeholders
        
        # STANDARD CASE: Replace placeholders trong template với #0, #1, #2...
        for arg_name, arg_data in arguments.items():
            placeholder_pattern = f"#{arg_name}"
            if placeholder_pattern in program:
                # CRITICAL FIX: Replace with #0, #1, ... (not with actual value)
                numbered_placeholder = f"#{placeholder_idx}"
                program = program.replace(placeholder_pattern, numbered_placeholder)
                
                # Handle both dict and list
                if isinstance(arg_data, dict):
                    # Store mapping: #0 -> actual value
                    placeholders[numbered_placeholder] = {
                        'value': arg_data['value'],
                        'text': arg_data.get('text', str(arg_data['value'])),
                        'node_id': arg_data.get('node_id', ''),
                        'context': arg_data.get('context', '')
                    }
                elif isinstance(arg_data, list):
                    # List case (shouldn't happen in standard flow, handled above)
                    placeholders[numbered_placeholder] = {
                        'value': [v['value'] for v in arg_data if isinstance(v, dict)],
                        'text': 'multiple values',
                        'node_id': '',
                        'context': ''
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
    
    def _find_associated_percentage(self,
                                    kg: nx.MultiDiGraph,
                                    candidate: Dict,
                                    qa: QuestionAnalysis) -> float | None:
        """
        PHASE 4: Find percentage associated with a value in the same row.
        
        This handles cases like:
        - Question: "what was the total operating expenses in 2018?"
        - Table shows: "fuel expense: $9896 (23.6% of total operating expenses)"
        - Need to calculate: 9896 / 0.236 = 41932 (the total)
        
        Args:
            kg: Knowledge graph
            candidate: The candidate dict with 'value', 'context', 'node_id'
            qa: Question analysis
            
        Returns:
            Percentage value (e.g., 23.6), or None if not found
        """
        # Strategy: Look for PERCENT entities in the same row
        # Check if contexts share the same "Row: X" marker
        
        context = candidate.get('context', '').lower()
        value_str = str(candidate['value'])
        
        # Extract row identifier from context (e.g., "Row: 2018")
        import re
        row_match = re.search(r'row:\s*([^|]+)', context)
        if not row_match:
            return None  # No row info, can't match
        
        candidate_row = row_match.group(1).strip()
        
        # Look for PERCENT entities in the same row
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') not in ['entity', 'cell']:
                continue
                
            # Check if it's a PERCENT entity
            if node_data.get('label') != 'PERCENT':
                continue
            
            # Get this node's context
            node_context = node_data.get('context', '').lower()
            
            # Check if same row
            node_row_match = re.search(r'row:\s*([^|]+)', node_context)
            if node_row_match:
                node_row = node_row_match.group(1).strip()
                
                if node_row == candidate_row:
                    # Same row! This is likely the associated percentage
                    percentage = node_data.get('value')
                    if percentage:
                        print(f"    OK Found percentage {percentage}% in same row (Row: {candidate_row})")
                        return percentage
        
        return None
    
    def _contexts_overlap(self, context1: str, context2: str, threshold: float = 0.3) -> bool:
        """Check if two contexts have significant overlap"""
        # Simple word-based overlap
        words1 = set(context1.split())
        words2 = set(context2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        overlap = len(intersection) / len(union) if union else 0
        return overlap >= threshold
    
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
        IMPROVED FALLBACK: Try harder to generate a program
        
        Strategies:
        1. If numbers mentioned in question -> use them
        2. If no numbers in question -> extract from KG
        3. Guess operation based on question keywords
        """
        print(f"\n[FALLBACK] Attempting fallback synthesis...")
        
        # Strategy 1: Use numbers mentioned in question
        if len(qa.numbers_mentioned) >= 2:
            print(f"  Strategy 1: Using numbers from question: {qa.numbers_mentioned[:2]}")
            return ProgramSynthesisResult(
                program=f"divide(#{0}, #{1})",
                placeholders={
                    '#0': {'value': qa.numbers_mentioned[0], 'text': str(qa.numbers_mentioned[0])},
                    '#1': {'value': qa.numbers_mentioned[1], 'text': str(qa.numbers_mentioned[1])}
                },
                explanation="Fallback: Using numbers from question with divide",
                confidence=0.3
            )
        
        # Strategy 2: Extract any numeric entities from KG
        all_numeric = []
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') == 'entity' and node_data.get('value') is not None:
                # Filter out dates
                if node_data.get('label') != 'DATE':
                    all_numeric.append({
                        'value': node_data['value'],
                        'text': node_data.get('text', str(node_data['value'])),
                        'context': node_data.get('context', '')
                    })
        
        if len(all_numeric) >= 2:
            print(f"  Strategy 2: Found {len(all_numeric)} numeric entities in KG")
            # Sort by value, take smallest and largest (often part/whole)
            sorted_nums = sorted(all_numeric, key=lambda x: x['value'])
            
            # Guess operation based on question
            operation = 'divide'  # Default
            if any(word in qa.question.lower() for word in ['change', 'increase', 'decrease', 'growth']):
                operation = 'divide(subtract(#0, #1), #1)'  # percentage change
                print(f"  Detected change question -> using percentage_change formula")
                return ProgramSynthesisResult(
                    program=operation,
                    placeholders={
                        '#0': sorted_nums[-1],  # new/larger
                        '#1': sorted_nums[0]    # old/smaller
                    },
                    explanation="Fallback: Detected percentage change from keywords",
                    confidence=0.4
                )
            elif any(word in qa.question.lower() for word in ['percentage', 'percent', 'proportion', 'ratio']):
                operation = 'divide'
                print(f"  Detected percentage/ratio question -> using divide")
            
            return ProgramSynthesisResult(
                program=f"{operation}(#0, #1)",
                placeholders={
                    '#0': sorted_nums[0],  # smaller (numerator)
                    '#1': sorted_nums[-1]  # larger (denominator)
                },
                explanation=f"Fallback: Using {operation} on KG entities",
                confidence=0.35
            )
        
        # Strategy 3: If only one number, try direct lookup
        if len(all_numeric) >= 1:
            print(f"  Strategy 3: Single value direct lookup")
            return ProgramSynthesisResult(
                program="#0",
                placeholders={
                    '#0': all_numeric[0]
                },
                explanation="Fallback: Direct value lookup",
                confidence=0.2
            )
        
        print(f"  ERROR All fallback strategies failed - no numeric entities found")
        return ProgramSynthesisResult(
            program="",
            placeholders={},
            explanation="Could not synthesize program - no numeric entities",
            confidence=0.0
        )
