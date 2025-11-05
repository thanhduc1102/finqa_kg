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
                'template': 'divide(subtract(#new, #old), #old)',
                'description': 'Percentage change = (new - old) / old (in decimal form)',
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
                'description': 'Percentage = part / whole (in decimal form)',
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
    
    def _retrieve_from_table(self, kg: nx.MultiDiGraph, 
                            row_keywords: List[str], 
                            col_keywords: List[str] = None,
                            exclude_labels: List[str] = None) -> List[float]:
        """
        Table-aware retrieval: tìm values từ table based on row/column labels
        
        CRITICAL FIX: Search 'entity' nodes (with proper context), not 'cell' nodes (empty context)
        
        Args:
            kg: Knowledge Graph
            row_keywords: Keywords to match row labels (e.g., ['revenue', '2007'])
            col_keywords: Keywords to match column headers (optional)
            exclude_labels: Entity labels to exclude (e.g., ['DATE'] to filter out years)
            
        Returns:
            List of matching cell values
        """
        print(f"\n  [Table Query] row_keywords={row_keywords}, col_keywords={col_keywords}")
        
        results = []
        exclude_labels = exclude_labels or []
        
        # CRITICAL FIX: Search entity nodes (which have proper context),
        # not cell nodes (which have empty text/context)
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') == 'entity':
                # Filter by label if specified
                entity_label = node_data.get('label', '')
                if entity_label in exclude_labels:
                    continue
                
                # Get entity context - must be from table
                context = node_data.get('context', '').lower()
                
                # CRITICAL: Only consider entities from tables
                if 'table[' not in context:
                    continue
                
                entity_text = node_data.get('text', '').lower()
                entity_value = node_data.get('value')
                
                if entity_value is None:
                    continue
                
                # Score based on keyword matches in text + context
                score = 0
                matched_keywords = []
                
                for keyword in row_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in entity_text or keyword_lower in context:
                        score += 10
                        matched_keywords.append(keyword)
                
                if col_keywords:
                    for keyword in col_keywords:
                        keyword_lower = keyword.lower()
                        if keyword_lower in entity_text or keyword_lower in context:
                            score += 5
                            matched_keywords.append(keyword)
                
                if score > 0:
                    results.append({
                        'value': entity_value,
                        'text': entity_text,
                        'score': score,
                        'matched': matched_keywords,
                        'context': context
                    })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            print(f"  Found {len(results)} table cells")
            for i, r in enumerate(results[:3]):
                print(f"    {i+1}. value={r['value']}, score={r['score']}, "
                      f"matched={r['matched']}, text={r['text'][:60]}")
        
        # CRITICAL FIX: Return full dicts, not just values
        # Each dict has: 'value', 'text', 'score', 'matched', 'context'
        return results
    
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
        # DEBUG LOGGING
        print(f"\n[DEBUG] Argument Retrieval:")
        print(f"  Question: {qa.question}")
        print(f"  Type: {qa.question_type}")
        print(f"  Required Args: {required_args}")
        print(f"  Entities mentioned: {qa.entities_mentioned}")
        print(f"  Temporal: {qa.temporal_entities}")
        print(f"  Numbers: {qa.numbers_mentioned}")
        print(f"  Entity Index sizes: by_text={len(entity_index['by_text'])}, "
              f"by_value={len(entity_index['by_value'])}, by_label={len(entity_index['by_label'])}")
        
        arguments = {}
        
        # PHASE 2.1: TRY TABLE-AWARE RETRIEVAL FIRST
        # For questions mentioning specific years/dates with financial concepts
        key_entities = [e for e in qa.entities_mentioned 
                      if not e.isdigit() and len(e) > 2]
        
        if qa.temporal_entities and key_entities:
            # Try table query with keywords
            table_keywords = key_entities + qa.temporal_entities
            table_values = self._retrieve_from_table(kg, table_keywords)
            
            if table_values:
                print(f"  ✓ Table query found {len(table_values)} values")
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
                            print(f"  ✓ Table query assigned: old={arguments['old']['value']} ({sorted_temporal[0]}), "
                                  f"new={arguments['new']['value']} ({sorted_temporal[-1]})")
                            return arguments
                
                # For other types, continue with context-based
                print(f"  → Continuing with context-based retrieval...")
        
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
                    print(f"  ⚠️  Found associated percentage: {percentage_value}%")
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
                # Use the LONGEST temporal entity (most specific, e.g., 'dec 29 2012' not just '20')
                if qa.temporal_entities:
                    # Sort by length to get most specific temporal
                    sorted_temporal = sorted(qa.temporal_entities, key=len, reverse=True)
                    temporal_text = sorted_temporal[0]
                    
                    # CRITICAL FIX: Normalize temporal text for matching
                    # "dec . 29 2012" → "dec292012" (remove spaces and punctuation)
                    import re
                    normalized_temporal = re.sub(r'[\s\.\,\-]', '', temporal_text.lower())
                    
                    # Filter candidates by temporal match - be more strict
                    # Remove generic matches like '20' which matches '2012', '2013', '2020', etc.
                    relevant = []
                    for c in all_candidates:
                        # Skip if only matches on very short strings like '20'
                        if len(normalized_temporal) <= 2:
                            continue
                        
                        # Normalize context for comparison
                        normalized_context = re.sub(r'[\s\.\,\-]', '', c['context'].lower())
                        
                        if normalized_temporal in normalized_context:
                            relevant.append(c)
                    
                    if len(relevant) >= 2:
                        all_candidates = relevant
                        print(f"  Filtered to {len(relevant)} candidates matching '{temporal_text}' (normalized: '{normalized_temporal}')")
                
                # PHASE 3 FIX: Enhanced assignment with context-based matching
                if 'part' in required_args and 'whole' in required_args:
                    # Try to match question keywords to context
                    # E.g., "available-for-sale" in question → find entity with "available" in row label
                    part_candidates = []
                    whole_candidates = []
                    
                    for c in all_candidates:
                        context_lower = c['context'].lower()
                        # Look for 'part' indicators: specific item names
                        # vs 'whole' indicators: 'total', 'sum', 'all'
                        if any(word in context_lower for word in ['total', 'sum', 'all']):
                            whole_candidates.append(c)
                        else:
                            part_candidates.append(c)
                    
                    # If we found clear part/whole distinction, use it
                    if part_candidates and whole_candidates:
                        # Take highest-scoring from each group
                        arguments['part'] = part_candidates[0]
                        arguments['whole'] = whole_candidates[0]
                        print(f"  Context-based: 'part': {arguments['part']['value']}, 'whole': {arguments['whole']['value']}")
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
        
        elif qa.question_type == 'percentage_change':
            # Need 2 values with temporal ordering: old and new
            if len(qa.temporal_entities) >= 2 and len(all_candidates) >= 2:
                sorted_temporal = sorted(qa.temporal_entities)
                print(f"  Temporal ordering: {sorted_temporal[0]} (old) → {sorted_temporal[-1]} (new)")
                
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
                        print(f"    ✓ Found percentage {percentage}% in same row (Row: {candidate_row})")
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
