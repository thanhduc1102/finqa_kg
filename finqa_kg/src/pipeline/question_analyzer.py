"""
Question Analyzer
Phân tích câu hỏi để xác định:
- Question type (calculation, comparison, percentage, etc)
- Entities mentioned
- Operations needed
- Temporal information
"""

import spacy
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import re

@dataclass
class QuestionAnalysis:
    """Kết quả phân tích question"""
    question: str
    question_type: str  # 'divide', 'multiply', 'subtract', 'add', 'percentage_change', 'ratio', 'average', etc
    operations: List[str]  # Danh sách operations cần thiết
    entities_mentioned: List[str]  # Entities được mention trong question
    numbers_mentioned: List[float]  # Numbers trong question
    temporal_entities: List[str]  # Years, quarters, dates
    keywords: List[str]  # Keywords quan trọng
    argument_order: List[str]  # Thứ tự arguments (important!)

class QuestionAnalyzer:
    """
    Analyze question để determine:
    1. What type of calculation?
    2. Which entities are involved?
    3. What operations are needed?
    4. What order should arguments be?
    """
    
    def __init__(self):
        """Initialize với Spacy và question patterns"""
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except:
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except:
                self.nlp = spacy.load("en_core_web_sm")
        
        # Define question patterns
        self.question_patterns = self._load_question_patterns()
    
    def _load_question_patterns(self) -> Dict[str, Dict]:
        """
        Load patterns để nhận diện question types
        """
        return {
            'percentage_change': {
                'keywords': ['percentage', 'percent', '%', 'growth rate', 'change rate', 
                            'increase rate', 'decrease rate', 'grew', 'declined'],
                'operations': ['subtract', 'divide', 'multiply'],
                'formula': '(new - old) / old * 100',
                'argument_order': ['new', 'old'],  # NEW - OLD!
                'temporal_indicators': ['from', 'to', 'between'],
                'examples': [
                    'What is the percentage change from X to Y?',
                    'What is the growth rate of X?',
                    'By what percent did X increase?'
                ]
            },
            'ratio': {
                'keywords': ['ratio', 'per', 'for each', 'for every', 'divide', 
                            'average per', 'per unit'],
                'operations': ['divide'],
                'formula': 'numerator / denominator',
                'argument_order': ['numerator', 'denominator'],
                'examples': [
                    'What is the ratio of X to Y?',
                    'What is X per Y?',
                    'How much X for each Y?'
                ]
            },
            'average': {
                'keywords': ['average', 'mean', 'typical'],
                'operations': ['divide', 'add'],
                'formula': 'sum / count',
                'examples': [
                    'What is the average X?',
                    'What is the mean X per Y?'
                ]
            },
            'sum': {
                'keywords': ['total', 'sum', 'combined', 'altogether', 'in total'],
                'operations': ['add'],
                'formula': 'a + b + c + ...',
                'examples': [
                    'What is the total X?',
                    'What is the sum of X and Y?'
                ]
            },
            'difference': {
                'keywords': ['difference', 'more than', 'less than', 'exceed', 
                            'greater than', 'lower than', 'change', 'increase', 'decrease'],
                'operations': ['subtract'],
                'formula': 'a - b',
                'argument_order': ['larger', 'smaller'],  # Or temporal order
                'temporal_indicators': ['from', 'to', 'than'],
                'examples': [
                    'What is the difference between X and Y?',
                    'How much more is X than Y?',
                    'What is the change from X to Y?'
                ]
            },
            'product': {
                'keywords': ['product', 'multiply', 'times', 'multiplied by'],
                'operations': ['multiply'],
                'formula': 'a * b',
                'examples': [
                    'What is X multiplied by Y?',
                    'What is the product of X and Y?'
                ]
            },
            'compound': {
                'keywords': ['calculate', 'compute', 'find', 'determine'],
                'operations': ['multiple'],  # Need further analysis
                'examples': [
                    'Calculate X based on Y and Z',
                    'What would be the result if...'
                ]
            },
            'percentage_of': {
                'keywords': ['what percent', 'what percentage', 'is what percent of'],
                'operations': ['divide', 'multiply'],
                'formula': '(part / whole) * 100',
                'argument_order': ['part', 'whole'],
                'examples': [
                    'X is what percent of Y?',
                    'What percentage of Y is X?'
                ]
            },
            'absolute_value': {
                'keywords': ['absolute', 'magnitude'],
                'operations': ['abs'],
                'formula': '|x|',
                'examples': [
                    'What is the absolute value of X?'
                ]
            },
            'comparison': {
                'keywords': ['exceed', 'greater than', 'less than', 'more than', 
                            'higher than', 'lower than', 'did', 'was', 'were',
                            'compare', 'comparison'],
                'operations': ['greater', 'less', 'equal'],
                'formula': 'a > b or a < b or a == b',
                'argument_order': ['value1', 'value2'],
                'examples': [
                    'Did X exceed Y?',
                    'Was X greater than Y?',
                    'Did X increase more than Y?'
                ]
            },
            'direct_lookup': {
                'keywords': ['what is', 'what was', 'what were', 'how much', 'how many',
                            'what are', "what's", 'value of'],
                'operations': ['lookup'],  # No calculation, just retrieve
                'formula': 'Extract from KG',
                'examples': [
                    'What is the X in year Y?',
                    'What was the X?',
                    'How much is X?'
                ]
            },
            'conversion': {
                'keywords': ['basis points', 'convert', 'equivalent'],
                'operations': ['divide', 'multiply'],
                'formula': 'conversion formula',
                'examples': [
                    'What is X basis points?',
                    'Convert X to Y'
                ]
            }
        }
    
    def analyze(self, question: str) -> QuestionAnalysis:
        """
        Main analysis function
        
        Args:
            question: Question string
            
        Returns:
            QuestionAnalysis object
        """
        # Process với Spacy
        doc = self.nlp(question.lower())
        
        # 1. Detect question type
        question_type, matched_pattern = self._detect_question_type(question.lower(), doc)
        
        # 2. Extract entities
        entities = self._extract_entities(doc)
        
        # 3. Extract numbers
        numbers = self._extract_numbers(question)
        
        # 4. Extract temporal entities
        temporal = self._extract_temporal_entities(doc)
        
        # 5. Extract keywords
        keywords = self._extract_keywords(doc, matched_pattern)
        
        # 6. Determine operations
        operations = matched_pattern.get('operations', []) if matched_pattern else []
        
        # 7. Determine argument order
        argument_order = self._determine_argument_order(
            question.lower(), question_type, temporal, entities
        )
        
        return QuestionAnalysis(
            question=question,
            question_type=question_type,
            operations=operations,
            entities_mentioned=entities,
            numbers_mentioned=numbers,
            temporal_entities=temporal,
            keywords=keywords,
            argument_order=argument_order
        )
    
    def _detect_question_type(self, question: str, doc) -> Tuple[str, Dict]:
        """
        Detect question type bằng keyword matching với scoring cải tiến
        """
        best_match = None
        best_score = 0
        
        # NEW: Early detection for comparison questions
        comparison_patterns = [
            r'\bdid\b.*\bexceed\b',
            r'\bwas\b.*\bgreater than\b',
            r'\bwere\b.*\bmore than\b',
            r'\bdid\b.*\bmore than\b',
            r'\bexceed\b',
            r'\bgreater than\b'
        ]
        if any(re.search(pattern, question.lower()) for pattern in comparison_patterns):
            return 'comparison', self.question_patterns.get('comparison', {})
        
        for q_type, pattern in self.question_patterns.items():
            score = 0
            keywords = pattern.get('keywords', [])
            
            # Count keyword matches
            matches = 0
            for keyword in keywords:
                if keyword in question:
                    matches += 1
                    # Bonus for longer/more specific keywords
                    score += len(keyword.split())
            
            # Bonus for matching multiple keywords
            if matches > 0:
                score += matches * 2
            
            # Special handling: "what is" questions
            if q_type == 'direct_lookup' and question.startswith(('what is', 'what was', 'what were')):
                # Check if it's really a lookup (no calculation keywords)
                calc_keywords = ['percentage', 'percent', 'ratio', 'average', 'total', 
                               'difference', 'change', 'growth', 'increase', 'decrease']
                has_calc = any(kw in question for kw in calc_keywords)
                if not has_calc:
                    score += 10  # Strong bonus for pure lookup
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = (q_type, pattern)
        
        if best_match:
            return best_match
        
        # Default to direct_lookup if starts with "what is/was"
        if question.startswith(('what is', 'what was', 'what were', 'how much', 'how many')):
            return 'direct_lookup', self.question_patterns.get('direct_lookup', {})
        
        return 'unknown', {}
    
    def _extract_entities(self, doc) -> List[str]:
        """Extract entities từ question"""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'PERCENT', 'DATE', 'TIME', 
                             'QUANTITY', 'CARDINAL', 'ORDINAL',
                             'ORG', 'PRODUCT', 'EVENT']:
                entities.append(ent.text)
        
        # Extract financial terms (không phải NER entities)
        financial_terms = [
            'revenue', 'income', 'profit', 'loss', 'expense', 'cost',
            'sales', 'earnings', 'dividend', 'interest', 'tax',
            'asset', 'liability', 'equity', 'cash flow', 'debt'
        ]
        
        for term in financial_terms:
            if term in doc.text:
                entities.append(term)
        
        return list(set(entities))
    
    def _extract_numbers(self, question: str) -> List[float]:
        """Extract numbers từ question"""
        numbers = []
        
        # Pattern cho numbers với %, $, etc
        patterns = [
            r'\$?\s?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s?%?',
            r'([0-9]+\.?[0-9]*)\s?(?:million|billion|thousand|M|B|K)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, question, re.IGNORECASE):
                try:
                    num_str = match.group(1).replace(',', '')
                    numbers.append(float(num_str))
                except:
                    continue
        
        return numbers
    
    def _extract_temporal_entities(self, doc) -> List[str]:
        """Extract temporal entities (years, quarters, dates)"""
        temporal = []
        
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:
                temporal.append(ent.text)
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', doc.text)
        temporal.extend(years)
        
        # Extract quarters
        quarters = re.findall(r'\bQ[1-4]\b|\b[1-4](?:st|nd|rd|th)\s+quarter\b', 
                             doc.text, re.IGNORECASE)
        temporal.extend(quarters)
        
        return list(set(temporal))
    
    def _extract_keywords(self, doc, pattern: Dict) -> List[str]:
        """Extract important keywords"""
        keywords = []
        
        # Keywords từ pattern
        if pattern:
            pattern_keywords = pattern.get('keywords', [])
            for kw in pattern_keywords:
                if kw in doc.text:
                    keywords.append(kw)
        
        # Important nouns và verbs
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop:
                keywords.append(token.text)
        
        return list(set(keywords))
    
    def _determine_argument_order(self, question: str, q_type: str, 
                                  temporal: List[str], entities: List[str]) -> List[str]:
        """
        Determine thứ tự arguments - CỰC KỲ QUAN TRỌNG!
        
        VD: "percentage change from 2008 to 2009" 
        -> Order: ['2008', '2009'] (old first, new second)
        """
        order = []
        
        if q_type == 'percentage_change' or q_type == 'difference':
            # Tìm temporal indicators
            if 'from' in question and 'to' in question:
                # "from A to B" -> [A, B] where A is old, B is new
                from_match = re.search(r'from\s+([^\s]+)\s+to\s+([^\s]+)', question)
                if from_match:
                    order = [from_match.group(1), from_match.group(2)]
            
            elif temporal and len(temporal) >= 2:
                # Sort temporal entities chronologically
                sorted_temporal = sorted(temporal)
                order = sorted_temporal
        
        elif q_type == 'ratio' or q_type == 'percentage_of':
            # "X per Y" or "X of Y" -> [X, Y] where X is numerator
            per_match = re.search(r'(\w+)\s+(?:per|of|to)\s+(\w+)', question)
            if per_match:
                order = [per_match.group(1), per_match.group(2)]
        
        return order
