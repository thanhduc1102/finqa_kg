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
    
    # NEW: Structured argument descriptions
    argument_structure: Dict[str, str] = None  # {'part': 'debt securities', 'whole': 'total revenues'}
    temporal_constraint: str = None  # '2013' or 'dec 29 2012'
    row_keywords: List[str] = None  # Keywords for table row
    col_keywords: List[str] = None  # Keywords for table column


class QuestionStructureExtractor:
    """
    Extract semantic structure from question to identify argument roles
    
    Example:
        Q: "what percentage of total net revenues were due to debt securities?"
        Output: {
            'argument_structure': {
                'whole': 'total net revenues',
                'part': 'debt securities'
            },
            'temporal_constraint': '2013',
            'row_keywords': ['debt securities'],
            'col_keywords': ['2013']
        }
    """
    
    # Question patterns for different types
    PATTERNS = {
        'percentage_of': [
            # Pattern 1: "percentage of X were/was Y"
            {
                'regex': r'percentage of\s+(?P<whole>.+?)\s+(?:were|was|consisted?|comprised)\s+(?:of\s+)?(?P<part>.+?)(?:\?|$)',
                'args': {'whole': 'whole', 'part': 'part'}
            },
            # Pattern 2: "percent of X due to Y"
            {
                'regex': r'percent(?:age)? of\s+(?P<whole>.+?)\s+(?:due to|attributed to|from)\s+(?P<part>.+?)(?:\?|$)',
                'args': {'whole': 'whole', 'part': 'part'}
            },
            # Pattern 3: "percent of X associated with Y"
            {
                'regex': r'percent(?:age)? of\s+(?P<whole>.+?)\s+(?:associated with|related to)\s+(?P<part>.+?)(?:\?|$)',
                'args': {'whole': 'whole', 'part': 'part'}
            },
            # Pattern 4: "X as a percentage of Y"
            {
                'regex': r'(?P<part>.+?)\s+as a (?:percentage|percent) of\s+(?P<whole>.+?)(?:\?|$)',
                'args': {'whole': 'whole', 'part': 'part'}
            },
        ],
        'percentage_change': [
            # Pattern 1: "change from X to Y"
            {
                'regex': r'(?:change|growth|increase|decrease)\s+(?:from|between)\s+(?P<old>\d{4})\s+(?:to|and)\s+(?P<new>\d{4})',
                'args': {'old': 'old_year', 'new': 'new_year'}
            },
            # Pattern 2: "growth rate in YEAR"
            {
                'regex': r'(?:growth|change)\s+rate\s+(?:in|for)\s+(?P<new>\d{4})',
                'args': {'new': 'new_year'}  # old = previous year
            },
        ],
        'difference': [
            # Pattern: "difference between X and Y"
            {
                'regex': r'difference between\s+(?P<value1>.+?)\s+and\s+(?P<value2>.+?)(?:\?|$)',
                'args': {'value1': 'first', 'value2': 'second'}
            },
            # Pattern: "change in X"
            {
                'regex': r'change in\s+(?P<concept>.+?)(?:\?|$)',
                'args': {'concept': 'main'}
            },
        ],
        'direct_lookup': [
            # Pattern: "what is/was X"
            {
                'regex': r'what (?:is|was|were)\s+(?:the\s+)?(?P<concept>.+?)(?:\?|$)',
                'args': {'concept': 'target'}
            },
        ],
    }
    
    def extract_structure(self, question: str, question_type: str) -> Dict:
        """
        Extract argument structure from question
        
        Args:
            question: Full question text
            question_type: Type from QuestionAnalyzer
        
        Returns:
            Dict with keys: argument_structure, temporal_constraint, row_keywords, col_keywords
        """
        result = {
            'argument_structure': {},
            'temporal_constraint': None,
            'row_keywords': [],
            'col_keywords': []
        }
        
        # Extract temporal constraint first (year, date, quarter)
        result['temporal_constraint'] = self._extract_temporal(question)
        
        # Try pattern matching for this question type
        if question_type in self.PATTERNS:
            for pattern_info in self.PATTERNS[question_type]:
                match = re.search(pattern_info['regex'], question, re.IGNORECASE)
                if match:
                    # Extract argument descriptions
                    groups = match.groupdict()
                    for arg_role, group_name in pattern_info['args'].items():
                        if group_name in groups:
                            raw_text = groups[group_name]
                            # Clean and extract keywords
                            keywords = self._extract_keywords(raw_text)
                            result['argument_structure'][arg_role] = {
                                'raw_text': raw_text,
                                'keywords': keywords
                            }
                    break  # Use first matching pattern
        
        # Extract row/column keywords for table navigation
        if result['argument_structure']:
            result['row_keywords'], result['col_keywords'] = self._identify_table_keywords(
                result['argument_structure'],
                result['temporal_constraint']
            )
        
        return result
    
    def _extract_temporal(self, question: str) -> str:
        """
        Extract temporal constraint (year, date, quarter)
        Priority: specific date > year > quarter
        """
        # Pattern 1: Full date (dec 29 2012, december 29, 2012)
        date_pattern = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\.?\s*\d{1,2}[,\s]+\d{4}'
        date_match = re.search(date_pattern, question, re.IGNORECASE)
        if date_match:
            return date_match.group(0)
        
        # Pattern 2: Just month and year (december 2012)
        month_year_pattern = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\.?\s*\d{4}'
        month_year_match = re.search(month_year_pattern, question, re.IGNORECASE)
        if month_year_match:
            return month_year_match.group(0)
        
        # Pattern 3: Year only (2013, 2012, etc.)
        year_pattern = r'\b(19\d{2}|20[0-2]\d)\b'
        year_matches = re.findall(year_pattern, question)
        if year_matches:
            # Return most recent/specific year mentioned
            return year_matches[-1]
        
        # Pattern 4: Quarter (Q1 2013, first quarter 2013)
        quarter_pattern = r'(?:Q[1-4]|(?:first|second|third|fourth) quarter)\s*(?:of\s*)?\d{4}'
        quarter_match = re.search(quarter_pattern, question, re.IGNORECASE)
        if quarter_match:
            return quarter_match.group(0)
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text phrase
        Remove stop words and keep content words
        """
        # Common stop words to remove
        stop_words = {
            'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'at', 'by', 'with',
            'from', 'as', 'is', 'was', 'were', 'are', 'been', 'be', 'being',
            'that', 'this', 'these', 'those', 'and', 'or', 'but'
        }
        
        # Split and clean
        words = text.lower().split()
        keywords = []
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                keywords.append(clean_word)
        
        return keywords
    
    def _identify_table_keywords(self, 
                                 argument_structure: Dict,
                                 temporal_constraint: str) -> Tuple[List[str], List[str]]:
        """
        Identify which keywords are for table rows vs columns
        
        Heuristic:
        - Temporal constraint -> column keyword
        - Argument keywords -> row keywords
        """
        row_keywords = []
        col_keywords = []
        
        # Temporal goes to column
        if temporal_constraint:
            col_keywords.append(temporal_constraint)
        
        # Argument keywords go to rows
        for arg_name, arg_info in argument_structure.items():
            if isinstance(arg_info, dict) and 'keywords' in arg_info:
                row_keywords.extend(arg_info['keywords'])
        
        return row_keywords, col_keywords


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
        
        # Initialize structure extractor
        self.structure_extractor = QuestionStructureExtractor()
    
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
                'keywords': ['what percent', 'what percentage', 'percent of', 'percentage of',
                            'what is the percent', 'what was the percent', 
                            'what is the percentage', 'what was the percentage'],
                'operations': ['divide', 'multiply'],
                'formula': '(part / whole) * 100',
                'argument_order': ['part', 'whole'],
                'negative_keywords': ['change', 'increase', 'decrease', 'growth'],  # Not percentage_change!
                'examples': [
                    'X is what percent of Y?',
                    'What percentage of Y is X?',
                    'What is the percent of X?'
                ]
            },
            'percentage_change': {
                'keywords': ['percentage change', 'percent change', 'growth rate', 'change rate',
                            'percentage increase', 'percentage decrease',
                            'what is the growth rate', 'what was the growth rate',
                            'what percentage decrease', 'what percentage increase'],
                'operations': ['subtract', 'divide', 'multiply'],
                'formula': '(new - old) / old * 100',
                'argument_order': ['new', 'old'],  # NEW - OLD!
                'temporal_indicators': ['from', 'to', 'between'],
                'examples': [
                    'What is the percentage change from X to Y?',
                    'What is the growth rate of X?',
                    'What percentage decrease occurred from 2011-2012?'
                ]
            },
            'difference': {
                'keywords': ['what is the change', 'what was the change', 
                            'what is the difference', 'what was the difference',
                            'how much change', 'how much difference',
                            'change in', 'difference between'],
                'operations': ['subtract'],
                'formula': 'a - b',
                'argument_order': ['larger', 'smaller'],  # Or temporal order
                'temporal_indicators': ['from', 'to', 'than', 'between'],
                'examples': [
                    'What is the difference between X and Y?',
                    'What is the change from X to Y?',
                    'What is the change in net assets from 2007 to 2008?'
                ]
            },
            'direct_lookup': {
                # CRITICAL FIX: Remove "what is/was/were" - these are TOO GENERAL!
                # Direct lookup should ONLY match very specific no-calculation questions
                'keywords': [],  # Will be detected by ABSENCE of calc keywords
                'operations': ['lookup'],  # No calculation, just retrieve
                'formula': 'Extract from KG',
                'negative_keywords': ['percentage', 'percent', 'ratio', 'per', 
                                    'change', 'growth', 'rate', 'difference',
                                    'total', 'sum', 'average', 'mean',
                                    'increase', 'decrease'],
                'examples': [
                    'What is the revenue in 2020?',  # Simple lookup with specific entity + year
                    'What was the value of X?'        # Simple lookup
                ]
            },
            'ratio': {
                'keywords': ['ratio', 'per', 'for each', 'for every', 
                            'average per', 'per unit', 'divided by'],
                'operations': ['divide'],
                'formula': 'numerator / denominator',
                'argument_order': ['numerator', 'denominator'],
                'examples': [
                    'What is the ratio of X to Y?',
                    'What is X per Y?',
                    'How much X for each Y?'
                ]
            },
            'sum': {
                'keywords': ['total', 'sum', 'combined', 'altogether', 'in total',
                            'what is the total', 'what was the total',
                            'total of'],
                'operations': ['add'],
                'formula': 'a + b + c + ...',
                'examples': [
                    'What is the total X?',
                    'In millions, what is the total of X?'
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
            'product': {
                'keywords': ['product', 'multiply', 'times', 'multiplied by'],
                'operations': ['multiply'],
                'formula': 'a * b',
                'examples': [
                    'What is X multiplied by Y?',
                    'What is the product of X and Y?'
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
            'absolute_value': {
                'keywords': ['absolute', 'magnitude'],
                'operations': ['abs'],
                'formula': '|x|',
                'examples': [
                    'What is the absolute value of X?'
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
        
        # 8. NEW: Extract structured argument descriptions
        structure_info = self.structure_extractor.extract_structure(question, question_type)
        
        return QuestionAnalysis(
            question=question,
            question_type=question_type,
            operations=operations,
            entities_mentioned=entities,
            numbers_mentioned=numbers,
            temporal_entities=temporal,
            keywords=keywords,
            argument_order=argument_order,
            # NEW structured fields
            argument_structure=structure_info.get('argument_structure', {}),
            temporal_constraint=structure_info.get('temporal_constraint'),
            row_keywords=structure_info.get('row_keywords', []),
            col_keywords=structure_info.get('col_keywords', [])
        )
    
    def _detect_question_type(self, question: str, doc) -> Tuple[str, Dict]:
        """
        Detect question type dựa trên PHÂN TÍCH DỮ LIỆU THỰC TẾ
        
        Key insights from data:
        - "what is/was" questions are MOSTLY calculations, NOT lookups!
        - "percentage of" questions -> divide operation
        - "growth rate" or "change" -> percentage_change (subtract + divide)
        - "change in" or "difference" -> subtract operation
        - Direct lookup is RARE and should be last resort
        """
        q_lower = question.lower()
        best_match = None
        best_score = 0
        
        # PRIORITY 1: Specific percentage questions (highest priority)
        if 'what percent' in q_lower or 'what percentage' in q_lower:
            if any(kw in q_lower for kw in ['change', 'increase', 'decrease', 'growth']):
                return 'percentage_change', self.question_patterns.get('percentage_change', {})
            else:
                return 'percentage_of', self.question_patterns.get('percentage_of', {})
        
        # PRIORITY 2: Growth rate / percentage change questions
        if any(kw in q_lower for kw in ['growth rate', 'change rate', 'percentage change', 'percent change',
                                          'percentage increase', 'percentage decrease']):
            return 'percentage_change', self.question_patterns.get('percentage_change', {})
        
        # PRIORITY 3: Change/Difference questions (subtract)
        if ('what is the change' in q_lower or 'what was the change' in q_lower or
            'what is the difference' in q_lower or 'what was the difference' in q_lower or
            'change in' in q_lower or 'change from' in q_lower):
            # But check if it's actually percentage change
            if 'percent' in q_lower or '%' in q_lower:
                return 'percentage_change', self.question_patterns.get('percentage_change', {})
            return 'difference', self.question_patterns.get('difference', {})
        
        # PRIORITY 4: Sum/Total questions  
        if 'total' in q_lower:
            # Check if it's "what is the total of X"
            if ' of ' in q_lower and 'what is the total of' not in q_lower:
                # "what percent of total" -> percentage_of
                return 'percentage_of', self.question_patterns.get('percentage_of', {})
            return 'sum', self.question_patterns.get('sum', {})
        
        # PRIORITY 5: Comparison questions
        comparison_patterns = [
            r'\bdid\b.*\bexceed\b',
            r'\bwas\b.*\bgreater than\b',
            r'\bwere\b.*\bmore than\b',
            r'\bdid\b.*\bmore than\b',
            r'\bexceed\b',
            r'\bgreater than\b'
        ]
        if any(re.search(pattern, q_lower) for pattern in comparison_patterns):
            return 'comparison', self.question_patterns.get('comparison', {})
        
        # PRIORITY 6: Ratio questions ("per", "for each")
        if re.search(r'\bper\b|\bfor each\b|\bfor every\b', q_lower):
            return 'ratio', self.question_patterns.get('ratio', {})
        
        # PRIORITY 7: Now do keyword-based scoring for remaining types
        for q_type, pattern in self.question_patterns.items():
            if q_type in ['direct_lookup']:  # Skip direct_lookup in this pass
                continue
                
            score = 0
            keywords = pattern.get('keywords', [])
            negative_keywords = pattern.get('negative_keywords', [])
            
            # Check negative keywords first
            if negative_keywords:
                if any(neg_kw in q_lower for neg_kw in negative_keywords):
                    continue  # Skip this type
            
            # Count keyword matches
            matches = 0
            for keyword in keywords:
                if keyword in q_lower:
                    matches += 1
                    # Longer keywords get more weight
                    score += len(keyword.split()) * 3
            
            # Bonus for multiple keyword matches
            if matches > 1:
                score += matches * 5
            elif matches == 1:
                score += 3
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = (q_type, pattern)
        
        if best_match and best_score > 0:
            return best_match
        
        # LAST RESORT: Check if it's truly a simple lookup
        # Only if NO calculation keywords detected
        calc_indicators = ['percentage', 'percent', '%', 'ratio', 'per', 
                          'change', 'growth', 'rate', 'difference',
                          'total', 'sum', 'average', 'mean',
                          'increase', 'decrease', 'exceed',
                          ' to ', ' between ', ' and ']
        
        has_calc = any(ind in q_lower for ind in calc_indicators)
        
        if not has_calc:
            # Might be direct lookup
            return 'direct_lookup', self.question_patterns.get('direct_lookup', {})
        
        # Still don't know - default to "other" (will use fallback in synthesizer)
        return 'other', {}
    
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
