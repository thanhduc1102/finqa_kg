"""
Text Processor for FinQA Knowledge Graph
Cung cấp các utilities cho text processing
"""

import re
from typing import List, Dict, Any
import unicodedata
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1000)
def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing characters"""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
    
    return text.strip()

@lru_cache(maxsize=1000)
def normalize_text(text: str) -> str:
    """Normalize text for better matching"""
    # Convert to lowercase
    text = text.lower()
    
    # Normalize numbers
    text = re.sub(r'(\d),(\d)', r'\1\2', text)  # Remove commas in numbers
    
    # Normalize dates
    text = re.sub(r'(\d)(st|nd|rd|th)\b', r'\1', text)  # Remove ordinal indicators
    
    # Normalize currency symbols
    currency_map = {
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '$': 'USD'
    }
    for symbol, code in currency_map.items():
        text = text.replace(symbol, code + ' ')
    
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences intelligently"""
    # Handle common abbreviations
    text = re.sub(r'(?<=Mr)\.', '@', text)
    text = re.sub(r'(?<=Ms)\.', '@', text)
    text = re.sub(r'(?<=Mrs)\.', '@', text)
    text = re.sub(r'(?<=Dr)\.', '@', text)
    text = re.sub(r'(?<=Prof)\.', '@', text)
    text = re.sub(r'(?<=Inc)\.', '@', text)
    text = re.sub(r'(?<=Ltd)\.', '@', text)
    text = re.sub(r'(?<=Corp)\.', '@', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore periods in abbreviations
    sentences = [s.replace('@', '.') for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]

def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract key phrases from text using simple heuristics"""
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Score sentences based on heuristics
    scored_sentences = []
    for sentence in sentences:
        score = 0
        # Presence of numbers
        score += len(re.findall(r'\d+', sentence))
        # Presence of financial terms
        financial_terms = ['revenue', 'profit', 'loss', 'cost', 'price', 
                         'sales', 'margin', 'growth', 'decrease', 'increase']
        score += sum(1 for term in financial_terms if term in sentence.lower())
        # Length penalty
        score /= (1 + len(sentence) / 100)  # Prefer shorter sentences
        
        scored_sentences.append((score, sentence))
    
    # Sort by score and take top phrases
    scored_sentences.sort(reverse=True)
    return [s for _, s in scored_sentences[:max_phrases]]

def segment_table_text(table_text: str) -> Dict[str, Any]:
    """Segment table text into structural components"""
    # Split header and data
    lines = table_text.strip().split('\n')
    if not lines:
        return {'headers': [], 'data': []}
        
    # Process headers
    headers = lines[0].strip().split('\t')
    headers = [clean_text(h) for h in headers]
    
    # Process data rows
    data = []
    for line in lines[1:]:
        cells = line.strip().split('\t')
        cells = [clean_text(c) for c in cells]
        if len(cells) == len(headers):  # Only add complete rows
            data.append(cells)
            
    return {
        'headers': headers,
        'data': data,
        'num_rows': len(data),
        'num_cols': len(headers)
    }

def analyze_text_structure(text: str) -> Dict[str, Any]:
    """Analyze the structure of a text passage"""
    text = clean_text(text)
    sentences = split_into_sentences(text)
    
    # Analyze sentence structure
    sentence_lengths = [len(s.split()) for s in sentences]
    num_sentences = len(sentences)
    
    # Find patterns
    patterns = {
        'numbered_lists': len(re.findall(r'^\d+\.', text, re.M)),
        'bullet_points': len(re.findall(r'^[•\-\*]', text, re.M)),
        'parenthetical': len(re.findall(r'\([^)]+\)', text)),
        'quotes': len(re.findall(r'"[^"]+"', text))
    }
    
    return {
        'num_sentences': num_sentences,
        'avg_sentence_length': sum(sentence_lengths) / num_sentences if num_sentences else 0,
        'patterns': patterns,
        'key_phrases': extract_key_phrases(text)
    }

def find_contextual_references(text: str) -> Dict[str, List[str]]:
    """Find contextual references in text"""
    references = {
        'dates': [],
        'amounts': [],
        'companies': [],
        'relative_refs': []
    }
    
    # Find date references
    references['dates'].extend(re.findall(
        r'\b(?:last|next|previous|following)\s+(?:year|quarter|month|week)\b',
        text,
        re.I
    ))
    
    # Find amount references
    references['amounts'].extend(re.findall(
        r'\b(?:increased|decreased|grew|declined)\s+by\s+[\d.]+%?\b',
        text,
        re.I
    ))
    
    # Find company references
    references['companies'].extend(re.findall(
        r'\b(?:the\s+company|we|our\s+company|it)\b',
        text,
        re.I
    ))
    
    # Find relative time/value references
    references['relative_refs'].extend(re.findall(
        r'\b(?:above|below|aforementioned|previously|subsequently)\b',
        text,
        re.I
    ))
    
    return references

def get_text_statistics(text: str) -> Dict[str, Any]:
    """Get comprehensive statistics about text"""
    clean = clean_text(text)
    
    stats = {
        'length': len(clean),
        'word_count': len(clean.split()),
        'sentence_count': len(split_into_sentences(clean)),
        'number_count': len(re.findall(r'\d+', clean)),
        'currency_count': len(re.findall(r'[\$\€\£\¥]\s*\d+', clean)),
        'percentage_count': len(re.findall(r'\d+\s*%', clean))
    }
    
    # Add structure analysis
    stats.update(analyze_text_structure(clean))
    
    # Add contextual references
    stats['references'] = find_contextual_references(clean)
    
    return stats