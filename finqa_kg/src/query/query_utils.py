"""
Query utilities for FinQA Knowledge Graph
"""

from typing import Dict, Any
import re
import numpy as np
from functools import lru_cache

def normalize_query(query: str) -> str:
    """Normalize query text"""
    # Convert to lowercase
    query = query.lower()
    
    # Normalize numbers
    query = re.sub(r'(\d),(\d)', r'\1\2', query)
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

def calculate_score(text1: str, text2: str, distance: int = 1) -> float:
    """Calculate relevance score between two text segments"""
    # Basic exact match scoring
    score = sum(
        word in text2.lower()
        for word in text1.lower().split()
    ) / len(text1.split())
    
    # Distance penalty
    score = score / (1 + np.log(1 + distance))
    
    return float(score)

@lru_cache(maxsize=1000)
def parse_number(text: str) -> float:
    """Parse number from text with currency and formatting"""
    try:
        # Remove currency symbols and commas
        clean = text.replace('$', '').replace(',', '').replace('€', '')
        clean = clean.replace('£', '').replace('¥', '')
        
        # Handle percentages
        if '%' in clean:
            clean = clean.replace('%', '')
            return float(clean) / 100
            
        return float(clean)
    except ValueError:
        return None

def get_node_importance(graph_data: Dict[str, Any]) -> float:
    """Calculate node importance score"""
    importance = 1.0
    
    # More weight to nodes with more connections
    if 'edges' in graph_data:
        importance *= (1 + len(graph_data['edges']) / 10)
        
    # More weight to nodes with rich content
    if 'content' in graph_data:
        importance *= (1 + len(str(graph_data['content'])) / 1000)
        
    # More weight to certain node types
    type_weights = {
        'table': 2.0,
        'header': 1.5,
        'number': 1.2,
        'date': 1.2
    }
    if 'type' in graph_data:
        importance *= type_weights.get(graph_data['type'], 1.0)
        
    return importance

def format_number(value: float) -> str:
    """Format number for display"""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.2f}K"
    elif abs(value) < 0.01:
        return f"{value:.2e}"
    else:
        return f"{value:.2f}"

def extract_calculation_steps(path: list) -> list:
    """Extract calculation steps from a path"""
    steps = []
    current_value = None
    
    for node in path:
        if node.get('type') == 'number' and 'value' in node:
            if current_value is None:
                current_value = node['value']
                steps.append({
                    'operation': 'start',
                    'value': current_value,
                    'text': node.get('content', '')
                })
            else:
                new_value = node['value']
                # Determine operation
                if abs(new_value - current_value) < 1e-10:
                    operation = 'equals'
                elif new_value > current_value:
                    operation = 'add'
                    amount = new_value - current_value
                else:
                    operation = 'subtract'
                    amount = current_value - new_value
                    
                steps.append({
                    'operation': operation,
                    'value': new_value,
                    'amount': amount,
                    'text': node.get('content', '')
                })
                current_value = new_value
                
    return steps

def analyze_trend(values: list) -> Dict[str, Any]:
    """Analyze numerical trend"""
    if not values:
        return {}
        
    analysis = {
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
        'count': len(values)
    }
    
    if len(values) > 1:
        # Calculate changes
        changes = [values[i] - values[i-1] for i in range(1, len(values))]
        analysis['trend'] = 'increasing' if sum(changes) > 0 else 'decreasing'
        
        # Calculate volatility
        if len(changes) > 1:
            analysis['volatility'] = float(np.std(changes))
            
        # Calculate growth rate
        if values[0] != 0:
            analysis['growth_rate'] = (values[-1] / values[0] - 1) * 100
            
    return analysis

def get_relation_importance(relation_type: str) -> float:
    """Get importance weight for different relation types"""
    weights = {
        'contains': 1.0,
        'has_cell': 1.2,
        'header_to_cell': 1.5,
        'contains_entity': 1.3,
        'semantically_related': 1.4,
        'supports_answer': 2.0
    }
    return weights.get(relation_type, 1.0)

def combine_scores(base_score: float, weights: Dict[str, float]) -> float:
    """Combine multiple scoring factors"""
    # Start with base score
    final_score = base_score
    
    # Apply weights multiplicatively
    for weight in weights.values():
        final_score *= weight
        
    # Normalize to [0,1]
    return min(1.0, max(0.0, final_score))