"""
Relation Extractor for FinQA Knowledge Graph
Sử dụng LLM và modern NLP techniques
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """Relation between entities or text blocks"""
    source: str
    target: str
    relation_type: str
    confidence: float
    metadata: Dict[str, Any] = None

class RelationExtractor:
    """Modern relation extraction using transformer models"""
    
    def __init__(self, use_gpu: bool = torch.cuda.is_available()):
        self.device = "cuda" if use_gpu else "cpu"
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all models for relation extraction"""
        try:
            # Text similarity model
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model.to(self.device)
            
            # Financial relation classifier
            self.relation_classifier = pipeline(
                "text-classification",
                model="relik-ie/relik-relation-extraction-small",
                device=0 if self.device == "cuda" else -1
            )
            
            # Zero-shot classifier for flexible relation types
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Relation extraction models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    @lru_cache(maxsize=1000)
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding with caching"""
        return self.sentence_model.encode(text)

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        emb1 = self._get_text_embedding(text1)
        emb2 = self._get_text_embedding(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    async def extract_financial_relations(self, source_text: str, target_text: str) -> List[Relation]:
        """Extract financial relations between texts"""
        relations = []
        try:
            # Combine texts for relation classification
            combined_text = f"{source_text} [SEP] {target_text}"
            results = self.relation_classifier(combined_text)
            
            for result in results:
                if result['score'] > 0.5:  # Confidence threshold
                    relations.append(Relation(
                        source=source_text,
                        target=target_text,
                        relation_type=result['label'],
                        confidence=result['score'],
                        metadata={'source': 'financial_classifier'}
                    ))
        except Exception as e:
            logger.warning(f"Financial relation extraction failed: {e}")
            
        return relations

    async def extract_zero_shot_relations(
        self, 
        source_text: str, 
        target_text: str,
        candidate_relations: List[str]
    ) -> List[Relation]:
        """Extract relations using zero-shot learning"""
        relations = []
        try:
            # Combine texts
            combined_text = f"{source_text} [SEP] {target_text}"
            
            # Get predictions
            result = self.zero_shot_classifier(
                combined_text,
                candidate_labels=candidate_relations,
                multi_label=True
            )
            
            # Create relations for high-confidence predictions
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.7:  # High confidence threshold
                    relations.append(Relation(
                        source=source_text,
                        target=target_text,
                        relation_type=label,
                        confidence=score,
                        metadata={'source': 'zero_shot'}
                    ))
                    
        except Exception as e:
            logger.warning(f"Zero-shot relation extraction failed: {e}")
            
        return relations

    def _get_numerical_relation(self, num1: float, num2: float) -> Tuple[str, float]:
        """Determine relation between two numbers"""
        if abs(num1 - num2) < 1e-10:  # Handle floating point comparison
            return "equals", 1.0
        elif num1 > num2:
            return "greater_than", 1.0
        else:
            return "less_than", 1.0

    async def extract_numerical_relations(
        self,
        source_number: float,
        target_number: float
    ) -> List[Relation]:
        """Extract relations between numerical values"""
        relation_type, confidence = self._get_numerical_relation(source_number, target_number)
        
        # Calculate percentage difference
        try:
            pct_diff = abs((source_number - target_number) / target_number * 100)
            metadata = {'percentage_difference': pct_diff}
        except ZeroDivisionError:
            metadata = {}
            
        return [Relation(
            source=str(source_number),
            target=str(target_number),
            relation_type=relation_type,
            confidence=confidence,
            metadata=metadata
        )]

    async def extract_table_relations(
        self,
        header: str,
        cell_value: str,
        candidate_relations: List[str]
    ) -> List[Relation]:
        """Extract relations between table headers and cells"""
        relations = []
        
        # First try semantic similarity
        similarity = self.compute_semantic_similarity(header, cell_value)
        if similarity > 0.8:
            relations.append(Relation(
                source=header,
                target=cell_value,
                relation_type="strongly_related",
                confidence=similarity,
                metadata={'source': 'semantic_similarity'}
            ))
            
        # Then try zero-shot classification
        zero_shot_relations = await self.extract_zero_shot_relations(
            header, cell_value, candidate_relations
        )
        relations.extend(zero_shot_relations)
        
        return relations

    async def extract_all_relations(
        self,
        source_text: str,
        target_text: str,
        source_type: str = None,
        target_type: str = None
    ) -> List[Relation]:
        """Extract all possible relations between two text segments"""
        relations = []
        
        # If both are numbers, extract numerical relations
        if source_type == "NUMBER" and target_type == "NUMBER":
            try:
                source_num = float(source_text.replace(',', '').replace('$', ''))
                target_num = float(target_text.replace(',', '').replace('$', ''))
                relations.extend(await self.extract_numerical_relations(source_num, target_num))
            except ValueError:
                pass
        
        # Extract financial relations
        relations.extend(await self.extract_financial_relations(source_text, target_text))
        
        # Extract general relations using zero-shot learning
        candidate_relations = [
            "describes", "elaborates", "contradicts", "supports",
            "precedes", "follows", "causes", "influences"
        ]
        relations.extend(await self.extract_zero_shot_relations(
            source_text, target_text, candidate_relations
        ))
        
        # Compute similarity and add if high enough
        similarity = self.compute_semantic_similarity(source_text, target_text)
        if similarity > 0.7:
            relations.append(Relation(
                source=source_text,
                target=target_text,
                relation_type="semantically_related",
                confidence=similarity,
                metadata={'source': 'semantic_similarity'}
            ))
            
        return relations

    def get_relation_statistics(self, relations: List[Relation]) -> Dict[str, Any]:
        """Get statistics about extracted relations"""
        stats = {
            'total_relations': len(relations),
            'relation_types': {},
            'confidence_stats': {
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0
            },
            'sources': {}
        }
        
        if not relations:
            return stats
            
        # Collect statistics
        confidences = []
        for rel in relations:
            # Relation type counts
            stats['relation_types'][rel.relation_type] = \
                stats['relation_types'].get(rel.relation_type, 0) + 1
                
            # Confidence stats
            conf = rel.confidence
            confidences.append(conf)
            stats['confidence_stats']['min'] = min(stats['confidence_stats']['min'], conf)
            stats['confidence_stats']['max'] = max(stats['confidence_stats']['max'], conf)
            
            # Source counts
            if rel.metadata and 'source' in rel.metadata:
                source = rel.metadata['source']
                stats['sources'][source] = stats['sources'].get(source, 0) + 1
                
        # Calculate average confidence
        stats['confidence_stats']['avg'] = sum(confidences) / len(confidences)
        
        return stats