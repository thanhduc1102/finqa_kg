"""
Entity Extractor for FinQA Knowledge Graph
Sử dụng LLM và modern NLP techniques
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import spacy
from transformers import pipeline
import torch
import re
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
from .text_processor import clean_text, normalize_text

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entity extracted from text"""
    text: str
    type: str
    start: int
    end: int
    value: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

class EntityExtractor:
    """Modern entity extraction using multiple techniques"""
    
    def __init__(self, use_gpu: bool = torch.cuda.is_available()):
        self.device = "cuda" if use_gpu else "cpu"
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all NLP models"""
        try:
            # SpaCy model for basic NLP
            self.nlp = spacy.load("en_core_web_trf")
            
            # Financial NER model
            self.financial_ner = pipeline(
                "ner",
                model="yiyanghkust/finbert-pretrained-ner",
                device=0 if self.device == "cuda" else -1
            )
            
            # General NER for backup
            self.general_ner = pipeline(
                "ner",
                model="jean-baptiste/roberta-large-ner-english",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Entity extraction models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    @lru_cache(maxsize=1000)
    def _extract_numbers(self, text: str) -> List[Entity]:
        """Extract numerical entities with regex"""
        entities = []
        patterns = [
            # Currency with symbols
            (r'(?:[\$\€\£\¥])\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', 'MONEY'),
            # Currency with codes
            (r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|EUR|GBP|JPY)', 'MONEY'),
            # Percentages
            (r'(-?\d+(?:\.\d+)?)\s*%', 'PERCENTAGE'),
            # Numbers with commas
            (r'(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?)', 'NUMBER'),
            # Decimal numbers
            (r'(-?\d+\.\d+)', 'NUMBER'),
            # Plain integers
            (r'(-?\d+)', 'NUMBER')
        ]
        
        for pattern, entity_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    num_str = match.group(1).replace(',', '')
                    value = float(num_str)
                    entities.append(Entity(
                        text=match.group(0),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        value=value,
                        confidence=1.0,
                        metadata={'pattern': pattern}
                    ))
                except (ValueError, IndexError):
                    continue
        
        return entities

    @lru_cache(maxsize=1000)
    def _extract_dates(self, text: str) -> List[Entity]:
        """Extract date entities with regex"""
        entities = []
        patterns = [
            # MM/DD/YYYY
            (r'\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b', 'DATE'),
            # Month DD, YYYY
            (r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
             r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
             r'Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,\s*(\d{4})\b', 'DATE'),
            # YYYY
            (r'\b(19[7-9]\d|20[0-2]\d)\b', 'YEAR')
        ]
        
        for pattern, entity_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(0),
                    type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    metadata={'pattern': pattern}
                ))
        
        return entities

    async def extract_financial_entities(self, text: str) -> List[Entity]:
        """Extract financial entities using FinBERT"""
        entities = []
        try:
            results = self.financial_ner(text)
            for result in results:
                entities.append(Entity(
                    text=result['word'],
                    type=result['entity'],
                    start=result['start'],
                    end=result['end'],
                    confidence=result['score'],
                    metadata={'source': 'finbert'}
                ))
        except Exception as e:
            logger.warning(f"Financial NER failed: {e}, falling back to general NER")
            
        return entities

    async def extract_general_entities(self, text: str) -> List[Entity]:
        """Extract general entities using RoBERTa"""
        entities = []
        try:
            results = self.general_ner(text)
            for result in results:
                entities.append(Entity(
                    text=result['word'],
                    type=result['entity'],
                    start=result['start'],
                    end=result['end'],
                    confidence=result['score'],
                    metadata={'source': 'roberta'}
                ))
        except Exception as e:
            logger.warning(f"General NER failed: {e}")
            
        return entities

    async def extract_spacy_entities(self, text: str) -> List[Entity]:
        """Extract entities using spaCy"""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.8,  # SpaCy doesn't provide confidence scores
                metadata={'source': 'spacy'}
            ))
            
        return entities

    def _merge_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities, preferring higher confidence ones"""
        if not entities:
            return []
            
        # Sort by start position and confidence
        sorted_entities = sorted(
            entities,
            key=lambda x: (x.start, -x.confidence)
        )
        
        merged = []
        current = sorted_entities[0]
        
        for next_entity in sorted_entities[1:]:
            if current.end <= next_entity.start:
                merged.append(current)
                current = next_entity
            else:
                # Overlap found - keep the one with higher confidence
                if next_entity.confidence > current.confidence:
                    current = next_entity
                
        merged.append(current)
        return merged

    async def extract_all_entities(self, text: str) -> List[Entity]:
        """Extract all entities using multiple techniques in parallel"""
        # Clean and normalize text
        text = clean_text(text)
        text = normalize_text(text)
        
        # Extract different types of entities in parallel
        with ThreadPoolExecutor() as executor:
            number_future = executor.submit(self._extract_numbers, text)
            date_future = executor.submit(self._extract_dates, text)
            
        # Get results from regex extraction
        entities = []
        entities.extend(number_future.result())
        entities.extend(date_future.result())
        
        # Add ML-based entities
        entities.extend(await self.extract_financial_entities(text))
        entities.extend(await self.extract_general_entities(text))
        entities.extend(await self.extract_spacy_entities(text))
        
        # Merge overlapping entities
        merged_entities = self._merge_overlapping_entities(entities)
        
        # Sort by position
        return sorted(merged_entities, key=lambda x: x.start)