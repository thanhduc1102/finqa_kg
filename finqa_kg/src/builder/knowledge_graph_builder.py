"""
Modern Knowledge Graph Builder for FinQA Dataset
Enhanced with LLM and advanced NLP techniques
"""

import json
import re
from typing import Dict, List, Tuple, Any, Set, Optional
import networkx as nx
from collections import defaultdict
import spacy
import asyncio
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityMention:
    """Data class for storing entity mentions"""
    text: str
    type: str
    start: int
    end: int
    value: Optional[float] = None
    metadata: Dict[str, Any] = None

class ModernFinQAKnowledgeGraph:
    """
    Enhanced Knowledge Graph Builder with modern NLP capabilities
    """

    def __init__(self, use_gpu: bool = torch.cuda.is_available()):
        self.device = "cuda" if use_gpu else "cpu"
        self.graph = nx.MultiDiGraph()
        self.entity_index = {}
        self.node_counter = 0
        self._initialize_nlp_components()
        
    def _initialize_nlp_components(self):
        """Initialize all NLP components"""
        try:
            # Basic NLP
            self.nlp = spacy.load("en_core_web_trf")
            
            # Semantic text similarity model
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model.to(self.device)
            
            # Named Entity Recognition - Using a simpler but reliable model
            self.ner_pipeline = pipeline("ner", 
                                      model="dslim/bert-base-NER",
                                      device=0 if self.device == "cuda" else -1)
            
            # Relation Extraction - Using a general purpose text classification model
            self.relation_model = pipeline("text-classification",
                                         model="distilbert-base-uncased-finetuned-sst-2-english",
                                         device=0 if self.device == "cuda" else -1)
            
            logger.info("All NLP components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            raise

    # ---------------------- Helper methods ----------------------
    def _create_node_id(self, prefix: str) -> str:
        """Create a unique node id with a prefix."""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    async def _add_entity_node(self, entity: EntityMention) -> str:
        """Add or reuse an entity node in the graph.

        Deduplicates entities by normalized text (lowercased). Returns the node id.
        """
        if entity is None or not entity.text:
            return self._create_node_id('ENTITY')

        key = entity.text.strip().lower()
        if key in self.entity_index:
            return self.entity_index[key]

        node_id = self._create_node_id('ENTITY')
        self.graph.add_node(
            node_id,
            type='entity',
            text=entity.text,
            entity_type=getattr(entity, 'type', None),
            value=getattr(entity, 'value', None),
            metadata=(entity.metadata or {})
        )
        # Index by normalized text
        self.entity_index[key] = node_id
        return node_id

    async def _add_qa_node(self, qa_data: Dict[str, Any], parent_id: str) -> str:
        """Add a QA node (question + answer) and link it to parent document."""
        qa_node_id = self._create_node_id('QA')
        question = qa_data.get('question') if isinstance(qa_data, dict) else None
        answer = qa_data.get('answer') if isinstance(qa_data, dict) else None
        self.graph.add_node(
            qa_node_id,
            type='qa',
            question=question,
            answer=answer,
            parent=parent_id
        )
        # Link to parent
        self.graph.add_edge(parent_id, qa_node_id, relation='has_qa')
        return qa_node_id

    def get_statistics(self) -> Dict[str, Any]:
        """Return basic graph statistics: node counts by type and edge count."""
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            t = data.get('type', 'unknown')
            node_types[t] += 1

        stats = {
            'nodes_total': self.graph.number_of_nodes(),
            'edges_total': self.graph.number_of_edges(),
            'nodes_by_type': dict(node_types)
        }
        return stats

    def _extract_numbers(self, text: str) -> List[Tuple[str, Optional[float]]]:
        """Find numbers and currency amounts in text and return (text, value).

        This is a lightweight extractor intended for tests and demo data.
        """
        results: List[Tuple[str, Optional[float]]] = []

        # Currency with dollar sign or other common formats: $1,234.56
        for m in re.finditer(r"\$?\d{1,3}(?:[,\d]{0,})(?:\.\d+)?%?", text):
            txt = m.group(0)
            # Skip lone commas
            if txt.strip() == ',':
                continue
            val = None
            try:
                cleaned = txt.replace('$', '').replace(',', '')
                if cleaned.endswith('%'):
                    val = float(cleaned[:-1]) / 100.0
                else:
                    val = float(cleaned)
            except Exception:
                val = None
            results.append((txt, val))

        # Plain integers/floats not caught above
        for m in re.finditer(r"(?<!\$)(?<!\w)(\d+\.?\d*)(?!\w)", text):
            txt = m.group(1)
            # Avoid duplicates
            if any(txt == r[0].replace('$', '').replace(',', '') for r in results):
                continue
            try:
                val = float(txt)
            except Exception:
                val = None
            results.append((txt, val))

        return results

    def _extract_dates(self, text: str) -> List[str]:
        """Extract simple date patterns (MM/DD/YYYY, YYYY, Month DD, YYYY)."""
        dates: Set[str] = set()
        # MM/DD/YYYY or M/D/YYYY
        for m in re.finditer(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
            dates.add(m.group(0))
        # YYYY
        for m in re.finditer(r"\b(19|20)\d{2}\b", text):
            dates.add(m.group(0))
        # Month name patterns e.g., January 12, 2020
        for m in re.finditer(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b", text, re.IGNORECASE):
            dates.add(m.group(0))

        return list(dates)

    @lru_cache(maxsize=1000)
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding with caching"""
        return self.sentence_model.encode(text)

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        emb1 = self._get_text_embedding(text1)
        emb2 = self._get_text_embedding(text2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    async def _extract_entities_llm(self, text: str) -> List[EntityMention]:
        """Extract entities using advanced NLP models"""
        entities = []
        
        # Run NER
        ner_results = self.ner_pipeline(text)
        for ent in ner_results:
            entities.append(EntityMention(
                text=ent['word'],
                type=ent['entity'],
                start=ent['start'],
                end=ent['end']
            ))
            
        # Extract numbers and dates using regex
        numbers = self._extract_numbers(text)
        dates = self._extract_dates(text)
        
        # Add regex results
        for num_text, num_value in numbers:
            entities.append(EntityMention(
                text=num_text,
                type='NUMBER',
                start=text.find(num_text),
                end=text.find(num_text) + len(num_text),
                value=num_value
            ))
            
        for date in dates:
            entities.append(EntityMention(
                text=date,
                type='DATE',
                start=text.find(date),
                end=text.find(date) + len(date)
            ))
            
        return entities

    async def _extract_relations_llm(self, source_text: str, target_text: str) -> List[Dict[str, Any]]:
        """Extract semantic relations between texts using sentiment analysis"""
        # Combine texts for relation analysis
        combined_text = f"{source_text} [SEP] {target_text}"
        
        # Get sentiment prediction as relationship type
        result = self.relation_model(combined_text)
        
        # Convert sentiment to relationship type
        sentiment_score = result[0]['score']
        if sentiment_score > 0.6:
            relation_type = 'POSITIVE_RELATION'
        elif sentiment_score < 0.4:
            relation_type = 'NEGATIVE_RELATION'
        else:
            relation_type = 'NEUTRAL_RELATION'
            
        return [{
            'relation': relation_type,
            'confidence': sentiment_score
        }]

    async def _process_text_block(self, text: str, block_type: str, doc_node_id: str) -> str:
        """Process a text block with advanced NLP"""
        text_node_id = self._create_node_id('TEXT')
        self.graph.add_node(
            text_node_id,
            type='text',
            block_type=block_type,
            content=text,
            parent=doc_node_id,
            embedding=self._get_text_embedding(text).tolist()
        )

        # Extract entities
        entities = await self._extract_entities_llm(text)
        for entity in entities:
            entity_node_id = await self._add_entity_node(entity)
            self.graph.add_edge(
                text_node_id,
                entity_node_id,
                relation='contains_entity',
                start=entity.start,
                end=entity.end
            )

        return text_node_id

    async def add_document(self, doc_data: Dict[str, Any]) -> str:
        """Enhanced document processing with semantic understanding"""
        doc_node_id = self._create_node_id('DOC')
        self.graph.add_node(
            doc_node_id,
            type='document',
            doc_id=doc_data.get('id', f"doc_{self.node_counter}"),
            filename=doc_data.get('filename', '')
        )

        # Process pre_text and post_text separately
        pre_text_nodes = []
        post_text_nodes = []
        
        # Process pre_text
        for idx, text in enumerate(doc_data.get('pre_text', [])):
            node_id = await self._process_text_block(text, 'pre_text', doc_node_id)
            pre_text_nodes.append((node_id, text))
            self.graph.add_edge(doc_node_id, node_id, relation='has_pre_text', order=idx)

        # Process post_text
        for idx, text in enumerate(doc_data.get('post_text', [])):
            node_id = await self._process_text_block(text, 'post_text', doc_node_id)
            post_text_nodes.append((node_id, text))
            self.graph.add_edge(doc_node_id, node_id, relation='has_post_text', order=idx)

        # Create semantic links between related text blocks
        await self._create_semantic_links(pre_text_nodes, post_text_nodes)

        # Process table if exists
        table = doc_data.get('table', [])
        if table:
            table_node_id = await self._add_table_node(table, doc_node_id)
            self.graph.add_edge(doc_node_id, table_node_id, relation='has_table')

        # Process QA
        qa_data = doc_data.get('qa', {})
        if qa_data:
            await self._add_qa_node(qa_data, doc_node_id)

        return doc_node_id

    async def _create_semantic_links(self, pre_text_nodes: List[Tuple[str, str]], 
                                   post_text_nodes: List[Tuple[str, str]]):
        """Create semantic links between related text blocks"""
        for pre_id, pre_text in pre_text_nodes:
            for post_id, post_text in post_text_nodes:
                # Compute semantic similarity
                similarity = self._compute_semantic_similarity(pre_text, post_text)
                
                # If texts are semantically related
                if similarity > 0.5:  # Threshold can be adjusted
                    # Extract specific relations
                    relations = await self._extract_relations_llm(pre_text, post_text)
                    
                    # Add semantic edges
                    for rel in relations:
                        self.graph.add_edge(
                            pre_id,
                            post_id,
                            relation=rel['relation'],
                            similarity=similarity,
                            confidence=rel['confidence']
                        )

    async def _add_table_node(self, table: List[List[str]], parent_id: str) -> str:
        """Enhanced table processing with semantic understanding"""
        table_node_id = self._create_node_id('TABLE')
        self.graph.add_node(
            table_node_id,
            type='table',
            rows=len(table),
            cols=len(table[0]) if table else 0,
            parent=parent_id
        )

        if not table:
            return table_node_id

        # Process headers
        headers = table[0]
        header_nodes = []
        for col_idx, header in enumerate(headers):
            header_node_id = self._create_node_id('HEADER')
            self.graph.add_node(
                header_node_id,
                type='table_header',
                content=header,
                col_index=col_idx,
                embedding=self._get_text_embedding(header).tolist()
            )
            self.graph.add_edge(table_node_id, header_node_id, relation='has_header', col=col_idx)
            header_nodes.append(header_node_id)

        # Process cells with parallel processing
        tasks = []
        for row_idx, row in enumerate(table[1:], start=1):
            for col_idx, cell in enumerate(row):
                task = self._process_cell(cell, row_idx, col_idx, table_node_id, header_nodes[col_idx])
                tasks.append(task)
        
        await asyncio.gather(*tasks)
        return table_node_id

    async def _process_cell(self, cell_content: str, row: int, col: int, 
                          table_id: str, header_id: str):
        """Process individual table cell with semantic understanding"""
        cell_node_id = self._create_node_id('CELL')
        self.graph.add_node(
            cell_node_id,
            type='table_cell',
            content=cell_content,
            row=row,
            col=col,
            table=table_id,
            embedding=self._get_text_embedding(cell_content).tolist()
        )
        
        # Link cell to table
        self.graph.add_edge(table_id, cell_node_id, relation='has_cell', row=row, col=col)
        
        # Link cell to header
        self.graph.add_edge(header_id, cell_node_id, relation='header_to_cell')
        
        # Extract entities from cell
        entities = await self._extract_entities_llm(cell_content)
        for entity in entities:
            entity_node_id = await self._add_entity_node(entity)
            self.graph.add_edge(
                cell_node_id,
                entity_node_id,
                relation='contains_entity',
                start=entity.start,
                end=entity.end
            )

    async def build_from_json(self, json_file_path: str, max_samples: int = None):
        """Build knowledge graph from FinQA JSON with modern processing"""
        logger.info(f"Loading data from {json_file_path}...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Accept either a list of documents or a single document (dict)
        if isinstance(data, dict):
            data_list = [data]
        elif isinstance(data, list):
            data_list = data
        else:
            # If the file contains other types (e.g., newline-delimited JSON strings), try to coerce
            try:
                data_list = list(data)
            except Exception:
                raise ValueError("Unsupported JSON format for knowledge graph input")

        if max_samples:
            data_list = data_list[:max_samples]

        logger.info(f"Building knowledge graph from {len(data_list)} documents...")

        # Process documents with progress bar
        for doc_data in tqdm(data_list, desc="Processing documents"):
            # Ensure each doc_data is a dict; if it's a string (e.g., path) try to load it
            if isinstance(doc_data, str):
                try:
                    with open(doc_data, 'r', encoding='utf-8') as df:
                        doc_data = json.load(df)
                except Exception:
                    # Skip invalid entries
                    logger.warning(f"Skipping invalid document entry: {doc_data}")
                    continue

            if not isinstance(doc_data, dict):
                logger.warning("Skipping non-dictionary document entry")
                continue

            await self.add_document(doc_data)

        logger.info("Knowledge graph built successfully!")
        stats = self.get_statistics()
        logger.info(f"Graph Statistics:\n{json.dumps(stats, indent=2)}")

    def save_graph(self, output_path: str):
        """Save the enhanced knowledge graph"""
        import pickle
        with open(output_path, 'wb') as f:
            # Don't save NLP components
            graph_data = {
                'graph': self.graph,
                'entity_index': self.entity_index,
                'node_counter': self.node_counter
            }
            pickle.dump(graph_data, f)
        logger.info(f"Graph saved to {output_path}")

    def load_graph(self, input_path: str):
        """Load the enhanced knowledge graph"""
        import pickle
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.entity_index = data['entity_index']
            self.node_counter = data['node_counter']
        logger.info(f"Graph loaded from {input_path}")
        self._initialize_nlp_components()  # Reinitialize NLP components

if __name__ == "__main__":
    # Example usage
    async def main():
        kg = ModernFinQAKnowledgeGraph()
        await kg.build_from_json(
            'FinQA/dataset/train.json',
            max_samples=10
        )
        kg.save_graph('finqa_modern_knowledge_graph.pkl')

    asyncio.run(main())