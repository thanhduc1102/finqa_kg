"""
Modern Knowledge Graph Query Engine for FinQA
Cung cấp các phương thức query nâng cao với semantic search
"""

from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
from functools import lru_cache
import logging
from dataclasses import dataclass
from .query_utils import normalize_query, calculate_score

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result from a knowledge graph query"""
    node_id: str
    score: float
    content: Any
    type: str
    metadata: Dict[str, Any] = None

class ModernFinQAGraphQuery:
    """Enhanced query engine with semantic search capabilities"""
    
    def __init__(self, graph: nx.MultiDiGraph, use_gpu: bool = torch.cuda.is_available()):
        self.graph = graph
        self.device = "cuda" if use_gpu else "cpu"
        self._initialize_models()
        self._build_index()
        
    def _initialize_models(self):
        """Initialize all models for querying"""
        try:
            # Semantic search model
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model.to(self.device)
            
            # Question answering model
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Query models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _build_index(self):
        """Build search index for all text nodes"""
        self.text_nodes = {}
        self.text_embeddings = []
        
        for node_id, data in self.graph.nodes(data=True):
            if 'content' in data:
                self.text_nodes[node_id] = data
                embedding = self._get_text_embedding(data['content'])
                self.text_embeddings.append(embedding)
                
        if self.text_embeddings:
            self.text_embeddings = np.vstack(self.text_embeddings)
        
        logger.info(f"Built search index for {len(self.text_nodes)} nodes")

    @lru_cache(maxsize=1000)
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding with caching"""
        return self.sentence_model.encode(text)

    def semantic_search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.5
    ) -> List[QueryResult]:
        """Perform semantic search across the knowledge graph"""
        # Normalize query
        query = normalize_query(query)
        
        # Get query embedding
        query_embedding = self._get_text_embedding(query)
        
        # Calculate similarities
        similarities = np.dot(self.text_embeddings, query_embedding)
        similarities = similarities / (
            np.linalg.norm(self.text_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k results
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            score = float(similarities[idx])
            if score < min_score:
                continue
                
            node_id = list(self.text_nodes.keys())[idx]
            node_data = self.text_nodes[node_id]
            
            results.append(QueryResult(
                node_id=node_id,
                score=score,
                content=node_data.get('content'),
                type=node_data.get('type'),
                metadata=node_data
            ))
            
        return results

    async def answer_question(
        self,
        question: str,
        context_window: int = 3
    ) -> Optional[str]:
        """Answer questions using relevant context from the knowledge graph"""
        # First find relevant context
        relevant_texts = self.semantic_search(question, k=context_window)
        
        if not relevant_texts:
            return None
            
        # Combine relevant texts
        context = " ".join(r.content for r in relevant_texts)
        
        # Use QA model
        try:
            answer = self.qa_model(
                question=question,
                context=context
            )
            
            if answer['score'] < 0.5:
                return None
                
            return answer['answer']
            
        except Exception as e:
            logger.warning(f"QA model failed: {e}")
            return None

    def get_entity_context(
        self,
        entity_text: str,
        context_window: int = 2
    ) -> List[QueryResult]:
        """Get contextual information about an entity"""
        results = []
        
        # Find nodes containing this entity
        for node_id, data in self.graph.nodes(data=True):
            if data.get('content') and entity_text.lower() in data['content'].lower():
                # Get neighboring context
                context_nodes = []
                for depth in range(1, context_window + 1):
                    neighbors = nx.single_source_shortest_path_length(
                        self.graph, node_id, cutoff=depth
                    )
                    for neighbor_id in neighbors:
                        if neighbor_id != node_id:
                            neighbor_data = self.graph.nodes[neighbor_id]
                            if 'content' in neighbor_data:
                                score = calculate_score(
                                    data['content'],
                                    neighbor_data['content'],
                                    depth
                                )
                                context_nodes.append(QueryResult(
                                    node_id=neighbor_id,
                                    score=score,
                                    content=neighbor_data['content'],
                                    type=neighbor_data.get('type'),
                                    metadata=neighbor_data
                                ))
                                
                results.extend(context_nodes)
                
        # Sort by score and remove duplicates
        results.sort(key=lambda x: x.score, reverse=True)
        seen = set()
        unique_results = []
        for r in results:
            if r.node_id not in seen:
                seen.add(r.node_id)
                unique_results.append(r)
                
        return unique_results

    def find_related_numbers(
        self,
        number: float,
        tolerance: float = 0.01,
        max_results: int = 10
    ) -> List[QueryResult]:
        """Find numbers related to a given number"""
        results = []
        
        # Find number nodes
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'number' and 'value' in data:
                value = data['value']
                
                # Check if numbers are related
                if abs(value - number) <= tolerance * abs(number):
                    # Get context for this number
                    edges = list(self.graph.in_edges(node_id, data=True))
                    for source, _, edge_data in edges:
                        source_data = self.graph.nodes[source]
                        if 'content' in source_data:
                            results.append(QueryResult(
                                node_id=node_id,
                                score=1.0 - (abs(value - number) / abs(number)),
                                content=source_data['content'],
                                type='number_context',
                                metadata={
                                    'number_value': value,
                                    'relation': edge_data.get('relation'),
                                    'context_type': source_data.get('type')
                                }
                            ))
                            
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def trace_calculation_path(
        self,
        start_number: float,
        end_number: float,
        max_depth: int = 5
    ) -> List[List[QueryResult]]:
        """Find possible calculation paths between two numbers"""
        paths = []
        
        # Find nodes for both numbers
        start_nodes = []
        end_nodes = []
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'number' and 'value' in data:
                if abs(data['value'] - start_number) < 1e-10:
                    start_nodes.append(node_id)
                elif abs(data['value'] - end_number) < 1e-10:
                    end_nodes.append(node_id)
                    
        # Find paths between number nodes
        for start in start_nodes:
            for end in end_nodes:
                # Get all paths
                node_paths = nx.all_simple_paths(
                    self.graph,
                    start,
                    end,
                    cutoff=max_depth
                )
                
                # Convert paths to results
                for path in node_paths:
                    result_path = []
                    for node_id in path:
                        data = self.graph.nodes[node_id]
                        result_path.append(QueryResult(
                            node_id=node_id,
                            score=1.0,  # Part of exact path
                            content=data.get('content'),
                            type=data.get('type'),
                            metadata=data
                        ))
                    paths.append(result_path)
                    
        return paths

    def analyze_numerical_trends(
        self,
        number_type: str = None,
        min_count: int = 3
    ) -> List[Dict[str, Any]]:
        """Analyze trends in numerical data"""
        trends = []
        
        # Group numbers by context
        number_groups = {}
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'number' and 'value' in data:
                # Get context
                edges = list(self.graph.in_edges(node_id, data=True))
                for source, _, edge_data in edges:
                    source_data = self.graph.nodes[source]
                    context_key = source_data.get('type', '')
                    if number_type and context_key != number_type:
                        continue
                        
                    if context_key not in number_groups:
                        number_groups[context_key] = []
                    number_groups[context_key].append((
                        data['value'],
                        source_data.get('content', ''),
                        edge_data.get('relation', '')
                    ))
                    
        # Analyze each group
        for context, numbers in number_groups.items():
            if len(numbers) < min_count:
                continue
                
            values = [n[0] for n in numbers]
            
            trend = {
                'context': context,
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'values': values,
                'contexts': [n[1] for n in numbers],
                'relations': [n[2] for n in numbers]
            }
            
            # Calculate trend direction
            if len(values) > 1:
                changes = [values[i] - values[i-1] for i in range(1, len(values))]
                trend['trend_direction'] = 'increasing' if sum(changes) > 0 else 'decreasing'
                trend['volatility'] = np.std(changes) if len(changes) > 1 else 0
                
            trends.append(trend)
            
        return trends

    def get_table_context(
        self,
        table_id: str,
        include_references: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive context for a table"""
        if not self.graph.has_node(table_id):
            return None
            
        table_data = self.graph.nodes[table_id]
        if table_data.get('type') != 'table':
            return None
            
        context = {
            'table_id': table_id,
            'metadata': table_data,
            'headers': [],
            'cells': [],
            'references': [] if include_references else None
        }
        
        # Get headers
        header_edges = [
            (_, target, data) for _, target, data in self.graph.edges(table_id, data=True)
            if data.get('relation') == 'has_header'
        ]
        
        for _, header_id, _ in sorted(header_edges, key=lambda x: x[2].get('col', 0)):
            header_data = self.graph.nodes[header_id]
            context['headers'].append({
                'content': header_data.get('content'),
                'metadata': header_data
            })
            
        # Get cells
        cell_edges = [
            (_, target, data) for _, target, data in self.graph.edges(table_id, data=True)
            if data.get('relation') == 'has_cell'
        ]
        
        cells_by_position = {}
        for _, cell_id, edge_data in cell_edges:
            row = edge_data.get('row', 0)
            col = edge_data.get('col', 0)
            cell_data = self.graph.nodes[cell_id]
            
            cells_by_position[(row, col)] = {
                'content': cell_data.get('content'),
                'metadata': cell_data
            }
            
        # Convert to 2D array
        if cells_by_position:
            max_row = max(pos[0] for pos in cells_by_position.keys())
            max_col = max(pos[1] for pos in cells_by_position.keys())
            
            for row in range(max_row + 1):
                row_cells = []
                for col in range(max_col + 1):
                    row_cells.append(cells_by_position.get((row, col), {'content': '', 'metadata': {}}))
                context['cells'].append(row_cells)
                
        # Get references if requested
        if include_references:
            # Find nodes referencing this table
            for source, _, edge_data in self.graph.in_edges(table_id, data=True):
                source_data = self.graph.nodes[source]
                if 'content' in source_data:
                    context['references'].append({
                        'node_id': source,
                        'content': source_data.get('content'),
                        'type': source_data.get('type'),
                        'relation': edge_data.get('relation')
                    })
                    
        return context