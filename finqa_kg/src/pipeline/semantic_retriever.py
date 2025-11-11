"""
Semantic Table Retriever - Modern approach using embeddings & reranking
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from dataclasses import dataclass
import re


@dataclass
class SemanticMatch:
    """Kết quả semantic matching"""
    value: float
    text: str
    context: str
    score: float
    node_id: str
    label: str
    semantic_score: float  # Bi-encoder score
    rerank_score: Optional[float] = None  # Cross-encoder score
    method: str = 'semantic'


class SemanticTableRetriever:
    """
    Semantic retrieval cho table cells using:
    1. Sentence embeddings (bi-encoder) - fast
    2. Cross-encoder reranking - accurate
    3. Hybrid scoring with fallback
    """
    
    def __init__(self, use_reranking: bool = True):
        """
        Args:
            use_reranking: Enable cross-encoder reranking (slower but more accurate)
        """
        print("[Semantic Retriever] Initializing models...")
        
        # Bi-encoder: Fast semantic search
        # Using all-mpnet-base-v2: good for general semantic similarity
        self.bi_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Cross-encoder: More accurate reranking
        self.use_reranking = use_reranking
        if use_reranking:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Cache embeddings to avoid recomputation
        self.cell_embeddings_cache = {}
        self.cache_enabled = True
        
        print(f"[Semantic Retriever] Models loaded (reranking={'ON' if use_reranking else 'OFF'})")
    
    def clear_cache(self):
        """Clear embedding cache (for new document)"""
        self.cell_embeddings_cache.clear()
    
    def _create_cell_text(self, node_data: Dict) -> str:
        """
        Create rich text representation of a table cell
        Combines: row header | column header | value | context
        ENHANCED: Better extraction and formatting
        """
        parts = []
        
        # Extract table info from context
        context = node_data.get('context', '')
        
        # Strategy 1: Try to extract from table[row,col] format
        match = re.search(r'[Tt]able\[(\d+),(\d+)\]:\s*(.+)', context)
        if match:
            row_num, col_num, rest = match.groups()
            
            # Extract row and column from rest
            # Format: "row: X | col: Y = value" or "Row: X | Col: Y = value"
            row_match = re.search(r'[Rr]ow:\s*([^|]+)', rest)
            col_match = re.search(r'[Cc]ol:\s*([^=]+)', rest)
            
            if row_match:
                row_header = row_match.group(1).strip()
                if row_header and row_header not in ['', 'None']:
                    parts.append(f"Row: {row_header}")
            
            if col_match:
                col_header = col_match.group(1).strip()
                if col_header and col_header not in ['', 'None']:
                    parts.append(f"Column: {col_header}")
        
        # Strategy 2: Check if it's a cell node with attributes
        if node_data.get('type') == 'cell':
            row_label = node_data.get('row_label', '')
            header = node_data.get('header', '')
            
            if row_label and row_label not in ['', 'None']:
                parts.append(f"Row: {row_label}")
            if header and header not in ['', 'None']:
                parts.append(f"Column: {header}")
        
        # Add value with its text representation
        value = node_data.get('value')
        text = node_data.get('text', '')
        if value is not None:
            if text and text not in ['', 'None', str(value)]:
                parts.append(f"Value: {text}")
            else:
                parts.append(f"Value: {value}")
        
        # Add label/type for semantic understanding
        label = node_data.get('label', '')
        if label and label not in ['', 'None', 'CARDINAL', 'QUANTITY']:
            # Only include meaningful labels (skip generic number labels)
            parts.append(f"Type: {label}")
        
        # Add cleaned context (limited to avoid noise)
        if context:
            # Clean context - remove table coordinates
            clean_context = re.sub(r'[Tt]able\[\d+,\d+\]:\s*', '', context)
            clean_context = re.sub(r'[Rr]ow:\s*[^|]+\s*\|\s*[Cc]ol:\s*[^=]+=\s*', '', clean_context)
            
            # Take meaningful parts only
            if len(clean_context) > 10 and clean_context.strip():
                # Limit to 100 chars
                if len(clean_context) > 100:
                    clean_context = clean_context[:100] + '...'
                parts.append(f"Context: {clean_context}")
        
        # Join all parts
        result = ' | '.join(parts) if parts else str(value)
        
        # Additional enhancement: For entity nodes, include surrounding text
        if node_data.get('type') == 'entity':
            entity_context = node_data.get('context', '')
            if entity_context and len(entity_context) > 20:
                # Extract surrounding words for better context
                words = entity_context.split()[:15]  # First 15 words
                result += f" | {' '.join(words)}"
        
        return result
    
    def _extract_table_cells(self, kg: nx.MultiDiGraph) -> List[Dict]:
        """
        Extract all table cells from KG with rich metadata
        ENHANCED: Better value extraction and normalization
        """
        cells = []
        
        for node_id, node_data in kg.nodes(data=True):
            # Only process entity nodes with values
            if node_data.get('type') not in ['entity', 'cell']:
                continue
            
            value = node_data.get('value')
            text = node_data.get('text', str(value))
            
            # CRITICAL FIX: Normalize value to float
            if value is None:
                continue
            
            # Convert value to float if it's string
            if isinstance(value, str):
                # Try to extract number from string like "$ 6427" or "23.6%"
                import re
                # Remove currency symbols and commas
                clean_value = re.sub(r'[$,\s%]', '', value)
                # Remove parentheses (accounting notation for negative)
                clean_value = clean_value.replace('(', '-').replace(')', '')
                try:
                    value = float(clean_value)
                except:
                    # Skip if can't parse
                    continue
            
            # Convert to float
            try:
                value = float(value)
            except:
                continue
            
            # Check if it's from a table - check for both patterns
            context = node_data.get('context', '').lower()
            # Match both "table[" and "Table[" patterns case-insensitively
            if not ('table[' in context or 'table[' in context.lower()):
                # Also check for cell nodes with row/col attributes
                if node_data.get('type') == 'cell' and 'row' in node_data:
                    # This is a table cell node - construct context
                    row_label = node_data.get('row_label', '')
                    header = node_data.get('header', '')
                    row_idx = node_data.get('row', '')
                    col_idx = node_data.get('col', '')
                    
                    context = f"table[{row_idx},{col_idx}]: row: {row_label} | col: {header} = {text}"
                else:
                    continue
            
            # Create rich cell representation
            cell_text = self._create_cell_text(node_data)
            
            cells.append({
                'node_id': node_id,
                'value': value,  # Now guaranteed to be float
                'text': text,
                'context': context,
                'label': node_data.get('label', ''),
                'cell_text': cell_text,
                'node_data': node_data
            })
        
        return cells
    
    def _compute_embeddings(self, cells: List[Dict]) -> np.ndarray:
        """
        Compute embeddings for all cells (with caching)
        """
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, cell in enumerate(cells):
            cell_id = cell['node_id']
            
            # Check cache
            if self.cache_enabled and cell_id in self.cell_embeddings_cache:
                embeddings.append(self.cell_embeddings_cache[cell_id])
            else:
                texts_to_encode.append(cell['cell_text'])
                indices_to_encode.append(i)
        
        # Encode uncached cells in batch
        if texts_to_encode:
            new_embeddings = self.bi_encoder.encode(
                texts_to_encode,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32
            )
            
            # Cache new embeddings
            for idx, embedding in zip(indices_to_encode, new_embeddings):
                cell_id = cells[idx]['node_id']
                if self.cache_enabled:
                    self.cell_embeddings_cache[cell_id] = embedding
                embeddings.insert(idx, embedding)
        
        return np.array(embeddings)
    
    def _semantic_search(self,
                        query: str,
                        cells: List[Dict],
                        top_k: int = 20,
                        exclude_year_values: bool = True) -> List[Tuple[int, float]]:
        """
        Fast semantic search using bi-encoder
        ENHANCED: Filter out year values (2007, 2008, etc.) for financial queries
        
        Args:
            query: Search query
            cells: List of cell dicts
            top_k: Number of results
            exclude_year_values: If True, filter out values that look like years
        
        Returns: List of (cell_index, similarity_score)
        """
        if not cells:
            return []
        
        # ENHANCEMENT: Filter out year values for financial queries
        if exclude_year_values:
            filtered_cells = []
            filtered_indices = []
            
            for i, cell in enumerate(cells):
                value = cell['value']
                label = cell['label']
                
                # Skip DATE labels with year-like values (1900-2100)
                if label == 'DATE' and isinstance(value, (int, float)) and 1900 <= value <= 2100:
                    continue
                
                # Skip values that are exactly 4 digits in range 1900-2100
                if isinstance(value, (int, float)) and value == int(value) and 1900 <= int(value) <= 2100:
                    # Check if it's NOT actually a monetary value in thousands
                    # (e.g., $2013 million would be acceptable)
                    if label not in ['MONEY', 'REVENUE', 'INCOME', 'EXPENSE']:
                        continue
                
                filtered_cells.append(cell)
                filtered_indices.append(i)
            
            if not filtered_cells:
                # Fallback: use all cells if filtering removed everything
                filtered_cells = cells
                filtered_indices = list(range(len(cells)))
        else:
            filtered_cells = cells
            filtered_indices = list(range(len(cells)))
        
        # Encode query
        query_embedding = self.bi_encoder.encode(
            query,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        
        # Get/compute cell embeddings for filtered cells
        cell_embeddings = self._compute_embeddings(filtered_cells)
        
        # Compute cosine similarities
        similarities = util.cos_sim(query_embedding, cell_embeddings)[0]
        
        # Get top-k
        top_results = torch.topk(similarities, k=min(top_k, len(filtered_cells)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            # Map back to original indices
            original_idx = filtered_indices[int(idx)]
            results.append((original_idx, float(score)))
        
        return results
    
    def _rerank(self,
               query: str,
               cells: List[Dict],
               initial_results: List[Tuple[int, float]],
               top_k: int = 10) -> List[Tuple[int, float, float]]:
        """
        Rerank using cross-encoder for better accuracy
        Returns: List of (cell_index, bi_score, cross_score)
        """
        if not self.use_reranking or not initial_results:
            return [(idx, bi_score, bi_score) for idx, bi_score in initial_results]
        
        # Prepare pairs for cross-encoder
        pairs = []
        for idx, _ in initial_results:
            cell = cells[idx]
            pairs.append([query, cell['cell_text']])
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        # Combine scores: 40% bi-encoder + 60% cross-encoder
        combined_results = []
        for (idx, bi_score), cross_score in zip(initial_results, cross_scores):
            combined_score = 0.4 * bi_score + 0.6 * cross_score
            combined_results.append((idx, bi_score, float(cross_score)))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: 0.4 * x[1] + 0.6 * x[2], reverse=True)
        
        return combined_results[:top_k]
    
    def retrieve_for_question(self,
                            question: str,
                            kg: nx.MultiDiGraph,
                            top_k: int = 5,
                            temporal_filter: Optional[str] = None) -> List[SemanticMatch]:
        """
        Main retrieval function for a complete question
        
        Args:
            question: Full question text
            kg: Knowledge graph
            top_k: Number of results to return
            temporal_filter: Optional temporal constraint (e.g., "2013")
        
        Returns:
            List of SemanticMatch objects, sorted by relevance
        """
        # Extract table cells
        cells = self._extract_table_cells(kg)
        
        if not cells:
            print("  [Semantic] No table cells found in KG")
            return []
        
        print(f"  [Semantic] Retrieved {len(cells)} table cells from KG")
        
        # Apply temporal filter if provided
        if temporal_filter:
            filtered_cells = []
            temporal_normalized = re.sub(r'[\s\.\,\-]', '', temporal_filter.lower())
            
            for cell in cells:
                cell_context = re.sub(r'[\s\.\,\-]', '', cell['context'].lower())
                if temporal_normalized in cell_context:
                    filtered_cells.append(cell)
            
            if filtered_cells:
                print(f"  [Semantic] Filtered to {len(filtered_cells)} cells matching '{temporal_filter}'")
                cells = filtered_cells
        
        # Semantic search
        initial_results = self._semantic_search(question, cells, top_k=20)
        
        if not initial_results:
            print("  [Semantic] No matching cells found")
            return []
        
        print(f"  [Semantic] Top {len(initial_results)} candidates from bi-encoder")
        
        # Rerank with cross-encoder
        reranked_results = self._rerank(question, cells, initial_results, top_k=top_k)
        
        # Convert to SemanticMatch objects
        matches = []
        for idx, bi_score, cross_score in reranked_results:
            cell = cells[idx]
            
            match = SemanticMatch(
                value=cell['value'],
                text=cell['text'],
                context=cell['context'],
                score=0.4 * bi_score + 0.6 * cross_score,  # Combined score
                node_id=cell['node_id'],
                label=cell['label'],
                semantic_score=bi_score,
                rerank_score=cross_score if self.use_reranking else None
            )
            matches.append(match)
        
        return matches
    
    def retrieve_for_subtask(self,
                           subtask: str,
                           question: str,
                           kg: nx.MultiDiGraph,
                           top_k: int = 3) -> List[SemanticMatch]:
        """
        Retrieve cells for a specific sub-task/entity
        
        Args:
            subtask: Specific entity or sub-question (e.g., "debt securities revenue")
            question: Full question for context
            kg: Knowledge graph
            top_k: Number of results
        
        Returns:
            List of SemanticMatch objects
        """
        # Combine subtask + question for better context
        combined_query = f"{subtask} | Context: {question}"
        
        return self.retrieve_for_question(
            combined_query,
            kg,
            top_k=top_k
        )
    
    def retrieve_multi_args(self,
                          question: str,
                          kg: nx.MultiDiGraph,
                          arg_descriptions: Dict[str, str],
                          temporal_filter: Optional[str] = None) -> Dict[str, SemanticMatch]:
        """
        Retrieve multiple arguments for a question
        
        Args:
            question: Full question
            kg: Knowledge graph
            arg_descriptions: Dict mapping arg names to descriptions
                e.g., {"part": "debt securities revenue", "whole": "total revenue"}
            temporal_filter: Optional temporal constraint
        
        Returns:
            Dict mapping arg names to SemanticMatch objects
        """
        results = {}
        
        for arg_name, arg_desc in arg_descriptions.items():
            # Create specific query for this argument
            query = f"{arg_desc} | Question: {question}"
            
            matches = self.retrieve_for_question(
                query,
                kg,
                top_k=3,
                temporal_filter=temporal_filter
            )
            
            if matches:
                results[arg_name] = matches[0]  # Take best match
                print(f"  [Semantic] {arg_name}: {matches[0].value} (score={matches[0].score:.3f})")
        
        return results


# Singleton instance for reuse across pipeline
_semantic_retriever_instance = None

def get_semantic_retriever(use_reranking: bool = True) -> SemanticTableRetriever:
    """Get or create singleton semantic retriever"""
    global _semantic_retriever_instance
    
    if _semantic_retriever_instance is None:
        _semantic_retriever_instance = SemanticTableRetriever(use_reranking=use_reranking)
    
    return _semantic_retriever_instance
