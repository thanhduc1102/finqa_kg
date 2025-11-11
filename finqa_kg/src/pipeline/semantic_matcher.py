"""
Semantic Entity Matcher - Use embeddings instead of keyword matching
This is a CRITICAL improvement to fix wrong value retrieval!
"""

from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticEntityMatcher:
    """
    Match question phrases to KG entities using semantic embeddings
    
    This replaces keyword-based matching which is too brittle:
    - "available-for-sale investments" matches "investments" (too broad!)
    - "net earnings" matches "earnings" (missing "net")
    
    Semantic matching understands:
    - "available-for-sale investments" ≈ "available-for-sale investments" (exact match)
    - "available-for-sale investments" ≉ "total investments" (different semantic)
    - "net earnings" ≈ "net earnings" but ≉ "gross earnings"
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with sentence-transformers model
        
        Models to consider:
        - 'all-MiniLM-L6-v2': Fast, 384 dim, good balance
        - 'all-mpnet-base-v2': Best quality, 768 dim, slower
        - 'paraphrase-MiniLM-L6-v2': Good for paraphrases
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"[SemanticMatcher] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Cache embeddings to avoid recomputation
        self.embedding_cache = {}
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        self.embedding_cache[text] = embedding
        return embedding
    
    def match_entities(self, 
                       query_text: str, 
                       candidate_entities: List[Dict],
                       top_k: int = 10,
                       min_similarity: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Match query text to candidate entities using semantic similarity
        
        Args:
            query_text: Text from question (e.g., "available-for-sale investments")
            candidate_entities: List of entity dicts with 'context', 'text', 'value' fields
            top_k: Return top K matches
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of (entity, similarity_score) tuples, sorted by similarity desc
        """
        if not candidate_entities:
            return []
        
        # Encode query
        query_emb = self.encode(query_text)
        
        # Encode candidates (use context for richer matching)
        candidate_texts = []
        for entity in candidate_entities:
            # Combine entity text + context for better matching
            text = entity.get('text', '')
            context = entity.get('context', '')
            
            # Use row label from context if available (for table entities)
            if 'table[' in context:
                # Extract: "table[available-for-sale investments]: col=..."
                import re
                match = re.search(r'table\[([^\]]+)\]', context)
                if match:
                    row_label = match.group(1)
                    combined = f"{row_label} {text}"
                else:
                    combined = f"{text} {context[:100]}"
            else:
                combined = f"{text} {context[:100]}"
            
            candidate_texts.append(combined)
        
        # Encode all candidates
        candidate_embs = np.array([self.encode(t) for t in candidate_texts])
        
        # Compute cosine similarities
        similarities = cosine_similarity([query_emb], candidate_embs)[0]
        
        # Filter by min similarity and sort
        results = []
        for i, sim in enumerate(similarities):
            if sim >= min_similarity:
                results.append((candidate_entities[i], float(sim)))
        
        # Sort by similarity descending
        results.sort(key=lambda x: -x[1])
        
        return results[:top_k]
    
    def match_part_and_whole(self,
                            part_text: str,
                            whole_text: str,
                            all_entities: List[Dict],
                            temporal_constraint: str = None) -> Tuple[Dict, Dict]:
        """
        CRITICAL FIX: Match both part and whole semantically
        
        This ensures:
        1. Part matches "available-for-sale investments" not "total investments"
        2. Whole matches "total cash and investments" not "available-for-sale"
        3. Both respect temporal constraint
        4. Part and whole are DIFFERENT entities
        
        Args:
            part_text: Text for part (e.g., "available-for-sale investments")
            whole_text: Text for whole (e.g., "total cash and investments")
            all_entities: All candidate entities from KG
            temporal_constraint: Temporal filter (e.g., "dec 29 2012")
            
        Returns:
            (part_entity, whole_entity) tuple
        """
        print(f"\n[SemanticMatcher] Matching part/whole:")
        print(f"  Part query: '{part_text}'")
        print(f"  Whole query: '{whole_text}'")
        print(f"  Temporal: {temporal_constraint}")
        print(f"  Candidates: {len(all_entities)} entities")
        
        # Filter by temporal if provided
        if temporal_constraint:
            # Normalize temporal
            import re
            normalized_temporal = re.sub(r'[\s\.\,\-]', '', temporal_constraint.lower())
            
            filtered = []
            for e in all_entities:
                context = e.get('context', '').lower()
                normalized_context = re.sub(r'[\s\.\,\-]', '', context)
                
                if normalized_temporal in normalized_context:
                    filtered.append(e)
            
            print(f"  After temporal filter: {len(filtered)} entities")
            all_entities = filtered if filtered else all_entities
        
        # Match part
        part_matches = self.match_entities(part_text, all_entities, top_k=10)
        
        # Match whole (excluding part matches to ensure different entities)
        part_node_ids = {m[0].get('node_id') for m in part_matches}
        whole_candidates = [e for e in all_entities if e.get('node_id') not in part_node_ids]
        
        whole_matches = self.match_entities(whole_text, whole_candidates, top_k=10)
        
        if not part_matches or not whole_matches:
            print(f"  ERROR: No matches found!")
            return None, None
        
        # Select best part and whole
        part_entity, part_sim = part_matches[0]
        whole_entity, whole_sim = whole_matches[0]
        
        print(f"  Part matched: value={part_entity.get('value')}, sim={part_sim:.3f}")
        print(f"    Context: {part_entity.get('context', '')[:80]}")
        print(f"  Whole matched: value={whole_entity.get('value')}, sim={whole_sim:.3f}")
        print(f"    Context: {whole_entity.get('context', '')[:80]}")
        
        # Validate part < whole (swap if needed)
        part_val = part_entity.get('value', 0)
        whole_val = whole_entity.get('value', 0)
        
        if part_val >= whole_val:
            print(f"  WARNING: part >= whole! Swapping...")
            part_entity, whole_entity = whole_entity, part_entity
        
        return part_entity, whole_entity
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.embedding_cache.clear()


# Global singleton instance
_semantic_matcher = None

def get_semantic_matcher(model_name: str = 'all-MiniLM-L6-v2') -> SemanticEntityMatcher:
    """
    Get global semantic matcher instance (singleton)
    
    Args:
        model_name: Model to use
        
    Returns:
        SemanticEntityMatcher instance
    """
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticEntityMatcher(model_name)
    return _semantic_matcher
