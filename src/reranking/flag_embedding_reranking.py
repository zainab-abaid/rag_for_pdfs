from typing import List, Dict, Any, Optional
import os
from transformers import AutoTokenizer, AutoModel

# Cache the reranker instance
_reranker_cache: Optional[Any] = None

def _get_reranker():
    """Lazy load and cache the reranker model."""
    global _reranker_cache
    if _reranker_cache is None:
        from FlagEmbedding import FlagReranker
        model_name = os.getenv("FLAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        cache_dir = "./models"
        _ = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        _ = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        _reranker_cache = FlagReranker(model_name, device="cpu")
    return _reranker_cache

def rerank_flag_embedding(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    """
    FlagEmbedding reranker using BAAI's reranker models.
    Returns indices sorted by reranker scores (descending).
    Compatible with CPU.
    """
    if not items:
        return []
    
    try:
        import numpy as np
        
        # Get cached reranker instance
        reranker = _get_reranker()
        
        # Prepare query-document pairs, filtering out empty texts
        texts = [item.get("text", "") or "" for item in items]
        pairs = [[query, text] for text in texts]
        
        # Compute scores
        batch_size = 16   # you can adjust this
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            batch_scores = reranker.compute_score(batch, normalize=True)
            scores.extend(batch_scores)        
        # Ensure scores are a list of floats
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [float(scores)]
        
        # Sort indices by score (descending)
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored_indices]
        
    except ImportError:
        print("WARNING: FlagEmbedding not installed. Install with: pip install -U FlagEmbedding")
        return list(range(len(items)))
    except Exception as e:
        print(f"WARNING: FlagEmbedding reranking failed: {e}")
        return list(range(len(items)))