# cohereranker.py
from typing import List, Dict, Any, Optional
import os
from ratelimit import limits, sleep_and_retry

# Cache the Cohere client
_cohere_client_cache: Optional[Any] = None

def _load_env_config():
    """Load environment configuration (lazy loaded)."""
    from dotenv import load_dotenv
    load_dotenv()

def _get_cohere_client():
    """Lazy load and cache the Cohere client."""
    global _cohere_client_cache
    if _cohere_client_cache is None:
        _load_env_config()
        
        try:
            import cohere
        except ImportError:
            raise ImportError("Cohere SDK not installed. Run: pip install cohere")
        
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Please set COHERE_API_KEY environment variable")
        
        _cohere_client_cache = cohere.Client(api_key)
    
    return _cohere_client_cache

# 20 requests per minute (free tier limit)
@sleep_and_retry
@limits(calls=20, period=60)
def _cohere_rerank_call(co, model, query, texts, top_n):
    """Rate-limited wrapper for Cohere rerank API."""
    return co.rerank(
        model=model,
        query=query,
        documents=texts,
        top_n=top_n
    )

def rerank_cohere(query: str, items: List[Dict[str, Any]], top_n: Optional[int] = None) -> List[int]:
    """
    Cohere Reranker with automatic rate limiting.
    
    Args:
        query: Query string to rerank against.
        items: List of dicts with at least a "text" key.
        top_n: Optional, how many top results to return. If None, returns all.

    Returns:
        List of indices of `items` sorted by rerank score (descending).
    """
    if not items:
        return []
    
    try:
        co = _get_cohere_client()
        texts = [item.get("text", "") or "" for item in items]
        
        # Get model from environment variable with fallback
        model = os.environ.get("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")
        
        # Rate limiting handled by decorator
        response = _cohere_rerank_call(
            co=co,
            model=model,
            query=query,
            texts=texts,
            top_n=top_n if top_n is not None else len(texts)
        )
        
        return [r.index for r in response.results]
        
    except Exception as e:
        print(f"WARNING: Cohere reranking failed: {e}")
        return list(range(len(items)))