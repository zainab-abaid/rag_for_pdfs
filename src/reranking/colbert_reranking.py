from typing import List, Dict, Any
import os

# Cache model globally
_reranker_model = None

def rerank_colbert(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    """
    ColBERT-based reranker using ragatouille.
    Returns indices sorted by ColBERT relevance scores (descending).
    """
    global _reranker_model
    try:
        from ragatouille import RAGPretrainedModel

        # Initialize model only once
        if _reranker_model is None:
            model_name = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
            _reranker_model = RAGPretrainedModel.from_pretrained(model_name)
        reranker = _reranker_model

        # Extract texts
        texts = [item.get("text", "") for item in items]

        # Rerank using ColBERT
        results = reranker.rerank(query=query, documents=texts, k=len(texts))

        # Attempt to extract content in a flexible way
        reranked_indices = []
        for result in results:
            # result could be dict or tuple/list
            if isinstance(result, dict):
                content = result.get("content", "")
            elif isinstance(result, (tuple, list)) and len(result) >= 2:
                content = result[1]  # assume (score, content)
            else:
                content = str(result)

            # Map back to original index
            for i, text in enumerate(texts):
                if i not in reranked_indices and text.strip() == content.strip():
                    reranked_indices.append(i)
                    break

        # Add any missing indices (fallback)
        for i in range(len(items)):
            if i not in reranked_indices:
                reranked_indices.append(i)

        return reranked_indices

    except ImportError:
        print("WARNING: ragatouille not installed. Install with: pip install ragatouille")
        return list(range(len(items)))
    except Exception as e:
        print(f"WARNING: ColBERT reranking failed: {e}")
        return list(range(len(items)))
