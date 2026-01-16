from typing import List, Dict, Any
import os


def rerank_llm(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    """
    LLM-based reranker using GPT to score relevance.
    Returns indices sorted by LLM-assigned relevance scores (descending).
    
    Note: This can be expensive for many items. Consider using only top-k items.
    """
    try:
        from openai import OpenAI
        import json
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("LLM_RERANKER_MODEL", "gpt-4o")
        max_items = int(os.getenv("LLM_RERANKER_MAX_ITEMS", "50"))  # Limit for cost control
        
        # Limit number of items to rerank (take top by orig_scores)
        if len(items) > max_items:
            scored_items = [(i, items[i], orig_scores[i]) for i in range(len(items))]
            scored_items.sort(key=lambda x: x[2], reverse=True)
            indices_to_rerank = [i for i, _, _ in scored_items[:max_items]]
            items_to_rerank = [item for _, item, _ in scored_items[:max_items]]
        else:
            indices_to_rerank = list(range(len(items)))
            items_to_rerank = items
        
        # Prepare prompt
        docs_text = ""
        for idx, item in enumerate(items_to_rerank):
            text = item.get("text", "")[:2000]  # Truncate to save tokens
            docs_text += f"\n\nDocument {idx}:\n{text}"
        
        prompt = f"""Given the query and documents below, score each document's relevance to the query on a scale of 0-10.
                Return ONLY a JSON array of scores in the same order as the documents, like: [8, 3, 9, 1, ...]

                Query: {query}

                Documents:{docs_text}

                JSON array of scores:"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a relevance scoring assistant. Return only JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        # Parse scores
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        scores = json.loads(content)
        
        if len(scores) != len(items_to_rerank):
            print(f"WARNING: LLM returned {len(scores)} scores but expected {len(items_to_rerank)}")
            return list(range(len(items)))
        
        # Sort reranked items by score
        scored_reranked = [(indices_to_rerank[i], scores[i]) for i in range(len(scores))]
        scored_reranked.sort(key=lambda x: x[1], reverse=True)
        reranked_indices = [i for i, _ in scored_reranked]
        
        # Add items that weren't reranked at the end
        for i in range(len(items)):
            if i not in reranked_indices:
                reranked_indices.append(i)
        
        return reranked_indices
        
    except ImportError:
        print("WARNING: openai not installed. Install with: pip install openai")
        return list(range(len(items)))
    except Exception as e:
        print(f"WARNING: LLM reranking failed: {e}")
        return list(range(len(items)))