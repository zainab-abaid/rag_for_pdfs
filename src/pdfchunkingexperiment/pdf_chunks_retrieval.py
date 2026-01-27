#!/usr/bin/env python3
"""
PDF Retrieval Evaluation with Hybrid Search + QFR + Reranking + Post-filtering

This script evaluates retrieval performance by:
1. Using hybrid search + Query Fusion Retrieval (QFR) to retrieve top-K chunks from PostgreSQL + pgvector.
2. Applying product post-filtering (none|soft|hard).
3. Applying reranking (none|entity|rrf|mmr|custom). Note: This script only contains entity-based reranking implementation.
4. Extracting top FINAL_K chunks.
5. Fuzzy matching retrieved chunks to JSON section_node_ids.
6. Comparing against ground truth section IDs to evaluate accuracy.

### Fuzzy Matching and Ground Truth Comparison:
- **Fuzzy Matching**: Uses the RapidFuzz library to compute similarity scores between retrieved chunks and ground truth section_node_ids. A match is considered valid if the similarity score exceeds the `FUZZY_THRESHOLD`.
- **Ground Truth Comparison**: Matches the retrieved chunks against the ground truth section IDs provided in DATASET_QUERIES. The logic for determining success is as follows:
  - If at least one ground truth section ID matches any of the retrieved chunks, the retrieval is considered successful, and the success score for that query is set to 1.
  - If no matches are found, the success score remains 0.

### Required Environment Variables:
- **PostgreSQL Settings**:
  - `PG_HOST`: PostgreSQL host (default: localhost).
  - `PG_PORT`: PostgreSQL port (default: 5432).
  - `PG_DB`: Database name.
  - `PG_SCHEMA`: Schema name.
  - `PG_USER`: Database user.
  - `PG_PASSWORD`: Database password.
  - `PG_TABLE_PDF`: Table name (default: pdf_chunks).
- **Embedding Model**:
  - `EMBED_MODEL`: OpenAI embedding model (default: text-embedding-3-small).
  - `EMBED_DIM`: Dimension of embedding vectors (default: 1536).
- **Retrieval and Evaluation Settings**:
  - `RETRIEVE_TOP_K`: Number of top chunks to retrieve (default: 30).
  - `SPARSE_TOP_K`: Number of sparse chunks to retrieve (default: 30).
  - `FINAL_K`: Number of final chunks to extract after reranking (default: 5).
  - `FUZZY_THRESHOLD`: Similarity threshold for fuzzy matching (default: 70).
  - `DATASET_QUERIES`: Path to the dataset queries file (default: ./data/questions_answers/query_dataset_with_qa.csv).
  - `CHUNKING_STRATEGY`: Strategy used for chunking (e.g., recursive).
  - `PDF_RETRIEVAL_LOG_CSV`: Path to the retrieval log file (default: logs/pdf_retrieval_log_<CHUNKING_STRATEGY>.csv).
  - `RERANKER_MODE`: Reranking mode (none|entity|rrf|mmr|custom, default: none).
  - `TEXT_SEARCH_CONFIG`: Text search configuration (default: english).
  - `VERBOSE_LOG`: Enable verbose logging (true/false, default: false).

### Usage Notes:
- Ensure the PostgreSQL database is set up and populated with chunked data before running this script.
- Adjust the `FUZZY_THRESHOLD` and `FINAL_K` values based on the desired precision and recall trade-offs.
"""
from __future__ import annotations
import os
import sys
import json
import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv
from typing import List, Dict, Any
from collections import defaultdict

# Import the existing retrieval pipeline
from src.reranking.entity_extraction import extract_entities_spacy
from src.reranking.entity_based_reranking import rerank_contexts as entity_rerank_contexts
from src.reranking.ner_extras import augment_entities
from src.postfiltering.product_postfilter import apply_postfilter
from pgvector.psycopg import register_vector
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

# ----------------------
# Global Stats for Error Tracking
# ----------------------
class EvaluationStats:
    """Track statistics during evaluation."""
    def __init__(self):
        self.json_parse_errors = 0
        self.retrieval_errors = 0
        self.fuzzy_match_errors = 0
    
    def report(self):
        """Print summary of errors encountered."""
        if any([self.json_parse_errors, self.retrieval_errors, self.fuzzy_match_errors]):
            print("\n Error Summary:")
            if self.json_parse_errors:
                print(f"JSON parsing errors: {self.json_parse_errors}")
            if self.retrieval_errors:
                print(f"Retrieval errors: {self.retrieval_errors}")
            if self.fuzzy_match_errors:
                print(f"Fuzzy matching errors: {self.fuzzy_match_errors}")

eval_stats = EvaluationStats()

# ----------------------
# Environment Variables
# ----------------------
PG_HOST = os.environ.get("PG_HOST")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB = os.environ.get("PG_DB")
PG_SCHEMA = os.environ.get("PG_SCHEMA")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_TABLE_PDF = os.environ.get("PG_TABLE_PDF", "pdf_chunks")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))

RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "30"))
SPARSE_TOP_K = int(os.environ.get("SPARSE_TOP_K", "30"))
FINAL_K = int(os.environ.get("FINAL_K", "5"))
FUZZY_THRESHOLD = int(os.environ.get("FUZZY_THRESHOLD", "70"))

DATASET_QUERIES = os.environ.get("DATASET_QUERIES", "./data/questions_answers/query_dataset_with_qa.csv")
CHUNKING_STRATEGY = os.environ.get("CHUNKING_STRATEGY", "recursive")
base_log = os.environ.get("PDF_RETRIEVAL_LOG_CSV", "logs/pdf_retrieval_log.csv")
stem, ext = os.path.splitext(base_log)
PDF_RETRIEVAL_LOG_CSV = f"{stem}_{CHUNKING_STRATEGY}{ext}"
RERANKER_MODE = os.environ.get("RERANKER_MODE", "none")
TEXT_SEARCH_CONFIG = os.environ.get("TEXT_SEARCH_CONFIG", "english").lower()
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "false").lower() == "true"

# ----------------------Helper: Env Var----------------
def _env_str(name: str, default: str = "") -> str:
    """Get environment variable with case-insensitive fallback."""
    v = os.getenv(name)
    if v is None:
        v = os.getenv(name.lower(), default)
    return v if v is not None else default

def _env_float(name: str, default: float) -> float:
    s = _env_str(name, "")
    try:
        return float(s) if s != "" else default
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    s = _env_str(name, "")
    try:
        return int(s) if s != "" else default
    except Exception:
        return default

# ----------------------Reranker Dispatch------------
RERANKER_DISPATCH = {
    "none": lambda query, chunks, scores, embed_model: list(range(len(chunks))),
    "entity": None,  # Will be set to actual function
    "rrf": lambda query, chunks, scores, embed_model: list(range(len(chunks))),
    "mmr": lambda query, chunks, scores, embed_model: list(range(len(chunks))),
    "custom": lambda query, chunks, scores, embed_model: list(range(len(chunks))),
}

RERANK_CFG = {
    "orig_weight": _env_float("RERANK_ORIG_WEIGHT", 0.60),
    "use_fuzzy": (_env_str("RERANK_USE_FUZZY", "true").lower() != "false"),
    "fuzzy_min": _env_int("RERANK_FUZZY_MIN", 85),
    "prod_exact": _env_float("RERANK_PRODUCT_EXACT_BONUS", 1.00),
    "prod_fuzzy": _env_float("RERANK_PRODUCT_FUZZY_BONUS", 0.35),
    "gen_exact": _env_float("RERANK_GENERIC_EXACT_BONUS", 0.45),
    "gen_fuzzy": _env_float("RERANK_GENERIC_FUZZY_BONUS", 0.15),
    "min_ent_toks": _env_int("RERANK_MIN_ENTITY_TOKENS", 1),
    "min_idf_weight": _env_float("RERANK_MIN_IDF_WEIGHT", 0.40),
    "min_active_ents": _env_int("RERANK_MIN_ACTIVE_ENTITIES", 1),
    "idf_mode": _env_str("RERANK_IDF_MODE", "log"),
}

def entity_based_reranking(query: str, chunks: List[Dict[str, Any]], orig_scores: List[float], embed_model=None) -> List[int]:
    """Entity-based reranking implementation. embed_model is intentionally unused (kept for dispatcher compatibility)."""
    cfg = RERANK_CFG
    try:
        base_ents = extract_entities_spacy(query) or []
        ents = augment_entities(query, base_ents)
        if not ents:
            return list(range(len(chunks)))
        order = entity_rerank_contexts(
            ents,
            chunks,
            orig_scores=orig_scores,
            orig_weight=cfg["orig_weight"],
            use_fuzzy=cfg["use_fuzzy"],
            fuzzy_min=cfg["fuzzy_min"],
            product_exact_bonus=cfg["prod_exact"],
            product_fuzzy_bonus=cfg["prod_fuzzy"],
            generic_exact_bonus=cfg["gen_exact"],
            generic_fuzzy_bonus=cfg["gen_fuzzy"],
            min_entity_tokens=cfg["min_ent_toks"],
            min_idf_weight=cfg["min_idf_weight"],
            min_active_entities=cfg["min_active_ents"],
            idf_mode=cfg["idf_mode"],
            debug=False,
            return_info=False,
        )
        return order
    except Exception as e:
        print(f"[RERANK][ENTITY] failed: {e}")
        return list(range(len(chunks)))


RERANKER_DISPATCH["entity"] = entity_based_reranking

# ----------------------Embedding Model Initialization-----------
def initialize_embedding_model():
    try:
        embed_model = OpenAIEmbedding(model=EMBED_MODEL, dimensions=EMBED_DIM)
        Settings.embed_model = embed_model
        return embed_model
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        sys.exit(1)


def hybrid_retrieve_QFR(query: str, k: int = RETRIEVE_TOP_K, sparse_k: int = SPARSE_TOP_K) -> List[Dict[str, Any]]:
    vector_store = PGVectorStore.from_params(
        database=PG_DB,
        host=PG_HOST,
        password=PG_PASSWORD,
        port=PG_PORT,
        user=PG_USER,
        table_name=PG_TABLE_PDF,
        schema_name=PG_SCHEMA,
        embed_dim=EMBED_DIM,
        hybrid_search=True,
        text_search_config=TEXT_SEARCH_CONFIG,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    vector_retriever = index.as_retriever(vector_store_query_mode="default", similarity_top_k=k)
    text_retriever = index.as_retriever(vector_store_query_mode="sparse", similarity_top_k=sparse_k)

    retriever = QueryFusionRetriever(
        [vector_retriever, text_retriever],
        similarity_top_k=k,
        num_queries=1,
        mode="relative_score",
        use_async=True
    )

    nodes_with_scores = retriever.retrieve(query)
    results = []
    for node_with_score in nodes_with_scores:
        node = node_with_score.node
        score = node_with_score.score or 0.0
        metadata = getattr(node, "metadata", {})
        results.append({
            "chunk_id": getattr(node, "id", f"{metadata.get('source_file','')}_{metadata.get('chunk_index','')}"),  # unique IDs
            "text": node.text,
            "source_file": metadata.get("source_file", ""),
            "chunk_index": metadata.get("chunk_index", ""),
            "score": float(score),
            "vector_score": float(score),
            "text_score": 0.0,
            "metadata": metadata
        })
    return results

# ----------------------Retrieval + Postfilter + Rerank Pipeline----
def retrieve_and_process_chunks(query: str, embed_model: Any, k: int = RETRIEVE_TOP_K, sparse_k: int = SPARSE_TOP_K,
                                final_k: int = FINAL_K, postfilter_mode: str = "none", reranker_mode: str = "none") -> List[Dict[str, Any]]:
    """Retrieve chunks using hybrid search, apply post-filtering and reranking. Returns final_k chunks in dictionary format."""
    try:
        chunks = hybrid_retrieve_QFR(query, k=k, sparse_k=sparse_k)
    except Exception as e:
        eval_stats.retrieval_errors += 1
        print(f"⚠ Retrieval failed: {e}")
        return []

    if not chunks:
        return []

    # Extract original scores
    orig_scores = [c.get("score", 0.0) for c in chunks]

    # Apply post-filtering if enabled
    if postfilter_mode.lower() != "none":
        try:
            chunks, _ = apply_postfilter(chunks, query, mode=postfilter_mode)
            orig_scores = [c.get("score", 0.0) for c in chunks]
        except Exception as e:
            print(f"Post-filtering failed: {e}, continuing without it")

    # Apply reranking if enabled
    reranker_fn = RERANKER_DISPATCH.get(reranker_mode.lower(), RERANKER_DISPATCH["none"])
    try:
        order = reranker_fn(query, chunks, orig_scores, embed_model)
        if not isinstance(order, list) or len(order) != len(chunks):
            print(f"Invalid reranker output, using original order")
            order = list(range(len(chunks)))
        chunks = [chunks[i] for i in order]
    except Exception as e:
        print(f"Reranking failed: {e}, using original order")

    return chunks[:final_k]

# ----------------------Chunk → Section Mapping-----------
def map_chunk_to_sections(chunk_text: str, json_path: str, json_cache: Dict[str, list], verbose: bool = True) -> List[str]:
    """Map a chunk to section_node_ids with detailed debug info."""
    if json_path in json_cache:
        data = json_cache[json_path]
    else:
        if not os.path.exists(json_path):
            if verbose:
                print(f"JSON path does not exist: {json_path}")
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            json_cache[json_path] = data
        except Exception as e:
            eval_stats.json_parse_errors += 1
            if verbose:
                print(f"Error reading JSON {json_path}: {e}")
            return []

    matched_sections = []
    for i, chunk_obj in enumerate(data):
        if not isinstance(chunk_obj, dict):
            continue
        chunk_json_text = chunk_obj.get("text", "")
        metadata = chunk_obj.get("metadata", {})
        section_node_id = metadata.get("section_node_id")  # make sure this key exists in your JSON
        if not chunk_json_text or not section_node_id:
            if verbose:
                print(f"Skipping JSON chunk {i}: missing text or section_node_id")
            continue
        try:
            score = max(
                fuzz.partial_ratio(chunk_text, chunk_json_text),
                fuzz.token_set_ratio(chunk_text, chunk_json_text),
            )

            if score >= FUZZY_THRESHOLD and section_node_id not in matched_sections:
                matched_sections.append(section_node_id)

        except Exception as e:
            eval_stats.fuzzy_match_errors += 1
            if verbose:
                print(f"Fuzzy matching exception: {e}")
            continue

    if verbose:
        print(f"Chunk mapped to sections: {matched_sections}\n")
    return matched_sections

def build_retrieved_context(mapped_chunks_json: str, retrieved_context_json: str, min_text_len: int = 80) -> List[Dict[str, Any]]:
    """Convert CSV-style retrieval output into a clean retrieved_context suitable for answer generation.
    
    Inputs:
      - mapped_chunks_json     : JSON string from CSV (mapped_chunks column)
      - retrieved_context_json : JSON string from CSV (retrieved_context column)
    Returns:
      List of section-level context blocks with globally unique text.
    """
    try:
        mapped_chunks = json.loads(mapped_chunks_json)
        retrieved_nodes = json.loads(retrieved_context_json)
    except Exception:
        return []

    # 1. Build chunk_index -> section_ids mapping (skip empty chunk_index)
    chunk_to_sections: Dict[str, List[str]] = {}
    for entry in mapped_chunks:
        chunk_index = entry.get("chunk_index")
        section_str = entry.get("mapped_sections", "")
        
        # Skip if chunk_index is None or empty string
        if not chunk_index or not section_str:
            continue
            
        chunk_to_sections[str(chunk_index)] = [s for s in section_str.split(";") if s]

    # 2. Group retrieved nodes by section_id
    sections: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "texts": [],
        "section_path": [],
        "doc_title": "",
    })

    seen_texts = set()  # global deduplication across all sections

    for node in retrieved_nodes:
        chunk_index = node.get("chunk_index")
        text = (node.get("text") or "").strip()

        # Skip invalid or duplicate text
        if not text or len(text) < min_text_len or text in seen_texts:
            continue
        
        # Skip if chunk_index is empty
        if not chunk_index:
            continue
            
        seen_texts.add(text)

        section_ids = chunk_to_sections.get(str(chunk_index), [])
        if not section_ids:
            continue

        for sid in section_ids:
            sections[sid]["texts"].append(text)
            sections[sid]["section_path"] = node.get("section_path", [])
            sections[sid]["doc_title"] = node.get("doc_title", "")

    # 3. Stitch texts per section
    retrieved_context: List[Dict[str, Any]] = []
    for section_id, data in sections.items():
        stitched_text = "\n\n".join(data["texts"])

        retrieved_context.append({
            "section_id": section_id,
            "section_path": data["section_path"],
            "doc_title": data["doc_title"],
            "text": stitched_text,
        })

    return retrieved_context


# ----------------------Evaluation----------------------
def evaluate_retrieval(embed_model) -> float:
    """Evaluate retrieval and save results to CSV with fresh data, including retrieved_context and answer."""
    reranker_mode = _env_str("RERANKER_MODE", "none")
    postfilter_mode = _env_str("POSTFILTER_MODE", "none")

    try:
        df_queries = pd.read_csv(DATASET_QUERIES)
    except Exception as e:
        print(f"Failed to read query dataset: {e}")
        return 0.0

    results_rows = []
    json_cache = {}
    successes = []

    # Clear CSV if exists
    if os.path.exists(PDF_RETRIEVAL_LOG_CSV):
        os.remove(PDF_RETRIEVAL_LOG_CSV)

    for idx, row in df_queries.iterrows():
        query = str(row.get("question", "")).strip()
        gt_section_id = str(row.get("gt_section_id", "")).strip()
        source_json = row.get("source_path", "")
        answer_text = str(row.get("answer", "")).strip()

        if not query or not source_json:
            print(f"Skipping row {idx}: missing query or JSON")
            continue

        print(f"\n=== Query {idx} ===\nQuestion: {query}\nGT Section: {gt_section_id}\nJSON: {source_json}\n")

        # Step 1: Retrieve raw chunks (pretrim)
        try:
            raw_chunks = retrieve_and_process_chunks(
                query,
                embed_model=embed_model,
                k=RETRIEVE_TOP_K,
                sparse_k=SPARSE_TOP_K,
                final_k=RETRIEVE_TOP_K,  # get all retrieved for pretrim
                postfilter_mode="none",
                reranker_mode="none"
            )
        except Exception as e:
            eval_stats.retrieval_errors += 1
            print(f"Retrieval failed for query: {e}")
            raw_chunks = []

        # Save full pretrim chunks
        retrieved_full_pretrim_out = []
        for chunk in raw_chunks:
            chunk_text = chunk.get("text", "").strip()
            retrieved_full_pretrim_out.append({
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk_text,
                "source_file": chunk.get("source_file", ""),
                "chunk_index": chunk.get("metadata", {}).get("chunk_index", ""),
                "score": chunk.get("score", 0.0),
                "vector_score": chunk.get("vector_score", 0.0),
                "text_score": chunk.get("text_score", 0.0),
                "metadata": chunk.get("metadata", {})
            })

        # Step 2: Process chunks (apply postfiltering, reranking, final_k trimming)
        processed_chunks = retrieve_and_process_chunks(
            query,
            embed_model=embed_model,
            k=RETRIEVE_TOP_K,
            sparse_k=SPARSE_TOP_K,
            final_k=FINAL_K,
            postfilter_mode=postfilter_mode,
            reranker_mode=reranker_mode
        )

        # Step 3: Map processed chunks to sections and build retrieved_context
        mapped_chunks_out = []
        retrieved_context_out = []

        seen_texts = set()
        retrieval_success = 0
        for chunk in processed_chunks:
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text or chunk_text in seen_texts:
                continue

            seen_texts.add(chunk_text)
            chunk_index = chunk.get("metadata", {}).get("chunk_index", "")
            source_file = chunk.get("source_file", "")
            doc_title = chunk.get("metadata", {}).get("doc_title", "")

            mapped_sections = map_chunk_to_sections(chunk_text, source_json, json_cache, verbose=False)
            if gt_section_id and gt_section_id in mapped_sections:
                retrieval_success = 1

            mapped_chunks_out.append({
                "chunk_index": chunk_index,
                "source_file": source_file,
                "doc_title": doc_title,
                "mapped_sections": ";".join(mapped_sections)
            })

            retrieved_context_out.append({
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk_text,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "score": chunk.get("score", 0.0),
                "vector_score": chunk.get("vector_score", 0.0),
                "text_score": chunk.get("text_score", 0.0),
                "metadata": chunk.get("metadata", {})
            })

        successes.append(retrieval_success)

        # Append row for CSV
        results_rows.append({
            "question": query,
            "answer": answer_text,
            "gt_section_id": gt_section_id,
            "retrieval_success": retrieval_success,
            "mapped_chunks": json.dumps(mapped_chunks_out, ensure_ascii=False),
            "retrieved_context": json.dumps(retrieved_context_out, ensure_ascii=False),
            "retrieved_full_pretrim": json.dumps(retrieved_full_pretrim_out, ensure_ascii=False),
            "reranker_mode": reranker_mode,
            "postfilter_mode": postfilter_mode
        })

        print(f"Retrieval success for this query: {retrieval_success}\n{'-'*60}")

    # Write all rows to CSV
    df_out = pd.DataFrame(results_rows)
    os.makedirs(os.path.dirname(PDF_RETRIEVAL_LOG_CSV), exist_ok=True)
    df_out.to_csv(PDF_RETRIEVAL_LOG_CSV, index=False)
    overall_score = sum(successes) / len(successes) if successes else 0.0
    print(f"\nOverall retrieval score: {overall_score:.4f} ({overall_score*100:.2f}%)")
    eval_stats.report()
    return overall_score

def main():
    embed_model = initialize_embedding_model()
    evaluate_retrieval(embed_model)

if __name__ == "__main__":
    main()
