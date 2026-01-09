#!/usr/bin/env python3
"""
PDF Retrieval Evaluation with Hybrid Search + QFR + Reranking + Post-filtering

This script evaluates retrieval performance by:
1. Using hybrid retrieval (vector + optional sparse/BM25 with RRF)
2. Applying product post-filtering (none|soft|hard)
3. Applying reranking (none|entity|rrf|mmr|custom)
4. Extracting top FINAL_K chunks
5. Fuzzy matching retrieved chunks to JSON section_node_ids
6. Comparing against ground truth section IDs
"""
from __future__ import annotations
import os
import sys
import json
import psycopg
import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv
from typing import List, Dict, Any

# Import the existing retrieval pipeline
from src.reranking.entity_extraction import extract_entities_spacy
from src.reranking.entity_based_reranking import rerank_contexts as entity_rerank_contexts
from src.reranking.ner_extras import augment_entities
from src.postfiltering.product_postfilter import apply_postfilter
from pgvector.psycopg import register_vector
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np

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
                print(f"   JSON parsing errors: {self.json_parse_errors}")
            if self.retrieval_errors:
                print(f"   Retrieval errors: {self.retrieval_errors}")
            if self.fuzzy_match_errors:
                print(f"   Fuzzy matching errors: {self.fuzzy_match_errors}")

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
PDF_RETRIEVAL_LOG_CSV = os.environ.get("PDF_RETRIEVAL_LOG_CSV", "logs/pdf_retrieval_log.csv")

RERANKER_MODE = os.environ.get("RERANKER_MODE", "none")
TEXT_SEARCH_CONFIG = os.environ.get("TEXT_SEARCH_CONFIG", "english").lower()

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

def rerank_with_entity(query: str, chunks: List[Dict[str, Any]], embed_model: Any, orig_scores: List[float]) -> List[int]:
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

RERANKER_DISPATCH["entity"] = rerank_with_entity

# ----------------------Embedding Model Initialization-----------
def initialize_embedding_model():
    try:
        embed_model = OpenAIEmbedding(model=EMBED_MODEL, dimensions=EMBED_DIM)
        Settings.embed_model = embed_model
        return embed_model
    except Exception as e:
        print(f" Failed to initialize embedding model: {e}")
        sys.exit(1)

# ----------------------Vector Retrieval-----------
def vector_retrieve_for_pdf(query: str, embed_model: Any, k: int = RETRIEVE_TOP_K, sparse_k: int = SPARSE_TOP_K) -> List[Dict[str, Any]]:
    query_embedding_raw = embed_model.get_text_embedding(query)
    # Convert to numpy array for pgvector
    query_embedding = np.array(query_embedding_raw, dtype=np.float32)
    
    
    with psycopg.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD) as conn:
        register_vector(conn)
        
        with conn.cursor() as cur:
            # Vector-only search
            cur.execute(
                f"""
                SELECT id, text, source_file, chunk_index,
                       1 - (embedding <=> %s::vector) AS similarity,
                       0.0 AS text_score,
                       1 - (embedding <=> %s::vector) AS rrf_score
                FROM {PG_SCHEMA}.{PG_TABLE_PDF}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, query_embedding, query_embedding, k)
            )
            rows = cur.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "chunk_id": row[0],
            "text": row[1],
            "source_file": row[2],
            "chunk_index": row[3],
            "vector_score": float(row[4]),
            "text_score": float(row[5]),
            "score": float(row[6]),
            "metadata": {"source_file": row[2], "chunk_index": row[3]}
        })
    return results

# ----------------------Retrieval + Postfilter + Rerank Pipeline----
def retrieve_and_process_chunks(query: str, embed_model: Any, k: int = RETRIEVE_TOP_K, sparse_k: int = SPARSE_TOP_K,
                                final_k: int = FINAL_K, postfilter_mode: str = "none", reranker_mode: str = "none") -> List[Dict[str, Any]]:
    try:
        chunks = vector_retrieve_for_pdf(query, embed_model, k=k, sparse_k=sparse_k)
    except Exception as e:
        eval_stats.retrieval_errors += 1
        print(f" Retrieval failed: {e}")
        return []

    if not chunks:
        return []

    orig_scores = [c["score"] for c in chunks]

    if postfilter_mode.lower() != "none":
        try:
            chunks, _ = apply_postfilter(chunks, query, mode=postfilter_mode)
            orig_scores = [c.get("score", 0.0) for c in chunks]
        except Exception as e:
            print(f" Post-filtering failed: {e}, continuing without it")

    reranker_fn = RERANKER_DISPATCH.get(reranker_mode.lower(), RERANKER_DISPATCH["none"])
    try:
        order = reranker_fn(query, chunks, orig_scores, embed_model)
        if not isinstance(order, list) or len(order) != len(chunks):
            order = list(range(len(chunks)))
        chunks = [chunks[i] for i in order]
    except Exception as e:
        print(f" Reranking failed: {e}, using original order")

    return chunks[:final_k]

# ----------------------Chunk → Section Mapping-----------
def map_chunk_to_sections(chunk_text: str, json_path: str, json_cache: Dict[str, list]) -> List[str]:
    if json_path in json_cache:
        data = json_cache[json_path]
    else:
        if not os.path.exists(json_path):
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            json_cache[json_path] = data
        except Exception as e:
            eval_stats.json_parse_errors += 1
            print(f" Error reading JSON {json_path}: {e}")
            return []

    matched_sections = []
    for chunk_obj in data:
        if not isinstance(chunk_obj, dict):
            continue
        chunk_json_text = chunk_obj.get("text", "")
        metadata = chunk_obj.get("metadata", {})
        section_node_id = metadata.get("section_node_id")
        if not chunk_json_text or not section_node_id:
            continue
        try:
            score = max(fuzz.partial_ratio(chunk_text, chunk_json_text),
                        fuzz.token_set_ratio(chunk_text, chunk_json_text))
            if score >= FUZZY_THRESHOLD and section_node_id not in matched_sections:
                matched_sections.append(section_node_id)
        except Exception:
            eval_stats.fuzzy_match_errors += 1
            continue
    return matched_sections

# ----------------------Evaluation----------------------

def evaluate_retrieval(embed_model) -> float:
    """
    Evaluate retrieval performance based on whether any of the top FINAL_K
    retrieved chunks map to the GT section ID.
    """
    reranker_mode = _env_str("RERANKER_MODE", "none")
    postfilter_mode = _env_str("POSTFILTER_MODE", "none")

    try:
        df_queries = pd.read_csv(DATASET_QUERIES)
    except Exception as e:
        print(f" Failed to read query dataset: {e}")
        return 0.0

    try:
        from tqdm import tqdm
        iterator = tqdm(df_queries.iterrows(), total=len(df_queries), desc="Evaluating")
    except ImportError:
        iterator = df_queries.iterrows()

    results_log = []
    successes = []
    json_cache = {}

    for idx, row in iterator:
        query = row["question"]
        gt_section_id = str(row["gt_section_id"]).strip()
        source_json = row["source_path"]

        if not gt_section_id or not source_json:
            continue  # skip if missing GT or JSON

        # Step 1: Retrieve chunks
        try:
            retrieved_chunks = retrieve_and_process_chunks(
                query,
                embed_model=embed_model,
                k=RETRIEVE_TOP_K,
                sparse_k=SPARSE_TOP_K,
                final_k=FINAL_K,
                postfilter_mode=postfilter_mode,
                reranker_mode=reranker_mode
            )
        except Exception as e:
            print(f" Retrieval failed for query {idx}: {e}")
            retrieved_chunks = []

        # Step 2: Map chunks to JSON sections
        success = 0
        mapped_chunks_info = []

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "")
            mapped_sections = map_chunk_to_sections(chunk_text, source_json, json_cache)

            mapped_chunks_info.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "mapped_sections": ";".join(mapped_sections),
                "text_snippet": chunk_text[:100],
                "source_file": chunk.get("source_file", "")
            })

            # Mark success if GT section is in any mapped sections
            if gt_section_id in mapped_sections:
                success = 1

        successes.append(success)

        # Step 3: Log results
        results_log.append({
            "question": query,
            "gt_section_id": gt_section_id,
            "retrieval_success": success,
            "mapped_chunks": json.dumps(mapped_chunks_info),
            "reranker_mode": reranker_mode,
            "postfilter_mode": postfilter_mode,
        })

        print(f"Query: {query[:50]}... | GT Section: {gt_section_id} | Success: {success}")
        for mc in mapped_chunks_info:
            print(f"  Chunk {mc['chunk_id']} | Source: {mc['source_file']} | Mapped Sections: {mc['mapped_sections']}")

    # Save log
    if results_log:
        try:
            os.makedirs(os.path.dirname(PDF_RETRIEVAL_LOG_CSV), exist_ok=True)
            pd.DataFrame(results_log).to_csv(PDF_RETRIEVAL_LOG_CSV, index=False)
            print(f"\n Results saved to: {PDF_RETRIEVAL_LOG_CSV}")
        except Exception as e:
            print(f" Failed to save results log: {e}")

    # Step 4: Compute overall success
    if not successes:
        print("\n No valid queries evaluated")
        return 0.0

    overall_score = sum(successes) / len(successes)
    print(f"\n{'=' * 60}")
    print(f" SECTION-BASED EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  Reranker Mode: {reranker_mode}")
    print(f"  Postfilter Mode: {postfilter_mode}")
    print(f"  Retrieve Top-K: {RETRIEVE_TOP_K}")
    print(f"  Final K: {FINAL_K}")
    print(f"  Fuzzy Threshold: {FUZZY_THRESHOLD}")
    print(f"\nResults:")
    print(f"  Total queries evaluated: {len(successes)}")
    print(f"  Successful retrievals: {sum(successes)}")
    print(f"  Failed retrievals: {len(successes) - sum(successes)}")
    print(f"  Overall retrieval score: {overall_score:.4f} ({overall_score * 100:.2f}%)")
    print(f"{'=' * 60}\n")

    return overall_score


# ----------------------
# Main
# ----------------------
def main():
    embed_model = initialize_embedding_model()
    evaluate_retrieval(embed_model)

if __name__ == "__main__":
    main()
