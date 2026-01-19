#!/usr/bin/env python3
"""
Hybrid Retrieval and Reranking Script

This script performs hybrid retrieval and reranking for section-wise chunks stored in a PostgreSQL database with pgvector.
It retrieves relevant sections based on a query, applies post-filtering and reranking strategies, and logs the results for 
further evaluation.

Usage:
    uv run python src/rerankingexperiments/retrieve_with_diff_rerankers.py

Environment Variables (REQUIRED):
    PG_HOST, PG_PORT, PG_DB, PG_SCHEMA, PG_USER, PG_PASSWORD: PostgreSQL connection details.
    PG_TABLE: Name of the table containing section-based chunks.
    DATASET_QUERIES: Path to the input dataset CSV file (e.g., `data/query_dataset_with_qa.csv`).
    RETRIEVAL_LOG_CSV: Path to save the retrieval log (e.g., `logs/retrieval_log_TOPK5_none.csv`).
    RERANKER_MODE: Reranking strategy to use (e.g., `none`, `entity`, `colbert`, `mmr`, `llm`, `flag`).
    POSTFILTER_MODE: Post-filtering strategy to use (e.g., `none`, `soft`, `hard`).

Environment Variables (OPTIONAL):
    RETRIEVE_TOP_K: Number of top results to retrieve in the initial dense search (default: 30).
    SPARSE_TOP_K: Number of top results to retrieve in the sparse search (default: 30).
    FINAL_K: Number of final results to return after reranking and trimming (default: 5).
    EMBED_MODEL, EMBED_DIM: Embedding model and dimensions for vector search.

Outputs:
    - Logs the retrieved contexts and metadata to the specified `RETRIEVAL_LOG_CSV` file.
    - Includes detailed telemetry such as ground truth ranks, post-filtering hits, and reranking results.
"""

from __future__ import annotations
import importlib
import os, json, csv, re
from typing import List, Dict, Any, Tuple, Optional, Callable

from dotenv import load_dotenv

from src.reranking.ner_extras import augment_entities
load_dotenv()

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

import psycopg

from src.reranking.entity_extraction import extract_entities_spacy
from  src.reranking.entity_based_reranking import rerank_contexts as entity_rerank_contexts
from src.postfiltering.product_postfilter import apply_postfilter
from src.reranking.ner_extras import augment_entities

# --- env ---

PG_HOST = os.environ["PG_HOST"]
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB = os.environ["PG_DB"]
PG_SCHEMA = os.environ["PG_SCHEMA"]
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

PG_TABLE = os.environ.get("PG_TABLE", "idx_section_based_chunking")
TEXT_SEARCH_CONFIG = os.environ.get("TEXT_SEARCH_CONFIG", "english").lower()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))

RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "30"))
SPARSE_TOP_K   = int(os.environ.get("SPARSE_TOP_K",   "30"))
FINAL_K = int(os.environ.get("FINAL_K", "5"))

DATASET_QUERIES = os.environ.get("dataset_queries", "./data/manual_queries_from_chunks.csv")
# Default output to logs/ directory with sensible naming
RETRIEVAL_LOG_CSV = os.environ.get("RETRIEVAL_LOG_CSV") or os.environ.get("retrieval_log_csv") or None  # Will be set in demo() based on dataset name

# Optional: set embed model if not configured globally elsewhere
try:
    from llama_index.core.settings import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, dimensions=EMBED_DIM)
except Exception:
    pass

# ---------------- Env helpers ----------------
def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        v = os.getenv(name.lower(), default)
    return (v if v is not None else default)

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

# ---------------- Reranker registry ----------------
# Unified wrapper signature for *all* rerankers we call:
#   rerank_fn(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]
# Must return a permutation (list of indices) for `items`.

def _rerank_none(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    return list(range(len(items)))

def _rerank_entity_based(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    if entity_rerank_contexts is None:
        # If module not importable, fall back
        return list(range(len(items)))
    #ents = extract_entities_spacy(query)  # products first, then others

    base = []
    try:
        base = extract_entities_spacy(query) or []
    except Exception:
        base = []
    ents = augment_entities(query, base)


    # knobs via env (all optional; defaults mirror your reranker defaults)
    orig_weight     = _env_float("RERANK_ORIG_WEIGHT", 0.60)
    use_fuzzy       = (_env_str("RERANK_USE_FUZZY", "true").lower() != "false")
    fuzzy_min       = _env_int("RERANK_FUZZY_MIN", 85)
    prod_exact      = _env_float("RERANK_PRODUCT_EXACT_BONUS", 1.00)
    prod_fuzzy      = _env_float("RERANK_PRODUCT_FUZZY_BONUS", 0.35)
    gen_exact       = _env_float("RERANK_GENERIC_EXACT_BONUS", 0.45)
    gen_fuzzy       = _env_float("RERANK_GENERIC_FUZZY_BONUS", 0.15)
    min_ent_toks    = _env_int("RERANK_MIN_ENTITY_TOKENS", 1)
    min_idf_weight  = _env_float("RERANK_MIN_IDF_WEIGHT", 0.40)
    min_active_ents = _env_int("RERANK_MIN_ACTIVE_ENTITIES", 1)
    idf_mode        = _env_str("RERANK_IDF_MODE", "log")

    order = entity_rerank_contexts(
        ents,
        items,
        orig_scores=orig_scores,
        orig_weight=orig_weight,
        use_fuzzy=use_fuzzy,
        fuzzy_min=fuzzy_min,
        product_exact_bonus=prod_exact,
        product_fuzzy_bonus=prod_fuzzy,
        generic_exact_bonus=gen_exact,
        generic_fuzzy_bonus=gen_fuzzy,
        min_entity_tokens=min_ent_toks,
        min_idf_weight=min_idf_weight,
        min_active_entities=min_active_ents,
        idf_mode=idf_mode,
        debug=False,
        return_info=False,
    )
    return order

def _load_custom_callable(spec: str) -> Optional[Callable[[str, List[Dict[str, Any]], List[float]], List[int]]]:
    """
    spec format: 'package.module:callable_name'
    callable must accept (query, items, orig_scores) and return List[int]
    """
    if not spec or ":" not in spec:
        return None
    mod_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        return None
    return fn

def _rerank_via_custom(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    spec = _env_str("RERANKER_SPEC", "")
    try:
        fn = _load_custom_callable(spec)
        if fn is None:
            return list(range(len(items)))
        return fn(query, items, orig_scores)
    except Exception:
        return list(range(len(items)))

# Optional: built-in hooks for 'rrf' and 'mmr' modes via custom modules.
# You can point them with RERANKER_SPEC too; these are convenience fallbacks.
def _rerank_rrf(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    # Try to import a provided RRF reranker if you have one
    spec = _env_str("RERANKER_SPEC", "")
    if spec:
        return _rerank_via_custom(query, items, orig_scores)
    # else, keep order (or write a simple RRF here if you expose per-modality ranks in meta)
    return list(range(len(items)))

def _rerank_mmr(query: str, items: List[Dict[str, Any]], orig_scores: List[float]) -> List[int]:
    spec = _env_str("RERANKER_SPEC", "")
    if spec:
        return _rerank_via_custom(query, items, orig_scores)
    return list(range(len(items)))

# Dispatcher
def _get_reranker() -> Callable[[str, List[Dict[str, Any]], List[float]], List[int]]:
    """
    Returns the appropriate reranker function based on RERANKER_MODE env var.
    Supported modes:
    - none/off: no reranking
    - entity/entity_based: entity-aware reranking
    - colbert: ColBERT reranking
    - flag/flag_embedding: FlagEmbedding reranking
    - llm/gpt: LLM-based reranking
    - rrf: Reciprocal Rank Fusion
    - mmr: Maximal Marginal Relevance
    - custom: custom reranker specified via RERANKER_SPEC
    """
    mode = _env_str("RERANKER_MODE", "none").lower()
    
    if mode in ("none", "", "off"):
        return _rerank_none
    
    if mode in ("entity", "entity_based", "entity-aware", "entity_aware"):
        return _rerank_entity_based
    
    if mode in ("colbert", "colbertv2"):
        from src.reranking.colbert_reranking import rerank_colbert
        return rerank_colbert
    
    if mode in ("flag", "flag_embedding", "flagembedding", "bge"):
        from src.reranking.flag_embedding_reranking import rerank_flag_embedding
        return rerank_flag_embedding
    
    if mode in ("llm", "gpt", "llm_rerank"):
        from src.reranking.llm_reranking import rerank_llm
        return rerank_llm
    
    if mode == "rrf":
        return _rerank_rrf
    
    if mode == "mmr":
        return _rerank_mmr
    
    if mode in ("custom",):
        return _rerank_via_custom
    
    # unknown -> no-op with warning
    print(f"WARNING: Unknown RERANKER_MODE '{mode}', using no reranking")
    return _rerank_none

# ---------- DB helpers ----------
def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return cur.fetchone() is not None

def _resolve_data_table(conn, schema: str, base_table: str) -> str:
    """
    Return the physical data table for PGVectorStore:
      - prefer <schema>.<base_table>
      - else <schema>.data_<base_table>
    Result is a fully-quoted FQTN: "schema"."table"
    """
    if _table_exists(conn, schema, base_table):
        return f'"{schema}"."{base_table}"'
    data_table = f"data_{base_table}"
    if _table_exists(conn, schema, data_table):
        return f'"{schema}"."{data_table}"'
    raise RuntimeError(
        f"Neither '{schema}.{base_table}' nor '{schema}.{data_table}' exists. "
        "Verify ingestion finished in this DB/schema."
    )

def _pg_conn():
    # Debug print to confirm envs used
    return psycopg.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )

# ---------- SQL fetch ----------
def _fetch_section_chunks(conn, section_node_id: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return all chunks for a section, ordered by chunk_index asc."""
    data_fqtn = _resolve_data_table(conn, PG_SCHEMA, PG_TABLE)
    sql = f"""
        SELECT id, text, metadata_
        FROM {data_fqtn}
        WHERE metadata_->>'section_node_id' = %s
        ORDER BY (metadata_->>'chunk_index')::int ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (section_node_id,))
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

def _fetch_any_chunk_with_section(conn, section_node_id: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    data_fqtn = _resolve_data_table(conn, PG_SCHEMA, PG_TABLE)
    sql = f"""
        SELECT id, text, metadata_
        FROM {data_fqtn}
        WHERE metadata_->>'section_node_id' = %s
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (section_node_id,))
        row = cur.fetchone()
        return (row[0], row[1], row[2]) if row else None

# ---------- Formatting / assembly ----------
_HEADER_PREFIX_RE = re.compile(
    r"^(?:Path:\s.*?\n)(?:Title:\s.*?\n)?\s*content:\s*\n",
    flags=re.IGNORECASE | re.DOTALL
)

def _strip_header_wrapper(text: str) -> str:
    """Remove the chunk's 'Path/Title/content:' header if present."""
    if not text:
        return text
    m = _HEADER_PREFIX_RE.match(text)
    return text[m.end():].lstrip() if m else text

def _strip_leading_heading(text: str) -> str:
    """
    If the first non-empty line is a markdown heading (e.g., '### Foo'), remove that line.
    """
    if not text:
        return text
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1
        # strip any blank lines following the heading
        while i < len(lines) and not lines[i].strip():
            i += 1
        return "\n".join(lines[i:]).lstrip()
    return text

def _assemble_section(rows: List[Tuple[str,str,Dict[str,Any]]]) -> Tuple[str, Dict[str,Any]]:
    """
    Given all rows (id, text, meta) of a section, return (assembled_text, canonical_meta).
    - Keep Path/Title/content and the section heading ONLY for the first split.
    - Strip header wrapper and leading heading from subsequent splits.
    - No '-----' separators inside.
    """
    if not rows:
        return "", {}

    assembled_parts: List[str] = []
    first_text, first_meta = rows[0][1], rows[0][2]
    # First split: keep as-is (already has Path/Title/content + heading)
    assembled_parts.append((first_text or "").strip())

    for (cid, ctext, cmeta) in rows[1:]:
        t = (ctext or "")
        t = _strip_header_wrapper(t)
        t = _strip_leading_heading(t)
        if t:
            assembled_parts.append(t.strip())

    return "\n\n".join(assembled_parts).strip(), first_meta

def _format_single_chunk(text: str) -> str:
    """For chunks not belonging to a known section, just pass-through."""
    return (text or "").strip()

# ---------- Retrieval ----------
def hybrid_retrieve(query: str, k: int = RETRIEVE_TOP_K, sparse_k: int = SPARSE_TOP_K) -> List[NodeWithScore]:
    vector_store = PGVectorStore.from_params(
        database=PG_DB,
        host=PG_HOST,
        password=PG_PASSWORD,
        port=PG_PORT,
        user=PG_USER,
        table_name=PG_TABLE,
        schema_name=PG_SCHEMA,
        embed_dim=EMBED_DIM,
        hybrid_search=True,
        text_search_config=TEXT_SEARCH_CONFIG,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    retriever = index.as_retriever(
        similarity_top_k=k,
        vector_store_query_mode="hybrid",
        vector_store_kwargs={"sparse_top_k": sparse_k},
    )
    return retriever.retrieve(query)

from typing import List
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.postgres import PGVectorStore

def hybrid_retrieve_QFR(query: str, k: int = RETRIEVE_TOP_K, sparse_k: int = SPARSE_TOP_K) -> List[NodeWithScore]:
    # Build PGVectorStore as before; keep hybrid enabled so sparse queries work
    vector_store = PGVectorStore.from_params(
        database=PG_DB,
        host=PG_HOST,
        password=PG_PASSWORD,
        port=PG_PORT,
        user=PG_USER,
        table_name=PG_TABLE,
        schema_name=PG_SCHEMA,
        embed_dim=EMBED_DIM,
        hybrid_search=True,
        text_search_config=TEXT_SEARCH_CONFIG,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # Two separate retrievers: dense + sparse
    vector_retriever = index.as_retriever(
        vector_store_query_mode="default",
        similarity_top_k=k,
    )
    text_retriever = index.as_retriever(
        vector_store_query_mode="sparse",
        similarity_top_k=sparse_k,   # interchangeable with sparse_top_k for sparse mode
    )

    # Fuse with calibrated scores (recommended for mixed scales)
    retriever = QueryFusionRetriever(
        [vector_retriever, text_retriever],
        similarity_top_k=k,  # final fused K to return
        num_queries=1,       # disable query generation for now
        mode="relative_score",  #or use "reciprocal_rerank" instead of relative_score
        use_async=True,      # parallelize underlying calls
        # Optional: only for relative_score mode, you can bias toward sparse if it hits more often for you, e.g. 40% dense / 60% sparse:
        # retriever_weights=[0.4, 0.6],
    )

    return retriever.retrieve(query)


def stitch_context(
    query: str,
    k: int = RETRIEVE_TOP_K,
    sparse_k: int = SPARSE_TOP_K,
    final_k: int = FINAL_K,
    postfilter_mode: str = "none",
    debug: bool = False,
):
    """
    Returns a triple:
      (trimmed_post, all_post_pretrim, all_pre_pretrim)

    - all_pre_pretrim  : pre-trim list BEFORE any post-filter or reranking
                         (score-desc after dedup & assembly)
    - all_post_pretrim : pre-trim list AFTER product post-filter + reranking
                         (permutation applied on filtered candidates)
    - trimmed_post     : all_post_pretrim[:final_k]

    Each block dict includes:
      {"section_id","section_path","doc_title","text","meta","orig_score"}
    """
    results = hybrid_retrieve_QFR(query, k=k, sparse_k=sparse_k) # hybrid_retrieve or hybrid_retrieve_QFR(query, k=k, sparse_k=sparse_k)
    if not results:
        return [], [], [], []

    # 1) dedup per section id, keep highest score
    best_by_section: Dict[str, NodeWithScore] = {}
    for r in results:
        node = r.node
        meta = node.metadata or {}
        sec_key = meta.get("section_node_id") or meta.get("chunk_id") or node.node_id
        prev = best_by_section.get(sec_key)
        if prev is None or getattr(r, "score", 0.0) > getattr(prev, "score", 0.0):
            best_by_section[sec_key] = r

    # 2) score-desc order, no trim yet
    ranked_all = sorted(best_by_section.items(), key=lambda kv: getattr(kv[1], "score", 0.0), reverse=True)

    # 3) assemble in THAT order, capturing orig_score aligned
    assembled: List[Dict[str, Any]] = []
    orig_scores: List[float] = []
    seen_ids: set[str] = set()

    with _pg_conn() as conn:
        for sec_key, r in ranked_all:
            node = r.node
            score = float(getattr(r, "score", 0.0))
            meta = node.metadata or {}
            section_node_id = meta.get("section_node_id")

            if section_node_id:
                rows = _fetch_section_chunks(conn, section_node_id)
                if rows:
                    text, first_meta = _assemble_section(rows)
                    if section_node_id not in seen_ids:
                        assembled.append({
                            "section_id": section_node_id,
                            "section_path": first_meta.get("section_path", []),
                            "doc_title": first_meta.get("doc_title", ""),
                            "text": text,
                            "meta": first_meta,
                            "orig_score": score,
                        })
                        orig_scores.append(score)
                        seen_ids.add(section_node_id)
                    continue

            # fallback: standalone chunk
            sid = meta.get("chunk_id") or meta.get("section_node_id") or node.node_id
            if sid not in seen_ids:
                assembled.append({
                    "section_id": sid,
                    "section_path": meta.get("section_path", []),
                    "doc_title": meta.get("doc_title", ""),
                    "text": _format_single_chunk(node.get_content()),
                    "meta": meta,
                    "orig_score": score,
                })
                orig_scores.append(score)
                seen_ids.add(sid)

    # 4) defensive dedup by (section_id, len(text)) preserving first occurrence
    def _dedup_preserve(seq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out, fp = [], set()
        for blk in seq:
            key = (blk["section_id"], len(blk.get("text", "") or ""))
            if key in fp:
                continue
            fp.add(key)
            out.append(blk)
        return out

    # This is the ORIGINAL pre-filter, pre-rerank baseline
    all_pre_pretrim = _dedup_preserve(assembled)

    # 5) PRODUCT POST-FILTER (env-driven: none|soft|hard)
    #    - If no clear product in the query -> passthrough.
    #    - soft: stable-partition (product hits first), order preserved within groups.
    #    - hard: keep only product hits; if none, fall back to original list.
    # Pass mode explicitly to ensure we use the required env value
    candidates, _pf_info = apply_postfilter(all_pre_pretrim, query, mode=postfilter_mode)

    # 6) RERANK via dispatcher (entity/mmr/rrf/none) on the post-filtered candidates
    rerank_fn = _get_reranker()
    try:
        rerank_order = rerank_fn(query, candidates, [b.get("orig_score", 0.0) for b in candidates])
        if not isinstance(rerank_order, list) or len(rerank_order) != len(candidates):
            rerank_order = list(range(len(candidates)))
    except Exception:
        rerank_order = list(range(len(candidates)))

    all_post_pretrim = [candidates[i] for i in rerank_order]

    # 7) trim
    trimmed_post = all_post_pretrim[:final_k]

    if debug:
        print("\n--- Debug GT Tracking ---")
        debug_gt_tracking(query, None, all_pre_pretrim, candidates, rerank_order)

    return trimmed_post, all_post_pretrim, all_pre_pretrim, rerank_order


def debug_gt_tracking(query, gt_section_id, candidates, postfilter_candidates=None, rerank_order=None):
    """
    Print GT presence at each stage:
    - candidates: list of dicts from hybrid/QFR retrieval
    - postfilter_candidates: list after postfiltering/trimming (optional)
    - rerank_order: list of indices returned by reranker (optional)
    """
    print(f"\nQuery: {query}")
    print(f"GT section: {gt_section_id}")

    # Stage 1: initial candidates
    candidate_ids = [c['section_id'] for c in candidates]
    print(f"Hybrid/QFR candidates ({len(candidate_ids)}): {candidate_ids[:10]}{'...' if len(candidate_ids) > 10 else ''}")
    if gt_section_id in candidate_ids:
        print(f"✅ GT found in hybrid/QFR candidates at index {candidate_ids.index(gt_section_id)}")
    else:
        print(f"❌ GT NOT found in hybrid/QFR candidates")

    # Stage 2: post-filter / trimming
    if postfilter_candidates is not None:
        pf_ids = [c['section_id'] for c in postfilter_candidates]
        print(f"Post-filter candidates ({len(pf_ids)}): {pf_ids[:10]}{'...' if len(pf_ids) > 10 else ''}")
        if gt_section_id in pf_ids:
            print(f"✅ GT found after post-filter at index {pf_ids.index(gt_section_id)}")
        else:
            print(f"❌ GT NOT found after post-filter")

    # Stage 3: after rerank
    if rerank_order is not None:
        reranked_ids = [candidates[i]['section_id'] for i in rerank_order]
        if gt_section_id in reranked_ids:
            print(f"✅ GT found after rerank at index {reranked_ids.index(gt_section_id)}")
        else:
            print(f"❌ GT NOT found after rerank")



def _pretty(blocks):
    """Full payload for logs — includes full text for answer generation."""
    return [
        {
            "section_id": b.get("section_id", ""),
            "text": b.get("text", ""),  # Full text, not truncated
            "section_path": b.get("section_path", []),
            "doc_title": b.get("doc_title", ""),
        }
        for b in blocks
    ]


def demo():
    def _env_str(name: str, default: str = "") -> str:
        v = os.getenv(name)
        if v is None:
            v = os.getenv(name.lower(), default)
        return v if v is not None else default

    def _rank_by_section_id(blocks, sid: str) -> int:
        if not sid:
            return -1
        for idx, b in enumerate(blocks):
            if (b.get("section_id") or "") == sid:
                return idx + 1  # 1-based
        return -1

    in_csv = _env_str("DATASET_QUERIES") or DATASET_QUERIES
    out_csv = _env_str("RETRIEVAL_LOG_CSV") or _env_str("retrieval_log_csv")
    if not out_csv:
        raise SystemExit("ERROR: RETRIEVAL_LOG_CSV or retrieval_log_csv environment variable is required.\n"
                        "Set it to the path where you want to save the retrieval log (e.g., logs/retrieval_log.csv).")
    
    # REQUIRED: reranker_mode (no default)
    reranker_mode = _env_str("RERANKER_MODE") or _env_str("reranker_mode")
    if not reranker_mode:
        raise SystemExit("ERROR: RERANKER_MODE or reranker_mode environment variable is required.\n"
                        "Set it to one of: none, entity, rrf, mmr, custom")
    reranker_mode = reranker_mode.lower()

    # Append reranker mode to filename
    base, ext = os.path.splitext(out_csv)
    out_csv = f"{base}_{reranker_mode}{ext}"
    
    # REQUIRED: postfilter_mode (no default)
    postfilter_mode_env = _env_str("POSTFILTER_MODE") or _env_str("postfilter_mode")
    if not postfilter_mode_env:
        raise SystemExit("ERROR: POSTFILTER_MODE or postfilter_mode environment variable is required.\n"
                        "Set it to one of: none, soft, hard")
    postfilter_mode_env = postfilter_mode_env.lower()

    # load dataset
    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"question", "answer", "context", "source_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing columns: {missing}")
        has_gt_id = "gt_section_id" in (reader.fieldnames or [])
        for row in reader:
            if has_gt_id and "gt_section_id" not in row:
                row["gt_section_id"] = ""
            rows.append(row)

    out_fields = [
        # Configuration metadata (from env) - at top for visibility
        "reranker_mode",              # RERANKER_MODE env var value
        "postfilter_mode",            # POSTFILTER_MODE env var value
        
        # Query and ground truth
        "question", "answer", "context", "source_path", "gt_section_id",

        # lists
        "postfilter_pretrim",         # list after product postfilter, BEFORE rerank/trim
        "retrieved_full_pretrim",     # post-rerank, pre-trim
        "retrieved_context",          # post-rerank, trimmed

        # telemetry
        "postfilter_product",
        "postfilter_hits",
        "postfilter_total",

        # ranks (1-based; -1 = not present)
        "gt_rank_postfilter_pre",     # rank in postfilter-pretrim list (before rerank)
        "gt_rank_postfilter_trimmed", # rank in top-K of postfilter-pretrim (before rerank)
        "gt_rank_pre_rerank",         # rank in all_pre_pretrim (baseline, before postfilter & before rerank)
        "gt_rank_post_rerank",        # rank in all_post_pretrim (after rerank)
        "entities",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        total = len(rows)
        for i, row in enumerate(rows, 1):
            q = (row.get("question") or "").strip()
            gt_sid = row.get("gt_section_id", "")

            # entities for log (best-effort)
            base = []
            try:
                base = extract_entities_spacy(q) or []
            except Exception:
                base = []
           # print(f"based ents: {base}")
            ents = augment_entities(q, base)
           # print(f"augmented ents: {ents}")
            if not q:
                writer.writerow({
                    "reranker_mode": reranker_mode,
                    "postfilter_mode": postfilter_mode_env,
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "context": row.get("context", ""),
                    "source_path": row.get("source_path", ""),
                    "gt_section_id": gt_sid,

                    "postfilter_pretrim": "[]",
                    "retrieved_full_pretrim": "[]",
                    "retrieved_context": "[]",

                    "postfilter_product": "",
                    "postfilter_hits": 0,
                    "postfilter_total": 0,

                    "gt_rank_postfilter_pre": -1,
                    "gt_rank_postfilter_trimmed": -1,
                    "gt_rank_pre_rerank": -1,
                    "gt_rank_post_rerank": -1,
                    "entities": json.dumps(ents, ensure_ascii=False),
                })
                print(f"[{i}/{total}] empty question")
                continue

            # stitch_context returns:
            #   trimmed_post (post-rerank, trimmed),
            #   blocks_all_post (post-rerank, pre-trim),
            #   blocks_all_pre (pre-postfilter & pre-rerank, pre-trim)
            blocks_trimmed_post, blocks_all_post, blocks_all_pre, _ = stitch_context(
                q, k=RETRIEVE_TOP_K, sparse_k=SPARSE_TOP_K, final_k=FINAL_K, postfilter_mode=postfilter_mode_env
            )
            debug_gt_tracking(
                query=q,
                gt_section_id=gt_sid,
                candidates=blocks_all_pre,
                postfilter_candidates=blocks_all_post,
                rerank_order=list(range(len(blocks_all_post)))  # if needed
            )

            # --- recompute product postfilter on the SAME baseline (blocks_all_pre) for logging parity ---
            # Pass mode explicitly to ensure we use the required env value
            pf_candidates, pf_info = apply_postfilter(blocks_all_pre, q, mode=postfilter_mode_env)
            pf_mode = pf_info.get("mode", postfilter_mode_env)
            pf_product = pf_info.get("product", "")
            pf_hits = int(pf_info.get("num_hits", 0))
            pf_total = int(pf_info.get("total", len(blocks_all_pre)))

            # ranks
            pre_rank_baseline = _rank_by_section_id(blocks_all_pre, gt_sid) if gt_sid else -1
            post_rank_rerank = _rank_by_section_id(blocks_all_post, gt_sid) if gt_sid else -1

            pf_rank_pre = _rank_by_section_id(pf_candidates, gt_sid) if gt_sid else -1
            # rank within the top-K *after postfilter but before rerank*:
            pf_topk = pf_candidates[:FINAL_K]
            pf_rank_trimmed = _rank_by_section_id(pf_topk, gt_sid) if gt_sid else -1

            writer.writerow({
                "reranker_mode": reranker_mode,
                "postfilter_mode": postfilter_mode_env,  # Use env value, not computed pf_mode
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "context": row.get("context", ""),
                "source_path": row.get("source_path", ""),
                "gt_section_id": gt_sid,

                "postfilter_pretrim": json.dumps(_pretty(pf_candidates), ensure_ascii=False),
                "retrieved_full_pretrim": json.dumps(_pretty(blocks_all_post), ensure_ascii=False),
                "retrieved_context": json.dumps(_pretty(blocks_trimmed_post), ensure_ascii=False),

                "postfilter_product": pf_product,
                "postfilter_hits": pf_hits,
                "postfilter_total": pf_total,

                "gt_rank_postfilter_pre": pf_rank_pre,
                "gt_rank_postfilter_trimmed": pf_rank_trimmed,
                "gt_rank_pre_rerank": pre_rank_baseline,
                "gt_rank_post_rerank": post_rank_rerank,
                "entities": json.dumps(ents, ensure_ascii=False),
            })

            # concise progress line (no DB config prints here)
            print(
                f"[{i}/{total}] base={len(blocks_all_pre)} | pf={pf_mode} hits={pf_hits}/{pf_total} "
                f"| pf_pre={len(pf_candidates)} | post_pre={len(blocks_all_post)} | "
                f"trimmed={len(blocks_trimmed_post)} (final_k={FINAL_K}) | "
                f"GT ranks: base={pre_rank_baseline} pf(pre/trim)={pf_rank_pre}/{pf_rank_trimmed} rerank={post_rank_rerank} "
                f"| ents={len(ents)} | rmode={reranker_mode}"
            )

    print(f"\nWrote log to: {out_csv}")



if __name__ == "__main__":
    demo()
