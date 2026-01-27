#!/usr/bin/env python3
"""
Ingest section-wise chunks into Postgres/pgvector via LlamaIndex.

- Reads JSON files emitted by section_wise_chunking.py (a JSON list per MD).
- Ingests ONLY the body content (drops the breadcrumb + heading prelude).
- Persists *all* metadata as JSON so retrieval/stitching can access it later.
- Enables hybrid (vector + BM25) search on the pgvector table.

Env (from .env):
  PG_HOST=localhost
  PG_PORT=5432
  PG_DB=extreme_pdfs
  PG_SCHEMA=extreme_pdfs_schema
  PG_USER=zainababaid
  PG_PASSWORD=zainababaid
  PG_TABLE=idx_section_based_chunking
  TEXT_SEARCH_CONFIG=english
  OVERWRITE_TABLE=true            # drop the table first if true

  EMBED_MODEL=text-embedding-3-small
  EMBED_DIM=1536

  OUTPUT_DIR=./data/sectionwise_chunks
"""

from __future__ import annotations
import os, re, json, glob
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# llamaindex
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore

# optional: set embedding model if not set globally
from llama_index.core.settings import Settings
try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except Exception:
    OpenAIEmbedding = None

# direct SQL for table drop
import psycopg

# --- env ---
PG_HOST = os.environ["PG_HOST"]
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB = os.environ["PG_DB"]
PG_SCHEMA = os.environ["PG_SCHEMA"]
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

PG_TABLE = os.environ.get("PG_TABLE", "idx_section_based_chunking")
TEXT_SEARCH_CONFIG = os.environ.get("TEXT_SEARCH_CONFIG", "english").lower()
OVERWRITE_TABLE = os.environ.get("OVERWRITE_TABLE", "false").lower() == "true"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./data/sectionwise_chunks")

if OpenAIEmbedding is not None:
    # set once for this process if not already set elsewhere
    try:
        Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, dimensions=EMBED_DIM)
    except Exception:
        pass


def iter_chunk_files(root: str) -> List[Path]:
    return [Path(p) for p in glob.glob(str(Path(root) / "**" / "*.json"), recursive=True)]


def to_nodes(chunks: List[Dict[str, Any]]) -> List[TextNode]:
    nodes: List[TextNode] = []
    for obj in chunks:
        raw_text = obj["text"]           # <-- keep full text (with Path + heading)
        meta = obj["metadata"]

        if not raw_text or not raw_text.strip():
            continue

        node_id = meta["chunk_id"]     # unique per split chunk
        ref_doc_id = meta["doc_id"]

        node = TextNode(
            id_=node_id,
            text=raw_text,               # <-- ingest full text
            metadata=meta,               # keep all stitching metadata
            ref_doc_id=ref_doc_id,
        )
        nodes.append(node)
    return nodes


def _drop_table_if_requested():
    if not OVERWRITE_TABLE:
        return
    fqtn = f'"{PG_SCHEMA}"."data_{PG_TABLE}"'
    print(f"[ingest] OVERWRITE_TABLE=true, dropping {fqtn} if it exists...")
    try:
        with psycopg.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
        ) as conn, conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS {fqtn} CASCADE;')
            conn.commit()
    except Exception as e:
        print(f"[warn] DROP TABLE failed (continuing): {e}")


def main():
    files = iter_chunk_files(OUTPUT_DIR)
    if not files:
        print(f"[ingest] No chunk files found under {OUTPUT_DIR}")
        return

    print(f"[ingest] Found {len(files)} chunk files. Building nodes...")

    all_nodes: List[TextNode] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                print(f"[warn] {f} is not a JSON list, skipping")
                continue
            nodes = to_nodes(data)
            all_nodes.extend(nodes)
        except Exception as e:
            print(f"[error] Failed reading {f}: {e}")

    if not all_nodes:
        print("[ingest] No nodes to ingest.")
        return

    # drop table first if requested
    _drop_table_if_requested()

    print(f"[ingest] Prepared {len(all_nodes)} nodes. Connecting to Postgres...")

    vector_store = PGVectorStore.from_params(
        database=PG_DB,
        host=PG_HOST,
        password=PG_PASSWORD,
        port=PG_PORT,
        user=PG_USER,
        table_name=PG_TABLE,
        schema_name=PG_SCHEMA,
        embed_dim=EMBED_DIM,
        hybrid_search=True,                 # hybrid ON
        text_search_config=TEXT_SEARCH_CONFIG,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Use a VectorStoreIndex to handle embedding & insert
    index = VectorStoreIndex([], storage_context=storage_context)
    print("[ingest] Inserting nodes (this will embed with your configured model via env)...")
    index.insert_nodes(all_nodes)

    print(f"[ingest] Done. Inserted {len(all_nodes)} nodes into {PG_SCHEMA}.{PG_TABLE}")


if __name__ == "__main__":
    main()
