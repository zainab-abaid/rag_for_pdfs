"""
Ingest annotated PDF nodes (with section_ids) into PostgreSQL using pgvector embeddings via LlamaIndex.

Workflow:

    PDF Files
       │
       ▼
    PDF Parsing → Nodes (.pkl files with text + metadata including section_ids)
       │
       ▼
    Ingest Script (this file)
       │
       ├─ Load nodes from `.pkl` files (supports both dict and object nodes)
       ├─ Generate embeddings for node text via OpenAI async API
       ├─ Batch nodes and retry failed API requests for reliability
       └─ Insert text, embeddings, and metadata (including section_ids) into PostgreSQL
       │
       ▼
    PostgreSQL + pgvector table (vector index for fast semantic search)
       │
       ▼
    Retrieval / RAG / Semantic Search (section-aware)

Features:

- Supports `.pkl` node files created by PDF parsing scripts, where each node contains:
    - `text`: chunk of PDF content
    - `metadata`: dict/object containing file info, chunk index, section_ids, etc.
- Node ingestion supports **dicts or objects** to handle flexible node formats.
- Generates embeddings asynchronously using OpenAI embeddings API (`text-embedding-3-small` by default).
- Persists text, embeddings, metadata, and section_ids into a PostgreSQL table.
- Automatically drops and recreates the table if `OVERWRITE_TABLE=True`.
- Creates a pgvector IVFFLAT index for fast similarity search with cosine distance.
- Uses batching and concurrency for efficient embedding generation and DB insertion.
- Includes retry logic for robust handling of transient API/network failures.
- Progress bars show ingestion and embedding status for visibility.

Environment variables (from .env):

  PG_HOST=localhost                # PostgreSQL host
  PG_PORT=5432                     # PostgreSQL port
  PG_DB=extreme_pdfs               # Database name
  PG_SCHEMA=extreme_pdfs_schema    # Schema name
  PG_TABLE_PDF=pdf_chunks          # Table to store chunks + embeddings
  PG_USER=xxx                      # Database user
  PG_PASSWORD=xxx                  # Database password
  OVERWRITE_TABLE=true             # Drop table before ingestion if true

  EMBED_MODEL=text-embedding-3-small    # OpenAI embedding model
  EMBED_DIM=1536                        # Embedding vector dimension
  EMBED_BATCH_SIZE=100                  # Number of nodes per embedding API request
  MAX_CONCURRENT=5                      # Max parallel API requests

  NODES_DIR=./data/annotated_nodes     # Directory containing `.pkl` node files

Design choices:

- Uses LlamaIndex nodes to preserve metadata and chunked text structure.
- Handles both dict and object node formats for compatibility.
- Async embedding + concurrency for speed without exceeding API limits.
- Batch processing ensures memory efficiency and high throughput.
- Executemany inserts + vector indexing for high-performance retrieval.
- Retry logic improves resilience against transient API/network errors.
- Progress bars provide operational visibility during ingestion.
- Section IDs stored as JSONB in PostgreSQL for section-aware retrieval.
"""

import os
import pickle
import json
import asyncio
from pathlib import Path
from typing import List, Any, Tuple
from tqdm import tqdm

import psycopg
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ---------------- Env/Configs ----------------
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_DB = os.getenv("PG_DB")
PG_SCHEMA = os.getenv("PG_SCHEMA", "extreme_pdfs_schema")
PG_TABLE_PDF = os.getenv("PG_TABLE_PDF", "pdf_chunking")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
OVERWRITE_TABLE = os.getenv("OVERWRITE_TABLE", "false").lower() == "true"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100")) 
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
NODES_DIR = Path(os.getenv("ANNOTATED_NODES_DIR", "./data/annotated_nodes"))

# ------------------ Utilities ----------------
def load_nodes(node_file: Path) -> List[Any]:
    """Load nodes from a .pkl file"""
    with open(node_file, "rb") as f:
        return pickle.load(f)

def create_table_if_needed(conn):
    """Create table for storing PDF chunks, embeddings, and section_ids"""
    fqtn = f'"{PG_SCHEMA}"."{PG_TABLE_PDF}"'
    with conn.cursor() as cur:
        if OVERWRITE_TABLE:
            print(f"[ingest] Dropping table {fqtn} (OVERWRITE_TABLE=True)...")
            cur.execute(f"DROP TABLE IF EXISTS {fqtn} CASCADE;")
        
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {fqtn} (
                id SERIAL PRIMARY KEY,
                source_file TEXT NOT NULL,
                chunk_index INT NOT NULL,
                text TEXT NOT NULL,
                section_ids JSONB NOT NULL,
                embedding vector({EMBED_DIM}) NOT NULL
            );
        """)

        # Check if index exists
        cur.execute(f"""
            SELECT indexname FROM pg_indexes 
            WHERE schemaname = %s AND tablename = %s AND indexname = 'pdf_chunks_embedding_index';
        """, (PG_SCHEMA, PG_TABLE_PDF))
        
        if not cur.fetchone():
            print(f"[ingest] Creating vector index (this may take a while)...")
            cur.execute(f"""
                CREATE INDEX pdf_chunks_embedding_index 
                ON {fqtn} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        
        conn.commit()
    print(f"[ingest] Table ready: {fqtn}")

# ------------------ Embedding logic ----------------
async def generate_embeddings_batch(client: AsyncOpenAI, texts: List[str], max_retries: int = 3) -> List[List[float]]:
    """Generate embeddings with retry logic"""
    for attempt in range(max_retries):
        try:
            response = await client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"[ingest] API error (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"[ingest] Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

async def process_batch(client: AsyncOpenAI, batch_data: List[Tuple], semaphore: asyncio.Semaphore) -> List[Tuple]:
    """Process a single batch with concurrency limiting"""
    async with semaphore:
        texts = [text for _, _, text, _ in batch_data]
        embeddings = await generate_embeddings_batch(client, texts)
        return [
            (src, idx, txt, json.dumps(section_ids), emb)
            for (src, idx, txt, section_ids), emb in zip(batch_data, embeddings)
        ]

async def insert_nodes_async(conn_params: dict, nodes: List[Any], client: AsyncOpenAI, batch_size: int):
    """Insert nodes into PostgreSQL with async embedding generation"""
    fqtn = f'"{PG_SCHEMA}"."{PG_TABLE_PDF}"'
    
    all_data = []
    for node in nodes:
        if isinstance(node, dict):
            text = node.get("text")
            meta = node.get("metadata", {})
        else:
            text = getattr(node, "text", None)
            meta = getattr(node, "metadata", {})

        if not text:
            continue
        source_file = meta.get("source_file", "unknown.pdf")
        chunk_index = meta.get("chunk_index", 0)
        section_ids = meta.get("section_ids", [])
        all_data.append((source_file, chunk_index, text, section_ids))
    
    # Split into batches
    batches = [all_data[i:i + batch_size] for i in range(0, len(all_data), batch_size)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    print(f"[ingest] Processing {len(batches)} batches ({len(nodes)} nodes)...")
    tasks = [process_batch(client, batch, semaphore) for batch in batches]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Embedding"):
        result = await coro
        results.append(result)

    # Insert into DB
    print(f"[ingest] Inserting {len(nodes)} nodes into database...")
    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            for batch_result in tqdm(results, desc="Inserting"):
                cur.executemany(
                    f"INSERT INTO {fqtn} (source_file, chunk_index, text, section_ids, embedding) VALUES (%s, %s, %s, %s, %s)",
                    batch_result
                )
        conn.commit()
    print(f"[ingest] ✓ Inserted {len(nodes)} nodes")

# ------------------ Main ----------------
async def main_async():
    conn_params = {
        "host": PG_HOST,
        "port": PG_PORT,
        "dbname": PG_DB,
        "user": PG_USER,
        "password": PG_PASSWORD
    }
    
    with psycopg.connect(**conn_params) as conn:
        create_table_if_needed(conn)
    
    async with AsyncOpenAI() as client:
        node_files = sorted(NODES_DIR.rglob("*.pkl"))
        print(f"[ingest] Found {len(node_files)} node files in {NODES_DIR}")
        print(f"[ingest] Embedding model: {EMBED_MODEL}, Batch size: {EMBED_BATCH_SIZE}, Max concurrent: {MAX_CONCURRENT}")
        print("-" * 60)
        
        total_nodes = 0
        for idx, node_file in enumerate(node_files, start=1):
            nodes = load_nodes(node_file)
            print(f"\n[ingest] ({idx}/{len(node_files)}) Processing {node_file.relative_to(NODES_DIR)} ({len(nodes)} nodes)")
            
            await insert_nodes_async(
                conn_params=conn_params,
                nodes=nodes,
                client=client,
                batch_size=EMBED_BATCH_SIZE  # <-- use config consistently
            )
            total_nodes += len(nodes)
        
        print("\n" + "=" * 60)
        print(f"[ingest] ✓ DONE — total nodes processed: {total_nodes}")
        print("=" * 60)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
