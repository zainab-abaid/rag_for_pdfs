#!/usr/bin/env python3
"""
Create HNSW indexes on vector columns for fast similarity search. This script is Optional.

HNSW (Hierarchical Navigable Small World) indexes are much faster than
sequential scans for vector similarity queries. This script creates
HNSW indexes on the vector embedding column.

The script is idempotent (safe to run multiple times) - it only creates
indexes that don't already exist.

Use --check flag to see current status without making changes:
  python src/setup/create_hnsw_indexes.py --check

Environment Variables (required):
  PG_HOST: PostgreSQL host (default: localhost)
  PG_PORT: PostgreSQL port (default: 5432)
  PG_DB: Database name
  PG_SCHEMA: Schema name
  PG_TABLE_PDF: Table name (default: pdf_chunks)
  PG_USER: Database user
  PG_PASSWORD: Database password
  HNSW_M: Number of connections per layer (default: 16)
  HNSW_EF_CONSTRUCTION: Size of candidate list during construction (default: 64)
  EMBED_DIM: Embedding dimensions (default: 1536)
"""

from __future__ import annotations
import os
import sys
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import psycopg
except ImportError:
    print("ERROR: psycopg is required. Install with: pip install psycopg[binary]")
    sys.exit(1)

# --- Environment variables ---
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB = os.environ.get("PG_DB")
PG_SCHEMA = os.environ.get("PG_SCHEMA")
PG_TABLE_PDF = os.environ.get("PG_TABLE_PDF", "pdf_chunks")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

# HNSW index parameters (can be tuned)
HNSW_M = int(os.environ.get("HNSW_M", "16"))  # Number of connections per layer (default: 16)
HNSW_EF_CONSTRUCTION = int(os.environ.get("HNSW_EF_CONSTRUCTION", "64"))  # Size of candidate list during construction
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))  # Embedding dimensions

# Validate required variables
if not PG_DB:
    print("ERROR: PG_DB environment variable is required")
    sys.exit(1)
if not PG_SCHEMA:
    print("ERROR: PG_SCHEMA environment variable is required")
    sys.exit(1)
if not PG_USER:
    print("ERROR: PG_USER environment variable is required")
    sys.exit(1)


def check_table_exists(conn, schema: str, table: str) -> bool:
    """Check if table exists."""
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


def check_data_table(conn, schema: str, base_table: str) -> str | None:
    """Find the actual data table."""
    if check_table_exists(conn, schema, base_table):
        return base_table
    data_table = f"data_{base_table}"
    if check_table_exists(conn, schema, data_table):
        return data_table
    return None


def get_vector_columns(conn, schema: str, table: str) -> list[str]:
    """Get all vector columns in the table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns c
            JOIN pg_type t ON c.udt_name = t.typname
            WHERE c.table_schema = %s 
              AND c.table_name = %s
              AND t.typname = 'vector'
            """,
            (schema, table),
        )
        return [row[0] for row in cur.fetchall()]


def index_exists(conn, schema: str, table: str, index_name: str) -> bool:
    """Check if an index already exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = %s 
              AND tablename = %s
              AND indexname = %s
            """,
            (schema, table, index_name),
        )
        return cur.fetchone() is not None


def check_hnsw_indexes(conn, schema: str, table: str, vector_cols: list[str]) -> dict[str, bool]:
    """
    Check which HNSW indexes exist for the vector columns.
    Returns dict mapping column_name -> exists (bool)
    """
    results = {}
    for col in vector_cols:
        index_name = f"{table}_{col}_hnsw_idx"
        results[col] = index_exists(conn, schema, table, index_name)
    return results


def verify_index_usage(conn, schema: str, table: str, vector_col: str) -> bool:
    """
    Verify that HNSW index is being used by running EXPLAIN ANALYZE.
    Returns True if index is being used, False otherwise.
    """
    fqtn = f'"{schema}"."{table}"'
    
    # Create a dummy embedding vector for testing
    dummy_vector = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"
    
    query = f"""
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
        SELECT id, {vector_col} <=> %s::vector AS distance
        FROM {fqtn}
        ORDER BY {vector_col} <=> %s::vector
        LIMIT 10
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, (dummy_vector, dummy_vector))
            plan = cur.fetchall()
            
            # Check if HNSW index is mentioned in the plan
            plan_text = "\n".join([row[0] for row in plan])
            if "Index Scan" in plan_text or "Bitmap Index Scan" in plan_text:
                return True
            return False
    except psycopg.Error:
        return False


def create_hnsw_index(conn, schema: str, table: str, column: str, distance_op: str = "vector_cosine_ops") -> bool:
    """
    Create HNSW index on a vector column.
    Returns True if index was created, False if it already existed.
    
    distance_op options:
    - vector_cosine_ops: for cosine similarity (most common)
    - vector_l2_ops: for L2 distance
    - vector_ip_ops: for inner product
    """
    index_name = f"{table}_{column}_hnsw_idx"
    
    if index_exists(conn, schema, table, index_name):
        print(f"  ✓ Index '{index_name}' already exists, skipping...")
        return False
    
    print(f"  Creating HNSW index '{index_name}' on {column}...")
    print(f"    This may take a while for large tables...")
    
    with conn.cursor() as cur:
        # Create index with HNSW
        sql = f'''
        CREATE INDEX "{index_name}"
        ON "{schema}"."{table}"
        USING hnsw ({column} {distance_op})
        WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})
        '''
        
        try:
            cur.execute(sql)
            conn.commit()
            print(f"  ✓ Index '{index_name}' created successfully")
            return True
        except psycopg.Error as e:
            conn.rollback()
            print(f"  ✗ Failed to create index: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Create HNSW indexes on vector columns for fast similarity search"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check status only (don't create anything)"
    )
    args = parser.parse_args()

    print("=" * 70)
    if args.check:
        print("HNSW Index Status Check")
    else:
        print("Create HNSW Indexes for Vector Search")
    print("=" * 70)
    print(f"Host: {PG_HOST}:{PG_PORT}")
    print(f"Database: {PG_DB}")
    print(f"Schema: {PG_SCHEMA}")
    print(f"Table: {PG_TABLE_PDF}")
    if not args.check:
        print(f"HNSW Parameters: m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}")
    print("=" * 70)
    print()

    try:
        with psycopg.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        ) as conn:
            # Find the actual table
            actual_table = check_data_table(conn, PG_SCHEMA, PG_TABLE_PDF)
            if not actual_table:
                print(f"✗ Table '{PG_SCHEMA}.{PG_TABLE_PDF}' or '{PG_SCHEMA}.data_{PG_TABLE_PDF}' not found!")
                print("\nMake sure you've run ingestion first:")
                print("  python src/ingestion/ingest_section_chunks.py")
                sys.exit(1)
            
            print(f"✓ Found table: {PG_SCHEMA}.{actual_table}")
            print()

            # Get vector columns
            vector_cols = get_vector_columns(conn, PG_SCHEMA, actual_table)
            if not vector_cols:
                print("✗ No vector columns found in table!")
                sys.exit(1)
            
            print(f"✓ Found {len(vector_cols)} vector column(s): {', '.join(vector_cols)}")
            print()

            if args.check:
                # Check-only mode
                index_status = check_hnsw_indexes(conn, PG_SCHEMA, actual_table, vector_cols)
                
                print("HNSW Index Status:")
                all_exist = True
                for col in vector_cols:
                    exists = index_status[col]
                    status = "✓ EXISTS" if exists else "✗ MISSING"
                    print(f"  {col}: {status}")
                    if not exists:
                        all_exist = False
                print()

                # Verify index usage if indexes exist
                if all_exist:
                    print("Verifying index usage...")
                    index_used = False
                    for col in vector_cols:
                        if verify_index_usage(conn, PG_SCHEMA, actual_table, col):
                            index_used = True
                            break
                    
                    if index_used:
                        print("  ✓ HNSW index is being used for vector queries")
                    else:
                        print("  ⚠ HNSW index exists but may not be used (this can happen with small tables)")
                    print()

                print("=" * 70)
                if all_exist:
                    print("✓ All HNSW indexes exist! Vector search should be fast.")
                else:
                    print("⚠ Some HNSW indexes are missing. This will slow down vector searches.")
                    print()
                    print("To create missing indexes, run:")
                    print("  python src/setup/create_hnsw_indexes.py")
                print("=" * 70)
            else:
                # Create mode
                print("NOTE: Creating indexes can take several minutes on large tables.")
                print("      This is a one-time operation that significantly speeds up retrieval.")
                print()

                # Create indexes
                created = 0
                for col in vector_cols:
                    if create_hnsw_index(conn, PG_SCHEMA, actual_table, col):
                        created += 1
                    print()

                print("=" * 70)
                if created > 0:
                    print(f"✓ Successfully created {created} HNSW index(es)")
                    print("  Vector similarity search should now be much faster!")
                else:
                    print("✓ All HNSW indexes already exist (nothing to create)")
                    print("  Vector search should be fast!")
                print("=" * 70)
                print()
                print("To verify indexes, run:")
                print("  python src/setup/create_hnsw_indexes.py --check")

    except psycopg.Error as e:
        print(f"ERROR: Database operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
