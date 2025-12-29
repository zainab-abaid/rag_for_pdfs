#!/usr/bin/env python3
"""
Check if PostgreSQL is using HNSW index for vector queries.

This script runs a sample vector similarity query and shows the execution plan
to verify if HNSW index is being used.
"""

from __future__ import annotations
import os
import sys

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
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

PG_TABLE = os.environ.get("PG_TABLE", "idx_section_based_chunking")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))

if not all([PG_DB, PG_SCHEMA, PG_USER]):
    print("ERROR: Missing required environment variables")
    sys.exit(1)


def find_data_table(conn) -> tuple[str, str] | None:
    """Find the actual data table name."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (PG_SCHEMA, PG_TABLE))
        if cur.fetchone():
            return PG_SCHEMA, PG_TABLE
        
        data_table = f"data_{PG_TABLE}"
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (PG_SCHEMA, data_table))
        if cur.fetchone():
            return PG_SCHEMA, data_table
    
    return None


def check_query_plan(conn, schema: str, table: str):
    """Run EXPLAIN ANALYZE on a sample vector query to see if HNSW is used."""
    fqtn = f'"{schema}"."{table}"'
    
    # Create a dummy embedding vector for testing
    dummy_vector = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"
    
    print("Running EXPLAIN ANALYZE on sample vector similarity query...")
    print("=" * 70)
    
    query = f"""
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
        SELECT id, embedding <=> %s::vector AS distance
        FROM {fqtn}
        ORDER BY embedding <=> %s::vector
        LIMIT 10
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (dummy_vector, dummy_vector))
        plan = cur.fetchall()
        
        print("\nQuery Execution Plan:")
        print("-" * 70)
        for row in plan:
            print(row[0])
        
        print("\n" + "=" * 70)
        
        # Check if HNSW index is mentioned in the plan
        plan_text = "\n".join([row[0] for row in plan])
        if "hnsw" in plan_text.lower() or "Index Scan" in plan_text:
            if "Index Scan" in plan_text or "Bitmap Index Scan" in plan_text:
                print("✓ Index is being used!")
                if "hnsw" in plan_text.lower():
                    print("✓ HNSW index detected in query plan")
                else:
                    print("⚠ Using index but HNSW not explicitly mentioned (might be using it)")
            else:
                print("✗ Index is NOT being used - doing sequential scan!")
                print("  This will be slow. Check index configuration.")
        else:
            print("⚠ Could not determine index usage from plan")


def main():
    print("=" * 70)
    print("Vector Query Plan Check")
    print("=" * 70)
    print(f"Host: {PG_HOST}:{PG_PORT}")
    print(f"Database: {PG_DB}")
    print(f"Schema: {PG_SCHEMA}")
    print(f"Table: {PG_TABLE}")
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
            table_info = find_data_table(conn)
            if not table_info:
                print(f"✗ ERROR: Table not found")
                sys.exit(1)
            
            schema, table = table_info
            print(f"Found table: {schema}.{table}\n")
            
            check_query_plan(conn, schema, table)
    
    except psycopg.Error as e:
        print(f"✗ Database error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

