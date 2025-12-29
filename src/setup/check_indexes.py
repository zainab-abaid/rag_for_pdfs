#!/usr/bin/env python3
"""
Check database indexes, especially HNSW indexes for vector similarity search.

This script checks:
1. If HNSW indexes exist on the vector column
2. What indexes exist on the table
3. Index sizes and usage statistics
"""

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
PG_TABLE = os.environ.get("PG_TABLE", "idx_section_based_chunking")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

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


def check_data_table(conn, schema: str, base_table: str) -> str:
    """
    Find the actual data table (could be base_table or data_base_table).
    Returns the table name or None if not found.
    """
    if check_table_exists(conn, schema, base_table):
        return base_table
    data_table = f"data_{base_table}"
    if check_table_exists(conn, schema, data_table):
        return data_table
    return None


def get_indexes(conn, schema: str, table: str):
    """Get all indexes on the table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                i.indexname,
                i.indexdef,
                pg_size_pretty(pg_relation_size(i.indexname::regclass)) as size
            FROM pg_indexes i
            WHERE i.schemaname = %s AND i.tablename = %s
            ORDER BY i.indexname
            """,
            (schema, table),
        )
        return cur.fetchall()


def get_vector_column_indexes(conn, schema: str, table: str):
    """Get indexes specifically on vector columns (HNSW or ivfflat)."""
    with conn.cursor() as cur:
        # Check for HNSW indexes
        cur.execute(
            """
            SELECT
                i.indexname,
                i.indexdef,
                pg_size_pretty(pg_relation_size(i.indexname::regclass)) as size,
                'HNSW' as index_type
            FROM pg_indexes i
            WHERE i.schemaname = %s 
              AND i.tablename = %s
              AND i.indexdef LIKE '%vector%'
              AND i.indexdef LIKE '%hnsw%'
            ORDER BY i.indexname
            """,
            (schema, table),
        )
        hnsw_indexes = cur.fetchall()
        
        # Check for IVFFlat indexes
        cur.execute(
            """
            SELECT
                i.indexname,
                i.indexdef,
                pg_size_pretty(pg_relation_size(i.indexname::regclass)) as size,
                'IVFFlat' as index_type
            FROM pg_indexes i
            WHERE i.schemaname = %s 
              AND i.tablename = %s
              AND i.indexdef LIKE '%vector%'
              AND i.indexdef LIKE '%ivfflat%'
            ORDER BY i.indexname
            """,
            (schema, table),
        )
        ivfflat_indexes = cur.fetchall()
        
        return hnsw_indexes, ivfflat_indexes


def get_vector_columns(conn, schema: str, table: str):
    """Get all vector columns in the table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.column_name
            FROM information_schema.columns c
            JOIN pg_type t ON c.udt_name = t.typname
            WHERE c.table_schema = %s 
              AND c.table_name = %s
              AND t.typname = 'vector'
            """,
            (schema, table),
        )
        return [row[0] for row in cur.fetchall()]


def main():
    print("=" * 70)
    print("Database Index Check (HNSW for Vector Search)")
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
            # Find the actual table name
            actual_table = check_data_table(conn, PG_SCHEMA, PG_TABLE)
            if not actual_table:
                print(f"✗ Table '{PG_SCHEMA}.{PG_TABLE}' or '{PG_SCHEMA}.data_{PG_TABLE}' not found!")
                print("\nMake sure you've run ingestion first:")
                print("  python src/ingestion/ingest_section_chunks.py")
                sys.exit(1)
            
            print(f"✓ Found table: {PG_SCHEMA}.{actual_table}")
            print()

            # Check for vector columns
            vector_cols = get_vector_columns(conn, PG_SCHEMA, actual_table)
            if not vector_cols:
                print("⚠ No vector columns found in table!")
                print("  This might mean the table structure is unexpected.")
            else:
                print(f"✓ Found {len(vector_cols)} vector column(s): {', '.join(vector_cols)}")
            print()

            # Check for HNSW/IVFFlat indexes
            hnsw_indexes, ivfflat_indexes = get_vector_column_indexes(conn, PG_SCHEMA, actual_table)
            
            if hnsw_indexes:
                print("✓ HNSW INDEXES FOUND:")
                print("-" * 70)
                for idx_name, idx_def, size, idx_type in hnsw_indexes:
                    print(f"  Index: {idx_name}")
                    print(f"  Type: {idx_type}")
                    print(f"  Size: {size}")
                    print(f"  Definition: {idx_def[:100]}...")
                    print()
            else:
                print("✗ NO HNSW INDEXES FOUND!")
                print()
                print("This is likely why retrieval is slow.")
                print("HNSW indexes significantly speed up vector similarity search.")
                print()
            
            if ivfflat_indexes:
                print("⚠ IVFFlat INDEXES FOUND (slower than HNSW):")
                print("-" * 70)
                for idx_name, idx_def, size, idx_type in ivfflat_indexes:
                    print(f"  Index: {idx_name}")
                    print(f"  Type: {idx_type}")
                    print(f"  Size: {size}")
                    print(f"  Definition: {idx_def[:100]}...")
                    print()
            
            # Show all indexes
            all_indexes = get_indexes(conn, PG_SCHEMA, actual_table)
            if all_indexes:
                print("ALL INDEXES ON TABLE:")
                print("-" * 70)
                for idx_name, idx_def, size in all_indexes:
                    print(f"  {idx_name} ({size})")
                    if 'vector' in idx_def.lower():
                        print(f"    → {idx_def[:120]}...")
                print()
            
            # Summary and recommendations
            print("=" * 70)
            if hnsw_indexes:
                print("✓ HNSW indexes are present. Vector search should be fast.")
            elif ivfflat_indexes:
                print("⚠ IVFFlat indexes found (slower than HNSW).")
                print("  Consider creating HNSW indexes for better performance.")
            else:
                print("✗ NO VECTOR INDEXES FOUND!")
                print()
                print("RECOMMENDATION: Create HNSW indexes to speed up retrieval.")
                print()
                print("To create HNSW indexes, run:")
                print(f"  python src/setup/create_hnsw_indexes.py")
                print()
                print("Or manually in psql:")
                for col in vector_cols:
                    print(f'  CREATE INDEX ON "{PG_SCHEMA}"."{actual_table}" USING hnsw ({col} vector_cosine_ops);')
            print("=" * 70)

    except psycopg.Error as e:
        print(f"ERROR: Database connection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

