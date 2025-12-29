#!/usr/bin/env python3
"""
Check and optionally create HNSW index for pgvector.

HNSW (Hierarchical Navigable Small World) is a fast approximate nearest neighbor
index that significantly speeds up vector similarity searches.

This script:
1. Checks if HNSW index exists on the vector column
2. Shows current index configuration
3. Optionally creates HNSW index if missing

Usage:
    python src/setup/check_hnsw_index.py          # Check status
    python src/setup/check_hnsw_index.py --create # Create if missing
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
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

PG_TABLE = os.environ.get("PG_TABLE", "idx_section_based_chunking")

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


def find_data_table(conn) -> tuple[str, str] | None:
    """Find the actual data table name (could be base_table or data_base_table)."""
    with conn.cursor() as cur:
        # Check for base table
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (PG_SCHEMA, PG_TABLE))
        if cur.fetchone():
            return PG_SCHEMA, PG_TABLE
        
        # Check for data_ prefix table
        data_table = f"data_{PG_TABLE}"
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (PG_SCHEMA, data_table))
        if cur.fetchone():
            return PG_SCHEMA, data_table
    
    return None


def find_vector_column(conn, schema: str, table: str) -> str | None:
    """Find the vector column name in the table."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            AND data_type = 'USER-DEFINED'
        """, (schema, table))
        
        # Check if it's a vector type
        for row in cur.fetchall():
            col_name = row[0]
            # Check if column is actually vector type
            cur.execute("""
                SELECT typname
                FROM pg_type t
                JOIN pg_attribute a ON a.atttypid = t.oid
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s
            """, (schema, table, col_name))
            type_row = cur.fetchone()
            if type_row and type_row[0] == 'vector':
                return col_name
    
    return None


def check_hnsw_index(conn, schema: str, table: str, vector_col: str) -> dict:
    """Check if HNSW index exists on the vector column."""
    with conn.cursor() as cur:
        # Find indexes on this table
        cur.execute("""
            SELECT
                i.indexname,
                i.indexdef,
                am.amname as access_method
            FROM pg_indexes i
            JOIN pg_class c ON c.relname = i.tablename
            JOIN pg_namespace n ON n.nspname = i.schemaname
            JOIN pg_index idx ON idx.indexrelid = (
                SELECT oid FROM pg_class WHERE relname = i.indexname
            )
            JOIN pg_am am ON am.oid = (
                SELECT relam FROM pg_class WHERE relname = i.indexname
            )
            WHERE i.schemaname = %s AND i.tablename = %s
        """, (schema, table))
        
        indexes = []
        hnsw_found = False
        hnsw_index_name = None
        
        for row in cur.fetchall():
            index_name, index_def, access_method = row
            indexes.append({
                "name": index_name,
                "def": index_def,
                "method": access_method
            })
            
            # Check if this is an HNSW index on the vector column
            if access_method == "hnsw" and vector_col in index_def:
                hnsw_found = True
                hnsw_index_name = index_name
        
        return {
            "exists": hnsw_found,
            "index_name": hnsw_index_name,
            "all_indexes": indexes
        }


def create_hnsw_index(conn, schema: str, table: str, vector_col: str, m: int = 16, ef_construction: int = 64):
    """Create HNSW index on the vector column."""
    index_name = f"{table}_{vector_col}_hnsw_idx"
    fqtn = f'"{schema}"."{table}"'
    
    with conn.cursor() as cur:
        # Check if index already exists
        cur.execute("""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = %s AND indexname = %s
        """, (schema, index_name))
        if cur.fetchone():
            print(f"Index '{index_name}' already exists")
            return False
        
        print(f"Creating HNSW index '{index_name}' on {fqtn}.{vector_col}...")
        print(f"  Parameters: m={m}, ef_construction={ef_construction}")
        print("  This may take a while for large tables...")
        
        sql = f"""
            CREATE INDEX {index_name}
            ON {fqtn}
            USING hnsw ({vector_col} vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction})
        """
        
        try:
            cur.execute(sql)
            conn.commit()
            print(f"✓ HNSW index '{index_name}' created successfully")
            return True
        except psycopg.Error as e:
            conn.rollback()
            print(f"✗ Failed to create HNSW index: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Check and optionally create HNSW index for pgvector"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create HNSW index if it doesn't exist"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=16,
        help="HNSW parameter m (default: 16, range: 4-64)"
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=64,
        help="HNSW parameter ef_construction (default: 64, range: 4-1000)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("HNSW Index Check for pgvector")
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
            # Find the data table
            table_info = find_data_table(conn)
            if not table_info:
                print(f"✗ ERROR: Table '{PG_TABLE}' or 'data_{PG_TABLE}' not found in schema '{PG_SCHEMA}'")
                print("  Make sure you've run ingestion first.")
                sys.exit(1)
            
            schema, table = table_info
            print(f"Found table: {schema}.{table}")
            
            # Find vector column
            vector_col = find_vector_column(conn, schema, table)
            if not vector_col:
                print(f"✗ ERROR: No vector column found in {schema}.{table}")
                print("  Make sure pgvector extension is enabled and table has vector data.")
                sys.exit(1)
            
            print(f"Found vector column: {vector_col}")
            print()
            
            # Check HNSW index
            index_info = check_hnsw_index(conn, schema, table, vector_col)
            
            print("Index Status:")
            if index_info["exists"]:
                print(f"  ✓ HNSW index found: {index_info['index_name']}")
            else:
                print("  ✗ HNSW index NOT found")
            
            print()
            print("All indexes on this table:")
            if index_info["all_indexes"]:
                for idx in index_info["all_indexes"]:
                    method = idx["method"]
                    status = "✓" if method == "hnsw" else " "
                    print(f"  {status} {idx['name']} ({method})")
            else:
                print("  (no indexes found)")
            
            print()
            
            # Create index if requested
            if args.create:
                if index_info["exists"]:
                    print("HNSW index already exists. Nothing to do.")
                else:
                    print("Creating HNSW index...")
                    success = create_hnsw_index(
                        conn, schema, table, vector_col,
                        m=args.m, ef_construction=args.ef_construction
                    )
                    if success:
                        print()
                        print("=" * 70)
                        print("✓ HNSW index created! Vector searches should be much faster now.")
                        print("=" * 70)
            else:
                if not index_info["exists"]:
                    print("=" * 70)
                    print("⚠ HNSW index is missing. This will slow down vector searches.")
                    print()
                    print("To create HNSW index, run:")
                    print(f"  python src/setup/check_hnsw_index.py --create")
                    print()
                    print("HNSW parameters (optional):")
                    print(f"  --m {args.m}              # Number of connections (4-64, default: 16)")
                    print(f"  --ef-construction {args.ef_construction}  # Build quality (4-1000, default: 64)")
                    print("=" * 70)
                else:
                    print("=" * 70)
                    print("✓ HNSW index is configured. Vector searches should be fast.")
                    print("=" * 70)
    
    except psycopg.Error as e:
        print(f"✗ Database error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

