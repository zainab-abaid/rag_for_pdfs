#!/usr/bin/env python3
"""
Database setup script for PostgreSQL with pgvector extension.

This script:
1. Connects to PostgreSQL (using PG_HOST, PG_PORT, PG_USER, PG_PASSWORD)
2. Creates database if not exists (using PG_DB)
3. Creates schema if not exists (using PG_SCHEMA)
4. Enables pgvector extension
5. Validates text search configuration (using TEXT_SEARCH_CONFIG)

Note: The table itself is created automatically by LlamaIndex's PGVectorStore
during ingestion (see ingest_section_chunks.py).

WHEN TO RUN THIS SCRIPT:
-----------------------
Run this script:
  - First time setup (before running ingestion)
  - After creating a new database
  - If you get errors about missing schema or pgvector extension

You DON'T need to run it if:
  - Database, schema, and pgvector extension already exist
  - You're just re-ingesting data into an existing setup

The script is idempotent (safe to run multiple times) - it only creates
what doesn't exist.

Use --check flag to see current status without making changes:
  python src/setup/create_db.py --check

Environment Variables (required):
  PG_HOST: PostgreSQL host (default: localhost)
  PG_PORT: PostgreSQL port (default: 5432)
  PG_DB: Database name
  PG_SCHEMA: Schema name
  PG_USER: Database user
  PG_PASSWORD: Database password
  TEXT_SEARCH_CONFIG: PostgreSQL text search config (default: english)
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Tuple

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

TEXT_SEARCH_CONFIG = os.environ.get("TEXT_SEARCH_CONFIG", "english").lower()

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


def check_database_status() -> Tuple[bool, bool, bool]:
    """
    Check status of database, schema, and pgvector extension.
    Returns (db_exists, schema_exists, pgvector_enabled)
    """
    db_exists = False
    schema_exists = False
    pgvector_enabled = False

    # Check if database exists
    try:
        with psycopg.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname="postgres",
            user=PG_USER,
            password=PG_PASSWORD
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (PG_DB,)
                )
                db_exists = cur.fetchone() is not None
    except psycopg.Error:
        return False, False, False

    if not db_exists:
        return False, False, False

    # Check schema and extension in target database
    try:
        with psycopg.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        ) as conn:
            with conn.cursor() as cur:
                # Check schema
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.schemata 
                    WHERE schema_name = %s
                    """,
                    (PG_SCHEMA,)
                )
                schema_exists = cur.fetchone() is not None

                # Check pgvector extension
                cur.execute(
                    """
                    SELECT 1 FROM pg_extension 
                    WHERE extname = 'vector'
                    """
                )
                pgvector_enabled = cur.fetchone() is not None
    except psycopg.Error:
        return db_exists, False, False

    return db_exists, schema_exists, pgvector_enabled


def create_database(check_only: bool = False):
    """Create database if it doesn't exist."""
    # Connect to default 'postgres' database to create target database
    try:
        with psycopg.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname="postgres",  # Connect to default database
            user=PG_USER,
            password=PG_PASSWORD
        ) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (PG_DB,)
                )
                exists = cur.fetchone() is not None

                if not exists:
                    if check_only:
                        print(f"✗ Database '{PG_DB}' does NOT exist")
                        return False
                    print(f"Creating database '{PG_DB}'...")
                    cur.execute(f'CREATE DATABASE "{PG_DB}"')
                    print(f"✓ Database '{PG_DB}' created")
                else:
                    print(f"✓ Database '{PG_DB}' already exists")
                return True
    except psycopg.Error as e:
        if check_only:
            print(f"✗ Cannot connect to PostgreSQL: {e}")
        else:
            print(f"ERROR: Failed to create database: {e}")
            sys.exit(1)
        return False


def setup_database(check_only: bool = False):
    """Set up schema, extensions, and configuration in the target database."""
    try:
        with psycopg.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        ) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check/create schema
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.schemata 
                    WHERE schema_name = %s
                    """,
                    (PG_SCHEMA,)
                )
                schema_exists = cur.fetchone() is not None

                if not schema_exists:
                    if check_only:
                        print(f"✗ Schema '{PG_SCHEMA}' does NOT exist")
                    else:
                        print(f"Creating schema '{PG_SCHEMA}'...")
                        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{PG_SCHEMA}"')
                        print(f"✓ Schema '{PG_SCHEMA}' created")
                else:
                    print(f"✓ Schema '{PG_SCHEMA}' already exists")

                # Check/enable pgvector extension
                cur.execute(
                    """
                    SELECT 1 FROM pg_extension 
                    WHERE extname = 'vector'
                    """
                )
                pgvector_exists = cur.fetchone() is not None

                if not pgvector_exists:
                    if check_only:
                        print("✗ pgvector extension is NOT enabled")
                    else:
                        print("Enabling pgvector extension...")
                        try:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                            print("✓ pgvector extension enabled")
                        except psycopg.Error as e:
                            print(f"WARNING: Failed to enable pgvector extension: {e}")
                            print("  Make sure pgvector is installed: https://github.com/pgvector/pgvector")
                            print("  Continuing anyway...")
                else:
                    print("✓ pgvector extension already enabled")

                # Validate text search configuration
                print(f"Validating text search config '{TEXT_SEARCH_CONFIG}'...")
                cur.execute(
                    """
                    SELECT 1 FROM pg_ts_config 
                    WHERE cfgname = %s
                    """,
                    (TEXT_SEARCH_CONFIG,)
                )
                if cur.fetchone() is None:
                    print(f"⚠ WARNING: Text search config '{TEXT_SEARCH_CONFIG}' not found")
                    print("  Available configs: simple, english, etc.")
                    print("  Continuing with default...")
                else:
                    print(f"✓ Text search config '{TEXT_SEARCH_CONFIG}' is available")

    except psycopg.Error as e:
        if check_only:
            print(f"✗ Cannot connect to database '{PG_DB}': {e}")
        else:
            print(f"ERROR: Failed to set up database: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Set up PostgreSQL database with pgvector extension for RAG system"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check status only (don't create anything)"
    )
    args = parser.parse_args()

    print("=" * 60)
    if args.check:
        print("PostgreSQL Database Status Check")
    else:
        print("PostgreSQL Database Setup for RAG System")
    print("=" * 60)
    print(f"Host: {PG_HOST}:{PG_PORT}")
    print(f"Database: {PG_DB}")
    print(f"Schema: {PG_SCHEMA}")
    print(f"User: {PG_USER}")
    print(f"Text Search Config: {TEXT_SEARCH_CONFIG}")
    print("=" * 60)
    print()

    if args.check:
        # Check-only mode
        db_exists, schema_exists, pgvector_enabled = check_database_status()
        
        print("Status:")
        print(f"  Database '{PG_DB}': {'✓ EXISTS' if db_exists else '✗ MISSING'}")
        if db_exists:
            print(f"  Schema '{PG_SCHEMA}': {'✓ EXISTS' if schema_exists else '✗ MISSING'}")
            print(f"  pgvector extension: {'✓ ENABLED' if pgvector_enabled else '✗ NOT ENABLED'}")
        
        print()
        if db_exists and schema_exists and pgvector_enabled:
            print("=" * 60)
            print("✓ Everything is set up! You can proceed with ingestion.")
            print("=" * 60)
        else:
            print("=" * 60)
            print("⚠ Setup incomplete. Run without --check to create missing components:")
            print("  python src/setup/create_db.py")
            print("=" * 60)
    else:
        # Setup mode
        db_ok = create_database(check_only=False)
        if db_ok:
            setup_database(check_only=False)

        print()
        print("=" * 60)
        print("✓ Database setup complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run section_wise_chunking.py to chunk your markdown files")
        print("2. Run ingest_section_chunks.py to load chunks into the database")
        print()


if __name__ == "__main__":
    main()

