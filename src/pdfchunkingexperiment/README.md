# PDF RAG System with different Chunking Strategies 

A Retrieval-Augmented Generation (RAG) system for querying PDF documents. This system directly converts PDFs into chunks using a flexible chunking strategy. It leverages hybrid search (dense vector embeddings + BM25 sparse retrieval) and entity-aware reranking to improve retrieval accuracy and deliver contextually relevant answers.


## Data Setup

**⚠️ IMPORTANT: The `data/` folder is not included in the GitHub repository.** You must create it locally and place your data files there.

Create a `data/` folder at the repository root with the following structure:

```
data/
├── eva-docs/                 # Place your PDF files here
│   ├── Aerohive/
│   ├── AP-datasheets/
│   └── ...
└── pdf_node_chunks/          # Must be created manually
    └── <chunking_strategy>/  # Created automatically when chunking PDFs `(pdf_documents_chunking.py)`
        ├── Aerohive/
        └── ...

```

**Note:** You can set the `CHUNKING_STRATEGY` environment variable.

## Workflow Overview

The system follows this pipeline:

1. **Preprocessing** (`pdf_documents_chunking.py`): Chunks PDF files into nodes
2. **Database Setup**: Create PostgreSQL database and schema with pgvector. You can skip it if database is already created
3. **Ingestion** (`pdf_chunks_ingestion.py`): Load chunks into PostgreSQL/pgvector
4. **HNSW Index Setup** (`pdf_create_hnsw_indexes.py`): Create indexes for fast vector search (required before retrieval)
5. **Retrieval** (`pdf_chunks_retrieval.py`): Query and retrieve relevant chunks
6. **Answer Generation** (`pdf_answers_generation.py`): Generate answers using retrieved contexts
7. **Answer Evaluation** (`pdf_answers_evaluation.py`): Evaluate answer quality using LLM judge (optional)

## Detailed Workflow

### 1. Preprocessing: PDF Files Chunking using chunking strategy

**Script**: `src/pdfchunkingexperiment/pdf_documents_chunking.py`

- **Input**: PDF files in `data/eva-docs/` (or path specified by `PDF_DATA_DIR`)
- **Output**: .pkl files in `data/pdf_node_chunks/{CHUNKING_STRATEGY}/` (one .pkl file per PDF file)
- **Process**:
  - Based on the `CHUNKING_STRATEGY`, it creates chunks of PDF files and creates node files with .pkl extension 


**Environment Variables**:
- `PDF_DATA_DIR`: Input directory (Set to `.data/eva-docs`)
- `PDF_PARSED_OUTPUT_DIR`: Output directory (Set to `.data/pdf_node_chunks/{CHUNKING_STRATEGY}/`)
- `CHUNK_SIZE`: Max chunk size (default: 800)
- `CHUNK_OVERLAP`: Overlap between splits (default: 150)
- `CHUNKING_STRATEGY`: chunking strategy (default: recursive)

### 2. Database Setup

**Script**: `src/setup/create_db.py`

This script sets up your PostgreSQL database for the RAG system:
- Creates PostgreSQL database (if it doesn't exist)
- Creates schema (if it doesn't exist)
- Enables pgvector extension (required for vector search)
- Validates text search configuration

**When to Run This Script**:
- ✅ **First time setup** - Before running ingestion for the first time
- ✅ **New database** - After creating a new PostgreSQL database
- ✅ **Missing components** - If you get errors about missing schema or pgvector extension
- ❌ **Not needed** - If database, schema, and pgvector extension already exist

**How to Check if Setup is Needed**:
```bash
# Check status without making changes
uv run python src/setup/create_db.py --check
```

This will show you what exists (✓) and what's missing (✗). If everything shows ✓, you're ready to proceed. If anything shows ✗, run the setup:

```bash
# Run setup (creates missing components)
uv run python src/setup/create_db.py
```

**Note**: The script is **idempotent** (safe to run multiple times) - it only creates what doesn't exist, so you won't break anything by running it.

**Required Environment Variables**:
- `PG_HOST`: PostgreSQL host (default: localhost)
- `PG_PORT`: PostgreSQL port (default: 5432)
- `PG_DB`: Database name
- `PG_SCHEMA`: Schema name
- `PG_USER`: Database user
- `PG_PASSWORD`: Database password
- `TEXT_SEARCH_CONFIG`: PostgreSQL text search config (default: english)


### 3. Ingestion: Load Chunks into Database

**Script**: `src/pdfchunkingexperiment/pdf_chunks_ingestion.py`

- **Input**: .pkl chunk files from preprocessing step
- **Output**: Chunks stored in PostgreSQL/pgvector table
- **Process**:
  - Reads all .pkl files from `PDF_PARSED_OUTPUT_DIR`
  - Converts chunks to LlamaIndex `TextNode` objects
  - Embeds text using OpenAI embeddings (configured via `EMBED_MODEL`)
  - Stores in PostgreSQL with hybrid search enabled (vector + BM25)

**Environment Variables**:
- `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_SCHEMA`, `PG_USER`, `PG_PASSWORD`: Database connection
- `PG_TABLE_PDF`: Table name (default: `pdf_chunks`)
- `TEXT_SEARCH_CONFIG`: PostgreSQL text search config (default: `english`)
- `OVERWRITE_TABLE`: Drop and recreate table if exists (default: false)
- `EMBED_MODEL`: Embedding model (default: `text-embedding-3-small`)
- `EMBED_DIM`: Embedding dimensions (default: 1536)
- `PDF_PARSED_OUTPUT_DIR`: Directory with Nodes .pkl files (default: `./data/pdf_node_chunks`)

### 4. HNSW Index Setup (Required Before Retrieval)

**Script**: `src/pdfchunkingexperiment/pdf_create_hnsw_indexes.py`

**⚠️ IMPORTANT: You must create or check for HNSW indexes before using the retrieval system.** HNSW (Hierarchical Navigable Small World) indexes significantly speed up vector similarity searches. Without indexes, retrieval will be very slow, especially on large datasets.

This script creates HNSW indexes on vector columns for fast retrieval.

**When to Run This Script**:
- ✅ **After ingestion** - Run this after you've loaded chunks into the database (indexes require data to be present)
- ✅ **Before retrieval** - **REQUIRED**: You should create indexes before running retrieval queries, otherwise retrieval will be extremely slow
- ✅ **First-time setup** - Essential step in the setup process
- ✅ **Performance optimization** - Significantly improves vector similarity query performance

**How to Check if Indexes Exist**:
```bash
# Check status without making changes
uv run python src/setup/create_hnsw_indexes.py --check
```

This will show you which indexes exist (✓) and which are missing (✗). It will also verify that indexes are being used by the query planner.

If indexes are missing, create them:
```bash
# Create HNSW indexes (idempotent - only creates missing ones)
uv run python src/setup/create_hnsw_indexes.py
```

**Note**: 
- The script is **idempotent** (safe to run multiple times) - it only creates indexes that don't already exist
- Creating indexes can take several minutes on large tables, but it's a one-time operation
- Indexes significantly improve retrieval speed for vector similarity queries

**Required Environment Variables**:
- `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_SCHEMA`, `PG_USER`, `PG_PASSWORD`: Database connection (same as Database Setup)
- `PG_TABLE_PDF`: Table name (default: `pdf_chunks`)
- `EMBED_DIM`: Embedding dimensions (default: 1536)

**Optional Environment Variables**:
- `HNSW_M`: Number of connections per layer (default: 16, range: 4-64)
- `HNSW_EF_CONSTRUCTION`: Size of candidate list during construction (default: 64, range: 4-1000)


### 5. Retrieval: Query and Retrieve Context

**Script**: `src/pdfchunkingexperiment/pdf_chunks_retrieval.py`

**⚠️ PREREQUISITE: Make sure you've created HNSW indexes before running retrieval!** Without indexes, vector similarity queries will be extremely slow. Run `python src/pdfchunkingexperiment/pdf_create_hnsw_indexes.py --check` to verify indexes exist, or `python src/pdfchunkingexperiment/pdf_create_hnsw_indexes.py` to create them.

- **Input**: CSV file with queries (or can be used programmatically)
- **Output**: CSV file with retrieved contexts and rankings
- **Process**:
  - Performs hybrid retrieval (vector + BM25) using QueryFusionRetriever
  - Applies optional product postfiltering (soft/hard/none)
  - Applies entity-based reranking
  - Returns top FINAL_K chunks per query
  - Fuzzy matches retrieved chunks to JSON section_node_ids
  - Comparing against ground truth section IDs

**Retrieval Pipeline**:
1. **Hybrid Search**: Combines dense (vector) and sparse (BM25) retrieval
2. **Postfiltering** (optional): Filters/ranks by product mentions
3. **Reranking** (optional): Entity-aware reranking
4. **Trimming**: Returns top `FINAL_K` sections
5. **Fuzzy Matching**: Matches retrieved chunks to JSON section_node_ids
6. **Comparing**: Compares against ground truth section ids. Sets success to 1 if any of the retrieved chunks match with ground truth

**Environment Variables**:
- Database: `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_SCHEMA`, `PG_USER`, `PG_PASSWORD`, `PG_TABLE_PDF`
- Embedding: `EMBED_MODEL`, `EMBED_DIM`
- Retrieval: `RETRIEVE_TOP_K` (default: 30), `SPARSE_TOP_K` (default: 30), `FINAL_K` (default: 5)
- Postfiltering: `POSTFILTER_MODE` (none/soft/hard), `POSTFILTER_FIELDS`
- Reranking: `RERANKER_MODE` (none/entity/rrf/mmr/custom), plus various `RERANK_*` knobs
- Paths: `DATASET_QUERIES`, `PDF_RETRIEVAL_LOG_CSV`


### 6. Answer Generation

**Script**: `src/pdfchunkingexperiment/pdf_answers_generation.py`

- **Input**: Retrieval log CSV (from `pdf_chunks_retrieval.py`)
- **Output**: Answer log CSV with generated answers
- **Process**:
  - Reads retrieval log with queries and retrieved contexts
  - Generates answers using LLM (OpenAI or Gemini models)
  - Uses retrieved contexts to answer each query
  - Supports caching to avoid regenerating existing answers
  - Includes retry logic with exponential backoff for API errors

**Features**:
- Supports OpenAI (gpt-4o, gpt-4o-mini, etc.), Gemini (gemini-2.5-flash, etc.), and Groq (llama-3.3-70b-versatile, etc.) models
- Automatic retry on rate limits and temporary API errors
- Caches generated answers to resume interrupted runs
- Uses top-k retrieved contexts (`retrieved_context` column)

**Environment Variables (REQUIRED)**:
- `PDF_RETRIEVAL_LOG_CSV`: Path to retrieval log CSV (from `pdf_chunks_retrieval.py`)
- Model selection (one of the following):
  - If `USE_GROQ=true`: `GROQ_MODEL` (e.g., `llama-3.3-70b-versatile`) and `GROQ_API_KEY`
  - If `USE_GROQ=false` or not set: `ANSWER_MODEL` (e.g., `gpt-4o`, `gemini-2.5-flash`)
- API Key (based on model selection):
  - `GROQ_API_KEY`: Required when `USE_GROQ=true`
  - `OPENAI_API_KEY`: Required for OpenAI models (when `USE_GROQ=false`)
  - `GEMINI_API_KEY`: Required for Gemini models (when `USE_GROQ=false`)

**Environment Variables (OPTIONAL)**:
- `USE_GROQ`: Set to `true` to use Groq API instead of OpenAI/Gemini (default: `false`)

**Output Format**:
CSV file with columns: `question`, `generated_answer`, `context_used`

**Note**: Answer log file automatically gets generated after the script. (e.g, `logs/answers_{answer_model}_query_dataset_with_qa_{chunking_strategy}.csv`)

### 7. Answer Evaluation

**Script**: `src/pdfchunkingexperiment/pdf_answers_evaluation.py`

- **Input**: Retrieval log CSV (for ground truth answers) and answer log CSV (generated answers)
- **Output**: Evaluation CSV with judge scores and reasoning
- **Process**:
  - Loads generated answers from answer log
  - Compares each generated answer against ground truth using LLM judge
  - Outputs scores (0/1) with reasoning for each answer
  - Provides summary statistics

**Features**:
- Uses LLM judge (default: gpt-4o) to evaluate answer accuracy
- Lenient evaluation: marks as accurate unless there's an obvious error
- Handles missing answers gracefully
- Supports o1 models (with special handling for system messages)
- Detailed error reporting

**Environment Variables (REQUIRED)**:
- `PDF_RETRIEVAL_LOG_CSV`: Path to retrieval log CSV (contains ground truth answers)
- `OPENAI_API_KEY`: Required for judge API calls

**Environment Variables (OPTIONAL)**:
- `JUDGE_MODEL`: Model to use for judging (default: `gpt-4o`)

**Output Format**:
CSV file with all columns from retrieval log plus: `generated_answer`, `judge_score`, `judge_reasoning`

**Note**: Answer Evaluation log file automatically gets generated after the script. (e.g, `logs/pdf_answers_eval_log_{answer_model}_{chunking_strategy}.csv`)

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   uv run python -m spacy download en_core_web_md
   ```

2. **Set up environment**:
   Copy `.env.example` to `.env` and fill in your database and API credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

3. **Set up database**:
   First, check if setup is needed:
   ```bash
   uv run python src/setup/create_db.py --check
   ```
   If anything is missing, run the setup:
   ```bash
   uv run python src/setup/create_db.py
   ```

4. **Prepare your data**:
   Create a `data/` folder at the repository root and place your pdf files in `data/eva-docs/` (see Data Setup section above).

5. **Run PDF Chunking Experiment**:
   Check for `run_pdf_chunking_experiment.py` file inside `pdfchunkingexperiment` folder. Run this file using different commands mentioned in it. It gives option to create database using `create-db` flag. By default it is set to False. Set to True if DB is not created. If any step is failed, it saves the state in a separate state file `pdf_chunking_state.json` and rerun only the skipped files. Check the file's docstring for better understanding.  

## Configuration

Most configuration is done via environment variables. See each script's docstring for details.

Key configuration areas:
- **Database**: Connection settings, table names, schema
- **Embeddings**: Model, dimensions
- **Chunking**: chunk size, chunk overlap, chunking strategies
- **Retrieval**: Top-K values, reranking mode, postfiltering mode
- **Reranking**: Entity weights, fuzzy matching thresholds
