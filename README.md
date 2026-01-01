# PDF Markdown RAG System

A RAG (Retrieval-Augmented Generation) system for querying PDF documentation that has been converted to Markdown format. This system uses section-wise chunking, hybrid search (vector + BM25), and optional entity-based reranking for improved retrieval accuracy.

## Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- OpenAI API key (for embeddings and optional dataset generation)
- spaCy English model: `uv run python -m spacy download en_core_web_md`

## Data Setup

**⚠️ IMPORTANT: The `data/` folder is not included in the GitHub repository.** You must create it locally and place your data files there.

Create a `data/` folder at the repository root with the following structure:

```
data/
├── eva_md_files/          # Place your markdown files here (input for chunking)
│   ├── Aerohive/
│   ├── AP-datasheets/
│   └── ...
└── sectionwise_chunks/     # Will be created automatically (output from chunking)
    └── ...
```

**Data Folder Structure:**
- `data/eva_md_files/`: Place your markdown (.md) files here. Organize them in subdirectories as needed.
- `data/sectionwise_chunks/`: This folder will be automatically created when you run the chunking script. It contains the processed JSON chunk files.

**Note:** You can customize the input directory by setting the `INPUT_DIR_FOR_CHUNKING` environment variable if your data is located elsewhere.

## Workflow Overview

The system follows this pipeline:

1. **Preprocessing** (`section_wise_chunking.py`): Chunks markdown files into sections
2. **Database Setup**: Create PostgreSQL database and schema with pgvector
3. **Ingestion** (`ingest_section_chunks.py`): Load chunks into PostgreSQL/pgvector
4. **HNSW Index Setup** (`create_hnsw_indexes.py`): Create indexes for fast vector search (required before retrieval)
5. **Dataset Generation** (`generate_dataset_llm.py`): Create Q&A dataset (optional)
6. **Retrieval** (`retrieve_and_stitch.py`): Query and retrieve relevant sections
7. **Retrieval Evaluation** (`retrieval_evaluation.py`): Evaluate retrieval performance (optional)
8. **Answer Generation** (`generate_answers.py`): Generate answers using retrieved contexts
9. **Answer Evaluation** (`evaluate_answers.py`): Evaluate answer quality using LLM judge (optional)

## Detailed Workflow

### 1. Preprocessing: Section-wise Chunking

**Script**: `src/preprocessing/section_wise_chunking.py`

- **Input**: Markdown files in `data/eva_md_files/` (or path specified by `INPUT_DIR_FOR_CHUNKING`)
- **Output**: JSON files in `data/sectionwise_chunks/` (one JSON list per MD file)
- **Process**:
  - Parses markdown structure (headings, sections)
  - Drops front-matter sections (preface, table of contents, etc.) via `DROP_TITLES` catalog
  - Splits long sections while keeping tables atomic
  - Generates rich metadata for reassembly (section IDs, parent/child relationships, sibling links)
  - Emits chunks with breadcrumb paths and document titles

**Key Metadata Fields Generated**:
- `section_node_id`: Stable ID for the entire section (constant across splits, used for reassembly)
- `chunk_id`: Unique ID for each chunk (if section is split, includes split index)
- `section_path`: Breadcrumb path (e.g., `["Introduction", "Getting Started"]`)
- `doc_title`, `doc_id`, `doc_path`: Document identification
- `chunk_index`, `chunk_count`: Split information (0-based index, total chunks in section)
- `prev_chunk_id`, `next_chunk_id`: Links to previous/next chunk in same section

**Environment Variables**:
- `INPUT_DIR_FOR_CHUNKING`: Input directory (default: `./data/md`, but typically set to `./data/eva_md_files`)
- `OUTPUT_DIR`: Output directory (default: `./data/sectionwise_chunks`)
- `MAX_CHARS`: Max chunk size (default: 4000)
- `SPLIT_OVERLAP_CHARS`: Overlap between splits (default: 180)
- `MIN_CHARS`: Minimum chunk size (default: 250)
- `KEEP_FRONT_MATTER`: Keep front-matter sections (default: false)

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

### 3. HNSW Index Setup (Required Before Retrieval)

**Script**: `src/setup/create_hnsw_indexes.py`

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
- `PG_TABLE`: Table name (default: `idx_section_based_chunking`)
- `EMBED_DIM`: Embedding dimensions (default: 1536)

**Optional Environment Variables**:
- `HNSW_M`: Number of connections per layer (default: 16, range: 4-64)
- `HNSW_EF_CONSTRUCTION`: Size of candidate list during construction (default: 64, range: 4-1000)

### 4. Ingestion: Load Chunks into Database

**Script**: `src/ingestion/ingest_section_chunks.py`

- **Input**: JSON chunk files from preprocessing step
- **Output**: Chunks stored in PostgreSQL/pgvector table
- **Process**:
  - Reads all JSON files from `OUTPUT_DIR`
  - Converts chunks to LlamaIndex `TextNode` objects
  - Embeds text using OpenAI embeddings (configured via `EMBED_MODEL`)
  - Stores in PostgreSQL with hybrid search enabled (vector + BM25)

**Environment Variables**:
- `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_SCHEMA`, `PG_USER`, `PG_PASSWORD`: Database connection
- `PG_TABLE`: Table name (default: `idx_section_based_chunking`)
- `TEXT_SEARCH_CONFIG`: PostgreSQL text search config (default: `english`)
- `OVERWRITE_TABLE`: Drop and recreate table if exists (default: false)
- `EMBED_MODEL`: Embedding model (default: `text-embedding-3-small`)
- `EMBED_DIM`: Embedding dimensions (default: 1536)
- `OUTPUT_DIR`: Directory with chunk JSON files (default: `./data/sectionwise_chunks`)

### 5. Dataset Generation (Optional)

**Script**: `src/utils/generate_dataset_llm.py`

- **Input**: JSON chunk files
- **Output**: CSV file with questions, answers, and ground truth context
- **Process**:
  - Randomly samples chunks from JSON files
  - Uses LLM to gate-keep chunks (filter for product-specific content)
  - Generates Q&A pairs from accepted chunks
  - Outputs CSV with `question`, `answer`, `context`, `source_path`, `gt_section_id`

**Environment Variables**:
- `chunks_dir`: Directory with chunk JSON files (required)
- `QUERY_DATASET_SIZE`: Target number of Q&A pairs (required)
- `OPENAI_API_KEY`: OpenAI API key (required for Q&A generation)
- `SKIP_JUDGE_MODEL`: Model for gate-keeping (default: `gpt-4o-mini`)
- `QA_MODEL`: Model for Q&A generation (default: `gpt-4o-mini`)
- `MAX_CONTEXT_CHARS`: Max context length (default: 12000)

**Command Line Arguments**:
- `--output_csv`: Output CSV path (default: `query_dataset_with_qa_200.csv`)
- `--seed`: Random seed
- `--accept_all`: Skip LLM gate-keeping
- `--allow_blank_qa`: Allow blank Q/A if no API key

### 6. Retrieval: Query and Retrieve Context

**Script**: `src/retrieval/retrieve_and_stitch.py`

**⚠️ PREREQUISITE: Make sure you've created HNSW indexes before running retrieval!** Without indexes, vector similarity queries will be extremely slow. Run `python src/setup/create_hnsw_indexes.py --check` to verify indexes exist, or `python src/setup/create_hnsw_indexes.py` to create them.

- **Input**: CSV file with queries (or can be used programmatically)
- **Output**: CSV file with retrieved contexts and rankings
- **Process**:
  - Performs hybrid retrieval (vector + BM25) using QueryFusionRetriever
  - Deduplicates by section ID
  - Assembles split chunks back into complete sections
  - Applies optional product postfiltering (soft/hard/none)
  - Applies optional entity-based reranking
  - Returns top-K sections per query

**Retrieval Pipeline**:
1. **Hybrid Search**: Combines dense (vector) and sparse (BM25) retrieval
2. **Deduplication**: Keeps highest-scoring chunk per section
3. **Assembly**: Stitches split chunks back into complete sections
4. **Postfiltering** (optional): Filters/ranks by product mentions
5. **Reranking** (optional): Entity-aware reranking
6. **Trimming**: Returns top `FINAL_K` sections

**Environment Variables**:
- Database: `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_SCHEMA`, `PG_USER`, `PG_PASSWORD`, `PG_TABLE`
- Embedding: `EMBED_MODEL`, `EMBED_DIM`
- Retrieval: `RETRIEVE_TOP_K` (default: 30), `SPARSE_TOP_K` (default: 30), `FINAL_K` (default: 5)
- Postfiltering: `POSTFILTER_MODE` (none/soft/hard), `POSTFILTER_FIELDS`
- Reranking: `RERANKER_MODE` (none/entity/rrf/mmr/custom), plus various `RERANK_*` knobs
- Paths: `DATASET_QUERIES`, `OUTPUT_CSV`

### 7. Retrieval Evaluation

**Script**: `src/evaluation/retrieval_evaluation.py`

- **Input**: Dataset CSV and retrieval log CSV
- **Output**: Augmented CSV with evaluation metrics
- **Metrics**:
  - Hit rate (trimmed vs pre-trim only)
  - Rank statistics (mean, median, best, worst)
  - Match status (trimmed/pretrim_only/missed)

**Environment Variables**:
- `DATASET_QUERIES`: Path to dataset CSV
- `OUTPUT_CSV`: Path to retrieval log CSV
- `EVAL_OUTPUT_CSV`: Output path (default: adds `_eval` suffix)

### 8. Answer Generation

**Script**: `src/evaluation/generate_answers.py`

- **Input**: Retrieval log CSV (from `retrieve_and_stitch.py`)
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
- `RETRIEVAL_LOG_CSV` or `retrieval_log_csv`: Path to retrieval log CSV (from `retrieve_and_stitch.py`)
- `ANSWER_LOG_CSV` or `answer_log_csv`: Path to save generated answers (e.g., `logs/answer_log.csv`)
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

### 9. Answer Evaluation

**Script**: `src/evaluation/evaluate_answers.py`

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
- `RETRIEVAL_LOG_CSV` or `retrieval_log_csv`: Path to retrieval log CSV (contains ground truth answers)
- `ANSWER_LOG_CSV` or `answer_log_csv`: Path to answer log CSV (from `generate_answers.py`)
- `ANSWER_EVAL_OUTPUT_CSV` or `answer_eval_output_csv`: Output path for evaluation results (e.g., `logs/answer_eval_log.csv`)
- `OPENAI_API_KEY`: Required for judge API calls

**Environment Variables (OPTIONAL)**:
- `JUDGE_MODEL`: Model to use for judging (default: `gpt-4o`)

**Output Format**:
CSV file with all columns from retrieval log plus: `generated_answer`, `judge_score`, `judge_reasoning`

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
   Create a `data/` folder at the repository root and place your markdown files in `data/eva_md_files/` (see Data Setup section above).

5. **Chunk markdown files**:
   Set the input directory (if using default location, you can skip this):
   ```bash
   export INPUT_DIR_FOR_CHUNKING=./data/eva_md_files
   ```
   Run the chunking script:
   ```bash
   uv run python src/preprocessing/section_wise_chunking.py
   ```

6. **Ingest chunks**:
   ```bash
   uv run python src/ingestion/ingest_section_chunks.py
   ```

7. **Create HNSW indexes** (recommended for faster retrieval):
   First, check if indexes exist:
   ```bash
   uv run python src/setup/create_hnsw_indexes.py --check
   ```
   If indexes are missing, create them:
   ```bash
   uv run python src/setup/create_hnsw_indexes.py
   ```

8. **Generate dataset** (optional):
   ```bash
   uv run python src/utils/generate_dataset_llm.py
   ```

9. **Run retrieval**:
   ```bash
   uv run python src/retrieval/retrieve_and_stitch.py
   ```

10. **Evaluate retrieval**:
   ```bash
   uv run python src/evaluation/retrieval_evaluation.py
   ```

11. **Generate answers**:
   ```bash
   # Option 1: Using OpenAI (default)
   export ANSWER_MODEL=gpt-4o-mini
   export OPENAI_API_KEY=your_openai_key
   export RETRIEVAL_LOG_CSV=logs/retrieval_log.csv
   export ANSWER_LOG_CSV=logs/answer_log.csv
   uv run python src/evaluation/generate_answers.py
   
   # Option 2: Using Groq (faster, lower cost)
   export USE_GROQ=true
   export GROQ_MODEL=llama-3.3-70b-versatile
   export GROQ_API_KEY=your_groq_key
   export RETRIEVAL_LOG_CSV=logs/retrieval_log.csv
   export ANSWER_LOG_CSV=logs/answer_log.csv
   uv run python src/evaluation/generate_answers.py
   ```

12. **Evaluate answers**:
    ```bash
    export RETRIEVAL_LOG_CSV=logs/retrieval_log.csv
    export ANSWER_LOG_CSV=logs/answer_log.csv
    export ANSWER_EVAL_OUTPUT_CSV=logs/answer_eval_log.csv
    uv run python src/evaluation/evaluate_answers.py
    ```

## Project Structure

```
src/
├── preprocessing/
│   └── section_wise_chunking.py    # Chunk markdown into sections
├── setup/
│   ├── create_db.py                 # Database setup script
│   └── create_hnsw_indexes.py      # Create HNSW indexes for vector search
├── ingestion/
│   └── ingest_section_chunks.py    # Load chunks into PostgreSQL
├── retrieval/
│   └── retrieve_and_stitch.py      # Query and retrieve contexts
├── evaluation/
│   ├── retrieval_evaluation.py     # Evaluate retrieval performance
│   ├── generate_answers.py         # Generate answers using retrieved contexts
│   └── evaluate_answers.py         # Evaluate answer quality using LLM judge
├── reranking/
│   ├── entity_based_reranking.py   # Entity-aware reranking
│   ├── entity_extraction.py         # Extract entities using spaCy
│   └── ner_extras.py                # Entity augmentation utilities
├── postfiltering/
│   └── product_postfilter.py       # Product-aware postfiltering
├── catalog/
│   ├── product_names.py            # Product name catalog
│   └── drop_titles.py              # Titles to drop from chunks
└── utils/
    └── generate_dataset_llm.py     # Generate Q&A dataset
```

## Configuration

Most configuration is done via environment variables. See each script's docstring for details.

Key configuration areas:
- **Database**: Connection settings, table names, schema
- **Embeddings**: Model, dimensions
- **Chunking**: Sizes, overlap, directories
- **Retrieval**: Top-K values, reranking mode, postfiltering mode
- **Reranking**: Entity weights, fuzzy matching thresholds

## Notes

- The system uses hybrid search (vector + BM25) for better retrieval
- Sections are reassembled from split chunks during retrieval
- Product-aware filtering and reranking can improve precision for product-specific queries
- Metadata includes rich section hierarchy for context-aware retrieval
