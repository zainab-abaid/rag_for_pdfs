# PDF Markdown RAG System

A RAG (Retrieval-Augmented Generation) system for querying PDF documentation that has been converted to Markdown format. This system uses section-wise chunking, hybrid search (vector + BM25), and optional entity-based reranking for improved retrieval accuracy.

## Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- OpenAI API key (for embeddings and optional dataset generation)
- spaCy English model: `uv run python -m spacy download en_core_web_md`

## Workflow Overview

The system follows this pipeline:

1. **Preprocessing** (`section_wise_chunking.py`): Chunks markdown files into sections
2. **Database Setup**: Create PostgreSQL database and schema with pgvector
3. **Ingestion** (`ingest_section_chunks.py`): Load chunks into PostgreSQL/pgvector
4. **Dataset Generation** (`generate_dataset_llm.py`): Create Q&A dataset (optional)
5. **Retrieval** (`retrieve_and_stitch.py`): Query and retrieve relevant sections
6. **Evaluation** (`retrieval_evaluation.py`): Evaluate retrieval performance

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
- `INPUT_DIR_FOR_CHUNKING`: Input directory (default: `./data/md`)
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

### 3. Ingestion: Load Chunks into Database

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

### 4. Dataset Generation (Optional)

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

### 5. Retrieval: Query and Retrieve Context

**Script**: `src/retrieval/retrieve_and_stitch.py`

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

### 6. Evaluation

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

4. **Chunk markdown files**:
   ```bash
   uv run python src/preprocessing/section_wise_chunking.py
   ```

5. **Ingest chunks**:
   ```bash
   uv run python src/ingestion/ingest_section_chunks.py
   ```

6. **Generate dataset** (optional):
   ```bash
   uv run python src/utils/generate_dataset_llm.py
   ```

7. **Run retrieval**:
   ```bash
   uv run python src/retrieval/retrieve_and_stitch.py
   ```

8. **Evaluate**:
   ```bash
   uv run python src/evaluation/retrieval_evaluation.py
   ```

## Project Structure

```
src/
├── preprocessing/
│   └── section_wise_chunking.py    # Chunk markdown into sections
├── setup/
│   └── create_db.py                 # Database setup script
├── ingestion/
│   └── ingest_section_chunks.py    # Load chunks into PostgreSQL
├── retrieval/
│   └── retrieve_and_stitch.py      # Query and retrieve contexts
├── evaluation/
│   └── retrieval_evaluation.py     # Evaluate retrieval performance
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
