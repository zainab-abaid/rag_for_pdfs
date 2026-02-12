# Reranking Experiment Pipeline

This pipeline orchestrates the entire process of reranking experiments, from chunking to answer evaluation. It is designed to be modular, resumable, and idempotent, ensuring that experiments can be run efficiently and reliably.

## Overview

The reranking experiment pipeline includes the following steps:

1. **Section-wise Chunking**: Preprocesses documents into manageable chunks.
2. **Database Setup**: Creates the PostgreSQL database and schema, and enables the `pgvector` extension for vector search.
3. **Ingestion**: Loads the section-based chunks into the database.
4. **HNSW Index Creation**: Builds HNSW indexes for fast vector similarity search.
5. **Retrieval with Different Rerankers**: Performs hybrid retrieval and applies various reranking strategies.
6. **Retrieval Evaluation**: Evaluates the retrieval results against ground truth.
7. **Answer Generation**: Generates answers based on the retrieved contexts.
8. **Answer Evaluation**: Evaluates the quality of the generated answers.

## Features

- **Resumable Execution**: The pipeline can resume from the last completed step, skipping already completed steps.
- **Reset Option**: Allows for a fresh run by clearing the pipeline state.
- **Consolidated Logging**: All steps log their outputs to a single consolidated log file.
- **Error Handling**: Stops execution on failure, with detailed logs for debugging.

## Prerequisites

- Python environment with required dependencies installed.
- PostgreSQL database with `pgvector` extension enabled.
- Input data prepared in the appropriate format.

## Pipeline Steps

### 1. Section-wise Chunking

**Script**: `src/preprocessing/section_wise_chunking.py`

- **Input**: Raw documents.
- **Output**: Section-based chunks stored in JSON files.

### 2. Database Setup

**Script**: `src/setup/create_db.py`

- Creates the database, schema, and enables the `pgvector` extension.
- Idempotent: Can be run multiple times without issues.

### 3. Ingestion

**Script**: `src/ingestion/ingest_section_chunks.py`

- Loads the section-based chunks into the PostgreSQL database.
- Supports hybrid search (vector + BM25).

### 4. HNSW Index Creation

**Script**: `src/setup/create_hnsw_indexes.py`

- Builds HNSW indexes for fast vector similarity search.
- Required for efficient retrieval.

### 5. Retrieval with Different Rerankers

**Script**: `src/rerankingexperiments/retrieve_with_diff_rerankers.py`

- Performs hybrid retrieval (vector + BM25).
- Applies various reranking strategies, such as:
  - None
  - Entity Based Reranking
  - Colbert Reranking
  - FlagEmbeddingsReranking
  - LLM Rerranking
  - MMR Reranking 

### 6. Retrieval Evaluation

**Script**: `src/rerankingexperiments/retrieval_evaluation_diff_rerankers.py`

- Evaluates the retrieval results against ground truth.
- Outputs metrics such as precision, recall, and F1-score.

### 7. Answer Generation

**Script**: `src/rerankingexperiments/generate_answers_diff_rerankers.py`

- Generates answers based on the retrieved contexts.
- Supports multiple answer generation models.

### 8. Answer Evaluation

**Script**: `src/rerankingexperiments/evaluate_answers_diff_rerankers.py`

- Evaluates the quality of the generated answers using an LLM judge.
- Outputs evaluation metrics to a log file.

## How to Run

### Fresh Run

```bash
uv run python src/rerankingexperiments/run_reranking_experiment.py
```

### Resume Previous Run

```bash
uv run python src/rerankingexperiments/run_reranking_experiment.py
```

### Fresh Run (Ignore Existing State)

```bash
uv run python src/rerankingexperiments/run_reranking_experiment.py --no-resume
```

### Completely Fresh Run (Reset State)

```bash
uv run python src/rerankingexperiments/run_reranking_experiment.py --reset
```

## Environment Variables

The following environment variables must be configured:

- **Database Configuration**:
  - `PG_HOST`: PostgreSQL host (default: `localhost`)
  - `PG_PORT`: PostgreSQL port (default: `5432`)
  - `PG_DB`: Database name
  - `PG_SCHEMA`: Schema name
  - `PG_USER`: Database user
  - `PG_PASSWORD`: Database password

- **Ingestion Configuration**:
  - `EMBED_MODEL`: Embedding model (default: `text-embedding-3-small`)
  - `EMBED_DIM`: Embedding dimensions (default: `1536`)

- **Retrieval Configuration**:
  - `RETRIEVE_TOP_K`: Number of top results to retrieve (default: `30`)
  - `SPARSE_TOP_K`: Number of sparse results to retrieve (default: `30`)
  - `FINAL_K`: Number of final results to return (default: `5`)

- **Reranking Configuration**:
  - `RERANKER_MODE`: Reranking mode (e.g., `none`, `entity`, `colbert`, `flag`, `llm`, `mmr`)

## Logs

Logs are saved in the `logs/` directory. Each run generates a timestamped log file, e.g., `logs/reranking_experiment_YYYYMMDD_HHMMSS.log`.

## Notes

- The pipeline is modular; each step can be run independently if needed.
- Ensure that the database and input data are properly set up before running the pipeline.
- Use the `--reset` flag for a completely fresh run, clearing all previous state.