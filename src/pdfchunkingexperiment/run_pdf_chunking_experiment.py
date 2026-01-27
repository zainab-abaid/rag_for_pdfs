#!/usr/bin/env python3
"""
Driver script for testing different PDF chunking strategies.

Runs the full pipeline for PDF-based RAG experiments:
1. Chunk PDF documents
2. (Optional) Create database (skipped by default)
3. Ingest PDF chunks
4. Create HNSW indexes (same script as usual but table name is changed)
5. Retrieve chunks
6. Generate answers
7. Evaluate answers

By default, database creation step is skipped if DB already exists.
Supports resuming from failed steps and maintains a consolidated log.

Resume / Reset behavior:
- resume=True  : Load existing state, skip completed steps, append to existing log
- resume=False : Do not load existing state; run all steps from the beginning using a new log file
- reset=True   : Delete existing state, then run all steps from the beginning using a new log file

Optional DB creation:
- By default, the "Create database" step is skipped if the state exists
- Will only run if reset=True or resume=False

Example commands:

1. Completely fresh run, delete state, DB skipped by default:
   uv run python src/pdfchunkingexperiment/run_pdf_chunking_experiment.py --reset

2. Completely fresh run, delete state, including optional DB creation:
   uv run python src/pdfchunkingexperiment/run_pdf_chunking_experiment.py --reset --create-db

3. Resume previous run (skip completed steps, DB skipped):
   uv run python src/pdfchunkingexperiment/run_pdf_chunking_experiment.py

4. Fresh run but ignore existing state (state file is preserved, DB skipped by default):
   uv run python src/pdfchunkingexperiment/run_pdf_chunking_experiment.py --no-resume

5. Fresh run (resume if state exists), including optional DB creation:
   uv run python src/pdfchunkingexperiment/run_pdf_chunking_experiment.py --create-db
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# -------------------------------------------------
# Paths
# -------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]   # src → project_root
SRC_DIR = PROJECT_ROOT / "src"
PDFCHUNK_DIR = SRC_DIR / "pdfchunkingexperiment"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

STATE_DIR = PROJECT_ROOT / "pipeline_state"
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "pdf_chunking_state.json"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PYTHON = sys.executable  # current venv python

# -------------------------------------------------
# Pipeline definition (ORDER MATTERS)
# -------------------------------------------------
PIPELINE = [
    ("PDF documents chunking",
     PDFCHUNK_DIR / "pdf_documents_chunking.py"),

    ("Create database (optional)",
     SRC_DIR / "setup" / "create_db.py"),

    ("PDF chunks ingestion",
     PDFCHUNK_DIR / "pdf_chunks_ingestion.py"),

    ("Create HNSW indexes",
     PDFCHUNK_DIR / "pdf_create_hnsw_indexes.py"),

    ("PDF chunks retrieval",
     PDFCHUNK_DIR / "pdf_chunks_retrieval.py"),

    ("PDF answers generation",
     PDFCHUNK_DIR / "pdf_answers_generation.py"),

    ("PDF answers evaluation",
     PDFCHUNK_DIR / "pdf_answers_evaluation.py"),
]

# -------------------------------------------------
# State Management
# -------------------------------------------------
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"completed_steps": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def reset_state():
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("✓ Pipeline state reset")

def is_step_completed(step_name, state):
    return step_name in state.get("completed_steps", [])

def mark_step_completed(step_name, state):
    if "completed_steps" not in state:
        state["completed_steps"] = []
    if step_name not in state["completed_steps"]:
        state["completed_steps"].append(step_name)
    save_state(state)

# -------------------------------------------------
# Runner
# -------------------------------------------------
def run_pipeline(resume=True, reset=False, create_db=False):
    if reset:
        reset_state()
        resume = False

    state = load_state() if resume else {"completed_steps": []}

    # Log file handling (Option A)
    if resume and "log_file" in state:
        LOG_FILE = PROJECT_ROOT / state["log_file"]
        log_mode = "a"
    else:
        LOG_FILE = LOG_DIR / f"pdf_chunking_experiment_{TIMESTAMP}.log"
        log_mode = "w"
        state["log_file"] = str(LOG_FILE.relative_to(PROJECT_ROOT))
        save_state(state)

    print(f"\n{'='*80}")
    print("PDF CHUNKING STRATEGIES PIPELINE")
    print(f"{'='*80}")
    print(f"Log file: {LOG_FILE}")

    if resume and state.get("completed_steps"):
        print(f"Resuming pipeline - {len(state['completed_steps'])} steps already completed")
    print()

    with open(LOG_FILE, log_mode, encoding="utf-8") as log:
        log.write("PDF CHUNKING STRATEGIES PIPELINE\n")
        log.write(f"Started at: {datetime.now()}\n")
        if resume and state.get("completed_steps"):
            log.write(f"Resuming - Previously completed: {state['completed_steps']}\n")
        log.write("=" * 100 + "\n\n")

        for idx, (step_name, script_path) in enumerate(PIPELINE, 1):
            # Skip database creation by default
            if "create database" in step_name.lower() and not create_db:
                print(f"⏭  [{idx}/{len(PIPELINE)}] Skipping DB creation (use --create-db to run): {step_name}")
                log.write(f"[STEP {idx}/{len(PIPELINE)}] SKIPPED (optional, --create-db not set): {step_name}\n")
                log.write("-" * 100 + "\n\n")
                mark_step_completed(step_name, state)
                continue

            if resume and is_step_completed(step_name, state):
                print(f"⏭  [{idx}/{len(PIPELINE)}] Skipping (already completed): {step_name}")
                log.write(f"[STEP {idx}/{len(PIPELINE)}] SKIPPED (already completed): {step_name}\n")
                log.write("-" * 100 + "\n\n")
                continue

            if not script_path.exists():
                error_msg = f"Script not found: {script_path}"
                log.write(f"[ERROR] {error_msg}\n")
                print(f"❌ ERROR: {error_msg}")
                print(f"📄 See log: {LOG_FILE}")
                sys.exit(1)

            cmd = [PYTHON, str(script_path)]

            log.write(f"[STEP {idx}/{len(PIPELINE)}] {step_name}\n")
            log.write(f"Script : {script_path}\n")
            log.write(f"Command: {' '.join(cmd)}\n\n")
            log.flush()

            print(f"▶ [{idx}/{len(PIPELINE)}] Running: {step_name}")

            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.stdout:
                log.write(process.stdout)
            if process.stderr:
                log.write("\n[STDERR]\n")
                log.write(process.stderr)

            if process.returncode != 0:
                log.write(f"\n[FAILED] {step_name} (exit code {process.returncode})\n")
                log.write("=" * 100 + "\n")
                print(f"\n❌ FAILED: {step_name}")
                print(f"📄 Full log: {LOG_FILE}")
                sys.exit(process.returncode)

            mark_step_completed(step_name, state)
            log.write(f"\n[SUCCESS] {step_name}\n")
            log.write("-" * 100 + "\n\n")
            print(f"  [OK] Completed\n")

        log.write("\nALL STEPS COMPLETED SUCCESSFULLY\n")
        log.write(f"Finished at: {datetime.now()}\n")

    reset_state()
    print(f"{'='*80}")
    print("✅ PDF chunking experiment completed successfully")
    print(f"📄 Log saved at: {LOG_FILE}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PDF chunking strategies pipeline"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset state and run all steps from beginning"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous state (but don't clear it)"
    )

    parser.add_argument(
        "--create-db",
        action="store_true",
        help="Run the database creation step"
    )

    args = parser.parse_args()

    run_pipeline(
        resume=not args.no_resume,
        reset=args.reset,
        create_db=args.create_db
    )

