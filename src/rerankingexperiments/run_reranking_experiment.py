#!/usr/bin/env python3
"""
Master driver script for reranking experiment.

Runs the full pipeline:
- chunking
- db setup
- ingestion
- indexing
- retrieval with different rerankers
- retrieval evaluation
- answer generation
- answer evaluation

Creates ONE consolidated log file.
Supports resuming from failed steps.

Resume / Reset behavior:
- resume=True  : Load existing state, skip completed steps, append to existing log
- resume=False : Do not load existing state; run all steps from the beginning using a new log file
- reset=True   : Delete existing state, then run all steps from the beginning using a new log file

Example commands (using uv):
- Fresh run (resume if state exists):
  uv run python src/rerankingexperiments/run_reranking_experiment.py

- Resume previous run:
  uv run python src/rerankingexperiments/run_reranking_experiment.py

- Fresh run, ignore existing state (state file is preserved):
  uv run python src/rerankingexperiments/run_reranking_experiment.py --no-resume

- Completely fresh run (state deleted):
  uv run python src/rerankingexperiments/run_reranking_experiment.py --reset
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
PROJECT_ROOT = THIS_FILE.parents[2]   # src/rerankingexperiments → src → project_root
SRC_DIR = PROJECT_ROOT / "src"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

STATE_DIR = PROJECT_ROOT / "pipeline_state"
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "reranking_experiment_state.json"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PYTHON = sys.executable  # current venv python

# -------------------------------------------------
# Pipeline definition (ORDER MATTERS)
# -------------------------------------------------
PIPELINE = [
    ("Section-wise chunking",
     SRC_DIR / "preprocessing" / "section_wise_chunking.py"),

    ("Create database",
     SRC_DIR / "setup" / "create_db.py"),

    ("Ingest section chunks",
     SRC_DIR / "ingestion" / "ingest_section_chunks.py"),

    ("Create HNSW indexes",
     SRC_DIR / "setup" / "create_hnsw_indexes.py"),

    ("Retrieve with different rerankers",
     SRC_DIR / "rerankingexperiments" / "retrieve_with_diff_rerankers.py"),

    ("Retrieval evaluation (diff rerankers)",
     SRC_DIR / "rerankingexperiments" / "retrieval_evaluation_diff_rerankers.py"),

    ("Generate answers (diff rerankers)",
     SRC_DIR / "rerankingexperiments" / "generate_answers_diff_rerankers.py"),

    ("Evaluate answers (diff rerankers)",
     SRC_DIR / "rerankingexperiments" / "evaluate_answers_diff_rerankers.py"),
]

# -------------------------------------------------
# State Management
# -------------------------------------------------
def load_state():
    """Load pipeline state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"completed_steps": []}

def save_state(state):
    """Save pipeline state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def reset_state():
    """Clear all pipeline state (force re-run from beginning)."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("✓ Pipeline state reset")

def is_step_completed(step_name, state):
    """Check if a step was previously completed."""
    return step_name in state.get("completed_steps", [])

def mark_step_completed(step_name, state):
    """Mark a step as completed."""
    if "completed_steps" not in state:
        state["completed_steps"] = []
    if step_name not in state["completed_steps"]:
        state["completed_steps"].append(step_name)
    save_state(state)

# -------------------------------------------------
# Runner
# -------------------------------------------------
def run_pipeline(resume=True, reset=False):
    """
    Run the pipeline.
    
    Args:
        resume: If True, skip already completed steps
        reset: If True, clear state and run all steps
    """
    if reset:
        reset_state()
        resume = False

    state = load_state() if resume else {"completed_steps": []}

    # ---------------- OPTION A LOGIC (ONLY ADDITION) ----------------
    if resume and "log_file" in state:
        LOG_FILE = PROJECT_ROOT / state["log_file"]
        log_mode = "a"
    else:
        LOG_FILE = LOG_DIR / f"reranking_experiment_{TIMESTAMP}.log"
        log_mode = "w"
        state["log_file"] = str(LOG_FILE.relative_to(PROJECT_ROOT))
        save_state(state)
    # ----------------------------------------------------------------

    print(f"\n{'='*80}")
    print("RERANKING EXPERIMENT PIPELINE")
    print(f"{'='*80}")
    print(f"Log file: {LOG_FILE}")

    if resume and state.get("completed_steps"):
        print(f"Resuming pipeline - {len(state['completed_steps'])} steps already completed")
    print()

    with open(LOG_FILE, log_mode, encoding="utf-8") as log:
        log.write("RERANKING EXPERIMENT PIPELINE\n")
        log.write(f"Started at: {datetime.now()}\n")
        if resume and state.get("completed_steps"):
            log.write(f"Resuming - Previously completed: {state['completed_steps']}\n")
        log.write("=" * 100 + "\n\n")

        for idx, (step_name, script_path) in enumerate(PIPELINE, 1):
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
                log.write(
                    f"\n[FAILED] {step_name} "
                    f"(exit code {process.returncode})\n"
                )
                log.write("=" * 100 + "\n")

                print(f"\n❌ FAILED: {step_name}")
                if process.stderr:
                    print(f"Error: {process.stderr[:500]}")
                print(f"📄 Full log: {LOG_FILE}")
                print(f"\n💡 To resume from this point, run the script again")
                print(f"💡 To start fresh, run: python {THIS_FILE.name} --reset")
                sys.exit(process.returncode)

            mark_step_completed(step_name, state)

            log.write(f"\n[SUCCESS] {step_name}\n")
            log.write("-" * 100 + "\n\n")
            print(f"  ✓ Completed\n")

        log.write("\nALL STEPS COMPLETED SUCCESSFULLY\n")
        log.write(f"Finished at: {datetime.now()}\n")

    reset_state()

    print(f"{'='*80}")
    print("✅ Reranking experiment completed successfully")
    print(f"📄 Log saved at: {LOG_FILE}")
    print(f"{'='*80}\n")

# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run reranking experiment pipeline")
    parser.add_argument("--reset", action="store_true",
                        help="Reset state and run all steps from beginning")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from previous state (but don't clear it)")

    args = parser.parse_args()

    run_pipeline(resume=not args.no_resume, reset=args.reset)
