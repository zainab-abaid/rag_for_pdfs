#!/usr/bin/env python3
"""
Recursively finds all PDFs under a directory, uploads/indexes them into a Gemini File Search store,
then reads a CSV (expects a column named 'question'), queries Gemini with File Search,
and writes a JSONL log of {question, answer, ...}.

Prereqs:
  pip install -U google-genai

Auth:
  export GEMINI_API_KEY="YOUR_KEY"   (or GOOGLE_API_KEY)

Usage example:
  python gemini_filesearch.py \
    --store_display_name "my-pdf-store" \
    --model "gemini-2.5-flash"

Note: 
  - PDFs are automatically read from data/eva-docs directory
  - Questions are read from dataset_queries env var (default: query_dataset_with_qa.csv)
  - Requires GEMINI_API_KEY environment variable
  - Outputs CSV in format compatible with evaluate_answers.py
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_pdf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".pdf"


def discover_pdfs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"PDF root not found: {root}")
    pdfs = [p for p in root.rglob("*.pdf") if is_pdf(p)]
    # Also catch .PDF
    pdfs += [p for p in root.rglob("*.PDF") if is_pdf(p)]
    # Deduplicate
    uniq = sorted({p.resolve() for p in pdfs})
    return uniq


def backoff_sleep(attempt: int, base: float = 1.2, cap: float = 30.0) -> None:
    # jittered exponential backoff
    delay = min(cap, (base ** attempt) + random.random())
    time.sleep(delay)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (overload, rate limit, etc.)."""
    error_str = str(error).lower()
    retryable_keywords = [
        "overloaded",
        "rate limit",
        "too many requests",
        "service unavailable",
        "503",
        "429",
        "quota exceeded",
        "temporarily unavailable",
    ]
    return any(keyword in error_str for keyword in retryable_keywords)


def exponential_backoff_sleep(attempt: int, base_delay: float = 10.0, max_delay: float = 120.0) -> None:
    """Sleep with exponential backoff and jitter."""
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    # Add jitter (random 0-20% of delay) to avoid thundering herd
    jitter = delay * 0.2 * random.random()
    total_delay = delay + jitter
    time.sleep(total_delay)


def wait_operation_done(client: genai.Client, operation: Any, poll_secs: float = 5.0) -> Any:
    """
    Polls a long-running operation until done.
    The File Search docs show checking operation.done and refreshing via client.operations.get().  # noqa
    """
    while True:
        try:
            if getattr(operation, "done", False):
                return operation
        except Exception:
            pass

        time.sleep(poll_secs)
        operation = client.operations.get(operation)


def safe_model_dump(obj: Any) -> Any:
    """
    Best-effort conversion for logging/debugging.
    """
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def upload_pdfs_to_store(
    client: genai.Client,
    store_name: str,
    pdf_paths: List[Path],
    chunking: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> List[Tuple[Path, str]]:
    """
    Uploads each PDF to the file search store; returns list of (path, status).
    """
    results: List[Tuple[Path, str]] = []

    for i, pdf in enumerate(pdf_paths, start=1):
        display_name = pdf.name  # shows up in citations per docs
        cfg: Dict[str, Any] = {"display_name": display_name}
        if chunking:
            cfg["chunking_config"] = chunking

        print(f"[{i}/{len(pdf_paths)}] Uploading: {pdf} (display_name={display_name})")
        ok = False
        last_err: Optional[str] = None

        for attempt in range(max_retries):
            try:
                op = client.file_search_stores.upload_to_file_search_store(
                    file=str(pdf),
                    file_search_store_name=store_name,
                    config=cfg,
                )
                op = wait_operation_done(client, op, poll_secs=5.0)
                # If operation has error info, log it
                op_dump = safe_model_dump(op)
                if isinstance(op_dump, dict) and op_dump.get("error"):
                    raise RuntimeError(f"Upload operation reported error: {op_dump.get('error')}")
                ok = True
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                print(f"  -> upload failed (attempt {attempt+1}/{max_retries}): {last_err}", file=sys.stderr)
                backoff_sleep(attempt + 1)

        results.append((pdf, "ok" if ok else f"failed: {last_err}"))
        if not ok:
            print(f"WARNING: giving up on {pdf} after {max_retries} attempts", file=sys.stderr)

    return results


def read_questions(csv_path: Path, column: str = "question") -> List[Dict[str, str]]:
    """
    Read questions and ground truth answers from CSV.
    Returns list of dicts with 'question' and 'answer' (ground truth) keys.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"CSV must contain a '{column}' column. Found: {reader.fieldnames}")
        for row in reader:
            q = (row.get(column) or "").strip()
            if q:
                rows.append({
                    "question": q,
                    "answer": (row.get("answer") or "").strip(),  # Ground truth answer
                })
    return rows


def ask_with_file_search(
    client: genai.Client,
    model: str,
    store_name: str,
    question: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Queries Gemini using File Search tool bound to the given store.
    Returns a dict suitable for JSONL logging.
    Includes retry logic for overload/rate limit errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[store_name]
                            )
                        )
                    ]
                ),
            )

            # Pull answer text
            answer_text = getattr(resp, "text", None)

            # Best-effort: include raw response dump for citations/grounding metadata if you want later parsing.
            # (Shape can evolve, so keep it as a blob.)
            resp_dump = safe_model_dump(resp)

            return {
                "ts_utc": utc_now_iso(),
                "question": question,
                "answer": answer_text,
                "model": model,
                "file_search_store": store_name,
                "raw_response": resp_dump,
            }

        except Exception as e:
            if is_retryable_error(e) and attempt < max_retries:
                wait_time = 10.0 * (2 ** (attempt - 1))  # 10s, 20s, 40s, 80s
                print(f"  -> Model overloaded/rate limited (attempt {attempt}/{max_retries}), retrying in {wait_time:.0f}s...", file=sys.stderr)
                exponential_backoff_sleep(attempt, base_delay=10.0)
                continue
            else:
                err = f"{type(e).__name__}: {e}"
                print(f"  -> query failed (attempt {attempt}/{max_retries}): {err}", file=sys.stderr)
                if attempt >= max_retries:
                    print(f"  -> Max retries ({max_retries}) exceeded", file=sys.stderr)
                # For non-retryable errors or final attempt, break and return error
                if attempt >= max_retries:
                    break
                backoff_sleep(attempt + 1)

    return {
        "ts_utc": utc_now_iso(),
        "question": question,
        "answer": None,
        "model": model,
        "file_search_store": store_name,
        "error": "Max retries exceeded",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_csv", default=None, help="CSV file with a 'question' column (default: from dataset_queries env var).")
    parser.add_argument("--question_column", default="question", help="CSV column name to read questions from.")
    parser.add_argument("--store_display_name", default="pdf-store", help="Human-friendly store display name.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name.")
    parser.add_argument("--answer_log_csv", default=None, help="Output CSV log file (default: logs/answers_gemini_{model}_{dataset}.csv).")
    parser.add_argument("--skip_upload", action="store_true", help="Skip uploading PDFs (assumes store already populated).")
    parser.add_argument("--chunk_tokens", type=int, default=None, help="Optional: max tokens per chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=None, help="Optional: max overlap tokens.")
    args = parser.parse_args()

    # Auth: Require GEMINI_API_KEY from env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY environment variable is required.\n"
                        "Set it in your .env file or export it: export GEMINI_API_KEY='your-key'")
    client = genai.Client(api_key=api_key)

    # PDF root is fixed to data/eva-docs directory
    pdf_root = Path("data/eva-docs").resolve()
    
    # Get questions CSV from env or argument
    csv_path = args.questions_csv
    if not csv_path:
        csv_path = os.getenv("dataset_queries") or os.getenv("DATASET_QUERIES")
        if not csv_path:
            raise SystemExit("ERROR: --questions_csv required or set dataset_queries environment variable.")
    csv_path = Path(csv_path).expanduser().resolve()
    
    # Determine output log path
    if args.answer_log_csv:
        answer_log_path = Path(args.answer_log_csv).expanduser().resolve()
    else:
        # Default: logs/answers_gemini_{model}_{dataset}.csv
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        model_safe = args.model.replace("/", "_").replace("\\", "_")
        dataset_name = csv_path.stem
        answer_log_path = logs_dir / f"answers_gemini_{model_safe}_{dataset_name}.csv"

    # 1) Create store (names are globally scoped; API returns a resource name like fileSearchStores/xxx)
    store = client.file_search_stores.create(config={"display_name": args.store_display_name})
    store_name = store.name
    print(f"Created File Search store: {store_name} (display_name={args.store_display_name})")

    # 2) Discover + upload PDFs from data/eva-docs
    pdfs = discover_pdfs(pdf_root)
    print(f"Discovered {len(pdfs)} PDF(s) under: {pdf_root}")

    chunking = None
    if args.chunk_tokens is not None or args.chunk_overlap is not None:
        if args.chunk_tokens is None or args.chunk_overlap is None:
            raise ValueError("--chunk_tokens and --chunk_overlap must be provided together.")
        chunking = {
            "white_space_config": {
                "max_tokens_per_chunk": args.chunk_tokens,
                "max_overlap_tokens": args.chunk_overlap,
            }
        }

    upload_status = []
    if not args.skip_upload:
        upload_status = upload_pdfs_to_store(client, store_name, pdfs, chunking=chunking)
        failed = [p for p, s in upload_status if not s.startswith("ok")]
        print(f"Upload complete. ok={len(upload_status)-len(failed)}, failed={len(failed)}")

    # 3) Read questions and ground truth answers
    question_rows = read_questions(csv_path, column=args.question_column)
    print(f"Loaded {len(question_rows)} question(s) from CSV: {csv_path}")

    # 4) Ask questions and save to CSV (matching format expected by evaluate_answers.py)
    answer_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for existing answers to avoid re-querying
    existing_answers = {}
    if answer_log_path.exists():
        try:
            with answer_log_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = (row.get("question") or "").strip()
                    ans = (row.get("generated_answer") or "").strip()
                    if q and ans:
                        existing_answers[q] = ans
            if existing_answers:
                print(f"Loaded {len(existing_answers)} existing answers from {answer_log_path}")
        except Exception as e:
            print(f"Warning: Could not load existing answers: {e}", file=sys.stderr)
    
    # Fieldnames matching generate_answers.py format
    fieldnames = ["question", "generated_answer", "context_used"]
    rows_to_write = []
    
    for idx, q_row in enumerate(question_rows, start=1):
        question = q_row["question"]
        print(f"[Q {idx}/{len(question_rows)}] {question[:90]}{'...' if len(question) > 90 else ''}")
        
        # Check if we already have an answer
        if question in existing_answers:
            generated_answer = existing_answers[question]
            print(f"  -> Using cached answer")
        else:
            # Query Gemini
            record = ask_with_file_search(client, args.model, store_name, question)
            generated_answer = record.get("answer") or ""
            if not generated_answer and "error" in record:
                print(f"  -> Error: {record.get('error', 'Unknown error')}", file=sys.stderr)
        
        # Save row (context_used can be empty or contain metadata)
        rows_to_write.append({
            "question": question,
            "generated_answer": generated_answer,
            "context_used": "",  # Gemini File Search doesn't expose context directly
        })
    
    # Write all rows to CSV
    with answer_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)
    
    print(f"\nDone. Wrote answer log: {answer_log_path}")
    print(f"  Total questions: {len(question_rows)}")
    print(f"  Generated answers: {len([r for r in rows_to_write if r['generated_answer']])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
