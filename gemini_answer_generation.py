#!/usr/bin/env python3
"""
Generate answers using an existing Gemini File Search store.

This script:
1. Connects to an existing Gemini File Search store (by name or display name)
2. Reads questions from CSV
3. Generates answers using the store
4. Saves answers to CSV (same format as generate_answers.py)

Use this when you've already uploaded PDFs and just want to generate answers,
or when the main gemini_filesearch.py script crashes during answer generation.

Usage:
    uv run python gemini_answer_generation.py --store_name "fileSearchStores/xxx"
    # OR
    uv run python gemini_answer_generation.py --store_display_name "pdf-store"

Environment Variables:
    GEMINI_API_KEY: Required
    dataset_queries: Path to questions CSV (default: query_dataset_with_qa.csv)
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
from typing import Any, Dict, List, Optional

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types


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


def extract_grounding(resp: Any) -> Dict[str, Any]:
    """
    Best-effort extraction of File Search grounding/citations.
    Returns a small JSON-serializable dict.
    """
    out: Dict[str, Any] = {"has_grounding": False, "citations": []}

    try:
        # Most common: resp.candidates[0].grounding_metadata
        if not hasattr(resp, "candidates") or not resp.candidates:
            return out
        
        cand0 = resp.candidates[0]
        gm = getattr(cand0, "grounding_metadata", None)
        if gm is None:
            return out

        gm_dump = safe_model_dump(gm)
        out["has_grounding"] = True
        out["grounding_metadata"] = gm_dump  # keep full grounding block (usually not enormous)

        # Optional: try to summarize into simple citations if present
        # Different SDK versions may store this differently, so keep it defensive.
        if isinstance(gm_dump, dict):
            # Common pattern: "groundingChunks" / "grounding_chunks" or "groundingSupports"
            chunks = gm_dump.get("grounding_chunks") or gm_dump.get("groundingChunks") or []
            supports = gm_dump.get("grounding_supports") or gm_dump.get("groundingSupports") or []

            # Pull file names/uris when available
            for ch in chunks:
                # attempt to extract something human-friendly
                if isinstance(ch, dict):
                    src = ch.get("retrieved_context") or ch.get("retrievedContext") or ch.get("web") or ch.get("file") or {}
                    out["citations"].append(src)

            # supports sometimes link answer spans to chunk indices
            if supports:
                out["supports"] = supports

        return out
    except Exception:
        return out


def ask_with_file_search(
    client: genai.Client,
    model: str,
    store_name: str,
    question: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Queries Gemini using File Search tool bound to the given store.
    Returns a dict with 'answer', 'grounding', and 'raw_response' keys.
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

            # Extract grounding/citations
            grounding = extract_grounding(resp)

            # Full raw response dump for debugging
            resp_dump = safe_model_dump(resp)

            return {
                "question": question,
                "answer": answer_text,
                "grounding": grounding,
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
                    break

    return {
        "question": question,
        "answer": None,
        "grounding": {"has_grounding": False, "citations": []},
        "raw_response": {},
        "error": "Max retries exceeded",
    }


def find_store_by_display_name(client: genai.Client, display_name: str) -> Optional[str]:
    """Find a store by its display name. Returns the store name (resource name) or None."""
    try:
        # List all stores and find one matching the display name
        stores = client.file_search_stores.list()
        for store in stores:
            if hasattr(store, "display_name") and store.display_name == display_name:
                return store.name
            # Also check config if available
            if hasattr(store, "config") and isinstance(store.config, dict):
                if store.config.get("display_name") == display_name:
                    return store.name
    except Exception as e:
        print(f"Warning: Could not list stores: {e}", file=sys.stderr)
    return None


def read_questions(csv_path: Path, column: str = "question") -> List[Dict[str, str]]:
    """Read questions from CSV."""
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate answers using an existing Gemini File Search store"
    )
    parser.add_argument("--store_name", default=None, help="Store resource name (e.g., fileSearchStores/xxx)")
    parser.add_argument("--store_display_name", default=None, help="Store display name (will search for matching store)")
    parser.add_argument("--questions_csv", default=None, help="CSV file with questions (default: from dataset_queries env var)")
    parser.add_argument("--question_column", default="question", help="CSV column name")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--answer_log_csv", default=None, help="Output CSV log file")
    args = parser.parse_args()

    # Auth
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY environment variable is required.")
    client = genai.Client(api_key=api_key)

    # Get store name
    store_name = args.store_name
    if not store_name and args.store_display_name:
        print(f"Searching for store with display_name: {args.store_display_name}...")
        store_name = find_store_by_display_name(client, args.store_display_name)
        if not store_name:
            raise SystemExit(f"ERROR: Could not find store with display_name: {args.store_display_name}\n"
                           f"Use --store_name with the full resource name (fileSearchStores/xxx)")
    
    if not store_name:
        raise SystemExit("ERROR: --store_name or --store_display_name required")
    
    print(f"Using store: {store_name}")

    # Get questions CSV
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
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        model_safe = args.model.replace("/", "_").replace("\\", "_")
        dataset_name = csv_path.stem
        answer_log_path = logs_dir / f"answers_gemini_{model_safe}_{dataset_name}.csv"

    # Read questions
    question_rows = read_questions(csv_path, column=args.question_column)
    print(f"Loaded {len(question_rows)} question(s) from CSV: {csv_path}")

    # Check for existing answers (preserve citations if available)
    existing_answers = {}
    existing_contexts = {}  # Store context_used (citations) for cached answers
    if answer_log_path.exists():
        try:
            with answer_log_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = (row.get("question") or "").strip()
                    ans = (row.get("generated_answer") or "").strip()
                    ctx = (row.get("context_used") or "").strip()
                    if q and ans:
                        existing_answers[q] = ans
                        if ctx:
                            existing_contexts[q] = ctx
            if existing_answers:
                print(f"Loaded {len(existing_answers)} existing answers from {answer_log_path}")
        except Exception as e:
            print(f"Warning: Could not load existing answers: {e}", file=sys.stderr)

    # Generate answers
    answer_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Raw response JSONL file for debugging
    raw_jsonl_path = answer_log_path.with_suffix(".raw.jsonl")
    
    fieldnames = ["question", "generated_answer", "context_used"]
    rows_to_write = []

    print(f"\nGenerating answers for {len(question_rows)} questions...")
    print(f"Model: {args.model}")
    print(f"Store: {store_name}")
    print(f"Answer log: {answer_log_path}")
    print(f"Raw response log: {raw_jsonl_path}\n")

    for idx, q_row in enumerate(question_rows, start=1):
        question = q_row["question"]
        print(f"[Q {idx}/{len(question_rows)}] {question[:90]}{'...' if len(question) > 90 else ''}")

        if question in existing_answers:
            generated_answer = existing_answers[question]
            # Preserve existing citations if available
            context_used = existing_contexts.get(question, "")
            print(f"  -> Using cached answer")
        else:
            record = ask_with_file_search(client, args.model, store_name, question)
            generated_answer = record.get("answer") or ""
            
            # Extract grounding/citations as JSON for context_used column
            grounding = record.get("grounding") or {}
            context_used = json.dumps(grounding, ensure_ascii=False)
            
            if not generated_answer and "error" in record:
                print(f"  -> Error: {record.get('error', 'Unknown error')}", file=sys.stderr)
            
            # Write full raw response to JSONL for debugging
            with raw_jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        rows_to_write.append({
            "question": question,
            "generated_answer": generated_answer,
            "context_used": context_used,
        })

    # Write CSV
    with answer_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"\nDone. Wrote answer log: {answer_log_path}")
    print(f"  Total questions: {len(question_rows)}")
    print(f"  Generated answers: {len([r for r in rows_to_write if r['generated_answer']])}")
    print(f"  Raw response log: {raw_jsonl_path}")
    print(f"\nNote: Citations/grounding metadata are in the 'context_used' column as JSON.")
    print(f"      Full raw responses (for debugging) are in: {raw_jsonl_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

