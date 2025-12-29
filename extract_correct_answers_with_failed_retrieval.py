#!/usr/bin/env python3
"""
Extract queries where answer is correct but retrieval judge failed.

Extracts the 14 queries where:
- Answer judge_score = 1 (correct answer)
- Retrieval judge_score = 0 (retrieval judge failed)

Outputs CSV with:
- question
- gt_context (ground truth context)
- gt_answer (ground truth answer)
- gemini_retrieved_context (extracted from context_used JSON)
- gemini_answer (generated answer)
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Increase CSV field size limit
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize // 2)


def read_csv_dict(path: Path) -> List[Dict[str, str]]:
    """Read CSV file and return list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_int(value: str, default: int = 0) -> int:
    """Safely parse integer from CSV value."""
    try:
        return int(value.strip()) if value.strip() else default
    except (ValueError, AttributeError):
        return default


def extract_retrieved_context_text(context_used: str) -> str:
    """
    Extract retrieved context text from context_used JSON.
    Returns concatenated text from all grounding chunks.
    """
    if not context_used or not context_used.strip():
        return ""

    try:
        obj = json.loads(context_used)
    except Exception:
        return ""

    # Navigate to grounding_chunks
    grounding_metadata = obj.get("grounding_metadata") or {}
    if not grounding_metadata:
        # Try alternative structure
        if "grounding" in obj and isinstance(obj["grounding"], dict):
            grounding_metadata = obj["grounding"]
        elif "grounding_chunks" in obj:
            grounding_metadata = {"grounding_chunks": obj["grounding_chunks"]}

    chunks = grounding_metadata.get("grounding_chunks") or []
    if not chunks:
        return ""

    # Extract text from each chunk
    texts = []
    for chunk in chunks:
        retrieved_context = chunk.get("retrieved_context") or {}
        text = retrieved_context.get("text") or ""
        if text:
            texts.append(text.strip())

    return "\n\n---\n\n".join(texts)


def main():
    retrieval_csv = Path("logs/gemini_filesearch_retrieval_eval_judge.csv")
    answer_csv = Path("logs/answer_eval_log_gemini_filesearch_api.csv")
    gt_csv = Path("query_dataset_with_qa.csv")
    answers_gemini_csv = Path("logs/answers_gemini_gemini-2.5-flash_query_dataset_with_qa.csv")
    output_csv = Path("logs/correct_answers_failed_retrieval.csv")

    # Check files exist
    for f in [retrieval_csv, answer_csv, gt_csv, answers_gemini_csv]:
        if not f.exists():
            raise SystemExit(f"ERROR: {f} not found")

    print("Reading CSV files...")
    retrieval_rows = read_csv_dict(retrieval_csv)
    answer_rows = read_csv_dict(answer_csv)
    gt_rows = read_csv_dict(gt_csv)
    answers_gemini_rows = read_csv_dict(answers_gemini_csv)

    # Build lookups
    print("Building lookups...")
    
    # GT: question -> (context, answer)
    gt_by_q: Dict[str, tuple] = {}
    for r in gt_rows:
        q = (r.get("question") or "").strip()
        if q:
            gt_by_q[q] = (r.get("context", ""), r.get("answer", ""))

    # Answer eval: question -> answer_judge_score
    answer_judge_by_q: Dict[str, int] = {}
    for r in answer_rows:
        q = (r.get("question") or "").strip()
        if q:
            answer_judge_by_q[q] = parse_int(r.get("judge_score", "0"))

    # Gemini answers: question -> (generated_answer, context_used)
    gemini_by_q: Dict[str, tuple] = {}
    for r in answers_gemini_rows:
        q = (r.get("question") or "").strip()
        if q:
            gemini_by_q[q] = (
                r.get("generated_answer", ""),
                r.get("context_used", "")
            )

    # Find queries where answer_judge=1 and retrieval_judge=0
    print("Finding matching queries...")
    extracted = []

    for r in retrieval_rows:
        q = (r.get("question") or "").strip()
        if not q:
            continue

        retrieval_judge = parse_int(r.get("judge_score", "0"))
        answer_judge = answer_judge_by_q.get(q, 0)

        # We want: answer_judge=1 AND retrieval_judge=0
        if answer_judge == 1 and retrieval_judge == 0:
            gt_context, gt_answer = gt_by_q.get(q, ("", ""))
            gemini_answer, context_used = gemini_by_q.get(q, ("", ""))
            retrieved_context = extract_retrieved_context_text(context_used)

            extracted.append({
                "question": q,
                "gt_context": gt_context,
                "gt_answer": gt_answer,
                "gemini_retrieved_context": retrieved_context,
                "gemini_answer": gemini_answer,
            })

    # Write output
    print(f"Found {len(extracted)} queries matching criteria")
    print(f"Writing to {output_csv}...")

    if not extracted:
        print("WARNING: No queries found matching the criteria!")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["question", "gt_context", "gt_answer", "gemini_retrieved_context", "gemini_answer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted)

    print(f"Done! Wrote {len(extracted)} rows to {output_csv}")


if __name__ == "__main__":
    main()

