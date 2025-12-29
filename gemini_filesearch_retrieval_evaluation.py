#!/usr/bin/env python3
"""
Evaluate Gemini File Search retrieval hit rate using grounding metadata,
with OpenAI GPT-4o as an external judge.

The judge answers:
"Do the retrieved contexts fully contain the evidence required to answer
the question, as defined by the ground-truth context and answer?"

STRICT SCORING:
- score = 1 only if ALL required evidence is present
- score = 0 if anything is missing, partial, or from a different product/model

Inputs:
1) Ground-truth CSV (minimum columns):
   - question
   - answer
   - context
   - source_path
   - gt_section_id

2) Answers CSV (minimum columns):
   - question
   - context_used   (JSON with grounding_metadata.grounding_chunks)

Environment:
- OPENAI_API_KEY (required)
- JUDGE_MODEL (optional, default: gpt-4o)

Output:
- CSV with doc_hit, judge_score, strict_hit, reasoning

Usage:
python gemini_filesearch_retrieval_evaluation.py \
  --groundtruth_csv query_dataset_with_qa.csv \
  --answers_csv logs/answers_gemini_*.csv \
  --out_csv logs/gemini_filesearch_retrieval_eval_judge.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI


# ============================================================
# Normalization + doc inference
# ============================================================

def normalize(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def infer_expected_doc_candidates(gt_row: Dict[str, str]) -> List[str]:
    """
    Infer expected document identity from:
    - source_path stem
    - 'Document_title:' line inside ground-truth context
    """
    candidates: List[str] = []

    source_path = (gt_row.get("source_path") or "").strip()
    if source_path:
        stem = Path(source_path).stem
        candidates.extend([stem, stem + ".pdf"])

    ctx = gt_row.get("context") or ""
    m = re.search(r"Document_title:\s*([^\n\r]+)", ctx)
    if m:
        doc = m.group(1).strip()
        candidates.extend([doc, doc + ".pdf"])

    seen = set()
    out = []
    for c in candidates:
        key = c.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out

def title_matches(retrieved_title: str, expected_candidates: List[str]) -> bool:
    rt = (retrieved_title or "").lower()
    for e in expected_candidates:
        ee = e.lower()
        if rt == ee or rt in ee or ee in rt:
            return True
    return False


# ============================================================
# Grounding metadata parsing
# ============================================================

def parse_grounding(context_used: str) -> Dict[str, Any]:
    """
    Extract grounding_metadata from stored JSON.
    """
    if not context_used:
        return {}
    try:
        obj = json.loads(context_used)
    except Exception:
        return {}

    if isinstance(obj, dict):
        if "grounding_metadata" in obj:
            return obj
        if "grounding" in obj and isinstance(obj["grounding"], dict):
            return obj["grounding"]
        if "grounding_chunks" in obj:
            return {"grounding_metadata": obj}

    return {}

def extract_retrieved_contexts(grounding_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    gm = grounding_obj.get("grounding_metadata") or {}
    chunks = gm.get("grounding_chunks") or []
    out: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks):
        rc = ch.get("retrieved_context") or {}
        out.append({
            "idx": idx,
            "title": rc.get("title"),
            "text": rc.get("text") or "",
            "uri": rc.get("uri"),
        })
    return out


# ============================================================
# Retrieved context cleanup for LLM judge
# ============================================================

def clean_pdf_text(s: str) -> str:
    s = normalize(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def format_retrieved_contexts(chunks: List[Dict[str, Any]], max_chars: int = 1800) -> str:
    blocks: List[str] = []
    for c in chunks:
        title = c.get("title") or "UNKNOWN_DOC"
        text = clean_pdf_text(c.get("text") or "")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n...[TRUNCATED]..."
        blocks.append(f"[CHUNK {c['idx']}] {title}\n{text}")
    return "\n\n---\n\n".join(blocks)


# ============================================================
# OpenAI Judge
# ============================================================

JUDGE_SYSTEM = """You are a strict retrieval evaluator for a RAG system.

Your task:
Determine whether the RETRIEVED CONTEXTS fully contain ALL information required
to answer the question correctly, as defined by the ground-truth answer and
ground-truth context.

Rules:
- score = 1 ONLY if all required evidence is present.
- score = 0 if anything is missing, partial, ambiguous, or refers to a different product/model.
- Be strict about product names, model numbers, part numbers, variants, and quantities.
- Similar-looking content is NOT sufficient.
- Use ONLY the retrieved contexts as evidence.
- Evidence may be distributed across multiple retrieved chunks.

VERY IMPORTANT:
The ground-truth context may contain additional procedural or descriptive details that are not required to answer the question.
Only evaluate whether the retrieved contexts contain the minimum sufficient evidence needed to produce the ground-truth answer.

Return JSON ONLY:
{
  "score": 0 or 1,
  "reasoning": "short, concrete explanation",
  "missing": "what evidence is missing if score=0, else empty string"
}
"""

def judge_retrieval_openai(
    client: OpenAI,
    model: str,
    question: str,
    gt_answer: str,
    gt_context: str,
    retrieved_contexts: str,
) -> Dict[str, Any]:

    user_prompt = f"""
QUESTION:
{question}

GROUND-TRUTH ANSWER:
{gt_answer}

GROUND-TRUTH CONTEXT:
{gt_context}

RETRIEVED CONTEXTS:
{retrieved_contexts}

Evaluate retrieval completeness.
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "score": 0,
            "reasoning": "Judge returned invalid JSON",
            "missing": "judge_parse_error",
        }


# ============================================================
# CSV helpers
# ============================================================

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth_csv", required=True)
    ap.add_argument("--answers_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is required.")

    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")
    judge_client = OpenAI(api_key=openai_key)

    gt_rows = read_csv(Path(args.groundtruth_csv))
    ans_rows = read_csv(Path(args.answers_csv))

    ans_by_q = {r["question"].strip(): r for r in ans_rows if r.get("question")}

    report: List[Dict[str, Any]] = []

    total = doc_hits = judge_hits = strict_hits = missing_grounding = 0

    # Filter out empty questions for accurate total count
    valid_gt_rows = [r for r in gt_rows if (r.get("question") or "").strip()]
    total_questions = len(valid_gt_rows)
    
    print(f"Processing {total_questions} questions...", file=sys.stderr)
    print(f"Judge model: {judge_model}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    for idx, gt in enumerate(valid_gt_rows, start=1):
        q = (gt.get("question") or "").strip()
        total += 1

        gt_answer = gt.get("answer", "")
        gt_context = gt.get("context", "")
        gt_section_id = gt.get("gt_section_id", "")

        expected_docs = infer_expected_doc_candidates(gt)

        # Progress indicator
        q_short = q[:70] + "..." if len(q) > 70 else q
        print(f"[{idx}/{total_questions}] {q_short}", file=sys.stderr, end="", flush=True)

        ans = ans_by_q.get(q, {})
        grounding = parse_grounding(ans.get("context_used", ""))

        if not grounding:
            missing_grounding += 1
            print(" -> MISSING GROUNDING", file=sys.stderr)
            report.append({
                "question": q,
                "gt_section_id": gt_section_id,
                "expected_doc_candidates": " | ".join(expected_docs),
                "doc_hit": 0,
                "judge_score": 0,
                "strict_hit": 0,
                "judge_reasoning": "no grounding metadata",
                "judge_missing": "no_grounding",
            })
            continue

        chunks = extract_retrieved_contexts(grounding)
        num_chunks = len(chunks)

        doc_hit = 1 if any(title_matches(c["title"], expected_docs) for c in chunks) else 0
        doc_hits += doc_hit

        print(f" -> {num_chunks} chunks, doc_hit={doc_hit}", file=sys.stderr, end="", flush=True)

        retrieved_formatted = format_retrieved_contexts(chunks)

        print(" [judging...]", file=sys.stderr, end="", flush=True)
        judge = judge_retrieval_openai(
            client=judge_client,
            model=judge_model,
            question=q,
            gt_answer=gt_answer,
            gt_context=gt_context,
            retrieved_contexts=retrieved_formatted,
        )

        score = 1 if int(judge.get("score", 0)) == 1 else 0
        judge_hits += score

        strict_hit = 1 if (doc_hit and score) else 0
        strict_hits += strict_hit

        print(f" -> judge={score}, strict={strict_hit} | running: doc={doc_hits}/{total} judge={judge_hits}/{total} strict={strict_hits}/{total}", file=sys.stderr)

        report.append({
            "question": q,
            "gt_section_id": gt_section_id,
            "expected_doc_candidates": " | ".join(expected_docs),
            "doc_hit": doc_hit,
            "judge_score": score,
            "strict_hit": strict_hit,
            "judge_reasoning": judge.get("reasoning", ""),
            "judge_missing": judge.get("missing", ""),
        })

    print("\n" + "=" * 80, file=sys.stderr)
    print("Writing results...", file=sys.stderr)
    write_csv(Path(args.out_csv), report)

    def pct(x: int) -> str:
        return f"{(100.0 * x / total):.1f}%" if total else "0.0%"

    print("\n=== Gemini File Search Retrieval Evaluation (OpenAI Judge) ===")
    print(f"Total questions:        {total}")
    print(f"Missing grounding:      {missing_grounding} ({pct(missing_grounding)})")
    print(f"Doc hit rate:           {doc_hits} ({pct(doc_hits)})")
    print(f"Judge section hit rate: {judge_hits} ({pct(judge_hits)})")
    print(f"Strict hit rate:        {strict_hits} ({pct(strict_hits)})")
    print(f"Judge model:            {judge_model}")
    print(f"Report written to:      {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
