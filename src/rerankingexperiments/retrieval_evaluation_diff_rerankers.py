#!/usr/bin/env python3
"""
Retrieval Evaluation Script

This script evaluates the retrieval performance of different reranking strategies by comparing the retrieved contexts against ground truth data.

Usage:
    uv run python src/rerankingexperiments/retrieval_evaluation_diff_rerankers.py

Environment Variables (REQUIRED):
    RERANKER_MODE: The reranking mode to evaluate (e.g., `entity`, `colbert`, `none`).
    DATASET_QUERIES: Path to the ground truth dataset CSV file (e.g., `data/questions_answers/query_dataset_with_qa.csv`).
    RETRIEVAL_LOG_CSV: Path to the retrieval log CSV file (e.g., `logs/retrieval_log_TOPK5_none.csv`).
    RETRIEVAL_EVAL_OUTPUT_CSV: Path to save the evaluation results (e.g., `logs/retrieval_eval_log_TOPK5_none.csv`).

    The script reads the ground truth dataset and the retrieval log, compares the retrieved contexts against the ground truth
    using both ID-based and text-based matching, and outputs evaluation metrics including hit rates and rank statistics.
"""

import os
import json
import statistics as stats
from pathlib import Path
from typing import Dict, List, Tuple

# CSV + large-field safety (should be tiny now, but keep it robust)
import sys, csv
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize // 2)

# Optional .env
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ---------- utils ----------
def get_env_path(up: str, low: str, default: str = "") -> str:
    return os.getenv(up) or os.getenv(low) or default

def with_suffix(path: str, suffix: str) -> str:
    """
    logs/file.csv + entity → logs/file_entity.csv
    """
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))

def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ----- robust text matching (fallback when IDs missing) -----
import re, unicodedata
from difflib import SequenceMatcher

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("™", "").replace("®", "")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_coverage(gt_norm: str, t_norm: str) -> float:
    gt_tokens = gt_norm.split()
    if not gt_tokens:
        return 0.0
    t_set = set(t_norm.split())
    hit = sum(1 for tok in gt_tokens if tok in t_set)
    return hit / len(gt_tokens)

def _partial_ratio(gt_norm: str, t_norm: str) -> float:
    if not gt_norm or not t_norm:
        return 0.0
    m = SequenceMatcher(None, gt_norm, t_norm).find_longest_match(0, len(gt_norm), 0, len(t_norm))
    return (m.size / max(1, len(gt_norm)))


def _is_match(gt: str, t: str) -> bool:
    gt_n = _normalize(gt)
    t_n = _normalize(t)
    if not gt_n or not t_n:
        return False
    if gt_n in t_n:
        return True
    gt_tokens = gt_n.split()
    if len(gt_tokens) >= 30 and _token_coverage(gt_n, t_n) >= 0.80:
        return True
    if _partial_ratio(gt_n, t_n) >= 0.85:
        return True
    return False

def _text_for_match(block: dict) -> str:
    # With the new minimal log, prefer text_head; fall back to full text if present.
    if not isinstance(block, dict):
        return ""
    return block.get("text", "") or block.get("text_head", "") or ""

def find_rank_in_blocks(gt: str, blocks_json: str) -> int:
    if not gt or not blocks_json:
        return -1
    try:
        blocks = json.loads(blocks_json)
        if not isinstance(blocks, list):
            return -1
    except Exception:
        return -1
    for i, b in enumerate(blocks):
        t = _text_for_match(b)
        if t and _is_match(gt, t):
            return i + 1
    return -1

# ----- ID-based helpers (preferred) -----
def extract_ids(blocks_json: str) -> List[str]:
    try:
        blocks = json.loads(blocks_json)
        if not isinstance(blocks, list):
            return []
        return [ (b.get("section_id") or "") for b in blocks if isinstance(b, dict) ]
    except Exception:
        return []

def find_rank_by_id(gt_id: str, blocks_json: str) -> int:
    if not gt_id:
        return -1
    ids = extract_ids(blocks_json)
    for i, sid in enumerate(ids):
        if sid and sid == gt_id:
            return i + 1
    return -1

# ---------- main ----------
def main():
    # ----- RERANKER MODE -----
    reranker_mode = os.getenv("RERANKER_MODE")
    if not reranker_mode:
        raise SystemExit(
            "ERROR: RERANKER_MODE is required "
            "(e.g., entity or colbert)"
        )
    reranker_mode = reranker_mode.lower()

    # REQUIRED: dataset_queries (ground truth dataset)
    dataset_path = get_env_path("DATASET_QUERIES", "dataset_queries")
    if not dataset_path:
        raise SystemExit("ERROR: DATASET_QUERIES or dataset_queries environment variable is required.\n"
                        "Set it to the path of your ground truth Q&A dataset CSV file.")
    if not Path(dataset_path).exists():
        raise SystemExit(f"ERROR: Dataset file not found: {dataset_path!r}")


    # REQUIRED: retrieval_log_csv (retrieval log from retrieve_and_stitch.py)
    base_log_path = get_env_path("RETRIEVAL_LOG_CSV", "retrieval_log_csv")
    if not base_log_path:
        raise SystemExit("ERROR: RETRIEVAL_LOG_CSV or retrieval_log_csv environment variable is required.\n"
                        "Set it to the path of your retrieval log CSV file (from retrieve_and_stitch.py).")
    log_path = with_suffix(base_log_path, reranker_mode)
    if not Path(log_path).exists():
        raise SystemExit(f"ERROR: Retrieval log file not found: {log_path!r}")


    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Dataset file (ground truth): {dataset_path}")
    print(f"Retrieval log file          : {log_path}")
    print("=" * 80)
    print()

    # REQUIRED: retrieval_eval_output_csv (retrieval evaluation output path)
    base_eval_out = get_env_path(
        "RETRIEVAL_EVAL_OUTPUT_CSV", "retrieval_eval_output_csv"
    )
    if not base_eval_out:
        raise SystemExit("ERROR: RETRIEVAL_EVAL_OUTPUT_CSV is required")

    eval_out = with_suffix(base_eval_out, reranker_mode)

    ds_rows = load_csv_rows(dataset_path)
    lg_rows = load_csv_rows(log_path)

    # sanity: required log columns
    need_cols = {"question", "retrieved_context", "retrieved_full_pretrim"}
    if lg_rows and not need_cols.issubset(lg_rows[0].keys()):
        raise SystemExit(f"Log file missing columns. Needed: {need_cols}")
    if not lg_rows:
        raise SystemExit("Log file has no rows.")

    # Build QUESTION -> (GT context, GT section id)
    q2gt: Dict[str, Tuple[str, str]] = {}
    for r in ds_rows:
        q = (r.get("question") or "").strip()
        q2gt[q] = (r.get("context", ""), r.get("gt_section_id", ""))

    total = 0
    hit_trimmed = 0
    hit_pretrim_only = 0
    missed = 0
    missing_gt = 0

    ranks_hit: List[int] = []

    # Prepare writer with same columns + 2 extra
    base_fields = list(lg_rows[0].keys())
    extra_fields = ["gt_match_status", "gt_rank_pretrim"]
    out_fields = base_fields + extra_fields

    with open(eval_out, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for r in lg_rows:
            q = (r.get("question") or "").strip()
            gt, gt_id = q2gt.get(q, ("", ""))

            status = "missed"
            rank = -1

            if not gt:
                missing_gt += 1
            else:
                total += 1
                pre = r.get("retrieved_full_pretrim", "")
                tri = r.get("retrieved_context", "")

                # Prefer ID-based rank
                if gt_id:
                    rank = find_rank_by_id(gt_id, pre)
                    if rank != -1:
                        # Found by ID in pre-trim; check trimmed by ID
                        if find_rank_by_id(gt_id, tri) != -1:
                            status = "trimmed"
                            hit_trimmed += 1
                        else:
                            status = "pretrim_only"
                            hit_pretrim_only += 1
                        ranks_hit.append(rank)
                    else:
                        # Fallback to robust text matching (using text_head now)
                        rank = find_rank_in_blocks(gt, pre)
                        if rank != -1:
                            if find_rank_in_blocks(gt, tri) != -1:
                                status = "trimmed"
                                hit_trimmed += 1
                            else:
                                status = "pretrim_only"
                                hit_pretrim_only += 1
                            ranks_hit.append(rank)
                        else:
                            missed += 1
                            status = "missed"
                else:
                    # No GT ID → text-only (uses text_head)
                    rank = find_rank_in_blocks(gt, pre)
                    if rank != -1:
                        if find_rank_in_blocks(gt, tri) != -1:
                            status = "trimmed"
                            hit_trimmed += 1
                        else:
                            status = "pretrim_only"
                            hit_pretrim_only += 1
                        ranks_hit.append(rank)
                    else:
                        missed += 1
                        status = "missed"

            out_row = dict(r)
            out_row["gt_match_status"] = status
            out_row["gt_rank_pretrim"] = rank
            writer.writerow(out_row)

    def pct(n):
        return 0.0 if total == 0 else round(100.0 * n / total, 2)

    print("\n=== Retrieval Evaluation Summary ===")
    print(f"Dataset file        : {dataset_path}")
    print(f"Reranker mode        : {reranker_mode}")
    print(f"Retrieval log file  : {log_path}")
    print(f"Eval log (augmented): {eval_out}\n")
    print(f"Evaluated queries   : {total}")
    print(f"Hit (trimmed)       : {hit_trimmed}  ({pct(hit_trimmed)}%)")
    print(f"Hit (pre-trim only) : {hit_pretrim_only}  ({pct(hit_pretrim_only)}%)")
    print(f"Missed              : {missed}  ({pct(missed)}%)")

    if ranks_hit:
        mean_rank = round(stats.mean(ranks_hit), 2)
        median_rank = stats.median(ranks_hit)
        best = min(ranks_hit)
        worst = max(ranks_hit)
        print("\nRank (pre-trim) for hits:")
        print(f"- mean   : {mean_rank}")
        print(f"- median : {median_rank}")
        print(f"- best   : {best}")
        print(f"- worst  : {worst}")

    if missing_gt:
        print(f"\nNote: Skipped {missing_gt} row(s) with missing GT context from dataset.")

if __name__ == "__main__":
    main()
