#!/usr/bin/env python3
"""
Meta-analysis script comparing retrieval quality vs answer quality.

Reads:
1. logs/gemini_filesearch_retrieval_eval_judge.csv - retrieval evaluation
2. logs/answer_eval_log_gemini_filesearch_api.csv - answer evaluation

Creates a breakdown table showing:
- For correct answers (answer judge_score=1): retrieval performance breakdown
- For incorrect answers (answer judge_score=0): retrieval performance breakdown

Usage:
    python analyze_retrieval_vs_answer_quality.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Increase CSV field size limit for large fields
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


def main():
    retrieval_csv = Path("logs/gemini_filesearch_retrieval_eval_judge.csv")
    answer_csv = Path("logs/answer_eval_log_gemini_filesearch_api.csv")

    if not retrieval_csv.exists():
        raise SystemExit(f"ERROR: {retrieval_csv} not found")
    if not answer_csv.exists():
        raise SystemExit(f"ERROR: {answer_csv} not found")

    # Read both CSVs
    retrieval_rows = read_csv_dict(retrieval_csv)
    answer_rows = read_csv_dict(answer_csv)

    # Build lookup: question -> (doc_hit, retrieval_judge_score)
    retrieval_by_q: Dict[str, Tuple[int, int]] = {}
    for r in retrieval_rows:
        q = (r.get("question") or "").strip()
        if q:
            doc_hit = parse_int(r.get("doc_hit", "0"))
            retrieval_judge = parse_int(r.get("judge_score", "0"))
            retrieval_by_q[q] = (doc_hit, retrieval_judge)

    # Build breakdown counters
    # Structure: answer_judge -> (doc_hit, retrieval_judge) -> count
    breakdown: Dict[int, Dict[Tuple[int, int], int]] = {
        0: defaultdict(int),  # Incorrect answers
        1: defaultdict(int),  # Correct answers
    }

    total_matched = 0
    total_unmatched = 0

    # Process answer evaluation
    for a in answer_rows:
        q = (a.get("question") or "").strip()
        if not q:
            continue

        answer_judge = parse_int(a.get("judge_score", "0"))
        answer_judge = 1 if answer_judge == 1 else 0  # Normalize to 0/1

        if q in retrieval_by_q:
            doc_hit, retrieval_judge = retrieval_by_q[q]
            breakdown[answer_judge][(doc_hit, retrieval_judge)] += 1
            total_matched += 1
        else:
            total_unmatched += 1

    # Print results
    print("=" * 80)
    print("RETRIEVAL QUALITY vs ANSWER QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nTotal matched questions: {total_matched}")
    if total_unmatched > 0:
        print(f"Total unmatched questions: {total_unmatched} (not in retrieval eval)")
    print()

    # Helper to format cell
    def fmt_cell(count: int, total: int) -> str:
        pct = (100.0 * count / total) if total > 0 else 0.0
        return f"{count:4d} ({pct:5.1f}%)"

    # Print breakdown for CORRECT answers
    correct_total = sum(breakdown[1].values())
    print("=" * 80)
    print("CORRECT ANSWERS (Answer Judge Score = 1)")
    print("=" * 80)
    print(f"Total correct answers: {correct_total}")
    print()
    print("Retrieval Performance Breakdown:")
    print("-" * 80)
    print(f"{'Doc Hit':<10} {'Retrieval Judge':<18} {'Count':<20} {'% of Correct'}")
    print("-" * 80)

    if correct_total > 0:
        # Order: (0,0), (0,1), (1,0), (1,1)
        for doc_hit in [0, 1]:
            for retrieval_judge in [0, 1]:
                count = breakdown[1][(doc_hit, retrieval_judge)]
                doc_str = "Yes" if doc_hit == 1 else "No"
                judge_str = "Pass" if retrieval_judge == 1 else "Fail"
                print(f"{doc_str:<10} {judge_str:<18} {fmt_cell(count, correct_total):<20}")
    else:
        print("No correct answers found")
    print()

    # Print breakdown for INCORRECT answers
    incorrect_total = sum(breakdown[0].values())
    print("=" * 80)
    print("INCORRECT ANSWERS (Answer Judge Score = 0)")
    print("=" * 80)
    print(f"Total incorrect answers: {incorrect_total}")
    print()
    print("Retrieval Performance Breakdown:")
    print("-" * 80)
    print(f"{'Doc Hit':<10} {'Retrieval Judge':<18} {'Count':<20} {'% of Incorrect'}")
    print("-" * 80)

    if incorrect_total > 0:
        for doc_hit in [0, 1]:
            for retrieval_judge in [0, 1]:
                count = breakdown[0][(doc_hit, retrieval_judge)]
                doc_str = "Yes" if doc_hit == 1 else "No"
                judge_str = "Pass" if retrieval_judge == 1 else "Fail"
                print(f"{doc_str:<10} {judge_str:<18} {fmt_cell(count, incorrect_total):<20}")
    else:
        print("No incorrect answers found")
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    # For correct answers
    if correct_total > 0:
        correct_both_1 = breakdown[1][(1, 1)]
        correct_doc_0 = breakdown[1][(0, 0)] + breakdown[1][(0, 1)]
        correct_judge_0 = breakdown[1][(0, 0)] + breakdown[1][(1, 0)]
        correct_both_0 = breakdown[1][(0, 0)]

        print("CORRECT ANSWERS:")
        print(f"  - Both retrieval checks pass (doc=1, judge=1): {correct_both_1} ({100.0*correct_both_1/correct_total:.1f}%)")
        print(f"  - Doc hit failed (doc=0): {correct_doc_0} ({100.0*correct_doc_0/correct_total:.1f}%)")
        print(f"  - Retrieval judge failed (judge=0): {correct_judge_0} ({100.0*correct_judge_0/correct_total:.1f}%)")
        print(f"  - Both retrieval checks failed (doc=0, judge=0): {correct_both_0} ({100.0*correct_both_0/correct_total:.1f}%)")
        print()

    # For incorrect answers
    if incorrect_total > 0:
        incorrect_both_1 = breakdown[0][(1, 1)]
        incorrect_doc_0 = breakdown[0][(0, 0)] + breakdown[0][(0, 1)]
        incorrect_judge_0 = breakdown[0][(0, 0)] + breakdown[0][(1, 0)]
        incorrect_both_0 = breakdown[0][(0, 0)]

        print("INCORRECT ANSWERS:")
        print(f"  - Both retrieval checks pass (doc=1, judge=1): {incorrect_both_1} ({100.0*incorrect_both_1/incorrect_total:.1f}%)")
        print(f"  - Doc hit failed (doc=0): {incorrect_doc_0} ({100.0*incorrect_doc_0/incorrect_total:.1f}%)")
        print(f"  - Retrieval judge failed (judge=0): {incorrect_judge_0} ({100.0*incorrect_judge_0/incorrect_total:.1f}%)")
        print(f"  - Both retrieval checks failed (doc=0, judge=0): {incorrect_both_0} ({100.0*incorrect_both_0/incorrect_total:.1f}%)")
        print()

    # Cross-analysis: What's the correlation?
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    if correct_total > 0 and incorrect_total > 0:
        correct_with_good_retrieval = breakdown[1][(1, 1)]
        incorrect_with_good_retrieval = breakdown[0][(1, 1)]
        total_with_good_retrieval = correct_with_good_retrieval + incorrect_with_good_retrieval

        if total_with_good_retrieval > 0:
            pct_correct_given_good_retrieval = 100.0 * correct_with_good_retrieval / total_with_good_retrieval
            print(f"When retrieval is perfect (doc=1, judge=1):")
            print(f"  - {correct_with_good_retrieval}/{total_with_good_retrieval} answers are correct ({pct_correct_given_good_retrieval:.1f}%)")

        correct_with_bad_retrieval = breakdown[1][(0, 0)]
        incorrect_with_bad_retrieval = breakdown[0][(0, 0)]
        total_with_bad_retrieval = correct_with_bad_retrieval + incorrect_with_bad_retrieval

        if total_with_bad_retrieval > 0:
            pct_correct_given_bad_retrieval = 100.0 * correct_with_bad_retrieval / total_with_bad_retrieval
            print(f"When retrieval fails completely (doc=0, judge=0):")
            print(f"  - {correct_with_bad_retrieval}/{total_with_bad_retrieval} answers are still correct ({pct_correct_given_bad_retrieval:.1f}%)")
    print()


if __name__ == "__main__":
    main()

