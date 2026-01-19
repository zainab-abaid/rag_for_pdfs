#!/usr/bin/env python3
"""
Answer Evaluation Script

This script:
1. Reads the retrieval log CSV (from `retrieve_with_diff_rerankers.py`)
2. Loads generated answers from the answer log CSV (from `generate_answers_diff_rerankers.py`)
3. Compares generated answers against ground truth using an LLM judge (e.g., `gpt-4o` or Groq models)
4. Outputs a simplified evaluation log with only essential columns:
   - `question`: The query/question being evaluated
   - `gt_answer`: Ground truth answer
   - `llm_answer`: Generated answer from the model
   - `retrieved_context`: Top-K context used for generating the answer
   - `gt_in_topK`: Whether the ground truth was in the top-K retrieved context
   - `judge_score`: Evaluation score (0 or 1)
   - `judge_reasoning`: Explanation for the score

Usage:
    uv run python src/rerankingexperiments/evaluate_answers_diff_rerankers.py

Environment Variables (REQUIRED):
    RETRIEVAL_LOG_CSV: Path to the retrieval log CSV (e.g., `logs/retrieval_log_TOPK5_none.csv`)
    ANSWER_LOG_CSV: Path to the answer log CSV (e.g., `logs/answers_gemini_query_dataset_with_qa_none.csv`)
    ANSWER_EVAL_OUTPUT_CSV: Output path for evaluation results (e.g., `logs/answers_eval_gemini_query_dataset_with_qa_none.csv`)
    OPENAI_API_KEY: Required for OpenAI API calls (when `USE_GROQ=false`)
    GROQ_API_KEY: Required for Groq API calls (when `USE_GROQ=true`)

Environment Variables (OPTIONAL):
    USE_GROQ: Set to `true` to use Groq API instead of OpenAI (default: `false`)
    JUDGE_MODEL: Model to use for judging when `USE_GROQ=false` (default: `gpt-4o`)
    GROQ_MODEL: Model to use for judging when `USE_GROQ=true` (required when `USE_GROQ=true`)
"""

import os
import json
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

# CSV field size limit
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize // 2)

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Groq client
try:
    from groq import Groq
except ImportError:
    Groq = None


def get_openai_client() -> OpenAI:
    """Get OpenAI client from environment."""
    if OpenAI is None:
        raise SystemExit("openai package not installed. Run: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def get_groq_client() -> Groq:
    """Get Groq client from environment."""
    if Groq is None:
        raise SystemExit("groq package not installed. Run: pip install groq")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set in environment.")
    return Groq(api_key=api_key)


def judge_answer(
    query: str,
    ground_truth_answer: str,
    generated_answer: str,
    client: OpenAI,
    model: str = "gpt-4o"
) -> Tuple[int, str]:
    """
    Use an LLM judge to evaluate if the generated answer accurately addresses "
    a question compared to a ground truth answer.
    Returns (score: 0 or 1, reasoning: str).
    
    The judge is lenient: it marks as accurate (1) unless there's an obvious mistake
    or something obviously wrong/missing.
    """
    if not ground_truth_answer or not generated_answer:
        return 0, "Missing ground truth or generated answer"
    
    system_prompt = (
        "You are a judge evaluating whether a generated answer accurately addresses "
        "a question compared to a ground truth answer.\n\n"
        "You should mark a semantically matching answer as ACCURATE (1) unless:\n"
        "- There is an obvious factual error or contradiction\n"
        "- Critical information is clearly missing that makes the answer incorrect\n"
        "- The answer is clearly wrong or misleading\n\n"
        "Minor differences in wording, additional context, or different phrasing "
        "that conveys the same meaning should still be marked as accurate.\n\n"
        "Return ONLY a JSON object with this exact format:\n"
        '{"score": 0 or 1, "reasoning": "brief explanation"}'
    )
    
    user_prompt = (
        f"Question: {query}\n\n"
        f"Ground Truth Answer: {ground_truth_answer}\n\n"
        f"Generated Answer: {generated_answer}\n\n"
        "Evaluate the generated answer. Return JSON only."
    )
    
    try:
        # o1 models don't support system messages, so combine into user message
        if model.startswith("o1"):
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined_prompt}]
            # o1 models don't support temperature or max_tokens parameters
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            except Exception as api_error:
                # Log the actual API error for debugging
                error_msg = str(api_error)
                print(f"ERROR: API call failed for {model}: {error_msg}", file=sys.stderr)
                return 0, f"API error: {error_msg[:150]}"
        else:
            try: 
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,  # Increased to allow for more detailed reasoning if needed
                )
            except Exception as e:
                print(f"WARNING: Judge API failed for question: {query[:60]}...")
                print(f"Reason: {str(e)[:200]}")
                return 0, "API error"
        
        if not response or not response.choices:
            return 0, "Empty response from API"
        
        content = (response.choices[0].message.content or "").strip()
        
        if not content:
            return 0, "Empty content in response"
        
        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            if content.startswith("```json"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            
            data = json.loads(content)
            score = int(data.get("score", 0))
            reasoning = str(data.get("reasoning", "")).strip()
            
            # Ensure score is 0 or 1
            score = 1 if score == 1 else 0
            return score, reasoning
        except json.JSONDecodeError as json_err:
            # Log the actual content for debugging
            print(f"WARNING: Failed to parse JSON from {model}. Content: {content[:200]}", file=sys.stderr)
            # Fallback: try to extract score from text
            if "score" in content.lower() and ("1" in content or "accurate" in content.lower()):
                return 1, "Parsed from text response"
            return 0, f"JSON parse error: {str(json_err)[:100]}"
    except Exception as e:
        # Log full error for debugging
        error_msg = str(e)
        print(f"ERROR: Unexpected error in judge_answer with {model}: {error_msg}", file=sys.stderr)
        return 0, f"Judge error: {error_msg[:150]}"


def get_env_path(up: str, low: str, default: str = "") -> str:
    """Get environment variable (case-insensitive)."""
    return os.getenv(up) or os.getenv(low) or default


def check_gt_in_topK(gt_section_id: str, retrieved_context_json: str) -> bool:
    """
    Check if ground truth section ID is in the top-K retrieved context.
    Returns True if GT is in top-K, False otherwise.
    """
    if not gt_section_id or not retrieved_context_json:
        return False
    
    try:
        blocks = json.loads(retrieved_context_json)
        if not isinstance(blocks, list):
            return False
        
        # Check if gt_section_id appears in any of the retrieved blocks
        for block in blocks:
            if isinstance(block, dict):
                section_id = block.get("section_id", "")
                if section_id == gt_section_id:
                    return True
        return False
    except Exception:
        return False


def load_answer_log(answer_log_path: Path) -> Dict[str, str]:
    """Load generated answers from log file."""
    if not answer_log_path.exists():
        raise SystemExit(f"Answer log not found: {answer_log_path}\n"
                        f"Run generate_answers.py first to create the answer log.")
    
    answers = {}
    try:
        with open(answer_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = (row.get("question") or "").strip()
                answer = (row.get("generated_answer") or "").strip()
                if question and answer:
                    answers[question] = answer
    except Exception as e:
        raise SystemExit(f"Error loading answer log: {e}")
    
    return answers

def resolve_answer_log_path(answer_model: str, reranker_mode: str) -> str:
    """Build filename: logs/answers_{ANSWER_MODEL}_query_dataset_with_qa_{RERANKER_MODE}.csv"""
    if "/" in answer_model:
        answer_model = answer_model.split("/")[0]
    safe_model = answer_model.replace("/", "_").replace(" ", "_")
    return f"logs/answers_{safe_model}_query_dataset_with_qa_{reranker_mode}.csv"

def resolve_answer_eval_log_path(answer_model: str, reranker_mode: str) -> str:
    """Build filename: logs/answers_eval_{ANSWER_MODEL}_query_dataset_with_qa_{RERANKER_MODE}.csv"""
    if "/" in answer_model:
        answer_model = answer_model.split("/")[0]
    safe_model = answer_model.replace("/", "_").replace(" ", "_")
    return f"logs/answers_eval_{safe_model}_query_dataset_with_qa_{reranker_mode}.csv"

def main():
    # -------------------- Setup filenames --------------------
    # Get RERANKER_MODE
    reranker_mode = get_env_path("RERANKER_MODE", "reranker_mode", "none")

    # Retrieval log (input)
    retrieval_log_base = get_env_path("RETRIEVAL_LOG_CSV", "retrieval_log_csv")
    if not retrieval_log_base:
        raise SystemExit("RETRIEVAL_LOG_CSV environment variable is required")
    base_path = Path(retrieval_log_base)
    retrieval_log_path = str(base_path.with_name(f"{base_path.stem}_{reranker_mode}{base_path.suffix}"))
    if not Path(retrieval_log_path).exists():
        raise SystemExit(f"Retrieval log file not found: {retrieval_log_path!r}")

    # Answer log (generated answers)
    answer_model = get_env_path("ANSWER_MODEL", "answer_model")
    answer_log_path = resolve_answer_log_path(answer_model, reranker_mode)

    if not Path(answer_log_path).exists():
        raise SystemExit(f"Answer log file not found: {answer_log_path!r}")

    # Evaluation output log
    eval_out = resolve_answer_eval_log_path(answer_model, reranker_mode)
    Path(eval_out).parent.mkdir(parents=True, exist_ok=True)

    # -------------------- API / Model Setup --------------------
    use_groq = (os.getenv("USE_GROQ") or "").strip().lower() == "true"
    if use_groq:
        client = get_groq_client()
        judge_model = os.getenv("GROQ_MODEL", "").strip()
        if not judge_model:
            raise SystemExit("GROQ_MODEL is required when USE_GROQ=true")
    else:
        client = get_openai_client()
        judge_model = os.getenv("JUDGE_MODEL", "gpt-4o").strip()

    # -------------------- Print info --------------------
    print("=" * 80)
    print("ANSWER EVALUATION")
    print("=" * 80)
    print(f"Retrieval log file  : {retrieval_log_path}")
    print(f"Answer log file     : {answer_log_path}")
    print(f"Evaluation output   : {eval_out}")
    print(f"Judge model         : {judge_model}")
    print(f"API provider        : {'Groq' if use_groq else 'OpenAI'}")
    print(f"Reranker mode       : {reranker_mode}")
    print("=" * 80)
    print()

    # -------------------- Load Logs --------------------
    # Retrieval log
    with open(retrieval_log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"question", "answer", "retrieved_context"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Log file missing required columns: {missing}")
        log_rows = list(reader)
    if not log_rows:
        raise SystemExit("Retrieval log file has no rows.")

    # Generated answers
    print(f"Loading answers from {answer_log_path}...")
    generated_answers = load_answer_log(Path(answer_log_path))
    print(f"Loaded {len(generated_answers)} answers with non-empty content from log\n")

    # -------------------- Prepare Output CSV --------------------
    out_fields = [
        "reranker_mode",
        "question",
        "gt_answer",
        "llm_answer",
        "retrieved_context",
        "gt_in_topK",
        "judge_score",
        "judge_reasoning"
    ]

    total = len(log_rows)
    accurate_count = 0
    missing_answers = 0

    print(f"Evaluating {total} queries...\n")
    print(f"Judge model: {judge_model}\n")


    # -------------------- Evaluation --------------------
    with open(eval_out, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for i, row in enumerate(log_rows, 1):
            question = (row.get("question") or "").strip()
            ground_truth_answer = (row.get("answer") or "").strip()
            generated_answer = generated_answers.get(question, "")
            retrieved_context = row.get("retrieved_context", "")
            gt_section_id = row.get("gt_section_id", "")

            # Check if GT is in top-K
            gt_in_topK = check_gt_in_topK(gt_section_id, retrieved_context)

            if not generated_answer:
                missing_answers += 1
                judge_score = 0
                judge_reasoning = "Answer not found in log"
            else:
                # Judge the answer
                judge_score, judge_reasoning = judge_answer(
                    question,
                    ground_truth_answer,
                    generated_answer,
                    client,
                    model=judge_model
                )
                if judge_score == 1:
                    accurate_count += 1

            # Write simplified output row
            out_row = {
                "reranker_mode": reranker_mode,
                "question": question,
                "gt_answer": ground_truth_answer,
                "llm_answer": generated_answer,
                "retrieved_context": retrieved_context,
                "gt_in_topK": "True" if gt_in_topK else "False",
                "judge_score": judge_score,
                "judge_reasoning": judge_reasoning
            }
            writer.writerow(out_row)

            # Progress print
            status = "✓" if judge_score == 1 else "✗"
            # Show error details if score is 0 and there's an error in reasoning
            error_indicator = ""
            if judge_score == 0 and "error" in judge_reasoning.lower():
                error_indicator = " [ERROR]"
            print(f"[{i}/{total}] {status} Score: {judge_score}{error_indicator} | {question[:60]}...")
            if error_indicator and i <= 5:  # Show first few errors in detail
                print(f"    → {judge_reasoning[:150]}")
    

    # -------------------- Summary --------------------
    print("\n" + "=" * 70)
    print("Answer Evaluation Summary")
    print("=" * 70)
    print(f"Total queries              : {total}")
    print(f"Accurate answers (1)       : {accurate_count} ({100.0 * accurate_count / total:.1f}%)")
    print(f"Inaccurate answers (0)     : {total - accurate_count} ({100.0 * (total - accurate_count) / total:.1f}%)")
    print(f"Missing answers            : {missing_answers}")
    print(f"Answer log file            : {answer_log_path}")
    print(f"Evaluation output file     : {eval_out}")



if __name__ == "__main__":
    main()

