#!/usr/bin/env python3
"""
Answer Generation and LLM-Judged Evaluation

This script:
1. Reads the retrieval log CSV (from retrieve_and_stitch.py)
2. For each query, generates an answer using ALL retrieved contexts (not truncated)
3. Saves generated answers to a separate log file
4. Compares the generated answer against the ground truth answer using an LLM judge (gpt-4o)
5. Outputs evaluation results with judge scores (0/1) and statistics
"""

import os
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    raise SystemExit("openai package not installed. Run: pip install openai")


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def extract_context_text(retrieved_context_json: str) -> str:
    """
    Extract full context text from retrieved_context JSON (top-k sections).
    CSV now contains full text (not truncated), so no DB lookup needed.
    Returns concatenated text from all retrieved sections with section titles.
    """
    if not retrieved_context_json:
        return ""
    
    try:
        blocks = json.loads(retrieved_context_json)
        if not isinstance(blocks, list):
            return ""
        
        # Extract full text from each block, preserving section structure
        sections = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            
            # Get full text from CSV (now includes full text, not just text_head)
            full_text = block.get("text", "")
            section_path = block.get("section_path", [])
            doc_title = block.get("doc_title", "")
            
            if not full_text:
                continue
            
            # Build section header with path and title
            header_parts = []
            if doc_title:
                header_parts.append(f"Document: {doc_title}")
            if section_path:
                if isinstance(section_path, list):
                    header_parts.append(f"Section: {' > '.join(section_path)}")
                else:
                    header_parts.append(f"Section: {section_path}")
            
            section_text = full_text.strip()
            if header_parts:
                section_text = "\n".join(header_parts) + "\n\n" + section_text
            
            sections.append(section_text)
        
        return "\n\n" + "="*80 + "\n\n".join(sections)
    except Exception as e:
        print(f"Error parsing retrieved_context: {e}", file=sys.stderr)
        return ""


def generate_answer(query: str, context: str, client: OpenAI, model: str = "gpt-4o-mini") -> Optional[str]:
    """
    Generate an answer to the query using the provided context.
    The context includes full text with section titles, product names, and all details.
    """
    if not query or not context:
        return None
    
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context from "
        "Extreme Networks product documentation.\n\n"
        "IMPORTANT: When answering:\n"
        "- Pay close attention to section titles, product names, model numbers, and version information\n"
        "- Preserve specific technical details, numbers, and exact terminology from the context\n"
        "- Include relevant section paths and document titles if they help clarify the answer\n"
        "- Answer accurately using only information from the context\n"
        "- If the context doesn't contain enough information to answer the question, say so clearly\n"
        "- Be precise and include fine details that are relevant to the question"
    )
    
    user_prompt = (
        f"Context from documentation:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based on the context above. Include relevant details, product names, and section information:"
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer if answer else None
    except Exception as e:
        print(f"Error generating answer: {e}", file=sys.stderr)
        return None


def judge_answer(
    query: str,
    ground_truth_answer: str,
    generated_answer: str,
    client: OpenAI,
    model: str = "gpt-4o"
) -> Tuple[int, str]:
    """
    Use an LLM judge to evaluate if the generated answer is accurate.
    Returns (score: 0 or 1, reasoning: str).
    
    The judge is lenient: it marks as accurate (1) unless there's an obvious mistake
    or something obviously wrong/missing.
    """
    if not ground_truth_answer or not generated_answer:
        return 0, "Missing ground truth or generated answer"
    
    system_prompt = (
        "You are a lenient judge evaluating whether a generated answer accurately addresses "
        "a question compared to a ground truth answer.\n\n"
        "You should mark the answer as ACCURATE (1) unless:\n"
        "- There is an obvious factual error or contradiction\n"
        "- Critical information is clearly missing that makes the answer incorrect\n"
        "- The answer is clearly wrong or misleading\n\n"
        "Be lenient: minor differences in wording, additional context, or different phrasing "
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
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
        )
        content = (response.choices[0].message.content or "").strip()
        
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
        except json.JSONDecodeError:
            # Fallback: try to extract score from text
            if "score" in content.lower() and ("1" in content or "accurate" in content.lower()):
                return 1, "Parsed from text response"
            return 0, f"Failed to parse JSON: {content[:100]}"
    except Exception as e:
        return 0, f"Judge error: {str(e)[:100]}"


def get_env_path(up: str, low: str, default: str = "") -> str:
    """Get environment variable (case-insensitive)."""
    return os.getenv(up) or os.getenv(low) or default


def load_answer_log(answer_log_path: Path) -> Dict[str, str]:
    """Load previously generated answers from log file."""
    if not answer_log_path.exists():
        return {}
    
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
        print(f"Warning: Could not load answer log: {e}", file=sys.stderr)
    
    return answers


def save_answer_log(answer_log_path: Path, rows: List[Dict[str, str]]):
    """Save generated answers to log file."""
    if not rows:
        return
    
    fieldnames = ["question", "generated_answer", "context_used"]
    with open(answer_log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "question": row.get("question", ""),
                "generated_answer": row.get("generated_answer", ""),
                "context_used": row.get("context_used", ""),
            })


def main():
    # Get input/output paths
    log_path = get_env_path("OUTPUT_CSV", "output_csv")
    if not log_path or not Path(log_path).exists():
        raise SystemExit(f"Retrieval log not found: {log_path!r}\n"
                        f"Set OUTPUT_CSV or output_csv environment variable.")
    
    # Answer log path (for saving/loading generated answers)
    p = Path(log_path)
    answer_log_path = p.with_name(p.stem + "_answers.csv")
    
    # Evaluation output path
    eval_out = get_env_path("ANSWER_EVAL_OUTPUT_CSV", "answer_eval_output_csv")
    if not eval_out:
        eval_out = str(p.with_name(p.stem + "_answer_eval" + p.suffix))
    
    # Model settings
    answer_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    judge_model = "gpt-4o"  # Always use gpt-4o for judge
    
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        raise SystemExit("OPENAI_API_KEY not set in environment.")
    
    # Load retrieval log
    log_rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Use retrieved_context (top-k, as intended)
        required = {"question", "answer", "retrieved_context"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Log file missing required columns: {missing}\n"
                           f"Available columns: {list(reader.fieldnames or [])}")
        log_rows = list(reader)
    
    if not log_rows:
        raise SystemExit("Log file has no rows.")
    
    # Load existing answers if available
    existing_answers = load_answer_log(answer_log_path)
    if existing_answers:
        print(f"Loaded {len(existing_answers)} existing answers from {answer_log_path}")
    
    # Prepare output CSV
    base_fields = list(log_rows[0].keys())
    extra_fields = ["generated_answer", "judge_score", "judge_reasoning"]
    out_fields = base_fields + extra_fields
    
    total = len(log_rows)
    accurate_count = 0
    failed_generation = 0
    reused_answers = 0
    new_answers = []
    
    print(f"Processing {total} queries...")
    print(f"Answer generation model: {answer_model}")
    print(f"Judge model: {judge_model}")
    print(f"Using top-k retrieved contexts (retrieved_context)\n")
    
    # First pass: generate answers (or load from cache)
    for i, row in enumerate(log_rows, 1):
        question = (row.get("question") or "").strip()
        ground_truth_answer = (row.get("answer") or "").strip()
        # Use retrieved_full_pretrim (all retrieved contexts, not just top-k)
        retrieved_context_json = row.get("retrieved_full_pretrim", "")
        
        # Check if we already have an answer for this question
        if question in existing_answers:
            generated_answer = existing_answers[question]
            reused_answers += 1
            print(f"[{i}/{total}] Using cached answer | {question[:60]}...")
        else:
            # Extract context text (full text is now in CSV, no DB lookup needed)
            context_text = extract_context_text(retrieved_context_json)
            
            # Generate answer
            generated_answer = None
            if question and context_text:
                generated_answer = generate_answer(question, context_text, client, model=answer_model)
            
            if not generated_answer:
                failed_generation += 1
                print(f"[{i}/{total}] ✗ Failed to generate answer | {question[:60]}...")
            else:
                print(f"[{i}/{total}] ✓ Generated answer | {question[:60]}...")
                # Save for answer log
                new_answers.append({
                    "question": question,
                    "generated_answer": generated_answer,
                    "context_used": retrieved_context_json[:200] + "..." if len(retrieved_context_json) > 200 else retrieved_context_json
                })
        
        # Store generated answer in row for evaluation
        row["_generated_answer"] = generated_answer or ""
    
    # Save answer log
    if new_answers:
        # Append to existing or create new
        all_answers = []
        if answer_log_path.exists():
            with open(answer_log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_answers = list(reader)
        
        # Update with new answers
        answer_dict = {a["question"]: a for a in all_answers}
        for a in new_answers:
            answer_dict[a["question"]] = a
        
        save_answer_log(answer_log_path, list(answer_dict.values()))
        print(f"\nSaved {len(new_answers)} new answers to {answer_log_path}")
    
    # Second pass: evaluate answers
    print(f"\nEvaluating answers...")
    with open(eval_out, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()
        
        for i, row in enumerate(log_rows, 1):
            question = (row.get("question") or "").strip()
            ground_truth_answer = (row.get("answer") or "").strip()
            generated_answer = row.get("_generated_answer", "")
            
            if not generated_answer:
                judge_score = 0
                judge_reasoning = "Failed to generate answer"
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
            
            # Write output row
            out_row = dict(row)
            out_row["generated_answer"] = generated_answer
            out_row["judge_score"] = judge_score
            out_row["judge_reasoning"] = judge_reasoning
            # Remove internal field
            out_row.pop("_generated_answer", None)
            writer.writerow(out_row)
            
            # Progress
            status = "✓" if judge_score == 1 else "✗"
            print(f"[{i}/{total}] {status} Score: {judge_score} | {question[:60]}...")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Answer Generation Evaluation Summary")
    print("=" * 70)
    print(f"Total queries              : {total}")
    print(f"Accurate answers (1)      : {accurate_count} ({100.0 * accurate_count / total:.1f}%)")
    print(f"Inaccurate answers (0)    : {total - accurate_count} ({100.0 * (total - accurate_count) / total:.1f}%)")
    print(f"Failed generation         : {failed_generation}")
    print(f"Reused cached answers     : {reused_answers}")
    print(f"New answers generated     : {len(new_answers)}")
    print(f"\nAnswer log file           : {answer_log_path}")
    print(f"Evaluation output file    : {eval_out}")


if __name__ == "__main__":
    main()
