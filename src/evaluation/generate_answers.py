#!/usr/bin/env python3
"""
Answer Generation Script

This script:
1. Reads the retrieval log CSV (from retrieve_and_stitch.py)
2. For each query, generates an answer using the retrieved contexts
3. Saves generated answers to a log file for later evaluation

Usage:
    uv run python src/evaluation/generate_answers.py

Environment Variables (REQUIRED):
    RETRIEVAL_LOG_CSV or retrieval_log_csv: Path to retrieval log CSV (from retrieve_and_stitch.py)
    ANSWER_LOG_CSV or answer_log_csv: Path to save generated answers (e.g., logs/answer_log.csv)
    ANSWER_MODEL: Model to use for answer generation (e.g., gpt-4o, gemini-2.5-flash)
    
    API Key (one of):
    - OPENAI_API_KEY: Required for OpenAI models (gpt-4o, gpt-4o-mini, etc.)
    - GEMINI_API_KEY: Required for Gemini models (gemini-2.5-flash, etc.)

Environment Variables (OPTIONAL):
    None - all required variables must be set
"""

import os
import json
import csv
import sys
import time
import random
from pathlib import Path
from typing import Optional

# CSV field size limit
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize // 2)

# Load .env file first (before reading env vars)
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

# Gemini client
try:
    from google import genai
except ImportError:
    genai = None


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client from environment."""
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_gemini_client():
    """Get Gemini client from environment."""
    if genai is None:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def is_gemini_model(model: str) -> bool:
    """Check if model name is a Gemini model."""
    if not model:
        return False
    return model.lower().startswith("gemini-")


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


def generate_answer_openai(query: str, context: str, client: OpenAI, model: str, max_retries: int = 5) -> Optional[str]:
    """Generate answer using OpenAI API with retry logic."""
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
    
    for attempt in range(1, max_retries + 1):
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
            if is_retryable_error(e) and attempt < max_retries:
                wait_time = 10.0 * (2 ** (attempt - 1))  # 10s, 20s, 40s, 80s
                print(f"  -> Model overloaded/rate limited (attempt {attempt}/{max_retries}), retrying in {wait_time:.0f}s...", file=sys.stderr)
                exponential_backoff_sleep(attempt, base_delay=10.0)
                continue
            else:
                print(f"Error generating answer with OpenAI: {e}", file=sys.stderr)
                if attempt >= max_retries:
                    print(f"  -> Max retries ({max_retries}) exceeded", file=sys.stderr)
                return None
    
    return None


def generate_answer_gemini(query: str, context: str, client, model: str, max_retries: int = 5) -> Optional[str]:
    """Generate answer using Gemini API with retry logic."""
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
    
    for attempt in range(1, max_retries + 1):
        try:
            # Gemini uses a different API structure
            response = client.models.generate_content(
                model=model,
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            answer = (getattr(response, "text", None) or "").strip()
            return answer if answer else None
        except Exception as e:
            if is_retryable_error(e) and attempt < max_retries:
                wait_time = 10.0 * (2 ** (attempt - 1))  # 10s, 20s, 40s, 80s
                print(f"  -> Model overloaded/rate limited (attempt {attempt}/{max_retries}), retrying in {wait_time:.0f}s...", file=sys.stderr)
                exponential_backoff_sleep(attempt, base_delay=10.0)
                continue
            else:
                print(f"Error generating answer with Gemini: {e}", file=sys.stderr)
                if attempt >= max_retries:
                    print(f"  -> Max retries ({max_retries}) exceeded", file=sys.stderr)
                return None
    
    return None


def generate_answer(query: str, context: str, client, model: str, is_gemini: bool = False) -> Optional[str]:
    """
    Generate an answer to the query using the provided context.
    Supports both OpenAI and Gemini models.
    """
    if is_gemini:
        return generate_answer_gemini(query, context, client, model)
    else:
        return generate_answer_openai(query, context, client, model)


def get_env_path(up: str, low: str, default: str = "") -> str:
    """Get environment variable (case-insensitive)."""
    return os.getenv(up) or os.getenv(low) or default


def load_answer_log(answer_log_path: Path) -> dict[str, str]:
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


def save_answer_log(answer_log_path: Path, rows: list[dict[str, str]]):
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
    # REQUIRED: retrieval_log_csv (retrieval log from retrieve_and_stitch.py)
    log_path = get_env_path("RETRIEVAL_LOG_CSV", "retrieval_log_csv")
    if not log_path:
        raise SystemExit("ERROR: RETRIEVAL_LOG_CSV or retrieval_log_csv environment variable is required.\n"
                        "Set it to the path of your retrieval log CSV file (from retrieve_and_stitch.py).")
    if not Path(log_path).exists():
        raise SystemExit(f"ERROR: Retrieval log file not found: {log_path!r}")
    
    # REQUIRED: answer_log_csv (where to save generated answers)
    answer_log_path = get_env_path("ANSWER_LOG_CSV", "answer_log_csv")
    if not answer_log_path:
        raise SystemExit("ERROR: ANSWER_LOG_CSV or answer_log_csv environment variable is required.\n"
                        "Set it to the path where you want to save generated answers (e.g., logs/answer_log.csv).")
    
    # Ensure logs directory exists
    answer_log_path_obj = Path(answer_log_path)
    answer_log_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Model settings - read from ANSWER_MODEL env var (required)
    answer_model = os.getenv("ANSWER_MODEL") or os.getenv("answer_model")
    if answer_model:
        answer_model = answer_model.strip()
    if not answer_model:
        raise SystemExit("ERROR: ANSWER_MODEL environment variable is required.\n"
                        "Set it to the model you want to use (e.g., gpt-4o, gemini-2.5-flash)")
    
    # Determine if it's a Gemini model
    use_gemini = is_gemini_model(answer_model)
    
    # Get appropriate client
    if use_gemini:
        if genai is None:
            raise SystemExit("ERROR: google-genai package not installed. Run: uv add google-genai")
        client = get_gemini_client()
        if not client:
            raise SystemExit("ERROR: GEMINI_API_KEY not set in environment (required for Gemini models).")
        api_provider = "Gemini"
    else:
        if OpenAI is None:
            raise SystemExit("ERROR: openai package not installed. Run: pip install openai")
        client = get_openai_client()
        if not client:
            raise SystemExit("ERROR: OPENAI_API_KEY not set in environment (required for OpenAI models).")
        api_provider = "OpenAI"
    
    print("=" * 80)
    print("ANSWER GENERATION")
    print("=" * 80)
    print(f"Retrieval log file  : {log_path}")
    print(f"Answer log file     : {answer_log_path}")
    print(f"Generation model    : {answer_model}")
    print(f"API provider        : {api_provider}")
    print("=" * 80)
    print()
    
    # Load retrieval log
    log_rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Use retrieved_context (top-k, as intended)
        required = {"question", "retrieved_context"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Log file missing required columns: {missing}\n"
                           f"Available columns: {list(reader.fieldnames or [])}")
        log_rows = list(reader)
    
    if not log_rows:
        raise SystemExit("Log file has no rows.")
    
    # Load existing answers if available
    existing_answers = load_answer_log(Path(answer_log_path))
    if existing_answers:
        print(f"Loaded {len(existing_answers)} existing answers from {answer_log_path}")
    
    total = len(log_rows)
    failed_generation = 0
    reused_answers = 0
    new_answers = []
    
    print(f"Generating answers for {total} queries...")
    print(f"Answer generation model: {answer_model}")
    print(f"Using top-k retrieved contexts (retrieved_context)\n")
    
    # Generate answers
    for i, row in enumerate(log_rows, 1):
        question = (row.get("question") or "").strip()
        retrieved_context_json = row.get("retrieved_context", "")
        
        # Check if we already have an answer for this question
        if question in existing_answers:
            reused_answers += 1
            print(f"[{i}/{total}] Using cached answer | {question[:60]}...")
        else:
            # Extract context text (full text is now in CSV, no DB lookup needed)
            context_text = extract_context_text(retrieved_context_json)
            
            # Generate answer
            generated_answer = None
            if question and context_text:
                generated_answer = generate_answer(question, context_text, client, answer_model, is_gemini=use_gemini)
            
            if not generated_answer:
                failed_generation += 1
                print(f"[{i}/{total}] ✗ Failed to generate answer | {question[:60]}...")
                # Still save the question with empty answer so we know it was attempted
                new_answers.append({
                    "question": question,
                    "generated_answer": "",  # Empty but recorded
                    "context_used": retrieved_context_json[:200] + "..." if len(retrieved_context_json) > 200 else retrieved_context_json
                })
            else:
                print(f"[{i}/{total}] ✓ Generated answer | {question[:60]}...")
                # Save for answer log
                new_answers.append({
                    "question": question,
                    "generated_answer": generated_answer,
                    "context_used": retrieved_context_json[:200] + "..." if len(retrieved_context_json) > 200 else retrieved_context_json
                })
    
    # Save answer log
    if new_answers:
        # Append to existing or create new
        all_answers = []
        answer_log_path_obj = Path(answer_log_path)
        if answer_log_path_obj.exists():
            with open(answer_log_path_obj, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_answers = list(reader)
        
        # Update with new answers
        answer_dict = {a["question"]: a for a in all_answers}
        for a in new_answers:
            answer_dict[a["question"]] = a
        
        save_answer_log(answer_log_path_obj, list(answer_dict.values()))
        print(f"\nSaved {len(new_answers)} new answers to {answer_log_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Answer Generation Summary")
    print("=" * 70)
    print(f"Total queries              : {total}")
    print(f"New answers generated     : {len(new_answers)}")
    print(f"Reused cached answers     : {reused_answers}")
    print(f"Failed generation         : {failed_generation}")
    print(f"\nAnswer log file           : {answer_log_path}")
    print("\nNext step: Run evaluation script:")
    print(f"  uv run python src/evaluation/evaluate_answers.py")


if __name__ == "__main__":
    main()

