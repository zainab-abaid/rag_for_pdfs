#!/usr/bin/env python3
"""
Answer Generation Script

This script:
1. Reads the retrieval log CSV (from `retrieve_with_diff_rerankers.py`)
2. For each query, generates an answer using the retrieved contexts
3. Saves generated answers to a log file for later evaluation

Usage:
    uv run python src/rerankingexperiments/generate_answers_diff_rerankers.py

Environment Variables (REQUIRED):
    RETRIEVAL_LOG_CSV: Path to the retrieval log CSV (e.g., `logs/retrieval_log_TOPK5_none.csv`)
    ANSWER_LOG_CSV: Path to save generated answers (e.g., `logs/answers_gemini_query_dataset_with_qa_none.csv`)

    Model selection (one of):
    - If `USE_GROQ=true`: `GROQ_MODEL` (e.g., `llama-3.3-70b-versatile`) and `GROQ_API_KEY`
    - If `USE_GROQ=false` or not set: `ANSWER_MODEL` (e.g., `gpt-4o`, `gemini-2.5-flash`)

    API Key (based on model selection):
    - `GROQ_API_KEY`: Required when `USE_GROQ=true`
    - `OPENAI_API_KEY`: Required for OpenAI models (when `USE_GROQ=false`)
    - `GEMINI_API_KEY`: Required for Gemini models (when `USE_GROQ=false`)

Environment Variables (OPTIONAL):
    USE_GROQ: Set to `true` to use Groq API instead of OpenAI/Gemini (default: `false`)
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

# Groq client
try:
    from groq import Groq
except ImportError:
    Groq = None

ANSWER_MAX_TOKENS = int(os.environ.get("ANSWER_MAX_TOKENS", "500"))


def get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI and api_key:
        return OpenAI(api_key=api_key)
    return None

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if genai and api_key:
        return genai.Client(api_key=api_key)
    return None

def get_groq_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY")
    if Groq and api_key:
        return Groq(api_key=api_key)
    return None

def is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini-") if model else False

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
        "overloaded", "rate limit", "too many requests",
        "service unavailable", "503", "429", "quota exceeded", "temporarily unavailable"
    ]
    return any(keyword in error_str for keyword in retryable_keywords)

def exponential_backoff_sleep(attempt: int, base_delay: float = 10.0, max_delay: float = 120.0):
    """Sleep with exponential backoff and jitter."""
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    # Add jitter (random 0-20% of delay) to avoid thundering herd
    jitter = delay * 0.2 * random.random()
    time.sleep(delay + jitter)

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

def generate_answer_groq(
    query: str, context: str, client, model: str, max_tokens: int = ANSWER_MAX_TOKENS, max_retries: int = 5
) -> Optional[str]:
    if not query or not context:
        return None
    
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context.\n"
        "Be precise, include technical details, section titles, product names, and version info.\n"
        "Answer ONLY using the context."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            if hasattr(resp, "choices") and resp.choices:
                return getattr(resp.choices[0].message, "content", "").strip()
            return None
        except Exception as e:
            if is_retryable_error(e) and attempt < max_retries:
                print(f"Groq API error (attempt {attempt}/{max_retries}): {e}", file=sys.stderr)
                exponential_backoff_sleep(attempt)
                continue
            else:
                print(f"Groq answer generation error: {e}", file=sys.stderr)
                return None
    
    return None

def generate_answer(query: str, context: str, client, model: str, is_gemini=False, is_groq=False) -> Optional[str]:
    """
    Generate an answer to the query using the provided context.
    Supports OpenAI, Gemini, and Groq models.
    """
    if is_groq:
        return generate_answer_groq(query, context, client, model)
    elif is_gemini:
        return generate_answer_gemini(query, context, client, model)
    else:
        return generate_answer_openai(query, context, client, model)

def resolve_answer_log_path(answer_model: str, reranker_mode: str) -> str:
    """Build filename: logs/answers_{ANSWER_MODEL}_query_dataset_with_qa_{RERANKER_MODE}.csv"""
    if "/" in answer_model:
        answer_model = answer_model.split("/")[0]
    safe_model = answer_model.replace("/", "_").replace(" ", "_")
    return f"logs/answers_{safe_model}_query_dataset_with_qa_{reranker_mode}.csv"

def get_env_path(up: str, low: str, default: str = "") -> str:
    """Get environment variable (case-insensitive)."""
    return os.getenv(up) or os.getenv(low) or default

def load_answer_log(answer_log_path: Path, model_filter: str = None) -> dict[str, str]:
    """
    Load previously generated answers from log file.
    If model_filter is provided, only loads answers that match the model.
    For backward compatibility: if model column is missing, ignores model_filter.
    """
    if not answer_log_path.exists():
        return {}
    answers = {}

    try:
        with open(answer_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            has_model_column = "model" in (reader.fieldnames or [])
            
            for row in reader:
                question = (row.get("question") or "").strip()
                answer = (row.get("generated_answer") or "").strip()
                
                # Filter by model if specified and model column exists
                if model_filter and has_model_column:
                    model = (row.get("model") or "").strip()
                    if model != model_filter:
                        continue
                # If model_filter is set but file doesn't have model column (old format),
                # don't load any cached answers to force regeneration
                elif model_filter and not has_model_column:
                    continue

                if question and answer:
                    answers[question] = answer
    except Exception as e:
        print(f"Warning: Could not load answer log: {e}", file=sys.stderr)
    return answers

def save_answer_log(answer_log_path: Path, rows: list[dict[str, str]], model: str, reranker_mode: str):
    """Save generated answers to log file. Includes model information."""
    if not rows:
        return
    fieldnames = ["reranker_mode", "question", "generated_answer", "context_used", "model"]
    existing_rows = []
    if answer_log_path.exists():
        try:
            with open(answer_log_path, "r", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
        except Exception:
            existing_rows = []

    # Merge existing rows with new rows (new rows override existing for same question+model)
    existing_dict = {(r.get("question","").strip(), r.get("model","").strip(), r.get("reranker_mode","").strip()): r for r in existing_rows}
    for row in rows:
        key = (row.get("question","").strip(), model, reranker_mode)
        existing_dict[key] = {
            "reranker_mode": reranker_mode,
            "question": row.get("question",""),
            "generated_answer": row.get("generated_answer",""),
            "context_used": row.get("context_used",""),
            "model": model,
        }
    with open(answer_log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_dict.values():
            writer.writerow(row)


# -------------------- Main --------------------

def main():
    # REQUIRED: retrieval_log_csv (retrieval log from retrieve_and_stitch.py)
    # Environment
    retrieval_log_base = get_env_path("RETRIEVAL_LOG_CSV", "retrieval_log_csv")
    reranker_mode = get_env_path("RERANKER_MODE", "reranker_mode", "none")

    if not retrieval_log_base:
        raise SystemExit("RETRIEVAL_LOG_CSV environment variable is required")

    # Insert reranker_mode into filename before extension
    base_path = Path(retrieval_log_base)
    retrieval_log_path = str(base_path.with_name(f"{base_path.stem}_{reranker_mode}{base_path.suffix}"))

    if not Path(retrieval_log_path).exists():
        raise SystemExit(f"Retrieval log file not found: {retrieval_log_path}")

    answer_model = get_env_path("ANSWER_MODEL", "answer_model")
    if not answer_model:
        raise SystemExit("ANSWER_MODEL environment variable is required")

    answer_log_path = resolve_answer_log_path(answer_model, reranker_mode)

    print(f"Retrieval log (input)  : {retrieval_log_path}")
    print(f"Answer log (output)    : {answer_log_path}")

    # Client selection
    use_groq = (os.getenv("USE_GROQ") or "").strip().lower() == "true"
    if use_groq:
        client = get_groq_client()
        if not client:
            raise SystemExit("GROQ_API_KEY not set")
        use_gemini = False
        provider = "Groq"
    else:
        # Determine if it's a Gemini model
        use_gemini = is_gemini_model(answer_model)
        if use_gemini:
            client = get_gemini_client()
            if not client:
                raise SystemExit("GEMINI_API_KEY not set")
            provider = "Gemini"
        else:
            client = get_openai_client()
            if not client:
                raise SystemExit("OPENAI_API_KEY not set")
            provider = "OpenAI"

    print("="*80)
    print("ANSWER GENERATION")

    print("="*80)
    print(f"Retrieval log file  : {retrieval_log_path}")
    print(f"Answer log file     : {answer_log_path}")
    print(f"Generation model    : {answer_model}")
    print(f"Reranker mode       : {reranker_mode}")
    print(f"API provider        : {provider}")
    print("="*80)
    print()

    # Load retrieval log
    with open(retrieval_log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"question", "retrieved_context"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in retrieval log: {missing}")
        log_rows = list(reader)

    if not log_rows:
        raise SystemExit("Retrieval log has no rows")

    # Load existing answers
    existing_answers = load_answer_log(Path(answer_log_path), model_filter=answer_model)
    if existing_answers:
        print(f"Loaded {len(existing_answers)} existing answers from {answer_log_path} (model: {answer_model})")
    
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
                generated_answer = generate_answer(question, context_text, client, answer_model, is_gemini=use_gemini, is_groq=use_groq)
            
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
    
    # Save answer log (includes model information)
    if new_answers:
        save_answer_log(Path(answer_log_path), new_answers, answer_model, reranker_mode)
        print(f"\nSaved {len(new_answers)} new answers to {answer_log_path} (model: {answer_model})")
    
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