#!/usr/bin/env python3
"""
Answer Generation Script

This script:
1. Reads the retrieval log CSV (from pdf_chunks_retrieval.py)
2. For each query, generates an answer using the retrieved contexts
3. Saves generated answers to a separate log file. This file will be automatically created through script, no need to set it via .env file 

Environment Variables (REQUIRED):
    PDF_RETRIEVAL_LOG_CSV: Path to retrieval log CSV
    CHUNKING_STRATEGY: Chunking strategy used (e.g., "recursive", "fixed", etc.) to create output filename

Optional Environment Variables:
    USE_GROQ: 'true' to use Groq API
    ANSWER_MODEL: Model to use for answer generation (OpenAI/Gemini) and also used to create output filename
    GROQ_MODEL: Model to use for Groq (if USE_GROQ=true)
    ANSWER_MAX_TOKENS: Max tokens for answer generation (default: 500)
"""

import os
import sys
import json
import csv
import time
import random
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List

# CSV field size limit
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(sys.maxsize // 2)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# API clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

CHUNKING_STRATEGY = os.environ.get("CHUNKING_STRATEGY", "recursive")
ANSWER_MODEL= os.environ.get("ANSWER_MODEL", "gpt-4o")
ANSWER_MAX_TOKENS = int(os.environ.get("ANSWER_MAX_TOKENS", "500"))
PDF_RETRIEVAL_LOG_CSV = os.environ.get("PDF_RETRIEVAL_LOG_CSV", f"logs/pdf_retrieval_log_{CHUNKING_STRATEGY}.csv")

def is_gemini_model(model: str) -> bool:
    return bool(model and model.lower().startswith("gemini-"))


def extract_context_text(retrieved_context_json: str) -> str:
    """Extract full concatenated text from retrieved_context JSON, removing duplicates"""
    if not retrieved_context_json:
        return ""
    try:
        blocks = json.loads(retrieved_context_json)
        if not isinstance(blocks, list):
            return ""

        seen_texts = set()
        sections = []

        for block in blocks:
            if not isinstance(block, dict):
                continue
            full_text = block.get("text", "")
            doc_title = block.get("doc_title", "")
            section_path = block.get("section_path", [])

            if not full_text:
                continue

            headers = []
            if doc_title:
                headers.append(f"Document: {doc_title}")
            if section_path:
                if isinstance(section_path, list):
                    headers.append(f"Section: {' > '.join(section_path)}")
                else:
                    headers.append(f"Section: {section_path}")

            section_text = full_text.strip()
            if headers:
                section_text = "\n".join(headers) + "\n\n" + section_text

            # Deduplicate
            if section_text not in seen_texts:
                sections.append(section_text)
                seen_texts.add(section_text)

        return ("\n\n" + "=" * 80 + "\n\n").join(sections)

    except Exception as e:
        print(f"Error parsing retrieved_context: {e}", file=sys.stderr)
        return ""


def is_retryable_error(error: Exception) -> bool:
    """Check for retryable errors"""
    msg = str(error).lower()
    return any(
        keyword in msg
        for keyword in [
            "overloaded",
            "rate limit",
            "too many requests",
            "service unavailable",
            "503",
            "429",
            "quota exceeded",
            "temporarily unavailable",
        ]
    )

def exponential_backoff_sleep(attempt: int, base_delay: float = 10.0, max_delay: float = 120.0):
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    jitter = delay * 0.2 * random.random()
    time.sleep(delay + jitter)

def generate_answer_openai(
    query: str, context: str, client: Any, model: str, max_tokens: int, max_retries: int = 5
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
                temperature=0,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
            )
            if hasattr(resp, "choices") and resp.choices:
                return getattr(resp.choices[0].message, "content", "").strip()
            return None
        except Exception as e:
            if is_retryable_error(e) and attempt < max_retries:
                exponential_backoff_sleep(attempt)
                continue
            else:
                print(f"OpenAI answer generation error: {e}", file=sys.stderr)
                return None
    return None


def generate_answer_gemini(
    query: str, context: str, client, model: str, max_tokens: int, max_retries: int = 5
) -> Optional[str]:
    if not query or not context:
        return None
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context.\n"
        "Be precise, include technical details, section titles, product names, and version info.\n"
        "Answer ONLY using the context."
    )
    user_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    for attempt in range(1, max_retries + 1):
        try:
            model_obj = client.models.get(model)
            resp = model_obj.generate_content(user_prompt, max_output_tokens=max_tokens)
            answer = getattr(resp, "text", None)
            return answer.strip() if answer else None
        except Exception as e:
            if is_retryable_error(e) and attempt < max_retries:
                exponential_backoff_sleep(attempt)
                continue
            else:
                print(f"Gemini answer generation error: {e}", file=sys.stderr)
                return None
    return None


def generate_answer_groq(
    query: str, context: str, client, model: str, max_tokens: int, max_retries: int = 5
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


def generate_answer(
    query: str, context: str, client, model: str, is_gemini=False, is_groq=False, max_tokens=500
) -> Optional[str]:
    if is_groq:
        return generate_answer_groq(query, context, client, model, max_tokens)
    elif is_gemini:
        return generate_answer_gemini(query, context, client, model, max_tokens)
    else:
        return generate_answer_openai(query, context, client, model, max_tokens)


def load_existing_answers(path: Path, model_filter: str = None) -> Dict[Tuple[str, str], str]:
    if not path.exists():
        return {}
    answers = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("question") or "").strip()
                a = (row.get("generated_answer") or "").strip()
                m = (row.get("model") or "").strip()
                if model_filter and m != model_filter:
                    continue
                if q:
                    answers[(q, m)] = a
    except Exception as e:
        print(f"Warning: Could not load answer log: {e}", file=sys.stderr)
    return answers

def save_answer_log(path: Path, rows: List[Dict], model: str):
    # Merge new rows with existing ones
    existing = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        except Exception:
            existing = []

    merged = {(r.get("question", ""), model): r for r in existing}
    for r in rows:
        key = (r.get("question", ""), model)
        merged[key] = {**r, "model": model}  # Keep all keys in r, plus model

    # Dynamically get fieldnames from merged data
    all_fieldnames = set()
    for r in merged.values():
        all_fieldnames.update(r.keys())
    fieldnames = list(all_fieldnames)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in merged.values():
            writer.writerow(r)

def resolve_answer_log_path(answer_model: str, chunking_strategy: str) -> str:
    """Build filename: logs/answers_{ANSWER_MODEL}_query_dataset_with_qa_{CHUNKING_STRATEGY}.csv"""
    if "/" in answer_model:
        answer_model = answer_model.split("/")[0]
    safe_model = answer_model.replace("/", "_").replace(" ", "_")
    return f"logs/answers_{safe_model}_query_dataset_with_qa_{chunking_strategy}.csv"

def main():
    if not PDF_RETRIEVAL_LOG_CSV or not Path(PDF_RETRIEVAL_LOG_CSV).exists():
        raise SystemExit("PDF Retrieval log CSV not found")
    
    PDF_ANSWER_LOG_CSV = resolve_answer_log_path(ANSWER_MODEL, CHUNKING_STRATEGY)
    Path(PDF_ANSWER_LOG_CSV).parent.mkdir(parents=True, exist_ok=True)
    if not PDF_ANSWER_LOG_CSV:
        raise SystemExit("PDF_ANSWER_LOG_CSV not set")

    use_groq = os.environ.get("USE_GROQ", os.environ.get("use_groq", "false")).lower() == "true"
    model = os.environ.get(
        "GROQ_MODEL" if use_groq else "ANSWER_MODEL",
        os.environ.get("groq_model" if use_groq else "answer_model", "")
    )
    if not model:
        raise SystemExit("Model not set")

    is_gemini = is_gemini_model(model)
    print(f"Answer generation will use OpenAI model: {model}")

    # Initialize client
    if use_groq:
        if Groq is None:
            raise SystemExit("Groq package not installed")
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    elif is_gemini:
        if genai is None:
            raise SystemExit("google-genai package not installed")
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        if OpenAI is None:
            raise SystemExit("openai package not installed")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load CSV
    with open(PDF_RETRIEVAL_LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    existing_answers: Dict[Tuple[str, str], str] = load_existing_answers(Path(PDF_ANSWER_LOG_CSV), model_filter=model)
    new_answers: List[Dict[str, str]] = []

    total = len(rows)
    for idx, row in enumerate(rows, 1):
        question = (row.get("question") or "").strip()
        if not question:
            print(f"[{idx}/{total}] Skipping empty question")
            continue
        retrieved_context_json = row.get("retrieved_context", "")

        if (question, model) in existing_answers:
            print(f"[{idx}/{total}] Using cached answer | {question[:60]}...")
            continue

        context_text = extract_context_text(retrieved_context_json)
        if not context_text.strip():
            print(f"[{idx}/{total}] Skipping (empty retrieved context)")
            continue

        answer_text = generate_answer(
            question, context_text, client, model, is_gemini, use_groq, ANSWER_MAX_TOKENS
        )
        if not answer_text:
            answer_text = ""

        new_answers.append({
            "question": question,
            "generated_answer": answer_text,
            "context_used": context_text
        })
        print(f"[{idx}/{total}] Generated answer | {question[:60]}...")

    if new_answers:
        save_answer_log(Path(PDF_ANSWER_LOG_CSV), new_answers, model)
        print(f"\nSaved {len(new_answers)} new answers to {PDF_ANSWER_LOG_CSV} (model: {model})")


if __name__ == "__main__":
    main()
