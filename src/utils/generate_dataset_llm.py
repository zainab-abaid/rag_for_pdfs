#!/usr/bin/env python3
import os
import csv
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# --- optional .env ---
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ------------------ OpenAI helpers ------------------ #
def get_openai_client():
    """
    Returns an OpenAI client (new SDK style) or None if not configured.
    """
    try:
        from openai import OpenAI  # pip install openai>=1.0.0
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def llm_gate_keep(text: str, client, model: str = "gpt-4o-mini") -> Tuple[bool, str, Optional[str]]:
    """
    Ask the LLM if this chunk is suitable for a product-specific Q&A.
    Returns (keep, reason, product_or_none).
    """
    if client is None:
        # No client => default keep, unknown product
        return True, "no-llm-default-keep", None

    sys = (
        "You are a strict gatekeeper building a RAG question dataset from Extreme Networks product documentation.\n"
        "APPROVE (keep=true) ONLY IF:\n"
        "1) A single product/model/version is identifiable from the passage itself (titles, headings, 'Path:' lines).\n"
        "   Examples: AP410C, AP510e, AP560i, X465, VSP series, ExtremeXOS 32.6, ExtremeCloud IQ, IQEngine.\n"
        "   Do NOT guess or invent; if ambiguous, do not approve.\n"
        "2) The passage is coherent and self-contained enough to support at least one clear factual Q&A strictly from the text.\n"
        "DECLINE if no unambiguous product, or if it's boilerplate/nav crumbs/mostly empty/needs missing context.\n"
        "Return JSON ONLY as: {\"keep\": true|false, \"reason\": \"<under 12 words>\", \"product\": \"<token or null>\"}."
    )
    user = (
        "PASSAGE START\n"
        f"{text}\n"
        "PASSAGE END\n\n"
        "Reply with JSON ONLY (no prose). If you cannot identify a single clear product/model from the passage itself, return:\n"
        "{\"keep\": false, \"reason\": \"no product\", \"product\": null}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            max_tokens=80,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        keep = bool(data.get("keep"))
        reason = str(data.get("reason", "")).strip()
        product = data.get("product")
        if isinstance(product, str):
            product = product.strip() or None
        else:
            product = None
        return keep, (reason or ("keep" if keep else "skip")), product
    except Exception:
        # on any error, default keep so pipeline continues
        return True, "llm-gate-error-default-keep", None

def llm_generate_qa(text: str, client, model: str = "gpt-4o-mini") -> Optional[Tuple[str, str]]:
    """
    Generate one product-specific Q&A. Return (q, a) or None.
    Returns None if the model outputs an empty pair or on failure.
    """
    if client is None:
        return None

    sys = (
        "You create ONE clear, factual, product-specific Q&A for a RAG dataset on Extreme Networks docs.\n"
        "Rules:\n"
        "1) Identify the product/model/version from the passage itself (e.g., AP410C, AP510e, X465, ExtremeXOS 32.6, ExtremeCloud IQ).\n"
        "   Use tokens exactly as written; include version if shown.\n"
        "   If you cannot identify the product with high confidence, return the EMPTY pair.\n"
        "2) QUESTION must explicitly include the product/model so it stands alone.\n"
        "   Good: “On the AP410C, what does …?”  Bad: “What does … on the port?”\n"
        "3) ANSWER must be strictly derived from the passage, concise (<= 30 words), no fluff.\n"
        "4) No hallucinations. If insufficient info, return EMPTY pair.\n"
        "Return JSON ONLY: {\"question\": \"...\", \"answer\": \"...\"}  (or empty pair)."
    )
    user = (
        "PASSAGE START\n"
        f"{text}\n"
        "PASSAGE END\n\n"
        "Return JSON ONLY. If the product/model cannot be identified with high confidence from the passage, return:\n"
        "{\"question\": \"\", \"answer\": \"\"}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            max_tokens=180,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        q = str(data.get("question", "")).strip()
        a = str(data.get("answer", "")).strip()
        if q and a:
            return q, a
        return None
    except Exception:
        return None

# ------------------ Core helpers ------------------ #
def parse_env_int(var_name: str, default: Optional[int] = None) -> int:
    val = os.environ.get(var_name)
    if val is None or str(val).strip() == "":
        if default is None:
            raise SystemExit(f"Environment variable '{var_name}' is required but not set.")
        return default
    try:
        return int(str(val).strip())
    except ValueError:
        raise SystemExit(f"Invalid integer for {var_name}: {val!r}")

def require_env_path(var_name: str) -> Path:
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise SystemExit(f"Environment variable '{var_name}' is required but not set.")
    p = Path(value)
    if not p.exists():
        raise SystemExit(f"Directory from env '{var_name}' not found: {p}")
    if not p.is_dir():
        raise SystemExit(f"Path from env '{var_name}' is not a directory: {p}")
    return p

def collect_json_files_recursive(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.json") if p.is_file()]

def read_json_list(file_path: Path) -> Optional[List[dict]]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with file_path.open("r", encoding="latin-1") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, list) else None

def pick_mean_length_chunk(chunks: List[dict]) -> Optional[dict]:
    pairs = []
    for c in chunks:
        t = c.get("text")
        if isinstance(t, str) and t.strip():
            pairs.append((c, len(t)))
    if not pairs:
        return None
    mean_len = sum(l for _, l in pairs) / len(pairs)
    pairs.sort(key=lambda cl: abs(cl[1] - mean_len))
    return pairs[0][0]  # chosen chunk dict

def write_csv(rows: List[List[str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "context", "source_path", "gt_section_id"])
        writer.writerows(rows)

# ------------------ Main ------------------ #
def main():
    # Required env
    chunks_dir = require_env_path("chunks_dir")
    target_size = parse_env_int("QUERY_DATASET_SIZE")  # exact number of rows to produce

    # Optional env
    gate_model = os.environ.get("SKIP_JUDGE_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    qa_model = os.environ.get("QA_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    max_context_chars = parse_env_int("MAX_CONTEXT_CHARS", 12000)

    parser = argparse.ArgumentParser(
        description="Build a CSV of QUERY_DATASET_SIZE rows from JSON chunk files; LLM-gate and product-specific Q&A."
    )
    parser.add_argument("--reported_prefix", default=str(chunks_dir),
                        help="Prefix used in source_path column (default: chunks_dir)")
    parser.add_argument("--output_csv", default="query_dataset_with_qa_200.csv",
                        help="Output CSV path (default: query_dataset_with_qa_200.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--allow_blank_qa", action="store_true",
                        help="If no OPENAI_API_KEY, allow writing blank Q/A (not recommended).")
    parser.add_argument("--accept_all", action="store_true",
                        help="Bypass LLM gate and accept all chunks before QA generation.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    client = get_openai_client()
    if client is None and not args.allow_blank_qa:
        raise SystemExit(
            "OPENAI_API_KEY not set. To populate 'question' and 'answer', set the key.\n"
            "If you intentionally want blanks, pass --allow_blank_qa."
        )
    elif client is None and args.allow_blank_qa:
        print("[warn] OPENAI_API_KEY not set. Will write BLANK question/answer (per --allow_blank_qa).")

    files = collect_json_files_recursive(chunks_dir)
    if not files:
        raise SystemExit(f"No .json files found under: {chunks_dir}")

    random.shuffle(files)

    accepted = 0
    skipped = 0
    attempts = 0
    rows: List[List[str]] = []

    idx = 0
    n = len(files)

    while accepted < target_size:
        if idx >= n:
            random.shuffle(files)
            idx = 0

        p = files[idx]
        idx += 1
        attempts += 1

        data = read_json_list(p)
        if not data:
            skipped += 1
            print(f"[{attempts}] {accepted}/{target_size} | skipped {skipped} | SKIP unreadable {p}")
            continue

        chosen = pick_mean_length_chunk(data)
        if not chosen:
            skipped += 1
            print(f"[{attempts}] {accepted}/{target_size} | skipped {skipped} | SKIP no-text {p}")
            continue

        text = chosen.get("text", "")
        meta: Dict = chosen.get("metadata", {}) or {}
        # canonical section id (same rule as retrieval)
        gt_section_id = meta.get("section_node_id") or meta.get("chunk_id") or ""

        # Trim overly long context for CSV
        context = text if len(text) <= max_context_chars else text[:max_context_chars]

        # Gate (unless bypassed)
        keep = True
        gate_reason = "accepted"
        if not args.accept_all and client is not None:
            keep, gate_reason, _product = llm_gate_keep(context, client, gate_model)

        if not keep:
            skipped += 1
            print(f"[{attempts}] {accepted}/{target_size} | skipped {skipped} | SKIP {p} :: {gate_reason}")
            continue

        # Generate Q&A (or blanks if allowed)
        if client is not None:
            qa = llm_generate_qa(context, client, qa_model)
            if qa is None:
                skipped += 1
                print(f"[{attempts}] {accepted}/{target_size} | skipped {skipped} | SKIP {p} :: qa-empty-or-failed")
                continue
            question, answer = qa
        else:
            question, answer = "", ""

        # Reported source_path relative to chunks_dir for visibility
        try:
            rel = p.relative_to(chunks_dir)
            reported = str(Path(args.reported_prefix) / rel)
        except ValueError:
            reported = str(Path(args.reported_prefix) / p.name)

        rows.append([question, answer, context, reported, gt_section_id])
        accepted += 1
        print(f"[{attempts}] {accepted}/{target_size} | skipped {skipped} | KEEP {p} :: {gate_reason} :: QA ok")

    write_csv(rows, Path(args.output_csv))
    print("\n--- Done ---")
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Total attempts: {attempts} | kept: {accepted} | skipped: {skipped}")
    print(f"Root directory (chunks_dir): {chunks_dir}")
    if client is not None:
        print(f"Models -> gate: {gate_model}, qa: {qa_model}, max_context_chars: {max_context_chars}")

if __name__ == "__main__":
    main()