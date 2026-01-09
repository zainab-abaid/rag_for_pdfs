"""
Do not use this script
This script annotates text nodes extracted from PDF documents with their corresponding sections.

Steps:
1. Loads nodes from PKL files and sections from JSON files.
2. Splits nodes into batches for parallel processing.
3. For each node, finds sections that match based on word overlap and fuzzy text similarity.
4. Annotates node metadata with matching section IDs and similarity scores.
5. Saves the annotated nodes back to PKL files, preserving folder structure.

Configuration (via environment variables):
- CHUNKING_STRATEGY: 'recursive' or other strategy folder
- OVERLAP_THRESHOLD: minimum fuzzy match score (0–1)
- MIN_WORD_OVERLAP: minimum shared words to consider a section
- BATCH_SIZE: number of nodes per batch
- MAX_WORKERS: number of parallel worker processes
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from rapidfuzz import fuzz

# ------------------------Configuration-------------------------
CHUNKING_STRATEGY = os.environ.get("CHUNKING_STRATEGY", "recursive")
OVERLAP_THRESHOLD = float(os.environ.get("OVERLAP_THRESHOLD", 0.3))
MIN_WORD_OVERLAP = int(os.environ.get("MIN_WORD_OVERLAP", 10))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 500))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", mp.cpu_count()))
PDF_PARSED_OUTPUT_DIR = Path(os.environ.get("PDF_PARSED_OUTPUT_DIR", "./data/pdf_node_chunks"))
SECTION_JSON_DIR = Path(os.environ.get("OUTPUT_DIR", "./data/sectionwise_chunks"))
ANNOTATED_NODES_DIR = Path(os.environ.get("ANNOTATED_NODES_DIR", "./data/annotated_nodes"))

# ------------------------Text helpers---------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().replace("\n", " ").split())

def get_word_set(text: str) -> Set[str]:
    return set(text.split())

# ------------------------Node helpers---------------------------
def get_node_text(node: Union[dict, object]) -> Optional[str]:
    return node.get("text") if isinstance(node, dict) else getattr(node, "text", None)

def get_node_metadata(node: Union[dict, object]) -> Optional[dict]:
    if isinstance(node, dict):
        return node.setdefault("metadata", {})
    return getattr(node, "metadata", None)

# ------------------------Section loading + indexing-------------
def load_sections(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = []
    for obj in data:
        sid = obj.get("metadata", {}).get("section_node_id")
        text = obj.get("text", "")
        if not sid or not text.strip():
            continue

        norm = normalize(text)
        sections.append({
            "section_id": sid,
            "norm_text": norm,
            "word_set": get_word_set(norm),
            "text_len": len(norm),
        })
    return sections

def build_section_index(sections: List[Dict]):
    """Returns: section_by_id: section_id -> section dict word_to_section_ids: word -> set(section_ids)"""
    section_by_id = {}
    word_to_section_ids: Dict[str, Set[str]] = {}

    for sec in sections:
        sid = sec["section_id"]
        section_by_id[sid] = sec
        for w in sec["word_set"]:
            word_to_section_ids.setdefault(w, set()).add(sid)

    return section_by_id, word_to_section_ids

# ------------------------Core matching logic (FAST)----------
def match_node_to_sections(
    normalized_node_text: str,
    node_word_set: Set[str],
    section_by_id: Dict[str, Dict],
    word_to_section_ids: Dict[str, Set[str]],
) -> Tuple[List[str], Dict[str, float]]:
    """Finds sections that best match a text node by shared words and fuzzy text similarity. Returns the matching section IDs and their similarity scores."""
    if not normalized_node_text or not node_word_set:
        return [], {}

    # 1️. Candidate section IDs via word_to_section_ids
    shared_word_count_by_section: Dict[str, int] = {}
    for w in node_word_set:
        for sid in word_to_section_ids.get(w, ()):
            shared_word_count_by_section[sid] = shared_word_count_by_section.get(sid, 0) + 1

    # 2️. Filter by min word overlap
    candidate_section_ids = [
        sid for sid, cnt in shared_word_count_by_section.items()
        if cnt >= MIN_WORD_OVERLAP
    ]

    matched_section_ids = []
    match_scores_by_section = {}
    node_len = len(normalized_node_text)

    # 3️. Fuzzy match on small candidate set
    for sid in candidate_section_ids:
        sec = section_by_id[sid]

        # Length pruning
        len_ratio = min(node_len, sec["text_len"]) / max(node_len, sec["text_len"])
        if len_ratio < 0.2:
            continue

        score = fuzz.partial_ratio(normalized_node_text, sec["norm_text"]) / 100
        if score >= OVERLAP_THRESHOLD:
            matched_section_ids.append(sid)
            match_scores_by_section[sid] = score

    return matched_section_ids, match_scores_by_section

# ---------------------------Worker - builds index per process-----
def process_node_batch(args):
    """Processes a batch of nodes and finds matching sections for each node. Returns a list of matches and similarity scores for all nodes in the batch."""
    nodes_batch, sections = args
    
    # Build index once per worker process (fast, avoids pickling issues)
    section_by_id, word_to_section_ids = build_section_index(sections)
    
    results = []
    for node in nodes_batch:
        text = get_node_text(node)
        if not text:
            results.append(([], {}))
            continue

        norm = normalize(text)
        words = set(norm.split())

        results.append(
            match_node_to_sections(norm, words, section_by_id, word_to_section_ids)
        )

    return results

# --------------------------Annotation--------------------------
def annotate_nodes_with_sections(pkl_path: str, json_path: str, output_pkl_path: str):
    """Annotates nodes with matching sections and overlap scores, processing in parallel, and saves the results to a pickle file."""
    print(f"\n{'─'*60}\nDocument: {Path(pkl_path).name}\n{'─'*60}")

    with open(pkl_path, "rb") as f:
        nodes = pickle.load(f)

    if not nodes:
        print("No nodes found")
        return

    sections = load_sections(json_path)
    if not sections:
        print("No sections found")
        return

    print(f"Loaded {len(nodes):,} nodes, {len(sections):,} sections")

    batches = [
        nodes[i:i + BATCH_SIZE]
        for i in range(0, len(nodes), BATCH_SIZE)
    ]

    start = time.time()
    processed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass sections (not index) - each worker builds its own index
        futures = {
            executor.submit(process_node_batch, (batch, sections)): idx
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            results = future.result()

            base = batch_idx * BATCH_SIZE
            for i, (section_ids, overlaps) in enumerate(results):
                idx = base + i
                if idx >= len(nodes):
                    break

                meta = get_node_metadata(nodes[idx])
                if meta is not None:
                    meta["section_ids"] = section_ids
                    meta["section_overlap_ratios"] = overlaps

            processed += len(results)
            elapsed = time.time() - start
            rate = processed / elapsed
            eta = (len(nodes) - processed) / rate if rate else 0

            print(
                f"  {processed:,}/{len(nodes):,} | "
                f"{rate:.0f} nodes/sec | ETA {eta:.0f}s",
                flush=True,
            )

    os.makedirs(Path(output_pkl_path).parent, exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(nodes, f)

    elapsed = time.time() - start
    print(f"✓ Done in {elapsed:.1f}s ({len(nodes)/elapsed:.0f} nodes/sec)")

# -----------------------Batch processing---------------------
def process_all_documents(chunking_strategy: str):
    """Processes all PKL node files for a given chunking strategy, annotates them with matching sections, and saves the annotated results."""
    
    base_pkl_dir = PDF_PARSED_OUTPUT_DIR / chunking_strategy
    base_json_dir = SECTION_JSON_DIR
    output_dir = ANNOTATED_NODES_DIR / chunking_strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(base_pkl_dir.rglob("*.pkl"))
    print(f"Found {len(pkl_files)} PKL files")

    for pkl_file in pkl_files:
        doc_name = pkl_file.stem
        parent = pkl_file.parent.name

        json_path = base_json_dir / parent / doc_name / "markdown" / f"{doc_name}.json"
        if not json_path.exists():
            print(f"Skipping {doc_name} (JSON not found)")
            continue

        out_file = output_dir / pkl_file.relative_to(base_pkl_dir)
        annotate_nodes_with_sections(str(pkl_file), str(json_path), str(out_file))

# ------------------------Entry------------------------
if __name__ == "__main__":
    process_all_documents(CHUNKING_STRATEGY)