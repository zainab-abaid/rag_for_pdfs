#!/usr/bin/env python3
"""
Section-wise Markdown Chunker for RAG Systems

This module chunks markdown documents into sections and subsections, preserving document
structure while enabling efficient retrieval and reassembly. The chunking strategy is
designed to maintain semantic coherence by respecting section boundaries and keeping
related content together.

CHUNKING STRATEGY:
-----------------
1. **Section Hierarchy**: Builds a tree structure from markdown headings (H1-H6, Setext)
   - Each section node tracks its title, level, character span, and parent-child relationships
   - Sections with children may have "intro spans" (prose before first child heading)

2. **Chunking Units**:
   - **Leaf sections**: Smallest subsections that have no children → chunked as-is
   - **Parent intro spans**: If a parent section has prose before its first child,
     that intro text becomes a separate chunk (marked as `is_intro_chunk=True`)
   - **Section body**: The main content of a leaf section (everything after the heading)

3. **Splitting Rules**:
   - Sections longer than MAX_CHARS are split using a table-aware algorithm:
     * Tables (Markdown and HTML) are kept atomic (never split, even if > MAX_CHARS)
     * Paragraphs are aggregated up to MAX_CHARS
     * If a paragraph exceeds MAX_CHARS, it's split at sentence boundaries
     * Overlap of SPLIT_OVERLAP_CHARS is added between consecutive splits
   - Sections shorter than MIN_CHARS are marked as `is_small_section=True`

4. **Front-matter Filtering**:
   - Sections matching titles in DROP_TITLES catalog are excluded (unless KEEP_FRONT_MATTER=true)
   - Common filtered sections: "Preface", "Table of Contents", "Legal Notice", etc.
   - Intro spans may have "on page X" lines stripped (common PDF artifacts)

5. **Text Formatting**:
   - Each chunk includes a breadcrumb path (e.g., "Path: Introduction > Getting Started")
   - Document title is included in the text payload
   - Section heading is preserved at the start of content

METADATA PRODUCED:
------------------
Each chunk includes rich metadata for retrieval and reassembly:

**Document Identity**:
  - doc_id: Stable hash of document path (constant across all chunks from same doc)
  - doc_title: Filename without extension
  - doc_path: Relative path from INPUT_DIR_FOR_CHUNKING

**Section Identity** (constant across all chunks from the same section):
  - section_node_id: Stable hash of (doc_id, section_path) - used to reassemble split sections
  - section_title: The heading text of this section
  - section_level: Heading level (1-6)
  - section_path: Breadcrumb list of all parent section titles
  - parent_section_id: section_node_id of parent section (None for top-level)

**Chunk Identity** (unique per chunk):
  - chunk_id: Unique ID for this specific chunk (if section was split, includes split index)
  - chunk_index: 0-based index within section (0 if section wasn't split)
  - chunk_count: Total number of chunks in this section (1 if not split)
  - prev_chunk_id: chunk_id of previous chunk in same section (None if first)
  - next_chunk_id: chunk_id of next chunk in same section (None if last)

**Role Hints**:
  - is_intro_chunk: True if this is an intro span (prose before children), False if body
  - is_small_section: True if section text < MIN_CHARS

HOW METADATA IS USED:
---------------------
1. **Ingestion** (ingest_section_chunks.py):
   - Uses `chunk_id` as the unique node identifier in the vector store
   - Stores all metadata as JSON for later retrieval

2. **Retrieval** (retrieve_and_stitch.py):
   - Uses `section_node_id` to fetch all chunks belonging to the same section
   - Orders chunks by `chunk_index` to reassemble split sections in correct order
   - Uses `section_path` and `doc_title` for product-aware postfiltering
   - Uses `prev_chunk_id`/`next_chunk_id` for navigation (currently unused but available)

3. **Evaluation** (retrieval_evaluation.py):
   - Uses `section_node_id` or `chunk_id` as ground truth identifiers
   - Matches retrieved sections against ground truth by ID

4. **Dataset Generation** (generate_dataset_llm.py):
   - Uses `section_node_id` (or `chunk_id` as fallback) as `gt_section_id` for evaluation

OUTPUT:
-------
For each input markdown file, emits one JSON file containing:
  - A list of chunk dictionaries, each with:
    * "text": The formatted chunk text (with breadcrumb, title, heading, content)
    * "metadata": All metadata fields described above
  - Chunks are ordered by document position (deterministic)

ENVIRONMENT VARIABLES:
---------------------
  INPUT_DIR_FOR_CHUNKING: Directory containing markdown files (default: ./data/md)
  OUTPUT_DIR: Directory to write JSON chunk files (default: ./data/sectionwise_chunks)
  MAX_CHARS: Maximum characters per chunk before splitting (default: 4000)
  SPLIT_OVERLAP_CHARS: Overlap between consecutive splits (default: 180)
  MIN_CHARS: Minimum section size to mark as small (default: 250)
  KEEP_FRONT_MATTER: If true, don't drop front-matter sections (default: false)
"""

from __future__ import annotations
import os, re, json, hashlib, unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

# your catalog of titles to drop
from src.catalog.drop_titles import DROP_TITLES

# optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- env/tunables ----------------
INPUT_DIR_FOR_CHUNKING = os.getenv("INPUT_DIR_FOR_CHUNKING", "./data/md")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data/sectionwise_chunks")
MAX_CHARS = int(os.getenv("MAX_CHARS", "4000"))
SPLIT_OVERLAP_CHARS = int(os.getenv("SPLIT_OVERLAP_CHARS", "180"))
MIN_CHARS = int(os.getenv("MIN_CHARS", "250"))
KEEP_FRONT_MATTER = os.getenv("KEEP_FRONT_MATTER", "false").lower() == "true"

# llamaindex sentence splitter (optional)
_LLAMA_SPLIT = None
try:
    from llama_index.core.node_parser.text.sentence import SentenceSplitter
    _LLAMA_SPLIT = "new"
except Exception:
    _LLAMA_SPLIT = None

# ---------------- utils ----------------
def norm_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()

def norm_title(s: str) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F'\"“”‘’`~^*_=+#<>|{}\[\]()/\\.,:;!?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stable_id(*parts: str) -> str:
    return hashlib.sha1(("||".join(parts)).encode("utf-8")).hexdigest()

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SETEXT_H1 = re.compile(r"^=+\s*$")
SETEXT_H2 = re.compile(r"^-+\s*$")

@dataclass
class SectionNode:
    title: str
    level: int
    start: int
    end: int = -1
    children: List["SectionNode"] = field(default_factory=list)
    parent: Optional["SectionNode"] = None
    intro_span: Optional[Tuple[int, int]] = None
    drop: bool = False
    def path_titles(self) -> List[str]:
        p, out = self, []
        while p and p.title != "_root_":
            out.append(p.title)
            p = p.parent
        return list(reversed(out))

# ---------------- parsing ----------------
def read_and_clean(md_path: Path) -> str:
    raw = md_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n","\n").replace("\r","\n")
    lines = raw.splitlines()

    # drop the top "Extreme" line variants
    if lines and norm_title(lines[0]) in {"extreme","巨 extreme","extreme networks"}:
        lines = lines[1:]

    # light cleanup of footer/page artifacts
    cleaned = []
    for ln in lines:
        # "for version X.Y ..." page liners
        if re.search(r"for version\s+\d+(?:\.\d+)*\s+\d+\s*$", ln, re.I): 
            continue
        if re.fullmatch(r"\d+\s*$", ln.strip()):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def find_headings(md: str) -> List[Tuple[int,int,int,str]]:
    lines = md.splitlines()
    out = []
    char_pos = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADING_RE.match(line)
        if m:
            out.append((i, char_pos, len(m.group(1)), m.group(2).strip()))
            char_pos += len(line) + 1
            i += 1
            continue
        if i+1 < len(lines):
            nxt = lines[i+1]
            if SETEXT_H1.match(nxt) and norm_text(line):
                out.append((i, char_pos, 1, norm_text(line)))
                char_pos += len(line)+1+len(nxt)+1
                i += 2
                continue
            if SETEXT_H2.match(nxt) and norm_text(line):
                out.append((i, char_pos, 2, norm_text(line)))
                char_pos += len(line)+1+len(nxt)+1
                i += 2
                continue
        char_pos += len(line) + 1
        i += 1
    return out

def _header_line_end(md: str, char_start: int) -> int:
    nl = md.find("\n", char_start)
    return (nl+1) if nl != -1 else len(md)

def build_tree(md: str, headings) -> List[SectionNode]:
    root = SectionNode("_root_", 0, 0)
    stack = [root]
    text_len = len(md)

    for _, start, level, title in headings:
        node = SectionNode(title, level, start)
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else root
        node.parent = parent
        parent.children.append(node)
        stack.append(node)

    root.end = text_len

    def close(n: SectionNode):
        for i, c in enumerate(n.children):
            c.end = n.children[i+1].start if i+1 < len(n.children) else (n.end if n.end!=-1 else text_len)
            close(c)
        if n.children:
            hdr_end = _header_line_end(md, n.start)
            first_child_start = n.children[0].start
            if hdr_end < first_child_start:
                n.intro_span = (hdr_end, first_child_start)
    close(root)
    return root.children

# ---------------- drop rules ----------------
_CANON_DROP = {t for t in DROP_TITLES if t}

def mark_dropped_sections(nodes: List[SectionNode]):
    def dfs(n: SectionNode):
        if norm_title(n.title) in _CANON_DROP and not KEEP_FRONT_MATTER:
            n.drop = True
        for c in n.children:
            dfs(c)
    for n in nodes:
        dfs(n)

# strip ToC-like lines inside intro spans (common in PDFs)
_TOC_LINE = re.compile(r"\s+on page\s+\d+\s*$", re.I)
def strip_toc_lines(s: str) -> str:
    lines = s.splitlines()
    keep = []
    for ln in lines:
        if _TOC_LINE.search(ln):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()

# ---------------- splitters ----------------
def _split_with_sentence_fallback(text: str) -> List[str]:
    """Generic splitter used by blockwise to break big paragraphs."""
    if not text or len(text) <= MAX_CHARS:
        return [text] if text else []
    # Try LlamaIndex SentenceSplitter if present
    if _LLAMA_SPLIT:
        try:
            splitter = SentenceSplitter(chunk_size=MAX_CHARS, chunk_overlap=SPLIT_OVERLAP_CHARS)
            return splitter.split_text(text)
        except Exception:
            pass
    # paragraph -> sentence fallback
    paras = re.split(r"\n{2,}", text)
    chunks, cur = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) > MAX_CHARS:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(cur)+len(s)+1 > MAX_CHARS:
                    if cur: chunks.append(cur.strip())
                    cur = s
                else:
                    cur = (cur + " " + s).strip() if cur else s
        else:
            if len(cur)+len(p)+2 > MAX_CHARS:
                if cur: chunks.append(cur.strip())
                cur = p
            else:
                cur = (cur + "\n\n" + p).strip() if cur else p
    if cur: chunks.append(cur.strip())
    # simple overlap
    if SPLIT_OVERLAP_CHARS > 0 and len(chunks) > 1:
        with_ov = []
        for i, ch in enumerate(chunks):
            if i == 0:
                with_ov.append(ch)
            else:
                prev_tail = chunks[i-1][-SPLIT_OVERLAP_CHARS:]
                with_ov.append((prev_tail + ch)[:MAX_CHARS])
        chunks = with_ov
    return chunks

# --- table-aware block splitting ---
_HTML_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")

def _is_md_table_sep(line: str) -> bool:
    line = line.strip()
    return bool(re.match(r"^\s*[:\-| ]+\s*$", line)) and ("-" in line) and ("|" in line)

def _is_md_table_row(line: str) -> bool:
    return "|" in line and bool(re.search(r"\S", line))

def _extract_md_table_blocks(text: str) -> list[tuple[str, str]]:
    """Yield ('table'|'para', content) where markdown tables are atomic, with caption above."""
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        # md table start? (row + sep)
        if i + 1 < len(lines) and _is_md_table_row(lines[i]) and _is_md_table_sep(lines[i+1]):
            start = i
            i += 2
            while i < len(lines) and _is_md_table_row(lines[i]):
                i += 1
            tbl = "\n".join(lines[start:i]).strip()
            # attach single-line caption 'Table n:' if present just above
            cap_idx = start - 1
            caption = None
            while cap_idx >= 0 and not lines[cap_idx].strip():
                cap_idx -= 1
            if cap_idx >= 0 and re.match(r"^\s*table\b.*:\s*", lines[cap_idx], re.I):
                caption = lines[cap_idx].strip()
                tbl = caption + "\n\n" + tbl
            blocks.append(("table", tbl))
            continue

        # accumulate paragraph until blank or next table start
        para_lines = [lines[i]]
        i += 1
        while i < len(lines):
            if i + 1 < len(lines) and _is_md_table_row(lines[i]) and _is_md_table_sep(lines[i+1]):
                break
            if not lines[i].strip():
                para_lines.append(lines[i])
                i += 1
                break
            para_lines.append(lines[i])
            i += 1
        para = "\n".join(para_lines).strip()
        if para:
            blocks.append(("para", para))
    return blocks

def _tokenize_blocks(text: str) -> list[tuple[str, str]]:
    """Split into ('table'|'para', content) across HTML tables + markdown tables."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _HTML_TABLE_RE.finditer(text):
        pre = text[pos:m.start()]
        tbl = m.group(0)
        if pre.strip():
            out.extend(_extract_md_table_blocks(pre))
        # attach caption above html table if present in last 3 lines of pre
        caption = None
        if pre:
            tail = pre.rstrip("\n").splitlines()[-3:]
            for ln in reversed(tail):
                if re.match(r"^\s*table\b.*:\s*", ln, re.I):
                    caption = ln.strip()
                    break
        if caption:
            out.append(("table", caption + "\n\n" + tbl.strip()))
        else:
            out.append(("table", tbl.strip()))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        out.extend(_extract_md_table_blocks(tail))
    return out

def split_text_blockwise(text: str) -> List[str]:
    """Block-aware splitter: tables atomic; paragraphs aggregated to MAX_CHARS with sentence fallback."""
    if not text:
        return []
    blocks = _tokenize_blocks(text)
    if all(kind != "table" for kind, _ in blocks):
        return _split_with_sentence_fallback(text)

    chunks, cur = [], ""
    def _flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
            cur = ""

    for kind, content in blocks:
        if kind == "table":
            _flush()
            # keep table atomic even if > MAX_CHARS
            chunks.append(content.strip())
            continue
        # para: try to pack
        para = content.strip()
        if not para: 
            continue
        if not cur:
            cur = para
        elif len(cur) + 2 + len(para) <= MAX_CHARS:
            cur = f"{cur}\n\n{para}"
        else:
            _flush()
            # para too big -> break further
            if len(para) > MAX_CHARS:
                for piece in _split_with_sentence_fallback(para):
                    if not cur:
                        cur = piece
                    elif len(cur) + 2 + len(piece) <= MAX_CHARS:
                        cur = f"{cur}\n\n{piece}"
                    else:
                        _flush()
                        cur = piece
            else:
                cur = para
    _flush()
    return chunks

# ---------------- chunk assembly ----------------
def extract_span(md: str, span: Optional[Tuple[int,int]]) -> str:
    if not span: return ""
    s,e = span
    return md[s:e].strip()

def visible_path(n: SectionNode) -> List[str]:
    return n.path_titles()

def chunkify(md_path: Path) -> List[Dict[str, Any]]:
    md = read_and_clean(md_path)
    headings = find_headings(md)

    # Document title is the filename without extension
    doc_title = Path(md_path).stem

    doc_id = stable_id(str(Path(md_path).resolve()))
    top_nodes = build_tree(md, headings)
    mark_dropped_sections(top_nodes)

    # ordered list of (node, is_intro, span)
    sections_linear: List[Tuple[SectionNode, bool, Optional[Tuple[int, int]]]] = []

    def add_node_recursive(node: SectionNode):
        """Recursively add section nodes to linear list, handling intro spans and leaf bodies."""
        if node.drop:
            return
        # If section has prose before its first child, add it as an intro chunk
        if node.intro_span:
            sections_linear.append((node, True, node.intro_span))
        # If section has children, recurse into them
        if node.children:
            for child in node.children:
                add_node_recursive(child)
        else:
            # Leaf section: add its body content (everything after the heading)
            body_span = (_header_line_end(md, node.start), node.end)
            sections_linear.append((node, False, body_span))

    # Add sections in natural document order (recursive traversal)
    for top_node in top_nodes:
        add_node_recursive(top_node)

    # Sort sections deterministically by document position (span start, then heading start)
    def _span_start(span: Optional[Tuple[int,int]]) -> int:
        """Extract start position from span tuple."""
        return span[0] if span else -1
    linear_with_pos = [(_span_start(span), node.start, node, is_intro, span) 
                       for node, is_intro, span in sections_linear]
    linear_with_pos.sort(key=lambda x: (x[0], x[1]))  # Sort by span start, then heading start
    sections_linear = [(node, is_intro, span) for _, __, node, is_intro, span in linear_with_pos]

    # Final de-duplication (defensive) after ordering to remove any duplicate spans
    unique_linear: List[Tuple[SectionNode, bool, Optional[Tuple[int, int]]]] = []
    seen_keys = set()
    for node, is_intro, span in sections_linear:
        key = (node.start, node.end, bool(is_intro))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_linear.append((node, is_intro, span))
    sections_linear = unique_linear

    # Helper functions to generate stable IDs for sections
    def node_id(node: SectionNode) -> str:
        """Generate stable section_node_id from doc_id and section path."""
        return stable_id(doc_id, " / ".join(visible_path(node)))

    def parent_id(node: SectionNode) -> Optional[str]:
        """Get section_node_id of parent section, or None if root-level."""
        if node.parent is None or node.parent.title == "_root_":
            return None
        return node_id(node.parent)

    out_chunks: List[Dict[str, Any]] = []

    # Emit chunks in strict document order
    for node, is_intro, span in sections_linear:
        raw_text = extract_span(md, span)
        if not raw_text.strip():
            continue
        # Strip "on page X" lines from intro spans (common PDF artifacts)
        text_for_split = strip_toc_lines(raw_text) if is_intro else raw_text

        # Split section text if needed (tables are kept atomic)
        splits = split_text_blockwise(text_for_split)

        # Generate IDs and paths for this section
        sec_path = visible_path(node)
        sec_node_id = node_id(node)
        parent_sec_id = parent_id(node)

        # Base group ID: section_node_id + role (intro/leaf)
        base_group = stable_id(sec_node_id, "intro" if is_intro else "leaf")

        # compute doc_path safely (handles absolute/relative mismatch)
        try:
            rel_path = Path(md_path).resolve().relative_to(Path(INPUT_DIR_FOR_CHUNKING).resolve())
        except Exception:
            rel_path = Path(os.path.relpath(Path(md_path).resolve(), Path(INPUT_DIR_FOR_CHUNKING).resolve()))

        for chunk_idx, part in enumerate(splits):
            if len(splits) == 1:
                # Single chunk: use base_group as chunk_id, no prev/next
                split_id = base_group
                prev_split_id = None
                next_split_id = None
            else:
                # Multiple chunks: append index to base_group for unique chunk_id
                split_id = stable_id(base_group, str(chunk_idx))
                prev_split_id = stable_id(base_group, str(chunk_idx-1)) if chunk_idx > 0 else None
                next_split_id = stable_id(base_group, str(chunk_idx+1)) if chunk_idx+1 < len(splits) else None

            # Format chunk text with breadcrumb, document title, and section heading
            # Format chunk text with breadcrumb path, document title, and section heading
            breadcrumb = "Path: " + " > ".join(sec_path)
            title_line = f"Document_title: {doc_title}"
            header_line = f"{('#' * node.level)} {node.title}".strip()
            chunk_text = f"{breadcrumb}\n{title_line}\n\ncontent:\n\n{header_line}\n\n{part}".strip()

            # Build metadata dictionary for this chunk
            meta = {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_path": str(rel_path),
                # Section identity (constant across all chunks in a section)
                "section_node_id": sec_node_id,  # Stable ID for entire section (used for reassembly)
                "section_title": node.title,
                "section_level": node.level,
                "section_path": sec_path,
                "parent_section_id": parent_sec_id,
                # chunk identity (unique per chunk)
                "chunk_id": split_id,                       # unique ID for this specific chunk
                # split info (if section was split)
                "chunk_index": chunk_idx,                   # 0-based index within section
                "chunk_count": len(splits),                 # total chunks in this section
                "prev_chunk_id": prev_split_id,             # previous chunk in same section
                "next_chunk_id": next_split_id,             # next chunk in same section
                # role hints
                "is_intro_chunk": bool(is_intro),
                "is_small_section": len(text_for_split) < MIN_CHARS,
            }

            out_chunks.append({"text": chunk_text, "metadata": meta})

    return out_chunks

# ---------------- io ----------------
def write_json(out_path: Path, chunks: List[Dict[str, Any]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # already in doc order; write as a list for easy human inspection
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def main():
    src_root = Path(INPUT_DIR_FOR_CHUNKING).resolve()
    out_root = Path(OUTPUT_DIR).resolve()
    md_files = [p for p in src_root.rglob("*.md") if p.is_file()]

    for md in md_files:
        rel = md.relative_to(src_root)
        out_path = out_root / rel.with_suffix(".json")
        chunks = chunkify(md)
        write_json(out_path, chunks)

    print(f"Done. Wrote chunks for {len(md_files)} files into {out_root}")

if __name__ == "__main__":
    main()
