# ner_extras.py
import re
from typing import List

# capture "quoted phrases" or 'quoted phrases'
_QUOTE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')

# detect version-like strings to DROP (24.1 / 24.1.0 / v24.1.0 / 24r1 / 24 r 1)
_DOTTED_VER_RE = re.compile(r'\b[vV]?\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b')
_RSTYLE_VER_RE = re.compile(r'\b\d{1,3}\s*[rR]\s*\d{1,3}\b')

# very number-y tokens (e.g., "24.1", "2024", "1/2/3"), we’ll usually drop
_MOSTLY_NUM_RE = re.compile(r'^[^A-Za-z]*\d[^A-Za-z]*$')

def _dedup(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq or []:
        k = " ".join(s.lower().split())
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

def _looks_versionish(s: str) -> bool:
    # normalize spacing/case a bit
    t = " ".join((s or "").strip().lower().split())
    return bool(_DOTTED_VER_RE.search(t) or _RSTYLE_VER_RE.search(t))

def extract_quoted_phrases(q: str, max_len: int = 80) -> List[str]:
    phrases = []
    for a, b in _QUOTE_RE.findall(q or ""):
        s = a or b
        s = " ".join(s.split())
        if 0 < len(s) <= max_len:
            phrases.append(s)
    return _dedup(phrases)

def _clean_base_entities(base_entities: List[str] | None) -> List[str]:
    out: List[str] = []
    for e in base_entities or []:
        e = " ".join((e or "").split())
        if not e:
            continue
        # drop version-like strings entirely
        if _looks_versionish(e):
            continue
        # drop pure/mostly numeric tokens (dates, codes, etc.)
        if _MOSTLY_NUM_RE.match(e):
            continue
        out.append(e)
    return _dedup(out)

def augment_entities(query: str, base_entities: List[str] | None) -> List[str]:
    """
    - Put quoted phrases first (most discriminative).
    - Append cleaned base entities (versions removed).
    - No version extraction/aliases at all.
    """
    quotes = extract_quoted_phrases(query)
    base   = _clean_base_entities(base_entities)
    return _dedup(quotes + base)
