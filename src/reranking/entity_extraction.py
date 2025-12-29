"""
Pure spaCy entity extraction with:
- Custom tokenizer that splits letter<->digit boundaries and hyphens.
- PhraseMatcher over your product catalog to capture CANONICAL product names first.
- Then standard spaCy NER (plus optional noun-chunk fallback).
- Works on str or dict-like (searches metadata/content; else all string fields).
- Returns a list[str]: products first (canonical), then other entities; deduped.

Usage:
    from app.scripts.entity_extraction_with_spacy import extract_entities_spacy
    ents = extract_entities_spacy(query_text)  # ['AP360i', 'Aerohive Switch', '...']

Assumes you have a module `product_names.py` somewhere importable that defines:
    PRODUCT_NAMES = ["AP360i", "AP130", "Hive Switch", ...]
"""

from __future__ import annotations
import re
from typing import Iterable, List, Optional, Set, Any, Dict

import spacy
from spacy.util import compile_infix_regex
from spacy.matcher import PhraseMatcher


# ------------- Catalog import (canonical product names) ----------------
try:
    from src.catalog.product_names import product_names as PRODUCT_NAMES
except Exception:
    PRODUCT_NAMES = []


# ------------- Runtime caches so we don't rebuild every call -----------
_NLP_CACHE: Dict[str, spacy.language.Language] = {}
_MATCHER_CACHE: Dict[str, PhraseMatcher] = {}


# ------------- Tokenizer: split letter<->digit and hyphens -------------
def _make_alnum_split_tokenizer(nlp: spacy.language.Language) -> spacy.tokenizer.Tokenizer:
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    infixes = list(nlp.Defaults.infixes)
    # split A|1 and 1|A boundaries; split hyphens between word chars
    infixes += [
        r"(?<=[A-Za-z])(?=\d)",    # letter | digit
        r"(?<=\d)(?=[A-Za-z])",    # digit  | letter
        r"(?<=\w)-(?=\w)",         # word   | - | word
    ]
    infix_re = compile_infix_regex(tuple(infixes))

    return spacy.tokenizer.Tokenizer(
        nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.Defaults.token_match,
    )


# ------------- PhraseMatcher over canonical product names --------------
def _build_product_matcher(nlp: spacy.language.Language) -> PhraseMatcher:
    key = f"{id(nlp)}:product_matcher"
    if key in _MATCHER_CACHE:
        return _MATCHER_CACHE[key]
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # helper: also add a space-separated variant for "AP360i" -> "AP 360 i"
    def _spaced_variants(name: str) -> List[str]:
        # Split on transitions letter<->digit to create "AP 360 i" style
        parts: List[str] = []
        buff = ""
        last_digit = None
        for ch in name:
            if ch.isalnum():
                is_digit = ch.isdigit()
                if last_digit is None:
                    buff = ch
                elif is_digit == last_digit:
                    buff += ch
                else:
                    parts.append(buff)
                    buff = ch
                last_digit = is_digit
            elif ch in "-_/" and buff:
                parts.append(buff)
                buff = ""
                last_digit = None
            else:
                # skip other punctuation
                pass
        if buff:
            parts.append(buff)
        variant = " ".join(parts) if len(parts) > 1 else None
        return [v for v in [name, variant] if v]

    # Add canonical names + their spaced variants as patterns.
    for canon in PRODUCT_NAMES or []:
        for variant in _spaced_variants(canon):
            matcher.add(canon, [nlp.make_doc(variant)])

    _MATCHER_CACHE[key] = matcher
    return matcher


# ------------- Load spaCy pipeline with custom tokenizer ----------------
def _load_nlp(model: str) -> spacy.language.Language:
    if model in _NLP_CACHE:
        return _NLP_CACHE[model]
    nlp = spacy.load(model)
    # swap tokenizer to the alnum-splitting one
    nlp.tokenizer = _make_alnum_split_tokenizer(nlp)
    _NLP_CACHE[model] = nlp
    return nlp


# ------------- Utilities ------------------------------------------------
def _gather_text(obj: Any) -> str:
    """
    Robustly turn a string OR dict-like OR list/tuple into a single text blob.
    - If dict has 'metadata' and/or 'content', prefer those (recursively).
    - Else, walk all values and collect strings.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return " ".join(_gather_text(x) for x in obj)
    if isinstance(obj, dict):
        chunks: List[str] = []
        if "metadata" in obj:
            chunks.append(_gather_text(obj.get("metadata")))
        if "content" in obj:
            chunks.append(_gather_text(obj.get("content")))
        if chunks:
            return " ".join(chunks)
        # fall back to all stringy values
        for v in obj.values():
            s = _gather_text(v)
            if s:
                chunks.append(s)
        return " ".join(chunks)
    # unknown types
    return str(obj)


def _dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in items:
        k = " ".join((s or "").split()).lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


# ------------- Public API ----------------------------------------------
def extract_entities_spacy(
    text_or_obj: Any,
    *,
    model: str = "en_core_web_md",
    ent_types: Optional[Set[str]] = None,
    include_products_first: bool = True,
    fallback_noun_chunks: bool = True,
) -> List[str]:
    """
    Return deduped entities as a list[str].
    - Product canonical names (PhraseMatcher hits) appear first (if include_products_first=True).
    - Then spaCy NER entities (optionally filtered by ent_types).
    - If nothing found and fallback_noun_chunks=True, append noun chunks.
    """
    nlp = _load_nlp(model)
    matcher = _build_product_matcher(nlp)

    text = _gather_text(text_or_obj)
    doc = nlp(text)

    out: List[str] = []
    seen = set()

    # 1) Product matches (canonical labels)
    if include_products_first and matcher:
        matches = matcher(doc)
        for match_id, start, end in matches:
            canon = nlp.vocab.strings[match_id]  # canonical product name
            k = canon.lower()
            if k not in seen:
                seen.add(k)
                out.append(canon)

    # 2) General NER
    for ent in doc.ents:
        if ent_types and ent.label_ not in ent_types:
            continue
        s = " ".join(ent.text.strip().split())
        k = s.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s)

    # 3) Noun-chunk fallback (only if we found nothing)
    if fallback_noun_chunks and len(out) == 0:
        for nc in doc.noun_chunks:
            s = " ".join(nc.text.strip().split())
            if len(s) < 3:  # tiny filter
                continue
            k = s.lower()
            if k and k not in seen:
                seen.add(k)
                out.append(s)

    return _dedup_preserve_order(out)
