#!/usr/bin/env python3
"""
Product-aware postfilter (soft|hard|none).

- Detect a canonical product in the query (using your extractor + catalog).
- If no product is found -> return items unchanged.
- If product found:
  * none  -> unchanged
  * soft  -> stable-partition: product hits first (original order), then the rest
  * hard  -> keep only product hits; if none, fall back to original list

Env (optional):
  POSTFILTER_MODE=none|soft|hard   (default: none)
  POSTFILTER_FIELDS=doc_title,section_path,text  (comma list; default as shown)

Public API:
  apply_postfilter(items, query, mode=None, fields=None) -> (items2, info)
    - items: list of dicts with keys like 'doc_title', 'section_path', 'text' (your stitched blocks)
    - query: str
    - mode: override env (None -> read env)
    - fields: override env as list[str]; any of {"doc_title","section_path","text","meta"}

  get_postfilter_mode() -> str
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import os
import re

# 1) entity extractor
from src.reranking.entity_extraction import extract_entities_spacy

# 2) canonical catalog
try:
    from src.catalog.product_names import product_names as PRODUCT_NAMES
except Exception:
    print("Warning: could not import product_names from src.catalog.product_names; product detection disabled.")
    PRODUCT_NAMES = []

# ---------------- utils ----------------
def _norm_spaces(s: str) -> str:
    return " ".join((s or "").lower().split())

_ALNUM_RE = re.compile(r"[^a-z0-9]+")
def _norm_alnum(s: str) -> str:
    return _ALNUM_RE.sub("", (s or "").lower())

def _is_catalog_product(name: str) -> bool:
    if not name:
        return False
    n = _norm_alnum(name)
    for p in PRODUCT_NAMES or []:
        if _norm_alnum(p) == n:
            return True
    return False

def _first_product_from_query(query: str) -> Optional[str]:
    """Return first entity that is a catalog product (canonical)."""
    try:
        ents = extract_entities_spacy(query) or []
    except Exception:
        ents = []
    for e in ents:
        if _is_catalog_product(e):
            return e
    return None

def _string_contains_product(val: str, product: str) -> bool:
    """Robust contains check: case-insensitive + alnum-substring."""
    if not val or not product:
        return False
    v = val.lower()
    p = product.lower()
    if p in v:
        return True
    return _norm_alnum(product) in _norm_alnum(val)

def _item_field_texts(item: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Extract (doc_title, section_path_joined, text, meta) from an assembled block.
    """
    meta = item.get("meta") or {}
    title = item.get("doc_title") or meta.get("doc_title") or ""
    sp = item.get("section_path") or meta.get("section_path") or []
    if isinstance(sp, list):
        path = " > ".join([str(x) for x in sp if x is not None])
    else:
        path = str(sp or "")
    body = item.get("text") or ""
    return title, path, body, meta

def _item_mentions_product(item: Dict[str, Any], product: str, fields: List[str]) -> bool:
    title, path, body, meta = _item_field_texts(item)
    for f in fields:
        if f == "doc_title" and _string_contains_product(title, product):
            return True
        if f == "section_path" and _string_contains_product(path, product):
            return True
        if f == "text" and _string_contains_product(body, product):
            return True
        if f == "meta" and any(_string_contains_product(str(v), product) for v in (meta or {}).values()):
            return True
    return False

def _fields_from_env(default: str = "doc_title,section_path,text") -> List[str]:
    val = os.getenv("POSTFILTER_FIELDS", default) or default
    return [s.strip() for s in val.split(",") if s.strip()]

def get_postfilter_mode() -> str:
    return (os.getenv("POSTFILTER_MODE") or "none").strip().lower()

# --------------- main API ---------------
def apply_postfilter(
    items: List[Dict[str, Any]],
    query: str,
    *,
    mode: Optional[str] = None,
    fields: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply product-aware postfilter. Returns (items2, info).
    info includes: product, mode, fields, num_hits, total, applied(bool).
    """
    total = len(items)
    mode_eff = (mode or get_postfilter_mode() or "none").lower()
    fields_eff = fields or _fields_from_env()

    # detect product
    product = _first_product_from_query(query)
    if not product or mode_eff == "none" or total == 0:
        return items, {
            "applied": False,
            "reason": "no_product" if not product else "mode_none",
            "product": product or "",
            "mode": mode_eff,
            "fields": fields_eff,
            "num_hits": 0,
            "total": total,
        }

    # partition by product mention (stable)
    hits: List[Dict[str, Any]] = []
    miss: List[Dict[str, Any]] = []
    for it in items:
        (hits if _item_mentions_product(it, product, fields_eff) else miss).append(it)

    num_hits = len(hits)

    if mode_eff == "soft":
        out = hits + miss  # keep original relative order within groups
        return out, {
            "applied": True,
            "reason": "soft_partition",
            "product": product,
            "mode": mode_eff,
            "fields": fields_eff,
            "num_hits": num_hits,
            "total": total,
        }

    if mode_eff == "hard":
        if num_hits > 0:
            out = hits
            return out, {
                "applied": True,
                "reason": "hard_filter",
                "product": product,
                "mode": mode_eff,
                "fields": fields_eff,
                "num_hits": num_hits,
                "total": total,
            }
        # fallback: no hits -> unchanged
        return items, {
            "applied": False,
            "reason": "no_hits_hard_fallback",
            "product": product,
            "mode": mode_eff,
            "fields": fields_eff,
            "num_hits": 0,
            "total": total,
        }

    # unknown mode -> passthrough
    return items, {
        "applied": False,
        "reason": "unknown_mode",
        "product": product,
        "mode": mode_eff,
        "fields": fields_eff,
        "num_hits": num_hits,
        "total": total,
    }
