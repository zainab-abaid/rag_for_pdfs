# reranker_entity_only.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
import re
import math
import unicodedata

# Optional: product catalog (kept for API compatibility; harmless if absent)
try:
    from catalog.product_names import PRODUCT_NAMES
except Exception:
    PRODUCT_NAMES = []

# ============================ Normalization ===============================

def _canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = s.replace("&", " and ").replace("@", " at ")
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_spaces(s: str) -> str:
    return _canon(s)

def _norm_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _canon(s))

def _token_set_ratio(needle_norm: str, hay_norm: str) -> float:
    ntoks = [t for t in (needle_norm or "").split() if t]
    if not ntoks:
        return 0.0
    htoks = set((hay_norm or "").split())
    found = sum(1 for t in ntoks if t in htoks)
    return 100.0 * found / max(1, len(ntoks))

def _contains_phrase(hay_spaces: str, needle_spaces: str) -> bool:
    if not needle_spaces:
        return False
    pat = re.compile(rf"(?<!\w){re.escape(needle_spaces)}(?!\w)")
    return bool(pat.search(hay_spaces or ""))

def _flatten_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (tuple, list)):
        return " ".join(_flatten_text(e) for e in x)
    if isinstance(x, dict):
        return " ".join(_flatten_text(v) for v in x.values())
    return str(x)

def _items_to_texts_and_scores(
    items: List[Any], orig_scores: List[float] | None
) -> Tuple[List[str], List[float]]:
    texts: List[str] = []
    scores: List[float] = []
    for i, it in enumerate(items):
        texts.append(_flatten_text(it))
        if orig_scores is not None:
            scores.append(float(orig_scores[i] if i < len(orig_scores) else 0.0))
        elif isinstance(it, dict):
            for k in ("orig_score", "score", "store_score"):
                if k in it:
                    try:
                        scores.append(float(it[k] or 0.0))
                        break
                    except Exception:
                        pass
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return texts, scores

def _dedup_norm_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq or []:
        k = _norm_spaces(s)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

def _catalog_norm_set() -> set[str]:
    return {_norm_spaces(s) for s in (PRODUCT_NAMES or [])} | {_norm_alnum(s) for s in (PRODUCT_NAMES or [])}

def _is_product_entity(e: str, catalog_norm: set[str]) -> bool:
    return (_norm_spaces(e) in catalog_norm) or (_norm_alnum(e) in catalog_norm)

# ============================== Fields ====================================

_HDR_RE = re.compile(r"^\s*#{1,3}\s+(.{1,160})", flags=re.M)

def _extract_header(text: str) -> str:
    t = text or ""
    m = _HDR_RE.search(t)
    if m:
        return m.group(1).strip()
    for line in t.splitlines():
        s = line.strip()
        if s:
            return s[:160]
    return ""

def _extract_fields(it: Any) -> Dict[str, str]:
    title = ""
    path = ""
    text = ""
    if isinstance(it, dict):
        title = str(it.get("doc_title", "") or "")
        sp = it.get("section_path", "")
        if isinstance(sp, list):
            path = " > ".join(str(x) for x in sp if x)
        elif isinstance(sp, str):
            path = sp
        text = str(it.get("text", "") or "")
    all_text = _flatten_text(it)
    header = _extract_header(text) if text else ""
    body = text or all_text
    return dict(title=title, path=path, header=header, body=body, all_text=all_text)

def _norm_fields(fd: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    spaces = {k: _norm_spaces(fd.get(k, "")) for k in ("title", "path", "header", "body")}
    alnum  = {k: _norm_alnum(fd.get(k, "")) for k in ("title", "path", "header", "body")}
    return {"spaces": spaces, "alnum": alnum}

def _short(s: str, n: int = 80) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

# =============================== Main =====================================

def rerank_contexts(
    entities: List[str],
    items: List[Any],
    *,
    orig_scores: List[float] | None = None,
    # blend
    orig_weight: float = 0.35,
    # matching
    use_fuzzy: bool = True,
    fuzzy_min: int = 92,
    # weights: product vs generic
    product_exact_bonus: float = 1.60,
    product_fuzzy_bonus: float = 0.60,
    generic_exact_bonus: float = 0.30,
    generic_fuzzy_bonus: float = 0.10,
    # IDF params (kept for API compatibility)
    min_entity_tokens: int = 1,
    min_idf_weight: float = 0.25,   # not used to hard-gate
    min_active_entities: int = 1,   # no hard gate here
    idf_mode: str = "log",          # "log" or "linear"
    # field multipliers
    title_mult: float = 3.0,
    path_mult: float = 2.0,
    header_mult: float = 1.75,
    body_mult: float = 1.0,
    # normalization
    use_minmax_entity_norm: bool = True,
    # debug
    debug: bool = False,
    debug_top: int = 10,
    # telemetry
    return_info: bool = False,
) -> List[int] | tuple[List[int], dict]:

    n = len(items)
    if n == 0:
        order = []
        return (order, {"applied": False, "reason": "no_items"}) if return_info else order

    _, store_scores = _items_to_texts_and_scores(items, orig_scores)
    fields = [_extract_fields(it) for it in items]
    norms = [_norm_fields(fd) for fd in fields]

    uniq_entities = _dedup_norm_order(entities or [])
    if len(uniq_entities) == 0:
        if debug:
            print("[RERANK] No entities; preserving store order")
        order = list(range(n))
        return (order, {"applied": False, "reason": "no_entities"}) if return_info else order

    catalog_norm = _catalog_norm_set()
    ent_is_product = [_is_product_entity(e, catalog_norm) for e in uniq_entities]
    ent_spaces = [_norm_spaces(e) for e in uniq_entities]
    ent_alnum  = [_norm_alnum(e) for e in uniq_entities]

    if debug:
        print(f"[RERANK] Entities ({len(uniq_entities)}): {uniq_entities}")
        print(f"[RERANK] fuzzy_min={fuzzy_min}, orig_weight={orig_weight}")

    FLDS = ("title", "path", "header", "body")
    FMULT = {
        "title": float(title_mult),
        "path": float(path_mult),
        "header": float(header_mult),
        "body": float(body_mult),
    }

    def _field_match_bonus(ej_norm: str, ej_alnum: str, i: int, field: str,
                           exact_b: float, fuzzy_b: float) -> float:
        # exact phrase on spaces
        if ej_norm and _contains_phrase(norms[i]["spaces"][field], ej_norm):
            return exact_b * FMULT[field]
        # exact substring on alnum
        if ej_alnum and ej_alnum in norms[i]["alnum"][field]:
            return exact_b * FMULT[field]
        # fuzzy tokenset on spaces
        if use_fuzzy and ej_norm:
            if _token_set_ratio(ej_norm, norms[i]["spaces"][field]) >= float(fuzzy_min):
                return fuzzy_b * FMULT[field]
        return 0.0

    def entity_hit_score(j: int, i: int) -> float:
        if ent_is_product[j]:
            exact_b = product_exact_bonus
            fuzzy_b = product_fuzzy_bonus
        else:
            exact_b = generic_exact_bonus
            fuzzy_b = generic_fuzzy_bonus

        best = 0.0
        for f in FLDS:
            sc = _field_match_bonus(ent_spaces[j], ent_alnum[j], i, f, exact_b, fuzzy_b)
            if sc > best:
                best = sc
        return best

    # ------------------ DF / True IDF (no floors, no gates) ----------------
    N = n
    E = len(uniq_entities)
    df = [0] * E
    for j in range(E):
        cnt = 0
        for i in range(N):
            if entity_hit_score(j, i) > 0.0:
                cnt += 1
        df[j] = cnt

    weights = [0.0] * E
    if idf_mode == "linear":
        for j in range(E):
            weights[j] = max(0.0, 1.0 - (df[j] / max(1, N)))
    else:
        denom = math.log(N + 1.0)
        for j in range(E):
            weights[j] = (math.log((N + 1.0) / (df[j] + 1.0)) / denom) if denom > 0 else 0.0

    # ------------------------ Entity scoring --------------------------------
    S_raw = [0.0] * N
    per_item_contrib: List[List[Tuple[str, float]]] = [[] for _ in range(N)]

    for i in range(N):
        s = 0.0
        for j in range(E):
            if min_entity_tokens > 1 and len(ent_spaces[j].split()) < min_entity_tokens:
                continue
            hit = entity_hit_score(j, i)
            if hit <= 1e-12:
                continue
            sc = weights[j] * hit
            s += sc
            per_item_contrib[i].append((uniq_entities[j], sc))
        S_raw[i] = s

    # ---------------- Normalization + blending -----------------------------
    if use_minmax_entity_norm:
        ent_min, ent_max = (min(S_raw), max(S_raw)) if S_raw else (0.0, 0.0)
        S_norm = [0.0] * N if ent_max - ent_min <= 1e-12 else [(s - ent_min) / (ent_max - ent_min) for s in S_raw]
    else:
        mx = max(S_raw) if S_raw else 0.0
        S_norm = [0.0] * N if mx <= 1e-12 else [min(1.0, s / mx) for s in S_raw]

    store_min, store_max = (min(store_scores), max(store_scores)) if store_scores else (0.0, 0.0)
    store_norm = [0.0] * N if store_max - store_min <= 1e-12 else [(s - store_min) / (store_max - store_min) for s in store_scores]

    a = max(0.0, min(1.0, float(orig_weight)))
    if max(S_raw) <= 1e-12 and a > 0.35:
        a = 0.35

    final = [a * store_norm[i] + (1.0 - a) * S_norm[i] for i in range(N)]
    order = sorted(range(N), key=lambda i: (final[i], -i), reverse=True)

    # ------------------------------ Debug ----------------------------------
    if debug:
        print(f"[RERANK] DF per entity: {df}")
        print("[RERANK] IDF weight per entity:", [round(w, 3) for w in weights])
        print(f"[RERANK] Entity score min/max: {round(min(S_raw),4)} / {round(max(S_raw),4)}")
        print("[RERANK] First 10 (orig idx): store_norm | S_raw -> S_norm | final")
        for i in range(min(10, N)):
            print(f"  i={i:>3}: {store_norm[i]:.3f} | {S_raw[i]:.4f} -> {S_norm[i]:.3f} | {final[i]:.3f}")

        print("[RERANK] Final order (top 15): (orig_idx) title/header | final")
        for rank, i in enumerate(order[:min(15, N)], 1):
            title = fields[i]["title"] or fields[i]["header"] or fields[i]["path"]
            print(f"  #{rank:>2} (i={i:>3}) {_short(title,60)} | final={final[i]:.3f} S_norm={S_norm[i]:.3f} store={store_norm[i]:.3f}")

        print(f"[RERANK] Detailed field matches for top {min(debug_top, N)} items:")
        for rank, i in enumerate(order[:min(debug_top, N)], 1):
            print(f"  -> Item i={i} rank={rank}")
            for j in range(E):
                if ent_is_product[j]:
                    exact_b = product_exact_bonus
                    fuzzy_b = product_fuzzy_bonus
                else:
                    exact_b = generic_exact_bonus
                    fuzzy_b = generic_fuzzy_bonus

                hits = []
                for f in FLDS:
                    val = 0.0
                    hows = []
                    if ent_spaces[j] and _contains_phrase(norms[i]['spaces'][f], ent_spaces[j]):
                        val = max(val, exact_b * FMULT[f]); hows.append("exact")
                    if ent_alnum[j] and ent_alnum[j] in norms[i]['alnum'][f]:
                        val = max(val, exact_b * FMULT[f]); hows.append("alnum")
                    if use_fuzzy and ent_spaces[j] and _token_set_ratio(ent_spaces[j], norms[i]['spaces'][f]) >= float(fuzzy_min):
                        val = max(val, fuzzy_b * FMULT[f]); hows.append("fuzzy")
                    if val > 0:
                        hits.append(f"{f}:{'+'.join(sorted(set(hows)))}+{val:.2f}")
                if hits:
                    print(f"     ENT '{uniq_entities[j]}': w={weights[j]:.3f} -> " + ", ".join(hits))
            if per_item_contrib[i]:
                contrib_str = " | ".join([f"{name}:{sc:.3f}" for name, sc in per_item_contrib[i]])
                print(f"     contrib: {contrib_str}")
            else:
                print("     contrib: (none)")

        print("[RERANK] Before/After rank changes (first 15 by final rank):")
        orig_order = sorted(range(N), key=lambda i: (store_scores[i], -i), reverse=True)
        orig_pos = {idx: r for r, idx in enumerate(orig_order, 1)}
        for rank, i in enumerate(order[:min(15, N)], 1):
            print(f"  final#{rank:>2} was orig#{orig_pos.get(i,'?'):>2} (i={i})")

    # ------------------------------ Return ---------------------------------
    if return_info:
        info = {
            "applied": True,
            "reason": "applied",
            "active_entities": uniq_entities,
            "active_entity_weights": [round(w, 3) for w in weights],
            "all_entities": uniq_entities,
            "all_entity_weights": [round(w, 3) for w in weights],
            "field_multipliers": dict(title=title_mult, path=path_mult, header=header_mult, body=body_mult),
            "blend_alpha": a,
            "use_minmax_entity_norm": use_minmax_entity_norm,
        }
        return order, info
    return order
