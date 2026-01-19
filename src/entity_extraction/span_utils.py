# entity_extraction/span_utils.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import unicodedata
from statistics import median


_WS = re.compile(r"\s+")
_PUNCT = str.maketrans({
    "“":"\"", "”":"\"", "‘":"'", "’":"'",
    "–":"-", "—":"-", "−":"-", "…":"...",
    "\u00A0":" ", "\u2009":" ", "\u200A":" ", "\u200B":" ",
})


def _ws_tokenize_with_offsets(text: str):
    """
    Whitespace tokenization with character offsets.
    Returns (tokens, token_spans) where token_spans[i] = (start_char, end_char).
    """
    tokens, spans = [], []
    pos = 0
    while True:
        m = _WS.search(text, pos)
        end = m.start() if m else len(text)
        if end > pos:  # non-empty
            tokens.append(text[pos:end])
            spans.append((pos, end))
        if not m:
            break
        pos = m.end()
    return tokens, spans


def _span_iou_char(a, b) -> float:
    s = max(int(a["start_char"]), int(b["start_char"]))
    e = min(int(a["end_char"]), int(b["end_char"]))
    inter = max(0, e - s)
    if inter <= 0:
        return 0.0
    ua = int(a["end_char"]) - int(a["start_char"])
    ub = int(b["end_char"]) - int(b["start_char"])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def iou_tuple(a: tuple[int,int], b: tuple[int,int]) -> float:
    (s1,e1),(s2,e2) = a,b
    inter = max(0, min(e1,e2) - max(s1,s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def merge_spans(primary, additions, iou_thr=0.5):
    merged = primary[:]
    for cand in additions:
        if not any(_span_iou_char(cand, m) >= iou_thr for m in merged):
            merged.append(cand)
    return merged


def overlaps_any(span: tuple[int,int], spans: list[tuple[int,int]], iou_thr: float = 0.5) -> bool:
    s, e = span
    for (s2, e2) in spans:
        if iou_tuple((s, e), (s2, e2)) >= iou_thr:
            return True
    return False


def dedupe_overlaps_longest(
    spans: List[Dict[str, Any]],
    iou_thr: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Greedy de-overlap: prefer longer spans.
    - Sort spans by length (desc).
    - Keep a span if its IoU with any already kept span is < iou_thr.
    - Finally, sort by start_char for deterministic ordering.
    """
    if not spans:
        return []

    # longest first
    ordered = sorted(
        spans,
        key=lambda s: int(s["end_char"]) - int(s["start_char"]),
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    for s in ordered:
        if not kept:
            kept.append(s)
            continue
        if not any(_span_iou_char(s, k) >= iou_thr for k in kept):
            kept.append(s)

    # nice deterministic order in output
    kept.sort(key=lambda s: int(s["start_char"]))
    return kept


def find_span_positions(text: str, span_text: str):
    """
    Find all positions of span_text in text using regex.
    Returns list of (start, end) tuples for all matches.
    """
    raw = span_text.strip()
    if not raw:
        return []

    def _wrap_word(token: str) -> str:
        if token and token[0].isalnum() and token[-1].isalnum():
            return r"\b" + token + r"\b"
        return token

    # 1) Exact (case-insensitive) match
    escaped_span = re.escape(raw)
    matches = [(m.start(), m.end()) for m in re.finditer(escaped_span, text, re.IGNORECASE)]
    if matches:
        return matches

    # 2) Flexible whitespace between tokens
    tokens = raw.split()
    if tokens:
        parts = [_wrap_word(re.escape(t)) for t in tokens]
        pattern = r"\s+".join(parts)
        matches = [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]
        if matches:
            return matches

    return []


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def _snap_to_candidate(span_text: str, candidates: List[Dict[str, Any]] | None) -> Dict[str, Any] | None:
    if not candidates:
        return None
    span_tokens = _tokenize_words(span_text)
    if len(span_tokens) < 2:
        return None

    best = None
    best_score = (-1.0, -1.0, float("inf"))
    for cand in candidates:
        cand_text = cand.get("text", "")
        cand_tokens = _tokenize_words(cand_text)
        if not cand_tokens:
            continue
        overlap = len(set(span_tokens) & set(cand_tokens))
        overlap_ratio = overlap / len(span_tokens)
        if overlap_ratio < 0.6:
            continue
        jaccard = overlap / (len(set(span_tokens)) + len(set(cand_tokens)) - overlap)
        score = (overlap_ratio, jaccard, len(cand_tokens))
        if score > best_score:
            best_score = score
            best = cand
    return best


def fix_span_indices(spans: list, sentence_text: str, candidates: List[Dict[str, Any]] | None = None) -> List[Dict]:
    """
    Fix span indices using regex matching.
    Returns updated spans with correct start_char and end_char.
    """

    fixed_spans = []
    for span in spans:
        if isinstance(span, str):
            span = {"text": span}
        if not isinstance(span, dict):
            continue
        span_text = span.get("text", "").strip()
        if not span_text:
            continue

        positions = find_span_positions(sentence_text, span_text)
        if positions:
            # Use the first available position
            start, end = positions[0]
            fixed_span = dict(span)
            fixed_span["start_char"] = start
            fixed_span["end_char"] = end
            fixed_span["text"] = sentence_text[start:end]  # Use actual text from sentence
            fixed_spans.append(fixed_span)
        else:
            snapped = _snap_to_candidate(span_text, candidates)
            if snapped:
                fixed_span = dict(span)
                fixed_span["start_char"] = int(snapped["start_char"])
                fixed_span["end_char"] = int(snapped["end_char"])
                fixed_span["text"] = sentence_text[fixed_span["start_char"]:fixed_span["end_char"]]
                fixed_spans.append(fixed_span)
            else:
                # Span text not found in sentence - log warning but keep original
                print(f"WARNING: Could not find span '{span_text}' in sentence: {sentence_text}")
                fixed_span = dict(span)
                fixed_span["start_char"] = 0
                fixed_span["end_char"] = 0
                fixed_span["text"] = span_text
                fixed_spans.append(fixed_span)

    return fixed_spans


def canon(s: str) -> str:
    # same canonicalization as for gold: normalize, collapse space, lowercase
    s = unicodedata.normalize("NFKC", s).translate(_PUNCT)
    s = _WS.sub(" ", s).strip().lower()
    return s


def consensus_merge_by_type(spans_list: List[List[Dict[str,Any]]],
                             iou_thr: float = 0.5) -> List[Dict[str,Any]]:
    """
    Merge many accepted-span lists (from passes or parallel samples) into one.
    Group by TYPE, then greedy IoU clustering. Boundaries = median of members.
    """
    pool = []
    for spans in spans_list:
        for s in spans or []:
            if "type" in s and s.get("type"):
                pool.append({**s, "confidence": float(s.get("confidence", 1.0))})

    # simple greedy clustering per type
    out = []
    used = [False]*len(pool)
    for i, si in enumerate(pool):
        if used[i]:
            continue
        group = [si]; used[i] = True
        for j, sj in enumerate(pool):
            if used[j] or sj.get("type") != si.get("type"):
                continue
            if _span_iou_char(si, sj) >= iou_thr:
                used[j] = True
                group.append(sj)
        starts = sorted(int(g["start_char"]) for g in group)
        ends   = sorted(int(g["end_char"]) for g in group)
        merged = {
            "text": group[0]["text"],  # optional: can slice original sentence later if needed
            "type": group[0]["type"],
            "start_char": int(median(starts)),
            "end_char":   int(median(ends)),
            "confidence": sum(g.get("confidence", 1.0) for g in group)/len(group),
            "votes": len(group)
        }
        out.append(merged)
    return out
