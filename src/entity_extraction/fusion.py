# entity_extraction/fusion.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from collections import defaultdict
import unicodedata

from span_utils import iou_tuple

IOU_THR = 0.7
DEFAULT_SOURCE_WEIGHTS = {
    "gazetteer": 0.9,
    "ner": 0.7,
    "chunks": 0.6,
    "bioc": 0.8,
    "unknown": 0.6,
}


def _normalize_text_min(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return " ".join(s.split())


def _noisy_or(ps: List[float]) -> float:
    prod = 1.0
    for p in ps:
        p = max(0.0, min(1.0, p))
        prod *= (1.0 - p)
    return 1.0 - prod


def _map_type(raw_type: Optional[str], tmap: Dict[str, str]) -> Optional[str]:
    if not raw_type:
        return None
    return tmap.get(raw_type.lower(), raw_type)


def fuse_candidates(
    cands: List[Dict[str, Any]],
    type_map: Optional[Dict[str, str]] = None,
    source_weights: Optional[Dict[str, float]] = None,
    iou_thr: float = IOU_THR,
) -> List[Dict[str, Any]]:
    if not cands:
        return []

    tmap = type_map or {}
    sweights = source_weights or DEFAULT_SOURCE_WEIGHTS

    items = []
    for c in cands:
        s = int(c["start_char"])
        e = int(c["end_char"])
        if e <= s:
            continue
        txt = c["text"]
        src = c.get("source", "unknown")
        score = float(c.get("score", 1.0))
        mapped = _map_type(c.get("type"), tmap)
        items.append(
            {
                "text": txt,
                "text_norm": _normalize_text_min(txt),
                "start_char": s,
                "end_char": e,
                "type": mapped,
                "source": src,
                "score": score,
                "meta": c.get("meta", {}),
            }
        )

    items.sort(key=lambda x: (x["text_norm"], x["start_char"], -(x["end_char"] - x["start_char"])))
    clusters: List[List[Dict[str, Any]]] = []

    for it in items:
        placed = False
        for cl in clusters:
            if cl[0]["text_norm"] != it["text_norm"]:
                continue
            if iou_tuple(
                (it["start_char"], it["end_char"]),
                (cl[0]["start_char"], cl[0]["end_char"]),
            ) >= iou_thr or any(
                m["start_char"] == it["start_char"] and m["end_char"] == it["end_char"] for m in cl
            ):
                cl.append(it)
                placed = True
                break
        if not placed:
            clusters.append([it])

    fused: List[Dict[str, Any]] = []
    for cl in clusters:
        can = max(cl, key=lambda x: (x["end_char"] - x["start_char"], x["score"]))
        s, e = can["start_char"], can["end_char"]
        text = can["text"]

        type_scores: Dict[str, List[float]] = defaultdict(list)
        sources = []
        for m in cl:
            t = m["type"]
            w = sweights.get(m["source"], sweights.get("unknown", 0.6))
            if t:
                type_scores[t].append(w * m["score"])
            sources.append(
                {
                    "name": m["source"],
                    "type": t,
                    "score": m["score"],
                    "meta": m.get("meta", {}),
                }
            )

        type_votes = {t: _noisy_or(vs) for t, vs in type_scores.items()}

        aliases = []
        for m in cl:
            if (m["start_char"], m["end_char"]) != (s, e):
                aliases.append(
                    {
                        "text": m["text"],
                        "start_char": m["start_char"],
                        "end_char": m["end_char"],
                        "source": m["source"],
                        "type": m["type"],
                    }
                )

        fused.append(
            {
                "text": text,
                "start_char": s,
                "end_char": e,
                "span_policy": "longest",
                "type_votes": type_votes,
                "sources": sources,
                "aliases": aliases,
            }
        )

    return fused
