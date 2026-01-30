# entity_extraction/candidates/bioc.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import json
import glob
import os
from nltk.tokenize import sent_tokenize
from span_utils import canon, fix_span_indices, find_span_positions

_BIOC_META_KEYS = [
    "concept_source",
    "preferred_term",
    "version",
    "concept_id",
    "evidence_code",
    "provenance",
    "provider",
    "score",
    "nature",
    "concept_form",
]


def load_bioc_index_from_dir(bioc_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for path in glob.glob(os.path.join(bioc_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        articles = doc.get("sibils_article_set") or doc.get("articles") or []
        for article in articles:
            for passage in article.get("passages", []) or []:
                text = passage.get("text") or ""
                if not text:
                    continue
                spans = []
                for ann in passage.get("annotations", []):
                    infons = ann.get("infons", {}) or {}
                    for loc in ann.get("locations", []):
                        start = int(loc.get("offset", 0))
                        length = int(loc.get("length", 0))
                        end = start + length
                        if end <= start or start < 0 or end > len(text):
                            continue
                        meta = {k: infons.get(k) for k in _BIOC_META_KEYS}
                        spans.append(
                            {
                                "start": start,
                                "end": end,
                                "text": text[start:end],
                                **meta,
                            }
                        )
                spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
                index[text] = spans
    return index


def _sentence_offsets(text: str) -> List[Tuple[str, int, int]]:
    try:
        sentences = sent_tokenize(text)
    except LookupError as exc:
        raise RuntimeError(
            "NLTK punkt tokenizer not found. Install it with: "
            "python -m nltk.downloader punkt"
        ) from exc

    out: List[Tuple[str, int, int]] = []
    cursor = 0
    for sent in sentences:
        if not sent:
            continue
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue
        idx = text.find(sent, cursor)
        if idx == -1:
            positions = find_span_positions(text, sent_stripped)
            if positions:
                idx, end = positions[0]
            else:
                continue
        else:
            end = idx + len(sent)
        if sent != sent_stripped:
            lstrip_len = len(sent) - len(sent.lstrip())
            rstrip_len = len(sent) - len(sent.rstrip())
            idx = idx + lstrip_len
            end = end - rstrip_len
        out.append((sent_stripped, idx, end))
        cursor = end
    return out


def load_bioc_sentence_index_from_dir(bioc_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for path in glob.glob(os.path.join(bioc_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        articles = doc.get("sibils_article_set") or doc.get("articles") or []
        for article in articles:
            for passage in article.get("passages", []) or []:
                text = passage.get("text") or ""
                if not text:
                    continue
                sent_offsets = _sentence_offsets(text)
                if not sent_offsets:
                    continue
                spans = []
                for ann in passage.get("annotations", []):
                    infons = ann.get("infons", {}) or {}
                    for loc in ann.get("locations", []):
                        start = int(loc.get("offset", 0))
                        length = int(loc.get("length", 0))
                        end = start + length
                        if end <= start or start < 0 or end > len(text):
                            continue
                        spans.append(
                            {
                                "start": start,
                                "end": end,
                                "text": text[start:end],
                                **{k: infons.get(k) for k in _BIOC_META_KEYS},
                            }
                        )
                if not spans:
                    continue
                for sent_text, s_start, s_end in sent_offsets:
                    sent_spans = []
                    for sp in spans:
                        if sp["start"] < s_start or sp["end"] > s_end:
                            continue
                        sent_spans.append(
                            {
                                "text": sp["text"],
                                "start": sp["start"] - s_start,
                                "end": sp["end"] - s_start,
                                "type": sp.get("type"),
                                "source": "bioc",
                                "meta": {k: sp.get(k) for k in _BIOC_META_KEYS},
                            }
                        )
                    if sent_spans:
                        index.setdefault(sent_text, []).extend(sent_spans)
    return index


def dedupe_bioc_index(bioc_index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    tmp: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}

    def _span_key(sp):
        return (int(sp["start"]), int(sp["end"]), sp.get("text", ""), sp.get("type"))

    for sent_text, spans in bioc_index.items():
        c = canon(sent_text)
        if c not in tmp:
            tmp[c] = (sent_text, [])
        key_text, merged = tmp[c]
        merged.extend(spans)

    deduped: Dict[str, List[Dict[str, Any]]] = {}
    for _, (orig_text, merged) in tmp.items():
        seen = set()
        uniq = []
        for sp in merged:
            k = _span_key(sp)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(sp)
        uniq.sort(key=lambda s: (int(s["start"]), -(int(s["end"]) - int(s["start"]))))
        deduped[orig_text] = uniq
    return deduped


def normalize_bioc_spans(spans: List[Dict[str, Any]], sent_text: str) -> List[Dict[str, Any]]:
    norm_spans = []
    for s in spans:
        if "text" in s and "start" in s and "end" in s:
            if isinstance(s.get("meta"), dict):
                meta = {k: s["meta"].get(k) for k in _BIOC_META_KEYS}
            else:
                meta = {k: s.get(k) for k in _BIOC_META_KEYS}
            assert set(meta.keys()) == set(_BIOC_META_KEYS)
            norm_spans.append(
                {
                    "text": s["text"].strip(),
                    "start_char": int(s["start"]),
                    "end_char": int(s["end"]),
                    "type": s.get("type"),
                    "source": "bioc",
                    "meta": meta,
                }
            )
    if not norm_spans:
        return []
    needs_fix = any(
        (int(sp["start_char"]) < 0)
        or (int(sp["end_char"]) > len(sent_text))
        or (int(sp["end_char"]) <= int(sp["start_char"]))
        for sp in norm_spans
    )
    return fix_span_indices(norm_spans, sent_text) if needs_fix else norm_spans
