# entity_extraction/candidates/bioc.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import json
import glob
import os
from span_utils import canon, fix_span_indices


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
                        spans.append(
                            {
                                "start": start,
                                "end": end,
                                "text": text[start:end],
                                "source": infons.get("concept_source"),
                                "concept_id": infons.get("concept_id"),
                                "preferred_term": infons.get("preferred_term"),
                            }
                        )
                spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
                index[text] = spans
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
            norm_spans.append(
                {
                    "text": s["text"].strip(),
                    "start_char": int(s["start"]),
                    "end_char": int(s["end"]),
                    "type": s.get("type"),
                    "source": "bioc",
                }
            )
    return fix_span_indices(norm_spans, sent_text)
