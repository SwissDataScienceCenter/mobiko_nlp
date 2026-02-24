# entity_extraction/candidates/bioc.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import json
import glob
import os
import re
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

_BIOC_EXCLUDED_SENTENCE_FIELDS = {
    "title",
    "keywords",
    "affiliations",
    "section_title",
    "table_caption",
    "table_value",
    "fig_caption",
    "references"
}


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
                                "start_char": start,
                                "end_char": end,
                                "text": text[start:end],
                                **meta,
                            }
                        )
                spans.sort(key=lambda s: (s["start_char"], -(s["end_char"] - s["start_char"])))
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


def _norm_section_title(title: str) -> str:
    t = (title or "").casefold()
    t = t.replace("&", " and ")
    t = re.sub(r"[-–—_/]+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _is_included_section_title(title: str) -> bool:
    norm = _norm_section_title(title)
    if not norm:
        return False

    # Exclude methods/methodology sections even if mixed with other words.
    if "methodolog" in norm or "materials and method" in norm or re.search(r"\bmethods?\b", norm):
        return False

    return any(k in norm for k in ("introduction", "results", "discussion", "conclusion"))


def _iter_section_content_ids(node: Any) -> List[str]:
    if not isinstance(node, dict):
        return []
    out: List[str] = []
    node_id = node.get("id")
    if isinstance(node_id, str) and node_id:
        out.append(node_id)
    for child in node.get("contents", []) or []:
        if not isinstance(child, dict):
            continue
        child_id = child.get("id")
        if isinstance(child_id, str) and child_id:
            out.append(child_id)
        if child.get("contents"):
            out.extend(_iter_section_content_ids(child))
    return out


def _collect_allowed_content_ids_from_document(article: Dict[str, Any]) -> Tuple[set[str], set[str], bool]:
    doc = article.get("document", {}) or {}
    body_sections = doc.get("body_sections", [])
    if not isinstance(body_sections, list) or not body_sections:
        return set(), set(), False

    allowed_content_ids: set[str] = set()
    allowed_section_ids: set[str] = set()

    for sec in body_sections:
        if not isinstance(sec, dict):
            continue
        sec_tag = str(sec.get("tag") or "").strip().lower()
        sec_title = str(sec.get("title") or "")
        include = sec_tag == "abstract" or _is_included_section_title(sec_title)
        if not include:
            continue

        sec_id = sec.get("id")
        if isinstance(sec_id, str) and sec_id:
            allowed_section_ids.add(sec_id)
        for cid in _iter_section_content_ids(sec):
            allowed_content_ids.add(cid)

    return allowed_content_ids, allowed_section_ids, True


def load_bioc_sentence_index_from_dir(bioc_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for path in glob.glob(os.path.join(bioc_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        articles = doc.get("sibils_article_set") or doc.get("articles") or []
        for article in articles:
            if article.get("sentences"):
                allowed_content_ids, allowed_section_ids, has_body_section_metadata = (
                    _collect_allowed_content_ids_from_document(article)
                )

                def _is_allowed_sentence_record(s: Dict[str, Any]) -> bool:
                    fld = str(s.get("field") or "").strip().lower()
                    if fld in _BIOC_EXCLUDED_SENTENCE_FIELDS:
                        return False
                    if fld == "abstract":
                        return True
                    if fld != "text":
                        return False

                    # Fallback for formats without document/body_sections metadata.
                    if not has_body_section_metadata:
                        return True

                    content_id = str(s.get("content_id") or "").strip()
                    if not content_id:
                        return False
                    if content_id in allowed_content_ids:
                        return True
                    return any(
                        content_id == sec_id or content_id.startswith(sec_id + ".")
                        for sec_id in allowed_section_ids
                    )

                sent_by_key: Dict[Tuple[str, int], str] = {}
                for s in article.get("sentences", []):
                    if not _is_allowed_sentence_record(s):
                        continue
                    fld = s.get("field") or ""
                    num = int(s.get("sentence_number", 0))
                    txt = s.get("sentence") or ""
                    if not txt:
                        continue
                    sent_by_key[(fld, num)] = txt

                ann_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
                for a in article.get("annotations", []):
                    fld = a.get("field") or ""
                    num = int(a.get("sentence_number", 0))
                    ann_by_key.setdefault((fld, num), []).append(a)

                for key, text in sent_by_key.items():
                    anns = ann_by_key.get(key, [])
                    if not anns:
                        continue
                    spans = []
                    for a in anns:
                        start = int(a.get("start_index", 0))
                        end = int(a.get("end_index", 0))
                        if end <= start or start < 0 or end > len(text):
                            continue
                        spans.append(
                            {
                                "text": text[start:end],
                                "start_char": start,
                                "end_char": end,
                                "type": a.get("type"),
                                "source": "bioc",
                                "meta": {k: a.get(k) for k in _BIOC_META_KEYS},
                            }
                        )
                    if spans:
                        index.setdefault(text, []).extend(spans)
                continue

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
                                "start_char": start,
                                "end_char": end,
                                "text": text[start:end],
                                "type": infons.get("type"),
                                **{k: infons.get(k) for k in _BIOC_META_KEYS},
                            }
                        )
                if not spans:
                    continue
                for sent_text, s_start, s_end in sent_offsets:
                    sent_spans = []
                    for sp in spans:
                        if sp["start_char"] < s_start or sp["end_char"] > s_end:
                            continue
                        sent_spans.append(
                            {
                                "text": sp["text"],
                                "start_char": sp["start_char"] - s_start,
                                "end_char": sp["end_char"] - s_start,
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
        return (int(sp["start_char"]), int(sp["end_char"]), sp.get("text", ""), sp.get("type"))

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
        uniq.sort(key=lambda s: (int(s["start_char"]), -(int(s["end_char"]) - int(s["start_char"]))))
        deduped[orig_text] = uniq
    return deduped


def normalize_bioc_spans(spans: List[Dict[str, Any]], sent_text: str) -> List[Dict[str, Any]]:
    norm_spans = []
    for s in spans:
        if "text" in s and "start_char" in s and "end_char" in s:
            if isinstance(s.get("meta"), dict):
                meta = {k: s["meta"].get(k) for k in _BIOC_META_KEYS}
            else:
                meta = {k: s.get(k) for k in _BIOC_META_KEYS}
            assert set(meta.keys()) == set(_BIOC_META_KEYS)
            norm_spans.append(
                {
                    "text": s["text"].strip(),
                    "start_char": int(s["start_char"]),
                    "end_char": int(s["end_char"]),
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
