import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


NONE_LABEL = "__NONE__"
UNKNOWN_LABEL = "__UNKNOWN__"
COLLAPSED_ENTITY_LABEL_MAP = {
    "ABIOTIC COLLECTIVE ENTITY": "ABIOTIC ENTITY",
    "BIOTIC COLLECTIVE ENTITY": "BIOTIC ENTITY",
}
DEFAULT_CANDIDATE_TYPE_MAP = {
    "biomes": "ABIOTIC ENTITY",
    "biota": "BIOTIC ENTITY",
    "mountains": "BIOTIC ENTITY",
    "mountainrange": "BIOTIC ENTITY",
    "geography": "SPATIAL ENTITY",
    "env_feature": "ABIOTIC PROPERTY",
    "population": "BIOTIC COLLECTIVE ENTITY",
    "taxon": "BIOTIC ENTITY",
    "location": "SPACIAL ENTITY",
    "habitat": "BIOTIC PROPERTY",
    "threat": "ANTHROPOGENIC PROCESS",
}
TOP_LEVEL_LABEL_PREFIXES = (
    "ABIOTIC",
    "BIOTIC",
    "ANTHROPOGENIC",
    "SPATIAL",
    "TEMPORAL",
    "CONCEPT",
)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def normalize_whitespace_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def load_docs(path: str) -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            doc_id = rec.get("doc_id")
            if not doc_id:
                raise ValueError(f"{path}:{line_no} missing doc_id")
            if doc_id in docs:
                raise ValueError(f"{path}:{line_no} duplicate doc_id={doc_id!r}")
            docs[doc_id] = rec
    return docs


def _get_pred_final_spans(sentence: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns (present, final_spans). present=False if no final_spans key is available.
    Supports sentence['llm']['final_spans'] and sentence['final_spans'].
    """
    if "final_spans" in sentence:
        v = sentence.get("final_spans")
        return True, (v or [])
    llm = sentence.get("llm") or {}
    if isinstance(llm, dict) and "final_spans" in llm:
        v = llm.get("final_spans")
        return True, (v or [])
    return False, []


def _get_sentence_candidates(sentence: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns (present, candidates). present=False if no candidates key is available.
    Supports sentence['candidates'] and sentence['llm']['candidates'].
    """
    if "candidates" in sentence:
        v = sentence.get("candidates")
        return True, (v or [])
    llm = sentence.get("llm") or {}
    if isinstance(llm, dict) and "candidates" in llm:
        v = llm.get("candidates")
        return True, (v or [])
    return False, []


def canonicalize_label(
    label: str,
    *,
    collapse_entity_variants: bool,
    collapse_to_top_level: bool,
) -> str:
    label = str(label)
    if collapse_entity_variants:
        label = COLLAPSED_ENTITY_LABEL_MAP.get(label, label)
    if collapse_to_top_level:
        for prefix in TOP_LEVEL_LABEL_PREFIXES:
            if label == prefix or label.startswith(prefix + " "):
                return prefix
    return label


def canonicalize_candidate_label(label: str, candidate_type_map: Dict[str, str]) -> str:
    mapped = candidate_type_map.get(str(label).strip().lower(), str(label))
    return str(mapped)


def normalize_gold_span(
    span: Dict[str, Any],
    *,
    context: str,
    collapse_entity_variants: bool = False,
    collapse_to_top_level: bool = False,
) -> Dict[str, Any]:
    if "start_char" not in span or "end_char" not in span:
        raise ValueError(f"{context}: gold span missing offsets: {span}")
    label = span.get("type")
    if not label:
        raise ValueError(f"{context}: gold span missing type: {span}")
    out = {
        "start_char": int(span["start_char"]),
        "end_char": int(span["end_char"]),
        "type": canonicalize_label(
            label,
            collapse_entity_variants=collapse_entity_variants,
            collapse_to_top_level=collapse_to_top_level,
        ),
        "text": span.get("text"),
    }
    if out["end_char"] <= out["start_char"]:
        raise ValueError(f"{context}: invalid gold span offsets: {span}")
    return out


def normalize_pred_span(
    span: Dict[str, Any],
    *,
    context: str,
    collapse_entity_variants: bool = False,
    collapse_to_top_level: bool = False,
) -> Dict[str, Any]:
    if "start_char" not in span or "end_char" not in span:
        raise ValueError(f"{context}: pred span missing offsets: {span}")
    label = span.get("type") or UNKNOWN_LABEL
    out = {
        "start_char": int(span["start_char"]),
        "end_char": int(span["end_char"]),
        "type": canonicalize_label(
            label,
            collapse_entity_variants=collapse_entity_variants,
            collapse_to_top_level=collapse_to_top_level,
        ),
        "text": span.get("text"),
    }
    if out["end_char"] <= out["start_char"]:
        raise ValueError(f"{context}: invalid pred span offsets: {span}")
    return out


def _prediction_span_key(span: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(span["start_char"]), int(span["end_char"]), str(span["type"]))


def _extract_pred_spans(
    sentence: Dict[str, Any],
    *,
    prediction_field: str,
    candidate_sources: Optional[Set[str]],
    candidate_type_map: Dict[str, str],
) -> Tuple[bool, List[Dict[str, Any]]]:
    if prediction_field == "final_spans":
        return _get_pred_final_spans(sentence)

    present, raw_candidates = _get_sentence_candidates(sentence)
    if not present:
        return False, []

    pred_spans: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, str]] = set()
    for cand in raw_candidates:
        if not isinstance(cand, dict):
            continue

        source_entries: List[Dict[str, Any]] = []
        raw_sources = cand.get("sources")
        if isinstance(raw_sources, list):
            source_entries.extend(s for s in raw_sources if isinstance(s, dict))

        if not source_entries and cand.get("type"):
            source_entries.append(
                {
                    "name": cand.get("source"),
                    "type": cand.get("type"),
                }
            )

        for src in source_entries:
            src_name = str(src.get("name") or "").strip()
            if candidate_sources and src_name not in candidate_sources:
                continue
            src_type = src.get("type")
            if not src_type:
                continue
            try:
                pred_span = {
                    "start_char": int(cand["start_char"]),
                    "end_char": int(cand["end_char"]),
                    "type": canonicalize_candidate_label(src_type, candidate_type_map),
                    "text": cand.get("text"),
                }
            except (KeyError, TypeError, ValueError):
                continue

            key = _prediction_span_key(pred_span)
            if key in seen:
                continue
            seen.add(key)
            pred_spans.append(pred_span)

    return True, pred_spans


def is_gold_sentence_annotated(sentence: Dict[str, Any]) -> bool:
    """
    Treat only sentences with a non-empty gold spans list as annotated.

    This matches the current partial-annotation workflow where unfinished
    sentences are represented as spans=[] and should be skipped.
    """
    spans = sentence.get("spans")
    return isinstance(spans, list) and len(spans) > 0


def align_sentences_by_text(
    gold_doc_id: str,
    gold_sents: List[Dict[str, Any]],
    model_sents: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """
    Align gold sentences to model sentences by exact text, in order.
    Returns:
      - pairs: list of (gold_idx, model_idx)
      - model_only_extras: skipped model sentences encountered during scan, plus trailing extras
    """
    pairs: List[Tuple[int, int]] = []
    extras: List[Dict[str, Any]] = []
    j = 0
    for gi, gs in enumerate(gold_sents):
        target = gs.get("text", "")
        found = False
        while j < len(model_sents):
            mt = model_sents[j].get("text", "")
            if mt == target:
                pairs.append((gi, j))
                j += 1
                found = True
                break
            extras.append(
                {
                    "doc_id": gold_doc_id,
                    "model_sent_idx": j,
                    "text": mt,
                    "reason": "model_only_extra_before_aligned_match",
                }
            )
            j += 1
        if not found:
            snippet = normalize_whitespace_one_line(target)[:200]
            raise ValueError(
                f"Alignment failed for doc_id={gold_doc_id!r}, gold_sent_idx={gi}: "
                f"could not find remaining model sentence with text={snippet!r}"
            )

    while j < len(model_sents):
        extras.append(
            {
                "doc_id": gold_doc_id,
                "model_sent_idx": j,
                "text": model_sents[j].get("text", ""),
                "reason": "model_only_extra_trailing",
            }
        )
        j += 1

    return pairs, extras


def span_key_full(span: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(span["start_char"]), int(span["end_char"]), str(span["type"]))


def boundary_key(span: Dict[str, Any]) -> Tuple[int, int]:
    return (int(span["start_char"]), int(span["end_char"]))


def top_level_label(label: str) -> str:
    label = str(label)
    for prefix in TOP_LEVEL_LABEL_PREFIXES:
        if label == prefix or label.startswith(prefix + " "):
            return prefix
    return label


def _is_trivial_gap(text: str) -> bool:
    return not any(ch.isalnum() for ch in (text or ""))


def load_spacy_model(model_name: str):
    try:
        import spacy
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "spaCy is not installed in this environment. Install requirements and a parser-enabled model first."
        ) from e
    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load spaCy model {model_name!r}. Install it with: python -m spacy download {model_name}"
        ) from e

    if "parser" not in getattr(nlp, "pipe_names", []):
        raise RuntimeError(f"spaCy model {model_name!r} does not have the parser enabled.")
    return nlp


def span_head_token(doc, start_char: int, end_char: int):
    span = doc.char_span(start_char, end_char, alignment_mode="expand")
    if span is None:
        tokens = [t for t in doc if t.idx < end_char and (t.idx + len(t)) > start_char]
        if not tokens:
            return None
        span = doc[tokens[0].i : tokens[-1].i + 1]
    return span.root


def build_span_heads(doc, spans: List[Dict[str, Any]]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    heads: List[Any] = []
    failures: List[Dict[str, Any]] = []
    for idx, span in enumerate(spans):
        head = span_head_token(doc, int(span["start_char"]), int(span["end_char"]))
        heads.append(head)
        if head is None:
            failures.append(
                {
                    "span_idx": idx,
                    "span": span,
                    "reason": "no_overlapping_tokens_for_span",
                }
            )
    return heads, failures


def _find_split_gold_match(
    gold_span: Dict[str, Any],
    pred_spans: List[Dict[str, Any]],
    matched_pred_idxs: set[int],
    sentence_text: str,
) -> List[int]:
    compatible = [
        (idx, span)
        for idx, span in enumerate(pred_spans)
        if idx not in matched_pred_idxs
        and top_level_label(span["type"]) == top_level_label(gold_span["type"])
        and int(span["start_char"]) >= int(gold_span["start_char"])
        and int(span["end_char"]) <= int(gold_span["end_char"])
    ]
    compatible.sort(key=lambda item: (int(item[1]["start_char"]), int(item[1]["end_char"]), item[0]))
    if len(compatible) < 2:
        return []

    first_start = int(compatible[0][1]["start_char"])
    last_end = int(compatible[-1][1]["end_char"])
    gold_start = int(gold_span["start_char"])
    gold_end = int(gold_span["end_char"])
    if first_start != gold_start or last_end != gold_end:
        return []

    selected: List[int] = []
    prev_end = gold_start
    for idx, span in compatible:
        start = int(span["start_char"])
        end = int(span["end_char"])
        if start < prev_end:
            return []
        if start > prev_end and not _is_trivial_gap(sentence_text[prev_end:start]):
            return []
        selected.append(idx)
        prev_end = end

    return selected if prev_end == gold_end else []


def _find_head_aware_pred_gold_matches(
    pred_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    pred_heads: List[Any],
    gold_heads: List[Any],
    matched_pred_idxs: set[int],
    matched_gold_idxs: set[int],
) -> List[Tuple[int, List[int], int]]:
    matches: List[Tuple[int, List[int], int]] = []
    pred_order = sorted(
        [idx for idx in range(len(pred_spans)) if idx not in matched_pred_idxs],
        key=lambda idx: (
            -(int(pred_spans[idx]["end_char"]) - int(pred_spans[idx]["start_char"])),
            int(pred_spans[idx]["start_char"]),
            idx,
        ),
    )

    for pred_idx in pred_order:
        pred_span = pred_spans[pred_idx]
        pred_head = pred_heads[pred_idx]
        if pred_head is None:
            continue

        contained_gold_idxs = [
            gold_idx
            for gold_idx, gold_span in enumerate(gold_spans)
            if gold_idx not in matched_gold_idxs
            and int(gold_span["start_char"]) >= int(pred_span["start_char"])
            and int(gold_span["end_char"]) <= int(pred_span["end_char"])
        ]
        if not contained_gold_idxs:
            continue

        anchor_gold_idxs = [
            gold_idx
            for gold_idx in contained_gold_idxs
            if gold_heads[gold_idx] is not None
            and gold_heads[gold_idx].i == pred_head.i
            and str(gold_spans[gold_idx]["type"]) == str(pred_span["type"])
        ]
        if not anchor_gold_idxs:
            continue

        anchor_gold_idxs.sort(
            key=lambda gold_idx: (
                -(int(gold_spans[gold_idx]["end_char"]) - int(gold_spans[gold_idx]["start_char"])),
                int(gold_spans[gold_idx]["start_char"]),
                gold_idx,
            )
        )
        anchor_gold_idx = anchor_gold_idxs[0]

        covered_gold_idxs: List[int] = []
        for gold_idx in contained_gold_idxs:
            gold_head = gold_heads[gold_idx]
            if gold_head is None:
                continue
            if gold_head.i == pred_head.i or pred_head.is_ancestor(gold_head):
                covered_gold_idxs.append(gold_idx)

        if not covered_gold_idxs:
            continue
        matches.append((pred_idx, sorted(covered_gold_idxs), anchor_gold_idx))
        matched_pred_idxs.add(pred_idx)
        matched_gold_idxs.update(covered_gold_idxs)

    return matches


def _increment_confusion_boundary(
    cm: Dict[str, Counter],
    gold_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
) -> None:
    gold_by_b: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    pred_by_b: Dict[Tuple[int, int], Counter] = defaultdict(Counter)

    for s in gold_spans:
        gold_by_b[boundary_key(s)][s["type"]] += 1
    for s in pred_spans:
        pred_by_b[boundary_key(s)][s["type"]] += 1

    for bkey in sorted(set(gold_by_b) | set(pred_by_b)):
        gc = gold_by_b.get(bkey, Counter()).copy()
        pc = pred_by_b.get(bkey, Counter()).copy()

        # Exact label matches on the same boundary -> diagonal
        for label in sorted(set(gc) & set(pc)):
            m = min(gc[label], pc[label])
            if m:
                cm[label][label] += m
                gc[label] -= m
                pc[label] -= m
                if gc[label] == 0:
                    del gc[label]
                if pc[label] == 0:
                    del pc[label]

        # Residual same-boundary labels -> off-diagonal confusion
        gold_residual = []
        pred_residual = []
        for label in sorted(gc):
            gold_residual.extend([label] * gc[label])
        for label in sorted(pc):
            pred_residual.extend([label] * pc[label])

        pair_n = min(len(gold_residual), len(pred_residual))
        for i in range(pair_n):
            cm[gold_residual[i]][pred_residual[i]] += 1

        # Unmatched residuals become FN / FP against NONE
        for i in range(pair_n, len(gold_residual)):
            cm[gold_residual[i]][NONE_LABEL] += 1
        for i in range(pair_n, len(pred_residual)):
            cm[NONE_LABEL][pred_residual[i]] += 1


def evaluate(
    gold_file: str,
    model_file: str,
    output_json: str,
    empty_final_spans_txt: str,
    confusion_matrix_csv: str,
    missed_gold_entities_csv: str,
    max_debug_examples: int = 10,
    prediction_field: str = "final_spans",
    candidate_sources: Optional[List[str]] = None,
    candidate_type_map: Optional[Dict[str, str]] = None,
    collapse_entity_variants: bool = False,
    collapse_to_top_level: bool = False,
    allow_split_family_matches: bool = False,
    allow_head_subspan_matches: bool = False,
    use_head_aware_relaxed_matching: bool = False,
    spacy_model: str = "en_core_web_trf",
) -> Dict[str, Any]:
    gold_docs = load_docs(gold_file)
    model_docs = load_docs(model_file)
    nlp = load_spacy_model(spacy_model) if use_head_aware_relaxed_matching else None
    candidate_source_set = {s.strip() for s in (candidate_sources or []) if s and s.strip()} or None
    candidate_type_map_normalized = {
        str(k).strip().lower(): str(v)
        for k, v in (candidate_type_map or DEFAULT_CANDIDATE_TYPE_MAP).items()
    }

    gold_doc_ids = sorted(gold_docs.keys())
    model_doc_ids = sorted(model_docs.keys())
    shared_doc_ids = [d for d in gold_doc_ids if d in model_docs]
    missing_model_docs = [d for d in gold_doc_ids if d not in model_docs]
    extra_model_docs = [d for d in model_doc_ids if d not in gold_docs]

    if missing_model_docs:
        raise ValueError(f"Model file is missing gold doc_ids: {missing_model_docs[:10]}")

    total_tp = total_fp = total_fn = 0
    per_type_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion: Dict[str, Counter] = defaultdict(Counter)

    empty_final_spans_sentences: List[str] = []
    untyped_pred_examples: List[Dict[str, Any]] = []
    missing_final_spans_examples: List[Dict[str, Any]] = []
    model_only_extra_examples: List[Dict[str, Any]] = []
    invalid_pred_span_examples: List[Dict[str, Any]] = []
    gold_missing_entity_examples: List[Dict[str, Any]] = []
    all_missed_gold_entities: List[Dict[str, Any]] = []
    head_relaxed_match_examples: List[Dict[str, Any]] = []
    head_parse_failure_examples: List[Dict[str, Any]] = []

    aligned_sentence_count = 0
    included_sentence_count = 0
    skipped_unannotated_gold_sentences = 0
    skipped_empty_final_spans = 0
    skipped_missing_final_spans = 0
    model_only_extra_sentences_ignored = 0
    skipped_invalid_pred_spans = 0
    head_relaxed_match_prediction_count = 0
    head_relaxed_match_gold_span_count = 0
    head_parse_failure_count = 0

    gold_sentence_total = 0
    model_sentence_total = 0

    for doc_id in shared_doc_ids:
        gold_sents = gold_docs[doc_id].get("sentences", []) or []
        model_sents = model_docs[doc_id].get("sentences", []) or []
        gold_sentence_total += len(gold_sents)
        model_sentence_total += len(model_sents)

        pairs, extras = align_sentences_by_text(doc_id, gold_sents, model_sents)
        aligned_sentence_count += len(pairs)
        model_only_extra_sentences_ignored += len(extras)
        for ex in extras[: max(0, max_debug_examples - len(model_only_extra_examples))]:
            model_only_extra_examples.append(
                {
                    "doc_id": ex["doc_id"],
                    "model_sent_idx": ex["model_sent_idx"],
                    "text": normalize_whitespace_one_line(ex.get("text", ""))[:300],
                    "reason": ex["reason"],
                }
            )

        for gi, mi in pairs:
            gsent = gold_sents[gi]
            msent = model_sents[mi]
            sent_text = msent.get("text") or gsent.get("text") or ""

            if not is_gold_sentence_annotated(gsent):
                skipped_unannotated_gold_sentences += 1
                continue

            has_prediction_spans, prediction_spans_raw = _extract_pred_spans(
                msent,
                prediction_field=prediction_field,
                candidate_sources=candidate_source_set,
                candidate_type_map=candidate_type_map_normalized,
            )
            if not has_prediction_spans:
                skipped_missing_final_spans += 1
                if len(missing_final_spans_examples) < max_debug_examples:
                    missing_final_spans_examples.append(
                        {
                            "doc_id": doc_id,
                            "gold_sent_idx": gi,
                            "model_sent_idx": mi,
                            "text": normalize_whitespace_one_line(sent_text)[:300],
                        }
                    )
                continue

            if len(prediction_spans_raw) == 0:
                skipped_empty_final_spans += 1
                empty_final_spans_sentences.append(normalize_whitespace_one_line(sent_text))
                continue

            gold_spans = [
                normalize_gold_span(
                    s,
                    context=f"{doc_id}:{gi}",
                    collapse_entity_variants=collapse_entity_variants,
                    collapse_to_top_level=collapse_to_top_level,
                )
                for s in (gsent.get("spans", []) or [])
            ]

            pred_spans: List[Dict[str, Any]] = []
            for p in prediction_spans_raw:
                if not (p.get("type")) and len(untyped_pred_examples) < max_debug_examples:
                    untyped_pred_examples.append(
                        {
                            "doc_id": doc_id,
                            "gold_sent_idx": gi,
                            "model_sent_idx": mi,
                            "sentence_text": normalize_whitespace_one_line(sent_text)[:300],
                            "span": p,
                        }
                    )
                try:
                    pred_spans.append(
                        normalize_pred_span(
                            p,
                            context=f"{doc_id}:{mi}",
                            collapse_entity_variants=collapse_entity_variants,
                            collapse_to_top_level=collapse_to_top_level,
                        )
                    )
                except ValueError as e:
                    skipped_invalid_pred_spans += 1
                    if len(invalid_pred_span_examples) < max_debug_examples:
                        invalid_pred_span_examples.append(
                            {
                                "doc_id": doc_id,
                                "gold_sent_idx": gi,
                                "model_sent_idx": mi,
                                "sentence_text": normalize_whitespace_one_line(sent_text)[:300],
                                "error": str(e),
                                "span": p,
                            }
                        )
                    continue

            included_sentence_count += 1

            matched_gold_idxs: set[int] = set()
            matched_pred_idxs: set[int] = set()
            matched_gold_labels: List[str] = []

            gold_key_to_idxs: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
            pred_key_to_idxs: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
            for idx, span in enumerate(gold_spans):
                gold_key_to_idxs[span_key_full(span)].append(idx)
            for idx, span in enumerate(pred_spans):
                pred_key_to_idxs[span_key_full(span)].append(idx)

            for key in sorted(set(gold_key_to_idxs) & set(pred_key_to_idxs)):
                gold_idxs = gold_key_to_idxs[key]
                pred_idxs = pred_key_to_idxs[key]
                for gold_idx, pred_idx in zip(gold_idxs, pred_idxs):
                    matched_gold_idxs.add(gold_idx)
                    matched_pred_idxs.add(pred_idx)
                    matched_gold_labels.append(gold_spans[gold_idx]["type"])

            if allow_split_family_matches:
                for gold_idx, gold_span in enumerate(gold_spans):
                    if gold_idx in matched_gold_idxs:
                        continue
                    split_pred_idxs = _find_split_gold_match(
                        gold_span,
                        pred_spans,
                        matched_pred_idxs,
                        sent_text,
                    )
                    if not split_pred_idxs:
                        continue
                    matched_gold_idxs.add(gold_idx)
                    matched_pred_idxs.update(split_pred_idxs)
                    matched_gold_labels.append(gold_span["type"])

            if allow_head_subspan_matches:
                for gold_idx, gold_span in enumerate(gold_spans):
                    if gold_idx in matched_gold_idxs:
                        continue
                    head_pred_idx = _find_head_subspan_match(
                        gold_span,
                        pred_spans,
                        matched_pred_idxs,
                    )
                    if head_pred_idx is None:
                        continue
                    matched_gold_idxs.add(gold_idx)
                    matched_pred_idxs.add(head_pred_idx)
                    matched_gold_labels.append(gold_span["type"])

            if use_head_aware_relaxed_matching:
                doc = nlp(sent_text)
                gold_heads, gold_head_failures = build_span_heads(doc, gold_spans)
                pred_heads, pred_head_failures = build_span_heads(doc, pred_spans)
                head_parse_failure_count += len(gold_head_failures) + len(pred_head_failures)

                for failure in gold_head_failures:
                    if len(head_parse_failure_examples) >= max_debug_examples:
                        break
                    head_parse_failure_examples.append(
                        {
                            "doc_id": doc_id,
                            "gold_sent_idx": gi,
                            "model_sent_idx": mi,
                            "sentence_text": normalize_whitespace_one_line(sent_text)[:300],
                            "span_source": "gold",
                            **failure,
                        }
                    )
                for failure in pred_head_failures:
                    if len(head_parse_failure_examples) >= max_debug_examples:
                        break
                    head_parse_failure_examples.append(
                        {
                            "doc_id": doc_id,
                            "gold_sent_idx": gi,
                            "model_sent_idx": mi,
                            "sentence_text": normalize_whitespace_one_line(sent_text)[:300],
                            "span_source": "pred",
                            **failure,
                        }
                    )

                relaxed_matches = _find_head_aware_pred_gold_matches(
                    pred_spans,
                    gold_spans,
                    pred_heads,
                    gold_heads,
                    matched_pred_idxs,
                    matched_gold_idxs,
                )
                for pred_idx, covered_gold_idxs, anchor_gold_idx in relaxed_matches:
                    head_relaxed_match_prediction_count += 1
                    head_relaxed_match_gold_span_count += len(covered_gold_idxs)
                    for gold_idx in covered_gold_idxs:
                        matched_gold_labels.append(gold_spans[gold_idx]["type"])
                    if len(head_relaxed_match_examples) < max_debug_examples:
                        pred_head = pred_heads[pred_idx]
                        head_relaxed_match_examples.append(
                            {
                                "doc_id": doc_id,
                                "gold_sent_idx": gi,
                                "model_sent_idx": mi,
                                "sentence_text": normalize_whitespace_one_line(sent_text)[:300],
                                "pred_span": pred_spans[pred_idx],
                                "pred_head": None
                                if pred_head is None
                                else {
                                    "text": pred_head.text,
                                    "idx": pred_head.i,
                                    "pos": pred_head.pos_,
                                },
                                "anchor_gold_span": gold_spans[anchor_gold_idx],
                                "covered_gold_spans": [gold_spans[idx] for idx in covered_gold_idxs],
                            }
                        )

            tp = len(matched_gold_idxs)
            fp = len(pred_spans) - len(matched_pred_idxs)
            fn = len(gold_spans) - len(matched_gold_idxs)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Collect examples for gold entities missed by the model (gold -> __NONE__).
            if len(gold_missing_entity_examples) < max_debug_examples:
                for gold_idx, gspan in enumerate(gold_spans):
                    if gold_idx in matched_gold_idxs:
                        continue
                    gold_missing_entity_examples.append(
                        {
                            "doc_id": doc_id,
                            "gold_sent_idx": gi,
                            "model_sent_idx": mi,
                            "sentence_text": normalize_whitespace_one_line(sent_text)[:400],
                            "gold_span": gspan,
                        }
                    )
                    if len(gold_missing_entity_examples) >= max_debug_examples:
                        break

            for gold_idx, gspan in enumerate(gold_spans):
                if gold_idx in matched_gold_idxs:
                    continue
                all_missed_gold_entities.append(
                    {
                        "doc_id": doc_id,
                        "gold_sent_idx": gi,
                        "model_sent_idx": mi,
                        "sentence_text": normalize_whitespace_one_line(sent_text),
                        "gold_type": gspan.get("type"),
                        "gold_start_char": gspan.get("start_char"),
                        "gold_end_char": gspan.get("end_char"),
                        "gold_text": gspan.get("text"),
                    }
                )

            for label in matched_gold_labels:
                per_type_counts[label]["tp"] += 1
                confusion[label][label] += 1
            for pred_idx, pspan in enumerate(pred_spans):
                if pred_idx not in matched_pred_idxs:
                    per_type_counts[pspan["type"]]["fp"] += 1
            for gold_idx, gspan in enumerate(gold_spans):
                if gold_idx not in matched_gold_idxs:
                    per_type_counts[gspan["type"]]["fn"] += 1

            remaining_gold_spans = [g for idx, g in enumerate(gold_spans) if idx not in matched_gold_idxs]
            remaining_pred_spans = [p for idx, p in enumerate(pred_spans) if idx not in matched_pred_idxs]
            _increment_confusion_boundary(confusion, remaining_gold_spans, remaining_pred_spans)

    # Build per-type metrics
    per_type_metrics: Dict[str, Dict[str, Any]] = {}
    all_types = sorted(per_type_counts.keys())
    for label in all_types:
        tp = per_type_counts[label]["tp"]
        fp = per_type_counts[label]["fp"]
        fn = per_type_counts[label]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        per_type_metrics[label] = {
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f1, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_support": tp + fn,
            "pred_support": tp + fp,
        }

    # Build confusion matrix
    observed_labels = sorted(set(per_type_counts.keys()) | set(confusion.keys()) | {k for row in confusion.values() for k in row.keys()})
    labels = [l for l in observed_labels if l != NONE_LABEL]
    labels.append(NONE_LABEL)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for g_label, row in confusion.items():
        for p_label, cnt in row.items():
            if g_label not in label_to_idx or p_label not in label_to_idx:
                continue
            matrix[label_to_idx[g_label]][label_to_idx[p_label]] = int(cnt)

    row_totals = [sum(r) for r in matrix]
    col_totals = [sum(matrix[r][c] for r in range(len(labels))) for c in range(len(labels))]

    P, R, F1 = prf(total_tp, total_fp, total_fn)

    results = {
        "metadata": {
            "gold_file": str(gold_file),
            "model_file": str(model_file),
            "output_json": str(output_json),
            "empty_final_spans_txt": str(empty_final_spans_txt),
            "confusion_matrix_csv": str(confusion_matrix_csv),
            "missed_gold_entities_csv": str(missed_gold_entities_csv),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluator": "evaluate_llm_final_spans.py",
            "matching_policy": "strict_exact_span_and_type",
            "alignment_policy": "exact_text_in_order",
            "gold_annotation_policy": "score_only_sentences_with_nonempty_gold_spans",
            "prediction_field": prediction_field,
            "candidate_sources": sorted(candidate_source_set) if candidate_source_set else [],
            "candidate_type_map": candidate_type_map_normalized if prediction_field == "candidates" else {},
            "collapse_entity_variants": collapse_entity_variants,
            "collapse_to_top_level": collapse_to_top_level,
            "allow_split_family_matches": allow_split_family_matches,
            "allow_head_subspan_matches": allow_head_subspan_matches,
            "use_head_aware_relaxed_matching": use_head_aware_relaxed_matching,
            "spacy_model": spacy_model if use_head_aware_relaxed_matching else None,
            "collapsed_entity_label_map": COLLAPSED_ENTITY_LABEL_MAP if collapse_entity_variants else {},
            "top_level_label_prefixes": list(TOP_LEVEL_LABEL_PREFIXES) if collapse_to_top_level else [],
            "untyped_prediction_policy": UNKNOWN_LABEL,
            "empty_final_spans_policy": "skip_and_export_sentence_text",
            "extra_model_only_sentences_policy": "ignored_for_scoring",
        },
        "dataset_stats": {
            "gold_doc_count": len(gold_doc_ids),
            "model_doc_count": len(model_doc_ids),
            "shared_doc_count": len(shared_doc_ids),
            "extra_model_doc_count": len(extra_model_docs),
            "gold_sentence_total": gold_sentence_total,
            "model_sentence_total": model_sentence_total,
            "aligned_sentence_count": aligned_sentence_count,
            "included_sentence_count": included_sentence_count,
            "skipped_unannotated_gold_sentences": skipped_unannotated_gold_sentences,
            "skipped_empty_final_spans": skipped_empty_final_spans,
            "skipped_missing_final_spans": skipped_missing_final_spans,
            "model_only_extra_sentences_ignored": model_only_extra_sentences_ignored,
            "empty_final_spans_txt_line_count": len(empty_final_spans_sentences),
            "skipped_invalid_pred_spans": skipped_invalid_pred_spans,
            "head_relaxed_match_prediction_count": head_relaxed_match_prediction_count,
            "head_relaxed_match_gold_span_count": head_relaxed_match_gold_span_count,
            "head_parse_failure_count": head_parse_failure_count,
            "missed_gold_entities_count": len(all_missed_gold_entities),
        },
        "overall": {
            "precision": round(P, 6),
            "recall": round(R, 6),
            "f1": round(F1, 6),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_type": per_type_metrics,
        "confusion_matrix": {
            "labels": labels,
            "counts": matrix,
            "row_totals": row_totals,
            "col_totals": col_totals,
            "none_label": NONE_LABEL,
        },
        "diagnostics": {
            "missing_model_doc_ids": missing_model_docs,
            "extra_model_doc_ids": extra_model_docs,
            "sample_model_only_extra_sentences": model_only_extra_examples,
            "sample_missing_final_spans_sentences": missing_final_spans_examples,
            "sample_untyped_predictions": untyped_pred_examples,
            "sample_invalid_pred_spans": invalid_pred_span_examples,
            "sample_gold_missing_entity_examples": gold_missing_entity_examples,
            "sample_head_relaxed_matches": head_relaxed_match_examples,
            "sample_head_parse_failures": head_parse_failure_examples,
        },
    }

    # Write outputs
    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    empty_txt_path = Path(empty_final_spans_txt)
    empty_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with empty_txt_path.open("w", encoding="utf-8") as f:
        for line in empty_final_spans_sentences:
            f.write(line)
            f.write("\n")

    cm_csv_path = Path(confusion_matrix_csv)
    cm_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with cm_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels, "row_total"])
        for label, row, row_total in zip(labels, matrix, row_totals):
            writer.writerow([label, *row, row_total])
        writer.writerow(["col_total", *col_totals, sum(row_totals)])

    missed_csv_path = Path(missed_gold_entities_csv)
    missed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with missed_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "gold_sent_idx",
                "model_sent_idx",
                "gold_type",
                "gold_start_char",
                "gold_end_char",
                "gold_text",
                "sentence_text",
            ],
        )
        writer.writeheader()
        writer.writerows(all_missed_gold_entities)

    return results


def _derive_default_outputs(
    model_file: str,
    output_json: str | None,
    empty_txt: str | None,
    confusion_csv: str | None,
    missed_gold_csv: str | None,
) -> Tuple[str, str, str, str]:
    model_path = Path(model_file)
    stem = model_path.stem
    base_dir = Path("output") / "eval"
    if output_json:
        out_json = output_json
        if empty_txt:
            out_txt = empty_txt
        else:
            out_txt = str(Path(output_json).with_name(Path(output_json).stem + "_empty_final_spans.txt"))
        if confusion_csv:
            out_csv = confusion_csv
        else:
            out_csv = str(Path(output_json).with_name(Path(output_json).stem + "_confusion_matrix.csv"))
        if missed_gold_csv:
            out_missed_csv = missed_gold_csv
        else:
            out_missed_csv = str(Path(output_json).with_name(Path(output_json).stem + "_missed_gold_entities.csv"))
    else:
        out_json = str(base_dir / f"{stem}_eval.json")
        out_txt = empty_txt or str(base_dir / f"{stem}_empty_final_spans.txt")
        out_csv = confusion_csv or str(base_dir / f"{stem}_confusion_matrix.csv")
        out_missed_csv = missed_gold_csv or str(base_dir / f"{stem}_missed_gold_entities.csv")
    return out_json, out_txt, out_csv, out_missed_csv


def pretty_print(results: Dict[str, Any], max_types: int = 20) -> None:
    ds = results["dataset_stats"]
    ov = results["overall"]
    md = results.get("metadata") or {}
    prediction_field = md.get("prediction_field") or "final_spans"
    print("=" * 70)
    print("LLM ENTITY EXTRACTION EVALUATION (final_spans)")
    print("=" * 70)
    if md.get("use_head_aware_relaxed_matching"):
        print(f"Head-aware relaxed matching: ON ({md.get('spacy_model')})")
    elif md.get("collapse_to_top_level"):
        print("Head-aware relaxed matching: OFF")
        print("Top-level label collapse:     ON")
    print("Overall (micro, strict exact span+type)")
    print(f"Precision: {ov['precision']:.4f}")
    print(f"Recall:    {ov['recall']:.4f}")
    print(f"F1:        {ov['f1']:.4f}")
    print(f"TP/FP/FN:  {ov['tp']} / {ov['fp']} / {ov['fn']}")
    print()
    print("Dataset stats")
    print(f"Aligned sentences:            {ds['aligned_sentence_count']}")
    print(f"Included sentences:           {ds['included_sentence_count']}")
    print(f"Skipped unannotated gold:     {ds.get('skipped_unannotated_gold_sentences', 0)}")
    print(f"Skipped empty {prediction_field}:    {ds['skipped_empty_final_spans']}")
    print(f"Skipped missing {prediction_field}:  {ds['skipped_missing_final_spans']}")
    print(f"Model-only extras ignored:    {ds['model_only_extra_sentences_ignored']}")
    print(f"Empty-{prediction_field} TXT lines:  {ds['empty_final_spans_txt_line_count']}")
    print(f"Skipped invalid pred spans:   {ds.get('skipped_invalid_pred_spans', 0)}")
    print(f"Head-relaxed pred matches:    {ds.get('head_relaxed_match_prediction_count', 0)}")
    print(f"Head-relaxed gold covered:    {ds.get('head_relaxed_match_gold_span_count', 0)}")
    print(f"Head parse failures:          {ds.get('head_parse_failure_count', 0)}")
    print()

    per_type = results.get("per_type", {})
    if per_type:
        print("Per-type metrics (sorted by gold support desc)")
        items = sorted(per_type.items(), key=lambda kv: (-kv[1]["gold_support"], kv[0]))
        for label, m in items[:max_types]:
            print(
                f"{label:28s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                f"tp={m['tp']} fp={m['fp']} fn={m['fn']} gold={m['gold_support']} pred={m['pred_support']}"
            )
        if len(items) > max_types:
            print(f"... ({len(items) - max_types} more labels)")

    cm = results.get("confusion_matrix") or {}
    labels = cm.get("labels") or []
    counts = cm.get("counts") or []
    if labels and counts:
        print()
        print("Confusion matrix (rows=gold, cols=pred)")
        # Keep it readable in terminal by truncating labels to a fixed width.
        col_w = 10
        row_label_w = 22
        short_labels = [lbl if len(lbl) <= col_w else lbl[: col_w - 1] + "…" for lbl in labels]
        header = " " * row_label_w + " ".join(f"{lbl:>{col_w}s}" for lbl in short_labels)
        print(header)
        for label, row in zip(labels, counts):
            row_name = label if len(label) <= row_label_w else label[: row_label_w - 1] + "…"
            print(f"{row_name:<{row_label_w}s}" + " ".join(f"{int(v):>{col_w}d}" for v in row))

    fn_examples = (results.get("diagnostics") or {}).get("sample_gold_missing_entity_examples") or []
    if fn_examples:
        print()
        print("Examples: missed gold entities (gold -> __NONE__)")
        for i, ex in enumerate(fn_examples, 1):
            gs = ex["gold_span"]
            print(
                f"{i}. {ex['doc_id']} gold_idx={ex['gold_sent_idx']} model_idx={ex['model_sent_idx']} "
                f"type={gs.get('type')} span=({gs.get('start_char')},{gs.get('end_char')}) "
                f"text={gs.get('text')!r}"
            )
            print(f"   sentence: {ex['sentence_text']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate entity extraction pipeline output against annotated spans using sentence final_spans or candidates."
    )
    ap.add_argument("--gold-file", required=True, help="Gold JSONL file with sentences[].spans")
    ap.add_argument(
        "--model-file",
        required=True,
        help="Model JSONL file with sentence-level predictions in llm.final_spans/final_spans or candidates.",
    )
    ap.add_argument(
        "--prediction-field",
        choices=["final_spans", "candidates"],
        default="final_spans",
        help="Which model sentence field to score against gold spans.",
    )
    ap.add_argument(
        "--candidate-sources",
        nargs="+",
        help=(
            "When --prediction-field candidates is used, keep only candidate source entries whose "
            "name matches one of these values, e.g. ner gazetteer."
        ),
    )
    ap.add_argument(
        "--candidate-type-map-json",
        help=(
            "Optional JSON file mapping candidate source labels to evaluation labels, "
            "e.g. {'TAXON':'BIOTIC ENTITY'}."
        ),
    )
    ap.add_argument("--output-json", help="Path for structured JSON evaluation report")
    ap.add_argument(
        "--empty-final-spans-txt",
        help="Path for TXT file with sentence text (one line per sentence) for aligned sentences with empty final_spans",
    )
    ap.add_argument(
        "--confusion-matrix-csv",
        help="Path for CSV file containing confusion matrix (rows=gold, cols=pred)",
    )
    ap.add_argument(
        "--missed-gold-entities-csv",
        help="Path for CSV file containing all missed gold entities (gold -> __NONE__)",
    )
    ap.add_argument(
        "--collapse-entity-variants",
        action="store_true",
        help=(
            "Treat collective entity labels as equivalent to their base entity labels "
            "(ABIOTIC, BIOTIC, ANTHROPOGENIC) during scoring."
        ),
    )
    ap.add_argument(
        "--collapse-to-top-level",
        action="store_true",
        help=(
            "Treat labels with the same top-level family as equivalent, such as "
            "BIOTIC ENTITY/PROPERTY/PROCESS -> BIOTIC."
        ),
    )
    ap.add_argument(
        "--allow-split-family-matches",
        action="store_true",
        help=(
            "Count one gold span as matched when multiple predicted subspans exactly cover it "
            "and all belong to the same top-level family."
        ),
    )
    ap.add_argument(
        "--allow-head-subspan-matches",
        action="store_true",
        help=(
            "Count one gold span as matched when a predicted subspan with the same label matches "
            "the phrase head as an exact prefix or suffix."
        ),
    )
    ap.add_argument(
        "--use-head-aware-relaxed-matching",
        action="store_true",
        help=(
            "Use a spaCy dependency parser to allow a broader predicted span to satisfy contained "
            "gold spans when the prediction head aligns with a compatible gold head."
        ),
    )
    ap.add_argument(
        "--spacy-model",
        default="en_core_web_trf",
        help="spaCy model with parser enabled, used only with --use-head-aware-relaxed-matching",
    )
    ap.add_argument("--max-debug-examples", type=int, default=10, help="Max examples to keep in diagnostics")
    args = ap.parse_args()

    output_json, empty_txt, confusion_csv, missed_gold_csv = _derive_default_outputs(
        args.model_file,
        args.output_json,
        args.empty_final_spans_txt,
        args.confusion_matrix_csv,
        args.missed_gold_entities_csv,
    )
    candidate_type_map = None
    if args.candidate_type_map_json:
        with open(args.candidate_type_map_json, "r", encoding="utf-8") as f:
            candidate_type_map = json.load(f)
    results = evaluate(
        gold_file=args.gold_file,
        model_file=args.model_file,
        output_json=output_json,
        empty_final_spans_txt=empty_txt,
        confusion_matrix_csv=confusion_csv,
        missed_gold_entities_csv=missed_gold_csv,
        max_debug_examples=args.max_debug_examples,
        prediction_field=args.prediction_field,
        candidate_sources=args.candidate_sources,
        candidate_type_map=candidate_type_map,
        collapse_entity_variants=args.collapse_entity_variants,
        collapse_to_top_level=args.collapse_to_top_level,
        allow_split_family_matches=args.allow_split_family_matches,
        allow_head_subspan_matches=args.allow_head_subspan_matches,
        use_head_aware_relaxed_matching=args.use_head_aware_relaxed_matching,
        spacy_model=args.spacy_model,
    )
    pretty_print(results)
    print()
    print(f"JSON report: {output_json}")
    print(f"Empty final_spans sentences TXT: {empty_txt}")
    print(f"Confusion matrix CSV: {confusion_csv}")
    print(f"Missed gold entities CSV: {missed_gold_csv}")


if __name__ == "__main__":
    main()
