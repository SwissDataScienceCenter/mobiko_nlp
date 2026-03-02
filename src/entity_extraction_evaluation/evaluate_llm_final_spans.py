import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


NONE_LABEL = "__NONE__"
UNKNOWN_LABEL = "__UNKNOWN__"


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


def normalize_gold_span(span: Dict[str, Any], *, context: str) -> Dict[str, Any]:
    if "start_char" not in span or "end_char" not in span:
        raise ValueError(f"{context}: gold span missing offsets: {span}")
    label = span.get("type")
    if not label:
        raise ValueError(f"{context}: gold span missing type: {span}")
    out = {
        "start_char": int(span["start_char"]),
        "end_char": int(span["end_char"]),
        "type": str(label),
        "text": span.get("text"),
    }
    if out["end_char"] <= out["start_char"]:
        raise ValueError(f"{context}: invalid gold span offsets: {span}")
    return out


def normalize_pred_span(span: Dict[str, Any], *, context: str) -> Dict[str, Any]:
    if "start_char" not in span or "end_char" not in span:
        raise ValueError(f"{context}: pred span missing offsets: {span}")
    label = span.get("type") or UNKNOWN_LABEL
    out = {
        "start_char": int(span["start_char"]),
        "end_char": int(span["end_char"]),
        "type": str(label),
        "text": span.get("text"),
    }
    if out["end_char"] <= out["start_char"]:
        raise ValueError(f"{context}: invalid pred span offsets: {span}")
    return out


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
) -> Dict[str, Any]:
    gold_docs = load_docs(gold_file)
    model_docs = load_docs(model_file)

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

    aligned_sentence_count = 0
    included_sentence_count = 0
    skipped_empty_final_spans = 0
    skipped_missing_final_spans = 0
    model_only_extra_sentences_ignored = 0
    skipped_invalid_pred_spans = 0

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

            has_final_spans, final_spans_raw = _get_pred_final_spans(msent)
            if not has_final_spans:
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

            if len(final_spans_raw) == 0:
                skipped_empty_final_spans += 1
                empty_final_spans_sentences.append(normalize_whitespace_one_line(sent_text))
                continue

            gold_spans = [
                normalize_gold_span(s, context=f"{doc_id}:{gi}")
                for s in (gsent.get("spans", []) or [])
            ]

            pred_spans: List[Dict[str, Any]] = []
            for p in final_spans_raw:
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
                    pred_spans.append(normalize_pred_span(p, context=f"{doc_id}:{mi}"))
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

            gold_counter = Counter(span_key_full(s) for s in gold_spans)
            pred_counter = Counter(span_key_full(s) for s in pred_spans)

            matched_counter = Counter()
            for k in set(gold_counter) & set(pred_counter):
                m = min(gold_counter[k], pred_counter[k])
                if m:
                    matched_counter[k] = m

            tp = sum(matched_counter.values())
            fp = sum((pred_counter - matched_counter).values())
            fn = sum((gold_counter - matched_counter).values())

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Collect examples for gold entities missed by the model (gold -> __NONE__).
            if len(gold_missing_entity_examples) < max_debug_examples:
                matched_left = matched_counter.copy()
                for gspan in gold_spans:
                    k = span_key_full(gspan)
                    if matched_left.get(k, 0) > 0:
                        matched_left[k] -= 1
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

            matched_left_all = matched_counter.copy()
            for gspan in gold_spans:
                k = span_key_full(gspan)
                if matched_left_all.get(k, 0) > 0:
                    matched_left_all[k] -= 1
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

            for (_, _, label), cnt in matched_counter.items():
                per_type_counts[label]["tp"] += cnt
            for (_, _, label), cnt in (pred_counter - matched_counter).items():
                per_type_counts[label]["fp"] += cnt
            for (_, _, label), cnt in (gold_counter - matched_counter).items():
                per_type_counts[label]["fn"] += cnt

            _increment_confusion_boundary(confusion, gold_spans, pred_spans)

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
            "skipped_empty_final_spans": skipped_empty_final_spans,
            "skipped_missing_final_spans": skipped_missing_final_spans,
            "model_only_extra_sentences_ignored": model_only_extra_sentences_ignored,
            "empty_final_spans_txt_line_count": len(empty_final_spans_sentences),
            "skipped_invalid_pred_spans": skipped_invalid_pred_spans,
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
    print("=" * 70)
    print("LLM ENTITY EXTRACTION EVALUATION (final_spans)")
    print("=" * 70)
    print("Overall (micro, strict exact span+type)")
    print(f"Precision: {ov['precision']:.4f}")
    print(f"Recall:    {ov['recall']:.4f}")
    print(f"F1:        {ov['f1']:.4f}")
    print(f"TP/FP/FN:  {ov['tp']} / {ov['fp']} / {ov['fn']}")
    print()
    print("Dataset stats")
    print(f"Aligned sentences:            {ds['aligned_sentence_count']}")
    print(f"Included sentences:           {ds['included_sentence_count']}")
    print(f"Skipped empty final_spans:    {ds['skipped_empty_final_spans']}")
    print(f"Skipped missing final_spans:  {ds['skipped_missing_final_spans']}")
    print(f"Model-only extras ignored:    {ds['model_only_extra_sentences_ignored']}")
    print(f"Empty-final-spans TXT lines:  {ds['empty_final_spans_txt_line_count']}")
    print(f"Skipped invalid pred spans:   {ds.get('skipped_invalid_pred_spans', 0)}")
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
        description="Evaluate entity extraction pipeline output against annotated spans using llm.final_spans."
    )
    ap.add_argument("--gold-file", required=True, help="Gold JSONL file with sentences[].spans")
    ap.add_argument("--model-file", required=True, help="Model JSONL file with sentences[].llm.final_spans")
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
    ap.add_argument("--max-debug-examples", type=int, default=10, help="Max examples to keep in diagnostics")
    args = ap.parse_args()

    output_json, empty_txt, confusion_csv, missed_gold_csv = _derive_default_outputs(
        args.model_file,
        args.output_json,
        args.empty_final_spans_txt,
        args.confusion_matrix_csv,
        args.missed_gold_entities_csv,
    )
    results = evaluate(
        gold_file=args.gold_file,
        model_file=args.model_file,
        output_json=output_json,
        empty_final_spans_txt=empty_txt,
        confusion_matrix_csv=confusion_csv,
        missed_gold_entities_csv=missed_gold_csv,
        max_debug_examples=args.max_debug_examples,
    )
    pretty_print(results)
    print()
    print(f"JSON report: {output_json}")
    print(f"Empty final_spans sentences TXT: {empty_txt}")
    print(f"Confusion matrix CSV: {confusion_csv}")
    print(f"Missed gold entities CSV: {missed_gold_csv}")


if __name__ == "__main__":
    main()
