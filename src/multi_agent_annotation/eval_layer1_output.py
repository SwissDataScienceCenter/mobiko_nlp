"""
Layer 1 — Output quality evaluation.

Compares agent annotations against human annotations at the span level.
Computes precision, recall, F1 per entity type and relation type.
Compares agent–human agreement to inter-human agreement.

Expected input formats
----------------------
Agent output (--agent-jsonl):  JSONL, one DeliberationRecord per line.
    {"sentence": "...", "final_entities": [{"text", "entity_type", ...}],
     "final_relations": [{"relation", "e1": {"text", "entity_type"}, "e2": ...}], ...}

Human annotations (--human-jsonl):  One or more files, one file per annotator.
  Annotator name is taken from each filename stem.
  Two file formats accepted:

  Native project JSON (single file):
    {"doc_id": "...", "sentences": [{"text": "...", "spans": [{"text", "type", ...}]}]}

  Per-annotator JSONL (one record per sentence):
    {"sentence": "...", "annotator": "A1",
     "entities": [{"text", "entity_type"}],
     "relations": [{"relation", "e1_text", "e1_type", "e2_text", "e2_type"}]}

Inter-human agreement is computed only on sentences annotated by ALL human annotators.
Agent-vs-human agreement is computed only on sentences present in both the agent output
and ALL human annotation files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from eval_utils import (
    bootstrap_ci,
    two_sample_bootstrap_p,
    fmt_ci,
    fmt_p,
)

SCHEMA_V1_TO_V2: Dict[str, str] = {
    "BIOTIC COLLECTIVE ENTITY":  "BIOTIC ENTITY",
    "ABIOTIC COLLECTIVE ENTITY": "ABIOTIC ENTITY",
}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_agent_records(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_human_annotations(path: Path) -> Dict[str, Dict[str, dict]]:
    """
    Returns {sentence_key: {annotator_id: {"entities": [...], "relations": [...]}}}
    sentence_key is the sentence text as-is.

    Accepts two formats:
    - Native project JSON: single object {doc_id, sentences:[{text, spans:[{text, type, ...}]}]}
    - Per-annotator JSONL: one record per line {sentence, annotator, entities, relations}
    """
    raw = path.read_text(encoding="utf8").strip()

    # Detect native project JSON format
    if raw.startswith("{"):
        try:
            doc = json.loads(raw)
            if "sentences" in doc:
                return _load_native_json_annotations(doc, path.stem)
        except json.JSONDecodeError:
            pass

    # Per-annotator JSONL format
    data: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        sent = obj["sentence"].strip()
        annotator = obj.get("annotator", "default")
        entities = [
            {**e, "entity_type": _normalize_label(e.get("entity_type", e.get("type", "")))}
            for e in obj.get("entities", [])
        ]
        data[sent][annotator] = {
            "entities": entities,
            "relations": obj.get("relations", []),
        }
    return dict(data)


def _load_native_json_annotations(doc: dict, annotator: str) -> Dict[str, Dict[str, dict]]:
    """Convert native {doc_id, sentences:[{text, spans}]} format."""
    data: Dict[str, Dict[str, dict]] = {}
    for sent_obj in doc.get("sentences", []):
        sent = sent_obj["text"].strip()
        entities = [
            {"text": span["text"], "entity_type": _normalize_label(span["type"])}
            for span in sent_obj.get("spans", [])
        ]
        data[sent] = {annotator: {"entities": entities, "relations": []}}
    return data


def load_all_human_annotations(paths: List[Path]) -> Dict[str, Dict[str, dict]]:
    """
    Load and merge annotations from multiple per-annotator files.
    Returns {sentence_key: {annotator_id: {"entities": [...], "relations": [...]}}}
    """
    merged: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for path in paths:
        for sent, annotators in load_human_annotations(path).items():
            merged[sent].update(annotators)
    return dict(merged)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_label(label: str) -> str:
    """Map V1 schema labels to V2 equivalents for human annotations."""
    upper = label.strip().upper()
    return SCHEMA_V1_TO_V2.get(upper, upper)


# ─────────────────────────────────────────────────────────────
# Span matching
# ─────────────────────────────────────────────────────────────

def _entity_key_exact(ent: dict) -> Tuple[str, str]:
    """(normalized_text, entity_type) for exact match."""
    text = ent.get("text", "")
    etype = ent.get("entity_type", ent.get("type", ""))
    return (_normalize(text), etype.strip().upper())


def _entity_key_text_only(ent: dict) -> str:
    return _normalize(ent.get("text", ""))


def _overlap_ratio(a: str, b: str) -> float:
    """Token-level overlap ratio (Jaccard on tokens)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _relation_key(rel: dict) -> Tuple[str, str, str, str, str]:
    """(relation, e1_text, e1_type, e2_text, e2_type) normalized."""
    # Handle both nested (agent) and flat (human) formats
    if "e1" in rel and isinstance(rel["e1"], dict):
        e1_text = rel["e1"].get("text", "")
        e1_type = rel["e1"].get("entity_type", "")
        e2_text = rel["e2"].get("text", "")
        e2_type = rel["e2"].get("entity_type", "")
    else:
        e1_text = rel.get("e1_text", "")
        e1_type = rel.get("e1_type", "")
        e2_text = rel.get("e2_text", "")
        e2_type = rel.get("e2_type", "")
    return (
        rel.get("relation", "").strip().upper(),
        _normalize(e1_text),
        e1_type.strip().upper(),
        _normalize(e2_text),
        e2_type.strip().upper(),
    )


# ─────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────

def compute_entity_metrics(
    pred_entities: List[dict],
    gold_entities: List[dict],
    match_mode: str = "exact",       # "exact" | "type_only" | "text_only"
    overlap_threshold: float = 0.5,  # for relaxed matching
) -> Dict[str, Any]:
    """
    Compute TP/FP/FN and P/R/F1 for entity spans.

    match_mode:
        exact     — (text, type) must match exactly
        type_only — text must overlap ≥ threshold, type must match
        text_only — text must overlap ≥ threshold, type ignored
    """
    if match_mode == "exact":
        pred_set = {_entity_key_exact(e) for e in pred_entities}
        gold_set = {_entity_key_exact(e) for e in gold_entities}
        tp = pred_set & gold_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set
        return _prf(len(tp), len(fp), len(fn))

    # Relaxed matching: greedy 1-to-1 alignment
    pred_matched = set()
    gold_matched = set()
    tp = 0

    for gi, g in enumerate(gold_entities):
        best_score = -1
        best_pi = -1
        for pi, p in enumerate(pred_entities):
            if pi in pred_matched:
                continue
            overlap = _overlap_ratio(
                p.get("text", ""), g.get("text", "")
            )
            if overlap < overlap_threshold:
                continue
            if match_mode == "type_only":
                p_type = (p.get("entity_type") or p.get("type", "")).strip().upper()
                g_type = (g.get("entity_type") or g.get("type", "")).strip().upper()
                if p_type != g_type:
                    continue
            if overlap > best_score:
                best_score = overlap
                best_pi = pi
        if best_pi >= 0:
            tp += 1
            pred_matched.add(best_pi)
            gold_matched.add(gi)

    fp = len(pred_entities) - len(pred_matched)
    fn = len(gold_entities) - len(gold_matched)
    return _prf(tp, fp, fn)


def compute_relation_metrics(
    pred_relations: List[dict],
    gold_relations: List[dict],
) -> Dict[str, Any]:
    """Compute TP/FP/FN for relations (exact match on all 5 fields)."""
    pred_set = {_relation_key(r) for r in pred_relations}
    gold_set = {_relation_key(r) for r in gold_relations}
    tp = pred_set & gold_set
    fp = pred_set - gold_set
    fn = gold_set - pred_set
    return _prf(len(tp), len(fp), len(fn))


def _prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


# ─────────────────────────────────────────────────────────────
# Per-type breakdown
# ─────────────────────────────────────────────────────────────

def compute_per_type_entity_metrics(
    pred_entities: List[dict],
    gold_entities: List[dict],
    match_mode: str = "exact",
) -> Dict[str, Dict[str, Any]]:
    """P/R/F1 broken down by entity type, respecting match_mode."""
    all_types: Set[str] = set()
    pred_by_type: Dict[str, list] = defaultdict(list)
    gold_by_type: Dict[str, list] = defaultdict(list)

    for e in pred_entities:
        t = (e.get("entity_type") or e.get("type", "")).strip().upper()
        all_types.add(t)
        pred_by_type[t].append(e)
    for e in gold_entities:
        t = (e.get("entity_type") or e.get("type", "")).strip().upper()
        all_types.add(t)
        gold_by_type[t].append(e)

    results = {}
    for t in sorted(all_types):
        results[t] = compute_entity_metrics(
            pred_by_type.get(t, []), gold_by_type.get(t, []), match_mode=match_mode
        )
    return results


def compute_per_type_relation_metrics(
    pred_relations: List[dict],
    gold_relations: List[dict],
) -> Dict[str, Dict[str, Any]]:
    """Exact-match P/R/F1 broken down by relation type."""
    all_types: Set[str] = set()
    pred_by_type: Dict[str, list] = defaultdict(list)
    gold_by_type: Dict[str, list] = defaultdict(list)

    for r in pred_relations:
        t = r.get("relation", "").strip().upper()
        all_types.add(t)
        pred_by_type[t].append(r)
    for r in gold_relations:
        t = r.get("relation", "").strip().upper()
        all_types.add(t)
        gold_by_type[t].append(r)

    results = {}
    for t in sorted(all_types):
        results[t] = compute_relation_metrics(
            pred_by_type.get(t, []), gold_by_type.get(t, [])
        )
    return results


# ─────────────────────────────────────────────────────────────
# Agent vs human, inter-human comparison
# ─────────────────────────────────────────────────────────────

def evaluate_agent_vs_humans(
    agent_records: List[dict],
    human_data: Dict[str, Dict[str, dict]],
    n_annotators: int,
    match_mode: str = "exact",
) -> Dict[str, Any]:
    """
    For each sentence present in both the agent output and ALL human annotators,
    compare agent output against each human annotator and compute inter-human agreement.

    n_annotators: total number of human annotation files — sentences must have
                  annotations from all of them to be included.
    """
    agent_vs_human_scores: List[Dict[str, Any]] = []
    inter_human_scores: List[Dict[str, Any]] = []
    per_type_agent: Dict[str, list] = defaultdict(list)
    per_type_human: Dict[str, list] = defaultdict(list)

    for rec in agent_records:
        sent = rec["sentence"].strip()
        if sent not in human_data:
            print(sent)
            continue

        annotators = human_data[sent]
        # Only evaluate sentences covered by all human annotators
        if len(annotators) < n_annotators:
            continue

        agent_ents = rec.get("final_entities", [])
        agent_rels = rec.get("final_relations", [])

        # Agent vs each human
        for ann_id, ann_data in annotators.items():
            ent_metrics = compute_entity_metrics(
                agent_ents, ann_data["entities"], match_mode=match_mode
            )
            rel_metrics = compute_relation_metrics(
                agent_rels, ann_data.get("relations", [])
            )
            agent_vs_human_scores.append({
                "sentence": sent[:80],
                "annotator": ann_id,
                "entity_f1": ent_metrics["f1"],
                "entity_precision": ent_metrics["precision"],
                "entity_recall": ent_metrics["recall"],
                "relation_f1": rel_metrics["f1"],
            })

            # Per-type accumulation
            type_metrics = compute_per_type_entity_metrics(
                agent_ents, ann_data["entities"], match_mode=match_mode
            )
            for etype, m in type_metrics.items():
                per_type_agent[etype].append(m["f1"])

        # Inter-human: pairwise (sentence already has all annotators)
        ann_ids = list(annotators.keys())
        for i in range(len(ann_ids)):
            for j in range(i + 1, len(ann_ids)):
                a_data = annotators[ann_ids[i]]
                b_data = annotators[ann_ids[j]]
                ent_m = compute_entity_metrics(
                    a_data["entities"], b_data["entities"], match_mode=match_mode
                )
                rel_m = compute_relation_metrics(
                    a_data.get("relations", []),
                    b_data.get("relations", []),
                )
                inter_human_scores.append({
                    "sentence": sent[:80],
                    "pair": f"{ann_ids[i]}↔{ann_ids[j]}",
                    "entity_f1": ent_m["f1"],
                    "relation_f1": rel_m["f1"],
                })

                type_metrics = compute_per_type_entity_metrics(
                    a_data["entities"], b_data["entities"], match_mode=match_mode
                )
                for etype, m in type_metrics.items():
                    per_type_human[etype].append(m["f1"])

    # Aggregate
    def _avg(scores, key):
        vals = [s[key] for s in scores if s[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    avh_ef1 = [s["entity_f1"] for s in agent_vs_human_scores]
    avh_rf1 = [s["relation_f1"] for s in agent_vs_human_scores]
    ih_ef1  = [s["entity_f1"] for s in inter_human_scores]
    ih_rf1  = [s["relation_f1"] for s in inter_human_scores]

    avh_ef1_ci = bootstrap_ci(avh_ef1)
    avh_rf1_ci = bootstrap_ci(avh_rf1)
    ih_ef1_ci  = bootstrap_ci(ih_ef1)
    ih_rf1_ci  = bootstrap_ci(ih_rf1)

    entity_f1_p = two_sample_bootstrap_p(avh_ef1, ih_ef1)
    relation_f1_p = two_sample_bootstrap_p(avh_rf1, ih_rf1)

    return {
        "n_sentences": len(agent_records),
        "n_matched": len({s["sentence"] for s in agent_vs_human_scores}),
        "agent_vs_human": {
            "mean_entity_f1": _avg(agent_vs_human_scores, "entity_f1"),
            "mean_entity_precision": _avg(agent_vs_human_scores, "entity_precision"),
            "mean_entity_recall": _avg(agent_vs_human_scores, "entity_recall"),
            "mean_relation_f1": _avg(agent_vs_human_scores, "relation_f1"),
            "entity_f1_ci95": list(avh_ef1_ci),
            "relation_f1_ci95": list(avh_rf1_ci),
            "per_sentence": agent_vs_human_scores,
        },
        "inter_human": {
            "mean_entity_f1": _avg(inter_human_scores, "entity_f1"),
            "mean_relation_f1": _avg(inter_human_scores, "relation_f1"),
            "entity_f1_ci95": list(ih_ef1_ci),
            "relation_f1_ci95": list(ih_rf1_ci),
            "per_pair": inter_human_scores,
        },
        "significance": {
            "entity_f1_p_value": entity_f1_p,
            "relation_f1_p_value": relation_f1_p,
            "note": "Two-sided permutation test: H0 = agent and inter-human F1 are equal.",
        },
        "per_type_agent_f1": {
            t: sum(v) / len(v) if v else 0.0
            for t, v in sorted(per_type_agent.items())
        },
        "per_type_human_f1": {
            t: sum(v) / len(v) if v else 0.0
            for t, v in sorted(per_type_human.items())
        },
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: Evaluate agent annotation quality against human annotations."
    )
    parser.add_argument("--agent-jsonl", type=Path, required=True)
    parser.add_argument("--human-jsonl", type=Path, nargs="+", required=True,
                        help="One or more per-annotator annotation files")
    parser.add_argument("--match-mode", choices=["exact", "type_only", "text_only"], default="exact")
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON")
    args = parser.parse_args()

    agent_records = load_agent_records(args.agent_jsonl)
    human_data = load_all_human_annotations(args.human_jsonl)
    n_annotators = len(args.human_jsonl)

    results = evaluate_agent_vs_humans(agent_records, human_data, n_annotators, args.match_mode)

    print(f"\n{'='*60}")
    print(f"  LAYER 1 — Output Quality ({args.match_mode} match)")
    print(f"{'='*60}")
    annotator_names = [p.stem for p in args.human_jsonl]
    print(f"  Annotators ({n_annotators}): {', '.join(annotator_names)}")
    print(f"  Sentences: {results['n_matched']} matched / {results['n_sentences']} total")

    avh = results["agent_vs_human"]
    ih = results["inter_human"]
    sig = results["significance"]
    print(f"\n  Agent vs Human (mean ± 95 % bootstrap CI):")
    print(f"    Entity P/R/F1:  {avh['mean_entity_precision']:.3f} / "
          f"{avh['mean_entity_recall']:.3f} / {avh['mean_entity_f1']:.3f} "
          f"  CI {fmt_ci(*avh['entity_f1_ci95'])}")
    print(f"    Relation F1:    {avh['mean_relation_f1']:.3f} "
          f"  CI {fmt_ci(*avh['relation_f1_ci95'])}")
    print(f"\n  Inter-Human (mean ± 95 % bootstrap CI):")
    print(f"    Entity F1:      {ih['mean_entity_f1']:.3f} "
          f"  CI {fmt_ci(*ih['entity_f1_ci95'])}")
    print(f"    Relation F1:    {ih['mean_relation_f1']:.3f} "
          f"  CI {fmt_ci(*ih['relation_f1_ci95'])}")

    gap = avh["mean_entity_f1"] - ih["mean_entity_f1"]
    print(f"\n  Gap (agent − human): {gap:+.3f} entity F1")
    print(f"    Significance (entity F1):   p = {fmt_p(sig['entity_f1_p_value'])}")
    print(f"    Significance (relation F1): p = {fmt_p(sig['relation_f1_p_value'])}")

    print(f"\n  Per-type entity F1 (agent | human):")
    all_types = sorted(set(results["per_type_agent_f1"]) | set(results["per_type_human_f1"]))
    for t in all_types:
        af = results["per_type_agent_f1"].get(t, 0)
        hf = results["per_type_human_f1"].get(t, 0)
        print(f"    {t:35s}  {af:.3f}  |  {hf:.3f}")

    if args.output:
        # Remove per-sentence detail for cleaner output file
        summary = {k: v for k, v in results.items()
                   if k not in ("agent_vs_human", "inter_human")}
        summary["agent_vs_human_summary"] = {
            k: v for k, v in avh.items() if k != "per_sentence"
        }
        summary["inter_human_summary"] = {
            k: v for k, v in ih.items() if k != "per_pair"
        }
        summary["significance"] = results["significance"]
        with args.output.open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
