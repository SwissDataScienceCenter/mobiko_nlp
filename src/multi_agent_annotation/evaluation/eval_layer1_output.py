"""
Layer 1 — Output quality evaluation.

Scores the agent's spans against human annotations as PRECISION / RECALL / F1
(agent = prediction, human = gold), using the SAME offset-based micro matching
as scripts/human_iaa_report.py — so Layer 1 reconciles with the inter-annotator
agreement (IAA) report and the human–human ceiling instead of diverging from it.

What changed (and why) vs the old Layer 1
-----------------------------------------
  * Matching is OFFSET-based (char spans), primary mode, in two flavours:
        boundary = (start_char, end_char)          — did the agent mark the span?
        strict   = (start_char, end_char, type)     — …with the right type?
    Text-based matching (surface form, position-agnostic) is kept as a SECONDARY
    lens only — it can't be reconciled with the offset IAA ceiling.
  * Aggregation is MICRO: TP/FP/FN are pooled across all sentences, then P/R/F1
    is computed once (with a bootstrap CI that resamples sentences). The old
    code averaged per-sentence F1, which produced incoherent triples
    (mean F1 below both mean P and mean R) and unstable numbers on short
    sentences.
  * Three gold references are reported SEPARATELY: each human annotator, plus
    the CONSENSUS = spans ALL humans agree on (per-sentence multiset
    intersection). This is the agent-vs-consensus alignment story.

This module is the single source of truth: scripts/agent_vs_consensus_prf.py is
a thin wrapper kept for back-compat and produces identical offset numbers.

Expected input formats
----------------------
Agent output (--agent-jsonl):  JSONL, one DeliberationRecord per line.
    {"sentence": "...",
     "final_entities":  [{"text", "entity_type", "start", "end", ...}],
     "final_relations": [{"relation", "e1": {...}, "e2": {...}}], ...}

Human annotations (--human-jsonl):  one or more files, one per annotator.
  Annotator name = filename stem. Two formats accepted:
    Native project JSON:  {"doc_id", "sentences":[{"text","spans":[{start_char,end_char,type,text}]}]}
    Per-annotator JSONL:  {"sentence","annotator","entities":[{text,type/entity_type,start_char,end_char}], ...}

Eval set (mirrors the IAA "doubly/triply-annotated" convention):
  - per individual human: sentences the agent AND that human both marked (non-empty)
  - consensus:            sentences the agent AND ALL humans marked (non-empty)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve flat sibling imports (eval_utils + the shared core at the package root)
# however this module is launched.
_PKG_ROOT = Path(__file__).resolve().parent.parent   # …/multi_agent_annotation
for _p in (_PKG_ROOT, _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_utils import fmt_ci, fmt_p, two_sample_bootstrap_p

SCHEMA_V1_TO_V2: Dict[str, str] = {
    "BIOTIC COLLECTIVE ENTITY":  "BIOTIC ENTITY",
    "ABIOTIC COLLECTIVE ENTITY": "ABIOTIC ENTITY",
}

N_BOOT = 5000
SEED = 42
RELATION_PREFIX = "RELATION"


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _normalize_label(label: str) -> str:
    upper = (label or "").strip().upper()
    return SCHEMA_V1_TO_V2.get(upper, upper)


def _norm_entity(e: dict) -> Optional[dict]:
    """
    Normalize an entity from any source to {text, type, start_char, end_char}.
    Accepts offset keys start/end (agent) or start_char/end_char (human native).
    Drops RELATION-typed pseudo-spans. start_char/end_char may be None when the
    source has no offsets (then offset modes skip it; text modes still work).
    """
    etype = _normalize_label(e.get("entity_type", e.get("type", "")))
    if etype.startswith(RELATION_PREFIX):
        return None
    start = e.get("start_char", e.get("start"))
    end = e.get("end_char", e.get("end"))
    return {
        "text": e.get("text", ""),
        "type": etype,
        "start_char": start,
        "end_char": end,
    }


def load_agent_records(path: Path) -> List[dict]:
    """Return [{sentence, entities:[norm], relations:[...]}] from agent JSONL."""
    records = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            rec = json.loads(line)
            ents = [_norm_entity(e) for e in rec.get("final_entities", [])]
            records.append({
                "sentence": rec.get("sentence", ""),
                "entities": [e for e in ents if e],
                "relations": rec.get("final_relations", []),
            })
    return records


def load_human_annotations(path: Path) -> Dict[str, Dict[str, dict]]:
    """{sentence_key: {annotator: {entities:[norm], relations:[...]}}} for one file."""
    raw = path.read_text(encoding="utf8").strip()

    # Native project JSON: {doc_id, sentences:[{text, spans:[...]}]}
    if raw.startswith("{"):
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            doc = None
        if doc and "sentences" in doc:
            data: Dict[str, Dict[str, dict]] = {}
            for s in doc["sentences"]:
                ents = [_norm_entity(sp) for sp in s.get("spans", [])]
                data[_normalize(s["text"])] = {
                    path.stem: {
                        "entities": [e for e in ents if e],
                        "relations": [r for r in s.get("relations", [])],
                    }
                }
            return data

    # Per-annotator JSONL: one record per sentence.
    data = defaultdict(dict)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        ents = [_norm_entity(e) for e in obj.get("entities", [])]
        annotator = obj.get("annotator", path.stem)
        data[_normalize(obj["sentence"])][annotator] = {
            "entities": [e for e in ents if e],
            "relations": obj.get("relations", []),
        }
    return dict(data)


def load_all_human_annotations(paths: List[Path]) -> Tuple[Dict[str, Dict[str, dict]], List[str]]:
    """Merge per-annotator files. Returns (merged, ordered_annotator_names)."""
    merged: Dict[str, Dict[str, dict]] = defaultdict(dict)
    names: List[str] = []
    for path in paths:
        loaded = load_human_annotations(path)
        for sent, annotators in loaded.items():
            merged[sent].update(annotators)
            for a in annotators:
                if a not in names:
                    names.append(a)
    return dict(merged), names


# ─────────────────────────────────────────────────────────────
# Matching keys (one per mode)
# ─────────────────────────────────────────────────────────────

def _has_offsets(e: dict) -> bool:
    return e.get("start_char") is not None and e.get("end_char") is not None


def _keyer(mode: str):
    """Return a key function for the given matching mode (or None to skip e)."""
    if mode == "boundary":
        return lambda e: (e["start_char"], e["end_char"]) if _has_offsets(e) else None
    if mode == "strict":
        return lambda e: (e["start_char"], e["end_char"], e["type"]) if _has_offsets(e) else None
    if mode == "text_type":
        return lambda e: (_normalize(e["text"]), e["type"])
    if mode == "text_only":
        return lambda e: _normalize(e["text"])
    raise ValueError(f"unknown mode {mode}")


def _counter(entities: List[dict], keyfn) -> Counter:
    c = Counter()
    for e in entities:
        k = keyfn(e)
        if k is not None:
            c[k] += 1
    return c


OFFSET_MODES = [("boundary", "BOUNDARY  (offset only)"),
                ("strict", "STRICT    (offset + type)")]
TEXT_MODES = [("text_type", "TEXT+TYPE (surface form + type)"),
              ("text_only", "TEXT-ONLY (surface form)")]


# ─────────────────────────────────────────────────────────────
# Micro P/R/F1 with sentence-bootstrap CI
# ─────────────────────────────────────────────────────────────

def _prf(tp: int, n_pred: int, n_gold: int) -> Dict[str, Any]:
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    f = (2 * p * r / (p + r)) if (p and r) else (0.0 if (n_pred or n_gold) else None)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def _bootstrap_f1_ci(per_sentence: List[Tuple[int, int, int]]) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for micro F1, resampling sentences (each = (tp, n_pred, n_gold))."""
    if len(per_sentence) < 2:
        return (None, None)
    rng = random.Random(SEED)
    n = len(per_sentence)
    f1s = []
    for _ in range(N_BOOT):
        tp = npred = ngold = 0
        for _ in range(n):
            t, p, g = per_sentence[rng.randrange(n)]
            tp += t; npred += p; ngold += g
        denom = npred + ngold
        f1s.append(2 * tp / denom if denom else 0.0)
    f1s.sort()
    return (f1s[int(0.025 * N_BOOT)], f1s[int(0.975 * N_BOOT)])


def sentence_f1(pred_ents: List[dict], gold_ents: List[dict], mode: str = "strict") -> Optional[float]:
    """Single-sentence F1 (used only for the two-sample significance test)."""
    keyfn = _keyer(mode)
    pc = _counter(pred_ents, keyfn)
    gc = _counter(gold_ents, keyfn)
    return _prf(sum((pc & gc).values()), sum(pc.values()), sum(gc.values()))["f1"]


def _cohen_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    """Cohen's κ on a list of (label_a, label_b) decisions."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({l for p in pairs for l in p})
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def label_only_agreement(
    eval_sentences: List[Tuple[List[dict], List[dict]]],
) -> Dict[str, Any]:
    """
    Labeling agreement CONDITIONED on a boundary-matched span — isolates the
    type decision from detection. Returns observed type agreement (accuracy)
    and chance-corrected Cohen's κ over all co-located span pairs.
    """
    pairs: List[Tuple[str, str]] = []
    for pred_ents, gold_ents in eval_sentences:
        pmap = {(e["start_char"], e["end_char"]): e["type"]
                for e in pred_ents if _has_offsets(e)}
        gmap = {(e["start_char"], e["end_char"]): e["type"]
                for e in gold_ents if _has_offsets(e)}
        for k in pmap.keys() & gmap.keys():
            pairs.append((pmap[k], gmap[k]))
    n = len(pairs)
    return {
        "n_boundary_matched_pairs": n,
        "observed_type_agreement": (sum(1 for a, b in pairs if a == b) / n) if n else None,
        "cohen_kappa": _cohen_kappa(pairs),
    }


def per_type_strict_f1(
    eval_sentences: List[Tuple[List[dict], List[dict]]],
) -> Dict[str, Dict[str, Any]]:
    """Micro strict-offset F1 per entity type. Returns {type: {f1, gold}}."""
    keyfn = _keyer("strict")
    tp = Counter(); pred = Counter(); gold = Counter()
    for pred_ents, gold_ents in eval_sentences:
        pc = _counter(pred_ents, keyfn)
        gc = _counter(gold_ents, keyfn)
        for k, c in pc.items():
            pred[k[2]] += c
        for k, c in gc.items():
            gold[k[2]] += c
        for k, c in (pc & gc).items():
            tp[k[2]] += c
    types = set(pred) | set(gold)
    return {
        t: {"f1": _prf(tp[t], pred[t], gold[t])["f1"], "gold": gold[t]}
        for t in types
    }


def score_reference(
    eval_sentences: List[Tuple[List[dict], List[dict]]],  # (pred_ents, gold_ents) per sentence
    mode: str,
) -> Dict[str, Any]:
    """Micro P/R/F1 (+F1 bootstrap CI) for one reference under one matching mode."""
    keyfn = _keyer(mode)
    tp = npred = ngold = 0
    per_sentence: List[Tuple[int, int, int]] = []
    for pred_ents, gold_ents in eval_sentences:
        pc = _counter(pred_ents, keyfn)
        gc = _counter(gold_ents, keyfn)
        s_tp = sum((pc & gc).values())
        s_pred = sum(pc.values())
        s_gold = sum(gc.values())
        tp += s_tp; npred += s_pred; ngold += s_gold
        per_sentence.append((s_tp, s_pred, s_gold))
    out = _prf(tp, npred, ngold)
    out["f1_ci95"] = list(_bootstrap_f1_ci(per_sentence))
    return out


# ─────────────────────────────────────────────────────────────
# Evaluation orchestration
# ─────────────────────────────────────────────────────────────

def evaluate(
    agent_records: List[dict],
    human_data: Dict[str, Dict[str, dict]],
    annotator_names: List[str],
) -> Dict[str, Any]:
    n_ann = len(annotator_names)
    consensus_label = (f"Consensus ({'∩'.join(annotator_names)})"
                       if n_ann <= 3 else f"Consensus (all {n_ann})")
    refs = annotator_names + [consensus_label]

    # Per-reference eval sentence lists of (agent_ents, gold_ents).
    eval_sents: Dict[str, List[Tuple[List[dict], List[dict]]]] = {r: [] for r in refs}
    # Inter-human eval (pairwise, doubly-annotated) — the human ceiling.
    human_pairs: List[Tuple[str, str]] = [
        (annotator_names[i], annotator_names[j])
        for i in range(n_ann) for j in range(i + 1, n_ann)
    ]
    human_eval: Dict[Tuple[str, str], List[Tuple[List[dict], List[dict]]]] = {
        p: [] for p in human_pairs
    }

    n_total = len(agent_records)
    for rec in agent_records:
        key = _normalize(rec["sentence"])
        annotators = human_data.get(key)
        if not annotators:
            continue
        agent_ents = rec["entities"]

        for name in annotator_names:
            ad = annotators.get(name)
            if ad is None:
                continue
            gold = ad["entities"]
            if agent_ents and gold:  # doubly-annotated
                eval_sents[name].append((agent_ents, gold))

        # Inter-human pairs (independent of agent coverage).
        for a, b in human_pairs:
            if annotators.get(a) and annotators.get(b) and \
               annotators[a]["entities"] and annotators[b]["entities"]:
                human_eval[(a, b)].append((annotators[a]["entities"], annotators[b]["entities"]))

    # Build consensus gold per mode lazily inside scoring (needs all annotator ents),
    # so re-collect consensus sentences with the raw annotator entity lists.
    consensus_raw: List[Tuple[List[dict], List[List[dict]]]] = []
    for rec in agent_records:
        key = _normalize(rec["sentence"])
        annotators = human_data.get(key)
        if not annotators:
            continue
        agent_ents = rec["entities"]
        if agent_ents and all(annotators.get(n) and annotators[n]["entities"] for n in annotator_names):
            consensus_raw.append((agent_ents, [annotators[n]["entities"] for n in annotator_names]))

    # ── score every (reference, mode) ──
    all_modes = OFFSET_MODES + TEXT_MODES
    results: Dict[str, Any] = {
        "n_sentences": n_total,
        "consensus_label": consensus_label,
        "references": {},
        "inter_human": {},
        "eval_sentence_counts": {},
    }

    for name in annotator_names:
        results["eval_sentence_counts"][name] = len(eval_sents[name])
        results["references"][name] = {
            mode: score_reference(eval_sents[name], mode) for mode, _ in all_modes
        }
    results["eval_sentence_counts"][consensus_label] = len(consensus_raw)
    results["references"][consensus_label] = {}
    for mode, _ in all_modes:
        keyfn = _keyer(mode)
        eval_pairs = []
        for agent_ents, ann_ent_lists in consensus_raw:
            # consensus gold = multiset intersection of all annotators' keys
            counters = [_counter(lst, keyfn) for lst in ann_ent_lists]
            cons = counters[0].copy()
            for c in counters[1:]:
                cons &= c
            # materialize consensus "entities" as repeated key tokens via a tiny shim:
            # score_reference recomputes counters with keyfn, so pass dicts that re-key
            # to the same value. Simplest: pass through a pre-counted path.
            eval_pairs.append((agent_ents, cons))
        results["references"][consensus_label][mode] = _score_against_counter(eval_pairs, keyfn)

    # ── inter-human ceiling (offset modes only; text modes available too) ──
    for (a, b), pairs in human_eval.items():
        results["inter_human"][f"{a}|{b}"] = {
            mode: score_reference(pairs, mode) for mode, _ in all_modes
        }

    # ── per-type strict-offset F1: agent vs each human + human ceiling ──
    results["per_type_strict_f1"] = {
        name: per_type_strict_f1(eval_sents[name]) for name in annotator_names
    }
    results["per_type_strict_f1"]["__human__"] = {
        f"{a}|{b}": per_type_strict_f1(pairs) for (a, b), pairs in human_eval.items()
    }

    # ── label-only agreement (type κ on boundary-matched spans) ──
    # Isolates labeling from detection. Agent vs each human + human ceiling.
    results["label_only"] = {
        name: label_only_agreement(eval_sents[name]) for name in annotator_names
    }
    for (a, b), pairs in human_eval.items():
        results["label_only"][f"{a} ↔ {b}"] = label_only_agreement(pairs)

    # ── significance: agent-vs-human vs inter-human (per-sentence strict F1) ──
    # Two-sample permutation test on per-sentence strict-offset F1. Agent values
    # pool agent-vs-each-human; human values are the inter-human pairs. This is
    # a distributional test only — the headline P/R/F1 above stay micro.
    agent_sent_f1 = [sentence_f1(p, g, "strict")
                     for name in annotator_names for p, g in eval_sents[name]]
    human_sent_f1 = [sentence_f1(p, g, "strict")
                     for pairs in human_eval.values() for p, g in pairs]
    agent_sent_f1 = [v for v in agent_sent_f1 if v is not None]
    human_sent_f1 = [v for v in human_sent_f1 if v is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    results["significance"] = {
        "agent_mean_sentence_strict_f1": mean(agent_sent_f1),
        "human_mean_sentence_strict_f1": mean(human_sent_f1),
        "n_agent_sentences": len(agent_sent_f1),
        "n_human_sentences": len(human_sent_f1),
        "strict_f1_p_value": two_sample_bootstrap_p(agent_sent_f1, human_sent_f1),
        "note": "Two-sided permutation test on per-sentence strict-offset F1; "
                "H0 = agent-vs-human and inter-human F1 have equal means.",
    }

    return results


def _score_against_counter(
    eval_pairs: List[Tuple[List[dict], Counter]],
    keyfn,
) -> Dict[str, Any]:
    """Like score_reference but gold is already a key Counter (the consensus set)."""
    tp = npred = ngold = 0
    per_sentence: List[Tuple[int, int, int]] = []
    for pred_ents, gold_counter in eval_pairs:
        pc = _counter(pred_ents, keyfn)
        s_tp = sum((pc & gold_counter).values())
        s_pred = sum(pc.values())
        s_gold = sum(gold_counter.values())
        tp += s_tp; npred += s_pred; ngold += s_gold
        per_sentence.append((s_tp, s_pred, s_gold))
    out = _prf(tp, npred, ngold)
    out["f1_ci95"] = list(_bootstrap_f1_ci(per_sentence))
    return out


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def _print_mode_table(results: Dict[str, Any], modes, refs: List[str]) -> None:
    for mode, label in modes:
        print(f"  {label}")
        print(f"    {'reference (gold)':<24}{'P':>8}{'R':>8}{'F1':>8}"
              f"{'F1 95% CI':>18}{'tp':>7}{'pred':>7}{'gold':>7}")
        for ref in refs:
            d = results["references"][ref][mode]
            ci = fmt_ci(*d["f1_ci95"]) if d["f1_ci95"][0] is not None else "      —      "
            p = f"{d['precision']:.3f}" if d['precision'] is not None else "  —  "
            r = f"{d['recall']:.3f}" if d['recall'] is not None else "  —  "
            f = f"{d['f1']:.3f}" if d['f1'] is not None else "  —  "
            print(f"    {ref:<24}{p:>8}{r:>8}{f:>8}{ci:>18}"
                  f"{d['tp']:>7}{d['n_pred']:>7}{d['n_gold']:>7}")
        print()


def print_report(results: Dict[str, Any], annotator_names: List[str]) -> None:
    consensus_label = results["consensus_label"]
    refs = annotator_names + [consensus_label]

    print(f"\n{'='*72}")
    print("  LAYER 1 — Output Quality (offset micro P/R/F1; agent = prediction)")
    print(f"{'='*72}")
    print(f"  Annotators ({len(annotator_names)}): {', '.join(annotator_names)}")
    print(f"  Sentences (agent total): {results['n_sentences']}")
    print("  Eval set per reference (doubly/triply-annotated, non-empty both sides):")
    for ref in refs:
        print(f"    {ref:<26} {results['eval_sentence_counts'][ref]} sentences")
    print()

    print("  ── PRIMARY: offset matching (comparable to the human IAA ceiling) ──\n")
    _print_mode_table(results, OFFSET_MODES, refs)

    if results["inter_human"]:
        print("  ── HUMAN CEILING: inter-annotator agreement (offset, symmetric F1) ──")
        for pair, modes_d in results["inter_human"].items():
            a, b = pair.split("|")
            bnd = modes_d["boundary"]["f1"]
            strict = modes_d["strict"]["f1"]
            print(f"    {a} ↔ {b:<18}  boundary F1={bnd:.3f}   strict F1={strict:.3f}")
        print()

    # ── label-only agreement (typing isolated from detection) ──
    lo = results.get("label_only", {})
    if lo:
        print("  ── LABEL-ONLY agreement: type decision on boundary-matched spans ──")
        print(f"    {'pair':<26}{'matched':>9}{'obs.agr':>9}{'Cohen κ':>9}")
        human_keys = [k for k in lo if "↔" in k]
        agent_keys = [k for k in lo if "↔" not in k]
        for k in human_keys + agent_keys:
            d = lo[k]
            label = k if "↔" in k else f"agent ↔ {k}"
            oa = f"{d['observed_type_agreement']:.3f}" if d['observed_type_agreement'] is not None else "—"
            kp = f"{d['cohen_kappa']:.3f}" if d['cohen_kappa'] is not None else "—"
            print(f"    {label:<26}{d['n_boundary_matched_pairs']:>9}{oa:>9}{kp:>9}")
        print("    (κ ~0.78 = 'substantial'; once a span is co-located, typing mostly agrees)\n")

    # ── significance vs the human ceiling ──
    sig = results.get("significance")
    if sig and sig.get("strict_f1_p_value") is not None:
        print("  ── SIGNIFICANCE: agent-vs-human vs inter-human (per-sentence strict F1) ──")
        print(f"    agent mean sentence F1:  {sig['agent_mean_sentence_strict_f1']:.3f} "
              f"(n={sig['n_agent_sentences']})")
        print(f"    human mean sentence F1:  {sig['human_mean_sentence_strict_f1']:.3f} "
              f"(n={sig['n_human_sentences']})")
        gap = sig['agent_mean_sentence_strict_f1'] - sig['human_mean_sentence_strict_f1']
        print(f"    gap (agent − human):     {gap:+.3f}")
        print(f"    two-sided permutation p: {fmt_p(sig['strict_f1_p_value'])}")
        print("    (p > 0.05 ⇒ agent indistinguishable from a human annotator)\n")

    # ── per-type strict-offset F1 ──
    pt = results.get("per_type_strict_f1", {})
    if pt:
        human_pt = pt.get("__human__", {})
        # merge human pairs into one ceiling column (first pair if 2 annotators)
        ceiling = next(iter(human_pt.values()), {})
        all_types = set()
        for name in annotator_names:
            all_types |= set(pt.get(name, {}))
        all_types |= set(ceiling)
        # sort by gold support under the first annotator (descending)
        first = pt.get(annotator_names[0], {})
        ordered = sorted(all_types, key=lambda t: -first.get(t, {}).get("gold", 0))
        print("  ── PER-TYPE strict-offset F1 (agent vs each human | human ceiling) ──")
        hdr = "    " + f"{'entity type':<26}" + "".join(f"{n[:10]:>11}" for n in annotator_names)
        hdr += f"{'HUMAN':>11}{'support':>9}"
        print(hdr)
        for t in ordered:
            row = f"    {t:<26}"
            for name in annotator_names:
                f1 = pt.get(name, {}).get(t, {}).get("f1")
                row += f"{(f'{f1:.3f}' if f1 is not None else '—'):>11}"
            cf1 = ceiling.get(t, {}).get("f1")
            row += f"{(f'{cf1:.3f}' if cf1 is not None else '—'):>11}"
            support = first.get(t, {}).get("gold", 0)
            row += f"{support:>9}"
            print(row)
        print()

    print("  ── SECONDARY lens: text matching (position-agnostic; NOT IAA-comparable) ──\n")
    _print_mode_table(results, TEXT_MODES, refs)

    print("  Reading:")
    print("   - Offset tables are the headline and use the same matching as the IAA")
    print("     report, so agent-vs-human F1 sits directly against the human ceiling.")
    print("   - Consensus = spans ALL humans agree on. RECALL is the headline there")
    print("     (fraction of agreed spans the agent recovered); precision is a lower")
    print("     bound (agent spans matching only one human count as false positives).")
    print("   - Text matching ignores position and dedups identical strings — a more")
    print("     lenient 'right mention regardless of extent' view, not comparable to IAA.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: agent annotation quality vs human gold (offset micro P/R/F1)."
    )
    parser.add_argument("--agent-jsonl", type=Path, required=True)
    parser.add_argument("--human-jsonl", type=Path, nargs="+", required=True,
                        help="One or more per-annotator annotation files")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for the human files, in the same order "
                             "(e.g. --names Mark Davnah). Default: filename stems.")
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON")
    # accepted for back-compat; matching mode is no longer a single choice
    parser.add_argument("--match-mode", default=None,
                        help="(deprecated) all modes are now reported together")
    args = parser.parse_args()

    if args.match_mode:
        print(f"  [note] --match-mode {args.match_mode} ignored: "
              "Layer 1 now reports offset (boundary+strict) and text modes together.")

    agent_records = load_agent_records(args.agent_jsonl)
    human_data, annotator_names = load_all_human_annotations(args.human_jsonl)

    if args.names:
        if len(args.names) != len(args.human_jsonl):
            parser.error(f"--names ({len(args.names)}) must match --human-jsonl "
                         f"({len(args.human_jsonl)}) count")
        stem_to_name = {p.stem: n for p, n in zip(args.human_jsonl, args.names)}
        if set(stem_to_name) != set(annotator_names):
            parser.error("--names only supported when each human file yields exactly "
                         "one annotator (its filename stem); got annotators "
                         f"{annotator_names}")
        for sent in human_data:
            human_data[sent] = {stem_to_name.get(a, a): v
                                for a, v in human_data[sent].items()}
        annotator_names = list(args.names)

    results = evaluate(agent_records, human_data, annotator_names)
    print_report(results, annotator_names)

    if args.output:
        with args.output.open("w", encoding="utf8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
