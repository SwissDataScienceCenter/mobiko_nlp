"""
Layer 2 — Disagreement correlation analysis.

Core question: Do agent disagreements predict human disagreements?

For each entity span, computes:
  - human_disagreed: did human annotators disagree on this span's type?
  - agent_disagreed: did the Critic challenge this span?

Then computes:
  - Per-span 2×2 contingency table
  - Odds ratio + Fisher's exact test
  - Spearman rank correlation at the sentence level
  - Breakdown by entity type (which types are hardest for both?)

Expected inputs
---------------
Agent output (--agent-jsonl): JSONL of DeliberationRecord.
Human annotations (--human-jsonl): multi-annotator JSONL (same as Layer 1).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from eval_utils import (
    bootstrap_ci,
    permutation_p_rho,
    fisher_exact_pvalue,
    fmt_ci,
    fmt_p,
    _mean,
)

SCHEMA_V1_TO_V2: Dict[str, str] = {
    "BIOTIC COLLECTIVE ENTITY":  "BIOTIC ENTITY",
    "ABIOTIC COLLECTIVE ENTITY": "ABIOTIC ENTITY",
}


# ─────────────────────────────────────────────────────────────
# Data loading (shared with Layer 1)
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

    Accepts two formats:
    - Native project JSON: single object {doc_id, sentences:[{text, spans:[{text, type, ...}]}]}
    - Per-annotator JSONL: one record per line {sentence, annotator, entities, relations}
    """
    raw = path.read_text(encoding="utf8").strip()

    if raw.startswith("{"):
        try:
            doc = json.loads(raw)
            if "sentences" in doc:
                return _load_native_json_annotations(doc, path.stem)
        except json.JSONDecodeError:
            pass

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
# Human disagreement computation
# ─────────────────────────────────────────────────────────────

def compute_human_span_disagreements(
    annotators: Dict[str, dict],
) -> Dict[str, Dict[str, Any]]:
    """
    For each unique entity span text across all annotators, determine
    whether annotators agreed on its type.

    Returns {normalized_text: {"types": {type: count}, "disagreed": bool,
             "present_in": int, "total_annotators": int}}
    """
    n_annotators = len(annotators)
    span_info: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    span_presence: Dict[str, int] = defaultdict(int)

    for ann_id, ann_data in annotators.items():
        seen_texts: Set[str] = set()
        for ent in ann_data.get("entities", []):
            text = _normalize(ent.get("text", ""))
            etype = (ent.get("entity_type") or ent.get("type", "")).strip().upper()
            if text and etype:
                span_info[text][etype] += 1
                if text not in seen_texts:
                    span_presence[text] += 1
                    seen_texts.add(text)

    results = {}
    for text, type_counts in span_info.items():
        n_types = len(type_counts)
        present_in = span_presence[text]
        # Disagreement if: multiple types assigned, OR not all annotators included it
        type_disagreed = n_types > 1
        presence_disagreed = present_in < n_annotators
        results[text] = {
            "types": dict(type_counts),
            "type_disagreed": type_disagreed,
            "presence_disagreed": presence_disagreed,
            "disagreed": type_disagreed or presence_disagreed,
            "present_in": present_in,
            "total_annotators": n_annotators,
        }
    return results


# ─────────────────────────────────────────────────────────────
# Agent disagreement extraction from deliberation messages
# ─────────────────────────────────────────────────────────────

def _try_parse_json(text: str) -> Optional[dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    last_obj = None
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "{":
            try:
                obj, end = decoder.raw_decode(cleaned, i)
                if isinstance(obj, dict):
                    last_obj = obj
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return last_obj


def extract_agent_span_disagreements(
    record: dict,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-span agent disagreement from the deliberation messages.

    A span is "agent_disagreed" if the Critic listed it in "disagreements"
    or "missing_annotations" in its last message.

    Returns {normalized_text: {"agent_disagreed": bool, "annotator_type": str,
             "critic_type": str | None, "severity": str | None}}
    """
    messages = record.get("messages", [])

    # Collect all entity spans proposed by the Annotator
    annotator_spans: Dict[str, str] = {}  # norm_text → entity_type
    for m in messages:
        if m.get("agent") != "Annotator":
            continue
        parsed = _try_parse_json(m.get("content", ""))
        if parsed and "entities" in parsed:
            for ent in parsed["entities"]:
                text = _normalize(ent.get("text", ""))
                etype = (ent.get("entity_type") or ent.get("type", "")).strip().upper()
                if text:
                    annotator_spans[text] = etype

    # Find the last Critic message and parse its disagreements
    last_critic = None
    for m in reversed(messages):
        if m.get("agent") == "Critic":
            last_critic = _try_parse_json(m.get("content", ""))
            if last_critic:
                break

    disputed_spans: Dict[str, Dict[str, Any]] = {}

    # Build result for all annotator spans
    for text, ann_type in annotator_spans.items():
        disputed_spans[text] = {
            "agent_disagreed": False,
            "annotator_type": ann_type,
            "critic_type": None,
            "severity": None,
        }

    if last_critic:
        # Mark disagreements
        for d in last_critic.get("disagreements", []):
            target = _normalize(d.get("target", ""))
            # Try to match target to an annotator span
            matched = None
            for span_text in annotator_spans:
                if target in span_text or span_text in target:
                    matched = span_text
                    break
            if matched is None:
                matched = target

            disputed_spans.setdefault(matched, {
                "agent_disagreed": True,
                "annotator_type": d.get("annotator_label", ""),
                "critic_type": d.get("proposed_label", ""),
                "severity": d.get("severity", ""),
            })
            disputed_spans[matched]["agent_disagreed"] = True
            disputed_spans[matched]["critic_type"] = d.get("proposed_label", "")
            disputed_spans[matched]["severity"] = d.get("severity", "")

        # Mark missing annotations as disagreements (presence disagreement)
        for miss in last_critic.get("missing_annotations", []):
            text = _normalize(miss.get("text", ""))
            if text:
                disputed_spans.setdefault(text, {
                    "agent_disagreed": True,
                    "annotator_type": "(missing)",
                    "critic_type": miss.get("entity_type", ""),
                    "severity": "missing",
                })
                disputed_spans[text]["agent_disagreed"] = True

    return disputed_spans


# ─────────────────────────────────────────────────────────────
# Correlation analysis
# ─────────────────────────────────────────────────────────────

def build_contingency_table(
    agent_records: List[dict],
    human_data: Dict[str, Dict[str, dict]],
) -> Dict[str, Any]:
    """
    Build a 2×2 contingency table:
        (human_agree, agent_agree)     → cell_aa
        (human_agree, agent_disagree)  → cell_ad
        (human_disagree, agent_agree)  → cell_da
        (human_disagree, agent_disagree) → cell_dd

    Also returns per-span details for further analysis.
    """
    cell_aa, cell_ad, cell_da, cell_dd = 0, 0, 0, 0
    span_details: List[Dict[str, Any]] = []
    per_type_cells: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for rec in agent_records:
        sent = rec["sentence"].strip()
        if sent not in human_data:
            continue
        annotators = human_data[sent]
        if len(annotators) < 2:
            continue  # need ≥2 annotators for human disagreement

        human_spans = compute_human_span_disagreements(annotators)
        agent_spans = extract_agent_span_disagreements(rec)

        # Union of all spans mentioned by either side
        all_spans = set(human_spans.keys()) | set(agent_spans.keys())

        for span_text in all_spans:
            h = human_spans.get(span_text, {})
            a = agent_spans.get(span_text, {})

            human_disagreed = h.get("disagreed", True)  # if not in human data, treat as disagreed (missing)
            agent_disagreed = a.get("agent_disagreed", False)

            # Determine majority type for per-type breakdown
            span_type = a.get("annotator_type", "")
            if not span_type and h.get("types"):
                span_type = max(h["types"], key=h["types"].get)

            if not human_disagreed and not agent_disagreed:
                cell_aa += 1
                per_type_cells[span_type][0] += 1
            elif not human_disagreed and agent_disagreed:
                cell_ad += 1
                per_type_cells[span_type][1] += 1
            elif human_disagreed and not agent_disagreed:
                cell_da += 1
                per_type_cells[span_type][2] += 1
            else:
                cell_dd += 1
                per_type_cells[span_type][3] += 1

            span_details.append({
                "sentence": sent[:60],
                "span": span_text,
                "human_disagreed": human_disagreed,
                "agent_disagreed": agent_disagreed,
                "human_types": h.get("types", {}),
                "agent_annotator_type": a.get("annotator_type"),
                "agent_critic_type": a.get("critic_type"),
            })

    # Odds ratio (with Haldane-Anscombe continuity correction)
    or_num = (cell_aa + 0.5) * (cell_dd + 0.5)
    or_den = (cell_ad + 0.5) * (cell_da + 0.5)
    odds_ratio = or_num / or_den if or_den > 0 else float("inf")

    total = cell_aa + cell_ad + cell_da + cell_dd
    concordance = (cell_aa + cell_dd) / total if total > 0 else 0.0

    # Fisher's exact test p-value (no continuity correction; exact)
    fisher_p = fisher_exact_pvalue(cell_aa, cell_ad, cell_da, cell_dd)

    # Bootstrap 95% CI on the odds ratio via span-level resampling
    def _or_from_spans(spans):
        aa = sum(1 for s in spans if not s["human_disagreed"] and not s["agent_disagreed"])
        ad = sum(1 for s in spans if not s["human_disagreed"] and s["agent_disagreed"])
        da = sum(1 for s in spans if s["human_disagreed"] and not s["agent_disagreed"])
        dd = sum(1 for s in spans if s["human_disagreed"] and s["agent_disagreed"])
        num = (aa + 0.5) * (dd + 0.5)
        den = (ad + 0.5) * (da + 0.5)
        return math.log(num / den) if den > 0 else float("nan")

    log_or_vals = [_or_from_spans(span_details)]  # placeholder for scalar CI
    log_or_ci: tuple = (float("nan"), float("nan"))
    if len(span_details) >= 4:
        # Resample span_details; compute log-OR for each bootstrap replicate
        import random as _rnd
        rng = _rnd.Random(42)
        n_sp = len(span_details)
        boot_log_ors = []
        for _ in range(2000):
            sample = [span_details[rng.randrange(n_sp)] for _ in range(n_sp)]
            boot_log_ors.append(_or_from_spans(sample))
        boot_log_ors = [v for v in boot_log_ors if not math.isnan(v)]
        if boot_log_ors:
            boot_log_ors.sort()
            lo_i = max(0, int(0.025 * len(boot_log_ors)))
            hi_i = min(len(boot_log_ors) - 1, int(0.975 * len(boot_log_ors)) - 1)
            log_or_ci = (boot_log_ors[lo_i], boot_log_ors[hi_i])

    log_or_obs = math.log(odds_ratio) if odds_ratio not in (0, float("inf")) else float("nan")
    or_ci = (
        math.exp(log_or_ci[0]) if not math.isnan(log_or_ci[0]) else float("nan"),
        math.exp(log_or_ci[1]) if not math.isnan(log_or_ci[1]) else float("nan"),
    )

    return {
        "contingency": {
            "human_agree_agent_agree": cell_aa,
            "human_agree_agent_disagree": cell_ad,
            "human_disagree_agent_agree": cell_da,
            "human_disagree_agent_disagree": cell_dd,
        },
        "total_spans": total,
        "concordance": concordance,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci95": list(or_ci),
        "log_odds_ratio": log_or_obs,
        "fisher_p_value": fisher_p,
        "per_type_contingency": {
            t: {"aa": c[0], "ad": c[1], "da": c[2], "dd": c[3]}
            for t, c in sorted(per_type_cells.items()) if t
        },
        "span_details": span_details,
    }


def compute_sentence_level_correlation(
    agent_records: List[dict],
    human_data: Dict[str, Dict[str, dict]],
) -> Dict[str, Any]:
    """
    For each sentence, compute:
        human_disagreement_rate = fraction of spans with human disagreement
        agent_disagreement_rate = agent agreement_score inverted (1 - score)

    Then compute Spearman rank correlation.
    """
    human_rates: List[float] = []
    agent_rates: List[float] = []
    sentence_details: List[Dict[str, Any]] = []

    for rec in agent_records:
        sent = rec["sentence"].strip()
        if sent not in human_data:
            continue
        annotators = human_data[sent]
        if len(annotators) < 2:
            continue

        human_spans = compute_human_span_disagreements(annotators)
        if not human_spans:
            continue

        h_disagree_rate = (
            sum(1 for s in human_spans.values() if s["disagreed"])
            / len(human_spans)
        )
        # Agent disagreement: use 1 - agreement_score, or compute from critic
        a_score = rec.get("agreement_score")
        if a_score is not None:
            a_disagree_rate = 1.0 - a_score
        else:
            # Fallback: compute from messages
            agent_spans = extract_agent_span_disagreements(rec)
            n_total = len(agent_spans)
            n_disagree = sum(1 for s in agent_spans.values() if s["agent_disagreed"])
            a_disagree_rate = n_disagree / n_total if n_total > 0 else 0.0

        human_rates.append(h_disagree_rate)
        agent_rates.append(a_disagree_rate)
        sentence_details.append({
            "sentence": sent[:60],
            "human_disagreement_rate": round(h_disagree_rate, 3),
            "agent_disagreement_rate": round(a_disagree_rate, 3),
        })

    # Spearman rank correlation + bootstrap CI + permutation p-value
    rho = _spearman_rho(human_rates, agent_rates)

    # Bootstrap CI: resample (human_rate, agent_rate) pairs together
    pairs = list(zip(human_rates, agent_rates))
    rho_ci: tuple = (float("nan"), float("nan"))
    rho_p: Optional[float] = None
    if len(pairs) >= 3:
        import random as _rnd
        rng = _rnd.Random(42)
        n_p = len(pairs)
        boot_rhos = []
        for _ in range(2000):
            sample = [pairs[rng.randrange(n_p)] for _ in range(n_p)]
            xs = [s[0] for s in sample]
            ys = [s[1] for s in sample]
            r = _spearman_rho(xs, ys)
            if r is not None:
                boot_rhos.append(r)
        if boot_rhos:
            boot_rhos.sort()
            lo_i = max(0, int(0.025 * len(boot_rhos)))
            hi_i = min(len(boot_rhos) - 1, int(0.975 * len(boot_rhos)) - 1)
            rho_ci = (boot_rhos[lo_i], boot_rhos[hi_i])

        rho_p = permutation_p_rho(human_rates, agent_rates)

    return {
        "n_sentences": len(human_rates),
        "spearman_rho": rho,
        "spearman_rho_ci95": list(rho_ci),
        "spearman_p_value": rho_p,
        "mean_human_disagreement_rate": (
            sum(human_rates) / len(human_rates) if human_rates else 0
        ),
        "mean_agent_disagreement_rate": (
            sum(agent_rates) / len(agent_rates) if agent_rates else 0
        ),
        "sentence_details": sentence_details,
    }


def _spearman_rho(x: List[float], y: List[float]) -> Optional[float]:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    if n < 3:
        return None

    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - (6 * d_sq) / (n * (n * n - 1))
    return round(rho, 4)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer 2: Correlate agent disagreements with human disagreements."
    )
    parser.add_argument("--agent-jsonl", type=Path, required=True)
    parser.add_argument("--human-jsonl", type=Path, nargs="+", required=True,
                        help="One or more per-annotator annotation files")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    agent_records = load_agent_records(args.agent_jsonl)
    human_data = load_all_human_annotations(args.human_jsonl)

    ct = build_contingency_table(agent_records, human_data)
    corr = compute_sentence_level_correlation(agent_records, human_data)

    print(f"\n{'='*60}")
    print(f"  LAYER 2 — Disagreement Correlation")
    print(f"{'='*60}")

    c = ct["contingency"]
    print(f"\n  Span-level contingency table ({ct['total_spans']} spans):")
    print(f"                        Agent agree  Agent disagree")
    print(f"    Human agree           {c['human_agree_agent_agree']:5d}        {c['human_agree_agent_disagree']:5d}")
    print(f"    Human disagree        {c['human_disagree_agent_agree']:5d}        {c['human_disagree_agent_disagree']:5d}")
    print(f"\n  Concordance:  {ct['concordance']:.3f}")
    or_ci = ct.get("odds_ratio_ci95", [float("nan"), float("nan")])
    print(f"  Odds ratio:   {ct['odds_ratio']:.2f}  (log OR: {ct['log_odds_ratio']:.2f})"
          f"  95% CI [{or_ci[0]:.2f}, {or_ci[1]:.2f}]")
    print(f"  Fisher exact: p = {fmt_p(ct['fisher_p_value'])}")
    if ct["odds_ratio"] > 1:
        print(f"  → Agent disagreement DOES predict human disagreement")
    else:
        print(f"  → Agent disagreement does NOT predict human disagreement")

    print(f"\n  Sentence-level correlation ({corr['n_sentences']} sentences):")
    rho_ci = corr.get("spearman_rho_ci95", [float("nan"), float("nan")])
    print(f"  Spearman ρ:   {corr['spearman_rho']}  "
          f"95% CI {fmt_ci(*rho_ci)}  p = {fmt_p(corr.get('spearman_p_value'))}")
    print(f"  Mean human disagreement rate: {corr['mean_human_disagreement_rate']:.3f}")
    print(f"  Mean agent disagreement rate: {corr['mean_agent_disagreement_rate']:.3f}")

    # Per-type breakdown
    print(f"\n  Per-type contingency (aa/ad/da/dd):")
    for t, cells in ct["per_type_contingency"].items():
        total_t = sum(cells.values())
        dd = cells["dd"]
        print(f"    {t:35s}  {cells['aa']:3d}/{cells['ad']:3d}/{cells['da']:3d}/{dd:3d}  "
              f"(co-disagree: {dd/total_t:.0%})" if total_t > 0 else "")

    if args.output:
        out = {
            "contingency": ct["contingency"],
            "concordance": ct["concordance"],
            "odds_ratio": ct["odds_ratio"],
            "odds_ratio_ci95": ct.get("odds_ratio_ci95"),
            "fisher_p_value": ct.get("fisher_p_value"),
            "spearman_rho": corr["spearman_rho"],
            "spearman_rho_ci95": corr.get("spearman_rho_ci95"),
            "spearman_p_value": corr.get("spearman_p_value"),
            "per_type_contingency": ct["per_type_contingency"],
            "sentence_correlations": corr["sentence_details"],
        }
        with args.output.open("w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
