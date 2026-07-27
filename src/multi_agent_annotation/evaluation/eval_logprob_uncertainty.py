"""
Logprob-uncertainty analysis: does the Annotator agent's TOKEN-PROBABILITY
uncertainty predict human annotation disagreement / difficulty?

Two uncertainty signals live in the agent JSONL (run-2 onward):
  annotator_mean_logprob          mean per-token logprob over the WHOLE Annotator
                                  round-1 JSON generation. Real spread → usable.
  annotator_entity_logprobs[].type_mean_logprob
                                  logprob over each entity's TYPE-label tokens.
                                  Saturated (~0) under constrained/tool_choice
                                  decoding → near-zero variance; reported for
                                  completeness but rarely usable.

Uncertainty is defined as the NEGATED mean logprob, so "more uncertain" is a
LARGER number and a POSITIVE correlation with difficulty matches the other
friction signals (rounds_used, disagreement counts).

Targets, per doubly-annotated sentence:
  human_difficulty     = 1 - pairwise inter-human F1 (text+type)
  human_disagree_rate  = fraction of unique human spans where the humans
                         disagreed (type or presence), via
                         compute_human_span_disagreements.
Internal / calibration checks (agent's own behaviour):
  agent_error          = 1 - mean agent-vs-human F1 (text+type)
  rounds_used, r1_disagreements, total_disagreements (deliberation friction)

Span level: is an entity's TYPE-logprob lower on spans where the two humans
disagree about the type? (group means + permutation test + point-biserial).

Usage:
  python eval_logprob_uncertainty.py --agent-jsonl <run2.jsonl> \
      --human-jsonl <Mark.json> <Davnah.json> [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Tuple

# Resolve flat imports: deliberation_history at the package root; eval siblings
# (eval_layer1/2, eval_utils) in evaluation/.
_PKG_ROOT = Path(__file__).resolve().parent.parent   # …/multi_agent_annotation
for _p in (_PKG_ROOT, _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deliberation_history import load_records, norm, record_signals
from eval_layer2_correlation import (
    load_all_human_annotations,
    compute_human_span_disagreements,
    _normalize,
)
from eval_layer1_output import sentence_f1
from eval_utils import _spearman_rho, permutation_p_rho, fmt_p

N_BOOT = 2000
SEED = 42


def _typed(ents: List[dict]) -> List[dict]:
    """Normalize any entity dict to {text, type} for text+type scoring."""
    return [{"text": e.get("text", ""),
             "type": (e.get("entity_type") or e.get("type") or "").strip().upper()}
            for e in ents]


# ─────────────────────────── per-sentence rows ───────────────────────────

def build_rows(records: List[dict], human: Dict[str, Dict[str, dict]]) -> List[dict]:
    hnorm = {norm(s): a for s, a in human.items()}
    rows: List[dict] = []
    for rec in records:
        sent = rec.get("sentence", "")
        sig = record_signals(rec)

        # --- agent uncertainty signals (negated logprob = uncertainty) ---
        mean_lp = rec.get("annotator_mean_logprob")
        ent_lps = [e["type_mean_logprob"] for e in rec.get("annotator_entity_logprobs", [])
                   if e.get("type_mean_logprob") is not None]
        # entropy fields (present only in runs with top_logprobs capture; higher
        # entropy = more uncertain, so NO negation, unlike logprob)
        ent_ents = [e["type_mean_entropy"] for e in rec.get("annotator_entity_logprobs", [])
                    if e.get("type_mean_entropy") is not None]
        ent_maxents = [e["type_max_entropy"] for e in rec.get("annotator_entity_logprobs", [])
                       if e.get("type_max_entropy") is not None]
        confs = [e["confidence"] for e in rec.get("final_entities", [])
                 if isinstance(e.get("confidence"), (int, float))]

        row: Dict[str, Any] = {
            "sentence": sent,
            "unc_whole": (-mean_lp) if mean_lp is not None else None,
            "unc_type_mean": (-st.mean(ent_lps)) if ent_lps else None,
            "unc_type_min": (-min(ent_lps)) if ent_lps else None,  # most-uncertain type token
            "unc_whole_entropy": rec.get("annotator_mean_entropy"),
            "unc_type_entropy_mean": st.mean(ent_ents) if ent_ents else None,
            "unc_type_entropy_max": max(ent_maxents) if ent_maxents else None,  # most-contested token
            "unc_selfconf": (1.0 - st.mean(confs)) if confs else None,  # self-reported (baseline)
            "rounds_used": sig.get("rounds_used"),
            "r1_disagreements": sig.get("r1_disagreements"),
            "total_disagreements": sig.get("total_disagreements_all_steps"),
        }

        # --- human targets (only doubly-annotated sentences) ---
        anns = hnorm.get(norm(sent))
        if anns and len(anns) >= 2:
            ids = list(anns.keys())
            f1 = sentence_f1(_typed(anns[ids[0]]["entities"]),
                             _typed(anns[ids[1]]["entities"]), mode="text_type") or 0.0
            row["human_difficulty"] = 1.0 - f1

            dis = compute_human_span_disagreements(anns)
            if dis:
                row["human_disagree_rate"] = sum(1 for v in dis.values() if v["disagreed"]) / len(dis)

            agent_typed = _typed(rec.get("final_entities", []))
            accs = [sentence_f1(agent_typed, _typed(anns[i]["entities"]), mode="text_type") or 0.0
                    for i in ids]
            row["agent_error"] = 1.0 - (sum(accs) / len(accs))

        rows.append(row)
    return rows


# ───────────────────────────── correlations ─────────────────────────────

def corr(rows: List[dict], sig: str, tgt: str) -> Tuple[int, Optional[float], Optional[float]]:
    pairs = [(r[sig], r[tgt]) for r in rows
             if r.get(sig) is not None and r.get(tgt) is not None]
    if len(pairs) < 3:
        return len(pairs), None, None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    return len(pairs), _spearman_rho(x, y), permutation_p_rho(x, y, n_boot=N_BOOT)


def _corr_table(rows: List[dict], signals: List[Tuple[str, str]],
                targets: List[Tuple[str, str]]) -> List[dict]:
    out = []
    for sig_key, sig_lbl in signals:
        entry: Dict[str, Any] = {"signal": sig_lbl}
        for tgt_key, _ in targets:
            n, rho, p = corr(rows, sig_key, tgt_key)
            entry[tgt_key] = {"n": n, "rho": rho, "p": p}
        out.append(entry)
    return out


# ─────────────────────────── span-level (type) ──────────────────────────

def _group_compare(pairs: List[Tuple[Optional[float], bool]],
                   higher_is_uncertain: bool) -> Dict[str, Any]:
    """
    pairs: [(metric_value, human_type_disagreed_bool), ...]; None values skipped.
    Compares the metric on human-type-disagreed vs agreed spans (group means +
    permutation p) and the point-biserial Spearman of UNCERTAINTY vs disagreement
    (uncertainty = metric if higher_is_uncertain else −metric).
    """
    vd = [v for v, d in pairs if d and v is not None]
    va = [v for v, d in pairs if (not d) and v is not None]
    xs = [(v if higher_is_uncertain else -v) for v, _ in pairs if v is not None]
    ys = [1.0 if d else 0.0 for v, d in pairs if v is not None]
    res: Dict[str, Any] = {
        "n_disagreed": len(vd), "n_agreed": len(va),
        "mean_uncertain_disagreed": st.mean(vd) if vd else None,
        "mean_uncertain_agreed": st.mean(va) if va else None,
        "point_biserial_rho": _spearman_rho(xs, ys) if len(xs) >= 3 else None,
        "point_biserial_p": permutation_p_rho(xs, ys, n_boot=N_BOOT) if len(xs) >= 3 else None,
        "mean_diff_perm_p": None,
    }
    if vd and va:
        obs = abs(st.mean(vd) - st.mean(va))
        pool = vd + va
        n_d = len(vd)
        rng = Random(SEED)
        count = 0
        for _ in range(N_BOOT):
            rng.shuffle(pool)
            if abs(st.mean(pool[:n_d]) - st.mean(pool[n_d:])) >= obs:
                count += 1
        res["mean_diff_perm_p"] = (count + 1) / (N_BOOT + 1)
    return res


def span_level(records: List[dict], human: Dict[str, Dict[str, dict]]) -> Dict[str, Any]:
    """Per agent entity: does its type-uncertainty (logprob / entropy) track
    whether the two humans disagree on that span's type?"""
    hnorm = {norm(s): a for s, a in human.items()}
    lp_pairs: List[Tuple[Optional[float], bool]] = []     # (type_mean_logprob, disagreed)
    ent_pairs: List[Tuple[Optional[float], bool]] = []    # (type_max_entropy, disagreed)
    n_matched = 0
    for rec in records:
        anns = hnorm.get(norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        dis = compute_human_span_disagreements(anns)
        for e in rec.get("annotator_entity_logprobs", []):
            info = dis.get(_normalize(e.get("text", "")))
            if info is None:
                continue  # agent span not located by either human
            n_matched += 1
            d = bool(info["type_disagreed"])
            lp_pairs.append((e.get("type_mean_logprob"), d))
            ent_pairs.append((e.get("type_max_entropy"), d))
    return {
        "n_matched_spans": n_matched,
        "logprob": _group_compare(lp_pairs, higher_is_uncertain=False),
        "entropy": _group_compare(ent_pairs, higher_is_uncertain=True),
    }


def saturation_summary(records: List[dict]) -> Dict[str, Any]:
    allent = [e["type_mean_logprob"] for r in records
              for e in (r.get("annotator_entity_logprobs") or [])
              if e.get("type_mean_logprob") is not None]
    maxents = [e["type_max_entropy"] for r in records
               for e in (r.get("annotator_entity_logprobs") or [])
               if e.get("type_max_entropy") is not None]
    means = [r["annotator_mean_logprob"] for r in records
             if r.get("annotator_mean_logprob") is not None]
    return {
        "n_records": len(records),
        "n_entities": len(allent),
        "whole_logprob": {
            "min": min(means), "median": st.median(means), "max": max(means),
            "stdev": st.pstdev(means),
        } if means else None,
        "type_logprob": {
            "min": min(allent), "median": st.median(allent), "max": max(allent),
            "frac_gt_-0.01": sum(1 for x in allent if x > -0.01) / len(allent),
        } if allent else None,
        "type_max_entropy": {
            "n": len(maxents), "min": min(maxents), "median": st.median(maxents),
            "max": max(maxents), "frac_gt_0.1": sum(1 for x in maxents if x > 0.1) / len(maxents),
        } if maxents else None,
    }


# ───────────────────────────── reporting ─────────────────────────────────

def _fmt_rho_p(cell: dict) -> str:
    rho, p = cell["rho"], cell["p"]
    if rho is None or cell.get("n", 0) < 3:
        return "n/a"
    star = " *" if (p is not None and p < 0.05) else ""
    p_s = f"{p:.4f}" if p is not None else "n/a"
    return f"{rho:+.3f}  {p_s}{star}"


def print_report(rep: Dict[str, Any]) -> None:
    print("\n" + "=" * 74)
    print("  AGENT LOGPROB UNCERTAINTY  →  HUMAN DISAGREEMENT / DIFFICULTY")
    print("=" * 74)

    sat = rep["saturation"]
    print(f"\n  Records: {sat['n_records']}   Entities with type-logprob: {sat['n_entities']}")
    if sat["whole_logprob"]:
        w = sat["whole_logprob"]
        print(f"  annotator_mean_logprob (whole output): "
              f"min {w['min']:.3f}  median {w['median']:.3f}  max {w['max']:.3f}  "
              f"stdev {w['stdev']:.3f}   ← usable spread")
    if sat["type_logprob"]:
        t = sat["type_logprob"]
        print(f"  type_mean_logprob (type tokens only):  "
              f"min {t['min']:.3f}  median {t['median']:.5f}  max {t['max']:.6f}")
        print(f"      {t['frac_gt_-0.01']*100:.1f}% of entities have logprob > -0.01 "
              f"(p>0.99) → SATURATED, near-zero variance")
    if sat["type_max_entropy"]:
        h = sat["type_max_entropy"]
        print(f"  type_max_entropy (top-k entropy):      "
              f"median {h['median']:.4f}  max {h['max']:.4f}  "
              f"({h['frac_gt_0.1']*100:.1f}% of entities > 0.1 nats — contested type)")
    else:
        print("  type_max_entropy: NOT PRESENT in this run (no top_logprobs capture) "
              "— re-run the pipeline with ANNOTATOR_TOP_LOGPROBS set to populate it.")

    tgts = rep["targets"]
    hdr = "".join(f"{lbl:>21}" for _, lbl in tgts)
    print("\n  ── Sentence-level Spearman ρ (signal vs target; * = p<0.05) ──")
    print(f"    {'signal':<24}" + hdr)
    for entry in rep["human_table"]:
        cells = "".join(f"{_fmt_rho_p(entry[k]):>21}" for k, _ in tgts)
        print(f"    {entry['signal']:<24}{cells}")
    n_human = rep["n_doubly_annotated"]
    print(f"    (n≈{n_human} doubly-annotated sentences; uncertainty = −mean logprob, "
          "so +ρ = more uncertain → harder)")

    itg = rep["internal_targets"]
    ihdr = "".join(f"{lbl:>21}" for _, lbl in itg)
    print("\n  ── Calibration / internal: does uncertainty track the agent's own"
          " behaviour? ──")
    print(f"    {'signal':<24}" + ihdr)
    for entry in rep["internal_table"]:
        cells = "".join(f"{_fmt_rho_p(entry[k]):>21}" for k, _ in itg)
        print(f"    {entry['signal']:<24}{cells}")
    print("    (agent_error = 1 − agent-vs-human F1; +ρ here = uncertainty predicts"
          " agent's own errors)")

    sl = rep["span_level"]
    lp = sl["logprob"]
    print("\n  ── Span level: type-logprob on spans where humans DISAGREE vs AGREE on type ──")
    print(f"    matched agent spans: {sl['n_matched_spans']}   "
          f"(human type-disagreed {lp['n_disagreed']} / "
          f"agreed {lp['n_agreed']})")
    if lp["mean_uncertain_disagreed"] is not None and lp["mean_uncertain_agreed"] is not None:
        print(f"    mean type-logprob  disagreed {lp['mean_uncertain_disagreed']:.5f}   "
              f"agreed {lp['mean_uncertain_agreed']:.5f}   "
              f"(perm p {fmt_p(lp.get('mean_diff_perm_p'))})")
    print(f"    point-biserial ρ(uncertainty, human_type_disagree): "
          f"{lp['point_biserial_rho']:+.3f}  p {fmt_p(lp['point_biserial_p'])}"
          if lp["point_biserial_rho"] is not None else "    point-biserial: n/a")


def main():
    ap = argparse.ArgumentParser(description="Agent logprob uncertainty vs human disagreement/difficulty.")
    ap.add_argument("--agent-jsonl", type=Path, required=True)
    ap.add_argument("--human-jsonl", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    records = load_records(args.agent_jsonl)
    human = load_all_human_annotations([Path(p) for p in args.human_jsonl])
    rows = build_rows(records, human)

    human_targets = [("human_difficulty", "human difficulty"),
                     ("human_disagree_rate", "human disagr.rate")]
    internal_targets = [("rounds_used", "rounds_used"),
                        ("r1_disagreements", "R1 disagreements"),
                        ("agent_error", "agent error")]
    signals = [("unc_whole", "logprob (whole)"),
               ("unc_type_min", "logprob (type-min)"),
               ("unc_type_mean", "logprob (type-mean)"),
               ("unc_selfconf", "self-confidence"),
               ("rounds_used", "rounds_used"),
               ("r1_disagreements", "R1 disagreements"),
               ("total_disagreements", "total disagreements")]
    internal_signals = [("unc_whole", "logprob (whole)"),
                        ("unc_type_min", "logprob (type-min)"),
                        ("unc_selfconf", "self-confidence")]

    report = {
        "n_doubly_annotated": sum(1 for r in rows if "human_difficulty" in r),
        "saturation": saturation_summary(records),
        "targets": human_targets,
        "internal_targets": internal_targets,
        "human_table": _corr_table(rows, signals, human_targets),
        "internal_table": _corr_table(rows, internal_signals, internal_targets),
        "span_level": span_level(records, human),
    }
    print_report(report)

    if args.output:
        with args.output.open("w", encoding="utf8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()