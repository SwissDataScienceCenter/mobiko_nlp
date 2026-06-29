"""
Deliberation-history analysis (standalone view).

Walks the full message history of every sentence and counts disagreement at
EVERY step — the friction that the Layer-1 final-output metrics hide (by the
last round the agents converge and agreement_score ~= 1.0).

All extraction/aggregation logic lives in the shared `deliberation_history`
module, which Layer 2 (eval_layer2_correlation) and Layer 3/4
(eval_layer34_deliberation) also use, so the three reports cannot drift apart.
This script is a thin CLI that prints the aggregate plus, optionally, the
disagreement-vs-human-difficulty correlation.

Usage:
    python eval_deliberation_history.py --agent-jsonl <results.jsonl> \
        [--human-jsonl <gold...>] [--output history.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Resolve flat imports: deliberation_history is at the package root; its
# correlate_with_human_difficulty lazily imports the eval/ siblings.
_PKG_ROOT = Path(__file__).resolve().parent.parent   # …/multi_agent_annotation
for _p in (_PKG_ROOT, _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deliberation_history import (
    load_records,
    analyze,
    correlate_with_human_difficulty,
)


def print_report(r: Dict[str, Any]) -> None:
    ps = r["per_step"]
    print(f"\n{'='*64}")
    print("  DELIBERATION HISTORY — disagreements at every step")
    print(f"{'='*64}")
    print(f"\n  Sentences: {r['n_sentences']}")

    print("\n  Per-step disagreement counts")
    print(f"    {'round':<10}{'critic turns':>14}{'disagree':>11}{'agree':>9}{'miss':>7}{'dis.rate':>11}")
    rates = ps["disagreement_rate_by_round"]
    for k in sorted(ps["critic_turns_by_round"], key=int):
        ki = int(k) if not isinstance(k, int) else k
        print(f"    R{ki:<9}{ps['critic_turns_by_round'][k]:>14}"
              f"{ps['disagreements_by_round'].get(k, 0):>11}"
              f"{ps['agreements_by_round'].get(k, 0):>9}"
              f"{ps['missing_by_round'].get(k, 0):>7}"
              f"{rates.get(k, float('nan')):>11.3f}")
    print(f"\n    Critic disagreements (all rounds): {ps['total_critic_disagreements']}")
    print(f"    Sentences adjudicated:             {ps['sentences_adjudicated']}")
    print(f"    Adjudicator resolutions:           {ps['adjudicator_resolutions']}")
    print(f"    TOTAL disagreements (all steps):   {ps['total_disagreements_all_steps']}")

    ct = r["convergence_trajectory"]
    print("\n  Convergence trajectory (round-1 disputes -> revised annotation)")
    print(f"    Followed:           {ct['round1_disputes_followed']}")
    print(f"    Resolved by round 2:{ct['resolved_by_round2']:>6}   (rate {ct['resolution_rate']})")
    print(f"    Persisted:          {ct['persisted_to_round2']:>6}")
    print(f"    Resolution direction: {ct['resolution_direction']}")

    print("\n  Disagreement taxonomy by round")
    for rd, data in r["taxonomy_by_round"].items():
        print(f"\n    Round {rd}  severity={data['severity']}")
        for c in data["top_confusions"][:10]:
            print(f"      {c['annotator']:30s} -> {c['critic']:30s}  x{c['count']}")

    gc = r["guideline_coverage"]
    print("\n  Guideline grounding coverage")
    print(f"    Annotator entities:     {gc['annotator_entities']}")
    print(f"    with guideline_step:    {gc['with_guideline_step']}  (coverage {gc['step_coverage']})")
    print(f"    with guideline_rule:    {gc['with_guideline_rule']}  (coverage {gc['rule_coverage']})")
    if gc["step_coverage"] == 0:
        print("    NOTE: guideline_step is unpopulated in this run — use guideline_rule "
              "(see eval_guideline_adherence.py for rule-grounding consistency).")

    diff = r.get("difficulty_prediction")
    if diff:
        print("\n  Does agent disagreement predict human difficulty?")
        print(f"    (difficulty = 1 - pairwise inter-human F1; n={diff['n_sentences']}, "
              f"mean human F1={diff['mean_human_f1']})")
        print(f"    {'agent signal':<34}{'Spearman ρ':>12}{'p':>9}")
        for c in diff["correlations"]:
            rho, p = c["spearman_rho"], c["p_value"]
            star = " *" if (p is not None and p < 0.05) else ""
            rho_s = f"{rho:+.3f}" if rho is not None else "   n/a"
            p_s = f"{p:.4f}" if p is not None else "  n/a"
            print(f"    {c['signal']:<34}{rho_s:>12}{p_s:>9}{star}")


def main():
    ap = argparse.ArgumentParser(description="Deliberation-history disagreement analysis.")
    ap.add_argument("--agent-jsonl", type=Path, required=True)
    ap.add_argument("--human-jsonl", type=Path, nargs="+", default=None,
                    help="Optional: human gold files to test disagreement vs difficulty.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    records = load_records(args.agent_jsonl)
    report = analyze(records)
    if args.human_jsonl:
        diff = correlate_with_human_difficulty(records, args.human_jsonl)
        if diff:
            report["difficulty_prediction"] = diff
    print_report(report)

    if args.output:
        with args.output.open("w", encoding="utf8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()