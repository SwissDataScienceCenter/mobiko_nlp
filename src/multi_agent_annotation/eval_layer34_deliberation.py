"""
Layers 3 & 4 — Deliberation process analysis and guideline improvement signal.

REBUILT to count disagreement from the FULL deliberation history (every Critic
round + the Adjudicator's resolutions) and to expose the number of rounds, via
the shared `deliberation_history` module. The previous version flattened all
rounds into one taxonomy, never counted Adjudicator resolutions, and computed a
broken "critic acceptance" rate (0/12) by comparing against the wrong round.

Layer 3 (process analysis):
  3a. Per-step disagreement counts + rates (round 1 vs round 2 vs adjudication)
      and the disagreement taxonomy split by round.
  3b. Tool usage patterns.
  3c. Convergence dynamics: of the round-1 disputes, how many resolved by round 2,
      and the resolution direction (accepted Critic / kept own / third / dropped).
  3d. Flagged-item analysis: are flagged items genuinely harder (needs human data)?

Layer 4 (guideline signal):
  - Guideline grounding coverage (guideline_step vs guideline_rule).
  - Top recurring confusion patterns → candidate guideline amendments.

Only requires agent output (deliberation JSONL). Human annotations are optional
(flagged-item IAA comparison).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval_utils import bootstrap_ci, two_sample_bootstrap_p, fmt_ci, fmt_p
from deliberation_history import (
    load_records,
    try_parse_json,
    analyze,
)


# ─────────────────────────────────────────────────────────────
# Layer 3b: Tool usage patterns
# ─────────────────────────────────────────────────────────────

def analyze_tool_usage(records: List[dict]) -> Dict[str, Any]:
    """Aggregate tool-call patterns from deliberation messages."""
    agent_tool_counts: Dict[str, Counter] = defaultdict(Counter)
    tool_results_empty: Counter = Counter()
    total_messages_with_tools = 0

    for rec in records:
        for m in rec.get("messages", []):
            agent = m.get("agent", "")
            tool_calls = m.get("tool_calls", [])
            if not tool_calls:
                continue
            total_messages_with_tools += 1
            for tc in tool_calls:
                tool_name = tc.get("tool_name", "unknown")
                agent_tool_counts[agent][tool_name] += 1
                if tc.get("result", "") in ("[]", "{}", "", "null", None):
                    tool_results_empty[tool_name] += 1

    return {
        "total_messages_with_tools": total_messages_with_tools,
        "per_agent_tool_usage": {
            agent: dict(counts.most_common())
            for agent, counts in sorted(agent_tool_counts.items())
        },
        "empty_tool_results": dict(tool_results_empty.most_common()),
    }


# ─────────────────────────────────────────────────────────────
# Layer 3d: Flagged items analysis (optional human data)
# ─────────────────────────────────────────────────────────────

def analyze_flagged_items(
    records: List[dict],
    human_data: Optional[Dict[str, Dict[str, dict]]] = None,
) -> Dict[str, Any]:
    """
    Items flagged by the Adjudicator for human review. If human data is given,
    compare inter-human IAA on flagged vs unflagged sentences.
    """
    flagged_items: List[Dict[str, Any]] = []
    flagged_sentences: set = set()

    for rec in records:
        sent = rec.get("sentence", "").strip()
        for m in rec.get("messages", []):
            if m.get("agent") != "Adjudicator":
                continue
            parsed = try_parse_json(m.get("content", ""))
            if parsed:
                for item in parsed.get("flagged_for_human_review", []):
                    flagged_items.append({"sentence": sent[:80], "item": item})
                    flagged_sentences.add(sent)

    result: Dict[str, Any] = {
        "total_flagged_items": len(flagged_items),
        "sentences_with_flags": len(flagged_sentences),
        "flagged_items": flagged_items[:50],
    }

    if human_data:
        from eval_layer1_output import sentence_f1

        # sentence_f1 expects {text, type}; the human loader emits {text,
        # entity_type}. Map to text+type (renamed compute_entity_metrics "exact").
        def _typed(ents):
            return [{"text": e.get("text", ""),
                     "type": e.get("entity_type", e.get("type", "")).strip().upper()}
                    for e in ents]

        flagged_f1s, unflagged_f1s = [], []
        for rec in records:
            sent = rec.get("sentence", "").strip()
            if sent not in human_data or len(human_data[sent]) < 2:
                continue
            ann_ids = list(human_data[sent].keys())
            pair_f1s = []
            for i in range(len(ann_ids)):
                for j in range(i + 1, len(ann_ids)):
                    f1 = sentence_f1(
                        _typed(human_data[sent][ann_ids[i]]["entities"]),
                        _typed(human_data[sent][ann_ids[j]]["entities"]),
                        mode="text_type",
                    ) or 0.0
                    pair_f1s.append(f1)
            avg_f1 = sum(pair_f1s) / len(pair_f1s) if pair_f1s else 0
            (flagged_f1s if sent in flagged_sentences else unflagged_f1s).append(avg_f1)

        if flagged_f1s and unflagged_f1s:
            result["flagged_mean_human_iaa"] = sum(flagged_f1s) / len(flagged_f1s)
            result["unflagged_mean_human_iaa"] = sum(unflagged_f1s) / len(unflagged_f1s)
            result["iaa_gap"] = result["unflagged_mean_human_iaa"] - result["flagged_mean_human_iaa"]
            result["flagged_iaa_ci95"] = list(bootstrap_ci(flagged_f1s))
            result["unflagged_iaa_ci95"] = list(bootstrap_ci(unflagged_f1s))
            result["iaa_gap_p_value"] = two_sample_bootstrap_p(unflagged_f1s, flagged_f1s)

    return result


# ─────────────────────────────────────────────────────────────
# Layer 4: Guideline improvement signal
# ─────────────────────────────────────────────────────────────

def build_guideline_recommendations(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Top recurring confusion patterns (across all rounds) → amendment hints."""
    recs = []
    for conf in report["top_confusions_all_rounds"][:10]:
        recs.append({
            "pattern": f"{conf['annotator']} → {conf['critic']}",
            "frequency": conf["count"],
            "recommendation": (
                f"Confusion between {conf['annotator']} and {conf['critic']} "
                f"occurred {conf['count']}× across all rounds. Consider adding a "
                f"disambiguation rule to the guideline."
            ),
        })
    return recs


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_report(report: Dict[str, Any], tools: Dict[str, Any],
                 flagged: Dict[str, Any]) -> None:
    ps = report["per_step"]
    conv = report["convergence_trajectory"]
    gc = report["guideline_coverage"]

    print(f"\n{'='*60}")
    print("  LAYER 3 — Deliberation Process Analysis")
    print(f"{'='*60}")

    print("\n  3a. Disagreement counts at every step")
    print(f"      {'round':<8}{'critic turns':>14}{'disagree':>11}{'agree':>9}{'miss':>7}{'dis.rate':>11}")
    rates = ps["disagreement_rate_by_round"]
    for k in sorted(ps["critic_turns_by_round"], key=int):
        print(f"      R{int(k):<7}{ps['critic_turns_by_round'][k]:>14}"
              f"{ps['disagreements_by_round'].get(k, 0):>11}"
              f"{ps['agreements_by_round'].get(k, 0):>9}"
              f"{ps['missing_by_round'].get(k, 0):>7}"
              f"{rates.get(k, float('nan')):>11.3f}")
    print(f"\n      Critic disagreements (all rounds): {ps['total_critic_disagreements']}")
    print(f"      Sentences adjudicated:             {ps['sentences_adjudicated']}")
    print(f"      Adjudicator resolutions:           {ps['adjudicator_resolutions']}")
    print(f"      TOTAL disagreements (all steps):   {ps['total_disagreements_all_steps']}")

    print("\n      Disagreement taxonomy by round:")
    for rd, data in report["taxonomy_by_round"].items():
        print(f"        Round {rd}  severity={data['severity']}")
        for c in data["top_confusions"][:8]:
            print(f"          {c['annotator']:30s} → {c['critic']:30s} ×{c['count']}")
    if report["missing_entity_types"]:
        print("\n      Missing entity types (Critic found, Annotator missed):")
        for t, c in list(report["missing_entity_types"].items())[:10]:
            print(f"        {t}: {c}")

    print("\n  3b. Tool usage")
    print(f"      Messages with tool calls: {tools['total_messages_with_tools']}")
    for agent, counts in tools["per_agent_tool_usage"].items():
        print(f"      {agent}: {counts}")

    print("\n  3c. Convergence dynamics")
    print(f"      Round-1 disputes followed:  {conv['round1_disputes_followed']}")
    print(f"      Resolved by round 2:        {conv['resolved_by_round2']}  "
          f"(rate {conv['resolution_rate']})")
    print(f"      Persisted to round 2:       {conv['persisted_to_round2']}")
    print(f"      Resolution direction:       {conv['resolution_direction']}")

    print("\n  3d. Flagged items")
    print(f"      Items flagged for human review: {flagged['total_flagged_items']}")
    print(f"      Sentences with flags:           {flagged['sentences_with_flags']}")
    if "iaa_gap" in flagged:
        f_ci = flagged.get("flagged_iaa_ci95") or [float("nan")] * 2
        u_ci = flagged.get("unflagged_iaa_ci95") or [float("nan")] * 2
        print(f"      Flagged sentences human IAA:   {flagged['flagged_mean_human_iaa']:.3f}  CI {fmt_ci(*f_ci)}")
        print(f"      Unflagged sentences human IAA: {flagged['unflagged_mean_human_iaa']:.3f}  CI {fmt_ci(*u_ci)}")
        print(f"      Gap: {flagged['iaa_gap']:.3f}  p = {fmt_p(flagged.get('iaa_gap_p_value'))}  "
              f"({'flagged harder ✓' if flagged['iaa_gap'] > 0 else 'no difference'})")

    print(f"\n{'='*60}")
    print("  LAYER 4 — Guideline Improvement Signal")
    print(f"{'='*60}")
    print("\n  Guideline grounding coverage")
    print(f"    Annotator entities:  {gc['annotator_entities']}")
    print(f"    guideline_step:      {gc['with_guideline_step']}  (coverage {gc['step_coverage']})")
    print(f"    guideline_rule:      {gc['with_guideline_rule']}  (coverage {gc['rule_coverage']})")
    if gc["step_coverage"] == 0:
        print("    NOTE: guideline_step unpopulated in this run — use guideline_rule "
              "(see eval_guideline_adherence.py).")

    print("\n  Recommendations for guideline amendments (top confusions, all rounds):")
    for i, r in enumerate(build_guideline_recommendations(report)[:5], 1):
        print(f"    {i}. {r['pattern']} (×{r['frequency']})")
        print(f"       {r['recommendation']}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layers 3+4: Deliberation analysis (history-based) + guideline signal."
    )
    parser.add_argument("--agent-jsonl", type=Path, required=True)
    parser.add_argument("--human-jsonl", type=Path, nargs="+", default=None,
                        help="Optional: human annotations for flagged-item IAA comparison")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.agent_jsonl)

    human_data = None
    if args.human_jsonl:
        from eval_layer2_correlation import load_all_human_annotations
        human_data = load_all_human_annotations(args.human_jsonl)

    report = analyze(records)
    tools = analyze_tool_usage(records)
    flagged = analyze_flagged_items(records, human_data)

    print_report(report, tools, flagged)

    if args.output:
        out = {
            **report,
            "tool_usage": tools,
            "flagged_items": {k: v for k, v in flagged.items() if k != "flagged_items"},
            "guideline_recommendations": build_guideline_recommendations(report),
        }
        with args.output.open("w", encoding="utf8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()