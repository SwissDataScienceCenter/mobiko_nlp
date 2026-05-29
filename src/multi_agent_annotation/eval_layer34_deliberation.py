"""
Layers 3 & 4 — Deliberation process analysis and guideline improvement signal.

Mines the multi-agent deliberation logs to extract:

Layer 3 (process analysis):
  - Disagreement taxonomy: Annotator→Critic type confusion patterns
  - Tool usage patterns: which tools are called during disagreements
  - Convergence dynamics: did the Annotator revise? toward human consensus?
  - Flagged-item analysis: are flagged items genuinely harder?

Layer 4 (guideline signal):
  - Guideline rule citation heat map: which rules are most contested
  - Top-K underspecified patterns: recurring disputes → guideline gaps
  - Category-specific recommendations for guideline amendments

Only requires agent output (deliberation JSONL).
Optionally accepts human annotations for convergence direction analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_utils import (
    bootstrap_ci,
    two_sample_bootstrap_p,
    binomial_wilson_ci,
    fmt_ci,
    fmt_p,
)


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


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


# ─────────────────────────────────────────────────────────────
# Layer 3a: Disagreement taxonomy
# ─────────────────────────────────────────────────────────────

def build_disagreement_taxonomy(
    records: List[dict],
) -> Dict[str, Any]:
    """
    Aggregate CriticDisagreement objects across all sentences into
    a taxonomy of type confusions.

    Returns:
        confusion_matrix: {(annotator_label, critic_label): count}
        severity_distribution: {severity: count}
        top_confusions: sorted list of (pattern, count)
        per_sentence_disputes: [{sentence, n_disputes, types}]
    """
    confusion: Counter = Counter()
    severity_counts: Counter = Counter()
    missing_types: Counter = Counter()
    per_sentence: List[Dict[str, Any]] = []

    for rec in records:
        messages = rec.get("messages", [])
        sentence = rec.get("sentence", "")[:80]

        sentence_disputes = []
        for m in messages:
            if m.get("agent") != "Critic":
                continue
            parsed = _try_parse_json(m.get("content", ""))
            if not parsed:
                continue

            for d in parsed.get("disagreements", []):
                ann_label = d.get("annotator_label", "?").strip().upper()
                crit_label = d.get("proposed_label", "?").strip().upper()
                severity = d.get("severity", "unspecified").lower()

                confusion[(ann_label, crit_label)] += 1
                severity_counts[severity] += 1
                sentence_disputes.append({
                    "target": d.get("target", ""),
                    "from": ann_label,
                    "to": crit_label,
                    "severity": severity,
                    "guideline_ref": d.get("guideline_reference", ""),
                })

            for miss in parsed.get("missing_annotations", []):
                mtype = (miss.get("entity_type") or "").strip().upper()
                if mtype:
                    missing_types[mtype] += 1

        if sentence_disputes:
            per_sentence.append({
                "sentence": sentence,
                "n_disputes": len(sentence_disputes),
                "disputes": sentence_disputes,
            })

    top_confusions = sorted(confusion.items(), key=lambda x: -x[1])

    return {
        "confusion_matrix": {f"{a} → {b}": c for (a, b), c in top_confusions},
        "total_disagreements": sum(confusion.values()),
        "unique_patterns": len(confusion),
        "severity_distribution": dict(severity_counts),
        "missing_entity_types": dict(missing_types.most_common()),
        "top_confusions": [
            {"annotator": a, "critic": b, "count": c}
            for (a, b), c in top_confusions[:20]
        ],
        "sentences_with_disputes": len(per_sentence),
        "per_sentence_disputes": per_sentence,
    }


# ─────────────────────────────────────────────────────────────
# Layer 3b: Tool usage patterns
# ─────────────────────────────────────────────────────────────

def analyze_tool_usage(records: List[dict]) -> Dict[str, Any]:
    """
    Aggregate tool call patterns from deliberation messages.

    For each agent, counts:
      - How often each tool is called
      - Whether tool calls co-occur with disagreements
      - Tool call sequences (what tools are called together)
    """
    agent_tool_counts: Dict[str, Counter] = defaultdict(Counter)
    tool_during_disagreement: Counter = Counter()
    tool_sequences: Counter = Counter()
    tool_results_empty: Counter = Counter()
    total_messages_with_tools = 0

    for rec in records:
        messages = rec.get("messages", [])

        # Find which messages have tool calls
        for m in messages:
            agent = m.get("agent", "")
            tool_calls = m.get("tool_calls", [])
            if not tool_calls:
                continue

            total_messages_with_tools += 1
            tools_in_msg = []

            for tc in tool_calls:
                tool_name = tc.get("tool_name", "unknown")
                agent_tool_counts[agent][tool_name] += 1
                tools_in_msg.append(tool_name)

                # Check if result was empty/unhelpful
                result = tc.get("result", "")
                if result in ("[]", "{}", "", "null", None):
                    tool_results_empty[tool_name] += 1

            # Record tool sequence
            if len(tools_in_msg) > 1:
                tool_sequences[tuple(sorted(tools_in_msg))] += 1

        # Check tool usage during critic disagreements
        for i, m in enumerate(messages):
            if m.get("agent") != "Critic":
                continue
            parsed = _try_parse_json(m.get("content", ""))
            if not parsed or not parsed.get("disagreements"):
                continue
            # Look at the Critic's tool calls in this or preceding message
            for j in range(max(0, i - 1), min(len(messages), i + 2)):
                if messages[j].get("agent") == "Critic":
                    for tc in messages[j].get("tool_calls", []):
                        tool_during_disagreement[tc.get("tool_name", "")] += 1

    return {
        "total_messages_with_tools": total_messages_with_tools,
        "per_agent_tool_usage": {
            agent: dict(counts.most_common())
            for agent, counts in sorted(agent_tool_counts.items())
        },
        "tools_during_disagreements": dict(tool_during_disagreement.most_common()),
        "tool_co_occurrences": {
            " + ".join(k): v for k, v in tool_sequences.most_common(10)
        },
        "empty_tool_results": dict(tool_results_empty.most_common()),
    }


# ─────────────────────────────────────────────────────────────
# Layer 3c: Convergence dynamics
# ─────────────────────────────────────────────────────────────

def analyze_convergence(records: List[dict]) -> Dict[str, Any]:
    """
    For multi-round deliberations, track what the Annotator changed
    after Critic feedback.

    Tracks:
      - How many entities were re-typed, added, removed between rounds
      - Whether the Annotator accepted or rejected Critic suggestions
      - Overall convergence rate
    """
    total_retyped = 0
    total_added = 0
    total_removed = 0
    total_critic_suggestions = 0
    total_accepted = 0
    multi_round_sentences = 0
    single_round_sentences = 0

    for rec in records:
        messages = rec.get("messages", [])
        rounds_used = rec.get("rounds_used", 0)

        if rounds_used <= 1:
            single_round_sentences += 1
            continue

        multi_round_sentences += 1

        # Collect annotator outputs by round
        annotator_outputs = []
        critic_outputs = []
        for m in messages:
            parsed = _try_parse_json(m.get("content", ""))
            if not parsed:
                continue
            if m.get("agent") == "Annotator" and "entities" in parsed:
                annotator_outputs.append(parsed)
            elif m.get("agent") == "Critic" and "disagreements" in parsed:
                critic_outputs.append(parsed)

        if len(annotator_outputs) < 2:
            continue

        # Compare round N to round N+1
        for r in range(len(annotator_outputs) - 1):
            prev_ents = {
                _normalize(e.get("text", "")): (e.get("entity_type") or e.get("type", "")).upper()
                for e in annotator_outputs[r].get("entities", [])
            }
            curr_ents = {
                _normalize(e.get("text", "")): (e.get("entity_type") or e.get("type", "")).upper()
                for e in annotator_outputs[r + 1].get("entities", [])
            }

            prev_texts = set(prev_ents.keys())
            curr_texts = set(curr_ents.keys())
            total_added += len(curr_texts - prev_texts)
            total_removed += len(prev_texts - curr_texts)

            for text in prev_texts & curr_texts:
                if prev_ents[text] != curr_ents[text]:
                    total_retyped += 1

        # Check if Critic suggestions were accepted
        if critic_outputs and len(annotator_outputs) >= 2:
            last_critic = critic_outputs[-1] if len(critic_outputs) >= 2 else critic_outputs[0]
            last_annotator = annotator_outputs[-1]

            last_ents = {
                _normalize(e.get("text", "")): (e.get("entity_type") or e.get("type", "")).upper()
                for e in last_annotator.get("entities", [])
            }

            for d in last_critic.get("disagreements", []):
                total_critic_suggestions += 1
                proposed = d.get("proposed_label", "").upper()
                target = _normalize(d.get("target", ""))
                # Check if annotator adopted the proposed label
                for span_text, span_type in last_ents.items():
                    if target in span_text or span_text in target:
                        if span_type == proposed:
                            total_accepted += 1
                        break

    acceptance_rate = (
        total_accepted / total_critic_suggestions
        if total_critic_suggestions > 0 else None
    )
    acceptance_rate_ci95 = (
        list(binomial_wilson_ci(total_accepted, total_critic_suggestions))
        if total_critic_suggestions > 0 else None
    )

    return {
        "multi_round_sentences": multi_round_sentences,
        "single_round_sentences": single_round_sentences,
        "total_entities_retyped": total_retyped,
        "total_entities_added": total_added,
        "total_entities_removed": total_removed,
        "total_critic_suggestions": total_critic_suggestions,
        "suggestions_accepted": total_accepted,
        "acceptance_rate": acceptance_rate,
        "acceptance_rate_ci95": acceptance_rate_ci95,
    }


# ─────────────────────────────────────────────────────────────
# Layer 3d: Flagged items analysis
# ─────────────────────────────────────────────────────────────

def analyze_flagged_items(
    records: List[dict],
    human_data: Optional[Dict[str, Dict[str, dict]]] = None,
) -> Dict[str, Any]:
    """
    Extract items flagged by the Adjudicator for human review.
    If human data is provided, compare agreement scores on flagged
    vs unflagged sentences.
    """
    flagged_items: List[Dict[str, Any]] = []
    flagged_sentences: set = set()

    for rec in records:
        sent = rec.get("sentence", "").strip()
        for m in rec.get("messages", []):
            if m.get("agent") != "Adjudicator":
                continue
            parsed = _try_parse_json(m.get("content", ""))
            if parsed:
                for item in parsed.get("flagged_for_human_review", []):
                    flagged_items.append({"sentence": sent[:80], "item": item})
                    flagged_sentences.add(sent)

    result: Dict[str, Any] = {
        "total_flagged_items": len(flagged_items),
        "sentences_with_flags": len(flagged_sentences),
        "flagged_items": flagged_items[:50],  # cap for readability
    }

    # If human data is available, compare IAA on flagged vs unflagged
    if human_data:
        from eval_layer1_output import compute_entity_metrics

        flagged_f1s = []
        unflagged_f1s = []

        for rec in records:
            sent = rec.get("sentence", "").strip()
            if sent not in human_data or len(human_data[sent]) < 2:
                continue

            annotators = human_data[sent]
            ann_ids = list(annotators.keys())

            # Compute pairwise inter-human F1
            pair_f1s = []
            for i in range(len(ann_ids)):
                for j in range(i + 1, len(ann_ids)):
                    m = compute_entity_metrics(
                        annotators[ann_ids[i]]["entities"],
                        annotators[ann_ids[j]]["entities"],
                    )
                    pair_f1s.append(m["f1"])

            avg_f1 = sum(pair_f1s) / len(pair_f1s) if pair_f1s else 0

            if sent in flagged_sentences:
                flagged_f1s.append(avg_f1)
            else:
                unflagged_f1s.append(avg_f1)

        if flagged_f1s and unflagged_f1s:
            result["flagged_mean_human_iaa"] = sum(flagged_f1s) / len(flagged_f1s)
            result["unflagged_mean_human_iaa"] = sum(unflagged_f1s) / len(unflagged_f1s)
            result["iaa_gap"] = result["unflagged_mean_human_iaa"] - result["flagged_mean_human_iaa"]

            # Bootstrap CI on each group's mean and permutation p-value for the gap
            result["flagged_iaa_ci95"] = list(bootstrap_ci(flagged_f1s))
            result["unflagged_iaa_ci95"] = list(bootstrap_ci(unflagged_f1s))
            result["iaa_gap_p_value"] = two_sample_bootstrap_p(unflagged_f1s, flagged_f1s)

    return result


# ─────────────────────────────────────────────────────────────
# Layer 4: Guideline improvement signal
# ─────────────────────────────────────────────────────────────

def extract_guideline_signals(records: List[dict]) -> Dict[str, Any]:
    """
    Identify where the annotation guideline is underspecified by
    mining Critic messages for:
      - Which guideline rules are cited during disagreements
      - Recurring confusion patterns that suggest missing rules
      - Tool searches that returned empty results (guideline gaps)
    """
    rule_citations: Counter = Counter()
    confusion_to_rules: Dict[str, List[str]] = defaultdict(list)
    empty_guideline_searches: List[Dict[str, Any]] = []
    pattern_examples: Dict[str, List[str]] = defaultdict(list)

    for rec in records:
        sentence = rec.get("sentence", "")[:80]
        messages = rec.get("messages", [])

        for m in messages:
            if m.get("agent") != "Critic":
                continue
            parsed = _try_parse_json(m.get("content", ""))
            if not parsed:
                continue

            for d in parsed.get("disagreements", []):
                ref = d.get("guideline_reference", "").strip()
                if ref:
                    rule_citations[ref] += 1
                    pattern = f"{d.get('annotator_label', '?')} → {d.get('proposed_label', '?')}"
                    confusion_to_rules[pattern].append(ref)
                    if len(pattern_examples[pattern]) < 3:
                        pattern_examples[pattern].append(
                            f"{d.get('target', '?')} in: {sentence}"
                        )

            # Check for empty guideline_search results in tool calls
            for tc in m.get("tool_calls", []):
                if tc.get("tool_name") == "guideline_search":
                    result = tc.get("result", "")
                    try:
                        parsed_result = json.loads(result) if result else []
                    except (json.JSONDecodeError, TypeError):
                        parsed_result = []
                    if not parsed_result:
                        empty_guideline_searches.append({
                            "query": tc.get("arguments", {}).get("query", ""),
                            "sentence": sentence,
                        })

    # Build recommendations
    taxonomy = build_disagreement_taxonomy(records)
    recommendations: List[Dict[str, Any]] = []

    for conf in taxonomy["top_confusions"][:10]:
        pattern = f"{conf['annotator']} → {conf['critic']}"
        rules_cited = confusion_to_rules.get(pattern, [])
        examples = pattern_examples.get(pattern, [])

        recommendations.append({
            "pattern": pattern,
            "frequency": conf["count"],
            "guideline_rules_cited": list(set(rules_cited)),
            "examples": examples,
            "recommendation": (
                f"Confusion between {conf['annotator']} and {conf['critic']} "
                f"occurred {conf['count']} times. "
                + (f"Rules cited: {', '.join(set(rules_cited))}. "
                   if rules_cited else "No guideline rule was cited — this pattern may lack a rule. ")
                + "Consider adding a disambiguation rule to the guideline."
            ),
        })

    return {
        "guideline_rule_citations": dict(rule_citations.most_common()),
        "total_rule_citations": sum(rule_citations.values()),
        "unique_rules_cited": len(rule_citations),
        "empty_guideline_searches": empty_guideline_searches,
        "n_empty_searches": len(empty_guideline_searches),
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────────────────────
# Combined report
# ─────────────────────────────────────────────────────────────

def full_deliberation_report(
    records: List[dict],
    human_data: Optional[Dict[str, Dict[str, dict]]] = None,
) -> Dict[str, Any]:
    """Run all Layer 3 + Layer 4 analyses and return a combined report."""
    return {
        "layer3_disagreement_taxonomy": build_disagreement_taxonomy(records),
        "layer3_tool_usage": analyze_tool_usage(records),
        "layer3_convergence": analyze_convergence(records),
        "layer3_flagged_items": analyze_flagged_items(records, human_data),
        "layer4_guideline_signals": extract_guideline_signals(records),
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layers 3+4: Deliberation analysis and guideline improvement signal."
    )
    parser.add_argument("--agent-jsonl", type=Path, required=True)
    parser.add_argument("--human-jsonl", type=Path, nargs="+", default=None,
                        help="Optional: human annotations for flagged-item IAA comparison")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_agent_records(args.agent_jsonl)

    human_data = None
    if args.human_jsonl:
        from eval_layer2_correlation import load_all_human_annotations
        human_data = load_all_human_annotations(args.human_jsonl)

    report = full_deliberation_report(records, human_data)

    # ── Print Layer 3 ─────────────────────────────────────
    tax = report["layer3_disagreement_taxonomy"]
    print(f"\n{'='*60}")
    print(f"  LAYER 3 — Deliberation Process Analysis")
    print(f"{'='*60}")

    print(f"\n  3a. Disagreement taxonomy")
    print(f"      Total disagreements: {tax['total_disagreements']}")
    print(f"      Unique confusion patterns: {tax['unique_patterns']}")
    print(f"      Severity: {tax['severity_distribution']}")
    print(f"\n      Top confusion patterns:")
    for conf in tax["top_confusions"][:10]:
        print(f"        {conf['annotator']:35s} → {conf['critic']:35s}  ×{conf['count']}")
    if tax["missing_entity_types"]:
        print(f"\n      Missing entity types (Critic found but Annotator missed):")
        for t, c in tax["missing_entity_types"].items():
            print(f"        {t}: {c}")

    tools = report["layer3_tool_usage"]
    print(f"\n  3b. Tool usage")
    print(f"      Messages with tool calls: {tools['total_messages_with_tools']}")
    for agent, counts in tools["per_agent_tool_usage"].items():
        print(f"      {agent}: {counts}")
    if tools["tools_during_disagreements"]:
        print(f"      Tools invoked during disagreements: {tools['tools_during_disagreements']}")
    if tools["empty_tool_results"]:
        print(f"      Tools returning empty results: {tools['empty_tool_results']}")

    conv = report["layer3_convergence"]
    print(f"\n  3c. Convergence dynamics")
    print(f"      Multi-round: {conv['multi_round_sentences']} | "
          f"Single-round: {conv['single_round_sentences']}")
    print(f"      Entities re-typed: {conv['total_entities_retyped']}")
    print(f"      Entities added: {conv['total_entities_added']} | "
          f"removed: {conv['total_entities_removed']}")
    if conv["acceptance_rate"] is not None:
        ar_ci = conv.get("acceptance_rate_ci95") or [float("nan"), float("nan")]
        print(f"      Critic suggestion acceptance rate: "
              f"{conv['suggestions_accepted']}/{conv['total_critic_suggestions']} "
              f"= {conv['acceptance_rate']:.0%}  "
              f"95% Wilson CI {fmt_ci(*ar_ci, decimals=2)}")

    flagged = report["layer3_flagged_items"]
    print(f"\n  3d. Flagged items")
    print(f"      Items flagged for human review: {flagged['total_flagged_items']}")
    print(f"      Sentences with flags: {flagged['sentences_with_flags']}")
    if "iaa_gap" in flagged:
        f_ci = flagged.get("flagged_iaa_ci95") or [float("nan"), float("nan")]
        u_ci = flagged.get("unflagged_iaa_ci95") or [float("nan"), float("nan")]
        print(f"      Flagged sentences human IAA:   {flagged['flagged_mean_human_iaa']:.3f}  "
              f"CI {fmt_ci(*f_ci)}")
        print(f"      Unflagged sentences human IAA: {flagged['unflagged_mean_human_iaa']:.3f}  "
              f"CI {fmt_ci(*u_ci)}")
        print(f"      Gap: {flagged['iaa_gap']:.3f}  "
              f"p = {fmt_p(flagged.get('iaa_gap_p_value'))}  "
              f"({'flagged harder ✓' if flagged['iaa_gap'] > 0 else 'no difference'})")

    # ── Print Layer 4 ─────────────────────────────────────
    gs = report["layer4_guideline_signals"]
    print(f"\n{'='*60}")
    print(f"  LAYER 4 — Guideline Improvement Signal")
    print(f"{'='*60}")

    print(f"\n  Guideline rules cited: {gs['total_rule_citations']} "
          f"({gs['unique_rules_cited']} unique)")
    if gs["guideline_rule_citations"]:
        print(f"  Most-cited rules:")
        for rule, count in list(gs["guideline_rule_citations"].items())[:10]:
            print(f"    {rule}: {count}×")

    print(f"\n  Empty guideline searches (potential gaps): {gs['n_empty_searches']}")
    for es in gs["empty_guideline_searches"][:5]:
        print(f"    Query: \"{es['query']}\" in: {es['sentence']}")

    print(f"\n  Recommendations for guideline amendments:")
    for i, r in enumerate(gs["recommendations"][:5], 1):
        print(f"\n    {i}. {r['pattern']} (×{r['frequency']})")
        print(f"       {r['recommendation']}")
        for ex in r["examples"][:2]:
            print(f"       Example: {ex}")

    if args.output:
        # Remove verbose per-sentence data for cleaner output
        clean = {
            "layer3_disagreement_taxonomy": {
                k: v for k, v in tax.items() if k != "per_sentence_disputes"
            },
            "layer3_tool_usage": tools,
            "layer3_convergence": conv,
            "layer3_flagged_items": {
                k: v for k, v in flagged.items() if k != "flagged_items"
            },
            "layer4_guideline_signals": gs,
        }
        with args.output.open("w") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
