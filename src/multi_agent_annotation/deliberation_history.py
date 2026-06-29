"""
Shared deliberation-history extraction — the single source of truth for how
disagreement is counted from the multi-agent logs.

The deliberation shape per sentence is:

    Annotator(R1) -> Critic(R1) -> [Annotator(R2) -> Critic(R2)] -> Adjudicator

The OLD approach read only the final state (or a single Critic message), which
hides almost all disagreement: the loop drives disputes to ~0 by the last round
(the Annotator capitulates) and the stored agreement_score then reports ~1.0.
This module instead walks the FULL history and counts disagreement at every
round, plus the Adjudicator's resolutions, and exposes the number of rounds.

Both Layer 2 (eval_layer2_correlation) and Layer 3/4 (eval_layer34_deliberation)
build on these helpers so the two reports cannot drift apart.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ───────────────────────── parsing helpers ─────────────────────────

def load_records(path: Path) -> List[dict]:
    out = []
    with Path(path).open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                out.append(json.loads(line))
    return out


def try_parse_json(text: str) -> Optional[dict]:
    """Return the last top-level JSON object embedded in `text` (ignores <think>)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    last = None
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "{":
            try:
                obj, end = decoder.raw_decode(cleaned, i)
                if isinstance(obj, dict):
                    last = obj
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return last


def norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def label_of(e: dict) -> str:
    return (e.get("entity_type") or e.get("type") or "").strip().upper()


def overlaps(a: str, b: str) -> bool:
    """Loose span-text match used to follow a target span across rounds."""
    if not a or not b:
        return False
    return a == b or a in b or b in a


# ───────────────────────── timeline reconstruction ─────────────────────────

def reconstruct_timeline(rec: dict) -> Dict[str, Any]:
    """
    Walk one record's messages in order. Returns:
      annotator_rounds: [ {norm_text -> label} ]  one map per Annotator turn
      critic_rounds:    [ {disagreements, agreements, missing} ]  per Critic turn
      adjudication:     {resolutions: [...]}
    """
    annotator_rounds: List[Dict[str, str]] = []
    critic_rounds: List[Dict[str, Any]] = []
    adjudication: Dict[str, Any] = {"resolutions": []}

    for m in rec.get("messages", []):
        agent = m.get("agent", "")
        parsed = try_parse_json(m.get("content", ""))
        if not parsed:
            continue

        if agent == "Annotator" and "entities" in parsed:
            ents = {}
            for e in parsed["entities"]:
                t = norm(e.get("text", ""))
                if t:
                    ents[t] = label_of(e)
            annotator_rounds.append(ents)

        elif agent == "Critic" and "disagreements" in parsed:
            critic_rounds.append({
                "disagreements": parsed.get("disagreements", []),
                "agreements": parsed.get("agreements", []),
                "missing": parsed.get("missing_annotations", []),
            })

        elif agent == "Adjudicator":
            res = parsed.get("disagreement_resolutions")
            if res:
                adjudication["resolutions"] = res

    return {
        "annotator_rounds": annotator_rounds,
        "critic_rounds": critic_rounds,
        "adjudication": adjudication,
    }


# ───────────────────────── per-record signals ─────────────────────────

def record_signals(rec: dict) -> Dict[str, Any]:
    """
    Sentence-level disagreement signals derived from the full history.
    Used by Layer 2 to correlate against human difficulty.
    """
    tl = reconstruct_timeline(rec)
    crit = tl["critic_rounds"]
    r1_dis = len(crit[0]["disagreements"]) if crit else 0
    r1_agr = len(crit[0]["agreements"]) if crit else 0
    r2_dis = len(crit[1]["disagreements"]) if len(crit) > 1 else 0
    all_critic = sum(len(c["disagreements"]) for c in crit)
    adj = len(tl["adjudication"]["resolutions"])
    rounds = rec.get("rounds_used", len(crit))
    return {
        "rounds_used": rounds,
        "r1_disagreements": r1_dis,
        "r1_agreements": r1_agr,
        "r1_rate": r1_dis / (r1_dis + r1_agr) if (r1_dis + r1_agr) else 0.0,
        "r2_disagreements": r2_dis,
        "critic_disagreements_all_rounds": all_critic,
        "adjudicator_resolutions": adj,
        "total_disagreements_all_steps": all_critic + adj,
        "flagged": 1 if rec.get("flagged_for_human_review") else 0,
    }


def agent_disagreed_spans(rec: dict) -> Dict[str, Dict[str, Any]]:
    """
    Per-span agent disagreement, UNION across ALL Critic rounds (history-based).

    A span is disagreed if the Critic listed it in `disagreements` or
    `missing_annotations` in ANY round. Returns
    {norm_text: {agent_disagreed, annotator_type, critic_type, severity}}.
    """
    tl = reconstruct_timeline(rec)

    # all spans the Annotator ever proposed (across rounds) → its label
    annotator_spans: Dict[str, str] = {}
    for ents in tl["annotator_rounds"]:
        annotator_spans.update(ents)

    spans: Dict[str, Dict[str, Any]] = {
        text: {"agent_disagreed": False, "annotator_type": atype,
               "critic_type": None, "severity": None}
        for text, atype in annotator_spans.items()
    }

    for cr in tl["critic_rounds"]:
        for d in cr["disagreements"]:
            target = norm(d.get("target", ""))
            matched = next((s for s in annotator_spans if overlaps(target, s)), target)
            slot = spans.setdefault(matched, {
                "agent_disagreed": True,
                "annotator_type": (d.get("annotator_label") or "").strip().upper(),
                "critic_type": None, "severity": None,
            })
            slot["agent_disagreed"] = True
            slot["critic_type"] = (d.get("proposed_label") or "").strip().upper()
            slot["severity"] = (d.get("severity") or "").lower()
        for miss in cr["missing"]:
            text = norm(miss.get("text", ""))
            if text:
                slot = spans.setdefault(text, {
                    "agent_disagreed": True, "annotator_type": "(missing)",
                    "critic_type": (miss.get("entity_type") or "").strip().upper(),
                    "severity": "missing",
                })
                slot["agent_disagreed"] = True

    return spans


# ───────────────────────── aggregate analysis ─────────────────────────

def analyze(records: List[dict]) -> Dict[str, Any]:
    """
    Aggregate the deliberation history across all records:
      - per-step disagreement counts + rates (round 1, round 2, adjudication)
      - disagreement taxonomy by round + combined
      - convergence trajectory + resolution direction (round-1 disputes followed
        into the revised annotation) — the corrected version of the old, broken
        "critic acceptance" metric
      - guideline grounding coverage (guideline_step vs guideline_rule)
    """
    dis_by_round: Counter = Counter()
    agr_by_round: Counter = Counter()
    miss_by_round: Counter = Counter()
    critic_turns_by_round: Counter = Counter()
    n_adjudications = 0
    adj_resolutions_total = 0

    taxonomy_by_round: Dict[int, Counter] = defaultdict(Counter)
    severity_by_round: Dict[int, Counter] = defaultdict(Counter)
    taxonomy_all: Counter = Counter()
    missing_types: Counter = Counter()

    traj = Counter()
    direction = Counter()

    n_ent = n_step = n_rule = 0

    for rec in records:
        tl = reconstruct_timeline(rec)
        critic_rounds = tl["critic_rounds"]
        annotator_rounds = tl["annotator_rounds"]

        for k, cr in enumerate(critic_rounds, start=1):
            critic_turns_by_round[k] += 1
            agr_by_round[k] += len(cr["agreements"])
            miss_by_round[k] += len(cr["missing"])
            for miss in cr["missing"]:
                mt = (miss.get("entity_type") or "").strip().upper()
                if mt:
                    missing_types[mt] += 1
            for d in cr["disagreements"]:
                dis_by_round[k] += 1
                a = (d.get("annotator_label") or "?").strip().upper()
                b = (d.get("proposed_label") or "?").strip().upper()
                taxonomy_by_round[k][(a, b)] += 1
                taxonomy_all[(a, b)] += 1
                severity_by_round[k][(d.get("severity") or "unspecified").lower()] += 1

        res = tl["adjudication"]["resolutions"]
        if res:
            n_adjudications += 1
            adj_resolutions_total += len(res)

        # convergence trajectory: follow each round-1 dispute into the revision
        if critic_rounds and len(annotator_rounds) >= 2:
            next_ents = annotator_rounds[1]
            for d in critic_rounds[0]["disagreements"]:
                target = norm(d.get("target", ""))
                proposed = (d.get("proposed_label") or "").strip().upper()
                annot = (d.get("annotator_label") or "").strip().upper()
                present, new_label = False, None
                for span, lab in next_ents.items():
                    if overlaps(target, span):
                        present, new_label = True, lab
                        break
                if not present:
                    traj["resolved"] += 1
                    direction["dropped"] += 1
                elif new_label == proposed:
                    traj["resolved"] += 1
                    direction["accepted_critic"] += 1
                elif new_label == annot:
                    traj["persisted"] += 1
                    direction["kept_own"] += 1
                else:
                    traj["resolved"] += 1
                    direction["third_label"] += 1

        for m in rec.get("messages", []):
            if m.get("agent") != "Annotator":
                continue
            parsed = try_parse_json(m.get("content", ""))
            if not parsed:
                continue
            for e in parsed.get("entities", []):
                if not label_of(e):
                    continue
                n_ent += 1
                if (e.get("guideline_step") or "").strip():
                    n_step += 1
                if (e.get("guideline_rule") or "").strip():
                    n_rule += 1

    def top(counter: Counter, n=15):
        return [
            {"annotator": a, "critic": b, "count": c}
            for (a, b), c in sorted(counter.items(), key=lambda x: -x[1])[:n]
        ]

    total_dis = sum(dis_by_round.values())
    followed = sum(traj.values())
    return {
        "n_sentences": len(records),
        "per_step": {
            "critic_turns_by_round": dict(critic_turns_by_round),
            "disagreements_by_round": dict(dis_by_round),
            "agreements_by_round": dict(agr_by_round),
            "missing_by_round": dict(miss_by_round),
            "disagreement_rate_by_round": {
                k: round(dis_by_round[k] / (dis_by_round[k] + agr_by_round[k]), 4)
                for k in sorted(critic_turns_by_round)
                if (dis_by_round[k] + agr_by_round[k]) > 0
            },
            "total_critic_disagreements": total_dis,
            "sentences_adjudicated": n_adjudications,
            "adjudicator_resolutions": adj_resolutions_total,
            "total_disagreements_all_steps": total_dis + adj_resolutions_total,
        },
        "taxonomy_by_round": {
            str(k): {"severity": dict(severity_by_round[k]),
                     "top_confusions": top(taxonomy_by_round[k])}
            for k in sorted(taxonomy_by_round)
        },
        "top_confusions_all_rounds": top(taxonomy_all, 20),
        "missing_entity_types": dict(missing_types.most_common()),
        "convergence_trajectory": {
            "round1_disputes_followed": followed,
            "resolved_by_round2": traj["resolved"],
            "persisted_to_round2": traj["persisted"],
            "resolution_rate": round(traj["resolved"] / followed, 4) if followed else None,
            "resolution_direction": dict(direction),
        },
        "guideline_coverage": {
            "annotator_entities": n_ent,
            "with_guideline_step": n_step,
            "with_guideline_rule": n_rule,
            "step_coverage": round(n_step / n_ent, 4) if n_ent else None,
            "rule_coverage": round(n_rule / n_ent, 4) if n_ent else None,
        },
    }


# ─────────────── disagreement vs human difficulty ───────────────

def correlate_with_human_difficulty(
    records: List[dict],
    human_paths: List[Path],
) -> Optional[Dict[str, Any]]:
    """
    Spearman correlation of each per-record history signal against human
    annotation difficulty (= 1 - pairwise inter-human entity F1). Returns None if
    helper modules are unavailable or fewer than 3 sentences match.
    """
    try:
        from eval_utils import _spearman_rho, permutation_p_rho
        from eval_layer1_output import sentence_f1
        from eval_layer2_correlation import load_all_human_annotations
    except Exception:
        return None

    human = load_all_human_annotations([Path(p) for p in human_paths])
    hnorm = {norm(s): a for s, a in human.items()}

    # sentence_f1 (formerly compute_entity_metrics) expects {text, type}; the
    # eval_layer2 human loader emits {text, entity_type}. Map to text+type
    # (the renamed function's default "exact" = surface form + type).
    def _typed(ents):
        return [{"text": e.get("text", ""),
                 "type": e.get("entity_type", e.get("type", "")).strip().upper()}
                for e in ents]

    rows = []
    for r in records:
        anns = hnorm.get(norm(r.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        ids = list(anns.keys())
        f1 = sentence_f1(_typed(anns[ids[0]]["entities"]),
                         _typed(anns[ids[1]]["entities"]),
                         mode="text_type") or 0.0
        sig = record_signals(r)
        sig["difficulty"] = 1.0 - f1
        rows.append(sig)

    if len(rows) < 3:
        return None

    y = [r["difficulty"] for r in rows]
    signals = [
        ("rounds_used", "rounds_used"),
        ("R1 disagreement count", "r1_disagreements"),
        ("R1 disagreement rate", "r1_rate"),
        ("total disagreements (all steps)", "total_disagreements_all_steps"),
        ("flagged_for_review (0/1)", "flagged"),
    ]
    out = []
    for lbl, key in signals:
        x = [r[key] for r in rows]
        out.append({
            "signal": lbl,
            "spearman_rho": _spearman_rho(x, y),
            "p_value": permutation_p_rho(x, y, n_boot=2000),
        })
    return {
        "n_sentences": len(rows),
        "mean_human_f1": round(sum(1 - r["difficulty"] for r in rows) / len(rows), 4),
        "correlations": out,
    }