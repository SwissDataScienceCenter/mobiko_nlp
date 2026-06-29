"""
Guideline-adherence evaluation.

Measures whether the agents actually ground their entity-type decisions in the
guideline, rather than pattern-matching a step number. Three signals:

  1. Rule coverage   — fraction of (Annotator) entities that carry a non-empty
                       `guideline_rule` citation (and, separately, `guideline_step`).
  2. Grounding match — of the entities whose cited rule text can be matched to a
                       decision-support row, the fraction whose best-matching row
                       is the SAME label they assigned. A low value means agents
                       cite a rule that actually belongs to a different category.
  3. Tool usage      — (optional, from a deliberation .txt log) how often
                       guideline_search / schema_lookup were actually called.

The cited rule may legitimately come from EITHER the decision-support definitions
OR the narrative guideline. Citations that match no decision-support row above the
threshold are counted as "narrative/unmatched" and excluded from the match rate
(they are likely quoting the narrative guideline, which this script does not index).

Usage:
    python eval_guideline_adherence.py \
        --agent-jsonl <results.jsonl> \
        [--decision-support Decision_support.csv] \
        [--deliberation-log deliberationsN.txt] \
        [--output adherence.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent                         # …/multi_agent_annotation (shared data)
_DEFAULT_DS = _PKG_ROOT / "Decision_support.csv"

MATCH_THRESHOLD = 0.6  # fraction of cited-rule tokens that must appear in a DS row


# ───────────────────────── parsing helpers ─────────────────────────

def _try_parse_json(text: str) -> Optional[dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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


def _tokens(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2}


def load_decision_support(path: Path) -> Dict[str, set]:
    """Return {UPPER_LABEL: token-set of Question+Examples+Definition+Comment}."""
    rows: Dict[str, set] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("LABEL") or "").strip().upper()
            if not label:
                continue
            blob = " ".join(
                row.get(c, "") or "" for c in ("Question", "Examples", "Definition", "Comment")
            )
            rows[label] = _tokens(blob)
    return rows


def annotator_entities(record: dict) -> List[dict]:
    """Entities from the LAST Annotator message that contains an 'entities' list."""
    latest: List[dict] = []
    for m in record.get("messages", []):
        if m.get("agent") != "Annotator":
            continue
        parsed = _try_parse_json(m.get("content", ""))
        if parsed and isinstance(parsed.get("entities"), list):
            latest = parsed["entities"]
    return latest


# ───────────────────────── metrics ─────────────────────────

def best_ds_label(rule_text: str, ds: Dict[str, set]) -> Optional[str]:
    """Label whose decision-support row best contains the cited rule's tokens."""
    rt = _tokens(rule_text)
    if not rt:
        return None
    best_label, best_score = None, 0.0
    for label, toks in ds.items():
        if not toks:
            continue
        score = len(rt & toks) / len(rt)   # fraction of cited tokens found in this row
        if score > best_score:
            best_label, best_score = label, score
    return best_label if best_score >= MATCH_THRESHOLD else None


def evaluate(records: List[dict], ds: Dict[str, set]) -> Dict[str, Any]:
    n_entities = 0
    n_with_rule = 0
    n_with_step = 0
    n_matched = 0          # cited rule matched some DS row
    n_consistent = 0       # best-match DS label == assigned label
    per_type_total: Counter = Counter()
    per_type_consistent: Counter = Counter()
    misgrounded: List[Dict[str, str]] = []

    for rec in records:
        for e in annotator_entities(rec):
            etype = (e.get("entity_type") or "").strip().upper()
            if not etype:
                continue
            n_entities += 1
            rule = (e.get("guideline_rule") or "").strip()
            step = (e.get("guideline_step") or "").strip()
            if rule:
                n_with_rule += 1
            if step:
                n_with_step += 1
            if not rule:
                continue
            match = best_ds_label(rule, ds)
            if match is None:
                continue   # narrative-cited or unmatched — not scored for DS consistency
            n_matched += 1
            per_type_total[etype] += 1
            if match == etype:
                n_consistent += 1
                per_type_consistent[etype] += 1
            else:
                misgrounded.append({
                    "text": e.get("text", ""),
                    "assigned": etype,
                    "cited_rule_matches": match,
                    "rule": rule[:120],
                })

    def frac(a, b):
        return round(a / b, 3) if b else None

    per_type = {
        t: {"matched": per_type_total[t], "consistent": per_type_consistent[t],
            "rate": frac(per_type_consistent[t], per_type_total[t])}
        for t in sorted(per_type_total)
    }

    return {
        "n_entities": n_entities,
        "rule_coverage": frac(n_with_rule, n_entities),
        "step_coverage": frac(n_with_step, n_entities),
        "n_rule_cited": n_with_rule,
        "n_matched_to_decision_support": n_matched,
        "grounding_match_rate": frac(n_consistent, n_matched),
        "per_type_match": per_type,
        "misgrounded_examples": misgrounded[:25],
    }


def count_tool_calls(log_path: Path) -> Dict[str, int]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "guideline_search": text.count("EXECUTING FUNCTION guideline_search"),
        "schema_lookup": text.count("EXECUTING FUNCTION schema_lookup"),
        "lookup_precedent": text.count("EXECUTING FUNCTION lookup_precedent"),
    }


# ───────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-jsonl", type=Path, required=True)
    ap.add_argument("--decision-support", type=Path, default=_DEFAULT_DS)
    ap.add_argument("--deliberation-log", type=Path, default=None,
                    help="Optional console .txt log to count tool invocations")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    records = []
    with args.agent_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))

    ds = load_decision_support(args.decision_support)
    result = evaluate(records, ds)

    if args.deliberation_log and args.deliberation_log.exists():
        result["tool_calls"] = count_tool_calls(args.deliberation_log)

    print(f"\n{'='*60}")
    print(f"  GUIDELINE ADHERENCE  ({len(records)} records, {result['n_entities']} entities)")
    print(f"{'='*60}")
    print(f"  Rule coverage  (entities citing guideline_rule): {result['rule_coverage']}")
    print(f"  Step coverage  (entities citing guideline_step): {result['step_coverage']}")
    print(f"  Cited rules matched to decision-support rows:    {result['n_matched_to_decision_support']}/{result['n_rule_cited']}")
    print(f"  Grounding match rate (cited rule's label == assigned label): {result['grounding_match_rate']}")
    if result["per_type_match"]:
        print(f"\n  Per-type grounding match rate:")
        for t, m in result["per_type_match"].items():
            print(f"    {t:32s} {m['rate']}  ({m['consistent']}/{m['matched']})")
    if result.get("tool_calls"):
        tc = result["tool_calls"]
        print(f"\n  Tool calls (from log): guideline_search={tc['guideline_search']}  "
              f"schema_lookup={tc['schema_lookup']}  lookup_precedent={tc['lookup_precedent']}")
    if result["misgrounded_examples"]:
        print(f"\n  Misgrounded examples (cited a rule that fits another label):")
        for ex in result["misgrounded_examples"][:10]:
            print(f"    '{ex['text']}' labelled {ex['assigned']} but rule matches {ex['cited_rule_matches']}")

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()