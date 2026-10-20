#!/usr/bin/env python
"""
What is the deliberation actually ARGUING ABOUT?

The Critic's top confusions are dominated by relation disputes — AFFECTS ->
REMOVE RELATION, HAS_PROPERTY -> REMOVE RELATION, CAUSES -> RELATED_TO — while
every target in the paper is an ENTITY-level quantity: entity F1 against the
human ceiling, and human disagreement about a span's entity TYPE.

That raises the possibility that the deliberation signals looked null not
because deliberation carries no information, but because we scored disputes
about relations against a target about entities. This splits the disputes and
re-tests only the entity-typed ones.

    python scripts/xrun/deliberation_taxonomy.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import eval_layer2_correlation as L2                     # noqa: E402
import xrun_analysis as X                                # noqa: E402
from clean_grid import CLEAN                             # noqa: E402
from deliberation_history import (                       # noqa: E402
    load_records, norm, reconstruct_timeline,
)
from eval_utils import _spearman_rho, permutation_p_rho  # noqa: E402
from label_norm import _FALLBACK_ENTITY_TYPES, canonicalize_entity_type  # noqa: E402

SCHEMA = set(_FALLBACK_ENTITY_TYPES)
N_PERM = 2000


def dispute_kind(annotator_label: str, critic_label: str) -> str:
    """entity | relation | other, from the two labels in one Critic dispute.

    'entity' requires BOTH sides to be schema entity types: a dispute where the
    Critic proposes REMOVE RELATION is not a disagreement about which entity
    type a span has, whatever the annotator originally said.
    """
    a = canonicalize_entity_type(annotator_label)
    c = canonicalize_entity_type(critic_label)
    if a in SCHEMA and c in SCHEMA:
        return "entity"
    blob = f"{a} {c}"
    if "RELATION" in blob or any(
            k in blob for k in ("AFFECTS", "HAS PROPERTY", "LOCATED IN", "CAUSES",
                                "RELATED TO", "ORIGINATES FROM", "DESTINATES TO",
                                "COMPARES TO", "PART OF", "REMOVE")):
        return "relation"
    return "other"


def split_disputes(rec: dict):
    """{kind: set(normalised target span)} plus a Counter of kinds."""
    kinds: Counter = Counter()
    spans = {"entity": set(), "relation": set(), "other": set()}
    for cr in reconstruct_timeline(rec)["critic_rounds"]:
        for d in cr["disagreements"]:
            k = dispute_kind(d.get("annotator_label"), d.get("proposed_label"))
            kinds[k] += 1
            spans[k].add(L2._normalize(d.get("target", "")))
    return spans, kinds


def main():
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): a for s, a in human.items()}

    print("What the Critic argues about, and whether the ENTITY disputes predict")
    print("human entity-type disagreement (the paper's actual target).\n")
    print(f"  {'run':30s} {'disputes':>8s} {'entity':>7s} {'relation':>9s} {'other':>6s}"
          f"  {'base':>6s} {'ent-disp':>9s} {'not':>6s} {'lift':>7s} {'p':>7s}")

    for _model, arm, path in CLEAN:
        if arm != "multi" or not path.exists():
            continue
        tot: Counter = Counter()
        xs, ys = [], []          # entity-dispute flag vs human type-disagreement
        for rec in load_records(path):
            anns = hnorm.get(norm(rec.get("sentence", "")))
            if not anns or len(anns) < 2:
                continue
            spans, kinds = split_disputes(rec)
            tot.update(kinds)
            dis = L2.compute_human_span_disagreements(anns)
            for key, info in dis.items():
                # score every span a human marked, flagged by whether the Critic
                # raised an ENTITY-type dispute about it
                xs.append(1.0 if key in spans["entity"] else 0.0)
                ys.append(1.0 if info["type_disagreed"] else 0.0)
        n = sum(tot.values()) or 1
        base = sum(ys) / len(ys) if ys else 0.0
        d = [y for x, y in zip(xs, ys) if x] or [0.0]
        nd = [y for x, y in zip(xs, ys) if not x] or [0.0]
        p = permutation_p_rho(xs, ys, n_boot=N_PERM) if len(set(xs)) > 1 else 1.0
        print(f"  {path.name[:30]:30s} {n:8d} {tot['entity'] / n:6.0%} "
              f"{tot['relation'] / n:8.0%} {tot['other'] / n:5.0%}  {base:6.3f} "
              f"{sum(d) / len(d):9.3f} {sum(nd) / len(nd):6.3f} "
              f"{sum(d) / len(d) - sum(nd) / len(nd):+7.3f} {p:7.4f}")

    print("\n  entity / relation / other = share of Critic disputes by what the two")
    print("  labels are. 'lift' = how much more often humans disagreed about a span")
    print("  the Critic raised an ENTITY dispute on, over one it did not.")


if __name__ == "__main__":
    main()
