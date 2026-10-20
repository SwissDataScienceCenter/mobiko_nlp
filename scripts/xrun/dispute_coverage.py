#!/usr/bin/env python
"""
Every Critic dispute, mapped to the entity spans it implicates.

An earlier split by dispute "kind" was wrong twice over. It classified a dispute
as relational whenever either label mentioned a relation or REMOVE, which put
`target="annual", TEMPORAL PROPERTY -> REMOVE (modifier of rainfall)` in the
relation bucket — a dispute about whether a span is an entity at all. And it
matched a dispute to human spans through the raw `target` field, which for 56%
of relation disputes names a RELATION ("species richness HAS_PROPERTY varying
response") and so could never match an entity span.

Relations are annotated over entity endpoints, so a dispute about a relation is
evidence about its endpoints. Human relation annotation is not finished, so
relations cannot be scored directly — but they can still be READ as evidence
about the entity spans they connect, which is the target we do have.

This resolves every dispute to entity spans:

  plain target        the span itself
  "X PREDICATE Y"     both endpoints, X and Y

and compares three predictors of human entity-type disagreement:

  literal      dispute target matched as-is        (the earlier weak measure)
  endpoints    relation targets split to endpoints (this file)
  entity-only  disputes whose two labels are both schema entity types

    python scripts/xrun/dispute_coverage.py
"""
from __future__ import annotations

import re
import sys
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
from eval_utils import permutation_p_rho                 # noqa: E402
from label_norm import _FALLBACK_ENTITY_TYPES, canonicalize_entity_type  # noqa: E402

SCHEMA = set(_FALLBACK_ENTITY_TYPES)
N_PERM = 2000

# The relation vocabulary, as it appears inside a target string.
PREDICATE = re.compile(
    r"\s+(AFFECTS|HAS_PROPERTY|LOCATED_IN|CAUSES|RELATED_TO|COMPARES_TO|PART_OF|"
    r"ORIGINATES_FROM|DESTINATES_TO|HAS PROPERTY|LOCATED IN|RELATED TO|"
    r"COMPARES TO|PART OF|ORIGINATES FROM|DESTINATES FROM)\s+")


def implicated_spans(target) -> list:
    """Entity spans a dispute target refers to.

    "X PREDICATE Y" yields both endpoints; anything else is a span in its own
    right. A non-string target (one run emitted a relation object) yields none.
    """
    if not isinstance(target, str) or not target.strip():
        return []
    parts = PREDICATE.split(target)
    if len(parts) >= 3:
        return [L2._normalize(parts[0]), L2._normalize(parts[-1])]
    return [L2._normalize(target)]


def is_entity_dispute(d: dict) -> bool:
    a = canonicalize_entity_type(d.get("annotator_label"))
    c = canonicalize_entity_type(d.get("proposed_label"))
    return a in SCHEMA and c in SCHEMA


def collect(rec: dict):
    """(literal, endpoints, entity_only) sets of normalised spans."""
    lit, ends, ent = set(), set(), set()
    for cr in reconstruct_timeline(rec)["critic_rounds"]:
        for d in cr["disagreements"]:
            t = d.get("target")
            if isinstance(t, str):
                lit.add(L2._normalize(t))
            spans = implicated_spans(t)
            ends.update(spans)
            if is_entity_dispute(d):
                ent.update(spans)
    return lit, ends, ent


def main():
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): a for s, a in human.items()}

    print("All Critic disputes as a predictor of human entity-type disagreement.")
    print("rate = share of implicated spans the two annotators typed differently.")
    print("cov = share of human-marked spans the predictor flags at all.\n")
    print(f"  {'run':30s} {'base':>6s} | {'literal':>7s} {'cov':>4s} | "
          f"{'endpts':>7s} {'cov':>4s} {'p':>7s} | {'ent-only':>8s} {'cov':>4s} {'p':>7s}")

    for _model, arm, path in CLEAN:
        if arm != "multi" or not path.exists():
            continue
        buckets = {"literal": ([], []), "endpoints": ([], []), "entity": ([], [])}
        for rec in load_records(path):
            anns = hnorm.get(norm(rec.get("sentence", "")))
            if not anns or len(anns) < 2:
                continue
            lit, ends, ent = collect(rec)
            for key, info in L2.compute_human_span_disagreements(anns).items():
                y = 1.0 if info["type_disagreed"] else 0.0
                for name, flagged in (("literal", key in lit),
                                      ("endpoints", key in ends),
                                      ("entity", key in ent)):
                    buckets[name][0].append(1.0 if flagged else 0.0)
                    buckets[name][1].append(y)

        ys = buckets["literal"][1]
        base = sum(ys) / len(ys) if ys else 0.0
        out = f"  {path.name[:30]:30s} {base:6.3f} |"
        for name in ("literal", "endpoints", "entity"):
            xs, yy = buckets[name]
            hit = [y for x, y in zip(xs, yy) if x]
            rate = (sum(hit) / len(hit)) if hit else float("nan")
            cov = len(hit) / len(yy) if yy else 0.0
            p = (permutation_p_rho(xs, yy, n_boot=N_PERM)
                 if len(set(xs)) > 1 and len(set(yy)) > 1 else 1.0)
            out += (f" {rate:7.3f} {cov:4.0%}" if name == "literal"
                    else f" {rate:7.3f} {cov:4.0%} {p:7.4f}")
            out += " |" if name != "entity" else ""
        print(out)

    print("\n  'endpoints' is the version that takes relations seriously: a dispute")
    print("  about a relation is evidence about the entities it connects. Compare")
    print("  its coverage against 'literal' — the gap is disputes the earlier")
    print("  measurement simply could not see.")


if __name__ == "__main__":
    main()
