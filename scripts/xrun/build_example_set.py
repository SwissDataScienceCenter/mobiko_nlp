#!/usr/bin/env python
"""
Select a leakage-free, balanced in-context example set — and measure whether it
is still current.

THE LEAKAGE CONSTRAINT. Of 326 annotated sentences, 154 carry both annotators
and those 154 ARE the evaluation set: the 0.497 ceiling, every agent score and
every RQ2 correlation come from them. Examples may therefore only be drawn from
the 172 singly-annotated sentences, which cannot support IAA and are useless for
evaluation. One of those 172 also appears in the grid input and is excluded.

WHY BALANCE BY ANNOTATOR. The ceiling is a symmetric mean over Mark and Davnah.
Examples skewed toward one of them would teach that annotator's conventions, so
agent-vs-that-annotator would be flattered and the parity comparison biased.
Mark contributes 121 pool sentences and Davnah 51, but 537 and 536 SPANS — so
balancing on spans rather than sentences is both possible and the right unit,
since spans are what the examples teach.

WHY BALANCE BY TYPE. A sample drawn at random reflects the corpus's own skew and
may contain no instance at all of the rarer types, which are exactly the ones a
model gets wrong. Selection is greedy on type coverage, subject to the annotator
balance.

AND THE REASON TO CHECK DRIFT. These sentences are older than the evaluation
set, and the annotators' practice has moved. Examples encoding a superseded
convention are worse than no examples: they actively teach the wrong label. So
this also reports what changed between the two pools, so the decision to use the
set is evidence-based rather than a guess.

    python scripts/xrun/build_example_set.py            # select + drift report
    python scripts/xrun/build_example_set.py --n 50 --out examples.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import xrun_analysis as X                                  # noqa: E402
from label_norm import normalize_text as norm              # noqa: E402
from label_norm import canonicalize_entity_type as canon   # noqa: E402
# SCHEMA_TYPES is populated at RUNTIME by _init_tool_state and is empty on
# import — using it here reported "17 of 0 types" and flagged every label as
# off-schema. _FALLBACK_ENTITY_TYPES is the static list.
from label_norm import _FALLBACK_ENTITY_TYPES as SCHEMA_TYPES   # noqa: E402

GRID = ROOT / "data" / "manually_labeled_last" / "overlap_sentences.txt"


def load(path: Path):
    """norm(sentence) -> (raw sentence, [span dicts])."""
    doc = json.loads(Path(path).read_text(encoding="utf8"))
    out = {}
    for s in doc.get("sentences", []):
        spans = [{"text": sp.get("text", ""),
                  "type_raw": sp.get("type"),
                  "type": canon(sp.get("type")),
                  "start_char": sp.get("start_char"),
                  "end_char": sp.get("end_char")}
                 for sp in s.get("spans", [])
                 if sp.get("start_char") is not None
                 and not str(sp.get("type", "")).startswith("RELATION")]
        out[norm(s["text"])] = (s["text"], spans)
    return out


def pools():
    mark, dav = load(Path(X.MARK)), load(Path(X.DAV))
    M = {k for k, (_, sp) in mark.items() if sp}
    D = {k for k, (_, sp) in dav.items() if sp}
    grid = {norm(l.strip()) for l in GRID.read_text().splitlines() if l.strip()} \
        if GRID.exists() else set()
    both = M & D
    cand = []
    for k in sorted(M - D):
        if k in grid:
            continue
        cand.append(("Mark", k) + mark[k])
    for k in sorted(D - M):
        if k in grid:
            continue
        cand.append(("Davnah", k) + dav[k])
    evalset = {k: (mark[k], dav[k]) for k in both}
    return cand, evalset, len(grid & (M - D)) + len(grid & (D - M))


def select(cand, n, seed=42):
    """Greedy: maximise type coverage while holding the annotator span split even."""
    rng = random.Random(seed)
    cand = cand[:]
    rng.shuffle(cand)
    chosen, have = [], Counter()
    spans_by_ann = Counter()

    def gain(item):
        _ann, _k, _raw, spans = item
        types = {s["type"] for s in spans if s["type"]}
        # a type not yet covered is worth far more than one already seen
        return sum(3.0 if have[t] == 0 else 1.0 / (1 + have[t]) for t in types)

    while len(chosen) < n and cand:
        # whichever annotator is behind on spans gets priority this round
        behind = min(("Mark", "Davnah"), key=lambda a: spans_by_ann[a])
        pref = [c for c in cand if c[0] == behind] or cand
        best = max(pref, key=gain)
        cand.remove(best)
        chosen.append(best)
        spans_by_ann[best[0]] += len(best[3])
        for s in best[3]:
            if s["type"]:
                have[s["type"]] += 1
    return chosen, have, spans_by_ann


def drifted_texts(cand, evalset):
    """Span texts the two pools label with no type in common."""
    pool, ev = defaultdict(set), defaultdict(set)
    for *_x, spans in cand:
        for s in spans:
            if s["text"] and s["type"]:
                pool[norm(s["text"])].add(s["type"])
    for (mk, dv) in evalset.values():
        for spans in (mk[1], dv[1]):
            for s in spans:
                if s["text"] and s["type"]:
                    ev[norm(s["text"])].add(s["type"])
    return {t for t in pool if t in ev and not (pool[t] & ev[t])}


def drift_report(cand, evalset):
    """What changed between the example pool and the evaluation set."""
    def tally(span_lists):
        c, off, n = Counter(), 0, 0
        for spans in span_lists:
            for s in spans:
                t = s["type"]
                if not t:
                    continue
                n += 1
                c[t] += 1
                if t not in SCHEMA_TYPES:
                    off += 1
        return c, off, n

    pool_types, pool_off, pool_n = tally([sp for *_x, sp in cand])
    ev_types, ev_off, ev_n = tally(
        [sp for (mk, dv) in evalset.values() for sp in (mk[1], dv[1])])

    print("=" * 96)
    print("DRIFT — example pool (older, single-annotated) vs evaluation set")
    print("=" * 96)
    print(f"  spans: pool {pool_n:,}   eval {ev_n:,}")
    print(f"  OFF-SCHEMA labels: pool {pool_off} ({pool_off/max(pool_n,1):.1%})   "
          f"eval {ev_off} ({ev_off/max(ev_n,1):.1%})")
    print()
    print(f"  {'entity type':28s} {'pool %':>8s} {'eval %':>8s} {'shift':>8s}")
    keys = sorted(set(pool_types) | set(ev_types),
                  key=lambda t: -(ev_types[t] / max(ev_n, 1)))
    for t in keys:
        p = pool_types[t] / max(pool_n, 1)
        e = ev_types[t] / max(ev_n, 1)
        flag = " *" if abs(p - e) > 0.03 else ""
        star = " OFF-SCHEMA" if t not in SCHEMA_TYPES else ""
        print(f"  {t[:28]:28s} {p:7.1%} {e:7.1%} {p-e:+7.1%}{flag}{star}")
    print("\n  * = the type's share moved by more than 3 points between pools.")

    # the sharpest drift evidence: same span text, different type
    pool_lab = defaultdict(set)
    for *_x, spans in cand:
        for s in spans:
            if s["text"] and s["type"]:
                pool_lab[norm(s["text"])].add(s["type"])
    ev_lab = defaultdict(set)
    for (mk, dv) in evalset.values():
        for spans in (mk[1], dv[1]):
            for s in spans:
                if s["text"] and s["type"]:
                    ev_lab[norm(s["text"])].add(s["type"])
    shared = [t for t in pool_lab if t in ev_lab]
    changed = [t for t in shared if not (pool_lab[t] & ev_lab[t])]
    print(f"\n  span texts appearing in BOTH pools: {len(shared)}")
    print(f"  labelled with NO type in common:    {len(changed)} "
          f"({len(changed)/max(len(shared),1):.0%})")
    for t in sorted(changed)[:12]:
        print(f"      {t[:34]:34s} pool={sorted(pool_lab[t])} eval={sorted(ev_lab[t])}")
    return len(changed), len(shared)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--exclude-drifted", action="store_true",
                    help="drop pool sentences carrying a label the annotators "
                         "have since moved away from (see the drift report)")
    args = ap.parse_args()

    cand, evalset, excluded = pools()

    if args.exclude_drifted:
        # CONCEPT fell from 27.4% of pool spans to 5.1% of evaluation spans, and
        # the current guideline now says explicitly that CONCEPT "is NEVER the
        # right label for an entity whose domain is merely uncertain" — so the
        # pool predates a rule the prompt now instructs against. Plus any span
        # text the two pools label with no type in common.
        drifted = drifted_texts(cand, evalset)
        before = len(cand)
        cand = [c for c in cand
                if not any(s["type"] == "CONCEPT" for s in c[3])
                and not any(norm(s["text"]) in drifted for s in c[3])]
        print(f"--exclude-drifted: {before} -> {len(cand)} candidate sentences "
              f"({before - len(cand)} dropped for carrying CONCEPT or one of "
              f"{len(drifted)} re-labelled span texts)\n")
    print(f"candidate pool: {len(cand)} singly-annotated sentences "
          f"({excluded} excluded for appearing in the grid input)")
    print(f"evaluation set: {len(evalset)} sentences — NEVER drawn from\n")

    chosen, have, by_ann = select(cand, args.n, args.seed)
    print("=" * 96)
    print(f"SELECTED {len(chosen)} SENTENCES")
    print("=" * 96)
    print(f"  Mark   {sum(1 for c in chosen if c[0]=='Mark'):3d} sentences, "
          f"{by_ann['Mark']:4d} spans")
    print(f"  Davnah {sum(1 for c in chosen if c[0]=='Davnah'):3d} sentences, "
          f"{by_ann['Davnah']:4d} spans")
    print(f"  span balance Mark:Davnah = "
          f"{by_ann['Mark']/max(by_ann['Davnah'],1):.2f} : 1")
    print(f"\n  entity types covered: {len(have)} of {len(SCHEMA_TYPES)} in schema")
    missing = sorted(set(SCHEMA_TYPES) - set(have))
    if missing:
        print(f"  NOT represented: {missing}")
    print(f"\n  {'type':30s} {'count':>6s}")
    for t, c in sorted(have.items(), key=lambda kv: -kv[1]):
        print(f"  {t[:30]:30s} {c:6d}")

    drift_report(cand, evalset)

    if args.out:
        with args.out.open("w", encoding="utf8") as f:
            for ann, _k, raw, spans in chosen:
                f.write(json.dumps({
                    "sentence": raw,
                    "annotator": ann,
                    "entities": [{"text": s["text"], "entity_type": s["type"],
                                  "start": s["start_char"], "end": s["end_char"]}
                                 for s in spans if s["type"]],
                }, ensure_ascii=False) + "\n")
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
