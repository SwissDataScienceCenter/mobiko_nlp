#!/usr/bin/env python
"""
Score a blind-pair run: the three measurements the design was built to produce.

1. AGENT-AGENT AGREEMENT against the two-human ceiling. Three agents annotate
   independently under the humans' condition (sentence plus its immediate
   neighbours, which is what Mark and Davnah had), so their agreement is
   comparable to the 0.497 the two experts reach. If the agents agree MORE than
   the humans that is shared bias, not accuracy — same model, same prompt — and
   the comparison is the point either way.

2. MAJORITY VOTE vs ADJUDICATOR, both against human gold. The vote is free and
   deterministic and settles 96.6% of type disagreements on the archived draws,
   so an LLM adjudicator that cannot beat it is cost and opacity with no
   decision content. Scored on the SAME contested set, because that is where
   they differ.

3. THE ANCHORING DELTA. How much closer the Critic's annotation moves to its
   peers after it sees them. This is the quantity the review pipeline could never
   produce: there the Critic never commits before exposure, so its agreement with
   the Annotator is anchoring and judgement inseparably mixed. Here the blind
   pass is kept, so the movement is measurable.

Scoring uses OFFSETS, not substring containment. Containment was understating
the agent false-positive rate by roughly half and most for short spans; offsets
are resolved by the harness for new runs and are in the record.

    python scripts/xrun/eval_blind_pair.py data/auto_annotated/grid_*_blind_r1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import xrun_analysis as X                                    # noqa: E402
from label_norm import normalize_text as norm                # noqa: E402
from span_offsets import human_offsets, iou                  # noqa: E402
from eval_difficulty_split import load_human_spans           # noqa: E402

ROLES = ("annotator_a", "annotator_b", "critic_blind", "critic_revised")
HUMAN_CEILING = 0.497


def spans_of(entities):
    return [(e["start"], e["end"], e.get("entity_type"))
            for e in entities if e.get("start") is not None]


def f1(pred, gold, typed=True, thr=0.5):
    """Span F1 by offset overlap at `thr` IoU, optionally requiring the type."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    used, tp = set(), 0
    for ps, pe, pt in pred:
        for i, (gs, ge, gt) in enumerate(gold):
            if i in used:
                continue
            if iou((ps, pe), (gs, ge)) >= thr and (not typed or pt == gt):
                used.add(i); tp += 1
                break
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def pairwise(a, b, typed=True):
    """Symmetric mean F1, the same estimand as the two-human ceiling — not
    best-of-two, which would have flattered the agents by +0.066."""
    return (f1(a, b, typed) + f1(b, a, typed)) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--typed", action="store_true", default=True)
    ap.add_argument("--untyped", dest="typed", action="store_false",
                    help="score span extent only, ignoring the entity type")
    args = ap.parse_args()

    mark, dav = load_human_spans(Path(X.MARK)), load_human_spans(Path(X.DAV))

    for path in args.runs:
        if not path.exists():
            print(f"{path}: missing"); continue
        recs = [json.loads(l) for l in path.open() if l.strip().startswith("{")]
        blind = [r for r in recs if r.get("blind_pair")]
        if not blind:
            print(f"{path.name}: no blind_pair records (is this a review-mode run?)")
            continue

        print("=" * 94)
        print(f"{path.name}   {len(blind)} blind records"
              f"   model: {(blind[0].get('run_meta') or {}).get('annotator_model')}")
        print("=" * 94)

        # ── 1. agent-agent agreement, and the human pair on the same sentences ──
        ab, with_human, human_pair = [], 0, []
        for r in blind:
            a = r["blind_pair"]["annotations"]
            if all(a[k]["parse_ok"] for k in ("annotator_a", "annotator_b")):
                ab.append(pairwise(spans_of(a["annotator_a"]["entities"]),
                                   spans_of(a["annotator_b"]["entities"]),
                                   args.typed))
            k = norm(r["sentence"])
            if k in mark and k in dav and mark[k] and dav[k]:
                with_human += 1
                human_pair.append(pairwise(
                    [(s, e, t) for (s, e), t in
                     zip(human_offsets(mark[k]), [x["type"] for x in mark[k]])],
                    [(s, e, t) for (s, e), t in
                     zip(human_offsets(dav[k]), [x["type"] for x in dav[k]])],
                    args.typed))

        print("\n1. AGREEMENT BETWEEN TWO INDEPENDENT ANNOTATORS")
        print(f"   agent A vs agent B      {mean(ab):.3f}   (n={len(ab)})"
              if ab else "   agent pair: no parseable pairs")
        if human_pair:
            print(f"   Mark vs Davnah         {mean(human_pair):.3f}   "
                  f"(n={with_human}, same sentences)")
            print(f"   corpus-wide human ceiling {HUMAN_CEILING:.3f} "
                  f"(all 154 double-annotated sentences)")
            if ab:
                d = mean(ab) - mean(human_pair)
                print(f"   difference             {d:+.3f}  "
                      f"{'agents agree MORE' if d > 0 else 'agents agree LESS'} "
                      f"than the humans")
                if d > 0:
                    print("   NOTE: more agreement is not more accuracy — same "
                          "model, same prompt, so this is shared bias.")

        # ── 2. majority vs adjudicator, against human gold ──
        rows = []
        for r in blind:
            k = norm(r["sentence"])
            if k not in mark or k not in dav or not mark[k] or not dav[k]:
                continue
            gold_m = [(s, e, t) for (s, e), t in
                      zip(human_offsets(mark[k]), [x["type"] for x in mark[k]])]
            gold_d = [(s, e, t) for (s, e), t in
                      zip(human_offsets(dav[k]), [x["type"] for x in dav[k]])]
            maj = spans_of(r["blind_pair"]["majority"]["entities"])
            fin = spans_of([{"start": e.get("start"), "end": e.get("end"),
                             "entity_type": e.get("entity_type")}
                            for e in r.get("final_entities") or []])
            score = lambda p: (f1(p, gold_m, args.typed) + f1(p, gold_d, args.typed)) / 2
            rows.append((score(maj), score(fin),
                         r["blind_pair"]["adjudicator_parse_ok"]))

        print("\n2. MAJORITY VOTE vs ADJUDICATOR, scored against both humans")
        if not rows:
            print("   no sentences with both human annotations")
        else:
            ok = [x for x in rows if x[2]]
            print(f"   sentences scored        {len(rows)}"
                  f"  (adjudicator parsed on {len(ok)})")
            print(f"   majority vote           {mean(r[0] for r in rows):.3f}")
            print(f"   adjudicator             {mean(r[1] for r in rows):.3f}")
            if ok:
                d = mean(r[1] - r[0] for r in ok)
                print(f"   adjudicator - majority  {d:+.3f}   "
                      f"(parsed sentences only, n={len(ok)})")
                wins = sum(1 for r in ok if r[1] > r[0])
                ties = sum(1 for r in ok if r[1] == r[0])
                print(f"   adjudicator better on {wins}, tied {ties}, "
                      f"worse on {len(ok)-wins-ties}")
                print("   A free deterministic rule is the baseline; an "
                      "adjudicator that does not beat it is not deciding anything.")

        # ── 3. the anchoring delta ──
        print("\n3. ANCHORING — does the Critic move toward its peers after seeing them?")
        before, after, moved = [], [], 0
        for r in blind:
            a = r["blind_pair"]["annotations"]
            if not all(a[k]["parse_ok"] for k in ROLES):
                continue
            peers = [spans_of(a["annotator_a"]["entities"]),
                     spans_of(a["annotator_b"]["entities"])]
            cb, cr = spans_of(a["critic_blind"]["entities"]), \
                spans_of(a["critic_revised"]["entities"])
            before.append(mean(pairwise(cb, p, args.typed) for p in peers))
            after.append(mean(pairwise(cr, p, args.typed) for p in peers))
            if set(cb) != set(cr):
                moved += 1
        if before:
            print(f"   critic_blind   vs peers  {mean(before):.3f}")
            print(f"   critic_revised vs peers  {mean(after):.3f}")
            print(f"   movement toward peers    {mean(after) - mean(before):+.3f}"
                  f"   (changed its annotation on {moved}/{len(before)} sentences)")
            print("   This is the quantity the review pipeline cannot produce: "
                  "there the\n   Critic never commits before exposure, so "
                  "anchoring and judgement are\n   inseparable.")
        else:
            print("   no sentence had all four roles parse")


if __name__ == "__main__":
    raise SystemExit(main())
