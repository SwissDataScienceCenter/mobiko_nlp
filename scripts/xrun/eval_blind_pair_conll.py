#!/usr/bin/env python
"""
Score a CoNLL blind-pair run. The MoBiKo sibling (eval_blind_pair.py) cannot do
this: it reads the two expert annotators by name from xrun_analysis.MARK/DAV,
and CoNLL has five crowd workers per sentence plus a separate ground truth.

The three measurements are the same, and the difference from MoBiKo matters for
reading them:

1. AGENT-AGENT AGREEMENT against the human pair. On MoBiKo the comparison is to
   two experts; here it is the mean over every available pair of the crowd
   workers who saw that sentence, which is a LOWER ceiling (crowd, single pass,
   no adjudication). Agents agreeing more than this crowd is close to expected
   and is not evidence of accuracy.

2. AGAINST GOLD. CoNLL has a real ground truth, which MoBiKo does not, so
   everything is scored against it directly rather than against two experts
   treated as two golds: the crowd workers, the two agents, the majority vote
   and the Adjudicator. The crowd-vs-gold row is the number that makes the
   agent rows interpretable — and the number most exposed to contamination,
   since these are 1996 Reuters sentences full of names a model has memorised.

3. THE ANCHORING DELTA, identical to MoBiKo.

Section 0 is the plumbing: parse rates, unresolved offsets, token source and
the invariant that CoNLL emits no relations. A run that fails there has no
scientific content and the sections below it are noise.

    python scripts/xrun/eval_blind_pair_conll.py run_a.jsonl run_b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("", "scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    _s = str(ROOT / _p) if _p else str(ROOT)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from eval_blind_pair import ROLES, f1, pairwise, spans_of   # noqa: E402
from label_norm import normalize_text as norm               # noqa: E402

GOLD = ROOT / "data/conll_mturk/all_gold.jsonl"
HUMANS = ROOT / "data/conll_mturk/all_humans.jsonl"


def _spans(entities):
    """(start, end, type) from the corpus's start_char/end_char records."""
    out = []
    for e in entities:
        s, t = e.get("start_char"), e.get("end_char")
        if s is None or t is None:
            continue
        out.append((s, t, e.get("entity_type")))
    return out


def load_corpus():
    """gold[key] -> spans;  humans[key] -> {annotator: spans}.

    Keyed on the normalised sentence, the same key the run records are looked up
    by, so a stray double space in the input file cannot silently drop a
    sentence from the join and shrink n without saying so.
    """
    gold, humans = {}, {}
    for line in GOLD.open():
        if not line.strip():
            continue
        r = json.loads(line)
        gold[norm(r["sentence"])] = _spans(r["entities"])
    for line in HUMANS.open():
        if not line.strip():
            continue
        r = json.loads(line)
        humans.setdefault(norm(r["sentence"]), {})[r["annotator"]] = \
            _spans(r["entities"])
    return gold, humans


def plumbing(blind, recs):
    """Section 0. Returns nothing; prints the checks that gate the rest."""
    parsed = Counter()
    for r in blind:
        for role in ROLES:
            parsed[role] += bool(r["blind_pair"]["annotations"][role]["parse_ok"])
    adj_ok = sum(1 for r in blind if r["blind_pair"]["adjudicator_parse_ok"])
    unresolved = sum(r["blind_pair"].get("unresolved_offsets") or 0 for r in blind)
    # token_usage is per-role with a "total" rollup; token_source there is a
    # LIST of the sources the roles reported, so that a run which fell back to
    # ag2 bookkeeping on one call cannot be read as fully api-metered.
    sources = Counter()
    for r in recs:
        tot = ((r.get("token_usage") or {}).get("total") or {})
        src = tot.get("token_source")
        for one in (src if isinstance(src, list) else [src]):
            sources[one] += 1
    rel_empty = all(not (r.get("final_relations") or []) for r in recs)
    toks = [((r.get("token_usage") or {}).get("total") or {}).get("total_tokens")
            for r in recs]
    toks = [t for t in toks if t]
    secs = [(r.get("timing") or {}).get("total_s") for r in recs]
    secs = [s for s in secs if s]

    print("\n0. PLUMBING")
    print(f"   records                  {len(blind)}")
    print("   parse ok                 " + "  ".join(
        f"{role.replace('annotator_', 'ann_')} {parsed[role]}/{len(blind)}"
        for role in ROLES))
    print(f"   adjudicator parsed       {adj_ok}/{len(blind)}")
    print(f"   unresolved offsets       {unresolved}"
          f"{'' if unresolved == 0 else '   <-- spans not located in the sentence'}")
    print(f"   token_source             {dict(sources)}")
    print(f"   final_relations empty    {rel_empty}"
          f"{'' if rel_empty else '   <-- CoNLL annotates no relations'}")
    if toks:
        print(f"   mean tokens/sentence     {mean(toks):,.0f}")
    if secs:
        print(f"   mean seconds/sentence    {mean(secs):.0f}")


def mentions(blind):
    """Section 1. The vote's own view of where the three agents differed."""
    agg = Counter()
    for r in blind:
        for k, v in (r["blind_pair"]["majority"]["summary"] or {}).items():
            agg[k] += v
    n = agg["mentions"]
    print("\n1. MENTIONS (the vote's clustering of all three blind passes)")
    if not n:
        print("   no mentions found by any agent")
        return
    print(f"   mentions total           {n}")
    for key, label in (("unanimous", "unanimous (all three)"),
                       ("detection_disagreement", "detection disagreement"),
                       ("extent_disagreement", "extent disagreement"),
                       ("type_disagreement", "type disagreement"),
                       ("type_tied", "type tied")):
        print(f"   {label:<24} {agg[key]:>4}   ({agg[key] / n:5.1%})")
    contested = n - agg["unanimous"]
    print(f"   contested                {contested:>4}   ({contested / n:5.1%})"
          "   <- all the vote and the Adjudicator have to work with")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--typed", action="store_true", default=True)
    ap.add_argument("--untyped", dest="typed", action="store_false",
                    help="score span extent only, ignoring the entity type")
    args = ap.parse_args()
    typed = args.typed

    gold, humans = load_corpus()

    for path in args.runs:
        if not path.exists():
            print(f"{path}: missing")
            continue
        recs = [json.loads(l) for l in path.open() if l.strip().startswith("{")]
        blind = [r for r in recs if r.get("blind_pair")]
        if not blind:
            print(f"{path.name}: no blind_pair records "
                  "(is this a review-mode run?)")
            continue
        meta = blind[0].get("run_meta") or {}

        print("=" * 94)
        print(f"{path.name}   {len(blind)} blind records")
        print(f"   model {meta.get('annotator_model')}"
              f"   corpus {meta.get('prompt_set')}"
              f"   nested_spans {meta.get('nested_spans')}"
              f"   context_radius {meta.get('context_radius')}"
              f"   commit {meta.get('git_commit')}")
        print("=" * 94)

        plumbing(blind, recs)
        mentions(blind)

        # ── 2. agreement: the agent pair against the crowd pair ──
        ab, hp, matched = [], [], 0
        for r in blind:
            a = r["blind_pair"]["annotations"]
            if all(a[k]["parse_ok"] for k in ("annotator_a", "annotator_b")):
                ab.append(pairwise(spans_of(a["annotator_a"]["entities"]),
                                   spans_of(a["annotator_b"]["entities"]), typed))
            k = norm(r["sentence"])
            who = humans.get(k) or {}
            if len(who) >= 2:
                matched += 1
                hp.append(mean(pairwise(x, y, typed)
                               for x, y in combinations(who.values(), 2)))

        print("\n2. AGREEMENT BETWEEN TWO INDEPENDENT ANNOTATORS")
        if ab:
            print(f"   agent A vs agent B       {mean(ab):.3f}   (n={len(ab)})")
        else:
            print("   agent pair: no parseable pairs")
        if hp:
            print(f"   crowd pairwise           {mean(hp):.3f}   "
                  f"(n={matched} sentences, all worker pairs)")
            if ab:
                d = mean(ab) - mean(hp)
                print(f"   difference               {d:+.3f}   "
                      f"agents agree {'MORE' if d > 0 else 'LESS'} than the crowd")
                print("   NOTE: agreement is not accuracy. Same model, same "
                      "prompt — see section 3\n   for whether it is also right.")

        # ── 3. everything against the CoNLL ground truth ──
        crowd, agents, maj_s, adj_s, pairs = [], [], [], [], []
        for r in blind:
            k = norm(r["sentence"])
            if k not in gold:
                continue
            g = gold[k]
            a = r["blind_pair"]["annotations"]
            for sp in (humans.get(k) or {}).values():
                crowd.append(f1(sp, g, typed))
            for role in ("annotator_a", "annotator_b"):
                if a[role]["parse_ok"]:
                    agents.append(f1(spans_of(a[role]["entities"]), g, typed))
            m = f1(spans_of(r["blind_pair"]["majority"]["entities"]), g, typed)
            fin = f1(spans_of([{"start": e.get("start"), "end": e.get("end"),
                                "entity_type": e.get("entity_type")}
                               for e in r.get("final_entities") or []]), g, typed)
            maj_s.append(m)
            adj_s.append(fin)
            if r["blind_pair"]["adjudicator_parse_ok"]:
                pairs.append((m, fin))

        print("\n3. AGAINST CoNLL GROUND TRUTH")
        if not maj_s:
            print("   no run sentence found in the gold file")
        else:
            print(f"   sentences scored         {len(maj_s)}")
            if crowd:
                print(f"   crowd annotators         {mean(crowd):.3f}   "
                      f"(n={len(crowd)} worker-sentences)")
            if agents:
                print(f"   agents A,B               {mean(agents):.3f}   "
                      f"(n={len(agents)} agent-sentences)")
            print(f"   majority vote            {mean(maj_s):.3f}")
            print(f"   adjudicator              {mean(adj_s):.3f}")
            if pairs:
                d = mean(f - m for m, f in pairs)
                better = sum(1 for m, f in pairs if f > m)
                tied = sum(1 for m, f in pairs if f == m)
                print(f"   adjudicator - majority   {d:+.3f}   "
                      f"(n={len(pairs)} parsed)")
                print(f"   adjudicator better on {better}, tied {tied}, "
                      f"worse on {len(pairs) - better - tied}")
                print("   The vote is free and deterministic; an adjudicator "
                      "that cannot beat it is\n   cost and opacity with no "
                      "decision content.")
            if crowd and agents and mean(agents) > mean(crowd):
                print("   NOTE: agents above the crowd on 1996 Reuters copy is "
                      "as consistent with\n   pretraining contamination as with "
                      "annotation skill. Not separable here.")

        # ── 4. the anchoring delta ──
        print("\n4. ANCHORING — does the Critic move toward its peers after "
              "seeing them?")
        before, after, moved = [], [], 0
        for r in blind:
            a = r["blind_pair"]["annotations"]
            if not all(a[k]["parse_ok"] for k in ROLES):
                continue
            peers = [spans_of(a["annotator_a"]["entities"]),
                     spans_of(a["annotator_b"]["entities"])]
            cb = spans_of(a["critic_blind"]["entities"])
            cr = spans_of(a["critic_revised"]["entities"])
            before.append(mean(pairwise(cb, p, typed) for p in peers))
            after.append(mean(pairwise(cr, p, typed) for p in peers))
            if set(cb) != set(cr):
                moved += 1
        if before:
            print(f"   critic_blind   vs peers  {mean(before):.3f}")
            print(f"   critic_revised vs peers  {mean(after):.3f}")
            print(f"   movement toward peers    {mean(after) - mean(before):+.3f}"
                  f"   (changed on {moved}/{len(before)} sentences)")
            if mean(before) > 0.95:
                print("   NOTE: the peers already agreed with the blind Critic "
                      "at this level, so\n   there was almost nothing for "
                      "revision to move toward.")
        else:
            print("   no sentence had all four roles parse")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
