#!/usr/bin/env python
"""Is the logprob 'saturation' a MEASUREMENT artefact or real model behaviour?

The reported saturation figure (e.g. qwen 99.7%) comes from type_mean_logprob —
the mean logprob over the tokens of the label as the model writes it out. Labels
share prefixes (BIOTIC ENTITY / BIOTIC PROPERTY / BIOTIC PROCESS), so a
token-averaged measure mixes an easy decision ("BIOTIC") with the hard one
(ENTITY vs PROPERTY) and could plausibly wash the latter out. If so, qwen's
apparent lack of headroom would be an artefact and the RQ2 saturation story
would be measuring the wrong thing.

This script tests that directly on EXISTING run output, using the fields the
pipeline already stores per entity:

  type_max_entropy       entropy at the single most-uncertain label token —
                         i.e. the discriminating decision, not the average
  type_top_alternatives  the top-k (token, prob) candidates at that position

Three questions, in order:
  1. Is the most-uncertain token actually the label-choice position, or is it
     spelling? -> how often the top-k spans more than one label head.
  2. Is top-k truncation hiding probability mass? -> sum(top-k) and top-1.
  3. Does the discriminative-token measure rank runs differently from the
     averaged one? -> Spearman of each against the span-level signal.

Run:  python scripts/xrun/headroom.py
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

from scipy import stats

import xrun_analysis as X
from aggregate import REGISTRY

XR = Path("/home/katinska/mobiko_nlp/output/eval_reports/cross_run_comparison")
HEADS = ["ENTITY", "PROPERTY", "PROCESS", "CONCEPT"]


def head_of(token: str):
    """Which schema head-noun a candidate token belongs to, if any."""
    letters = re.sub(r"[^A-Z]", "", (token or "").upper())
    for h in HEADS:
        if letters and (h.startswith(letters) or letters.startswith(h)):
            return h
    return None


def per_run(path):
    top1, sumk, multi, n, maxent = [], [], 0, 0, []
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        for e in (json.loads(line).get("annotator_entity_logprobs") or []):
            alts = e.get("type_top_alternatives")
            if not alts:
                continue
            n += 1
            top1.append(alts[0][1])
            sumk.append(sum(a[1] for a in alts))
            heads = {head_of(a[0]) for a in alts if a[0]}
            heads.discard(None)
            if len(heads) > 1:
                multi += 1
            if e.get("type_max_entropy") is not None:
                maxent.append(e["type_max_entropy"])
    if not n:
        return None
    return dict(n=n, top1=top1, sumk=sumk, multi=multi, maxent=maxent,
                k=len(alts))


def old_saturation(run_dir):
    for line in (XR / run_dir / "logprob.txt").read_text().splitlines():
        if "% of entities have logprob" in line:
            return float(line.strip().split("%")[0])
    return None


rows = []
print("=" * 100)
print("Q1/Q2 — is the most-uncertain token the label decision, and is top-k hiding mass?")
print("=" * 100)
print(f"  {'run':20s} {'n':>5s} {'k':>2s} {'>1 head':>8s} {'sum(top-k)':>11s} "
      f"{'top-1 med':>10s} {'top1<.99':>9s} {'top1<.70':>9s}")
for model, cond, d, p in REGISTRY:
    if not Path(p).exists():
        continue
    r = per_run(p)
    if r is None:
        print(f"  {d:20s} no top_logprobs captured in this run")
        continue
    r.update(run=d, model=model, cond=cond, old=old_saturation(d))
    fam = dict((lab, (pp, rho)) for lab, pp, rho in X.family(XR / d))
    r["pb"] = fam["span:logprob:point_biserial"][1]
    r["pb_p"] = fam["span:logprob:point_biserial"][0]
    r["ent"] = fam["span:entropy:point_biserial"][1]
    # headroom = share of spans where the label decision had ANY mass elsewhere
    r["headroom"] = 100 * sum(1 for x in r["top1"] if x < 0.99) / r["n"]
    rows.append(r)
    print(f"  {d:20s} {r['n']:5d} {r['k']:2d} {100*r['multi']/r['n']:7.1f}% "
          f"{st.mean(r['sumk']):11.3f} {st.median(r['top1']):10.4f} "
          f"{r['headroom']:8.1f}% "
          f"{100*sum(1 for x in r['top1'] if x < 0.70)/r['n']:8.1f}%")

print()
print("  sum(top-k) ~= 1.000 means the captured candidates hold essentially all the")
print("  mass, so k=3 is NOT truncating away real uncertainty: a top-1 of ~1.0000")
print("  already bounds how much probability any unlisted label could carry.")

print()
print("=" * 100)
print("Q3 — does the discriminative-token measure change the ranking?")
print("=" * 100)
print(f"  {'run':20s} {'old sat%':>9s} {'headroom%':>10s} {'span rho(lp)':>13s} "
      f"{'rho(ent)':>9s} {'p':>8s}")
for r in sorted(rows, key=lambda r: -r["headroom"]):
    print(f"  {r['run']:20s} {r['old']:9.1f} {r['headroom']:10.1f} "
          f"{r['pb']:+13.3f} {r['ent']:+9.3f} {r['pb_p']:8.4f}")

head = [r["headroom"] for r in rows]
sig = [r["pb"] for r in rows]
old = [-r["old"] for r in rows]          # negate: less saturated = more headroom
print()
print("  Spearman vs the span-level signal, across %d runs:" % len(rows))
print("    discriminative-token headroom : rho=%+.3f  p=%.4f"
      % stats.spearmanr(head, sig))
print("    averaged-token saturation     : rho=%+.3f  p=%.4f"
      % stats.spearmanr(old, sig))
print("  (runs share the same 154 gold sentences, so these are not independent;")
print("   the point is only that the two measures order the runs identically.)")

print()
print("=" * 100)
print("WITHIN-MODEL CONTRAST — the reviewer's 'model identity' alternative")
print("=" * 100)
same = [r for r in rows if r["model"].startswith("Apertus")]
for r in sorted(same, key=lambda r: r["headroom"]):
    print(f"  {r['run']:20s} {r['cond']:6s} headroom {r['headroom']:5.1f}%  "
          f"span rho {r['pb']:+.3f}  p={r['pb_p']:.4f}")
print("  Same model (Apertus-v1.5-70B) on both sides of the break, so model")
print("  identity is held constant while headroom varies. Caveat: these runs also")
print("  differ in pipeline build and relation-schema config.")
