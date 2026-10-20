#!/usr/bin/env python
"""
Recompute RQ2 against a POSITION-based human target, on the archived runs.

The RQ2 question is whether an agent's signals predict where the humans
disagreed. That makes the human-disagreement target the dependent variable, and
it has been measured three incompatible ways:

    text-keyed     eval_layer2_correlation — two annotators marking one mention
                   at different extents become two unrelated entities, each
                   internally unanimous, so the type conflict vanishes
    exact-offset   split_buckets — the same pair lands in detection_only, "one
                   annotator missed it", so the type conflict vanishes differently
    clustered      here — one mention, with presence, extent and type recorded
                   separately

On MoBiKo's 154 doubly-annotated sentences those find 128, 122 and 180-233 type
disagreements. The agents are unchanged; only the target moves. Any shift in the
correlations below is therefore a measurement artefact being removed, not a new
finding about the models.

Agent spans are resolved to offsets with span_offsets (the harness records them
for new runs; archived runs predate the field) and matched to mentions by
overlap, so an agent that marked a real mention at a slightly different extent
counts as talking about that mention rather than as a miss.

    python scripts/xrun/rq2_recompute.py
    python scripts/xrun/rq2_recompute.py --nested
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import human_agreement as H                                   # noqa: E402
import span_offsets as SO                                     # noqa: E402
import xrun_analysis as X                                     # noqa: E402
import eval_layer2_correlation as L2                          # noqa: E402
from deliberation_history import (load_records, norm,          # noqa: E402
                                  try_parse_json, agent_disagreed_spans)
from eval_difficulty_split import load_human_spans            # noqa: E402

DIRS = [ROOT / "data" / "auto_annotated_clean",
        ROOT / "data" / "auto_annotated_clean_multi_run"]


def annotator_first_pass(rec):
    for m in rec.get("messages", []):
        if m.get("agent") != "Annotator":
            continue
        p = try_parse_json(m.get("content", ""))
        if not p or "entities" not in p:
            continue
        return p["entities"]
    return []


def collect(path, mark, dav, raws, nested):
    """One row per agent span matched to a human mention."""
    rows = []
    for rec in load_records(path):
        k = norm(rec.get("sentence", ""))
        if k not in mark or k not in dav or not mark[k] or not dav[k]:
            continue
        sent = raws.get(k, rec.get("sentence", ""))
        mentions = H.mention_disagreements([mark[k], dav[k]], nested=nested)
        # text-keyed target, for the side-by-side
        text_dis = L2.compute_human_span_disagreements({
            "m": {"entities": [{"text": s["text"], "entity_type": s["type"]}
                               for s in mark[k]]},
            "d": {"entities": [{"text": s["text"], "entity_type": s["type"]}
                               for s in dav[k]]}})
        lp = {norm(e.get("text", "")): e for e in
              (rec.get("annotator_entity_logprobs") or [])}
        disp = agent_disagreed_spans(rec)
        for e in annotator_first_pass(rec):
            raw = (e.get("text") or "").strip()
            t = norm(raw)
            if not t:
                continue
            occ = SO.resolve(raw, sent)[0]
            if not occ:
                continue
            mm = H.match(mentions, occ[0], nested=nested)
            old = text_dis.get(t)
            l = lp.get(t) or {}
            rows.append({
                "new_type": (1 if mm and mm["type_disagreed"] else 0) if mm else None,
                "old_type": (1 if old["type_disagreed"] else 0) if old else None,
                "new_graded": mm["graded_type"] if mm else None,
                "challenged": 1 if (disp.get(t) or {}).get("agent_disagreed") else 0,
                "logprob": l.get("type_mean_logprob"),
                "entropy": l.get("type_max_entropy"),
                "conf": e.get("confidence"),
                "len": max(1, len(t.split())),
            })
    return rows


def pb(y, x):
    """Point-biserial Spearman with a permutation p, as the paper reports it."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    if len(y) < 30 or len(set(y.tolist())) < 2 or len(set(x.tolist())) < 2:
        return None, None, len(y)
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(0)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic
                     for _ in range(2000)])
    p = (np.sum(np.abs(null) >= abs(rho)) + 1) / 2001
    return float(rho), float(p), len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nested", action="store_true",
                    help="treat a span contained in another as the same mention")
    args = ap.parse_args()

    mark, dav = load_human_spans(Path(X.MARK)), load_human_spans(Path(X.DAV))
    raws = SO.raw_sentences(Path(X.MARK))
    runs = []
    for d in DIRS:
        if not d.exists():
            continue
        tag = "PF " if "multi_run" in str(d) else "   "
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name.startswith("multi_pass"):
                runs.append((tag + f.name, f))

    rule = "IoU>=0.5 or nested" if args.nested else "IoU>=0.5"
    print("=" * 108)
    print(f"RQ2 RECOMPUTED — human target grouped by POSITION ({rule})")
    print("  the agents are unchanged; only the target moves")
    print("=" * 108)
    print(f"{'run':30s} {'n':>5s} | {'TEXT-KEYED target':^24s} | "
          f"{'CLUSTERED target':^24s} | {'graded':^15s}")
    print(f"{'':30s} {'':>5s} | {'rate':>6s} {'rho':>8s} {'p':>7s} | "
          f"{'rate':>6s} {'rho':>8s} {'p':>7s} | {'rho':>7s} {'p':>7s}")
    for name, path in runs:
        rows = collect(path, mark, dav, raws, args.nested)
        lp = [r for r in rows if isinstance(r["logprob"], (int, float))]
        old = [r for r in lp if r["old_type"] is not None]
        new = [r for r in lp if r["new_type"] is not None]
        ro, po, no = pb([r["old_type"] for r in old], [-r["logprob"] for r in old])
        rn, pn, nn = pb([r["new_type"] for r in new], [-r["logprob"] for r in new])
        rg, pg, _ = pb([r["new_graded"] for r in new], [-r["logprob"] for r in new])
        f = lambda v, s="{:+.3f}": s.format(v) if isinstance(v, float) else "   -  "
        star = lambda p: "*" if isinstance(p, float) and p < 0.05 else " "
        print(f"{name[:30]:30s} {len(new):5d} | "
              f"{np.mean([r['old_type'] for r in old]) if old else float('nan'):6.1%} "
              f"{f(ro):>8s} {f(po,'{:.4f}'):>6s}{star(po)}| "
              f"{np.mean([r['new_type'] for r in new]) if new else float('nan'):6.1%} "
              f"{f(rn):>8s} {f(pn,'{:.4f}'):>6s}{star(pn)}| "
              f"{f(rg):>7s} {f(pg,'{:.4f}'):>6s}{star(pg)}")

    print()
    print("  rho = point-biserial Spearman of NEGATED annotator type-logprob against")
    print("  human type-disagreement; positive means lower model confidence where the")
    print("  humans disagreed. 'graded' replaces the binary target with 1 - modal")
    print("  label share, which does not saturate as annotator count grows.")


if __name__ == "__main__":
    raise SystemExit(main())
