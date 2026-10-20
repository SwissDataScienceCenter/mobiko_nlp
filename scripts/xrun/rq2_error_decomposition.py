#!/usr/bin/env python
"""
Does the Critic tell a boundary slip from an invented span?

With real offsets on both sides, an agent span no longer has to be scored by one
match threshold. Against the UNION of the two humans' spans it falls into exactly
one of three states:

    exact      some human span has identical (start, end)        -> correct
    boundary   overlaps a human span, but no exact match         -> extent error
    none       overlaps no human span at all                     -> invention

Everything reported so far pooled `boundary` and `none` into "no human marked
it", which is what the iou50/overlap/exact policies were sliding between: iou50
counts most boundary errors as matches, `exact` counts all of them as misses.
Pooling them asks whether the Critic objects to spans that are wrong. Separating
them asks whether the Critic knows HOW they are wrong, which is the thing that
would make its objections worth acting on — dropping an invented span is right,
dropping a mis-delimited real entity throws away a true positive.

The discrimination test is the challenge rate on `none` against the challenge
rate on `boundary`, restricted to those two categories, stratified by token
length (Cochran-Mantel-Haenszel). A ratio near 1 means the Critic objects to both
kinds equally and cannot distinguish them.

AMBIGUOUS SPANS. Archived runs carry no offsets (the harness only began
recording them per round after this analysis), so positions are recovered by
literal search. Where a span text occurs more than once, the BEST achievable
category across candidates is taken — exact over boundary over none. That is
deliberately charitable: it biases against finding inventions, so it cannot
manufacture the effect. A sensitivity pass excluding ambiguous spans is reported
alongside.

    python scripts/xrun/rq2_error_decomposition.py
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

import span_offsets as SO                                   # noqa: E402
import xrun_analysis as X                                   # noqa: E402
from deliberation_history import (load_records, norm,        # noqa: E402
                                  try_parse_json, agent_disagreed_spans)
from eval_difficulty_split import load_human_spans          # noqa: E402

DIRS = [ROOT / "data" / "auto_annotated_clean",
        ROOT / "data" / "auto_annotated_clean_multi_run"]
RANK = {"exact": 2, "boundary": 1, "none": 0}


def annotator_first_pass(rec):
    for m in rec.get("messages", []):
        if m.get("agent") != "Annotator":
            continue
        parsed = try_parse_json(m.get("content", ""))
        if not parsed or "entities" not in parsed:
            continue
        out = {}
        for e in parsed["entities"]:
            raw = (e.get("text") or "").strip()
            t = norm(raw)
            if t:
                out[t] = raw
        return out
    return {}


def categorise(cand, human):
    """Best state this span can be in against the human span union."""
    best, width = "none", None
    for c in cand:
        for h in human:
            if c == h:
                return "exact", 0
            if min(c[1], h[1]) > max(c[0], h[0]):
                if RANK["boundary"] > RANK[best]:
                    best = "boundary"
                    width = (c[1] - c[0]) - (h[1] - h[0])
    return best, width


def collect(path, mark, dav, raws):
    rows = []
    for rec in load_records(path):
        k = norm(rec.get("sentence", ""))
        if k not in mark or k not in dav or not mark[k] or not dav[k]:
            continue
        sent = raws.get(k, rec.get("sentence", ""))
        human = SO.human_offsets(mark[k]) + SO.human_offsets(dav[k])
        disp = agent_disagreed_spans(rec)
        for t, raw in annotator_first_pass(rec).items():
            cand, status = SO.resolve(raw, sent)
            cat, width = categorise(cand, human)
            slot = disp.get(t) or {}
            rows.append({"cat": cat, "width": width, "status": status,
                         "ch": 1 if slot.get("agent_disagreed") else 0,
                         "sev": slot.get("severity") or "none",
                         "len": max(1, len(t.split()))})
    return rows


def cmh(y, c, L):
    num = den = a_s = e_s = v_s = 0.0
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 99)):
        m = (L >= lo) & (L <= hi)
        if not m.any():
            continue
        ys, cs = y[m], c[m]
        a = ((cs == 1) & (ys == 1)).sum(); b = ((cs == 1) & (ys == 0)).sum()
        cc = ((cs == 0) & (ys == 1)).sum(); d = ((cs == 0) & (ys == 0)).sum()
        nn = a + b + cc + d
        r1, r2, c1, c2 = a + b, cc + d, a + cc, b + d
        if nn < 2 or min(r1, r2, c1, c2) == 0:
            continue
        num += a * d / nn; den += b * cc / nn
        a_s += a; e_s += r1 * c1 / nn
        v_s += r1 * r2 * c1 * c2 / (nn * nn * (nn - 1))
    if den == 0 or v_s == 0:
        return None, None
    chi = (abs(a_s - e_s) - 0.5) ** 2 / v_s
    return num / den, float(stats.chi2.sf(chi, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude-ambiguous", action="store_true")
    args = ap.parse_args()

    mark = load_human_spans(Path(X.MARK))
    dav = load_human_spans(Path(X.DAV))
    raws = SO.raw_sentences(Path(X.MARK))
    runs = []
    for d in DIRS:
        if not d.exists():
            continue
        tag = "PF " if "multi_run" in str(d) else "   "
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name.startswith("multi_pass"):
                runs.append((tag + f.name, f))
    cache = {}
    for n, p in runs:
        rows = collect(p, mark, dav, raws)
        if args.exclude_ambiguous:
            rows = [r for r in rows if r["status"] != "ambiguous"]
        cache[n] = rows

    print("=" * 112)
    print("1. WHERE AGENT SPANS LAND against the union of both humans' spans")
    print("=" * 112)
    print(f"{'run':30s} {'spans':>6s} {'exact':>7s} {'boundary':>9s} {'none':>6s} "
          f"| {'median width delta':>19s}")
    for name, _ in runs:
        rows = cache[name]
        t = len(rows) or 1
        sh = {k: sum(1 for r in rows if r["cat"] == k) / t
              for k in ("exact", "boundary", "none")}
        w = [r["width"] for r in rows if r["cat"] == "boundary"
             and r["width"] is not None]
        wd = f"{np.median(w):+.0f} chars" if w else "-"
        print(f"{name[:30]:30s} {len(rows):6d} {sh['exact']:6.0%} "
              f"{sh['boundary']:8.0%} {sh['none']:5.0%} | {wd:>19s}")
    print("  width delta = agent span length minus the human span it overlaps;")
    print("  positive means the agent over-extends.")

    print()
    print("=" * 112)
    print("2. CHALLENGE RATE WITHIN EACH STATE — is the Critic graded?")
    print("=" * 112)
    print(f"{'run':30s} {'chal|exact':>11s} {'chal|boundary':>14s} {'chal|none':>10s} "
          f"{'none/bound':>11s} {'none/exact':>11s}")
    for name, _ in runs:
        rows = cache[name]
        r = {}
        for k in ("exact", "boundary", "none"):
            sub = [x["ch"] for x in rows if x["cat"] == k]
            r[k] = float(np.mean(sub)) if sub else None
        f = lambda v: f"{v:.0%}" if isinstance(v, float) else "  -  "
        nb = (r["none"] / r["boundary"]) if r["none"] is not None and r["boundary"] else None
        ne = (r["none"] / r["exact"]) if r["none"] is not None and r["exact"] else None
        g = lambda v: f"{v:.2f}x" if isinstance(v, float) else "  -  "
        print(f"{name[:30]:30s} {f(r['exact']):>11s} {f(r['boundary']):>14s} "
              f"{f(r['none']):>10s} {g(nb):>11s} {g(ne):>11s}")

    print()
    print("=" * 112)
    print("3. DISCRIMINATION TEST — among WRONG spans only (boundary + none),")
    print("   does a challenge predict invention rather than a boundary slip?")
    print("   CMH odds ratio over token-length strata; ~1.0 = cannot distinguish")
    print("=" * 112)
    print(f"{'run':30s} {'n wrong':>8s} {'none%':>6s} {'none|chal':>10s} "
          f"{'none|not':>9s} {'CMH OR':>8s} {'p':>8s}")
    ors = []
    for name, _ in runs:
        rows = [r for r in cache[name] if r["cat"] in ("boundary", "none")]
        if len(rows) < 40:
            print(f"{name[:30]:30s} {len(rows):8d}   too few")
            continue
        y = np.array([1 if r["cat"] == "none" else 0 for r in rows])
        c = np.array([r["ch"] for r in rows])
        L = np.array([r["len"] for r in rows])
        a = y[c == 1].mean() if (c == 1).any() else float("nan")
        b = y[c == 0].mean() if (c == 0).any() else float("nan")
        orv, pv = cmh(y, c, L)
        if orv:
            ors.append(orv)
        print(f"{name[:30]:30s} {len(rows):8d} {y.mean():5.0%} {a:9.0%} {b:8.0%} "
              + (f"{orv:8.2f} {pv:7.4f}{'*' if pv < 0.05 else ' '}" if orv
                 else f"{'-':>8s} {'-':>8s}"))
    if ors:
        print(f"\n  CMH OR across runs: median {np.median(ors):.2f}, "
              f"range {min(ors):.2f} to {max(ors):.2f}, "
              f"above 1.0 in {sum(1 for o in ors if o > 1)}/{len(ors)}")

    print()
    print("=" * 112)
    print("4. SEVERITY vs STATE — does the Critic's own grade track the error kind?")
    print("=" * 112)
    print(f"{'run':30s} {'major|exact':>12s} {'major|bound':>12s} {'major|none':>11s}")
    for name, _ in runs:
        rows = cache[name]
        out = []
        for k in ("exact", "boundary", "none"):
            sub = [1 if x["sev"] == "major" else 0
                   for x in rows if x["cat"] == k and x["ch"] == 1]
            out.append(f"{np.mean(sub):.0%}" if sub else "  -  ")
        print(f"{name[:30]:30s} {out[0]:>12s} {out[1]:>12s} {out[2]:>11s}")


if __name__ == "__main__":
    main()
