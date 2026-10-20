#!/usr/bin/env python
"""
Redo the two load-bearing results with OFFSET matching instead of containment.

rq2_detection.py found that a Critic challenge concentrates on spans neither
human marked (2-4x odds, 15/17 runs), and rq2_critic_vs_confidence.py found that
the Critic adds +0.004 AUC over the Annotator's own self-reported confidence.
Both used substring containment to decide whether a human marked a span, because
the Annotator emits no offsets. Containment lets a one-token agent span match any
human span containing that token, which understates false positives for short
spans — and short spans are where the effect lives.

This re-runs both under span_offsets, which resolves agent span text to real
character positions and matches against the humans' own offsets. Three policies
are reported as bounds (overlap / iou50 / exact) alongside the old containment
number, so the effect of the fix is visible rather than asserted.

    python scripts/xrun/rq2_offset_matched.py
    python scripts/xrun/rq2_offset_matched.py --policy exact
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import span_offsets as SO                                     # noqa: E402
import xrun_analysis as X                                     # noqa: E402
from deliberation_history import (load_records, norm, overlaps,  # noqa: E402
                                  try_parse_json, agent_disagreed_spans)
from eval_difficulty_split import load_human_spans            # noqa: E402

DIRS = [ROOT / "data" / "auto_annotated_clean",
        ROOT / "data" / "auto_annotated_clean_multi_run"]
POLICIES = ["containment", "overlap", "iou50", "exact"]


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
            if not t:
                continue
            c = e.get("confidence")
            out[t] = {"raw": raw,
                      "conf": float(c) if isinstance(c, (int, float)) else None}
        return out
    return {}


def collect(path, mark, dav, raws):
    rows = []
    for rec in load_records(path):
        k = norm(rec.get("sentence", ""))
        if k not in mark or k not in dav or not mark[k] or not dav[k]:
            continue
        sent = raws.get(k, rec.get("sentence", ""))
        HAo, HBo = SO.human_offsets(mark[k]), SO.human_offsets(dav[k])
        HAt = {norm(s.get("text", "")) for s in mark[k] if norm(s.get("text", ""))}
        HBt = {norm(s.get("text", "")) for s in dav[k] if norm(s.get("text", ""))}
        disp = agent_disagreed_spans(rec)
        for t, info in annotator_first_pass(rec).items():
            cands, status = SO.resolve(info["raw"], sent)
            slot = disp.get(t) or {}
            n = {}
            # old behaviour, for side-by-side
            n["containment"] = int(t in HAt or any(overlaps(t, u) for u in HAt)) \
                + int(t in HBt or any(overlaps(t, u) for u in HBt))
            for pol in ("overlap", "iou50", "exact"):
                n[pol] = int(SO.matches(cands, HAo, pol)) + \
                         int(SO.matches(cands, HBo, pol))
            rows.append({"n": n, "status": status,
                         "ch": 1 if slot.get("agent_disagreed") else 0,
                         "sev": slot.get("severity") or "none",
                         "conf": info["conf"],
                         "len": max(1, len(t.split()))})
    return rows


def cmh_or(y, c, L):
    """CMH odds ratio over token-length strata + chi-square p."""
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


def cv_auc(y, Xm, repeats=5):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2 or Xm.shape[0] < 80:
        return None
    if any(len(set(Xm[:, j].tolist())) < 2 for j in range(Xm.shape[1])):
        return None
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000))])
    out = []
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        try:
            p = cross_val_predict(pipe, Xm, y, cv=cv, method="predict_proba")[:, 1]
        except Exception:
            return None
        out.append(roc_auc_score(y, p))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="iou50", choices=POLICIES[1:])
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
    cache = {n: collect(p, mark, dav, raws) for n, p in runs}

    print("=" * 112)
    print("0. OFFSET RESOLUTION, and how often each policy says 'a human marked this'")
    print("=" * 112)
    print(f"{'run':30s} {'spans':>6s} {'uniq':>6s} {'ambig':>6s} {'absent':>7s} "
          f"| {'FP% containment':>16s} {'overlap':>8s} {'iou50':>7s} {'exact':>7s}")
    for name, _ in runs:
        rows = cache[name]
        t = len(rows)
        st = {k: sum(1 for r in rows if r["status"] == k) / t
              for k in ("unique", "ambiguous", "absent")}
        fps = {p: np.mean([1 if r["n"][p] == 0 else 0 for r in rows]) for p in POLICIES}
        print(f"{name[:30]:30s} {t:6d} {st['unique']:5.0%} {st['ambiguous']:5.0%} "
              f"{st['absent']:6.0%} | {fps['containment']:15.0%} "
              f"{fps['overlap']:7.0%} {fps['iou50']:6.0%} {fps['exact']:6.0%}")

    print()
    print("=" * 112)
    print("1. B REVISITED — challenge -> false positive, CMH OR over length strata")
    print("=" * 112)
    print(f"{'run':30s} " + " ".join(f"{p:>17s}" for p in POLICIES))
    for name, _ in runs:
        rows = cache[name]
        L = np.array([r["len"] for r in rows])
        c = np.array([r["ch"] for r in rows])
        cells = []
        for p in POLICIES:
            y = np.array([1 if r["n"][p] == 0 else 0 for r in rows])
            orv, pv = cmh_or(y, c, L)
            cells.append(f"{orv:6.2f} {pv:7.4f}{'*' if pv and pv < 0.05 else ' '}"
                         if orv else f"{'-':>17s}")
        print(f"{name[:30]:30s} " + " ".join(f"{x:>17s}" for x in cells))

    print()
    print("=" * 112)
    print(f"2. REDUNDANCY REVISITED under policy '{args.policy}'")
    print("   does the Critic add over the Annotator's own confidence?")
    print("=" * 112)
    print(f"{'run':30s} {'n':>5s} {'FP%':>5s} {'conf':>7s} {'chal':>7s} {'both':>7s} "
          f"{'both-conf':>10s}")
    deltas = []
    for name, _ in runs:
        rows = [r for r in cache[name] if r["conf"] is not None]
        if len(rows) < 80:
            continue
        y = [1 if r["n"][args.policy] == 0 else 0 for r in rows]
        cf = np.array([[r["conf"]] for r in rows], float)
        ch = np.array([[r["ch"]] for r in rows], float)
        bo = np.array([[r["ch"], r["conf"]] for r in rows], float)
        a_cf, a_ch, a_bo = cv_auc(y, cf), cv_auc(y, ch), cv_auc(y, bo)
        d = (a_bo - a_cf) if (a_bo is not None and a_cf is not None) else None
        if d is not None:
            deltas.append(d)
        fm = lambda v, s="{:.3f}": s.format(v) if isinstance(v, float) else "  -  "
        print(f"{name[:30]:30s} {len(rows):5d} {np.mean(y):5.0%} {fm(a_cf):>7s} "
              f"{fm(a_ch):>7s} {fm(a_bo):>7s} {fm(d,'{:+.3f}'):>10s}")
    if deltas:
        print(f"\n  both - conf: median {np.median(deltas):+.3f}, range "
              f"{min(deltas):+.3f} to {max(deltas):+.3f}, positive in "
              f"{sum(1 for d in deltas if d>0)}/{len(deltas)}")

    print()
    print("=" * 112)
    print("3. ABSENT SPANS — surface string not in the sentence at all")
    print("   is the Critic challenging these more than resolvable spans?")
    print("=" * 112)
    print(f"{'run':30s} {'absent n':>9s} {'chal | absent':>14s} "
          f"{'chal | resolved':>16s} {'lift':>6s}")
    for name, _ in runs:
        rows = cache[name]
        ab = [r["ch"] for r in rows if r["status"] == "absent"]
        ok = [r["ch"] for r in rows if r["status"] != "absent"]
        if not ab or not ok:
            print(f"{name[:30]:30s} {len(ab):9d}")
            continue
        a, b = float(np.mean(ab)), float(np.mean(ok))
        print(f"{name[:30]:30s} {len(ab):9d} {a:13.0%} {b:15.0%} "
              f"{(a/b if b else float('nan')):5.2f}x")


if __name__ == "__main__":
    main()
