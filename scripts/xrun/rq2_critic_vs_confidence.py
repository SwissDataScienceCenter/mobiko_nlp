#!/usr/bin/env python
"""
Does the Critic's objection add anything over the Annotator's own uncertainty?

rq2_detection.py established that a Critic challenge concentrates on spans
neither human marked, at 2-4x odds, in 15 of 17 runs, independent of span
length. That is only interesting if a SECOND agent is needed to get it. The
Annotator already reports a per-span confidence and already carries token
logprobs for the type it chose — both available from one pass, at a third of
the cost. If those predict false positives as well as the Critic does, the
multi-agent architecture is not earning its keep on this signal.

Four single instruments, same target (no human marked the span), same
population (spans the Annotator proposed in its FIRST round):

  challenged       the Critic disputed this span in any round         (binary)
  1 - confidence   the Annotator's own self-reported confidence     (continuous)
  -type_mean_logprob    token logprob of the type it emitted        (continuous)
  type_mean_entropy     predictive entropy over that type           (continuous)

Then the question that decides it: cross-validated AUC for confidence alone,
challenge alone, and both together, plus the challenge odds ratio adjusted for
confidence and length. If `both` does not beat `confidence`, the Critic is
redundant here.

Self-reported confidence is read from the Annotator's own first-round message,
NOT from `final_entities`. `final_entities` is the Adjudicator's output: its
confidences have already been revised by the deliberation under test, so using
them would leak the Critic's influence into the "single pass" baseline.

A binary predictor's AUC is mechanically capped (it has two ranks), so the
single-instrument AUCs are not a fair head-to-head against a continuous score.
The CV comparison and the adjusted OR are the ones to read.

    python scripts/xrun/rq2_critic_vs_confidence.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import xrun_analysis as X                                      # noqa: E402
from deliberation_history import (load_records, norm, overlaps,  # noqa: E402
                                  try_parse_json, agent_disagreed_spans)
from eval_difficulty_split import load_human_spans             # noqa: E402

DIRS = [ROOT / "data" / "auto_annotated_clean",
        ROOT / "data" / "auto_annotated_clean_multi_run"]


def annotator_first_pass(rec: dict):
    """norm_text -> {confidence}. The Annotator's OWN first message only."""
    for m in rec.get("messages", []):
        if m.get("agent") != "Annotator":
            continue
        parsed = try_parse_json(m.get("content", ""))
        if not parsed or "entities" not in parsed:
            continue
        out = {}
        for e in parsed["entities"]:
            t = norm(e.get("text", ""))
            if not t:
                continue
            c = e.get("confidence")
            out[t] = {"conf": float(c) if isinstance(c, (int, float)) else None}
        return out
    return {}


def logprob_map(rec: dict):
    out = {}
    for e in rec.get("annotator_entity_logprobs") or []:
        t = norm(e.get("text", ""))
        if t:
            out[t] = {"lp": e.get("type_mean_logprob"),
                      "ent": e.get("type_mean_entropy")}
    return out


def human_texts(spans):
    return {norm(s.get("text", "")) for s in spans if norm(s.get("text", ""))}


def in_set(t, texts):
    return t in texts or any(overlaps(t, u) for u in texts)


def collect(path: Path, mark, dav):
    rows = []
    for rec in load_records(path):
        k = norm(rec.get("sentence", ""))
        if k not in mark or k not in dav or not mark[k] or not dav[k]:
            continue
        HA, HB = human_texts(mark[k]), human_texts(dav[k])
        first = annotator_first_pass(rec)
        lps = logprob_map(rec)
        disp = agent_disagreed_spans(rec)
        for t, info in first.items():
            slot = disp.get(t) or {}
            lp = lps.get(t) or {}
            rows.append({
                "fp": 1 if not (in_set(t, HA) or in_set(t, HB)) else 0,
                "ch": 1 if slot.get("agent_disagreed") else 0,
                "sev": (slot.get("severity") or "none"),
                "conf": info["conf"],
                "lp": lp.get("lp"),
                "ent": lp.get("ent"),
                "len": max(1, len(t.split())),
            })
    return rows


def auc(y, score):
    s = np.asarray(score, dtype=float)
    ok = ~np.isnan(s)
    y2 = np.asarray(y)[ok]
    if len(set(y2.tolist())) < 2 or len(set(s[ok].tolist())) < 2:
        return None
    return float(roc_auc_score(y2, s[ok]))


def cv_auc(y, Xm, repeats=5, seed=0):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2 or Xm.shape[0] < 50:
        return None
    if any(len(set(Xm[:, j].tolist())) < 2 for j in range(Xm.shape[1])):
        return None
    # cross_val_predict needs every row predicted exactly once, so a repeated
    # splitter is invalid here. Repeat plain StratifiedKFold over seeds instead
    # and average the per-repeat AUCs.
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000))])
    scores = []
    for r in range(repeats):
        cvs = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + r)
        try:
            pr = cross_val_predict(pipe, Xm, y, cv=cvs,
                                   method="predict_proba")[:, 1]
        except Exception:
            return None
        scores.append(roc_auc_score(y, pr))
    return float(np.mean(scores)) if scores else None


def adjusted(y, Xm):
    """exp(coef) on the FIRST column, unregularised."""
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    try:
        m = LogisticRegression(C=np.inf, max_iter=3000).fit(Xm, y)
    except Exception:
        return None
    return float(np.exp(m.coef_[0][0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    mark = load_human_spans(Path(X.MARK))
    dav = load_human_spans(Path(X.DAV))
    runs = []
    for d in DIRS:
        if not d.exists():
            continue
        tag = "PF " if "multi_run" in str(d) else "   "
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name.startswith("multi_pass"):
                runs.append((tag + f.name, f))
    cache = {n: collect(p, mark, dav) for n, p in runs}

    # ── 1. severity gradation ──
    print("=" * 112)
    print("1. SEVERITY GRADATION — FP rate by the Critic's own grade")
    print("   does major outrank minor outrank unchallenged?")
    print("=" * 112)
    print(f"{'run':30s} {'none n':>7s} {'none':>6s} {'minor n':>8s} {'minor':>6s} "
          f"{'major n':>8s} {'major':>6s} {'monotone':>9s}")
    for name, _ in runs:
        rows = cache[name]
        g = {}
        for key, pred in (("none", lambda r: r["ch"] == 0),
                          ("minor", lambda r: r["ch"] == 1 and r["sev"] == "minor"),
                          ("major", lambda r: r["ch"] == 1 and r["sev"] == "major")):
            sub = [r["fp"] for r in rows if pred(r)]
            g[key] = (len(sub), float(np.mean(sub)) if sub else None)
        vals = [g[k][1] for k in ("none", "minor", "major")]
        mono = ("yes" if all(v is not None for v in vals)
                and vals[0] <= vals[1] <= vals[2] else "no")
        fm = lambda v: f"{v:.3f}" if isinstance(v, float) else "  -  "
        print(f"{name[:30]:30s} {g['none'][0]:7d} {fm(g['none'][1]):>6s} "
              f"{g['minor'][0]:8d} {fm(g['minor'][1]):>6s} "
              f"{g['major'][0]:8d} {fm(g['major'][1]):>6s} {mono:>9s}")

    # ── 2. single instruments ──
    print()
    print("=" * 112)
    print("2. SINGLE INSTRUMENTS — AUC for predicting 'no human marked this span'")
    print("   a binary predictor's AUC is capped by construction; read section 3 instead")
    print("=" * 112)
    print(f"{'run':30s} {'n':>5s} {'FP%':>5s} {'challenge':>10s} {'1-conf':>8s} "
          f"{'-logprob':>9s} {'entropy':>8s} {'conf n':>7s}")
    for name, _ in runs:
        rows = cache[name]
        y = [r["fp"] for r in rows]
        nconf = sum(1 for r in rows if r["conf"] is not None)
        a_ch = auc(y, [r["ch"] for r in rows])
        a_cf = auc(y, [1.0 - r["conf"] if r["conf"] is not None else np.nan
                       for r in rows])
        a_lp = auc(y, [-r["lp"] if isinstance(r["lp"], (int, float)) else np.nan
                       for r in rows])
        a_en = auc(y, [r["ent"] if isinstance(r["ent"], (int, float)) else np.nan
                       for r in rows])
        fm = lambda v: f"{v:.3f}" if isinstance(v, float) else "  -  "
        print(f"{name[:30]:30s} {len(rows):5d} {np.mean(y):5.0%} {fm(a_ch):>10s} "
              f"{fm(a_cf):>8s} {fm(a_lp):>9s} {fm(a_en):>8s} {nconf:7d}")

    # ── 3. the decisive comparison ──
    print()
    print("=" * 112)
    print("3. DOES THE CRITIC ADD OVER THE ANNOTATOR'S OWN CONFIDENCE?")
    print("   5-fold CV AUC x5 repeats; adj OR is on challenge, controlling conf + length")
    print("=" * 112)
    print(f"{'run':30s} {'n':>5s} {'conf':>7s} {'chal':>7s} {'both':>7s} "
          f"{'both-conf':>10s} {'adj OR':>8s}")
    deltas = []
    for name, _ in runs:
        rows = [r for r in cache[name] if r["conf"] is not None]
        if len(rows) < 80:
            print(f"{name[:30]:30s} {len(rows):5d}   too few with confidence")
            continue
        y = [r["fp"] for r in rows]
        cf = np.array([[r["conf"]] for r in rows], dtype=float)
        ch = np.array([[r["ch"]] for r in rows], dtype=float)
        bo = np.array([[r["ch"], r["conf"]] for r in rows], dtype=float)
        adj = np.array([[r["ch"], r["conf"], r["len"]] for r in rows], dtype=float)
        a_cf, a_ch, a_bo = cv_auc(y, cf, args.repeats), cv_auc(y, ch, args.repeats), cv_auc(y, bo, args.repeats)
        orv = adjusted(y, adj)
        d = (a_bo - a_cf) if (a_bo is not None and a_cf is not None) else None
        if d is not None:
            deltas.append(d)
        fm = lambda v, s="{:.3f}": s.format(v) if isinstance(v, float) else "  -  "
        print(f"{name[:30]:30s} {len(rows):5d} {fm(a_cf):>7s} {fm(a_ch):>7s} "
              f"{fm(a_bo):>7s} {fm(d,'{:+.3f}'):>10s} {fm(orv,'{:.2f}'):>8s}")
    if deltas:
        print(f"\n  both - conf across runs: median {np.median(deltas):+.3f}, "
              f"range {min(deltas):+.3f} to {max(deltas):+.3f}, "
              f"positive in {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
        print("  adj OR > 1 means a challenge still predicts an FP at fixed")
        print("  self-reported confidence and fixed span length.")


if __name__ == "__main__":
    main()
