#!/usr/bin/env python
"""
RQ2 on DETECTION, not just typing.

Everything we have reported about the Critic-challenge signal was measured on
TYPE disagreement inside the 644 spans BOTH annotators found: 122 disagreements.
But on the 154 doubly-annotated sentences there are 880 further decisions where
only ONE annotator marked a span at all. Detection disagreement outnumbers typing
disagreement roughly 7:1, so the triage result so far covers ~12% of the
disagreement in the corpus.

Three questions, same predictor (did the Critic challenge this span in any
round?), all answerable on runs already on disk:

  A  DETECTION DISAGREEMENT
     Among agent spans at least one human also marked: does a challenge predict
     that only ONE of the two marked it, rather than both?

  B  AGENT FALSE POSITIVES
     Over all agent spans: does a challenge predict that NEITHER human marked
     it? If so the Critic flags the Annotator's own invented spans, which is a
     precision signal and more directly useful than the typing lift.

  C  THE CRITIC'S missing_annotations CLAIMS
     The mirror of B. When the Critic says the Annotator MISSED a span, how often
     did a human actually mark it? This measures the Critic's recall claims, and
     unlike A and B it is a property of the Critic alone.

Matching. Human spans carry char offsets; agent spans carry only text, so spans
are matched on normalised text with the same exact-then-containment rule the rest
of evaluation/ uses (deliberation_history.overlaps). Two human spans with the
same text at different offsets therefore collapse to one. That is a real
limitation and it is the same one every other span-level eval in this project
has; it cannot be fixed without offsets in the agent output.

Only sentences where BOTH annotators marked at least one span are used — on the
rest, "no human marked it" would mean "nobody looked".

    python scripts/xrun/rq2_detection.py
    python scripts/xrun/rq2_detection.py --boot 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import xrun_analysis as X                                   # noqa: E402
from deliberation_history import (load_records, norm, overlaps,  # noqa: E402
                                  agent_disagreed_spans)
from eval_difficulty_split import load_human_spans          # noqa: E402

DIRS = [ROOT / "data" / "auto_annotated_clean",
        ROOT / "data" / "auto_annotated_clean_multi_run"]


def human_text_set(spans) -> set:
    return {norm(s.get("text", "")) for s in spans if norm(s.get("text", ""))}


def in_set(text: str, texts: set) -> bool:
    if text in texts:
        return True
    return any(overlaps(text, t) for t in texts)


def collect(path: Path, mark, dav):
    """One row per agent span: (n_humans, challenged, is_missing_claim, text)."""
    rows = []
    for rec in load_records(path):
        k = norm(rec.get("sentence", ""))
        if k not in mark or k not in dav or not mark[k] or not dav[k]:
            continue
        HA, HB = human_text_set(mark[k]), human_text_set(dav[k])
        for text, slot in agent_disagreed_spans(rec).items():
            if not text:
                continue
            n = int(in_set(text, HA)) + int(in_set(text, HB))
            rows.append((n,
                         1 if slot.get("agent_disagreed") else 0,
                         slot.get("annotator_type") == "(missing)",
                         text))
    return rows


def contrast(y: np.ndarray, c: np.ndarray, boot: int, seed: int = 0):
    """Rate among challenged vs not, lift, bootstrap CI, permutation p."""
    if len(y) == 0 or c.sum() == 0 or (1 - c).sum() == 0:
        return None
    a, b = y[c == 1].mean(), y[c == 0].mean()
    lift = a / b if b else float("nan")
    rng = np.random.default_rng(seed)
    lifts = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        ys, cs = y[i], c[i]
        if cs.sum() == 0 or (1 - cs).sum() == 0:
            continue
        bb = ys[cs == 0].mean()
        if bb:
            lifts.append(ys[cs == 1].mean() / bb)
    ci = (np.percentile(lifts, 2.5), np.percentile(lifts, 97.5)) if len(lifts) > 50 else (None, None)
    obs = a - b
    null = np.empty(boot)
    for j in range(boot):
        cs = rng.permutation(c)
        null[j] = y[cs == 1].mean() - y[cs == 0].mean()
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (boot + 1)
    return dict(n=len(y), n_ch=int(c.sum()), a=a, b=b, lift=lift,
                ci=ci, p=float(p))


def show(title, note, results):
    print()
    print("=" * 112)
    print(title)
    print(note)
    print("=" * 112)
    print(f"{'run':34s} {'n':>5s} {'n chal':>7s} {'chal':>7s} {'not':>7s} "
          f"{'lift':>6s} {'95% CI':>14s} {'perm p':>8s}")
    for name, r in results:
        if r is None:
            print(f"{name[:34]:34s} {'insufficient data':>48s}")
            continue
        ci = f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" if r["ci"][0] is not None else "-"
        star = "*" if r["p"] < 0.05 else " "
        print(f"{name[:34]:34s} {r['n']:5d} {r['n_ch']:7d} {r['a']:7.3f} "
              f"{r['b']:7.3f} {r['lift']:5.2f}x {ci:>14s} {r['p']:7.4f}{star}")


# ───────────────────── length control ─────────────────────
# Both A and B could be artefacts of span extent: a long or oddly delimited span
# is plausibly more likely to attract a Critic challenge AND less likely to match
# what a human marked. Two ways of asking whether the signal is more than length:
#   stratified  recompute the contrast inside token-length strata and combine
#               them with Cochran-Mantel-Haenszel, which is exactly the
#               "same association, every stratum" estimator for this shape.
#   adjusted    logistic regression of the outcome on challenge + length, so the
#               challenge coefficient is read at fixed length.

LEN_BINS = [(1, 1), (2, 2), (3, 3), (4, 99)]


def tok_len(text: str) -> int:
    return max(1, len(text.split()))


def bin_of(n: int) -> str:
    for lo, hi in LEN_BINS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else f"{lo}+"
    return "4+"


def cmh(strata):
    """Cochran-Mantel-Haenszel odds ratio and chi-square p over 2x2 strata.

    strata: [(a, b, c, d)] with a = exposed+event, b = exposed+no event,
    c = unexposed+event, d = unexposed+no event.
    """
    num = den = 0.0
    a_sum = e_sum = v_sum = 0.0
    used = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        r1, r2 = a + b, c + d
        c1, c2 = a + c, b + d
        if min(r1, r2, c1, c2) == 0:
            continue
        num += a * d / n
        den += b * c / n
        a_sum += a
        e_sum += r1 * c1 / n
        v_sum += r1 * r2 * c1 * c2 / (n * n * (n - 1))
        used += 1
    if den == 0 or v_sum == 0:
        return None
    from scipy import stats as _st
    chi = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
    return dict(or_mh=num / den, p=float(_st.chi2.sf(chi, 1)), strata=used)


def adjusted_or(y, c, L):
    """exp(coef) on challenge in logit(y) ~ challenge + length, unregularised."""
    from sklearn.linear_model import LogisticRegression
    if len(set(y)) < 2 or len(set(c)) < 2:
        return None
    X = np.column_stack([c, L]).astype(float)
    try:
        m = LogisticRegression(penalty=None, max_iter=2000).fit(X, y)
    except Exception:
        return None
    return dict(or_adj=float(np.exp(m.coef_[0][0])),
                or_len=float(np.exp(m.coef_[0][1])))


def length_control(name, rows, target):
    """rows: [(n_humans, challenged, is_missing, text)]; target: 'fp' or 'det'."""
    if target == "fp":
        pop = [r for r in rows if not r[2]]
        y = np.array([1 if r[0] == 0 else 0 for r in pop])
    else:
        pop = [r for r in rows if not r[2] and r[0] >= 1]
        y = np.array([1 if r[0] == 1 else 0 for r in pop])
    if not pop:
        return None
    c = np.array([r[1] for r in pop])
    L = np.array([tok_len(r[3]) for r in pop])
    strata, per_bin = [], {}
    for lo, hi in LEN_BINS:
        m = (L >= lo) & (L <= hi)
        if not m.any():
            continue
        ys, cs = y[m], c[m]
        a = int(((cs == 1) & (ys == 1)).sum()); b = int(((cs == 1) & (ys == 0)).sum())
        cc = int(((cs == 0) & (ys == 1)).sum()); d = int(((cs == 0) & (ys == 0)).sum())
        strata.append((a, b, cc, d))
        key = f"{lo}" if lo == hi else f"{lo}+"
        per_bin[key] = dict(n=int(m.sum()),
                            ch_rate=float(cs.mean()),
                            y_rate=float(ys.mean()),
                            lift=(ys[cs == 1].mean() / ys[cs == 0].mean()
                                  if (cs == 1).any() and (cs == 0).any()
                                  and ys[cs == 0].mean() else float("nan")))
    return dict(per_bin=per_bin, cmh=cmh(strata), adj=adjusted_or(y, c, L),
                n=len(pop))


def show_length(title, note, results):
    print()
    print("=" * 112)
    print(title)
    print(note)
    print("=" * 112)
    print("  marginals — is length itself associated with challenge and with the outcome?")
    print(f"  {'run':30s} " + " ".join(f"{b:>19s}" for b in ("1 tok", "2 tok", "3 tok", "4+ tok")))
    print(f"  {'':30s} " + " ".join(f"{'n  chal%  out%':>19s}" for _ in range(4)))
    for name, r in results:
        if r is None:
            continue
        cells = []
        for key in ("1", "2", "3", "4+"):
            d = r["per_bin"].get(key)
            cells.append(f"{d['n']:5d} {d['ch_rate']:5.0%} {d['y_rate']:5.0%}"
                         if d else f"{'-':>19s}")
        print(f"  {name[:30]:30s} " + " ".join(f"{x:>19s}" for x in cells))

    print()
    print("  lift within each length stratum, then combined")
    print(f"  {'run':30s} {'1 tok':>7s} {'2 tok':>7s} {'3 tok':>7s} {'4+ tok':>7s} "
          f"{'CMH OR':>8s} {'p':>8s} {'adj OR':>8s} {'OR/token':>9s}")
    for name, r in results:
        if r is None:
            continue
        cells = []
        for key in ("1", "2", "3", "4+"):
            d = r["per_bin"].get(key)
            v = d["lift"] if d else float("nan")
            cells.append(f"{v:5.2f}x" if v == v else f"{'-':>7s}")
        mh, ad = r["cmh"], r["adj"]
        star = "*" if mh and mh["p"] < 0.05 else " "
        mh_s = f"{mh['or_mh']:8.2f} {mh['p']:7.4f}{star}" if mh else f"{'-':>8s} {'-':>8s}"
        ad_s = f"{ad['or_adj']:8.2f} {ad['or_len']:9.2f}" if ad else f"{'-':>8s} {'-':>9s}"
        print(f"  {name[:30]:30s} " + " ".join(f"{x:>7s}" for x in cells)
              + f" {mh_s} {ad_s}")



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
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

    cache = {name: collect(p, mark, dav) for name, p in runs}

    # ── A: detection disagreement, among spans ≥1 human marked ──
    res_a = []
    for name, _ in runs:
        rows = [r for r in cache[name] if not r[2] and r[0] >= 1]
        y = np.array([1 if n == 1 else 0 for n, _c, _m, _t in rows])
        c = np.array([_c for _n, _c, _m, _t in rows])
        res_a.append((name, contrast(y, c, args.boot)))
    show("A — DETECTION DISAGREEMENT",
         "  P(only ONE human marked this span | Critic challenged it)  vs  | it did not\n"
         "  population: agent-produced spans that at least one human also marked",
         res_a)

    # ── B: agent false positives, over all agent spans ──
    res_b = []
    for name, _ in runs:
        rows = [r for r in cache[name] if not r[2]]
        y = np.array([1 if n == 0 else 0 for n, _c, _m, _t in rows])
        c = np.array([_c for _n, _c, _m, _t in rows])
        res_b.append((name, contrast(y, c, args.boot)))
    show("B — AGENT FALSE POSITIVES",
         "  P(NEITHER human marked this span | Critic challenged it)  vs  | it did not\n"
         "  population: every span the Annotator produced",
         res_b)

    # ── C: the Critic's missing_annotations claims ──
    print()
    print("=" * 112)
    print("C — THE CRITIC'S missing_annotations CLAIMS")
    print("  when the Critic says the Annotator missed a span, how many humans marked it?")
    print("=" * 112)
    print(f"{'run':34s} {'claims':>7s} {'0 humans':>10s} {'1 human':>9s} "
          f"{'2 humans':>10s} {'>=1 (hit rate)':>16s}")
    for name, _ in runs:
        miss = [r for r in cache[name] if r[2]]
        if not miss:
            print(f"{name[:34]:34s} {0:7d}")
            continue
        n0 = sum(1 for n, _c, _m, _t in miss if n == 0)
        n1 = sum(1 for n, _c, _m, _t in miss if n == 1)
        n2 = sum(1 for n, _c, _m, _t in miss if n == 2)
        t = len(miss)
        print(f"{name[:34]:34s} {t:7d} {n0/t:9.0%} {n1/t:8.0%} {n2/t:9.0%} "
              f"{(n1+n2)/t:15.0%}")
    print("\n  A claim no human marked is either a genuine human miss or a Critic")
    print("  invention; this cannot separate those without adjudication.")

    # ── base rates, for reading the lifts above ──
    print()
    print("=" * 112)
    print("BASE RATES (agent-produced spans, missing-claims excluded)")
    print("=" * 112)
    print(f"{'run':34s} {'spans':>6s} {'0 humans':>10s} {'1 human':>9s} "
          f"{'2 humans':>10s} {'challenged':>11s}")
    for name, _ in runs:
        rows = [r for r in cache[name] if not r[2]]
        if not rows:
            print(f"{name[:34]:34s} {0:6d}")
            continue
        t = len(rows)
        n0 = sum(1 for n, _c, _m, _t in rows if n == 0)
        n1 = sum(1 for n, _c, _m, _t in rows if n == 1)
        n2 = sum(1 for n, _c, _m, _t in rows if n == 2)
        ch = sum(_c for _n, _c, _m, _t in rows)
        print(f"{name[:34]:34s} {t:6d} {n0/t:9.0%} {n1/t:8.0%} {n2/t:9.0%} "
              f"{ch/t:10.0%}")


    # ── length control on B and A ──
    show_length("LENGTH CONTROL on B — agent false positives",
                "  'out%' is P(neither human marked the span) inside that length bin.\n"
                "  CMH OR combines the four strata; adj OR is from logit(FP) ~ challenge + length.",
                [(name, length_control(name, cache[name], "fp")) for name, _ in runs])

    show_length("LENGTH CONTROL on A — detection disagreement",
                "  'out%' is P(only one human marked it) inside that length bin,\n"
                "  among agent spans at least one human marked.",
                [(name, length_control(name, cache[name], "det")) for name, _ in runs])

    print()
    print("  OR/token > 1 means longer spans are independently more likely to show the")
    print("  outcome. If adj OR stays near the unadjusted lift, length is not the story.")


if __name__ == "__main__":
    main()
