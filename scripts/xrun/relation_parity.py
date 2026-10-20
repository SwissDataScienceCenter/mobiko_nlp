#!/usr/bin/env python
"""
Two questions about ONE run, on the relation gold:

  parity   is the pipeline statistically indistinguishable from a human
           annotator? Per sentence, agent = mean(F1 vs Mark, F1 vs Davnah) and
           human = F1(Mark, Davnah) — the SAME paired design as
           xrun_analysis.parity, extended from entities to relation pairs and
           triplets. Paired t, Wilcoxon, and TOST at ±0.05 / ±0.03: a small
           p on the difference is not evidence of sameness, only equivalence
           testing can affirm that, and with n≈130 sentences the test is weak —
           it is reported with its CI so the width is visible.

  predict  does the run's own deliberation predict where the HUMANS disagree?
           Spearman of each signal against per-sentence human disagreement
           (1 − F1 between Mark and Davnah), plus ROC-AUC on the binary
           "harder than the median" label, with Benjamini-Hochberg over the
           whole family of tests so that 20 signals do not manufacture a hit.

Signals come from deliberation_history.record_signals (the same extractor the
difficulty model uses) plus the record's own logprob/entropy fields, so nothing
here re-derives a feature that already exists elsewhere.

    python scripts/xrun/relation_parity.py
    python scripts/xrun/relation_parity.py --run multi_pass_deepseek_1 --layer triplets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import eval_relation_agent as RA        # noqa: E402
import eval_relation_iaa as RI          # noqa: E402
from deliberation_history import record_signals  # noqa: E402

GOLD = {
    "Mark": ROOT / "data/aug_runs/Mark_relation_input_relation/Mark/relations_merged.jsonl",
    "Davnah": ROOT / "data/aug_runs/Davnah_relation_input_relation/Davnah/relations_merged.jsonl",
}
RUN_DIR = ROOT / "data/auto_annotated_clean_multi_run"
LAYERS = ("entities", "pairs", "triplets")


# ── per-sentence agreement, one layer ───────────────────────────────────────

def f1(tp: int, na: int, nb: int) -> float:
    """Symmetric per-sentence F1 — the same 2·agree/(n_a+n_b) used everywhere
    here. Two empty annotations count as perfect agreement (nothing to find,
    nothing found); those sentences are excluded before testing anyway."""
    return (2 * tp / (na + nb)) if (na + nb) else 1.0


def sentence_scores(a: dict, b: dict, layer: str, mode: str, view: str) -> float:
    if layer == "entities":
        ea = {(s.start, s.end, s.type) for s in RA.entity_spans(a)}
        eb = {(s.start, s.end, s.type) for s in RA.entity_spans(b)}
        return f1(len(ea & eb), len(ea), len(eb))
    ra, _, _ = RI.sentence_relations(a, view)
    rb, _, _ = RI.sentence_relations(b, view)
    m = RI.match_relations(ra, rb, mode, labeled=(layer == "triplets"))
    return f1(len(m), len(ra), len(rb))


def paired_block(d: np.ndarray) -> dict:
    n = len(d)
    mean, se = float(np.mean(d)), float(stats.sem(d))
    crit = stats.t.ppf(0.975, n - 1)
    out = {"n": n, "mean": mean, "se": se,
           "ci": (mean - crit * se, mean + crit * se),
           "t_p": float(stats.ttest_1samp(d, 0.0).pvalue)}
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    for margin in (0.05, 0.03):
        # TOST: two one-sided t-tests; equivalence is affirmed when BOTH reject.
        lo = float(stats.ttest_1samp(d + margin, 0.0, alternative="greater").pvalue)
        hi = float(stats.ttest_1samp(d - margin, 0.0, alternative="less").pvalue)
        out[f"tost_{margin}"] = max(lo, hi)
    return out


def parity(agent: dict, golds: dict, layer: str, mode: str, view: str) -> dict:
    a_scores, h_scores = [], []
    mark, dav = golds["Mark"], golds["Davnah"]
    for key in sorted(set(mark) & set(dav) & set(agent)):
        # Score only where there is something to agree about: at least one of
        # the three parties produced an item of this layer for this sentence.
        parties = [agent[key], mark[key], dav[key]]
        if layer == "entities":
            sizes = [len(RA.entity_spans(p)) for p in parties]
        else:
            sizes = [len(RI.sentence_relations(p, view)[0]) for p in parties]
        if not any(sizes):
            continue
        a_scores.append((sentence_scores(agent[key], mark[key], layer, mode, view)
                         + sentence_scores(agent[key], dav[key], layer, mode, view)) / 2)
        h_scores.append(sentence_scores(mark[key], dav[key], layer, mode, view))
    a, h = np.array(a_scores), np.array(h_scores)
    r = paired_block(a - h)
    r.update(agent_mean=float(a.mean()), human_mean=float(h.mean()))
    return r


# ── can the run predict human disagreement? ─────────────────────────────────

def signals(rec: dict) -> dict:
    out = dict(record_signals(rec))
    for role in ("annotator", "critic", "adjudicator"):
        for field in ("mean_logprob", "mean_entropy"):
            v = rec.get(f"{role}_{field}")
            if isinstance(v, (int, float)):
                out[f"{role}_{field}"] = float(v)
    out["agreement_score"] = rec.get("agreement_score") or 0.0
    out["flagged"] = 1.0 if rec.get("flagged_for_human_review") else 0.0
    out["n_entities"] = len(rec.get("final_entities") or [])
    out["n_relations"] = len(rec.get("final_relations") or [])
    out["n_messages"] = len(rec.get("messages") or [])
    return {k: v for k, v in out.items() if isinstance(v, (int, float, bool))}


def bh(pvals: list) -> list:
    """Benjamini-Hochberg q-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, pvals[i] * m / (m - rank + 1))
        q[i] = prev
    return q


def auc(y: np.ndarray, x: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties handled by rank averages)."""
    pos, neg = x[y == 1], x[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def predict(records: dict, golds: dict, layer: str, mode: str, view: str) -> dict:
    mark, dav = golds["Mark"], golds["Davnah"]
    keys, target = [], []
    for key in sorted(set(mark) & set(dav) & set(records)):
        if layer == "entities":
            sizes = [len(RA.entity_spans(p)) for p in (mark[key], dav[key])]
        else:
            sizes = [len(RI.sentence_relations(p, view)[0]) for p in (mark[key], dav[key])]
        if not all(sizes):          # both humans must have worked this layer
            continue
        keys.append(key)
        target.append(1.0 - sentence_scores(mark[key], dav[key], layer, mode, view))
    y = np.array(target)
    hard = (y >= np.median(y)).astype(int)

    feats = [signals(records[k]) for k in keys]
    names = sorted(set().union(*[set(f) for f in feats])) if feats else []
    rows = []
    for name in names:
        x = np.array([float(f.get(name, 0.0)) for f in feats])
        if np.all(x == x[0]):
            continue
        rho, p = stats.spearmanr(x, y)
        rows.append({"signal": name, "rho": float(rho), "p": float(p),
                     "auc": auc(hard, x), "n": len(y)})
    for row, q in zip(rows, bh([r["p"] for r in rows])):
        row["q"] = q
    rows.sort(key=lambda r: -abs(r["rho"]))
    return {"n": len(y), "target_mean": float(y.mean()),
            "target_zero_frac": float((y == 0).mean()), "signals": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="multi_pass_deepseek_kimi_kimi_1")
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--label-view", choices=RI.LABEL_VIEWS, default="migration")
    ap.add_argument("--mode", choices=RI.ARG_MODES, default="boundary")
    ap.add_argument("--layer", choices=LAYERS + ("all",), default="all")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    path = args.run_dir / args.run
    agent = RA.load_agent(path)
    records = {}
    with path.open(encoding="utf8") as f:
        for line in f:
            if line.strip().startswith("{"):
                rec = json.loads(line)
                records[" ".join((rec.get("sentence") or "").lower().split())] = rec
    golds = {n: RA.load_gold(p) for n, p in GOLD.items()}
    layers = LAYERS if args.layer == "all" else (args.layer,)

    print(f"\n{'=' * 78}")
    print(f"  PARITY AND DISAGREEMENT PREDICTION — {args.run}")
    print(f"{'=' * 78}")
    print(f"  label view: {args.label_view}   argument match: {args.mode}")

    out = {"run": args.run, "label_view": args.label_view, "mode": args.mode,
           "parity": {}, "prediction": {}}

    print("\n  PARITY — per-sentence, agent = mean(vs Mark, vs Davnah), human = Mark vs Davnah")
    print(f"    {'layer':<11}{'n':>5}{'agent':>8}{'human':>8}{'gap':>8}"
          f"{'95% CI':>18}{'t p':>8}{'wilcox':>8}{'TOST.05':>9}{'TOST.03':>9}")
    for layer in layers:
        r = parity(agent, golds, layer, args.mode, args.label_view)
        out["parity"][layer] = r
        ci = "[%+.3f, %+.3f]" % (r["ci"][0], r["ci"][1])
        print(f"    {layer:<11}{r['n']:>5}{r['agent_mean']:>8.3f}{r['human_mean']:>8.3f}"
              f"{r['mean']:>+8.3f}{ci:>18}"
              f"{r['t_p']:>8.3f}{r['wilcoxon_p']:>8.3f}"
              f"{r['tost_0.05']:>9.3f}{r['tost_0.03']:>9.3f}")
    print("    TOST p < 0.05 affirms equivalence within that margin; a large t p alone does not.")

    for layer in layers:
        pr = predict(records, golds, layer, args.mode, args.label_view)
        out["prediction"][layer] = pr
        print(f"\n  PREDICTING HUMAN DISAGREEMENT — {layer} "
              f"(target = 1 − Mark↔Davnah F1, n={pr['n']}, "
              f"mean {pr['target_mean']:.3f}, perfect-agreement sentences "
              f"{pr['target_zero_frac']:.0%})")
        print(f"    {'signal':<34}{'rho':>8}{'p':>8}{'BH q':>8}{'AUC':>8}")
        for row in pr["signals"][:12]:
            star = "  *" if row["q"] < 0.05 else ""
            print(f"    {row['signal']:<34}{row['rho']:>+8.3f}{row['p']:>8.3f}"
                  f"{row['q']:>8.3f}{row['auc']:>8.3f}{star}")
        hits = [r for r in pr["signals"] if r["q"] < 0.05]
        print(f"    signals surviving BH at q<0.05: {len(hits)}"
              + (f"  ({', '.join(r['signal'] for r in hits)})" if hits else ""))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf8")
        print(f"\n  JSON written to {args.output}")
    print()


if __name__ == "__main__":
    main()
