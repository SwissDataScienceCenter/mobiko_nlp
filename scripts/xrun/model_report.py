#!/usr/bin/env python
"""
Everything we measured, organised per MODEL.

The per-run and pooled views both hid a pattern the per-model view makes
obvious: the pipeline pays off on the model that actually uses it. Kimi argues
three times as much as Qwen, is the only model whose Annotator gets a second
pass, the only one whose severity grading points the right way in every
replicate, and the only one where deliberation significantly improves F1.
Pooling those four models into one verdict was the wrong summary.

Sources, per model, replicates shown as a range:
  layer1/parity_cell  quality against the two-human ceiling
  ablation            full pipeline vs the same model's Annotator alone
  layer34 + history   rounds, disputes, revisions, severity, missed spans
  guideline           whether cited rules are real
  logprob             span-level token uncertainty
  computed            self-confidence, Critic/Adjudicator logprobs, dispute routing
  diffmodel           sentence-level difficulty models

    python scripts/xrun/model_report.py
    python scripts/xrun/model_report.py --model Kimi-2.7
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import dispute_coverage as DC              # noqa: E402
import eval_layer2_correlation as L2       # noqa: E402
import pipeline_signals as PS              # noqa: E402
import xrun_analysis as X                  # noqa: E402
from clean_grid import CLEAN, build_rows_deep  # noqa: E402
from deliberation_history import (         # noqa: E402
    load_records, norm, record_signals, reconstruct_timeline,
)

REPORTS = ROOT / "output" / "eval_reports_clean"
MODELS = ["Qwen3.6-35B", "DeepSeek", "Kimi-2.7", "GPT-OSS"]
CEILING = 0.497


def rep(run, stem):
    p = REPORTS / run / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def g(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def cells(arm):
    out = {}
    for model, a, path in CLEAN:
        if a == arm and path.exists():
            out.setdefault(model, []).append(path)
    return out


def rng(vals, spec="{:+.3f}"):
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return "n/a"
    return spec.format(v[0]) if len(v) == 1 else \
        f"{spec.format(min(v))} .. {spec.format(max(v))}"


def rho_p(xs, ys):
    """Spearman with its asymptotic p. Permutation is unnecessary at n~800 and
    would make this report take an hour."""
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, 1.0
    r = stats.spearmanr(xs, ys)
    return float(r.statistic), float(r.pvalue)


def span_signals(path, hnorm):
    """rho and significance for every per-span signal on one run."""
    rows = build_rows_deep(path, hnorm)
    base = [r for r in rows if r["logprob"] is not None]
    out = {}
    for key, hi in (("logprob", False), ("entropy", True), ("selfconf", True),
                    ("critic_lp", False), ("adj_lp", False)):
        use = [r for r in base if r.get(key) is not None]
        if len(use) < 30:
            out[key] = (None, 1.0)
            continue
        xs = [(r[key] if hi else -r[key]) for r in use]
        out[key] = rho_p(xs, [r["target"] for r in use])
    return out, len(base)


def routing(path, hnorm):
    """Precision / recall / effort for the Critic's disputes as a routing gate."""
    narrow = {"flag": 0, "hit": 0}
    wide = {"flag": 0, "hit": 0}
    tot = dis = 0
    for rec in load_records(path):
        anns = hnorm.get(norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        _lit, ends, ent = DC.collect(rec)
        for key, info in L2.compute_human_span_disagreements(anns).items():
            y = 1 if info["type_disagreed"] else 0
            tot += 1
            dis += y
            for bucket, flags in ((narrow, ent), (wide, ends)):
                if key in flags:
                    bucket["flag"] += 1
                    bucket["hit"] += y
    def stat(b):
        return (b["hit"] / b["flag"] if b["flag"] else 0.0,
                b["hit"] / dis if dis else 0.0,
                b["flag"] / tot if tot else 0.0)
    return stat(narrow), stat(wide), (dis / tot if tot else 0.0)


def severity(path, hnorm):
    """(share graded major, major hit rate, minor hit rate)."""
    ct = Counter()
    buckets = {"major": [], "minor": []}
    for rec in load_records(path):
        anns = hnorm.get(norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        sev = {}
        for cr in reconstruct_timeline(rec)["critic_rounds"]:
            for d in cr["disagreements"]:
                lab = str(d.get("severity") or "").strip().lower() or "none"
                ct[lab] += 1
                for k in DC.implicated_spans(d.get("target")):
                    sev.setdefault(k, set()).add(lab)
        for key, info in L2.compute_human_span_disagreements(anns).items():
            if key not in sev:
                continue
            y = 1 if info["type_disagreed"] else 0
            if "major" in sev[key]:
                buckets["major"].append(y)
            elif "minor" in sev[key]:
                buckets["minor"].append(y)
    tot = sum(ct.values()) or 1
    m = lambda v: (sum(v) / len(v)) if v else None
    return ct.get("major", 0) / tot, m(buckets["major"]), m(buckets["minor"])


def rounds_hist(path):
    c = Counter(record_signals(r).get("rounds_used") for r in load_records(path))
    return dict(sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0])))


def flagged_rate(path):
    recs = load_records(path)
    return sum(1 for r in recs if r.get("flagged_for_human_review")) / max(len(recs), 1)


def report(model, multi, single, hnorm):
    runs = multi.get(model, [])
    solo = single.get(model, [])
    print("\n" + "=" * 88)
    print(f"  {model}    full pipeline R={len(runs)}    annotator-only R={len(solo)}")
    print("=" * 88)

    # ── quality ─────────────────────────────────────────────────────────────
    r = X.parity_cell(runs, canon=True, empties="zero")
    print(f"\n  QUALITY (ceiling {CEILING:.3f})")
    print(f"    agent F1              {r['agent_mean']:.3f}")
    print(f"    gap                   {r['mean']:+.3f}  "
          f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  p={r['t_p']:.3f}")
    print(f"    replicate spread      {r.get('run_spread', 0.0):.3f}")
    if solo:
        rs = X.parity_cell(solo, canon=True, empties="zero")
        sm = [X.load(p, canon=True, empties="zero") for p in runs]
        ss = [X.load(p, canon=True, empties="zero") for p in solo]
        keys = set(sm[0])
        for st in sm[1:] + ss:
            keys &= set(st)
        f = lambda st, k: (X.f1(st[k][0], st[k][1]) + X.f1(st[k][0], st[k][2])) / 2
        d = np.array([np.mean([f(st, k) for st in sm]) for k in sorted(keys)]) - \
            np.array([np.mean([f(st, k) for st in ss]) for k in sorted(keys)])
        p = float(stats.ttest_1samp(d, 0.0).pvalue)
        print(f"    annotator alone       {rs['agent_mean']:.3f}  (gap {rs['mean']:+.3f})")
        print(f"    DELIBERATION EFFECT   {d.mean():+.3f}  p={p:.3f}"
              f"{'   <- significant' if p < 0.05 else ''}")
    else:
        print("    annotator alone       no solo arm — cannot run the ablation")

    # ── process ─────────────────────────────────────────────────────────────
    print("\n  DELIBERATION PROCESS")
    print(f"    rounds used           " +
          "  ".join(f"{k}:{v}" for k, v in rounds_hist(runs[0]).items()) +
          "   (first replicate)")
    print(f"    critic disputes       "
          f"{rng([g(rep(p.name,'layer34'),'per_step','total_critic_disagreements') for p in runs], '{:.0f}')}")
    print(f"    adjudicator resolves  "
          f"{rng([g(rep(p.name,'layer34'),'per_step','adjudicator_resolutions') for p in runs], '{:.0f}')}")
    fol = [g(rep(p.name, "history"), "convergence_trajectory", "round1_disputes_followed")
           for p in runs]
    print(f"    REVISION LOOP         {rng(fol, '{:.0f}')} disputes reached a second pass"
          f"{'   <- loop never runs' if all(x == 0 for x in fol if x is not None) else ''}")
    print(f"    flagged for human     {rng([flagged_rate(p) for p in runs], '{:.0%}')}")
    sev = [severity(p, hnorm) for p in runs]
    dirs = {"+" if (a is not None and b is not None and a > b) else "-"
            for _s, a, b in sev}
    print(f"    graded major          {rng([s[0] for s in sev], '{:.0%}')}")
    print(f"    major vs minor rate   {rng([s[1] for s in sev], '{:.3f}')} vs "
          f"{rng([s[2] for s in sev], '{:.3f}')}   "
          f"{'consistent' if len(dirs) == 1 and '+' in dirs else 'FLIPS across replicates'}")
    miss = Counter()
    for p in runs:
        miss.update(g(rep(p.name, "history"), "missing_entity_types") or {})
    print(f"    missed-span claims    {sum(miss.values())}   top: "
          + ", ".join(f"{k} {v}" for k, v in miss.most_common(3)))
    print(f"    guideline grounding   "
          f"{rng([g(rep(p.name,'guideline'),'grounding_match_rate') for p in runs], '{:.3f}')}")

    # ── span-level signals ──────────────────────────────────────────────────
    print("\n  UNCERTAINTY, PER SPAN   (rho vs human type-disagreement, k/R at p<0.05)")
    sig = [span_signals(p, hnorm) for p in runs]
    for key, lbl in (("logprob", "annotator logprob"), ("entropy", "annotator entropy"),
                     ("selfconf", "self-confidence"), ("critic_lp", "critic logprob"),
                     ("adj_lp", "adjudicator logprob")):
        vals = [s[0][key][0] for s in sig]
        k = sum(1 for s in sig if s[0][key][1] < 0.05)
        print(f"    {lbl:22s}{rng(vals):>20s}   {k}/{len(runs)}")
    if solo:
        ssig = [span_signals(p, hnorm) for p in solo]
        print(f"    {'(solo) logprob':22s}"
              f"{rng([s[0]['logprob'][0] for s in ssig]):>20s}   "
              f"{sum(1 for s in ssig if s[0]['logprob'][1] < 0.05)}/{len(solo)}")

    # ── routing ─────────────────────────────────────────────────────────────
    print("\n  DISPUTES AS A ROUTING GATE")
    rt = [routing(p, hnorm) for p in runs]
    base = rt[0][2]
    print(f"    base disagreement     {base:.3f}")
    for i, lbl in ((0, "entity disputes"), (1, "all disputes")):
        print(f"    {lbl:22s}precision {rng([x[i][0] for x in rt], '{:.3f}')}   "
              f"recall {rng([x[i][1] for x in rt], '{:.1%}')}   "
              f"effort {rng([x[i][2] for x in rt], '{:.1%}')}")

    # ── sentence level ──────────────────────────────────────────────────────
    print("\n  UNCERTAINTY, PER SENTENCE")
    print(f"    confidence only       "
          f"{rng([g(rep(p.name,'diffmodel'),'models','baseline_conf_mean','spearman','mean') for p in runs])}")
    print(f"    all 25 features       "
          f"{rng([g(rep(p.name,'diffmodel'),'models','RidgeCV (all feats)','spearman','mean') for p in runs])}")
    for name in ("rounds_used", "flagged_for_review (0/1)"):
        vs = []
        for p in runs:
            for c in (g(rep(p.name, "history"), "difficulty_prediction", "correlations") or []):
                if c.get("signal") == name:
                    vs.append(c.get("spearman_rho"))
        print(f"    {name:22s}{rng(vs):>20s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS)
    a = ap.parse_args()
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): v for s, v in human.items()}
    multi, single = cells("multi"), cells("single")
    for m in ([a.model] if a.model else MODELS):
        if m in multi:
            report(m, multi, single, hnorm)


if __name__ == "__main__":
    main()
