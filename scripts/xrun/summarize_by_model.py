#!/usr/bin/env python
"""
Per-MODEL summary of the clean grid — one row per model, not per run.

Replicates are aggregated the way the paper aggregates them, which is not the
same as averaging the per-run numbers:

  RQ1     each sentence's agent score is averaged across the model's runs and
          the paired test is run on those averages (xrun_analysis.parity_cell),
          so n stays the number of sentences rather than sentences x runs.
  RQ2     correlations are reported as a RANGE with a count of how many runs
          reach significance. Averaging rho would hide the replicate spread,
          which on this data reaches 0.098 — wider than the gaps between models.
  counts  means, with the range alongside.

p-values are never averaged.

    python scripts/xrun/summarize_by_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import xrun_analysis as X          # noqa: E402
from clean_grid import CLEAN       # noqa: E402

REPORTS = ROOT / "output" / "eval_reports_clean"
MODELS = ["Qwen3.6-35B", "DeepSeek", "Kimi-2.7", "GPT-OSS"]


def rep(run: str, stem: str):
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
        return "-"
    if len(v) == 1:
        return spec.format(v[0])
    return f"{spec.format(min(v))} to {spec.format(max(v))}"


def main():
    multi, single = cells("multi"), cells("single")

    print("=" * 92)
    print("PER-MODEL SUMMARY — clean grid")
    print("=" * 92)

    print("\nRQ1 — distance from the two-human ceiling (0.497), full pipeline")
    print("cell-level paired test on per-sentence scores averaged across replicates")
    print(f"  {'Model':14s} {'R':>2s} {'agent':>7s} {'gap':>8s} {'95% CI':>20s} "
          f"{'p':>7s} {'spread':>7s}")
    for m in MODELS:
        if m not in multi:
            continue
        r = X.parity_cell(multi[m], canon=True, empties="zero")
        print(f"  {m:14s} {r['n_runs']:2d} {r['agent_mean']:7.3f} {r['mean']:+8.3f} "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {r['t_p']:7.3f} "
              f"{r.get('run_spread', 0.0):7.3f}")

    print("\nDELIBERATION — does the pipeline beat its own Annotator alone?")
    print(f"  {'Model':14s} {'R multi':>8s} {'R solo':>7s} {'full':>7s} {'solo':>7s} "
          f"{'diff':>8s} {'p':>7s}")
    for m in MODELS:
        if m not in multi or m not in single:
            print(f"  {m:14s} {len(multi.get(m, [])):8d} {len(single.get(m, [])):7d} "
                  f"{'-':>7s} {'-':>7s} {'-':>8s} {'-':>7s}   no solo arm")
            continue
        sm = [X.load(p, canon=True, empties="zero") for p in multi[m]]
        ss = [X.load(p, canon=True, empties="zero") for p in single[m]]
        keys = set(sm[0])
        for st in sm[1:] + ss:
            keys &= set(st)
        f = lambda st, k: (X.f1(st[k][0], st[k][1]) + X.f1(st[k][0], st[k][2])) / 2
        mv = [np.mean([f(st, k) for st in sm]) for k in sorted(keys)]
        sv = [np.mean([f(st, k) for st in ss]) for k in sorted(keys)]
        d = np.array(mv) - np.array(sv)
        p = float(stats.ttest_1samp(d, 0.0).pvalue)
        print(f"  {m:14s} {len(sm):8d} {len(ss):7d} {np.mean(mv):7.3f} "
              f"{np.mean(sv):7.3f} {d.mean():+8.3f} {p:7.3f}")

    print("\nRQ2 — span-level type logprob vs human type-disagreement")
    print("range across replicates, with the count reaching p<0.05")
    print(f"  {'Model':14s} {'arm':6s} {'R':>2s} {'rho range':>24s} {'sig':>6s}")
    for arm, group in (("multi", multi), ("solo", single)):
        for m in MODELS:
            if m not in group:
                continue
            rhos, sig = [], 0
            for path in group[m]:
                d = rep(path.name, "logprob")
                rho = g(d, "span_level", "logprob", "point_biserial_rho")
                pv = g(d, "span_level", "logprob", "point_biserial_p")
                if rho is not None:
                    rhos.append(rho)
                if isinstance(pv, float) and pv < 0.05:
                    sig += 1
            print(f"  {m:14s} {arm:6s} {len(group[m]):2d} {rng(rhos):>24s} "
                  f"{sig:>3d}/{len(group[m])}")

    print("\nDELIBERATION VOLUME — how much arguing actually happened (per run)")
    print(f"  {'Model':14s} {'R':>2s} {'critic disputes':>17s} {'adj resolutions':>17s} "
          f"{'sentences adj':>14s}")
    for m in MODELS:
        if m not in multi:
            continue
        cd = [g(rep(p.name, "layer34"), "per_step", "total_critic_disagreements")
              for p in multi[m]]
        ar = [g(rep(p.name, "layer34"), "per_step", "adjudicator_resolutions")
              for p in multi[m]]
        sa = [g(rep(p.name, "layer34"), "per_step", "sentences_adjudicated")
              for p in multi[m]]
        print(f"  {m:14s} {len(multi[m]):2d} {rng(cd, '{:.0f}'):>17s} "
              f"{rng(ar, '{:.0f}'):>17s} {rng(sa, '{:.0f}'):>14s}")

    print("\nGUIDELINE ADHERENCE and DIFFICULTY MODEL")
    print(f"  {'Model':14s} {'grounded':>18s} {'conf-only model':>20s} "
          f"{'all 25 features':>20s}")
    for m in MODELS:
        if m not in multi:
            continue
        gr = [g(rep(p.name, "guideline"), "grounding_match_rate") for p in multi[m]]
        b = [g(rep(p.name, "diffmodel"), "models", "baseline_conf_mean",
               "spearman", "mean") for p in multi[m]]
        a = [g(rep(p.name, "diffmodel"), "models", "RidgeCV (all feats)",
               "spearman", "mean") for p in multi[m]]
        print(f"  {m:14s} {rng(gr, '{:.3f}'):>18s} {rng(b):>20s} {rng(a):>20s}")

    print("\n  'conf-only' is the Annotator's mean self-reported confidence, one")
    print("  feature. 'all 25' adds every deliberation signal. Compare the two.")

    deliberation_history(multi)


def deliberation_history(multi):
    """Everything eval_deliberation_history reports, aggregated per model."""
    import sys as _s
    from collections import Counter
    import dispute_coverage as DC
    import eval_layer2_correlation as _L2
    from deliberation_history import load_records, norm, reconstruct_timeline

    human = _L2.load_all_human_annotations([X.MARK, X.DAV])
    hn = {norm(t): a for t, a in human.items()}

    print("\n" + "=" * 92)
    print("DELIBERATION HISTORY, PER MODEL")
    print("=" * 92)

    print("\nDoes the revision loop actually run?")
    print(f"  {'Model':14s} {'R':>2s} {'r1 disputes followed':>21s} {'resolved':>10s}")
    for m in MODELS:
        if m not in multi:
            continue
        fol = [g(rep(p_.name, "history"), "convergence_trajectory",
                 "round1_disputes_followed") for p_ in multi[m]]
        res = [g(rep(p_.name, "history"), "convergence_trajectory",
                 "resolved_by_round2") for p_ in multi[m]]
        print(f"  {m:14s} {len(multi[m]):2d} {rng(fol, '{:.0f}'):>21s} "
              f"{rng(res, '{:.0f}'):>10s}")
    print("  0 means the Critic objected and the Annotator never got a second pass:")
    print("  the pipeline ran Annotator -> Critic -> Adjudicator, with no revision.")

    print("\nSeverity: is the Critic's own grading calibrated?")
    print("  human type-disagreement rate among spans it graded major vs minor")
    print(f"  {'Model':14s} {'% major':>8s} {'major rate':>22s} {'minor rate':>22s} {'dir':>5s}")
    for m in MODELS:
        if m not in multi:
            continue
        maj_r, min_r, shares, dirs = [], [], [], []
        for p_ in multi[m]:
            sev_ct = Counter()
            buckets = {"major": [], "minor": []}
            for rec in load_records(p_):
                anns = hn.get(norm(rec.get("sentence", "")))
                if not anns or len(anns) < 2:
                    continue
                sev = {}
                for cr in reconstruct_timeline(rec)["critic_rounds"]:
                    for d in cr["disagreements"]:
                        lab = str(d.get("severity") or "").strip().lower() or "none"
                        sev_ct[lab] += 1
                        for k in DC.implicated_spans(d.get("target")):
                            sev.setdefault(k, set()).add(lab)
                for key, info in _L2.compute_human_span_disagreements(anns).items():
                    if key not in sev:
                        continue
                    y = 1 if info["type_disagreed"] else 0
                    if "major" in sev[key]:
                        buckets["major"].append(y)
                    elif "minor" in sev[key]:
                        buckets["minor"].append(y)
            tot = sum(sev_ct.values()) or 1
            shares.append(sev_ct.get("major", 0) / tot)
            a = sum(buckets["major"]) / len(buckets["major"]) if buckets["major"] else None
            b = sum(buckets["minor"]) / len(buckets["minor"]) if buckets["minor"] else None
            maj_r.append(a); min_r.append(b)
            if a is not None and b is not None:
                dirs.append("+" if a > b else "-")
        consistent = "same" if len(set(dirs)) == 1 else "FLIPS"
        print(f"  {m:14s} {rng(shares, '{:.0%}'):>8s} {rng(maj_r, '{:.3f}'):>22s} "
              f"{rng(min_r, '{:.3f}'):>22s} {consistent:>5s}")
    print("  'dir' = whether major outranks minor in every replicate of that model.")

    print("\nMissed-span claims the Critic filed")
    print(f"  {'Model':14s} {'total':>6s}  top types")
    for m in MODELS:
        if m not in multi:
            continue
        c = Counter()
        for p_ in multi[m]:
            c.update(g(rep(p_.name, "history"), "missing_entity_types") or {})
        top = ", ".join(f"{k} {v}" for k, v in c.most_common(4))
        print(f"  {m:14s} {sum(c.values()):6d}  {top}")


if __name__ == "__main__":
    main()
