#!/usr/bin/env python
"""
RQ1 and RQ2 recomputed on the clean grid (data/auto_annotated_clean).

These runs replace the old heterogeneous set: one configuration throughout —
guideline v4 (09db658b...), no dependency hints, no relation schema in the
prompt, input 863216e1... — with machine-written provenance in every record,
so nothing here rests on a file name.

They also carry the deliberation ablation the paper never had: three models run
BOTH as the full Annotator/Critic/Adjudicator pipeline and as the Annotator
alone. The single-pass arm uses the prompt written for it ("solo"), so the
contrast is the full pipeline against a properly prompted single annotator
rather than against a crippled one — the comparison a reviewer will ask for.

    python scripts/xrun/clean_grid.py            RQ1, ablation, RQ2
    python scripts/xrun/clean_grid.py --rq1
    python scripts/xrun/clean_grid.py --rq2
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

import xrun_analysis as X                     # noqa: E402
import eval_layer2_correlation as L2          # noqa: E402
import pipeline_signals as PS                 # noqa: E402
from deliberation_history import load_records, norm  # noqa: E402
from eval_utils import permutation_p_rho      # noqa: E402

D = ROOT / "data" / "auto_annotated_clean"

# (model, arm, path). The duplicate files — single_pass_kimi27_no_deliberation
# and ..._1, single_pass_qwen35B_no_deliberation and ..._1 — are byte-identical
# (md5), so only one of each is listed: counting them twice would fake a
# replicate and shrink the standard errors.
CLEAN = [
    ("Qwen3.6-35B", "multi",  D / "multi_pass_qwen3-35B-vllm_1"),
    ("Qwen3.6-35B", "multi",  D / "multi_pass_qwen3-35B-vllm_2"),
    ("Qwen3.6-35B", "multi",  D / "multi_pass_qwen3-35B-vllm_3"),
    ("DeepSeek",    "multi",  D / "multi_pass_deepseek_1"),
    ("DeepSeek",    "multi",  D / "multi_pass_deepseek_2"),
    ("DeepSeek",    "multi",  D / "multi_pass_deepseek_3"),
    ("Kimi-2.7",    "multi",  D / "multi_pass_rcp-kimi-2.7_1"),
    ("Kimi-2.7",    "multi",  D / "multi_pass_rcp-kimi-2.7_2"),
    ("GPT-OSS",     "multi",  D / "multi_pass_gpt-oss_1"),
    ("GPT-OSS",     "multi",  D / "multi_pass_gpt-oss_2"),
    ("Qwen3.6-35B", "single", D / "single_pass_qwen35B_no_deliberation"),
    ("DeepSeek",    "single", D / "single_pass_deepseek_no_deliberation_1"),
    # NOTE the "+" — the file really is named single_pass+deepseek_..., not
    # single_pass_deepseek_.... Any glob on "single_pass_*" silently misses it.
    ("DeepSeek",    "single", D / "single_pass+deepseek_no_deliberation_2"),
    ("Kimi-2.7",    "single", D / "single_pass_kimi27_no_deliberation"),
    ("GPT-OSS",     "single", D / "single_pass_gpt-oss_no_deliberation_1"),
]

CANON, EMPTIES = True, "zero"


def cells(arm):
    out = {}
    for model, a, path in CLEAN:
        if a == arm and path.exists():
            out.setdefault(model, []).append(path)
    return out


def rq1():
    print("=" * 78)
    print("RQ1 — distance from the two-human ceiling, clean grid")
    print("paired per sentence; 0 = ceiling; empties scored F1=0; labels canonicalised")
    print("=" * 78)
    for arm, label in (("multi", "FULL PIPELINE (Annotator + Critic + Adjudicator)"),
                       ("single", "ANNOTATOR ONLY (no deliberation)")):
        print(f"\n{label}")
        print(f"  {'Model':16s} {'R':>2s} {'n':>4s} {'agent':>7s} {'gap':>8s} "
              f"{'95% CI':>20s} {'p':>7s} {'TOST.03':>8s}")
        for model, paths in sorted(cells(arm).items()):
            r = X.parity_cell(paths, canon=CANON, empties=EMPTIES)
            print(f"  {model:16s} {r['n_runs']:2d} {r['n']:4d} {r['agent_mean']:7.3f} "
                  f"{r['mean']:+8.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] "
                  f"{r['t_p']:7.3f} {r['tost03']:8.3f}")
        if arm == "multi":
            r = X.parity_cell(list(cells("multi").values())[0], canon=CANON, empties=EMPTIES)
            print(f"  ceiling (mean per-sentence human F1) = {r['human_mean']:.3f}")


def ablation():
    """Does deliberation help? Paired, same sentences, same model."""
    print("\n" + "=" * 78)
    print("DELIBERATION ABLATION — full pipeline minus annotator-only")
    print("paired per sentence on the sentences BOTH arms scored")
    print("=" * 78)
    m, s = cells("multi"), cells("single")
    shared = sorted(set(m) & set(s))
    print(f"\n  {'Model':16s} {'n':>4s} {'multi':>7s} {'single':>7s} {'diff':>8s} "
          f"{'95% CI':>20s} {'p':>7s}")
    for model in shared:
        stores_m = [X.load(p, canon=CANON, empties=EMPTIES) for p in m[model]]
        stores_s = [X.load(p, canon=CANON, empties=EMPTIES) for p in s[model]]
        keys = set(stores_m[0])
        for st in stores_m[1:] + stores_s:
            keys &= set(st)
        keys = sorted(keys)
        mv, sv = [], []
        for k in keys:
            mv.append(np.mean([(X.f1(st[k][0], st[k][1]) + X.f1(st[k][0], st[k][2])) / 2
                               for st in stores_m]))
            sv.append(np.mean([(X.f1(st[k][0], st[k][1]) + X.f1(st[k][0], st[k][2])) / 2
                               for st in stores_s]))
        d = np.array(mv) - np.array(sv)
        se = float(stats.sem(d))
        crit = stats.t.ppf(0.975, len(d) - 1)
        p = float(stats.ttest_1samp(d, 0.0).pvalue)
        print(f"  {model:16s} {len(d):4d} {np.mean(mv):7.3f} {np.mean(sv):7.3f} "
              f"{d.mean():+8.3f} [{d.mean()-crit*se:+.3f}, {d.mean()+crit*se:+.3f}] {p:7.3f}")
    print("\n  positive = deliberation helps. The paper claims deliberation as its")
    print("  method; this is the comparison that tests it.")


def build_rows_deep(path, hnorm):
    """PS.build_rows plus the Critic's and Adjudicator's OWN token probabilities.

    PS's `critic`/`adjudicated` columns are binary event flags — did the Critic
    dispute this span, did the Adjudicator touch it. These are the graded
    versions: the logprob each agent carried on its own type decision for the
    same span, newly captured. Matched on normalised span text, the same key the
    human disagreement target uses.
    """
    rows = PS.build_rows(path, hnorm)
    by_key = {}
    for rec in load_records(path):
        anns = hnorm.get(norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        for e in (rec.get("critic_entity_logprobs") or []):
            k = L2._normalize(e.get("text", ""))
            # A span the Critic ruled on more than once (agreement in one round,
            # dispute in another) keeps its most uncertain reading rather than
            # whichever happened to come last.
            lp = e.get("type_mean_logprob")
            if lp is not None:
                cur = by_key.setdefault(k, {})
                cur["critic_lp"] = min(cur.get("critic_lp", 0.0), lp)
        for e in (rec.get("adjudicator_entity_logprobs") or []):
            k = L2._normalize(e.get("text", ""))
            lp = e.get("type_mean_logprob")
            if lp is not None:
                by_key.setdefault(k, {})["adj_lp"] = lp
    # rows carry no key, so rebuild the join in the same order build_rows used
    i = 0
    for rec in load_records(path):
        anns = hnorm.get(norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        dis = L2.compute_human_span_disagreements(anns)
        for e in (rec.get("annotator_entity_logprobs") or []):
            k = L2._normalize(e.get("text", ""))
            if dis.get(k) is None:
                continue
            extra = by_key.get(k, {})
            rows[i]["critic_lp"] = extra.get("critic_lp")
            rows[i]["adj_lp"] = extra.get("adj_lp")
            i += 1
    return rows


def rq2_deep():
    """The graded Critic/Adjudicator signals, against the Annotator baseline."""
    print("\n" + "=" * 78)
    print("RQ2b — whose token probabilities carry the signal?")
    print("Annotator vs Critic vs Adjudicator logprob, same spans, same target")
    print("=" * 78)
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): a for s, a in human.items()}
    print(f"\n  {'run':34s} {'n':>5s} {'ann':>10s} {'critic':>10s} {'adjud':>10s}"
          f"  {'cov crit':>9s} {'cov adj':>8s}")
    for model, arm, path in CLEAN:
        if not path.exists() or arm != "multi":
            continue
        rows = build_rows_deep(path, hnorm)
        base = [r for r in rows if r["logprob"] is not None]
        line = f"  {path.name[:34]:34s} {len(base):5d}"
        for key in ("logprob", "critic_lp", "adj_lp"):
            r = PS.score(rows, key, False, restrict_to="logprob")
            if r is None or r["rho"] is None:
                line += f"{'-':>10s}"
                continue
            star = "*" if (r["p"] is not None and r["p"] < 0.05) else " "
            line += f"{r['rho']:>9.3f}{star}"
        cc = sum(1 for r in base if r.get("critic_lp") is not None) / max(len(base), 1)
        ca = sum(1 for r in base if r.get("adj_lp") is not None) / max(len(base), 1)
        print(line + f"  {cc:9.0%} {ca:8.0%}")
    print("\n  cov = share of the Annotator's scored spans on which that agent also")
    print("  made a recorded type decision. A low coverage makes the column a")
    print("  different sample, not a fair comparison.")


def rq2():
    print("\n" + "=" * 78)
    print("RQ2 — span-level uncertainty vs human type-disagreement, clean grid")
    print("rho = point-biserial Spearman on spans a human also marked; * = perm p<0.05")
    print("=" * 78)
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): a for s, a in human.items()}
    sig = [("logprob", False), ("entropy", True), ("selfconf", True),
           ("critic", True), ("adjudicated", True)]
    print(f"\n  {'run':34s} {'arm':7s} {'spans':>6s}" +
          "".join(f"{k[:9]:>11s}" for k, _h in sig))
    for model, arm, path in CLEAN:
        if not path.exists():
            continue
        rows = PS.build_rows(path, hnorm)
        line = f"  {path.name[:34]:34s} {arm:7s} {len(rows):6d}"
        for key, hi in sig:
            r = PS.score(rows, key, hi, restrict_to="logprob")
            if r is None or r["rho"] is None:
                line += f"{'-':>11s}"
                continue
            star = "*" if (r["p"] is not None and r["p"] < 0.05) else " "
            line += f"{r['rho']:>10.3f}{star}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1", action="store_true")
    ap.add_argument("--rq2", action="store_true")
    a = ap.parse_args()
    both = not (a.rq1 or a.rq2)
    if a.rq1 or both:
        rq1(); ablation()
    if a.rq2 or both:
        rq2(); rq2_deep()


if __name__ == "__main__":
    main()
