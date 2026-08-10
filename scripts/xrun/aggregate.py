#!/usr/bin/env python
"""Paper-ready AGGREGATED tables, one cell per (model x dependency-hint condition).

  aggregate.py            both tables as fixed-width text
  aggregate.py --latex    both tables as LaTeX (booktabs)

TO ADD A RUN: append one line to REGISTRY. Nothing else changes — cells with a
single run print no dispersion, cells with none print "not yet run".

AGGREGATION METHOD
  Runs within a cell are replicates on the SAME 154 gold sentences, so pooling
  their sentences would fake independence and shrink the SEs. Instead:
    RQ1  each sentence's agent score is AVERAGED ACROSS THE CELL'S RUNS, then
         tested paired against the Mark<->Davnah ceiling on that sentence.
         n stays the number of sentences. Between-run dispersion is reported
         separately as the spread of the per-run values (SD for R>2, half-range
         for R=2), so replicate noise is visible rather than absorbed.
    RQ2  rho is averaged across the cell's runs and reported with its range;
         significance is reported as "k/R runs surviving BH q<0.05 within that
         run's own 36-test family". Rho values are NOT pooled into one test —
         each run's test stays its own test.

  Primary numbers use CANONICALISED predicted labels (underscores -> spaces,
  whitespace collapsed). That measures typing content rather than a known
  emit-path formatting bug, and it is required for the dep contrast to be fair:
  the apertus dep arms shipped 72/57 malformed labels against the control's 4.
  The as-is column is printed alongside so nothing is hidden.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats

import xrun_analysis as X

ROOT = Path("/home/katinska/mobiko_nlp")
D = ROOT / "data"
XR = ROOT / "output/eval_reports/cross_run_comparison"

# ── (model, condition, report dir, source JSONL) ────────────────────────────
REGISTRY = [
    ("Qwen3.6-35B", "no-dep", "qwen36_35B",
     D / "auto_annotated/datademo_manually_labeled_swissai-qwen3-35B-vllm"),
    ("Qwen3.6-35B", "no-dep", "qwen36_35B_v3",
     D / "auto_annotated/datademo_manually_labeled_qwen3-35B-vllm_v3"),
    ("Qwen3.6-35B", "dep", "qwen36_35B_dep",
     D / "auto_annotated/datademo_manually_labeled_qwen3-35B-vllm_v1_dep"),

    # Apertus is split by PIPELINE BUILD, not just by condition. The 2026-08-04
    # no-dep rerun beats the 2026-07-20 no-dep run by +0.052 F1 (p=0.0014,
    # ~3.3 se) — far outside the 0.012-0.014 replicate noise floor. They are
    # therefore NOT replicates and must not share a cell. Keeping them apart is
    # what makes the eventual same-build dep contrast interpretable.
    ("Apertus70B Jul", "no-dep", "apertus70B_nodep",
     D / "auto_annotated/datademo_manually_labeled_swissai-apertus-70B.jsonl"),
    ("Apertus70B Jul", "dep", "apertus70B_dep",
     D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_dep_relation_schema"),
    ("Apertus70B Jul", "dep", "apertus70B_dep_v2",
     D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_dep_v2.dedup.jsonl"),

    # NOTE the variant: this run also has the RELATION SCHEMA DISABLED
    # (filename suffix _no_relation_schema), which is why it emits 55 free-form
    # relation types against the Jul runs' 7-12. So it differs from the Jul
    # no-dep run in at least TWO ways, build and relation-schema config. Its dep
    # counterpart must be run with the SAME no-relation-schema setting or the
    # contrast is confounded all over again.
    ("Apertus70B Aug-noRS", "no-dep", "apertus70B_nodep_v3",
     D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_v3_no_relation_schema"),
    # The same-build, same-no-relation-schema dep arm the rerun asked for. Added
    # 2026-08-04. This is the FIRST apertus dep contrast in which the control and
    # the treated run share a build and a relation-schema setting, so it is the
    # only one that estimates the hints and nothing else.
    # Raw file is …_v3_dep (169 lines, 155 unique sentences); a resume overlap
    # re-annotated 14 sentences in one contiguous block. Scored from the
    # keep-first .dedup.jsonl — same rule as dep_v2 — because load_agent_records
    # appends every line and would micro-count those 14 sentences twice.
    ("Apertus70B Aug-noRS", "dep", "apertus70B_dep_v3",
     D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_v3_dep.dedup.jsonl"),

    ("Kimi-2.7", "no-dep", "kimi27_v2",
     D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v2"),
    ("Kimi-2.7", "no-dep", "kimi27_v3",
     D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v3"),
    ("Kimi-2.7", "dep", "kimi27_dep",
     D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v1_dep"),

    # openai/gpt-oss-120b via RCP, added 2026-08-06. FIRST RUN WHOSE ARM IS
    # VERIFIABLE FROM THE DATA: its records carry run_meta with
    # use_dependency_relation_hints=true and include_relation_schema=true, so
    # this cell does not rest on a filename (caveat 7). Relation-schema ON is
    # independently confirmed by its 8 relation types. No no-dep arm exists yet,
    # so this model contributes no dep contrast.
    ("GPT-OSS-120B", "dep", "gptoss_dep",
     D / "auto_annotated_local/datademo_manually_labeled_rcp-gpt-oss_v1_dep"),

    ("Qwen3.5 (legacy)", "no-dep", "qwen35_v2",
     D / "auto_annotated/datademo_manually_labeled2.jsonl"),
]

MODELS = list(OrderedDict((m, None) for m, _, _, _ in REGISTRY))
CONDITIONS = ["no-dep", "dep"]

# short header codes for Table 2 — derived names would collide (two Apertus rows)
MODEL_SHORT = {
    "Qwen3.6-35B": "Qwen3.6", "Apertus70B Jul": "ApJul",
    "Apertus70B Aug-noRS": "ApAug", "Kimi-2.7": "Kimi",
    "GPT-OSS-120B": "GPToss", "Qwen3.5 (legacy)": "Qwen3.5",
}

RQ2_SIGNALS = [
    ("span type-logprob -> type-disagr", "span:logprob:point_biserial"),
    ("self-conf -> difficulty",          "lp:self-confidence->human_difficulty"),
    ("self-conf -> disagr rate",         "lp:self-confidence->human_disagree_rate"),
    ("logprob type-min -> difficulty",   "lp:logprob (type-min)->human_difficulty"),
    ("entity count -> typing diff",      "typ:agent entity count"),
    ("critic disagr -> human (OR)",      "layer2:fisher_or"),
]


def cell_runs(model, cond):
    return [(d, p) for m, c, d, p in REGISTRY if m == model and c == cond]


def spread(vals):
    """Dispersion across runs: SD for R>2, half-range for R=2, None for R<2."""
    if len(vals) < 2:
        return None
    if len(vals) == 2:
        return abs(vals[1] - vals[0]) / 2.0
    return float(np.std(vals, ddof=1))


# ── RQ1 ─────────────────────────────────────────────────────────────────────
def rq1_cell(model, cond, canon=True):
    runs = cell_runs(model, cond)
    if not runs:
        return None
    stores = [X.load(p, canon=canon) for _, p in runs]
    keys = sorted(set.intersection(*(set(s) for s in stores)))
    if not keys:
        return None

    agent, human, per_run = [], [], [[] for _ in stores]
    for k in keys:
        vals = []
        for i, s in enumerate(stores):
            a, mk, dv = s[k]
            v = (X.f1(a, mk) + X.f1(a, dv)) / 2.0
            vals.append(v)
            per_run[i].append(v)
        agent.append(float(np.mean(vals)))
        _, mk, dv = stores[0][k]
        human.append(X.f1(mk, dv))

    agent, human = np.array(agent), np.array(human)
    r = X.paired_block(agent - human)
    r.update(model=model, cond=cond, R=len(runs), agent_mean=float(agent.mean()),
             human_mean=float(human.mean()),
             run_means=[float(np.mean(v)) for v in per_run],
             run_dirs=[d for d, _ in runs])
    r["between_run"] = spread(r["run_means"])
    return r


def headline_cell(model, cond, canon=True):
    """Micro strict F1 and typing kappa, computed per run on the SAME label
    variant as the gap column, then averaged across the cell's runs.

    Recomputed here rather than read from the stored layer1.json because those
    are as-is only; mixing an as-is F1 with a canonicalised gap in one row would
    be incoherent. With canon=False this reproduces layer1.json exactly.
    """
    runs = cell_runs(model, cond)
    if not runs:
        return None
    acc = {k: [] for k in ("sMark", "sDav", "kMark", "kDav")}
    for _, p in runs:
        store = X.load(p, canon=canon)
        pairs_m = [(a, mk) for a, mk, _ in store.values()]
        pairs_d = [(a, dv) for a, _, dv in store.values()]
        acc["sMark"].append(X.L1.score_reference(pairs_m, "strict")["f1"])
        acc["sDav"].append(X.L1.score_reference(pairs_d, "strict")["f1"])
        acc["kMark"].append(X.L1.label_only_agreement(pairs_m)["cohen_kappa"])
        acc["kDav"].append(X.L1.label_only_agreement(pairs_d)["cohen_kappa"])
    return {k: (float(np.mean(v)), spread(v)) for k, v in acc.items()}


def dep_contrast(model, canon=True):
    """dep minus no-dep, paired per sentence on cell-averaged scores."""
    a, b = cell_runs(model, "no-dep"), cell_runs(model, "dep")
    if not a or not b:
        return None
    SA = [X.load(p, canon=canon) for _, p in a]
    SB = [X.load(p, canon=canon) for _, p in b]
    keys = sorted(set.intersection(*(set(s) for s in SA + SB)))
    if not keys:
        return None
    va, vb = [], []
    for k in keys:
        for stores, out in ((SA, va), (SB, vb)):
            vals = []
            for s in stores:
                ent, mk, dv = s[k]
                vals.append((X.f1(ent, mk) + X.f1(ent, dv)) / 2.0)
            out.append(float(np.mean(vals)))
    va, vb = np.array(va), np.array(vb)
    r = X.paired_block(vb - va)
    r.update(model=model, nodep_mean=float(va.mean()), dep_mean=float(vb.mean()))
    return r


# ── RQ2 ─────────────────────────────────────────────────────────────────────
def rq2_cell(model, cond):
    runs = cell_runs(model, cond)
    if not runs:
        return None
    out = {"R": len(runs), "signals": {}, "surv": [], "raw": [], "sat": []}
    per_run = []
    for d, _ in runs:
        tests = X.family(XR / d)
        surv, qv, raw, m = X.bh(tests)
        per_run.append(({l: (p, rho) for l, p, rho in tests}, qv))
        out["surv"].append(len(surv))
        out["raw"].append((raw, m))
        for line in (XR / d / "logprob.txt").read_text().splitlines():
            if "% of entities have logprob" in line:
                out["sat"].append(float(line.strip().split("%")[0]))
                break
    for label, key in RQ2_SIGNALS:
        rhos, nsig = [], 0
        for cells, qv in per_run:
            p, rho = cells.get(key, (None, None))
            if p is None or rho is None:
                continue
            rhos.append(rho)
            if qv.get(key, 1) < 0.05:
                nsig += 1
        out["signals"][label] = (rhos, nsig, len(per_run))
    return out


# ── rendering ───────────────────────────────────────────────────────────────
def fmt_pm(mean, disp, nd=3):
    if mean is None:
        return "--"
    s = f"{mean:.{nd}f}"
    return s if disp is None else f"{s}+-{disp:.{nd}f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "  n/a"
    return "<.001" if p < 0.001 else f"{p:.3f}"


def table_rq1_text():
    L = []
    L.append("  model                cond    R    n  strict F1 (M)  strict F1 (D)  "
             "typing k (M)   hyg     gap      95% CI          t p  TOST")
    for model in MODELS:
        for cond in CONDITIONS:
            r = rq1_cell(model, cond)
            if r is None:
                L.append(f"  {model:20s} {cond:6s}  --   --  "
                         f"{'---- not yet run ----':^45s}")
                continue
            h = headline_cell(model, cond, canon=True)
            hraw = headline_cell(model, cond, canon=False)
            hyg = h["sMark"][0] - hraw["sMark"][0]
            eq = "EQ" if r["tost05"] < 0.05 else "- "
            L.append(f"  {model:20s} {cond:6s} {r['R']:2d}  {r['n']:3d}  "
                     f"{fmt_pm(*h['sMark']):>13s}  {fmt_pm(*h['sDav']):>13s}  "
                     f"{fmt_pm(*h['kMark']):>13s}  {hyg:+.3f}  "
                     f"{r['mean']:+.3f}  [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                     f"{fmt_p(r['t_p']):>5s}  {eq}")
    L.append("  " + "-" * 120)
    L.append(f"  {'HUMAN CEILING':20s} {'':6s}  --   --  "
             f"{'0.478':>13s}  {'0.478':>13s}  {'0.781':>13s}"
             f"      --   0.000   (by definition)")
    return "\n".join(L)


def table_rq2_text():
    cells = {(m, c): rq2_cell(m, c) for m in MODELS for c in CONDITIONS}
    live = [(m, c) for m in MODELS for c in CONDITIONS if cells[(m, c)]]
    head = [f"{MODEL_SHORT.get(m, m)}/{'dep' if c == 'dep' else 'no'}"
            for m, c in live]
    W = 34
    L = ["  " + "signal".ljust(W) + "".join(h.rjust(13) for h in head)]
    for label, key in RQ2_SIGNALS:
        is_or = key == "layer2:fisher_or"
        line = "  " + label.ljust(W)
        for m, c in live:
            rhos, nsig, R = cells[(m, c)]["signals"][label]
            if not rhos:
                line += "           --"
                continue
            # odds ratios are multiplicative: average them geometrically, not
            # arithmetically, and print without a sign (1.00 = no association).
            if is_or:
                cell = f"{float(np.exp(np.mean(np.log(rhos)))):.2f}"
            else:
                cell = f"{float(np.mean(rhos)):+.3f}"
            cell += f" {nsig}/{R}"
            line += cell.rjust(13)
        L.append(line)
    L.append("  " + "-" * (W + 13 * len(live)))
    for name, key in (("BH survivors (mean of runs)", "surv"),
                      ("logprob saturation %", "sat")):
        line = "  " + name.ljust(W)
        for m, c in live:
            v = cells[(m, c)][key]
            line += (f"{np.mean(v):.1f}" if key == "sat"
                     else f"{np.mean(v):.1f}").rjust(13)
        L.append(line)
    line = "  " + "runs in cell (R)".ljust(W)
    for m, c in live:
        line += str(cells[(m, c)]["R"]).rjust(13)
    L.append(line)
    return "\n".join(L)


def _pm_tex(t):
    m_, d_ = t
    return f"${m_:.3f}$" if d_ is None else f"${m_:.3f}\\pm{d_:.3f}$"


def latex_rq1():
    out = [r"% RQ1 --- entity quality and distance from the human ceiling.",
           r"% Canonicalised labels. R = runs aggregated; dispersion is the",
           r"% between-run half-range (R=2) or SD (R>2).",
           r"\begin{tabular}{llrrcccrc}",
           r"\toprule",
           r"Model & Hints & $R$ & $n$ & Strict F1 (M) & Strict F1 (D) & "
           r"Typing $\kappa$ (M) & Gap & 95\% CI \\",
           r"\midrule"]
    for model in MODELS:
        for cond in CONDITIONS:
            r = rq1_cell(model, cond)
            if r is None:
                out.append(f"{model} & {cond} & \\multicolumn{{7}}{{c}}"
                           f"{{\\emph{{not yet run}}}} \\\\")
                continue
            h = headline_cell(model, cond, canon=True)
            star = "" if r["t_p"] >= 0.05 else "^{*}"
            out.append(f"{model} & {cond} & {r['R']} & {r['n']} & "
                       f"{_pm_tex(h['sMark'])} & {_pm_tex(h['sDav'])} & "
                       f"{_pm_tex(h['kMark'])} & ${r['mean']:+.3f}{star}$ & "
                       f"$[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]$ \\\\")
    out += [r"\midrule",
            r"Human ceiling & --- & --- & --- & $0.478$ & $0.478$ & $0.781$ "
            r"& $0.000$ & --- \\",
            r"\bottomrule", r"\end{tabular}",
            r"% $^{*}$ gap differs from the human ceiling at $p<0.05$ "
            r"(paired $t$, two-sided)."]
    return "\n".join(out)


def latex_rq2():
    cells = {(m, c): rq2_cell(m, c) for m in MODELS for c in CONDITIONS}
    live = [(m, c) for m in MODELS for c in CONDITIONS if cells[(m, c)]]
    out = [r"% RQ2 --- difficulty-prediction signals. Cell = mean Spearman rho",
           r"% across the cell's runs; k/R = runs surviving BH q<0.05 within",
           r"% their own 36-test family. Last row is a Fisher odds ratio",
           r"% (geometric mean; 1.00 = no association).",
           r"\begin{tabular}{l" + "c" * len(live) + "}",
           r"\toprule",
           "Signal & " + " & ".join(
               f"\\shortstack{{{m}\\\\{c}}}" for m, c in live) + r" \\",
           r"\midrule"]
    for label, key in RQ2_SIGNALS:
        is_or = key == "layer2:fisher_or"
        row = [label.replace("->", "$\\rightarrow$").replace("_", "\\_")]
        for m, c in live:
            rhos, nsig, R = cells[(m, c)]["signals"][label]
            if not rhos:
                row.append("---")
                continue
            v = (float(np.exp(np.mean(np.log(rhos)))) if is_or
                 else float(np.mean(rhos)))
            s = f"${v:.2f}$" if is_or else f"${v:+.3f}$"
            row.append(s + (f"\\,{nsig}/{R}" if nsig else ""))
        out.append(" & ".join(row) + r" \\")
    out.append(r"\midrule")
    out.append("Logprob saturation (\\%) & " + " & ".join(
        f"${np.mean(cells[(m, c)]['sat']):.1f}$" for m, c in live) + r" \\")
    out.append("Runs aggregated ($R$) & " + " & ".join(
        f"${cells[(m, c)]['R']}$" for m, c in live) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def latex():
    return latex_rq1() + "\n\n" + latex_rq2()


if __name__ == "__main__":
    if "--latex" in sys.argv:
        print(latex())
        raise SystemExit(0)
    print("=" * 120)
    print("TABLE 1 (RQ1) — entity quality and distance from the human ceiling, "
          "aggregated per model x condition")
    print("=" * 120)
    print(table_rq1_text())
    print()
    print("=" * 120)
    print("TABLE 2 (RQ2) — difficulty-prediction signals, aggregated per "
          "model x condition (rho, k/R runs BH-significant)")
    print("=" * 120)
    print(table_rq2_text())
    print()
    print("=" * 120)
    print("DEP-HINT CONTRAST (dep minus no-dep, paired per sentence, "
          "cell-averaged, canonicalised labels)")
    print("=" * 120)
    for model in MODELS:
        for canon in (True, False):
            r = dep_contrast(model, canon=canon)
            if r is None:
                if canon:
                    print(f"  {model:20s} no contrast available "
                          f"(needs >=1 run in both conditions)")
                continue
            tag = "canon" if canon else "as-is"
            print(f"  {model:20s} {tag:6s} n={r['n']:3d}  "
                  f"{r['nodep_mean']:.4f} -> {r['dep_mean']:.4f}  "
                  f"diff {r['mean']:+.4f}  CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
                  f"t p={fmt_p(r['t_p'])}  se={r['se']:.4f}")
