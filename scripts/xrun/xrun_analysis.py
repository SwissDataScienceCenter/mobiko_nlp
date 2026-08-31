#!/usr/bin/env python
"""
Cross-run analysis rebuild for CROSS_RUN_COMPARISON.txt.

Three blocks, all reusing eval_layer1_output's own normalisation + sentence_f1
so numbers are consistent with the stored per-run reports:

  parity     paired per-sentence strict F1: agent (pooled over the two
             annotators) vs the Mark<->Davnah ceiling on the same sentence.
             gap, 95% CI, paired t, Wilcoxon, TOST at +/-0.05 and +/-0.03.
  replicate  two runs paired on the sentences BOTH scored non-emptily.
  bh         Benjamini-Hochberg over the 36-test difficulty family.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/katinska/mobiko_nlp")
sys.path.insert(0, str(ROOT / "src" / "multi_agent_annotation"))
sys.path.insert(0, str(ROOT / "src" / "multi_agent_annotation" / "evaluation"))

import eval_layer1_output as L1  # noqa: E402
from label_norm import canonicalize_entity_type  # noqa: E402

MARK = ROOT / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
DAV = ROOT / "data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl"
NAMES = ["Mark", "Davnah"]


# ── data ────────────────────────────────────────────────────────────────────
# The emit-time canonicaliser itself, not a local re-implementation of it. The
# local copy handled underscores and whitespace only, so it missed hyphens, the
# missing-separator case and the observed misspellings, and it could disagree
# with what the pipeline had already written.
_canon_type = canonicalize_entity_type


def load(agent_path: Path, canon: bool = False, empties: str = "drop"):
    """Return {norm_sentence: (agent_ents, mark_ents, davnah_ents)}.

    canon=True canonicalises entity types at load, which measures typing CONTENT
    rather than shipped output (see CROSS_RUN_SUMMARY caveat 3). It is threaded
    into the loader rather than applied afterwards so that RQ1 and the RQ2 stack
    reach labels through the same code path.

    empties decides what happens to a sentence the agent ATTEMPTED and returned
    no entities for:
      "drop"  the layer1 'Consensus' convention — the sentence is excluded. This
              answers "how good is the output it produces?".
      "zero"  the sentence is kept and scores F1=0. Both humans annotated it, so
              an empty output is a recall failure, not a defensible abstention.
              This answers "is this system interchangeable with a human
              annotator?", which is RQ1's question, and it is the convention the
              paper reports.
    Either way, sentences the agent never attempted (absent from its JSONL) are
    not added, so incomplete coverage is never charged as error.
    """
    if empties not in ("drop", "zero"):
        raise ValueError(f"empties must be 'drop' or 'zero', got {empties!r}")
    agent = L1.load_agent_records(Path(agent_path), canon=canon)
    human, keys = L1.load_all_human_annotations([MARK, DAV], canon=canon)
    # loader names annotators after their filenames; MARK was passed first.
    mark_key, dav_key = keys[0], keys[1]
    out = {}
    for rec in agent:
        key = L1._normalize(rec["sentence"])
        anns = human.get(key)
        if not anns:
            continue
        if empties == "drop" and not rec["entities"]:
            continue
        if not all(anns.get(n) and anns[n]["entities"] for n in (mark_key, dav_key)):
            continue
        out[key] = (rec["entities"], anns[mark_key]["entities"], anns[dav_key]["entities"])
    return out


def f1(a, b):
    v = L1.sentence_f1(a, b, "strict")
    return 0.0 if v is None else v


# ── tests ───────────────────────────────────────────────────────────────────
def tost(d, margin):
    """Two one-sided t-tests; p = max of the two one-sided p-values."""
    n, m, se = len(d), float(np.mean(d)), stats.sem(d)
    if se == 0:
        return 0.0
    lo = stats.t.sf((m - (-margin)) / se, n - 1)   # H0: mu <= -margin
    hi = stats.t.cdf((m - margin) / se, n - 1)     # H0: mu >= +margin
    return max(lo, hi)


def paired_block(d):
    n, m, se = len(d), float(np.mean(d)), float(stats.sem(d))
    t_p = float(stats.ttest_1samp(d, 0.0).pvalue)
    try:
        w_p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        w_p = float("nan")
    crit = stats.t.ppf(0.975, n - 1)
    return dict(n=n, mean=m, se=se, sd=float(np.std(d, ddof=1)),
                ci=(m - crit * se, m + crit * se), t_p=t_p, w_p=w_p,
                tost05=tost(d, 0.05), tost03=tost(d, 0.03))


def parity(agent_path, canon: bool = False, empties: str = "drop"):
    data = load(agent_path, canon=canon, empties=empties)
    agent, human, n_empty = [], [], 0
    for a, mk, dv in data.values():
        if not a:
            n_empty += 1
        agent.append((f1(a, mk) + f1(a, dv)) / 2.0)
        human.append(f1(mk, dv))
    agent, human = np.array(agent), np.array(human)
    r = paired_block(agent - human)
    r.update(agent_mean=float(agent.mean()), human_mean=float(human.mean()),
             empties=empties, n_empty_scored=n_empty)
    return r


def parity_cell(paths, canon: bool = False, empties: str = "zero"):
    """Parity for a GROUP of runs treated as one cell (e.g. all runs of a model).

    Runs within a cell score the SAME sentences, so pooling their sentences would
    fake independence and shrink the standard errors. Instead each sentence's
    agent score is averaged across the runs that scored it, and the paired test
    is run on those averages: n stays the number of sentences, not sentences x
    runs. This is the method aggregate.py documents for its RQ1 cells.

    Restricted to sentences EVERY run in the cell scored, so a run that stopped
    early cannot shift the cell mean by contributing a different sentence set.
    Between-run dispersion is returned separately rather than absorbed, because
    the whole point of showing a cell is that the spread stays visible.
    """
    stores = [load(p, canon=canon, empties=empties) for p in paths]
    shared = set(stores[0])
    for s in stores[1:]:
        shared &= set(s)
    shared = sorted(shared)
    agent, human, per_run = [], [], [[] for _ in stores]
    for k in shared:
        scores = []
        for i, s in enumerate(stores):
            a, mk, dv = s[k]
            v = (f1(a, mk) + f1(a, dv)) / 2.0
            scores.append(v)
            per_run[i].append(v)
        agent.append(sum(scores) / len(scores))
        _, mk, dv = stores[0][k]
        human.append(f1(mk, dv))
    agent, human = np.array(agent), np.array(human)
    r = paired_block(agent - human)
    run_means = [float(np.mean(v)) for v in per_run]
    r.update(agent_mean=float(agent.mean()), human_mean=float(human.mean()),
             n_runs=len(stores), run_means=run_means,
             run_spread=(max(run_means) - min(run_means)) if len(stores) > 1 else 0.0)
    return r


def replicate(path_a, path_b, ref="pooled", canon: bool = False):
    """Paired comparison of two runs on shared sentences."""
    A, B = load(path_a, canon=canon), load(path_b, canon=canon)
    keys = sorted(set(A) & set(B))
    va, vb = [], []
    for k in keys:
        for store, out in ((A, va), (B, vb)):
            a, mk, dv = store[k]
            out.append(f1(a, mk) if ref == "mark" else (f1(a, mk) + f1(a, dv)) / 2.0)
    va, vb = np.array(va), np.array(vb)
    r = paired_block(vb - va)          # B - A
    r.update(a_mean=float(va.mean()), b_mean=float(vb.mean()))
    return r


# ── BH over the 36-test difficulty family ───────────────────────────────────
def family(run_dir: Path):
    """The 36 difficulty-prediction tests: 16 diffsplit univariate Spearman,
    14 logprob human_table (7 signals x 2 human targets), 4 logprob span-level,
    layer2 Fisher + layer2 Spearman."""
    tests = []
    ds = json.load(open(run_dir / "diffsplit.json"))
    for block, tag in (("detection_univariate", "det"), ("typing_univariate", "typ")):
        for row in ds[block]:
            tests.append((f"{tag}:{row['signal']}", row["p"], row.get("rho")))

    lp = json.load(open(run_dir / "logprob.json"))
    for row in lp["human_table"]:
        for tgt in ("human_difficulty", "human_disagree_rate"):
            cell = row[tgt]
            tests.append((f"lp:{row['signal']}->{tgt}", cell["p"], cell.get("rho")))
    for measure in ("logprob", "entropy"):
        sl = lp["span_level"][measure]
        tests.append((f"span:{measure}:point_biserial", sl["point_biserial_p"],
                      sl.get("point_biserial_rho")))
        tests.append((f"span:{measure}:mean_diff_perm", sl["mean_diff_perm_p"], None))

    l2 = json.load(open(run_dir / "layer2.json"))
    tests.append(("layer2:fisher_or", l2["fisher_p_value"], l2.get("odds_ratio")))
    tests.append(("layer2:spearman", l2["spearman_p_value"], l2.get("spearman_rho")))
    return tests


def bh(tests, q=0.05):
    # A test whose p is null carries no information and must not inflate the
    # family size. qwen35_v2 has no span-level ENTROPY stats (older run, no
    # type_mean_entropy in annotator_entity_logprobs), so its family is 34.
    tests = [t for t in tests
             if t[1] is not None and not (isinstance(t[1], float) and np.isnan(t[1]))]
    ps = np.array([t[1] for t in tests], float)
    order = np.argsort(ps)
    m = len(ps)
    thresh, k = None, 0
    for i, idx in enumerate(order, start=1):
        if ps[idx] <= q * i / m:
            thresh, k = ps[idx], i
    survivors = [tests[idx] for idx in order[:k]] if thresh is not None else []
    qvals = {}
    running = 1.0
    for i in range(m, 0, -1):
        idx = order[i - 1]
        running = min(running, ps[idx] * m / i)
        qvals[tests[idx][0]] = running
    return survivors, qvals, int((ps < 0.05).sum()), m


# ── relation descriptives ───────────────────────────────────────────────────
def relations(agent_path):
    from collections import Counter
    tot, sents, kinds = 0, 0, Counter()
    for line in open(agent_path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        rels = rec.get("final_relations") or []
        tot += len(rels)
        if rels:
            sents += 1
        for r in rels:
            kinds[r.get("relation") or r.get("type") or "?"] += 1
    return tot, sents, kinds


# ── CLI: the RQ1 parity table ───────────────────────────────────────────────
def _parity_table(canon: bool = True, empties: str = "zero") -> None:
    """Print the RQ1 parity table for every run in aggregate.REGISTRY.

    Exists because the F1=0 column was computed ad hoc when it was first
    reported, and CROSS_RUN_SUMMARY flags that it must be scripted before being
    quoted. It is now the paper's primary parity convention, so it has to be
    regenerable from the run files rather than retyped.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aggregate import REGISTRY  # noqa: E402

    print(f"  convention: empties={empties}  labels="
          f"{'canonicalised' if canon else 'as-is'}")
    print(f"  {'run':10s} {'n':>4s} {'empty':>6s} {'agent':>7s} {'human':>7s} "
          f"{'gap':>7s} {'95% CI':>18s} {'t p':>7s} {'TOST.05':>8s} {'TOST.03':>8s}")
    rows = []
    for _model, _cond, tag, path in REGISTRY:
        if not Path(path).exists():
            print(f"  {tag:10s} <missing: {path}>")
            continue
        r = parity(path, canon=canon, empties=empties)
        rows.append((tag, r))
        print(f"  {tag:10s} {r['n']:4d} {r['n_empty_scored']:6d} "
              f"{r['agent_mean']:7.3f} {r['human_mean']:7.3f} {r['mean']:+7.3f} "
              f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] {r['t_p']:7.3f} "
              f"{r['tost05']:8.3f} {r['tost03']:8.3f}")
    below = [t for t, r in rows if r["t_p"] < 0.05 and r["mean"] < 0]
    print(f"\n  significantly BELOW the ceiling: {len(below)}/{len(rows)}"
          f"  {sorted(below)}")
    eq03 = [t for t, r in rows if r["tost03"] < 0.05]
    print(f"  equivalence affirmed at +/-0.03 : {len(eq03)}/{len(rows)}"
          f"  {sorted(eq03) or '(none)'}")


def _parity_latex(canon: bool = True, empties: str = "zero") -> None:
    """Emit the RQ1 table as booktabs LaTeX, sorted by gap.

    Generated rather than typed for the same reason aggregate.py emits its own:
    every hand-copied table in this project has drifted from its source at least
    once. Rows are grouped by whether the run is distinguishable from the ceiling,
    which is the claim the table exists to support.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aggregate import REGISTRY  # noqa: E402

    # Qwen3.5 is a different model from the Qwen3.6 studied here and contributes
    # a single run. It is excluded so the table reports only models the paper
    # actually studies — which costs a row that happened to be one of the two
    # closest to the ceiling, so the exclusion is stated rather than silent.
    EXCLUDE_MODELS = {"Qwen3.5 (legacy)"}
    rows_paths = [(m, c, t, p) for m, c, t, p in REGISTRY
                  if Path(p).exists() and m not in EXCLUDE_MODELS]
    # One row per MODEL, not per run. Fourteen rows is too many for a main table,
    # and most of the count comes from a configuration variable this paper does
    # not study (dependency hints), so per-run rows would invite a reader to
    # compare arms the paper makes no claim about. Collapsing by averaging rather
    # than by picking each model's best run matters: selecting the max of three
    # runs at se~0.013 biases the reported gap upward by roughly one se.
    groups = {}
    for model, _cond, _tag, path in rows_paths:
        groups.setdefault(model, []).append(path)
    cells = {m: parity_cell(ps, canon=canon, empties=empties)
             for m, ps in groups.items()}
    order = sorted(cells, key=lambda m: -cells[m]["mean"])

    print("% GENERATED by xrun_analysis.py --latex. Do not edit by hand.")
    print(f"% convention: empties={empties}, "
          f"labels={'canonicalised' if canon else 'as-is'}")
    print("\\begin{tabular}{lrrrrr}")
    print("\\toprule")
    print("Model & $R$ & $n$ & Gap & 95\\% CI & $p$ \\\\")
    print("\\midrule")
    for model in order:
        r = cells[model]
        p = r["t_p"]
        ps = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        dag = "^{\\dagger}" if p >= 0.05 else ""
        spread = (f"\\,$\\pm${r['run_spread'] / 2:.3f}" if r["n_runs"] > 1 else "")
        print(f"{model.replace(' (legacy)', '')} & {r['n_runs']} & {r['n']} & "
              f"${r['mean']:+.3f}{dag}$ & "
              f"$[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]$ & {ps} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--empties", choices=("drop", "zero"), default="zero",
                    help="how to score sentences the agent attempted and left "
                         "empty (default zero — the paper's convention)")
    ap.add_argument("--as-is", action="store_true",
                    help="use shipped labels instead of canonicalised ones")
    ap.add_argument("--latex", action="store_true",
                    help="emit the RQ1 table as booktabs LaTeX")
    args = ap.parse_args()
    if args.latex:
        _parity_latex(canon=not args.as_is, empties=args.empties)
    else:
        _parity_table(canon=not args.as_is, empties=args.empties)
