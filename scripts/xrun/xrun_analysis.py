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

MARK = ROOT / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
DAV = ROOT / "data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl"
NAMES = ["Mark", "Davnah"]


# ── data ────────────────────────────────────────────────────────────────────
def _canon_type(t):
    """The emit-time canonicaliser: underscores -> spaces, whitespace collapsed.
    Applied to PREDICTIONS only; gold is already clean."""
    return re.sub(r"\s+", " ", (t or "").replace("_", " ")).strip().upper()


def load(agent_path: Path, canon: bool = False):
    """Return {norm_sentence: (agent_ents, mark_ents, davnah_ents)} for sentences
    where the agent is non-empty and BOTH annotators are non-empty — i.e. exactly
    the layer1 'Consensus' eval set.

    canon=True canonicalises predicted entity types in memory, which measures
    typing CONTENT rather than shipped output (see CROSS_RUN_SUMMARY caveat 3)."""
    agent = L1.load_agent_records(Path(agent_path))
    if canon:
        for rec in agent:
            for e in rec["entities"]:
                e["type"] = _canon_type(e["type"])
    human, keys = L1.load_all_human_annotations([MARK, DAV])
    # loader names annotators after their filenames; MARK was passed first.
    mark_key, dav_key = keys[0], keys[1]
    out = {}
    for rec in agent:
        key = L1._normalize(rec["sentence"])
        anns = human.get(key)
        if not anns or not rec["entities"]:
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


def parity(agent_path, canon: bool = False):
    data = load(agent_path, canon=canon)
    agent, human = [], []
    for a, mk, dv in data.values():
        agent.append((f1(a, mk) + f1(a, dv)) / 2.0)
        human.append(f1(mk, dv))
    agent, human = np.array(agent), np.array(human)
    r = paired_block(agent - human)
    r.update(agent_mean=float(agent.mean()), human_mean=float(human.mean()))
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
