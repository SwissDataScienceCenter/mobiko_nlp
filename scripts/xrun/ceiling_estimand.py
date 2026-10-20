#!/usr/bin/env python
"""
TODO 1.1 — audit of the RQ1 ceiling comparison.

The draft says the agent is scored "pooled over the two annotators" while the
humans are scored against each other. If "pooled" meant union or best-of-two,
the agent would get two chances where each human got one, and the headline
-0.014 gap would be inflated by more than its own size.

This script recomputes every RQ1 cell under six ways of resolving the agent
against two annotators, so the choice is measured rather than asserted:

  mean    (F1(A,M) + F1(A,D)) / 2      the draft's convention
  mark     F1(A,M)                     against one annotator only
  davnah   F1(A,D)                     against the other
  max     max(F1(A,M), F1(A,D))        best-of-two — the inflation the referee feared
  min     min(F1(A,M), F1(A,D))        the pessimistic bound
  union    F1(A, M u D)                 the other thing "pooled" could have meant

The ceiling is F1(M,D) in every case, so only the agent side changes.

Also reports what the "both annotators non-empty" filter in xrun_analysis.load
costs, since that filter sets n=154 and shifts the ceiling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/katinska/mobiko_nlp")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import xrun_analysis as X  # noqa: E402
from aggregate import REGISTRY  # noqa: E402

EXCLUDE_MODELS = {"Qwen3.5 (legacy)"}
ESTIMANDS = ("mean", "mark", "davnah", "max", "min", "union")


def _union(m, d):
    """Multiset union of two annotators' entities, keyed strictly."""
    seen, out = set(), []
    for e in list(m) + list(d):
        k = (e.get("start_char"), e.get("end_char"), (e.get("type") or "").strip().upper())
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def agent_score(a, m, d, how):
    fm, fd = X.f1(a, m), X.f1(a, d)
    if how == "mean":
        return (fm + fd) / 2.0
    if how == "mark":
        return fm
    if how == "davnah":
        return fd
    if how == "max":
        return max(fm, fd)
    if how == "min":
        return min(fm, fd)
    if how == "union":
        return X.f1(a, _union(m, d))
    raise ValueError(how)


def cell(paths, how, canon=True, empties="zero"):
    """Parity for one model cell under one estimand (mirrors X.parity_cell)."""
    stores = [X.load(p, canon=canon, empties=empties) for p in paths]
    shared = set(stores[0])
    for s in stores[1:]:
        shared &= set(s)
    agent, human = [], []
    for k in sorted(shared):
        scores = [agent_score(*s[k], how) for s in stores]
        agent.append(sum(scores) / len(scores))
        _, m, d = stores[0][k]
        human.append(X.f1(m, d))
    agent, human = np.array(agent), np.array(human)
    r = X.paired_block(agent - human)
    r.update(agent_mean=float(agent.mean()), human_mean=float(human.mean()),
             n_runs=len(stores))
    return r


def main():
    groups = {}
    for model, _cond, _tag, path in REGISTRY:
        if model in EXCLUDE_MODELS or not Path(path).exists():
            continue
        groups.setdefault(model, []).append(path)

    results = {m: {h: cell(ps, h) for h in ESTIMANDS} for m, ps in groups.items()}
    order = sorted(results, key=lambda m: -results[m]["mean"]["mean"])

    print("RQ1 gap (agent - ceiling) under six ways of resolving two annotators")
    print("convention: empties=zero, labels=canonicalised (the paper's)\n")
    head = f"  {'Model':22s} {'R':>2s} {'n':>4s}" + "".join(f"{h:>9s}" for h in ESTIMANDS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for m in order:
        r = results[m]
        row = f"  {m:22s} {r['mean']['n_runs']:2d} {r['mean']['n']:4d}"
        row += "".join(f"{r[h]['mean']:+9.3f}" for h in ESTIMANDS)
        print(row)

    print(f"\n  ceiling (mean per-sentence F1(M,D)) = "
          f"{results[order[0]]['mean']['human_mean']:.3f}")

    print("\n  p-values (paired t vs the ceiling)\n")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for m in order:
        r = results[m]
        row = f"  {m:22s} {r['mean']['n_runs']:2d} {r['mean']['n']:4d}"
        row += "".join(f"{r[h]['t_p']:9.3f}" for h in ESTIMANDS)
        print(row)

    print("\n  how many cells are NOT distinguishable from the ceiling (p>=0.05)")
    for h in ESTIMANDS:
        ns = [m for m in order if results[m][h]["t_p"] >= 0.05]
        print(f"    {h:7s} {len(ns)}/{len(order)}  {ns}")

    print("\n  inflation of the best cell if 'pooled' had meant best-of-two or union")
    best = order[0]
    base = results[best]["mean"]["mean"]
    for h in ("max", "union"):
        print(f"    {best}: {base:+.3f} (mean) -> {results[best][h]['mean']:+.3f} "
              f"({h}), difference {results[best][h]['mean'] - base:+.3f}")

    print("\n  per-annotator asymmetry (agent mean F1 against each)")
    print(f"  {'Model':22s} {'vs Mark':>9s} {'vs Davnah':>10s} {'diff':>8s}")
    for m in order:
        a, b = results[m]["mark"]["agent_mean"], results[m]["davnah"]["agent_mean"]
        print(f"  {m:22s} {a:9.3f} {b:10.3f} {a - b:+8.3f}")


if __name__ == "__main__":
    main()
