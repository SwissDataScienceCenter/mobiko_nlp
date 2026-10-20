#!/usr/bin/env python
"""
TODO 1.2 — does the pipeline's OWN uncertainty predict human disagreement, and
does it beat the token logprob?

The paper reads uncertainty out of the decoder (span type logprob) and never
tests the signals the deliberation itself produces, although every run records
them: the Critic's per-span disputes, the Adjudicator's audit of what it changed,
the Annotator's self-reported confidence, and the sentence's review flag.

This matters beyond bookkeeping. If a pipeline signal predicts human
disagreement as well as the logprob does, the three agents are load-bearing for
the paper's central claim. If only the logprob works, the same result could have
been obtained from a single annotator model, and the paper should say so.

Every signal is scored on EXACTLY the rows the paper's span-level logprob
analysis uses — agent spans whose normalised text at least one annotator also
marked — so the comparison is like for like. Target is the same one the paper
uses: did the two annotators assign this span different types.

Usage:
    python scripts/xrun/pipeline_signals.py [--runs TAG ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import xrun_analysis as X                     # noqa: E402
import eval_layer2_correlation as L2          # noqa: E402
from aggregate import REGISTRY                # noqa: E402
from deliberation_history import (            # noqa: E402
    load_records, norm, agent_disagreed_spans, record_signals, try_parse_json,
)
import numpy as np  # noqa: E402

from eval_utils import _spearman_rho, permutation_p_rho  # noqa: E402

N_PERM = 2000

# "entity \"centres\": SPATIAL ENTITY" -> centres
_AUDIT_ENTITY = re.compile(r'entity\s+"([^"]+)"')

# label, higher_is_uncertain
SIGNALS = [
    ("logprob",      "type logprob (paper's)",   False),
    ("entropy",      "type max entropy",         True),
    ("selfconf",     "1 - self-confidence",      True),
    ("critic",       "Critic disputed span",     True),
    ("adjudicated",  "Adjudicator touched span", True),
    ("flagged",      "sentence review-flagged",  True),
]


def audit_touched(rec: dict) -> set:
    """Normalised text of every entity the Adjudicator changed or rejected.

    preserved_consensus is deliberately excluded: those are the spans nothing
    happened to, which is the negative class.
    """
    audit = rec.get("adjudication_audit") or {}
    out = set()
    for key in ("allowed_changes", "rejected_changes"):
        for line in audit.get(key) or []:
            for m in _AUDIT_ENTITY.finditer(str(line)):
                out.add(L2._normalize(m.group(1)))
    return out


def build_rows(path: Path, hnorm: dict) -> list:
    rows = []
    for rec in load_records(path):
        anns = hnorm.get(norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        dis = L2.compute_human_span_disagreements(anns)
        critic = agent_disagreed_spans(rec)
        touched = audit_touched(rec)
        flagged = 1.0 if rec.get("flagged_for_human_review") else 0.0
        # Self-reported confidence must come from the ANNOTATOR'S FIRST
        # MESSAGE, not from final_entities.
        #
        # final_entities is the ADJUDICATOR's output: its confidences have
        # already been revised by the deliberation. Reading them here put the
        # Critic's influence inside the feature called "selfconf", so any
        # comparison of self-confidence against the Critic was circular — the
        # baseline contained the thing it was being compared with. The logprob
        # and entropy in the same row come from annotator_entity_logprobs, which
        # IS the first pass, so the row mixed two different points in the
        # pipeline under one label.
        #
        # Parsed from messages rather than read from record.annotator_rounds
        # because the archived runs predate that field.
        conf = {}
        for _m in rec.get("messages") or []:
            if _m.get("agent") != "Annotator":
                continue
            _p = try_parse_json(_m.get("content", ""))
            if not _p or "entities" not in _p:
                continue
            conf = {L2._normalize(e.get("text", "")): e.get("confidence")
                    for e in _p["entities"]
                    if isinstance(e.get("confidence"), (int, float))}
            break

        for e in (rec.get("annotator_entity_logprobs") or []):
            key = L2._normalize(e.get("text", ""))
            info = dis.get(key)
            if info is None:
                continue        # same exclusion the paper's span_level applies
            c = conf.get(key)
            rows.append({
                "target": 1.0 if info["type_disagreed"] else 0.0,
                "logprob": e.get("type_mean_logprob"),
                "entropy": e.get("type_max_entropy"),
                "selfconf": (1.0 - c) if c is not None else None,
                "critic": 1.0 if critic.get(key, {}).get("agent_disagreed") else 0.0,
                "adjudicated": 1.0 if key in touched else 0.0,
                "flagged": flagged,
            })
    return rows


def score(rows, key, higher_is_uncertain, restrict_to=None):
    """Point-biserial Spearman of uncertainty vs human type-disagreement."""
    use = [r for r in rows if r[key] is not None
           and (restrict_to is None or r[restrict_to] is not None)]
    if len(use) < 3:
        return None
    xs = [(r[key] if higher_is_uncertain else -r[key]) for r in use]
    ys = [r["target"] for r in use]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return dict(n=len(use), rho=None, p=None, base=sum(ys) / len(ys))
    return dict(n=len(use), rho=_spearman_rho(xs, ys),
                p=permutation_p_rho(xs, ys, n_boot=N_PERM),
                base=sum(ys) / len(ys))


def auc(scores, labels):
    """P(a disagreed span scores higher than an agreed one); ties count half.

    Reported alongside rho because the paper's own replicate analysis found AUC
    the steadier of the two — it moves by at most 0.035 across identical
    configurations where the correlations move by up to 0.098.
    """
    pos = sum(1 for l in labels if l == 1)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = np.empty(len(scores))
    i = 0
    while i < len(order):                       # average ranks within ties
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rp = sum(ranks[i] for i, l in enumerate(labels) if l == 1)
    return (rp - pos * (pos + 1) / 2) / (pos * neg)


def bh_survivors(ps, q=0.05):
    """Indices surviving Benjamini-Hochberg — the correction the paper applies
    within each run's own test family."""
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    m, thr = len(ps), None
    for rank, i in enumerate(idx, 1):
        if ps[i] <= q * rank / m:
            thr = rank
    return {idx[i] for i in range(thr)} if thr else set()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", help="restrict to these registry tags")
    args = ap.parse_args()

    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): a for s, a in human.items()}

    runs = [(t, p) for _m, _c, t, p in REGISTRY
            if Path(p).exists() and (not args.runs or t in args.runs)]

    print("Span-level: pipeline signals vs the token logprob, same rows, same target")
    print("rho = point-biserial Spearman vs human type-disagreement; * = perm p < 0.05\n")
    head = f"  {'run':24s}" + "".join(f"{lbl.split()[0][:9]:>11s}" for _k, lbl, _h in SIGNALS)
    print(head)
    print("  " + "-" * (len(head) - 2))

    totals = {k: [] for k, _l, _h in SIGNALS}
    for tag, path in runs:
        rows = build_rows(Path(path), hnorm)
        line = f"  {tag:24s}"
        for key, _lbl, hi in SIGNALS:
            # every signal scored on the rows where the LOGPROB also exists,
            # so no signal gets an easier row set than the baseline
            r = score(rows, key, hi, restrict_to="logprob")
            if r is None or r["rho"] is None:
                line += f"{'-':>11s}"
                continue
            star = "*" if (r["p"] is not None and r["p"] < 0.05) else " "
            line += f"{r['rho']:>10.3f}{star}"
            totals[key].append(r["rho"])
        print(line)

    # ── AUC on identical rows, BH-corrected ─────────────────────────────────
    keys = [(k, h) for k, _l, h in SIGNALS if k != "entropy"]
    print("\n\nAUC on the rows carrying BOTH a logprob and a self-confidence,")
    print("so no signal is scored on an easier subset. 0.5 = uninformative;")
    print("* survives Benjamini-Hochberg within the run's own family.\n")
    print(f"  {'run':22s} {'n':>5s} {'base':>6s}"
          + "".join(f"{k[:9]:>11s}" for k, _h in keys))
    aucs_all = {k: [] for k, _h in keys}
    beats = 0
    for tag, path in runs:
        rows = [r for r in build_rows(Path(path), hnorm)
                if r["logprob"] is not None and r["selfconf"] is not None]
        if not rows:
            continue
        ys = [r["target"] for r in rows]
        vals, ps = {}, []
        for k, hi in keys:
            xs = [(r[k] if hi else -r[k]) for r in rows]
            vals[k] = auc(xs, ys)
            ps.append(permutation_p_rho(xs, ys, n_boot=N_PERM)
                      if len(set(xs)) > 1 else 1.0)
        keep = bh_survivors(ps)
        line = f"  {tag:22s} {len(rows):5d} {sum(ys) / len(ys):6.2f}"
        for i, (k, _h) in enumerate(keys):
            line += (f"{vals[k]:10.3f}{'*' if i in keep else ' '}"
                     if vals[k] is not None else f"{'-':>11s}")
            if vals[k] is not None:
                aucs_all[k].append(vals[k])
        print(line)
        if vals["selfconf"] and vals["logprob"] and vals["selfconf"] > vals["logprob"]:
            beats += 1

    print(f"\n  {'signal':14s} {'runs':>5s} {'mean':>7s} {'min':>7s} {'max':>7s} {'>0.5':>9s}")
    for k, _h in keys:
        v = aucs_all[k]
        if v:
            print(f"  {k:14s} {len(v):5d} {sum(v) / len(v):7.3f} {min(v):7.3f} "
                  f"{max(v):7.3f} {sum(1 for x in v if x > 0.5):5d}/{len(v)}")
    print(f"\n  self-confidence beats the logprob on AUC in {beats}/{len(runs)} runs,")
    print("  but the point is the range, not the mean: see min/max above.")

    print("\n  summary across runs")
    print(f"  {'signal':26s} {'runs':>5s} {'mean rho':>9s} {'min':>8s} {'max':>8s}")
    for key, lbl, _h in SIGNALS:
        v = totals[key]
        if not v:
            print(f"  {lbl:26s} {0:5d} {'-':>9s}")
            continue
        print(f"  {lbl:26s} {len(v):5d} {sum(v)/len(v):9.3f} "
              f"{min(v):8.3f} {max(v):8.3f}")


if __name__ == "__main__":
    main()
