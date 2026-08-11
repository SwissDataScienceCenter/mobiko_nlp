#!/usr/bin/env python
"""RQ2 signal x run matrix.

  matrix.py             full 36 x 9 matrix, every test in every run's family
  matrix.py --summary   the 7-row block pasted into CROSS_RUN_SUMMARY.txt
"""
import json
import sys
from pathlib import Path

import xrun_analysis as X

XR = Path("/home/katinska/mobiko_nlp/output/eval_reports/cross_run_comparison")

# Which runs exist is owned by aggregate.REGISTRY — the single source of truth,
# so adding a run there automatically adds a column here. Short display codes
# are optional; a run with no entry falls back to its report-dir name.
from aggregate import REGISTRY  # noqa: E402

SHORT = {
    "qwen35_v2": "q35v2", "qwen36_35B": "q36", "qwen36_35B_v3": "q36v3",
    "qwen36_35B_dep": "q36dep", "kimi27_v2": "k27v2", "kimi27_v3": "k27v3",
    "kimi27_dep": "k27dep", "apertus70B_nodep": "apNO",
    "apertus70B_dep": "apD1", "apertus70B_dep_v2": "apD2",
    "apertus70B_nodep_v3": "apNO3", "apertus70B_dep_v3": "apD3", "apertus70B_dep_v3": "apD3",
    "gptoss_dep": "gptD", "deepseek_dep": "dsD",
}
# report order: no-dep runs of a model, then its dep runs, models in registry order
ORDER = [d for _, c, d, _ in REGISTRY if c == "no-dep"] + \
        [d for _, c, d, _ in REGISTRY if c == "dep"]
ORDER = sorted(dict.fromkeys(ORDER), key=lambda d: (
    [m for m, _, dd, _ in REGISTRY if dd == d][0],
    [c for _, c, dd, _ in REGISTRY if dd == d][0] == "dep", d))
CODES = [SHORT.get(d, d) for d in ORDER]

# the rows carried in CROSS_RUN_SUMMARY.txt's RQ2 table
SUMMARY_ROWS = [
    ("SPAN type-logprob -> type-disagr", "span:logprob:point_biserial", "rho"),
    ("SPAN type-entropy -> type-disagr", "span:entropy:point_biserial", "rho"),
    ("logprob type-min -> difficulty", "lp:logprob (type-min)->human_difficulty", "rho"),
    ("self-confidence -> difficulty", "lp:self-confidence->human_difficulty", "rho"),
    ("self-confidence -> disagr rate", "lp:self-confidence->human_disagree_rate", "rho"),
    ("entity count -> typing difficulty", "typ:agent entity count", "rho"),
    ("critic disagr -> human disagr (OR)", "layer2:fisher_or", "or"),
]


def saturation(run):
    for line in (XR / run / "logprob.txt").read_text().splitlines():
        if "% of entities have logprob" in line:
            return line.strip().split("%")[0] + "%"
    return "?"


def summary_table():
    """Emit the exact block used in CROSS_RUN_SUMMARY.txt (aligned, 7-char cells)."""
    cells, qs, meta = {}, {}, {}
    for code, run in zip(CODES, ORDER):
        tests = X.family(XR / run)
        surv, qv, raw, m = X.bh(tests)
        cells[code] = {lab: (p, rho) for lab, p, rho in tests}
        qs[code], meta[code] = qv, (raw, m, len(surv))

    W = 36
    out = ["  " + "signal".ljust(W) + "".join(c.rjust(7) for c in CODES)]
    for lab, key, kind in SUMMARY_ROWS:
        line = "  " + lab.ljust(W)
        for code in CODES:
            p, rho = cells[code].get(key, (None, None))
            if p is None:
                line += "--".rjust(6) + " "
            else:
                star = "*" if qs[code].get(key, 1) < 0.05 else " "
                v = f"{rho:.2f}" if kind == "or" else f"{rho:+.3f}"
                line += v.rjust(6) + star
        out.append(line.rstrip())
    out.append("  " + "-" * (W + 63))
    out.append("  " + "raw p<0.05 / family".ljust(W)
               + "".join(f"{meta[c][0]}/{meta[c][1]}".rjust(7) for c in CODES))
    out.append("  " + "BH q<0.05 survivors".ljust(W)
               + "".join(str(meta[c][2]).rjust(7) for c in CODES))
    out.append("  " + "logprob saturation".ljust(W)
               + "".join(saturation(r).rjust(7) for r in ORDER))
    return "\n".join(out)


if "--summary" in sys.argv:
    print(summary_table())
    raise SystemExit(0)

# collect every test in the 36-family for every run, keyed by test label
cells, qs = {}, {}
for run in ORDER:
    tests = X.family(XR / run)
    surv, qv, raw, m = X.bh(tests)
    cells[run] = {lab: (p, rho) for lab, p, rho in tests}
    qs[run] = qv
    print(f"# {run:20s} raw p<.05 {raw:2d}/{m}  BH survivors {len(surv)}")

labels = sorted({lab for run in ORDER for lab in cells[run]})
print()
for lab in labels:
    row = []
    for run in ORDER:
        p, rho = cells[run].get(lab, (None, None))
        if p is None:
            row.append("    --      ")
            continue
        rs = f"{rho:+.3f}" if isinstance(rho, (int, float)) else "  .  "
        star = "*" if qs[run].get(lab, 1) < 0.05 else " "
        row.append(f"{rs} p{p:.3f}{star}")
    print(f"{lab:46s} " + " ".join(row))

print()
print("# saturation (% entities with type-logprob > -0.01)")
for run in ORDER:
    txt = (XR / run / "logprob.txt").read_text()
    for line in txt.splitlines():
        if "% of entities have logprob" in line:
            print(f"#   {run:20s} {line.strip()}")
            break
