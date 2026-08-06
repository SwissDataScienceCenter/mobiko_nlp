#!/usr/bin/env python
"""Print every cross-run number needed for CROSS_RUN_COMPARISON.txt."""
import json
from pathlib import Path

import xrun_analysis as X

ROOT = Path("/home/katinska/mobiko_nlp")
D = ROOT / "data"
XR = ROOT / "output/eval_reports/cross_run_comparison"

RUNS = {
    "qwen35_v2":         D / "auto_annotated/datademo_manually_labeled2.jsonl",
    "qwen36_35B":        D / "auto_annotated/datademo_manually_labeled_swissai-qwen3-35B-vllm",
    "qwen36_35B_v3":     D / "auto_annotated/datademo_manually_labeled_qwen3-35B-vllm_v3",
    "kimi27_v2":         D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v2",
    "kimi27_v3":         D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v3",
    "kimi27_dep":        D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v1_dep",
    "apertus70B_nodep":  D / "auto_annotated/datademo_manually_labeled_swissai-apertus-70B.jsonl",
    "apertus70B_dep":    D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_dep",
    "apertus70B_dep_v2": D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_dep_v2.dedup.jsonl",
}

print("=" * 78)
print("MAPPING CHECK  (parity n must equal layer1.txt Mark eval-sentence count)")
print("=" * 78)
for name in RUNS:
    rp = XR / name / "layer1.txt"
    mark_n = "?"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip().startswith("Mark ") and "sentences" in line:
                mark_n = line.split()[-2]
                break
    print(f"  {name:20s} file exists={RUNS[name].exists()}  layer1 Mark n={mark_n}")

print()
print("=" * 78)
print("PARITY — paired per-sentence strict F1 (agent pooled) vs Mark<->Davnah")
print("=" * 78)
print(f"  {'run':20s} {'n':>4s} {'agent':>7s} {'human':>7s} {'gap':>8s} "
      f"{'95% CI':>20s} {'t p':>7s} {'W p':>7s} {'TOST.05':>8s} {'TOST.03':>8s} {'sd':>6s}")
par = {}
for name, path in RUNS.items():
    r = X.parity(path)
    par[name] = r
    print(f"  {name:20s} {r['n']:4d} {r['agent_mean']:7.4f} {r['human_mean']:7.4f} "
          f"{r['mean']:+8.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
          f"{r['t_p']:7.3f} {r['w_p']:7.3f} {r['tost05']:8.4f} {r['tost03']:8.4f} {r['sd']:6.3f}")

print()
print("=" * 78)
print("REPLICATE / CONFIG PAIRS — paired on sentences BOTH runs scored")
print("=" * 78)
PAIRS = [
    ("qwen36_35B", "qwen36_35B_v3", "NEW qwen replicate pair (same config)"),
    ("kimi27_v2",  "kimi27_v3",     "kimi replicate pair (same config)"),
    ("kimi27_v3",  "kimi27_dep",    "kimi dep-hints vs no-hints"),
    ("kimi27_v2",  "kimi27_dep",    "kimi dep-hints vs no-hints (other control)"),
    ("apertus70B_dep", "apertus70B_dep_v2", "apertus replicate pair (same config)"),
    ("apertus70B_nodep", "apertus70B_dep",   "apertus dep vs nodep (v1)"),
    ("apertus70B_nodep", "apertus70B_dep_v2", "apertus dep vs nodep (v2)"),
    ("qwen35_v2", "qwen36_35B", "cross-model (different qwen builds)"),
]
for a, b, note in PAIRS:
    r = X.replicate(RUNS[a], RUNS[b])
    print(f"  {a} -> {b}   [{note}]")
    print(f"     n={r['n']:3d}  {r['a_mean']:.4f} -> {r['b_mean']:.4f}  "
          f"diff {r['mean']:+.4f}  CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
          f"t p={r['t_p']:.3f}  W p={r['w_p']:.3f}  se={r['se']:.4f}  "
          f"|d|/se={abs(r['mean'])/r['se']:.2f}")

print()
print("=" * 78)
print("BH OVER THE 36-TEST DIFFICULTY FAMILY")
print("=" * 78)
for name in RUNS:
    rd = XR / name
    if not (rd / "diffsplit.json").exists():
        print(f"  {name:20s} reports missing")
        continue
    tests = X.family(rd)
    surv, qv, raw, m = X.bh(tests)
    print(f"  {name:20s} m={m}  raw p<0.05 = {raw}  BH q<0.05 survivors = {len(surv)}")
    for label, p, rho in surv:
        rs = f"rho={rho:+.3f}" if isinstance(rho, (int, float)) else "        "
        print(f"       {label:44s} p={p:.5f}  q={qv[label]:.4f}  {rs}")

print()
print("=" * 78)
print("KEY RQ2 SIGNALS PER RUN (rho, p) — the ones tracked across sections 7-10")
print("=" * 78)
TRACK = ["span:logprob:point_biserial", "layer2:fisher_or"]
for name in RUNS:
    rd = XR / name
    if not (rd / "diffsplit.json").exists():
        continue
    tests = dict((t[0], t) for t in X.family(rd))
    surv, qv, raw, m = X.bh(tests.values() if False else X.family(rd))
    lp = json.load(open(rd / "logprob.json"))
    sat = lp.get("saturation") or lp.get("pct_saturated")
    print(f"  --- {name}")
    for key in TRACK:
        if key in tests:
            _, p, rho = tests[key]
            rs = f"{rho:+.3f}" if isinstance(rho, (int, float)) else "  n/a "
            print(f"      {key:32s} {rs}  p={p:.5f}  q={qv[key]:.4f}")
    # self-confidence rows live in the logprob human_table
    for row in lp["human_table"]:
        if "conf" in row["signal"].lower():
            for tgt in ("human_difficulty", "human_disagree_rate"):
                c = row[tgt]
                k = f"lp:{row['signal']}->{tgt}"
                print(f"      {k:32s} {c['rho']:+.3f}  p={c['p']:.5f}  q={qv[k]:.4f}")

print()
print("=" * 78)
print("RELATIONS (descriptive) + LABEL HYGIENE")
print("=" * 78)
for name, path in RUNS.items():
    tot, sents, kinds = X.relations(path)
    print(f"  {name:20s} total={tot:4d}  sents_with_rel={sents:3d}  "
          + " ".join(f"{k}={v}" for k, v in kinds.most_common(8)))
