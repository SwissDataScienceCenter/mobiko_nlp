#!/usr/bin/env python
"""Write label-canonicalised copies of runs, then re-score entity headlines.

Canonicaliser (the fix recommended in section 10b, applied to final_entities
entity_type only — nothing else is touched):
    re.sub(r"\\s+", " ", t.replace("_", " ")).strip().upper()
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import xrun_analysis as X  # noqa: E402

ROOT = Path("/home/katinska/mobiko_nlp")
D = ROOT / "data"
OUT = Path(__file__).parent / "canon"
OUT.mkdir(exist_ok=True)

RUNS = {
    "kimi27_dep":        D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v1_dep",
    "apertus70B_nodep":  D / "auto_annotated/datademo_manually_labeled_swissai-apertus-70B.jsonl",
    "apertus70B_dep":    D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_dep",
    "apertus70B_dep_v2": D / "auto_annotated/datademo_manually_labeled_dep_apertus70B_dep_v2.dedup.jsonl",
}


def canon(t):
    return re.sub(r"\s+", " ", (t or "").replace("_", " ")).strip().upper()


paths = {}
for name, src in RUNS.items():
    dst = OUT / (name + ".canon.jsonl")
    changed = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            s = line.strip()
            if not s.startswith("{"):
                continue
            rec = json.loads(s)
            for e in rec.get("final_entities") or []:
                for key in ("entity_type", "type"):
                    if key in e and e[key]:
                        new = canon(e[key])
                        if new != e[key]:
                            changed += 1
                        e[key] = new
            fout.write(json.dumps(rec) + "\n")
    paths[name] = dst
    print(f"  {name:20s} canonicalised {changed} entity_type strings -> {dst.name}")

print()
print("=" * 92)
print("COST OF LABEL HYGIENE — as-is vs canonicalised, same eval set")
print("=" * 92)
print(f"  {'run':20s} {'variant':6s} {'n':>4s} {'agent':>7s} {'gap':>8s} {'t p':>7s} "
      f"{'TOST.05':>8s}")
res = {}
for name, src in RUNS.items():
    for variant, p in (("as-is", src), ("canon", paths[name])):
        r = X.parity(p)
        res[(name, variant)] = r
        print(f"  {name:20s} {variant:6s} {r['n']:4d} {r['agent_mean']:7.4f} "
              f"{r['mean']:+8.4f} {r['t_p']:7.3f} {r['tost05']:8.4f}")

print()
print("=" * 92)
print("APERTUS DEP vs NODEP — recomputed on CANONICALISED labels (paired)")
print("=" * 92)
for a, b, note in (("apertus70B_nodep", "apertus70B_dep", "dep v1 vs nodep"),
                   ("apertus70B_nodep", "apertus70B_dep_v2", "dep v2 vs nodep"),
                   ("apertus70B_dep", "apertus70B_dep_v2", "dep replicate pair")):
    for variant in ("as-is", "canon"):
        pa = RUNS[a] if variant == "as-is" else paths[a]
        pb = RUNS[b] if variant == "as-is" else paths[b]
        r = X.replicate(pa, pb)
        print(f"  {note:20s} {variant:6s} n={r['n']:3d}  {r['a_mean']:.4f} -> {r['b_mean']:.4f}  "
              f"diff {r['mean']:+.4f}  CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
              f"t p={r['t_p']:.3f}  se={r['se']:.4f}")

print()
print("=" * 92)
print("KIMI DEP — canonicalised, vs its no-hint controls (paired)")
print("=" * 92)
KIMI = {
    "kimi27_v2": D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v2",
    "kimi27_v3": D / "auto_annotated_local/datademo_manually_labeled_rcp-kimi-2.7_v3",
}
for ctrl, cp in KIMI.items():
    r = X.replicate(cp, paths["kimi27_dep"])
    print(f"  {ctrl} -> kimi27_dep(canon)  n={r['n']:3d}  {r['a_mean']:.4f} -> {r['b_mean']:.4f}  "
          f"diff {r['mean']:+.4f}  CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
          f"t p={r['t_p']:.3f}  se={r['se']:.4f}")
