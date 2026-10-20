"""Why does the agent agree more with Mark than with Davnah? And what does the
'both annotators non-empty' filter cost?"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/katinska/mobiko_nlp")
sys.path.insert(0, str(ROOT / "scripts" / "xrun"))
import xrun_analysis as X
from aggregate import REGISTRY
sys.path.insert(0, str(ROOT / "src" / "multi_agent_annotation"))
sys.path.insert(0, str(ROOT / "src" / "multi_agent_annotation" / "evaluation"))
import eval_layer1_output as L1

# ---- filter cost -----------------------------------------------------------
path = [p for m, c, t, p in REGISTRY if m == "DeepSeek guid-v4"][0]
agent = L1.load_agent_records(Path(path))
human, keys = L1.load_all_human_annotations([X.MARK, X.DAV])
mk, dv = keys[0], keys[1]
attempted = both = one_empty = no_human = 0
dropped_ceiling = []
for rec in agent:
    k = L1._normalize(rec["sentence"])
    anns = human.get(k)
    attempted += 1
    if not anns:
        no_human += 1
        continue
    has_m = bool(anns.get(mk) and anns[mk]["entities"])
    has_d = bool(anns.get(dv) and anns[dv]["entities"])
    if has_m and has_d:
        both += 1
    else:
        one_empty += 1
        if has_m or has_d:
            dropped_ceiling.append(0.0)   # one annotated, one didn't -> F1(M,D)=0
print("=== what the 'both annotators non-empty' filter costs ===")
print(f"  sentences the agent attempted        {attempted}")
print(f"  no human annotation record            {no_human}")
print(f"  at least one annotator empty          {one_empty}"
      f"   (of which exactly one empty: {len(dropped_ceiling)})")
print(f"  kept (n for the paired test)          {both}")
if dropped_ceiling:
    print(f"  those {len(dropped_ceiling)} sentences have F1(M,D)=0 by construction, so")
    print(f"  keeping them would pull the ceiling DOWN from 0.497 toward "
          f"{0.497*both/(both+len(dropped_ceiling)):.3f} and favour the agent.")

# ---- per-annotator profile -------------------------------------------------
print("\n=== annotator profiles on the 154 kept sentences ===")
store = X.load(path, canon=True, empties="zero")
nm = np.mean([len(m) for _, m, _ in store.values()])
nd = np.mean([len(d) for _, _, d in store.values()])
na = np.mean([len(a) for a, _, _ in store.values()])
print(f"  mean spans/sentence   Mark {nm:.2f}   Davnah {nd:.2f}   agent {na:.2f}")

def prf(pred, gold):
    k = L1._keyer("strict")
    pc, gc = L1._counter(pred, k), L1._counter(gold, k)
    return L1._prf(sum((pc & gc).values()), sum(pc.values()), sum(gc.values()))

print(f"\n  {'contrast':22s} {'P':>7s} {'R':>7s} {'F1':>7s}")
for name, get in (("agent -> Mark", lambda t: (t[0], t[1])),
                  ("agent -> Davnah", lambda t: (t[0], t[2])),
                  ("Mark -> Davnah", lambda t: (t[1], t[2]))):
    tp = pn = gn = 0
    kf = L1._keyer("strict")
    for t in store.values():
        p, g = get(t)
        pc, gc = L1._counter(p, kf), L1._counter(g, kf)
        tp += sum((pc & gc).values()); pn += sum(pc.values()); gn += sum(gc.values())
    r = L1._prf(tp, pn, gn)
    print(f"  {name:22s} {r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f}")

# ---- is the Mark-vs-Davnah agent gap consistent across models? -------------
print("\n=== per-sentence paired test: does the agent really prefer Mark? ===")
from scipy import stats
groups = {}
for model, _c, _t, p in REGISTRY:
    if model == "Qwen3.5 (legacy)" or not Path(p).exists():
        continue
    groups.setdefault(model, []).append(p)
print(f"  {'Model':22s} {'d(Mark-Dav)':>12s} {'95% CI':>18s} {'p':>8s}")
for model, paths in groups.items():
    stores = [X.load(p, canon=True, empties="zero") for p in paths]
    shared = set(stores[0])
    for s in stores[1:]:
        shared &= set(s)
    d = []
    for k in sorted(shared):
        vals = [X.f1(s[k][0], s[k][1]) - X.f1(s[k][0], s[k][2]) for s in stores]
        d.append(sum(vals) / len(vals))
    d = np.array(d)
    m, se = d.mean(), stats.sem(d)
    crit = stats.t.ppf(0.975, len(d) - 1)
    p = stats.ttest_1samp(d, 0.0).pvalue
    print(f"  {model:22s} {m:+12.3f} [{m-crit*se:+.3f}, {m+crit*se:+.3f}] {p:8.4f}")
