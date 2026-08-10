#!/usr/bin/env python
"""Harvest headline per-run metrics from the stored reports + raw JSONL."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
D = ROOT / "data"
XR = ROOT / "output/eval_reports/cross_run_comparison"

# Runs are owned by aggregate.REGISTRY — see the note in driver.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from aggregate import REGISTRY  # noqa: E402

RUNS = {d: p for _, _, d, p in REGISTRY}
RUNS["apertus15_70_v1"] = None   # raw data truncated; reports only, not in REGISTRY

SCHEMA = {
    "BIOTIC ENTITY", "ABIOTIC ENTITY", "ANTHROPOGENIC ENTITY", "SPATIAL ENTITY",
    "TEMPORAL ENTITY", "CONCEPT",
    "BIOTIC PROPERTY", "ABIOTIC PROPERTY", "ANTHROPOGENIC PROPERTY",
    "SPATIAL PROPERTY", "TEMPORAL PROPERTY",
    "QUALITATIVE PROPERTY", "QUANTITATIVE PROPERTY",
    "BIOTIC PROCESS", "ABIOTIC PROCESS", "ANTHROPOGENIC PROCESS",
}

print("=" * 100)
print("LAYER 1 HEADLINES  (from stored layer1.json)")
print("=" * 100)
hdr = ("run", "evalN", "sMark", "sDav", "bMark", "bDav", "kMark", "kDav")
print("  {:20s} {:>5s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}".format(*hdr))
for name in RUNS:
    f = XR / name / "layer1.json"
    if not f.exists():
        print(f"  {name:20s} -- no layer1.json")
        continue
    j = json.load(open(f))
    refs = j["references"]
    mk, dv = "Mark", "Davnah"
    ec = j["eval_sentence_counts"]
    lo = j["label_only"]
    print("  {:20s} {:5d} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(
        name, ec[mk],
        refs[mk]["strict"]["f1"], refs[dv]["strict"]["f1"],
        refs[mk]["boundary"]["f1"], refs[dv]["boundary"]["f1"],
        lo[mk]["cohen_kappa"], lo[dv]["cohen_kappa"]))
    ci = refs[mk]["strict"].get("f1_ci")
    print(f"       strict-Mark 95% CI {ci}   ceiling strict "
          f"{j['inter_human']['Mark|Davnah']['strict']['f1']:.3f} "
          f"boundary {j['inter_human']['Mark|Davnah']['boundary']['f1']:.3f} "
          f"kappa {lo['Mark ↔ Davnah']['cohen_kappa']:.3f}")

print()
print("=" * 100)
print("GUIDELINE GROUNDING  (from guideline.json / .txt)")
print("=" * 100)
for name in RUNS:
    f = XR / name / "guideline.txt"
    if not f.exists():
        continue
    txt = f.read_text()
    hits = re.findall(r"(grounded|grounding|adherence)[^\n]*", txt, re.I)[:3]
    print(f"  {name:20s} " + " | ".join(h.strip()[:80] for h in hits))

print()
print("=" * 100)
print("RAW-FILE HYGIENE + COUNTS")
print("=" * 100)
for name, path in RUNS.items():
    if path is None or not path.exists():
        print(f"  {name:20s} (raw data unavailable)")
        continue
    n = ents = rels = empty = 0
    types = Counter()
    fails = Counter()
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        n += 1
        fe = rec.get("final_entities") or []
        ents += len(fe)
        rels += len(rec.get("final_relations") or [])
        if not fe:
            empty += 1
        for e in fe:
            types[(e.get("entity_type") or e.get("type") or "").strip()] += 1
        blob = json.dumps(rec)
        for tag in ("adjudicator_parse_failed", "annotator_parse_failed",
                    "critic_parse_failed", "critic_missing_fallback"):
            if tag in blob:
                fails[tag] += 1
    offschema = {t: c for t, c in types.items() if t not in SCHEMA}
    print(f"  {name:20s} recs={n:3d} ents={ents:5d} rels={rels:4d} empty={empty:3d} "
          f"types={len(types):2d} off-schema spans={sum(offschema.values()):3d}")
    if offschema:
        print("       off-schema:", ", ".join(f"{t!r}x{c}" for t, c in
                                              sorted(offschema.items(), key=lambda kv: -kv[1])))
    if fails:
        print("       parse:", dict(fails))
