#!/usr/bin/env python
"""Does deliberation help? Annotator's proposal vs the adjudicated output.

The paper is titled multi-agent deliberation but never shows the Critic and
Adjudicator earn their cost. This measures it WITHIN each run, needing no new
model calls and no second run to compare against: every record already stores the
Annotator's own proposal (annotator_entity_logprobs) alongside the final
adjudicated labels (final_entities). Same sentence, same model, same guideline,
same everything -- the only difference is whether the other two agents have acted
yet. That makes it a true ablation rather than a comparison of two systems.

MATCHING MODE. The annotator record carries span text and type but no character
offsets, so strict F1 (offsets + type) cannot be computed on that side. Both arms
are scored in text_type mode. Those numbers run HIGHER than the strict F1 reported
elsewhere in the paper and are not comparable to it; only the within-row
difference is meaningful.

EMPTY OUTPUTS are scored F1=0 on whichever side they occur, matching the paper's
convention. A record where the annotator proposed nothing and the pipeline also
produced nothing contributes 0 to both arms and so does not move the contrast.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "multi_agent_annotation"))
sys.path.insert(0, str(ROOT / "src" / "multi_agent_annotation" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_layer1_output as L1                                   # noqa: E402
from xrun_analysis import MARK, DAV, _canon_type, paired_block    # noqa: E402

MODE = "text_type"


def _f1(pred, gold):
    v = L1.sentence_f1(pred, gold, MODE)
    return 0.0 if v is None else v


def _drop_nested(ents):
    """The pipeline's own nested-entity rule, applied symmetrically.

    multi_agent_annotation_ag2._remove_nested_entities drops any span whose text
    is strictly contained in a longer co-annotated span ("species" inside "species
    richness"). It runs AFTER adjudication, on the final output only, so comparing
    raw annotator spans against post-processed final spans credits deliberation
    with a deterministic filter it did not perform. On one run that filter alone
    accounts for 17 of the differing spans on records the adjudicator never
    touched -- every one in the same direction. Both arms get it here, so the
    contrast isolates the Critic and Adjudicator.

    Text-based containment, matching the pipeline's own fallback when offsets are
    absent (the annotator record has none).
    """
    keep, texts = [], [(e["text"] or "").lower().strip() for e in ents]
    for i, e in enumerate(ents):
        ti = texts[i]
        if any(ti != tj and ti in tj for j, tj in enumerate(texts) if i != j):
            continue
        keep.append(e)
    return keep


def _norm(raw, canon):
    out = []
    for e in raw or []:
        d = {"text": e.get("text", ""),
             "entity_type": e.get("entity_type") or e.get("type") or ""}
        if canon:
            d["entity_type"] = _canon_type(d["entity_type"])
        n = L1._norm_entity(d)
        if n:
            out.append(n)
    return out


def _records(path: Path):
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def measure(path: Path, canon: bool = True):
    human, keys = L1.load_all_human_annotations([MARK, DAV])
    mk_k, dv_k = keys[0], keys[1]
    solo, pipe, ceil = [], [], []
    n_no_proposal = n_empty_final = 0
    seen = set()
    for rec in _records(path):
        key = L1._normalize(rec.get("sentence", ""))
        if key in seen:
            continue
        anns = human.get(key)
        if not anns or not all(anns.get(n) and anns[n]["entities"] for n in (mk_k, dv_k)):
            continue
        seen.add(key)
        mk, dv = anns[mk_k]["entities"], anns[dv_k]["entities"]
        a = _drop_nested(_norm(rec.get("annotator_entity_logprobs"), canon))
        f = _drop_nested(_norm(rec.get("final_entities"), canon))
        # An absent proposal alongside a non-empty final output is MISSING
        # MEASUREMENT, not an observed empty annotator. Scoring it 0 would hand
        # the pipeline a free win on a record where we simply failed to capture
        # what the annotator said. Excluded. A record empty on BOTH sides is a
        # genuine empty and stays, contributing 0 to each arm.
        if not a and f:
            n_no_proposal += 1
            continue
        n_empty_final += (not f)
        solo.append((_f1(a, mk) + _f1(a, dv)) / 2.0)
        pipe.append((_f1(f, mk) + _f1(f, dv)) / 2.0)
        ceil.append(_f1(mk, dv))
    if len(solo) < 3:
        return None
    solo, pipe = np.array(solo), np.array(pipe)
    d = paired_block(pipe - solo)
    return dict(n=len(solo), solo=float(solo.mean()), pipe=float(pipe.mean()),
                ceiling=float(np.mean(ceil)), eff=d,
                no_proposal=n_no_proposal, empty_final=n_empty_final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, help="one run dir/file; omit for all runs")
    ap.add_argument("--as-is", action="store_true", help="do not canonicalise labels")
    args = ap.parse_args()
    canon = not args.as_is

    if args.run:
        targets = [("run", args.run)]
    else:
        from aggregate import REGISTRY
        targets = [(t, p) for _m, _c, t, p in REGISTRY if Path(p).exists()]

    print("=" * 96)
    print("  DELIBERATION ABLATION — annotator proposal vs adjudicated output, same record")
    print(f"  mode: {MODE}, labels {'canonicalised' if canon else 'as-is'}; "
          "NOT comparable to strict F1 elsewhere")
    print("=" * 96)
    print(f"  {'run':24s} {'n':>4s} {'solo':>6s} {'pipe':>6s} {'ceil':>6s} "
          f"{'effect':>8s} {'95% CI':>18s} {'p':>7s}  {'noprop':>6s}")
    rows = []
    for tag, path in targets:
        r = measure(Path(path), canon=canon)
        if not r:
            print(f"  {tag:24s} <no scoreable overlap>")
            continue
        rows.append((tag, r))
        d = r["eff"]
        print(f"  {tag:24s} {r['n']:4d} {r['solo']:6.3f} {r['pipe']:6.3f} "
              f"{r['ceiling']:6.3f} {d['mean']:+8.3f} "
              f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] {d['t_p']:7.3f}  "
              f"{r['no_proposal']:6d}")
    if len(rows) > 1:
        pos = [t for t, r in rows if r["eff"]["t_p"] < 0.05 and r["eff"]["mean"] > 0]
        neg = [t for t, r in rows if r["eff"]["t_p"] < 0.05 and r["eff"]["mean"] < 0]
        eff = np.array([r["eff"]["mean"] for _t, r in rows])
        print(f"\n  {len(rows)} runs: {len(pos)} significantly POSITIVE, "
              f"{len(neg)} significantly NEGATIVE, "
              f"{len(rows)-len(pos)-len(neg)} not resolved")
        print(f"  effect range {eff.min():+.3f} to {eff.max():+.3f}, mean {eff.mean():+.3f}")
        print("  noprop = missing annotator proposal with non-empty final output; excluded")


if __name__ == "__main__":
    main()
