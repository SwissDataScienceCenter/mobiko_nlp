#!/usr/bin/env python
"""
Audit: is the RQ1 path (scripts/xrun) computing the same thing as the RQ2 path
(src/multi_agent_annotation/evaluation)?

Both halves of the paper are scored from the same run files, but through two
different stacks:

  RQ1  xrun_analysis -> eval_layer1_output          strict offsets + type
  RQ2  eval_logprob_uncertainty -> eval_layer2_correlation   normalised span text

This script measures every place they could diverge, so the paper can state the
conventions instead of assuming they match. Run it after any change to either
stack.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import xrun_analysis as X            # noqa: E402
import eval_layer1_output as L1      # noqa: E402
import eval_layer2_correlation as L2  # noqa: E402
from aggregate import REGISTRY       # noqa: E402
from deliberation_history import load_records, norm  # noqa: E402
from eval_utils import permutation_p_rho  # noqa: E402

PROBE = ["  Foo   BAR ", "A\tB\nC", "Ünïcode  X", ""]


def check_normalisers():
    """Both stacks must reach text and labels through label_norm, not copies."""
    import label_norm
    print("1. one definition of text and label normalisation")
    same = (L1._normalize is L2._normalize is norm is label_norm.normalize_text)
    print(f"   L1._normalize is L2._normalize is deliberation_history.norm"
          f" is label_norm.normalize_text : {same}")
    assert same, "a copy of the sentence normaliser has come back"

    print("\n1b. both stacks canonicalise labels identically")
    import eval_logprob_uncertainty as LP
    cases = ["BIOTIC_ENTITY", "biotic  entity", " Biotic-Entity ",
             "BIOTIC COLLECTIVE ENTITY", "QUANTITATIVEPROPERTY"]
    for raw in cases:
        rq1 = L1._normalize_label(raw)                      # RQ1 path
        rq2 = LP._typed([{"text": "x", "type": raw}])[0]["type"]   # RQ2 path
        flag = "ok " if rq1 == rq2 else "MISMATCH"
        print(f"   {flag} {raw!r:30s} -> {rq1!r}")
        assert rq1 == rq2, (raw, rq1, rq2)
    print("   as-is convention still available: "
          f"{L1._normalize_label('BIOTIC_ENTITY', canon=False)!r}")


def check_sentence_sets(run):
    print("\n2. sentence sets")
    rq1 = X.load(run, canon=True, empties="zero")
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): a for s, a in human.items()}
    rq2 = {norm(r.get("sentence", "")) for r in load_records(run)
           if (a := hnorm.get(norm(r.get("sentence", "")))) and len(a) >= 2}
    print(f"   RQ1 (both annotators non-empty, empties scored 0) n = {len(rq1)}")
    print(f"   RQ2 (both annotators present)                     n = {len(rq2)}")
    print(f"   RQ2 \\ RQ1 = {len(rq2 - set(rq1))}   RQ1 \\ RQ2 = {len(set(rq1) - rq2)}")
    print("   RQ1 requires non-EMPTY annotations; RQ2 only requires both present")


def check_offsets_and_relations():
    print("\n3. entity dicts: offsets and RELATION pseudo-spans")
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    ents = [e for a in human.values() for ann in a.values() for e in ann["entities"]]
    off = sum(1 for e in ents if e.get("start_char") is not None)
    rel = sum(1 for e in ents
              if (e.get("entity_type") or e.get("type") or "").upper().startswith("RELATION"))
    print(f"   human entities in the L2 path carrying start_char: {off}/{len(ents)}")
    print("   -> RQ2 CANNOT match on offsets; text matching is forced, not chosen")
    print(f"   RELATION pseudo-spans in the L2-loaded human side: {rel}")
    print("   L1._norm_entity drops RELATION spans; L2's loader does not."
          " Currently 0, so latent.")


def check_graded_vs_binary():
    print("\n4. graded vs binary disagreement target (MoBiKo has 2 annotators)")
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    tot = same = 0
    for a in human.values():
        if len(a) < 2:
            continue
        for info in L2.compute_human_span_disagreements(a).values():
            tot += 1
            g, _ = L2.graded_disagreement(info)
            same += int((g > 0) == info["type_disagreed"])
    print(f"   graded>0 agrees with binary type_disagreed on {same}/{tot} spans")
    print("   -> monotone at R=2, so Spearman is identical; the MoBiKo binary target"
          " and the CoNLL graded target are the same measure")


def check_span_dropping():
    print("\n5. span-level RQ2 drops agent spans neither annotator marked.")
    print("   Is the dropped set biased on the predictor?\n")
    human = L2.load_all_human_annotations([X.MARK, X.DAV])
    hnorm = {norm(s): a for s, a in human.items()}
    print(f"   {'run':24s} {'spans':>6s} {'kept':>6s} {'drop%':>6s} "
          f"{'lp kept':>9s} {'lp dropped':>11s} {'perm p':>8s}")
    for _model, _cond, tag, path in REGISTRY:
        if not Path(path).exists():
            continue
        kept, dropped = [], []
        for r in load_records(path):
            a = hnorm.get(norm(r.get("sentence", "")))
            if not a or len(a) < 2:
                continue
            dis = L2.compute_human_span_disagreements(a)
            for e in (r.get("annotator_entity_logprobs") or []):
                lp = e.get("type_mean_logprob")
                if lp is None:
                    continue
                (kept if dis.get(L2._normalize(e.get("text", ""))) is not None
                 else dropped).append(lp)
        n = len(kept) + len(dropped)
        if not (kept and dropped):
            print(f"   {tag:24s} {n:6d} {len(kept):6d} {'-':>6s}")
            continue
        p = permutation_p_rho(kept + dropped,
                              [0.0] * len(kept) + [1.0] * len(dropped), n_boot=2000)
        print(f"   {tag:24s} {n:6d} {len(kept):6d} {len(dropped)/n:6.1%} "
              f"{st.mean(kept):9.4f} {st.mean(dropped):11.4f} {p:8.4f}")


def main():
    run = ROOT / "data/auto_annotated/datademo_manually_labeled_qwen3-35B-vllm_v3"
    check_normalisers()
    check_sentence_sets(run)
    check_offsets_and_relations()
    check_graded_vs_binary()
    check_span_dropping()


if __name__ == "__main__":
    main()
