#!/usr/bin/env python
"""P3 scorer — does SAMPLING variability predict human type-disagreement?

Pairs each agent span against the two humans exactly as
eval_logprob_uncertainty.span_level does (same normalisation, same
compute_human_span_disagreements, same _group_compare), so the rho printed here
is directly comparable to the span:logprob / span:entropy point-biserials
already in the RQ2 tables (+0.158 … +0.272 where headroom exists, null where not).

Uncertainty measures, per span, over the K annotator samples:
  typing    1 - modal_label_frequency, among samples that DETECTED the span.
            The direct analogue of type-logprob uncertainty.
  entropy   Shannon entropy (nats) of the empirical label distribution.
  detection 1 - detection_rate, i.e. how often the span was missed. Reported
            against human DETECTION disagreement, a different target.

Typing spans are restricted to those detected in at least --min-detections
samples: a label distribution over 1 or 2 observations is noise, and a span
detected once is a detection event, not a typing decision.

THE DIAGNOSTIC THAT DECIDES THE EXPERIMENT is printed first. If almost every
span is unanimous (modal frequency K/K), the model has no sampling variability
to correlate and the test cannot run for that model — which is itself the
answer: the signal is model-dependent, not headroom-dependent.

Usage:
    python scripts/xrun/selfconsistency_score.py --samples <file.jsonl>
    python scripts/xrun/selfconsistency_score.py --self-test   # no LLM, no data
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "src" / "multi_agent_annotation",
           _REPO / "src" / "multi_agent_annotation" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_logprob_uncertainty as LP  # noqa: E402
import eval_layer1_output as L1  # noqa: E402

MARK = _REPO / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
DAV = _REPO / "data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl"


def dedupe_records(records):
    """One record per sentence — the copy with the most usable samples.

    Shard files are opened in APPEND mode under --resume, so a shard launched
    twice (or whose sentence range moved between launches) can end up holding the
    same sentence more than once; `cat`-ing the shards then carries the duplicates
    into the merge. This is not cosmetic: repeated sentences contribute their spans
    twice, and duplicated observations are not independent, so they inflate n and
    bias both rho and p. The qwen K=10 run merged 30 records from 28 sentences
    exactly this way.

    Deduping in the scorer rather than at merge time means any merged file is
    scored correctly however it was produced — including the ones already written.

    Returns (deduped records in first-seen order, [(sentence, times_seen), ...]).
    """
    best, order, seen = {}, [], Counter()
    for rec in records:
        key = rec.get("sentence", "")
        seen[key] += 1
        usable = sum(1 for s in rec.get("samples", []) if s is not None)
        if key not in best:
            best[key] = (usable, rec)
            order.append(key)
        elif usable > best[key][0]:
            best[key] = (usable, rec)
    return [best[k][1] for k in order], [(k, n) for k, n in seen.items() if n > 1]


def span_distributions(samples, min_detections):
    """{normalised span text: {labels: Counter, n_detected, k_usable}}"""
    usable = [s for s in samples if s is not None]
    k = len(usable)
    out = {}
    for sample in usable:
        seen = set()
        for ent in sample:
            key = L1._normalize(ent.get("text", ""))
            if not key or key in seen:
                continue          # count a span once per sample
            seen.add(key)
            rec = out.setdefault(key, {"labels": Counter(), "n_detected": 0,
                                       "top1": [], "maxent": []})
            rec["labels"][ent.get("entity_type", "")] += 1
            rec["n_detected"] += 1
            # reported confidence for THIS pass, when the sampler captured it
            if ent.get("top1_prob") is not None:
                rec["top1"].append(ent["top1_prob"])
            if ent.get("type_max_entropy") is not None:
                rec["maxent"].append(ent["type_max_entropy"])
    for rec in out.values():
        rec["k_usable"] = k
        rec["detection_rate"] = rec["n_detected"] / k if k else 0.0
        n = rec["n_detected"]
        modal = max(rec["labels"].values()) if rec["labels"] else 0
        rec["modal_freq"] = modal / n if n else 0.0
        rec["typing_uncertainty"] = 1.0 - rec["modal_freq"]
        rec["label_entropy"] = -sum(
            (c / n) * math.log(c / n) for c in rec["labels"].values() if c) if n else 0.0
        rec["usable_for_typing"] = n >= min_detections
        rec["mean_top1"] = (sum(rec["top1"]) / len(rec["top1"])) if rec["top1"] else None
        rec["mean_maxent"] = (sum(rec["maxent"]) / len(rec["maxent"])) if rec["maxent"] else None
    return out


def score(records, human, min_detections):
    hnorm = {LP.norm(s): a for s, a in human.items()}
    typing_pairs, entropy_pairs, det_pairs = [], [], []
    modal_freqs, n_spans, n_sent = [], 0, 0
    mech = []      # (flipped_across_samples, mean reported top-1, mean max-entropy)
    for rec in records:
        anns = hnorm.get(LP.norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        n_sent += 1
        dis = LP.compute_human_span_disagreements(anns)
        dists = span_distributions(rec.get("samples", []), min_detections)
        for key, d in dists.items():
            info = dis.get(key)
            if info is None:
                continue          # span not located by either human
            n_spans += 1
            det_pairs.append((1.0 - d["detection_rate"],
                              bool(info.get("presence_disagreed", False))))
            if not d["usable_for_typing"]:
                continue
            modal_freqs.append(d["modal_freq"])
            if d["mean_top1"] is not None:
                mech.append((d["modal_freq"] < 0.999, d["mean_top1"], d["mean_maxent"]))
            disagreed = bool(info["type_disagreed"])
            typing_pairs.append((d["typing_uncertainty"], disagreed))
            entropy_pairs.append((d["label_entropy"], disagreed))
    return {
        "n_sentences": n_sent,
        "n_matched_spans": n_spans,
        "n_typing_spans": len(typing_pairs),
        "modal_freqs": modal_freqs,
        "mechanism": mech,
        "typing": LP._group_compare(typing_pairs, higher_is_uncertain=True),
        "entropy": LP._group_compare(entropy_pairs, higher_is_uncertain=True),
        "detection": LP._group_compare(det_pairs, higher_is_uncertain=True),
    }


def report(res, k_hint=None):
    mf = res["modal_freqs"]
    print("=" * 78)
    print("  DIAGNOSTIC — is there any sampling variability to correlate?")
    print("=" * 78)
    if not mf:
        print("  No spans usable for typing. Either K is too small or no agent span")
        print("  was located by both humans. Nothing can be concluded.")
        return
    unanimous = sum(1 for x in mf if x >= 0.999) / len(mf)
    print(f"  spans scored for typing        {len(mf)}")
    print(f"  unanimous across samples       {100*unanimous:.1f}%")
    print(f"  mean modal-label frequency     {sum(mf)/len(mf):.4f}")
    print(f"  spans with modal freq < 0.80   "
          f"{100*sum(1 for x in mf if x < 0.80)/len(mf):.1f}%")
    if unanimous > 0.95:
        print()
        print("  >>> ESSENTIALLY NO VARIABILITY. This model reproduces the same label")
        print("      on >95% of spans across independent samples, so there is nothing")
        print("      for the uncertainty signal to track. That is a RESULT, not a")
        print("      failure: for this model the signal is unmeasurable by any")
        print("      instrument, and the RQ2 finding must be reported as")
        print("      model-dependent rather than headroom-dependent.")
    print()
    print("=" * 78)
    print("  SIGNAL — sampling uncertainty vs human disagreement")
    print("  (rho comparable to span:logprob / span:entropy in the RQ2 tables)")
    print("=" * 78)
    print(f"  {'measure':28s} {'rho':>8s} {'p':>9s} {'n':>6s}  target")
    for name, key, target in (
        ("typing (1 - modal freq)", "typing", "human type-disagreement"),
        ("typing (label entropy)", "entropy", "human type-disagreement"),
        ("detection (1 - det rate)", "detection", "human span-disagreement"),
    ):
        g = res[key] or {}
        rho, p = g.get("point_biserial_rho"), g.get("point_biserial_p")
        n = res["n_typing_spans"] if key != "detection" else res["n_matched_spans"]
        rs = f"{rho:+.3f}" if isinstance(rho, (int, float)) else "   n/a"
        ps = f"{p:.4f}" if isinstance(p, (int, float)) else "     n/a"
        print(f"  {name:28s} {rs:>8s} {ps:>9s} {n:6d}  {target}")
    print()
    print(f"  sentences paired with both humans: {res['n_sentences']}")

    mech = res.get("mechanism") or []
    if mech:
        flip = [m for m in mech if m[0]]
        same = [m for m in mech if not m[0]]
        print()
        print("=" * 78)
        print("  MECHANISM — what did the model REPORT on spans it actually flipped?")
        print("=" * 78)
        print("  If the reported top-1 probability is ~1.0 even on spans where the")
        print("  label demonstrably changed between samples, then token-level logprobs")
        print("  are not measuring decision uncertainty: the label is already settled")
        print("  by the reasoning before that token is emitted, so its probability is")
        print("  ~1.0 whichever label the reasoning happened to land on.")
        print()
        print(f"  {'spans':28s} {'n':>5s} {'mean top-1':>11s} {'mean max-entropy':>17s}")
        for name, grp in (("FLIPPED across samples", flip),
                          ("unanimous across samples", same)):
            if not grp:
                print(f"  {name:28s} {0:5d}          --                --")
                continue
            t1 = sum(m[1] for m in grp) / len(grp)
            me = [m[2] for m in grp if m[2] is not None]
            ms = f"{sum(me)/len(me):17.5f}" if me else f"{'--':>17s}"
            print(f"  {name:28s} {len(grp):5d} {t1:11.5f} {ms}")
        if flip:
            t1f = sum(m[1] for m in flip) / len(flip)
            if t1f > 0.99:
                print()
                print("  >>> CONFIRMED. Spans the model flipped were reported at top-1 "
                      f"{t1f:.5f}.")
                print("      The logprob cannot see this uncertainty. Saturation "
                      "measures how")
                print("      early the decision is settled, NOT how certain the model is.")


# ── offline self-test: validates the logic with no LLM and no data ──────────
def self_test():
    """Synthetic samples with a KNOWN relationship, so the scorer can be trusted
    before any LLM budget is spent. Span A is unanimous, span B is split; only B
    is marked as a human type-disagreement, so uncertainty must track it."""
    K = 10
    def s(label_a, label_b):
        return [{"text": "alpha", "entity_type": label_a},
                {"text": "beta", "entity_type": label_b}]
    samples = [s("BIOTIC ENTITY", "BIOTIC ENTITY" if i < 5 else "BIOTIC PROPERTY")
               for i in range(K)]
    dists = span_distributions(samples, min_detections=5)
    a, b = dists["alpha"], dists["beta"]
    assert a["modal_freq"] == 1.0, a
    assert a["typing_uncertainty"] == 0.0, a
    assert abs(b["modal_freq"] - 0.5) < 1e-9, b
    assert abs(b["typing_uncertainty"] - 0.5) < 1e-9, b
    assert abs(b["label_entropy"] - math.log(2)) < 1e-9, b
    assert a["detection_rate"] == 1.0 and b["detection_rate"] == 1.0
    # a span seen in only 2 of 10 samples is a detection event, not a typing one
    partial = [s("BIOTIC ENTITY", "BIOTIC ENTITY") if i < 2
               else [{"text": "alpha", "entity_type": "BIOTIC ENTITY"}] for i in range(K)]
    d2 = span_distributions(partial, min_detections=5)
    assert d2["beta"]["usable_for_typing"] is False, d2["beta"]
    assert abs(d2["beta"]["detection_rate"] - 0.2) < 1e-9, d2["beta"]
    # None samples (parse failures) must not count toward K
    d3 = span_distributions([samples[0], None, samples[1]], min_detections=1)
    assert d3["alpha"]["k_usable"] == 2, d3["alpha"]
    # duplicate records collapse to one, keeping the copy with most usable samples
    full = {"sentence": "s1", "samples": samples}
    partial = {"sentence": "s1", "samples": samples[:3] + [None] * 7}
    other = {"sentence": "s2", "samples": samples}
    ded, dups = dedupe_records([partial, full, other])
    assert len(ded) == 2, ded
    assert ded[0]["samples"] is samples, "kept the partial copy over the complete one"
    assert dups == [("s1", 2)], dups
    ded2, dups2 = dedupe_records([other, full])   # a clean file is left alone
    assert len(ded2) == 2 and dups2 == [], (ded2, dups2)
    print("self-test OK — span distributions, thresholds, entropy, "
          "parse-failure handling and record dedupe all behave as specified")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--min-detections", type=int, default=5,
                    help="min samples a span must appear in to be scored for typing")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.samples:
        ap.error("--samples is required (or use --self-test)")

    records = [json.loads(l) for l in args.samples.read_text().splitlines()
               if l.strip().startswith("{")]
    records, dups = dedupe_records(records)
    if dups:
        n_dropped = sum(n - 1 for _, n in dups)
        print(f"!! DROPPED {n_dropped} duplicate record(s) across {len(dups)} "
              f"sentence(s) — scoring {len(records)} unique sentences.")
        print("   (a shard file was appended to more than once; see dedupe_records)")
        for s, n in dups:
            print(f"     x{n}  {s[:70]!r}")
        print()
    human, names = L1.load_all_human_annotations([MARK, DAV])
    res = score(records, human, args.min_detections)
    meta = records[0].get("run_meta", {}) if records else {}
    print(f"samples: {args.samples}")
    print(f"model:   {meta.get('annotator_model')}  t={meta.get('annotator_temperature')}  "
          f"K={meta.get('selfconsistency_k')}")
    print()
    report(res)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dump = {k: v for k, v in res.items() if k != "modal_freqs"}
        dump["run_meta"] = meta
        args.output.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
