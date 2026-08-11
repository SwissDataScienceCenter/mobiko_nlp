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
from random import Random

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "src" / "multi_agent_annotation",
           _REPO / "src" / "multi_agent_annotation" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_logprob_uncertainty as LP  # noqa: E402
import eval_layer1_output as L1  # noqa: E402

import eval_layer2_correlation as L2  # noqa: E402

# MoBiKo defaults. Overridable with --human so the same scorer runs on any corpus
# whose annotations are in the per-annotator JSONL shape (CoNLL-MTurk included,
# where one file carries all annotators via the per-record "annotator" field).
MARK = _REPO / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
DAV = _REPO / "data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl"


def auc_lower(flipped, unanimous, n_boot=LP.N_BOOT, seed=LP.SEED):
    """P(a flipped span is reported LESS confident than a unanimous one), + perm p.

    The mechanism question is a DISCRIMINATION question — can the reported number
    tell the two groups apart — so score it as one. Comparing group MEANS cannot:
    on a saturated model both means round to ~1.0 and the difference lands in the
    4th decimal, which looks like "no signal" whether or not one exists.

    0.5 = the report is uninformative about flipping; 1.0 = perfect separation.
    Ties score 0.5, so this equals Mann-Whitney U/(n1*n2) under mid-ranks.

    p is a permutation test on the group labels, matching permutation_p_rho's
    convention (same N_BOOT and SEED) so it sits on the same footing as the rho
    values printed above it rather than importing a second statistical toolkit.
    """
    if not flipped or not unanimous:
        return None, None

    def _auc(fl, un):
        wins = sum((1.0 if u > f else 0.5 if u == f else 0.0)
                   for f in fl for u in un)
        return wins / (len(fl) * len(un))

    obs = _auc(flipped, unanimous)
    pool = list(flipped) + list(unanimous)
    n_f = len(flipped)
    rng = Random(seed)
    count = 0
    for _ in range(n_boot):
        rng.shuffle(pool)
        if _auc(pool[:n_f], pool[n_f:]) >= obs:
            count += 1
    return obs, (count + 1) / (n_boot + 1)


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


def _correlate(pairs, target):
    """Uncertainty vs human disagreement, one dict shape for both target types.

    Binary keeps the existing point-biserial path so published MoBiKo numbers do
    not move. Graded uses _spearman_rho + permutation_p_rho — the same tie-correct
    machinery already behind the RQ2 tables, so the two remain comparable. The
    key names are shared deliberately; "measure" says which was used.
    """
    if target != "graded":
        out = LP._group_compare(pairs, higher_is_uncertain=True)
        if out is not None:
            out["measure"] = "point_biserial"
        return out
    xs = [float(v) for v, _ in pairs if v is not None]
    ys = [float(t) for v, t in pairs if v is not None]
    if len(xs) < 3:
        return {"point_biserial_rho": None, "point_biserial_p": None,
                "n": len(xs), "measure": "spearman"}
    return {
        "point_biserial_rho": LP._spearman_rho(xs, ys),
        "point_biserial_p": LP.permutation_p_rho(xs, ys, n_boot=LP.N_BOOT),
        "n": len(xs),
        "mean_uncertainty": sum(xs) / len(xs),
        "mean_target": sum(ys) / len(ys),
        "measure": "spearman",
    }


def score(records, human, min_detections, target="binary", min_span_annotators=1):
    """target: 'binary' (MoBiKo, fixed 2 annotators) or 'graded' (variable count).

    See L2.graded_disagreement for why a corpus with a variable annotator count
    must not use the binary target. min_span_annotators drops spans marked by
    fewer than that many annotators: on CoNLL 35% of union spans are marked by
    exactly one of five, and those are largely idiosyncratic rather than
    contested, so including them changes what "disagreement" means.
    """
    hnorm = {LP.norm(s): a for s, a in human.items()}
    typing_pairs, entropy_pairs, det_pairs = [], [], []
    modal_freqs, n_spans, n_sent = [], 0, 0
    n_dropped_singleton = 0
    ann_counts = Counter()   # annotators per SCORED sentence, not per loaded one
    mech = []      # (flipped_across_samples, mean reported top-1, mean max-entropy)
    for rec in records:
        anns = hnorm.get(LP.norm(rec.get("sentence", "")))
        if not anns or len(anns) < 2:
            continue
        n_sent += 1
        ann_counts[len(anns)] += 1
        dis = LP.compute_human_span_disagreements(anns)
        dists = span_distributions(rec.get("samples", []), min_detections)
        for key, d in dists.items():
            info = dis.get(key)
            if info is None:
                continue          # span not located by any human
            if info.get("present_in", 0) < min_span_annotators:
                n_dropped_singleton += 1
                continue
            n_spans += 1
            grad_type, grad_pres = L2.graded_disagreement(info)
            det_target = (grad_pres if target == "graded"
                          else bool(info.get("presence_disagreed", False)))
            det_pairs.append((1.0 - d["detection_rate"], det_target))
            if not d["usable_for_typing"]:
                continue
            modal_freqs.append(d["modal_freq"])
            if d["mean_top1"] is not None:
                mech.append((d["modal_freq"] < 0.999, d["mean_top1"], d["mean_maxent"]))
            typ_target = grad_type if target == "graded" else bool(info["type_disagreed"])
            typing_pairs.append((d["typing_uncertainty"], typ_target))
            entropy_pairs.append((d["label_entropy"], typ_target))
    # Discrimination of the REPORTED confidence, computed here rather than in
    # report() so the number lands in the saved JSON and can be cited directly.
    fl, un = [m for m in mech if m[0]], [m for m in mech if not m[0]]
    fl_t1, un_t1 = [m[1] for m in fl], [m[1] for m in un]
    fl_me = [m[2] for m in fl if m[2] is not None]
    un_me = [m[2] for m in un if m[2] is not None]
    auc_t1, p_t1 = auc_lower(fl_t1, un_t1)
    # entropy runs the other way: higher = less certain, so swap the groups
    auc_me, p_me = auc_lower(un_me, fl_me) if (fl_me and un_me) else (None, None)
    return {
        "n_sentences": n_sent,
        "n_matched_spans": n_spans,
        "n_typing_spans": len(typing_pairs),
        "modal_freqs": modal_freqs,
        "mechanism": mech,
        "mechanism_auc": {
            "n_flipped": len(fl), "n_unanimous": len(un),
            "top1_auc": auc_t1, "top1_perm_p": p_t1,
            "maxentropy_auc": auc_me, "maxentropy_perm_p": p_me,
            "top1_mean_gap": ((sum(un_t1) / len(un_t1)) - (sum(fl_t1) / len(fl_t1)))
                             if (fl_t1 and un_t1) else None,
        },
        "target": target,
        "annotators_per_scored_sentence": dict(sorted(ann_counts.items())),
        "min_span_annotators": min_span_annotators,
        "n_dropped_below_min_annotators": n_dropped_singleton,
        "typing": _correlate(typing_pairs, target),
        "entropy": _correlate(entropy_pairs, target),
        "detection": _correlate(det_pairs, target),
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
        print("  Sampling shows which spans the model is actually unsure about. The")
        print("  question here is whether the REPORTED confidence knew: can top-1 tell")
        print("  a span the model flipped from one it repeated K/K?")
        print()
        print("  Read AUC, not the group means. On a saturated model both means round")
        print("  to ~1.0 and their difference sits in the 4th decimal, which reads as")
        print("  'no signal' whether or not one is there. AUC asks the discrimination")
        print("  question directly: 0.5 = the report is uninformative, 1.0 = perfect.")
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
        disc = res.get("mechanism_auc") or {}
        auc, p = disc.get("top1_auc"), disc.get("top1_perm_p")
        if auc is not None:
            auc_e, p_e = disc.get("maxentropy_auc"), disc.get("maxentropy_perm_p")
            spread = disc.get("top1_mean_gap")
            print()
            print(f"  {'discrimination':28s} {'AUC':>11s} {'perm p':>17s}")
            print(f"  {'reported top-1':28s} {auc:11.3f} {p:17.4f}")
            if auc_e is not None:
                print(f"  {'reported max-entropy':28s} {auc_e:11.3f} {p_e:17.4f}")
            print(f"  {'usable range (mean gap)':28s} {spread:11.5f}")
            print()
            if auc >= 0.85:
                print("  >>> THE REPORT TRACKS THE FLIPS. Reported confidence separates")
                print("      flipped from unanimous spans well, so for this model the")
                print("      logprob is a working uncertainty instrument and should agree")
                print("      with the sampling measure.")
            elif p < 0.05:
                print("  >>> SIGNAL PRESENT BUT COMPRESSED. The report does rank flipped")
                print(f"      spans below unanimous ones (AUC {auc:.3f}, p={p:.4f}), but the")
                print(f"      whole separation spans {spread:.5f} of probability mass. Squeezed")
                print("      into that range it survives a rank test against a clean binary")
                print("      target and dies against a noisy one like human disagreement —")
                print("      which is how the RQ2 logprob correlation can be null here while")
                print("      the sampling correlation is not. Report this as limited dynamic")
                print("      range, NOT as the logprob being blind.")
            else:
                print("  >>> THE REPORT IS BLIND TO THE FLIPS. Reported confidence cannot")
                print(f"      distinguish spans the model flipped (AUC {auc:.3f}, p={p:.4f}).")
                print("      The label is settled by the reasoning before the token is")
                print("      emitted, so its probability is ~1.0 whichever way it landed.")


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
    # AUC: perfect separation, blindness, and the all-ties case a saturated
    # model produces (every reported value identical -> must read 0.5, not 1.0)
    a_perf, p_perf = auc_lower([0.1] * 8, [0.9] * 8, n_boot=200)
    assert a_perf == 1.0, a_perf
    a_tie, p_tie = auc_lower([0.999] * 8, [0.999] * 8, n_boot=200)
    assert a_tie == 0.5, a_tie
    assert p_tie > 0.05, p_tie          # ties carry no evidence of separation
    a_rev, _ = auc_lower([0.9] * 8, [0.1] * 8, n_boot=200)
    assert a_rev == 0.0, a_rev          # flipped MORE confident -> 0, not 1
    # half-tie: 1 of 2 unanimous above the flipped value, 1 equal -> 0.75
    a_half, _ = auc_lower([0.5], [0.5, 0.9], n_boot=200)
    assert abs(a_half - 0.75) < 1e-9, a_half
    # graded targets: unanimous span -> 0, evenly split -> 1-1/k, and presence
    # scaled by how many annotators marked it (this is what the binary flag
    # cannot express, and why a variable annotator count needs it)
    gt, gp = L2.graded_disagreement(
        {"types": {"PER": 5}, "present_in": 5, "total_annotators": 5})
    assert gt == 0.0 and gp == 0.0, (gt, gp)
    gt, gp = L2.graded_disagreement(
        {"types": {"PER": 3, "ORG": 2}, "present_in": 5, "total_annotators": 5})
    assert abs(gt - 0.4) < 1e-9 and gp == 0.0, (gt, gp)
    gt, gp = L2.graded_disagreement(
        {"types": {"PER": 1}, "present_in": 1, "total_annotators": 5})
    assert gt == 0.0 and abs(gp - 0.8) < 1e-9, (gt, gp)
    # the binary flag reads the SAME for these two, the graded one does not
    lo = L2.graded_disagreement({"types": {"PER": 4, "ORG": 1},
                                 "present_in": 5, "total_annotators": 5})[0]
    hi = L2.graded_disagreement({"types": {"PER": 3, "ORG": 2},
                                 "present_in": 5, "total_annotators": 5})[0]
    assert lo < hi, (lo, hi)
    print("self-test OK — span distributions, thresholds, entropy, parse-failure "
          "handling, record dedupe and discrimination AUC all behave as specified")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--min-detections", type=int, default=5,
                    help="min samples a span must appear in to be scored for typing")
    ap.add_argument("--human", type=Path, nargs="+", default=[MARK, DAV],
                    help="human annotation JSONL(s). Default is the two MoBiKo "
                         "annotators. For CoNLL-MTurk pass the single file that "
                         "carries all annotators, e.g. "
                         "data/conll_mturk/sample_n5_w10_400/humans.jsonl")
    ap.add_argument("--target", choices=("binary", "graded"), default="binary",
                    help="binary = 'did ANY annotator differ' (correct only at a "
                         "FIXED annotator count, e.g. MoBiKo's 2). graded = "
                         "1-modal_share and 1-detection_share, required when the "
                         "annotator count varies. See L2.graded_disagreement.")
    ap.add_argument("--min-span-annotators", type=int, default=1,
                    help="drop spans marked by fewer than this many annotators. "
                         "On CoNLL 35%% of union spans are marked by exactly one "
                         "of five; 2 restricts to spans at least two people saw.")
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
    human, names = L1.load_all_human_annotations(list(args.human))
    res = score(records, human, args.min_detections,
                target=args.target, min_span_annotators=args.min_span_annotators)
    meta = records[0].get("run_meta", {}) if records else {}
    print(f"samples: {args.samples}")
    print(f"model:   {meta.get('annotator_model')}  t={meta.get('annotator_temperature')}  "
          f"K={meta.get('selfconsistency_k')}")
    print(f"human:   {len(names)} annotator(s) from "
          f"{', '.join(p.name for p in args.human)}")
    print(f"target:  {args.target}"
          + (f"   (min {args.min_span_annotators} annotators/span, "
             f"{res['n_dropped_below_min_annotators']} spans dropped)"
             if args.min_span_annotators > 1 else ""))
    # The binary target is only meaningful at a fixed annotator count; warn rather
    # than silently producing a number confounded with how many people saw each
    # sentence. This is the failure mode that runs clean and reports nonsense.
    # Counted over the sentences actually SCORED — the loaded file may contain
    # singly-annotated sentences that score() already skips, and warning on those
    # would cry wolf on every MoBiKo run.
    counts = res["annotators_per_scored_sentence"]
    if args.target == "binary" and len(counts) > 1:
        print()
        print(f"  !! WARNING: annotator count VARIES across scored sentences {counts}")
        print("     but --target binary was used. Binary disagreement base rates")
        print("     climb with annotator count, so this correlation is partly")
        print("     measuring how many people saw each sentence. Use --target graded")
        print("     and/or restrict the corpus to one annotator count.")
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
