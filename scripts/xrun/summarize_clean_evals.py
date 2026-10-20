#!/usr/bin/env python
"""
Cross-run summary of the full eval suite on the clean grid.

`run_evals.sh` produces eight reports per run; this pulls the headline number
out of each and lays them side by side, so a claim can be checked against every
run at once rather than one JSON at a time. Nothing is recomputed here — every
figure is read from the stored reports, which carry their own provenance block.

    python scripts/xrun/summarize_clean_evals.py
    python scripts/xrun/summarize_clean_evals.py --section logprob
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
REPORTS = ROOT / "output" / "eval_reports_clean"


def label(run: str) -> str:
    return (run + "  [PARTIAL]") if is_partial(run) else run


def load(run: str, stem: str):
    p = REPORTS / run / f"{stem}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


EXPECTED_SENTENCES = 155


def is_partial(run: str) -> bool:
    """True when the run scored fewer sentences than the corpus has.

    A run still being written scores a different sentence set, so its gap and
    correlations are not comparable to a complete run's and must not be read as
    a replicate. multi_pass_deepseek_3 was mid-flight when this suite ran.
    """
    n = g(load(run, "layer34"), "n_sentences")
    return isinstance(n, int) and n < EXPECTED_SENTENCES


def runs():
    """Multi-pass runs first, then the annotator-only control."""
    all_runs = sorted(d.name for d in REPORTS.iterdir() if d.is_dir())
    return ([r for r in all_runs if r.startswith("multi")],
            [r for r in all_runs if r.startswith("single")])


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(v, spec="7.3f"):
    """Format a number, or a dash of the same width when it is absent.

    Width has to be parsed out of the spec rather than reused verbatim: a
    sign-carrying spec like '+6.3f' is illegal for strings, which is how the
    annotator-only rows (no Critic, so no challenge rate) used to crash this."""
    if isinstance(v, (int, float)):
        return f"{v:{spec}}"
    width = spec.split(".")[0].lstrip("+-<>^") or "7"
    return f"{'-':>{width}s}"


def sec_layer1(rs):
    print("\nLAYER 1 — agent vs the two-human ceiling (permutation on per-sentence strict F1)")
    print(f"  {'run':46s} {'agent':>7s} {'human':>7s} {'gap':>7s} {'p':>7s}")
    for r in rs:
        d = load(r, "layer1")
        a = g(d, "significance", "agent_mean_sentence_strict_f1")
        h = g(d, "significance", "human_mean_sentence_strict_f1")
        p = g(d, "significance", "strict_f1_p_value")
        gap = (a - h) if isinstance(a, float) and isinstance(h, float) else None
        print(f"  {label(r)[:46]:46s} {fmt(a)} {fmt(h)} {fmt(gap,'+7.3f')} {fmt(p)}")


def sec_layer2(rs):
    print("\nLAYER 2 — does AGENT disagreement track HUMAN disagreement? (per span)")
    print(f"  {'run':46s} {'concord':>8s} {'odds':>6s} {'fisher p':>9s} {'rho':>7s} {'rho p':>7s}")
    for r in rs:
        d = load(r, "layer2")
        print(f"  {label(r)[:46]:46s} {fmt(g(d,'concordance'),'8.3f')} {fmt(g(d,'odds_ratio'),'6.2f')} "
              f"{fmt(g(d,'fisher_p_value'),'9.4f')} {fmt(g(d,'spearman_rho'))} "
              f"{fmt(g(d,'spearman_p_value'))}")


def sec_layer34(rs):
    print("\nLAYER 3/4 — how much deliberation actually happened")
    print(f"  {'run':46s} {'critic dis':>10s} {'adj resol':>10s} {'sent adj':>9s} {'all steps':>10s}")
    for r in rs:
        d = load(r, "layer34")
        print(f"  {label(r)[:46]:46s} {fmt(g(d,'per_step','total_critic_disagreements'),'10.0f')} "
              f"{fmt(g(d,'per_step','adjudicator_resolutions'),'10.0f')} "
              f"{fmt(g(d,'per_step','sentences_adjudicated'),'9.0f')} "
              f"{fmt(g(d,'per_step','total_disagreements_all_steps'),'10.0f')}")


def sec_guideline(rs):
    print("\nGUIDELINE ADHERENCE — are the cited rules real?")
    print(f"  {'run':46s} {'entities':>9s} {'rule cov':>9s} {'grounded':>9s} {'matched':>8s}")
    for r in rs:
        d = load(r, "guideline")
        n = g(d, "n_entities")
        m = g(d, "n_matched_to_decision_support")
        frac = (m / n) if isinstance(n, (int, float)) and n and isinstance(m, (int, float)) else None
        print(f"  {label(r)[:46]:46s} {fmt(n,'9.0f')} {fmt(g(d,'rule_coverage'),'9.3f')} "
              f"{fmt(g(d,'grounding_match_rate'),'9.3f')} {fmt(frac,'8.3f')}")


def sec_diffmodel(rs):
    print("\nDIFFICULTY MODEL — can any combination predict human difficulty?")
    print("  (out-of-fold; baseline = the annotator's own mean confidence alone)")
    d0 = load(rs[0], "diffmodel") if rs else None
    names = list((g(d0, "models") or {}).keys())
    print(f"  {'run':38s} {'best|rho|':>9s} " + "".join(f"{n[:15]:>16s}" for n in names))
    for r in rs:
        d = load(r, "diffmodel")
        line = f"  {label(r)[:38]:38s} {fmt(g(d,'best_single_feature_abs_rho'),'9.3f')} "
        for n in names:
            line += f"{fmt(g(d, 'models', n, 'spearman', 'mean'), '16.3f')}"
        print(line)


def sec_diffsplit(rs):
    print("\nDIFFICULTY SPLIT — does a Critic challenge mark a span humans dispute?")
    print(f"  {'run':46s} {'spans':>6s} {'base':>6s} {'challenged':>11s} {'not':>6s} {'lift':>6s}")
    for r in rs:
        d = load(r, "diffsplit")
        b = g(d, "span_typing", "base_rate")
        c = g(d, "span_typing", "p_disagree_given_critic_challenged")
        n = g(d, "span_typing", "p_disagree_given_not_challenged")
        lift = (c - n) if isinstance(c, float) and isinstance(n, float) else None
        print(f"  {label(r)[:46]:46s} {fmt(g(d,'span_typing','n_spans'),'6.0f')} {fmt(b,'6.3f')} "
              f"{fmt(c,'11.3f')} {fmt(n,'6.3f')} {fmt(lift,'+6.3f')}")


def sec_logprob(rs):
    print("\nLOGPROB UNCERTAINTY — span-level type logprob vs human type-disagreement")
    print(f"  {'run':46s} {'spans':>6s} {'rho':>7s} {'p':>8s} {'entropy rho':>12s} {'p':>8s}")
    for r in rs:
        d = load(r, "logprob")
        print(f"  {label(r)[:46]:46s} {fmt(g(d,'span_level','n_matched_spans'),'6.0f')} "
              f"{fmt(g(d,'span_level','logprob','point_biserial_rho'))} "
              f"{fmt(g(d,'span_level','logprob','point_biserial_p'),'8.4f')} "
              f"{fmt(g(d,'span_level','entropy','point_biserial_rho'),'12.3f')} "
              f"{fmt(g(d,'span_level','entropy','point_biserial_p'),'8.4f')}")


def sec_history(rs):
    """The sentence-level deliberation signals, as eval_deliberation_history
    reports them. This is the granularity at which the deliberation features are
    natively defined, so it is the fairest test of them."""
    print("\nDELIBERATION HISTORY — sentence-level signals vs human difficulty")
    d0 = load(rs[0], "history") if rs else None
    sigs = [c.get("signal") for c in (g(d0, "difficulty_prediction", "correlations") or [])]
    print(f"  {'run':46s} " + "".join(f"{(s or '?')[:13]:>15s}" for s in sigs))
    for r in rs:
        d = load(r, "history")
        cors = {c.get("signal"): c for c in
                (g(d, "difficulty_prediction", "correlations") or [])}
        line = f"  {label(r)[:46]:46s} "
        for sname in sigs:
            c = cors.get(sname) or {}
            rho, pv = c.get("spearman_rho"), c.get("p_value")
            star = "*" if isinstance(pv, (int, float)) and pv < 0.05 else " "
            line += (f"{rho:14.3f}{star}" if isinstance(rho, (int, float)) else f"{'-':>15s}")
        print(line)
    print("\n  convergence: round-1 disputes followed into a round-2 revision")
    print(f"  {'run':46s} {'followed':>9s} {'resolved':>9s} {'rate':>7s}")
    for r in rs:
        d = load(r, "history")
        ct = g(d, "convergence_trajectory") or {}
        print(f"  {label(r)[:46]:46s} {fmt(ct.get('round1_disputes_followed'),'9.0f')} "
              f"{fmt(ct.get('resolved_by_round2'),'9.0f')} {fmt(ct.get('resolution_rate'))}")


SECTIONS = {"layer1": sec_layer1, "layer2": sec_layer2, "layer34": sec_layer34,
            "guideline": sec_guideline, "diffmodel": sec_diffmodel,
            "diffsplit": sec_diffsplit, "logprob": sec_logprob,
            "history": sec_history}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=sorted(SECTIONS))
    a = ap.parse_args()
    multi, single = runs()
    todo = [a.section] if a.section else list(SECTIONS)
    for name in todo:
        fn = SECTIONS[name]
        fn(multi)
        if name in ("layer1", "logprob", "diffmodel", "diffsplit"):
            print("  --- annotator-only control ---")
            fn(single)


if __name__ == "__main__":
    main()
