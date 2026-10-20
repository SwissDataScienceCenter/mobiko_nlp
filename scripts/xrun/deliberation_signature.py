#!/usr/bin/env python
"""
Does the deliberation log know more than the Annotator's first pass?

The paper is built on multi-agent deliberation and nothing yet justifies that
framing on evidence: on the clean grid deliberation buys +0.006 to +0.036 strict
F1 at ~5x the calls, and its span-level signals were near-null. But those were
SPAN-level measurements, and deliberation signals are natively sentence-shaped —
how many rounds it took, whether the sentence was flagged, how many disputes
arose. This asks the question at the granularity where the signals actually live.

Three feature sets, same target, same folds, fitted per run:

  baseline      everything available from the Annotator's FIRST pass, before any
                Critic or Adjudicator exists: its token logprobs and entropies,
                the confidences it self-reported, its own uncertain_cases.
  deliberation  what only the pipeline produces: round counts, Critic disputes
                and their severity, Adjudicator resolutions, label churn across
                rounds, the review flag, and the Critic's and Adjudicator's own
                token probabilities.
  both

The comparison that matters is `both` against `baseline`. eval_difficulty_model
compares against the best single deliberation feature, which can only show that
combining deliberation signals beats one deliberation signal. If the pipeline's
extra signals do not beat what one annotator pass already gives you, the
architecture is not carrying the uncertainty claim.

    python scripts/xrun/deliberation_signature.py            # phase A + control
    python scripts/xrun/deliberation_signature.py --repeats 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("scripts/xrun", "src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import eval_layer1_output as L1            # noqa: E402
import eval_difficulty_model as EDM        # noqa: E402
import xrun_analysis as X                  # noqa: E402
import label_norm                       # noqa: E402
# multi_pass_gpt-oss_2 carries 3 Critic disagreements whose "target" is a
# relation OBJECT rather than a span string, in 2 of its 155 records. That
# reaches normalize_text and raises. Coerce non-strings to empty so one
# malformed record cannot drop a whole run from the comparison.
_orig_normalize_text = label_norm.normalize_text
label_norm.normalize_text = lambda t: _orig_normalize_text(t if isinstance(t, str) else "")
import deliberation_history as _DH        # noqa: E402
_DH.norm = label_norm.normalize_text
# eval_difficulty_model did `from deliberation_history import norm`, so it holds
# its own reference that the line above does not reach. Its _resolution_direction
# and _label_churn are exactly where the malformed target lands.
EDM.norm = label_norm.normalize_text
from deliberation_history import (         # noqa: E402
    load_records, record_signals, try_parse_json,
)

D = ROOT / "data" / "auto_annotated_clean"

# Run set comes from clean_grid.CLEAN, not a copy. A local list here silently
# went stale: it predated multi_pass_deepseek_3 completing and the arrival of
# single_pass+deepseek_no_deliberation_2, so this analysis was running on an
# older grid than every other script.
from clean_grid import CLEAN  # noqa: E402

MULTI = [(m, p.name) for m, arm, p in CLEAN if arm == "multi"]
# Negative control: no Critic, no Adjudicator, so the deliberation block must
# find nothing here. If it does, those features encode something else.
SINGLE = [(m, p.name) for m, arm, p in CLEAN if arm == "single"]


LOW_CONF = EDM.LOW_CONF


# ───────────────────────────── features ─────────────────────────────

def _first_annotator_json(rec: dict):
    """The Annotator's round-1 output, parsed. This is the pre-deliberation state."""
    for m in rec.get("messages", []):
        if m.get("agent") != "Annotator":
            continue
        parsed = try_parse_json(m.get("content", ""))
        if parsed and ("entities" in parsed or "uncertain_cases" in parsed):
            return parsed
    return None


def _agg(vals, prefix, out, default=0.0):
    if vals:
        out[f"{prefix}_mean"] = float(mean(vals))
        out[f"{prefix}_min"] = float(min(vals))
        out[f"{prefix}_max"] = float(max(vals))
        out[f"{prefix}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    else:
        for s in ("mean", "min", "max", "std"):
            out[f"{prefix}_{s}"] = default


def baseline_features(rec: dict) -> dict:
    """What one annotator pass gives you, with no Critic and no Adjudicator.

    Deliberately does NOT read final_entities: in a multi-pass run those are the
    Adjudicator's output, so using their confidences would leak deliberation into
    the baseline and make the comparison meaningless.
    """
    f: dict = {}
    f["ann_mean_logprob"] = float(rec.get("annotator_mean_logprob") or 0.0)
    f["ann_mean_entropy"] = float(rec.get("annotator_mean_entropy") or 0.0)

    lps = [e["type_mean_logprob"] for e in (rec.get("annotator_entity_logprobs") or [])
           if e.get("type_mean_logprob") is not None]
    ents_h = [e["type_max_entropy"] for e in (rec.get("annotator_entity_logprobs") or [])
              if e.get("type_max_entropy") is not None]
    _agg(lps, "ann_lp", f)
    _agg(ents_h, "ann_ent", f)
    f["n_ann_spans"] = float(len(rec.get("annotator_entity_logprobs") or []))

    parsed = _first_annotator_json(rec)
    ents = (parsed or {}).get("entities") or []
    confs = [float(e["confidence"]) for e in ents
             if isinstance(e.get("confidence"), (int, float))]
    _agg(confs, "a_conf", f, default=1.0)
    f["a_n_entities"] = float(len(ents))
    f["a_n_low_conf"] = float(sum(1 for c in confs if c < LOW_CONF))
    f["a_frac_low_conf"] = (f["a_n_low_conf"] / len(confs)) if confs else 0.0
    f["a_n_uncertain"] = float(len((parsed or {}).get("uncertain_cases") or []))
    return f


def deliberation_features(rec: dict) -> dict:
    """What only the pipeline produces. Reuses eval_difficulty_model's extractors."""
    f: dict = {}
    sig = record_signals(rec)
    for k in ("rounds_used", "r1_disagreements", "r1_agreements", "r1_rate",
              "r2_disagreements", "critic_disagreements_all_rounds",
              "adjudicator_resolutions", "total_disagreements_all_steps", "flagged"):
        f[k] = float(sig[k])

    n_major, n_minor, sev_w, n_missing = EDM._severity_and_missing(rec)
    f.update(n_major=float(n_major), n_minor=float(n_minor),
             severity_weighted=float(sev_w), n_missing=float(n_missing))

    rd = EDM._resolution_direction(rec)
    for k, v in rd.items():
        f[f"n_{k}"] = float(v)

    cmax, cmean, nch = EDM._label_churn(rec)
    f.update(churn_max=float(cmax), churn_mean=float(cmean), n_churned=float(nch))

    f["n_messages"] = float(len(rec.get("messages", [])))

    # The Critic's and Adjudicator's own token probabilities — captured only
    # since the logprob-per-role change, so absent on the older runs. Aggregated
    # rather than per-span because the target here is the sentence.
    for role, key in (("crit", "critic_entity_logprobs"),
                      ("adj", "adjudicator_entity_logprobs")):
        lps = [e["type_mean_logprob"] for e in (rec.get(key) or [])
               if e.get("type_mean_logprob") is not None]
        hs = [e["type_max_entropy"] for e in (rec.get(key) or [])
              if e.get("type_max_entropy") is not None]
        _agg(lps, f"{role}_lp", f)
        _agg(hs, f"{role}_ent", f)
        f[f"n_{role}_spans"] = float(len(rec.get(key) or []))
    return f


# ───────────────────────────── dataset ─────────────────────────────

def build(path: Path, human: dict):
    """(X_base, X_delib, y, names_base, names_delib) for one run."""
    rowsb, rowsd, y = [], [], []
    for rec in load_records(path):
        key = L1._normalize(rec.get("sentence", ""))
        anns = human.get(key)
        if not anns or len(anns) < 2:
            continue
        ids = list(anns.keys())
        f1 = L1.sentence_f1(anns[ids[0]]["entities"], anns[ids[1]]["entities"],
                            mode="text_type")
        if f1 is None:
            continue
        y.append(1.0 - f1)
        rowsb.append(baseline_features(rec))
        rowsd.append(deliberation_features(rec))

    def mat(rows):
        names = sorted(rows[0].keys())
        M = np.array([[r[n] for n in names] for r in rows], dtype=float)
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        keep = [i for i in range(M.shape[1]) if M[:, i].std() > 0]
        return M[:, keep], [names[i] for i in keep]

    Xb, nb = mat(rowsb)
    Xd, nd = mat(rowsd)
    return Xb, Xd, np.array(y), nb, nd


def model_fn():
    return Pipeline([("s", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-2, 3, 20)))])


def cv_score(Xm, y, hard, repeats, splits=5):
    """Out-of-fold Spearman and ROC-AUC, averaged over repeated K-fold."""
    if Xm.shape[1] == 0:
        return None
    rhos, aucs = [], []
    for r in range(repeats):
        kf = KFold(n_splits=splits, shuffle=True, random_state=r)
        pred = cross_val_predict(model_fn(), Xm, y, cv=kf)
        rho = spearmanr(pred, y).statistic
        if rho == rho:
            rhos.append(rho)
        if len(set(hard)) > 1:
            aucs.append(roc_auc_score(hard, pred))
    return {"rho": float(mean(rhos)) if rhos else float("nan"),
            "auc": float(mean(aucs)) if aucs else float("nan")}


def best_univariate(Xm, names, y):
    """Strongest single feature, by |Spearman|. No fitting, so no overfitting.

    Reported because n=155 against 30+ features is a thin ratio: a null
    out-of-fold score could mean the signal is absent OR that the model is
    drowning it. If the best single feature is strong while the fitted model is
    not, the problem is capacity, not the data.
    """
    if Xm.shape[1] == 0:
        return None, 0.0
    best, br = None, 0.0
    for i, nm in enumerate(names):
        r = spearmanr(Xm[:, i], y).statistic
        if r == r and abs(r) > abs(br):
            best, br = nm, r
    return best, br


def run(rows, human, repeats, title):
    print(f"\n{title}")
    print(f"  {'run':32s} {'n':>4s} {'nB':>3s} {'nD':>3s} "
          f"{'base':>7s} {'delib':>7s} {'both':>7s} {'gain':>7s} "
          f"{'best single delib feature':>34s}")
    out = []
    for model, name in rows:
        path = D / name
        if not path.exists():
            continue
        Xb, Xd, y, nb, nd = build(path, human)
        hard = (y >= np.median(y)).astype(int)
        Xboth = np.hstack([Xb, Xd]) if Xd.shape[1] else Xb
        sb = cv_score(Xb, y, hard, repeats)
        sd = cv_score(Xd, y, hard, repeats)
        sx = cv_score(Xboth, y, hard, repeats)
        gain = (sx["rho"] - sb["rho"]) if (sb and sx) else float("nan")
        out.append((model, name, sb, sd, sx, gain))
        bu, br = best_univariate(Xd, nd, y)
        f = lambda v: f"{v:7.3f}" if v is not None and v == v else f"{'-':>7s}"
        print(f"  {name[:32]:32s} {len(y):4d} {Xb.shape[1]:3d} {Xd.shape[1]:3d} "
              f"{f(sb['rho'] if sb else None)}{f(sd['rho'] if sd else None)}"
              f"{f(sx['rho'] if sx else None)} {gain:+7.3f} "
              f"{(bu or '-'):>26s} {br:+7.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=25,
                    help="repeated K-fold repeats (default 25)")
    args = ap.parse_args()

    human, _ = L1.load_all_human_annotations([X.MARK, X.DAV])
    print("Sentence-level: does the deliberation log beat the Annotator's first pass?")
    print("target = 1 - inter-human entity F1 (text+type). Out-of-fold, "
          f"{args.repeats}x 5-fold, RidgeCV.")
    print("nB / nD = non-constant features in the baseline / deliberation set.")

    res = run(MULTI, human, args.repeats, "FULL PIPELINE")
    ctl = run(SINGLE, human, args.repeats,
              "NEGATIVE CONTROL — annotator only (deliberation set should be empty/null)")

    print("\n  per-model replicate agreement on the gain (both - baseline):")
    by = {}
    for model, _n, _b, _d, _x, g in res:
        by.setdefault(model, []).append(g)
    for model, gs in by.items():
        agree = "yes" if all(g > 0 for g in gs) or all(g <= 0 for g in gs) else "NO"
        print(f"    {model:14s} {[f'{g:+.3f}' for g in gs]}  consistent: {agree}")
    print("\n  A gain that does not hold across a model's replicates is the failure")
    print("  mode that broke the Adjudicator result. Treat it as noise.")
    if ctl:
        print(f"\n  control: deliberation features surviving as non-constant = "
              f"{[x[4] is not None for x in ctl]}")


if __name__ == "__main__":
    main()
