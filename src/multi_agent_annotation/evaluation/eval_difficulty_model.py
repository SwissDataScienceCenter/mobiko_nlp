"""
Multivariate difficulty-prediction model.

Question: can we predict human annotation difficulty BETTER than the best single
agent signal (univariate Spearman rho ~0.27 for `rounds_used`) by combining all
available deliberation/uncertainty signals?

Target
------
difficulty(sentence) = 1 - mean pairwise inter-human entity F1 (exact match),
averaged over ALL annotator pairs that co-annotated the sentence. A binary
"hard" label (difficulty >= median) is derived for ROC-AUC.

Features (per sentence) — all from the agent record; NO wall-clock timing
(timing is confounded by network latency / model load, so it is excluded):
  History (deliberation_history.record_signals):
    rounds_used, r1_disagreements, r1_agreements, r1_rate, r2_disagreements,
    critic_disagreements_all_rounds, adjudicator_resolutions,
    total_disagreements_all_steps, flagged
  Confidence (final_entities[].confidence):
    n_entities, conf_mean, conf_min, conf_std, n_low_conf(<0.7), frac_low_conf
  Severity-weighted disagreement (Critic disagreements[].severity):
    n_major, n_minor, severity_weighted (2*major + 1*minor)
  Explicit uncertainty (Annotator uncertain_cases):
    n_uncertain
  Resolution direction (round-1 disputes followed into the revision):
    n_dropped, n_third_label, n_accepted_critic, n_kept_own
  Label churn (distinct labels a span passes through across rounds):
    churn_max, churn_mean, n_churned
  Critic-flagged missing spans:
    n_missing
  Token effort (NOT timing):
    total_completion_tokens, n_messages
  Hard-type prior (types everyone struggles with):
    n_hard_types, frac_hard_types

Evaluation
----------
Repeated K-fold CV. For each repeat we collect out-of-fold predictions and score
Spearman(pred, difficulty) and ROC-AUC(hard, pred); we report mean and 95%
spread across repeats. Baseline = the single best univariate feature. Feature
importances come from permutation importance on a full-data refit.

Usage
-----
    python eval_difficulty_model.py \
        --agent-jsonl ../../data/auto_annotated/datademo_manually_labeled1.jsonl \
        --human-jsonl ../../data/aug_runs/combined_M_D_Mark_postprocessed.jsonl \
                      ../../data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl \
        --output ../../output/eval_reports/difficulty_model.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from eval_utils import stamp_provenance
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# deliberation_history lives at the package root; eval siblings in evaluation/.
_PKG_ROOT = Path(__file__).resolve().parent.parent   # …/multi_agent_annotation
for _p in (_PKG_ROOT, _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from deliberation_history import (  # noqa: E402
    reconstruct_timeline, record_signals, norm, label_of, overlaps, try_parse_json,
)


# ───────────────────────── data loading (inlined, self-contained) ─────────────────────────
# Inlined rather than imported from eval_layer1/eval_layer2 so this script does
# not break when those sibling modules' APIs change.

def load_agent_records(path: Path) -> List[dict]:
    out = []
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                out.append(json.loads(line))
    return out


def load_all_human_annotations(paths: List[Path]) -> Dict[str, Dict[str, dict]]:
    """native {doc_id, sentences:[{text, spans:[{text,type}]}]} per annotator file
    -> {stripped_sentence: {annotator_name: {"entities": [{text, type}]}}}."""
    data: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf8"))
        name = Path(p).stem
        for s in doc.get("sentences", []):
            sent = s["text"].strip()
            ents = [
                {"text": sp.get("text", ""), "type": (sp.get("type") or "").strip().upper()}
                for sp in s.get("spans", [])
                if not str(sp.get("type", "")).startswith("RELATION")
            ]
            data[sent][name] = {"entities": ents}
    return dict(data)


def _exact_f1(a: List[dict], b: List[dict]) -> float:
    """Exact-match entity F1 on (normalized text, uppercased type)."""
    sa = {(norm(e["text"]), e["type"]) for e in a}
    sb = {(norm(e["text"]), e["type"]) for e in b}
    if not sa and not sb:
        return 1.0  # both marked nothing → perfect agreement
    tp = len(sa & sb)
    p = tp / len(sa) if sa else 0.0
    r = tp / len(sb) if sb else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


HARD_TYPES = {
    "CONCEPT", "QUALITATIVE PROPERTY", "QUANTITATIVE PROPERTY",
    "ANTHROPOGENIC ENTITY", "TEMPORAL PROPERTY",
}
LOW_CONF = 0.7


# ───────────────────────── feature extraction ─────────────────────────

def _resolution_direction(rec: dict) -> Dict[str, int]:
    """Per-record version of the convergence-trajectory direction counter."""
    tl = reconstruct_timeline(rec)
    crit, ann = tl["critic_rounds"], tl["annotator_rounds"]
    out = {"dropped": 0, "third_label": 0, "accepted_critic": 0, "kept_own": 0}
    if not crit or len(ann) < 2:
        return out
    nxt = ann[1]
    for d in crit[0]["disagreements"]:
        target = norm(d.get("target", ""))
        proposed = (d.get("proposed_label") or "").strip().upper()
        annot = (d.get("annotator_label") or "").strip().upper()
        present, new_label = False, None
        for span, lab in nxt.items():
            if overlaps(target, span):
                present, new_label = True, lab
                break
        if not present:
            out["dropped"] += 1
        elif new_label == proposed:
            out["accepted_critic"] += 1
        elif new_label == annot:
            out["kept_own"] += 1
        else:
            out["third_label"] += 1
    return out


def _label_churn(rec: dict) -> Tuple[float, float, int]:
    """Distinct labels each span passes through across all rounds + adjudication."""
    tl = reconstruct_timeline(rec)
    span_labels: Dict[str, set] = {}
    for ents in tl["annotator_rounds"]:
        for t, lab in ents.items():
            span_labels.setdefault(t, set()).add(lab)
    for cr in tl["critic_rounds"]:
        for d in cr["disagreements"]:
            tgt = norm(d.get("target", ""))
            match = next((s for s in span_labels if overlaps(tgt, s)), tgt)
            prop = (d.get("proposed_label") or "").strip().upper()
            if prop:
                span_labels.setdefault(match, set()).add(prop)
    for e in rec.get("final_entities", []):
        t = norm(e.get("text", ""))
        lab = label_of(e)
        if t and lab:
            span_labels.setdefault(t, set()).add(lab)
    counts = [len(v) for v in span_labels.values()] or [0]
    n_churned = sum(1 for c in counts if c > 1)
    return (max(counts), mean(counts), n_churned)


def _n_uncertain(rec: dict) -> int:
    """Length of the Annotator's `uncertain_cases` list (round 1 if present)."""
    for m in rec.get("messages", []):
        if m.get("agent") != "Annotator":
            continue
        parsed = try_parse_json(m.get("content", ""))
        if parsed and "uncertain_cases" in parsed:
            return len(parsed.get("uncertain_cases") or [])
    return 0


def _severity_and_missing(rec: dict) -> Tuple[int, int, int, int]:
    tl = reconstruct_timeline(rec)
    n_major = n_minor = n_missing = 0
    for cr in tl["critic_rounds"]:
        for d in cr["disagreements"]:
            sev = (d.get("severity") or "").lower()
            if sev == "major":
                n_major += 1
            elif sev == "minor":
                n_minor += 1
        n_missing += len(cr["missing"])
    return n_major, n_minor, 2 * n_major + n_minor, n_missing


def extract_features(rec: dict) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    sig = record_signals(rec)
    for k in ("rounds_used", "r1_disagreements", "r1_agreements", "r1_rate",
              "r2_disagreements", "critic_disagreements_all_rounds",
              "adjudicator_resolutions", "total_disagreements_all_steps", "flagged"):
        feat[k] = float(sig[k])

    ents = rec.get("final_entities", [])
    confs = [float(e["confidence"]) for e in ents if e.get("confidence") is not None]
    feat["n_entities"] = float(len(ents))
    feat["conf_mean"] = mean(confs) if confs else 1.0
    feat["conf_min"] = min(confs) if confs else 1.0
    feat["conf_std"] = pstdev(confs) if len(confs) > 1 else 0.0
    feat["n_low_conf"] = float(sum(1 for c in confs if c < LOW_CONF))
    feat["frac_low_conf"] = (feat["n_low_conf"] / len(confs)) if confs else 0.0

    n_major, n_minor, sev_w, n_missing = _severity_and_missing(rec)
    feat["n_major"] = float(n_major)
    feat["n_minor"] = float(n_minor)
    feat["severity_weighted"] = float(sev_w)
    feat["n_missing"] = float(n_missing)

    feat["n_uncertain"] = float(_n_uncertain(rec))

    rd = _resolution_direction(rec)
    feat["n_dropped"] = float(rd["dropped"])
    feat["n_third_label"] = float(rd["third_label"])
    feat["n_accepted_critic"] = float(rd["accepted_critic"])
    feat["n_kept_own"] = float(rd["kept_own"])

    cmax, cmean, nch = _label_churn(rec)
    feat["churn_max"] = float(cmax)
    feat["churn_mean"] = float(cmean)
    feat["n_churned"] = float(nch)

    comp = sum((m.get("token_usage") or {}).get("completion_tokens", 0)
               for m in rec.get("messages", []))
    feat["total_completion_tokens"] = float(comp)
    feat["n_messages"] = float(len(rec.get("messages", [])))

    n_hard = sum(1 for e in ents if label_of(e) in HARD_TYPES)
    feat["n_hard_types"] = float(n_hard)
    feat["frac_hard_types"] = (n_hard / len(ents)) if ents else 0.0
    return feat


# ───────────────────────── target ─────────────────────────

def sentence_difficulty(annotators: Dict[str, dict]) -> float:
    """1 - mean pairwise inter-human entity F1 (exact), over all annotator pairs."""
    ids = list(annotators.keys())
    f1s = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            f1s.append(_exact_f1(annotators[ids[i]]["entities"],
                                 annotators[ids[j]]["entities"]))
    if not f1s:
        return float("nan")
    return 1.0 - (sum(f1s) / len(f1s))


def build_dataset(agent_records, human_data):
    rows, y = [], []
    for rec in agent_records:
        sent = rec["sentence"].strip()
        anns = human_data.get(sent)
        if not anns or len(anns) < 2:
            continue
        diff = sentence_difficulty(anns)
        if diff != diff:  # NaN
            continue
        rows.append(extract_features(rec))
        y.append(diff)
    names = sorted(rows[0].keys())
    X = np.array([[r[n] for n in names] for r in rows], dtype=float)
    # Drop constant columns (no information; break Spearman / waste model capacity)
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
    dropped = [names[i] for i in range(len(names)) if i not in keep]
    X = X[:, keep]
    names = [names[i] for i in keep]
    return X, np.array(y), names, dropped


# ───────────────────────── evaluation ─────────────────────────

def repeated_cv(model_fn, X, y, hard, n_repeats=25, n_splits=5, seed0=0):
    rhos, aucs = [], []
    for r in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed0 + r)
        pred = cross_val_predict(model_fn(), X, y, cv=kf)
        rho = spearmanr(pred, y).statistic
        rhos.append(rho)
        if len(set(hard)) > 1:
            aucs.append(roc_auc_score(hard, pred))
    def summ(v):
        v = [x for x in v if x == x]
        v_sorted = sorted(v)
        lo = v_sorted[int(0.025 * len(v_sorted))]
        hi = v_sorted[min(len(v_sorted) - 1, int(0.975 * len(v_sorted)))]
        return {"mean": mean(v), "lo": lo, "hi": hi}
    return summ(rhos), (summ(aucs) if aucs else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-jsonl", type=Path, required=True)
    ap.add_argument("--human-jsonl", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=25)
    args = ap.parse_args()

    agent_records = load_agent_records(args.agent_jsonl)
    human_data = load_all_human_annotations(args.human_jsonl)
    X, y, names, dropped = build_dataset(agent_records, human_data)
    hard = (y >= np.median(y)).astype(int)

    print(f"\n{'='*64}")
    print("  MULTIVARIATE DIFFICULTY PREDICTION")
    print(f"{'='*64}")
    print(f"  Sentences: {len(y)}   Features: {len(names)} "
          f"(dropped {len(dropped)} constant: {', '.join(dropped) or 'none'})")
    print(f"  Target: 1 - inter-human exact F1  (mean {1-mean(y):.3f} F1, "
          f"difficulty mean {mean(y):.3f})")
    print(f"  Binary 'hard' = difficulty >= median ({(hard==1).sum()} hard / "
          f"{(hard==0).sum()} easy)")

    # ── univariate baseline ──
    print(f"\n  Univariate baseline (per-feature Spearman vs difficulty):")
    uni = []
    for i, n in enumerate(names):
        rho = spearmanr(X[:, i], y).statistic
        uni.append((n, rho))
    uni.sort(key=lambda t: -abs(t[1] if t[1] == t[1] else 0))
    for n, rho in uni[:10]:
        print(f"    {n:28s} rho={rho:+.3f}")
    best_name, best_uni = uni[0][0], abs(uni[0][1]) if uni[0][1] == uni[0][1] else 0.0
    best_idx = names.index(best_name)
    print(f"  Best single feature (in-sample): {best_name}  |rho|={best_uni:.3f}")

    # ── models, repeated CV (all out-of-sample, directly comparable) ──
    CURATED = ["conf_mean", "conf_min", "frac_low_conf", "frac_hard_types",
               "n_churned", "n_dropped", "rounds_used", "n_uncertain",
               "total_disagreements_all_steps"]
    curated_idx = [names.index(n) for n in CURATED if n in names]
    models = {
        "RidgeCV (all feats)": (lambda: Pipeline(
            [("sc", StandardScaler()),
             ("m", RidgeCV(alphas=np.logspace(-1, 3, 25)))]), None),
        "LassoCV (sparse)": (lambda: Pipeline(
            [("sc", StandardScaler()), ("m", LassoCV(cv=5, max_iter=50000))]), None),
        "Ridge (curated 9)": (lambda: Pipeline(
            [("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]), curated_idx),
        "GradBoost": (lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=0), None),
    }
    print(f"\n  Out-of-sample performance ({args.repeats}× 5-fold CV, mean [95% spread]):")
    print(f"    {'model':<22}{'Spearman rho':>22}{'ROC-AUC':>22}")
    # CV-matched single-feature baseline (apples-to-apples vs the models)
    base_rho, base_auc = repeated_cv(
        lambda: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        X[:, [best_idx]], y, hard, n_repeats=args.repeats)
    print(f"    {'baseline: '+best_name[:11]:<22}"
          f"{base_rho['mean']:+.3f} [{base_rho['lo']:+.3f},{base_rho['hi']:+.3f}]   "
          f"{base_auc['mean']:.3f} [{base_auc['lo']:.3f},{base_auc['hi']:.3f}]")
    results = {"baseline_" + best_name: {"spearman": base_rho, "auc": base_auc}}
    for label, (fn, cols) in models.items():
        Xs = X[:, cols] if cols else X
        rho_s, auc_s = repeated_cv(fn, Xs, y, hard, n_repeats=args.repeats)
        results[label] = {"spearman": rho_s, "auc": auc_s}
        rho_str = f"{rho_s['mean']:+.3f} [{rho_s['lo']:+.3f},{rho_s['hi']:+.3f}]"
        auc_str = (f"{auc_s['mean']:.3f} [{auc_s['lo']:.3f},{auc_s['hi']:.3f}]"
                   if auc_s else "  n/a")
        print(f"    {label:<22}{rho_str:>22}{auc_str:>22}")

    # ── permutation importance (Ridge, full-data refit) ──
    ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    ridge.fit(X, y)
    perm = permutation_importance(ridge, X, y, n_repeats=50, random_state=0,
                                  scoring="r2")
    imp = sorted(zip(names, perm.importances_mean), key=lambda t: -t[1])
    print(f"\n  Permutation importance (Ridge, drop in R² when feature shuffled):")
    for n, v in imp[:12]:
        print(f"    {n:28s} {v:+.4f}")

    if args.output:
        out = {
            "n_sentences": len(y), "n_features": len(names),
            "target_mean_difficulty": mean(y),
            "univariate_spearman": {n: (r if r == r else None) for n, r in uni},
            "best_single_feature_abs_rho": best_uni,
            "models": {
                k: {"spearman": v["spearman"], "auc": v["auc"]}
                for k, v in results.items()
            },
            "permutation_importance": {n: v for n, v in imp},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            stamp_provenance(out, args.agent_jsonl)
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()