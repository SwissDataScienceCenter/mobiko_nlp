"""
Split human annotation difficulty into DETECTION vs TYPING, and test how well
agent signals predict each separately.

Motivation
----------
The combined target (1 - exact F1) conflates two failures:
  - detection: do annotators mark the SAME spans (same offsets), ignoring type?
  - typing:    GIVEN a co-located span, do they agree on its label?
The agent deliberates almost entirely about TYPING (the Critic challenges
labels, not span existence), so it should predict typing difficulty far better
than detection difficulty. Splitting the target tests that directly.

Targets (offset-based, between the two human annotators)
-------------------------------------------------------
Per sentence, bucket every span in the union of both annotators' spans:
    co-located + same type  -> agreed
    co-located + diff type   -> TYPING disagreement
    only one annotator marks -> DETECTION disagreement
  detection_difficulty = |detection-disagree| / |union|   (span-presence disagreement
                          rate / Jaccard distance; monotone in, but not equal to, 1 - boundary F1)
  typing_difficulty    = |typing-disagree| / |co-located|       (= 1 - type-agree rate)
                         (NaN, dropped, if no co-located spans)

Span-level typing model
-----------------------
For each CO-LOCATED human span, target = (type_Mark != type_Davnah). Features are
BEHAVIORAL (what the deliberation DID), NOT the model's self-reported numeric
confidence:
    critic_challenged   the Critic disputed this span in any round (0/1)
    severity_major      that dispute was 'major' (0/1)
    churn               distinct labels the span passed through across rounds
    is_hard_type        either human label is a known-hard type (0/1)
    agent_marked        the agent's final output included this span (0/1)
    agent_type_differs  agent's final label != both humans' labels (0/1)

Usage
-----
    python eval_difficulty_split.py \
        --agent-jsonl ../../data/auto_annotated/datademo_manually_labeled1.jsonl \
        --mark   ../../data/aug_runs/combined_M_D_Mark_postprocessed.jsonl \
        --davnah ../../data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl \
        --output ../../output/eval_reports/difficulty_split.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
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
    reconstruct_timeline, record_signals, agent_disagreed_spans,
    norm, label_of, overlaps,
)

HARD_TYPES = {
    "CONCEPT", "QUALITATIVE PROPERTY", "QUANTITATIVE PROPERTY",
    "ANTHROPOGENIC ENTITY", "TEMPORAL PROPERTY",
}


# ───────────────────────── loading ─────────────────────────

def load_agent_records(path: Path) -> List[dict]:
    out = []
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                out.append(json.loads(line))
    return out


def load_human_spans(path: Path) -> Dict[str, List[dict]]:
    """norm(sentence) -> [{start_char, end_char, type, text}] (non-RELATION)."""
    doc = json.loads(Path(path).read_text(encoding="utf8"))
    out: Dict[str, List[dict]] = {}
    for s in doc.get("sentences", []):
        spans = [
            {"start_char": sp["start_char"], "end_char": sp["end_char"],
             "type": (sp.get("type") or "").strip().upper(), "text": sp.get("text", "")}
            for sp in s.get("spans", [])
            if sp.get("start_char") is not None
            and not str(sp.get("type", "")).startswith("RELATION")
        ]
        out[norm(s["text"])] = spans
    return out


# ───────────────────────── target decomposition ─────────────────────────

def split_buckets(spans_a: List[dict], spans_b: List[dict]):
    """Return (co_located, detection_only). co_located = [(key, typeA, typeB)]."""
    A = {(s["start_char"], s["end_char"]): s for s in spans_a}
    B = {(s["start_char"], s["end_char"]): s for s in spans_b}
    co_located, detection_only = [], []
    for k in set(A) | set(B):
        if k in A and k in B:
            co_located.append((k, A[k]["type"], B[k]["type"], A[k]["text"]))
        else:
            detection_only.append(k)
    return co_located, detection_only


def sentence_targets(spans_a, spans_b) -> Tuple[Optional[float], Optional[float], int, int]:
    co, det = split_buckets(spans_a, spans_b)
    union = len(co) + len(det)
    detection = (len(det) / union) if union else None
    typing = (sum(1 for _, ta, tb, _ in co if ta != tb) / len(co)) if co else None
    return detection, typing, union, len(co)


# ───────────────────────── per-span agent signals ─────────────────────────

def span_churn_map(rec: dict) -> Dict[str, int]:
    tl = reconstruct_timeline(rec)
    m: Dict[str, set] = defaultdict(set)
    for ents in tl["annotator_rounds"]:
        for t, lab in ents.items():
            m[t].add(lab)
    for cr in tl["critic_rounds"]:
        for d in cr["disagreements"]:
            tgt = norm(d.get("target", ""))
            match = next((s for s in m if overlaps(tgt, s)), tgt)
            prop = (d.get("proposed_label") or "").strip().upper()
            if prop:
                m[match].add(prop)
    for e in rec.get("final_entities", []):
        t = norm(e.get("text", ""))
        lab = label_of(e)
        if t and lab:
            m[t].add(lab)
    return {k: len(v) for k, v in m.items()}


def agent_final_labels(rec: dict) -> Dict[str, str]:
    return {norm(e.get("text", "")): label_of(e)
            for e in rec.get("final_entities", []) if e.get("text")}


def _lookup(d: Dict[str, Any], text: str):
    """Exact norm-text match, else fuzzy containment match."""
    if text in d:
        return d[text]
    for k, v in d.items():
        if overlaps(text, k):
            return v
    return None


# ───────────────────────── evaluation helpers ─────────────────────────

def repeated_auc(model_fn, X, y, n_repeats=25, n_splits=5):
    aucs = []
    for r in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=r)
        proba = cross_val_predict(model_fn(), X, y, cv=kf, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, proba))
    aucs.sort()
    return {"mean": mean(aucs),
            "lo": aucs[int(0.025 * len(aucs))],
            "hi": aucs[min(len(aucs) - 1, int(0.975 * len(aucs)))]}


def _spearman_table(signal_rows: List[Dict[str, float]], target: List[float], keys):
    out = []
    for label, key in keys:
        x = [r[key] for r in signal_rows]
        rho = spearmanr(x, target).statistic
        p = spearmanr(x, target).pvalue
        out.append((label, rho, p))
    out.sort(key=lambda t: -(abs(t[1]) if t[1] == t[1] else 0))
    return out


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-jsonl", type=Path, required=True)
    ap.add_argument("--mark", type=Path, required=True)
    ap.add_argument("--davnah", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=25)
    args = ap.parse_args()

    agent_records = load_agent_records(args.agent_jsonl)
    mark = load_human_spans(args.mark)
    davnah = load_human_spans(args.davnah)

    # ── sentence-level targets + sentence signals ──
    det_y, typ_y = [], []
    det_rows, typ_rows = [], []
    # ── span-level typing dataset ──
    span_feat: List[Dict[str, float]] = []
    span_y: List[int] = []

    SENT_KEYS = [
        ("rounds_used", "rounds_used"),
        ("R1 disagreements", "r1_disagreements"),
        ("total disagreements (all steps)", "total_disagreements_all_steps"),
        ("critic missing_annotations", "n_missing"),
        ("severity-weighted disagreement", "severity_weighted"),
        ("mean label churn", "churn_mean"),
        ("frac hard-type entities", "frac_hard_types"),
        ("agent entity count", "n_entities"),
    ]

    for rec in agent_records:
        k = norm(rec.get("sentence", ""))
        if k not in mark or k not in davnah:
            continue
        sa, sb = mark[k], davnah[k]
        detection, typing, union, n_co = sentence_targets(sa, sb)
        if union == 0:
            continue

        # sentence-level agent signals (behavioral; no numeric confidence)
        sig = record_signals(rec)
        tl = reconstruct_timeline(rec)
        n_missing = sum(len(cr["missing"]) for cr in tl["critic_rounds"])
        n_major = sum(1 for cr in tl["critic_rounds"] for d in cr["disagreements"]
                      if (d.get("severity") or "").lower() == "major")
        n_minor = sum(1 for cr in tl["critic_rounds"] for d in cr["disagreements"]
                      if (d.get("severity") or "").lower() == "minor")
        churn = span_churn_map(rec)
        ents = rec.get("final_entities", [])
        n_hard = sum(1 for e in ents if label_of(e) in HARD_TYPES)
        row = {
            "rounds_used": float(sig["rounds_used"]),
            "r1_disagreements": float(sig["r1_disagreements"]),
            "total_disagreements_all_steps": float(sig["total_disagreements_all_steps"]),
            "n_missing": float(n_missing),
            "severity_weighted": float(2 * n_major + n_minor),
            "churn_mean": float(mean(churn.values()) if churn else 0.0),
            "frac_hard_types": float(n_hard / len(ents)) if ents else 0.0,
            "n_entities": float(len(ents)),
        }

        det_y.append(detection)
        det_rows.append(row)
        if typing is not None:
            typ_y.append(typing)
            typ_rows.append(row)

        # span-level typing rows (co-located spans only)
        co, _ = split_buckets(sa, sb)
        adis = agent_disagreed_spans(rec)
        afinal = agent_final_labels(rec)
        for _key, ta, tb, text in co:
            nt = norm(text)
            d = _lookup(adis, nt) or {}
            challenged = 1 if d.get("agent_disagreed") else 0
            sev_major = 1 if (d.get("severity") or "").lower() == "major" else 0
            ch = _lookup(churn, nt) or 1
            agent_lab = _lookup(afinal, nt)
            agent_marked = 1 if agent_lab else 0
            agent_diff = 1 if (agent_lab and agent_lab != ta and agent_lab != tb) else 0
            is_hard = 1 if (ta in HARD_TYPES or tb in HARD_TYPES) else 0
            span_feat.append({
                "critic_challenged": float(challenged),
                "severity_major": float(sev_major),
                "churn": float(ch),
                "is_hard_type": float(is_hard),
                "agent_marked": float(agent_marked),
                "agent_type_differs": float(agent_diff),
            })
            span_y.append(1 if ta != tb else 0)

    print(f"\n{'='*68}")
    print("  DIFFICULTY SPLIT — detection vs typing")
    print(f"{'='*68}")
    print(f"  Sentences (agent ∩ both humans, non-empty union): {len(det_y)}")
    print(f"    with ≥1 co-located span (typing defined):       {len(typ_y)}")

    det_arr = np.array(det_y)
    typ_arr = np.array(typ_y)
    print(f"\n  Target levels (mean):")
    print(f"    detection difficulty: {det_arr.mean():.3f}  (span-presence disagreement rate, det_only/union)")
    print(f"    typing difficulty:    {typ_arr.mean():.3f}  (1 - type-agree rate on co-located)")
    # are the two targets independent?
    paired_det = [d for d, t in zip(det_y, typ_y + [None]*(len(det_y)-len(typ_y)))]
    # recompute pairing cleanly on sentences that have both defined
    both = [(r_d, ty) for r_d, ty in zip(det_y[:len(typ_y)], typ_y)]
    if len(typ_y) >= 3:
        rho_dt = spearmanr([d for d, _ in both], [t for _, t in both]).statistic
        print(f"    Spearman(detection, typing) across sentences: {rho_dt:+.3f}"
              f"  → {'largely independent' if abs(rho_dt) < 0.3 else 'correlated'}")

    print(f"\n  Univariate Spearman of agent signals vs DETECTION difficulty (n={len(det_y)}):")
    for lbl, rho, p in _spearman_table(det_rows, det_y, SENT_KEYS):
        star = " *" if (p == p and p < 0.05) else ""
        print(f"    {lbl:34s} ρ={rho:+.3f}  p={p:.3f}{star}")

    print(f"\n  Univariate Spearman of agent signals vs TYPING difficulty (n={len(typ_y)}):")
    for lbl, rho, p in _spearman_table(typ_rows, typ_y, SENT_KEYS):
        star = " *" if (p == p and p < 0.05) else ""
        print(f"    {lbl:34s} ρ={rho:+.3f}  p={p:.3f}{star}")

    # ── span-level typing model ──
    names = sorted(span_feat[0].keys())
    Xs = np.array([[r[n] for n in names] for r in span_feat], dtype=float)
    ys = np.array(span_y)
    base = ys.mean()
    print(f"\n{'='*68}")
    print(f"  SPAN-LEVEL TYPING MODEL  (co-located spans only)")
    print(f"{'='*68}")
    print(f"  Spans: {len(ys)}   human type-disagreement base rate: {base:.3f}")

    # The money table: does 'Critic challenged' predict human type-disagreement?
    ci = names.index("critic_challenged")
    ch_mask = Xs[:, ci] == 1
    p_ch = ys[ch_mask].mean() if ch_mask.any() else float("nan")
    p_no = ys[~ch_mask].mean() if (~ch_mask).any() else float("nan")
    print(f"\n  P(humans disagree on type | Critic challenged span)   = {p_ch:.3f}  "
          f"(n={int(ch_mask.sum())})")
    print(f"  P(humans disagree on type | Critic did NOT challenge) = {p_no:.3f}  "
          f"(n={int((~ch_mask).sum())})")
    if p_no and p_no == p_no and p_ch == p_ch:
        lift = p_ch / p_no if p_no > 0 else float("inf")
        print(f"  → lift = {lift:.2f}×")

    print(f"\n  Single-feature AUC (each behavioral signal alone):")
    for i, n in enumerate(names):
        col = Xs[:, i]
        if len(set(col)) > 1:
            auc = roc_auc_score(ys, col)
            print(f"    {n:22s} AUC={auc:.3f}")

    auc = repeated_auc(
        lambda: Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=1.0, max_iter=1000))]),
        Xs, ys, n_repeats=args.repeats)
    print(f"\n  Logistic model (all behavioral feats, {args.repeats}× 5-fold CV):")
    print(f"    ROC-AUC = {auc['mean']:.3f} [{auc['lo']:.3f}, {auc['hi']:.3f}]")
    print(f"    (vs 0.50 chance; base rate {base:.2f})")

    if args.output:
        out = {
            "n_sentences": len(det_y), "n_sentences_typing": len(typ_y),
            "mean_detection_difficulty": float(det_arr.mean()),
            "mean_typing_difficulty": float(typ_arr.mean()),
            "detection_univariate": [
                {"signal": l, "rho": (r if r == r else None), "p": (p if p == p else None)}
                for l, r, p in _spearman_table(det_rows, det_y, SENT_KEYS)],
            "typing_univariate": [
                {"signal": l, "rho": (r if r == r else None), "p": (p if p == p else None)}
                for l, r, p in _spearman_table(typ_rows, typ_y, SENT_KEYS)],
            "span_typing": {
                "n_spans": len(ys), "base_rate": float(base),
                "p_disagree_given_critic_challenged": (None if p_ch != p_ch else float(p_ch)),
                "p_disagree_given_not_challenged": (None if p_no != p_no else float(p_no)),
                "model_auc": auc,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()