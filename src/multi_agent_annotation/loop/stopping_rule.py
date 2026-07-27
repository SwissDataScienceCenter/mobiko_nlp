"""
stopping_rule.py — the dual stopping rule for the cold-start reconstruction loop
(RQ-D, spec §11.3).

The loop (``reconstruct_loop.py``) needs a principled answer to "when has the
reconstructed guideline stopped improving?" Spec §11.3 gives a *dual* rule, plus
two monitoring-only signals:

  PRIMARY  (internal, honest, DRIVES the stop):  friction convergence. Stop when
           the relative reduction in TOTAL ROUND-1 DISAGREEMENTS between G_i and
           G_{i+1} is < ``eps`` (default 5%) for ``friction_patience`` (default
           2) consecutive iterations. This never peeks at the expert guideline —
           it is a property of the agents' own deliberation.

  GUARD    (external, OVERRIDES):  held-out F1 echo-chamber guard. If the
           agent-vs-expert F1 on a HELD-OUT split (distinct from the working set
           used to mine confusions, §11.7) DECREASES for ``f1_guard_patience``
           (default 2) consecutive iterations, stop *immediately* and flag
           echo-chamber behaviour. Friction dropping while F1 drops is the
           failure mode; the guard catches it.

  MONITORING ONLY (logged, NEVER a stop signal):
    * guideline text delta — rules added/changed + embedding-similarity drift.
      An LLM amender can rephrase or oscillate without converging, so text
      stability is explicitly NOT a stop signal (spec §11.3).
    * rule coverage — fraction of the enumerated expert disambiguations matched
      (§11.5b); only computed when an expert-rules file is supplied.

This module is import-light on purpose: the pure decision functions
(``relative_reductions``, ``consecutive_tail_decreases``, ``decide_stop``,
``friction_from_report``, ``rule_coverage``) pull in nothing heavy, so they are
cheap to unit-test. The held-out F1 reuses the Layer-1 offset micro-F1 (the same
matching as the IAA ceiling) and the embedding drift / semantic-coverage
lazy-import sentence-transformers, fully guarded so a monitoring-only metric can
never crash the loop or affect the stop.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent          # …/multi_agent_annotation/loop
_PKG_ROOT = _THIS_DIR.parent                         # …/multi_agent_annotation (shared core)
# Make the flat siblings importable however this module was imported: the shared
# core (deliberation_history) lives at the package root and the held-out F1
# reuses eval_layer1_output from the evaluation/ subdir.
for _p in (_PKG_ROOT, _PKG_ROOT / "loop", _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deliberation_history import record_signals  # noqa: E402  (light, no torch)

logger = logging.getLogger("stopping_rule")

# Same embedding model the pipeline uses for guideline search (kept consistent so
# the monitoring numbers are comparable to the rest of the system).
_DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────
# Friction (PRIMARY signal source)
# ─────────────────────────────────────────────────────────────

def _round_value(by_round: Any, k: int) -> int:
    """Read round ``k`` from a disagreements-by-round map, tolerating int OR str
    keys (in-process ``analyze`` yields int keys; reloaded JSON yields str)."""
    if not isinstance(by_round, dict):
        return 0
    if k in by_round:
        return by_round[k]
    if str(k) in by_round:
        return by_round[str(k)]
    return 0


def friction_from_report(report: Dict[str, Any], records: List[dict]) -> Dict[str, Any]:
    """Canonical per-iteration friction dict (spec §11.3 "Friction" line).

    PRIMARY stop reads ``r1_disagreements`` (total round-1 disagreements over the
    fixed working set). Also logs the rounds-used rate and the all-steps total so
    the friction trajectory is fully auditable.
    """
    per_step = report.get("per_step", {}) or {}
    raw_by_round = per_step.get("disagreements_by_round", {}) or {}
    # PRIMARY signal reads the de-noised count: genuine disagreements only (real
    # type change or extent correction, severity != "none"). This excludes Critic
    # "disagreements" that just echo the annotator's label (proposed_label ==
    # annotator_label), which would otherwise inflate friction and mask real
    # convergence. Falls back to the raw count ONLY for older reports that predate
    # the genuine_* field — an empty genuine dict is a real "zero genuine
    # disagreements" and must be used, not treated as missing. NOTE: same-label
    # *extent-only* corrections are also excluded (no structured proposed-extent
    # field distinguishes them), matching the confusion-taxonomy predicate.
    if "genuine_disagreements_by_round" in per_step:
        by_round = per_step["genuine_disagreements_by_round"] or {}
    else:
        by_round = raw_by_round

    sigs = [record_signals(r) for r in records]
    n = len(sigs)
    rounds = [s["rounds_used"] for s in sigs]
    n_multi = sum(1 for s in sigs if s["rounds_used"] and s["rounds_used"] > 1)

    return {
        "n_sentences": report.get("n_sentences", n),
        # PRIMARY signal (de-noised — see above):
        "r1_disagreements": _round_value(by_round, 1),
        # raw round-1 count including same-label/none entries, for auditing:
        "r1_disagreements_raw": _round_value(raw_by_round, 1),
        # context / auditing:
        "total_critic_disagreements": per_step.get("total_critic_disagreements", 0),
        "total_disagreements_all_steps": per_step.get("total_disagreements_all_steps", 0),
        "disagreements_by_round": by_round,
        "disagreements_by_round_raw": raw_by_round,
        "mean_rounds_used": round(sum(rounds) / n, 4) if n else None,
        # "rounds_used rate" = fraction of sentences that needed a 2nd round:
        "rounds_used_rate": round(n_multi / n, 4) if n else None,
    }


# ─────────────────────────────────────────────────────────────
# Held-out agent-vs-expert F1 (GUARD signal source)
# ─────────────────────────────────────────────────────────────

def compute_held_out_f1(
    agent_jsonl: Path,
    expert_files: Sequence[Path],
    mode: str = "strict",
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Agent-vs-expert micro F1 on the held-out split (spec §11.3 / §11.7).

    Reuses the Layer-1 offset matching so the held-out F1 reconciles with the IAA
    ceiling and the rest of the eval. ``mode``:
      * ``strict``   = (start, end, type)  — detection AND typing (the headline).
      * ``boundary`` = (start, end)        — detection only.
      * ``text_type``/``text_only``        — position-agnostic fallbacks.

    The headline ``f1`` pools every (agent, expert) sentence pair across all
    expert files into one micro F1 (support-weighted); ``per_expert`` keeps the
    individual numbers. Returns ``f1=None`` when nothing overlaps — callers must
    treat ``None`` as "no signal", never as a decrease.
    """
    from eval_layer1_output import (  # local import: pulls eval_utils only (light)
        load_agent_records,
        load_all_human_annotations,
        score_reference,
        _normalize,
    )

    agent_records = load_agent_records(Path(agent_jsonl))
    human_data, ann_names = load_all_human_annotations([Path(p) for p in expert_files])

    # Optional display-name remap (mirrors eval_layer1_output.main), only valid
    # when each file yields exactly its filename stem.
    if names:
        if len(names) != len(expert_files):
            raise ValueError("names must match expert_files count")
        stem_to_name = {Path(p).stem: n for p, n in zip(expert_files, names)}
        if set(stem_to_name) == set(ann_names):
            human_data = {
                sent: {stem_to_name.get(a, a): v for a, v in anns.items()}
                for sent, anns in human_data.items()
            }
            ann_names = list(names)

    pooled: List[Tuple[List[dict], List[dict]]] = []
    per_expert: Dict[str, Any] = {}
    for name in ann_names:
        pairs: List[Tuple[List[dict], List[dict]]] = []
        for rec in agent_records:
            anns = human_data.get(_normalize(rec["sentence"]))
            if not anns:
                continue
            ad = anns.get(name)
            if ad is None:
                continue
            gold = ad["entities"]
            agent_ents = rec["entities"]
            if agent_ents and gold:  # doubly-annotated (agent + this expert)
                pairs.append((agent_ents, gold))
        d = score_reference(pairs, mode)
        per_expert[name] = {"f1": d["f1"], "precision": d["precision"],
                            "recall": d["recall"], "n_eval_sentences": len(pairs)}
        pooled.extend(pairs)

    overall = score_reference(pooled, mode)
    return {
        "mode": mode,
        "f1": overall["f1"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "n_eval_sentences": len(pooled),
        "n_agent_sentences": len(agent_records),
        "experts": ann_names,
        "per_expert": per_expert,
    }


# ─────────────────────────────────────────────────────────────
# Embedding helper (shared by text-delta drift and semantic coverage)
# ─────────────────────────────────────────────────────────────

_EMBED_MODEL = None
_EMBED_MODEL_NAME: Optional[str] = None


def _embed(texts: Sequence[str], model_name: str = _DEFAULT_EMBED_MODEL):
    """Normalised embeddings for ``texts`` (list of lists). Raises on failure —
    callers wrap this so monitoring metrics degrade to ``None`` rather than
    crash."""
    global _EMBED_MODEL, _EMBED_MODEL_NAME
    if _EMBED_MODEL is None or _EMBED_MODEL_NAME != model_name:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_name)
        _EMBED_MODEL_NAME = model_name
    vecs = _EMBED_MODEL.encode(list(texts), normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


# ─────────────────────────────────────────────────────────────
# Text delta (MONITORING ONLY — never a stop signal, spec §11.3)
# ─────────────────────────────────────────────────────────────

def embedding_drift(
    prev_text: str,
    curr_text: str,
    model_name: str = _DEFAULT_EMBED_MODEL,
) -> Dict[str, Any]:
    """Cosine drift between two guideline versions (monitoring only).

    Fully guarded: any failure (model not installed, offline, OOM) degrades to
    ``{"drift": None, "error": ...}`` and is logged — a monitoring metric must
    never crash the loop or leak into the stop decision.
    """
    try:
        v = _embed([prev_text, curr_text], model_name)
        cos = _cosine(v[0], v[1])
        return {"cosine_similarity": round(cos, 6),
                "drift": round(1.0 - cos, 6), "model": model_name}
    except Exception as exc:  # noqa: BLE001 — monitoring metric, swallow everything
        logger.warning("embedding drift unavailable (%s); skipping (monitoring only).", exc)
        return {"cosine_similarity": None, "drift": None,
                "model": model_name, "error": str(exc)}


def text_delta(
    prev_guideline: Optional[Path],
    curr_guideline: Path,
    rules_added: int,
    table_injections: int,
    drift: bool = True,
    model_name: str = _DEFAULT_EMBED_MODEL,
) -> Dict[str, Any]:
    """Guideline text delta (spec §11.3 "Guideline text delta", monitoring only).

    ``rules_added``/``table_injections`` count what produced ``curr_guideline``
    (the previous iteration's accepted amendments). Embedding drift compares
    ``curr_guideline`` against ``prev_guideline`` (skipped for G0 / when off).
    """
    out: Dict[str, Any] = {"rules_added": rules_added, "table_injections": table_injections}
    if drift and prev_guideline is not None and Path(prev_guideline).exists():
        out.update(embedding_drift(
            Path(prev_guideline).read_text(encoding="utf-8"),
            Path(curr_guideline).read_text(encoding="utf-8"),
            model_name=model_name,
        ))
    else:
        out["drift"] = None
    return out


# ─────────────────────────────────────────────────────────────
# Rule coverage vs enumerated expert disambiguations (§11.5b, monitoring only)
# ─────────────────────────────────────────────────────────────

def _norm_label(s: str) -> str:
    return " ".join((s or "").strip().upper().split())


def load_expert_rules(path: Path) -> List[Dict[str, Any]]:
    """Load the enumerated expert disambiguation rules (§11.5b reference set).

    Accepts two formats:
      * JSON list:  [{"id"?, "labels": ["A", "B"], "rule"?: "..."}, ...]
        (also tolerates {"rules": [...]} wrapping or "label_a"/"label_b" keys).
      * CSV with a header containing label_a,label_b[,rule[,id]].

    Each rule is normalised to {"id", "pair" (frozenset of 2 labels), "labels"
    (sorted list), "rule" (text)}. Rules without a clean label pair are dropped
    with a warning (a disambiguation is between two competing types).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    items: List[Dict[str, Any]] = []

    if path.suffix.lower() == ".csv" or (raw and not raw.lstrip().startswith(("[", "{"))):
        import csv
        import io
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            low = {(k or "").strip().lower(): v for k, v in row.items()}
            items.append({
                "id": low.get("id"),
                "labels": [low.get("label_a", ""), low.get("label_b", "")],
                "rule": low.get("rule", "") or low.get("disambiguation", ""),
            })
    else:
        doc = json.loads(raw)
        if isinstance(doc, dict):
            doc = doc.get("rules", doc.get("disambiguations", []))
        for d in doc:
            labels = d.get("labels")
            if not labels:
                labels = [d.get("label_a", ""), d.get("label_b", "")]
            items.append({"id": d.get("id"), "labels": labels, "rule": d.get("rule", "")})

    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        labs = [_norm_label(x) for x in (it.get("labels") or []) if _norm_label(x)]
        if len(set(labs)) != 2:
            logger.warning("expert rule #%d has no clean label pair (%r) — skipping.",
                           i, it.get("labels"))
            continue
        out.append({
            "id": it.get("id") or f"rule_{i}",
            "pair": frozenset(labs),
            "labels": sorted(set(labs)),
            "rule": (it.get("rule") or "").strip(),
        })
    return out


def rule_coverage(
    expert_rules: List[Dict[str, Any]],
    covered_triples: Sequence[Tuple[str, str, str]],
    semantic: bool = False,
    semantic_threshold: float = 0.5,
    model_name: str = _DEFAULT_EMBED_MODEL,
) -> Dict[str, Any]:
    """Fraction of enumerated expert disambiguations matched (§11.5b).

    ``covered_triples`` = the CUMULATIVE accepted (annotator_label, critic_label,
    decision_test) the loop has folded into the guideline so far.

    Two coverage levels:
      * ``pair_coverage`` — an expert rule is matched if the loop produced any
        accepted decision_test for the same competing label PAIR. Cheap and
        deterministic.
      * ``semantic_coverage`` (optional) — among pair-matched rules, the expert
        rule text must also be embedding-similar (cosine ≥ threshold) to one of
        the reconstructed decision_tests for that pair. Guarded; if embeddings
        are unavailable it degrades to ``None`` and only pair coverage is kept.
    """
    # Map label-pair → list of reconstructed decision_test texts.
    by_pair: Dict[frozenset, List[str]] = {}
    for a, b, test in covered_triples:
        pa, pb = _norm_label(a), _norm_label(b)
        if pa and pb and pa != pb:
            by_pair.setdefault(frozenset((pa, pb)), []).append(test or "")

    total = len(expert_rules)
    per_rule: List[Dict[str, Any]] = []
    n_pair = 0
    n_sem = 0
    sem_ok = semantic and total > 0

    for er in expert_rules:
        tests = by_pair.get(er["pair"], [])
        pair_matched = len(tests) > 0
        n_pair += int(pair_matched)
        row = {"id": er["id"], "labels": er["labels"],
               "pair_matched": pair_matched, "semantic_matched": None,
               "best_cosine": None}
        if sem_ok and pair_matched and er["rule"]:
            try:
                vecs = _embed([er["rule"]] + tests, model_name)
                best = max(_cosine(vecs[0], v) for v in vecs[1:])
                row["best_cosine"] = round(best, 4)
                row["semantic_matched"] = bool(best >= semantic_threshold)
                n_sem += int(row["semantic_matched"])
            except Exception as exc:  # noqa: BLE001 — monitoring only
                logger.warning("semantic coverage unavailable (%s); pair-only.", exc)
                sem_ok = False
        per_rule.append(row)

    return {
        "n_expert_rules": total,
        "pair_coverage": round(n_pair / total, 4) if total else None,
        "n_pair_matched": n_pair,
        "semantic_coverage": (round(n_sem / total, 4) if (sem_ok and total) else None),
        "n_semantic_matched": (n_sem if sem_ok else None),
        "semantic_threshold": semantic_threshold if semantic else None,
        "per_rule": per_rule,
    }


# ─────────────────────────────────────────────────────────────
# Pure decision logic (the dual stopping rule)
# ─────────────────────────────────────────────────────────────

def relative_reductions(series: Sequence[float]) -> List[float]:
    """Relative reduction r_k = (s[k-1] - s[k]) / s[k-1] for each consecutive pair.

    A positive value = friction fell; ~0 = plateau; negative = friction rose.
    When the previous value is 0 (no friction left to reduce) the reduction is 0
    if it stays at 0, else -1.0 (friction reappeared) — both count as "< eps".
    """
    out: List[float] = []
    for k in range(1, len(series)):
        prev, cur = series[k - 1], series[k]
        if prev == 0:
            out.append(0.0 if cur == 0 else -1.0)
        else:
            out.append((prev - cur) / prev)
    return out


def consecutive_tail_decreases(series: Sequence[Optional[float]]) -> int:
    """Number of consecutive STRICT decreases at the tail of ``series``.

    A ``None`` (no F1 signal that iteration) or any non-decrease breaks the
    chain, so a plateau near the ceiling does NOT trip the guard — only a genuine
    sustained drop does.
    """
    count = 0
    for k in range(len(series) - 1, 0, -1):
        a, b = series[k - 1], series[k]
        if a is None or b is None:
            break
        if b < a:
            count += 1
        else:
            break
    return count


def decide_stop(
    friction_series: Sequence[float],
    f1_series: Sequence[Optional[float]],
    eps: float = 0.05,
    friction_patience: int = 2,
    f1_guard_patience: int = 2,
) -> Dict[str, Any]:
    """Evaluate the dual stopping rule over the metric history so far.

    ``friction_series[i]`` = total round-1 disagreements under guideline G_i.
    ``f1_series[i]``       = held-out agent-vs-expert F1 under G_i (``None`` if
                              the held-out split is not configured / no overlap).

    Returns a dict: ``stop`` (bool), ``reason`` (str|None), ``echo_chamber``
    (bool — guard fired), ``primary_converged`` (bool — friction converged), and
    the computed ``friction_reductions`` / ``f1_consecutive_decreases`` for the
    report. The GUARD is checked first and OVERRIDES the primary rule.
    """
    reductions = relative_reductions(friction_series)
    decreases = consecutive_tail_decreases(f1_series)
    result: Dict[str, Any] = {
        "stop": False,
        "reason": None,
        "echo_chamber": False,
        "primary_converged": False,
        "friction_reductions": [round(r, 4) for r in reductions],
        "f1_consecutive_decreases": decreases,
    }

    # GUARD (overrides): held-out F1 fell for f1_guard_patience consecutive iters.
    if decreases >= f1_guard_patience:
        result["stop"] = True
        result["echo_chamber"] = True
        result["reason"] = (
            f"GUARD: held-out F1 decreased for {decreases} consecutive iterations "
            f"(≥ {f1_guard_patience}) — echo-chamber behaviour; stopping immediately."
        )
        return result

    # PRIMARY: last `friction_patience` relative reductions all below eps.
    if len(reductions) >= friction_patience:
        window = reductions[-friction_patience:]
        if all(r < eps for r in window):
            result["stop"] = True
            result["primary_converged"] = True
            pct = ", ".join(f"{r:+.1%}" for r in window)
            result["reason"] = (
                f"PRIMARY: friction converged — last {friction_patience} relative "
                f"reductions in round-1 disagreements ({pct}) all < {eps:.0%}."
            )
    return result


def best_f1_iteration(f1_series: Sequence[Optional[float]]) -> Optional[int]:
    """Index of the highest held-out F1 (the guideline to recommend if the guard
    fired). ``None`` when there is no F1 signal at all."""
    best_i, best_v = None, None
    for i, v in enumerate(f1_series):
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_i, best_v = i, v
    return best_i


# ─────────────────────────────────────────────────────────────
# Reporting (friction trajectory ALONGSIDE held-out F1, spec §11.3 "REPORT")
# ─────────────────────────────────────────────────────────────

def build_report(per_iter: List[Dict[str, Any]], decision: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the stopping-rule report dict from the per-iteration metric rows.

    Each row in ``per_iter`` is expected to carry: ``iteration``, ``guideline``,
    ``friction`` (dict), ``held_out_f1`` (dict|None), ``text_delta`` (dict|None),
    ``rule_coverage`` (dict|None).
    """
    friction_r1 = [row["friction"]["r1_disagreements"] for row in per_iter]
    f1_series = [
        (row.get("held_out_f1") or {}).get("f1") if row.get("held_out_f1") else None
        for row in per_iter
    ]
    coverage_series = [
        (row.get("rule_coverage") or {}).get("pair_coverage") if row.get("rule_coverage") else None
        for row in per_iter
    ]
    return {
        "friction_r1_series": friction_r1,
        "held_out_f1_series": f1_series,
        "rule_pair_coverage_series": coverage_series,
        "friction_reductions": decision.get("friction_reductions"),
        "f1_consecutive_decreases": decision.get("f1_consecutive_decreases"),
        "best_held_out_f1_iteration": best_f1_iteration(f1_series),
        "echo_chamber_flag": decision.get("echo_chamber", False),
        "primary_converged": decision.get("primary_converged", False),
        "stop_reason": decision.get("reason"),
        "iterations": per_iter,
    }


def format_trajectory(report: Dict[str, Any]) -> str:
    """Human-readable friction ∥ held-out-F1 (∥ coverage) trajectory table (spec
    §11.3 REPORT: so readers see whether 'the system decided it was done'
    coincided with 'quality plateaued near the ceiling')."""
    rows = report.get("iterations", [])
    lines = [
        "Stopping-rule trajectory (friction ∥ held-out F1 ∥ coverage):",
        f"  {'iter':>4}  {'guide':<6}{'R1 dis':>8}{'Δfrict':>9}{'held-F1':>9}"
        f"{'ΔF1':>8}{'cover':>8}{'rules+':>8}{'drift':>8}",
    ]
    prev_r1: Optional[float] = None
    prev_f1: Optional[float] = None
    for row in rows:
        fr = row["friction"]
        r1 = fr["r1_disagreements"]
        d_frict = "—" if prev_r1 in (None, 0) else f"{(prev_r1 - r1) / prev_r1:+.0%}"
        hf = row.get("held_out_f1") or {}
        f1 = hf.get("f1")
        f1s = f"{f1:.3f}" if f1 is not None else "—"
        d_f1 = (f"{f1 - prev_f1:+.3f}" if (f1 is not None and prev_f1 is not None) else "—")
        rc = row.get("rule_coverage") or {}
        cov = rc.get("pair_coverage")
        cov_s = f"{cov:.2f}" if cov is not None else "—"
        td = row.get("text_delta") or {}
        rules = td.get("rules_added")
        rules_s = "—" if rules is None else str(rules)
        drift = td.get("drift")
        drift_s = f"{drift:.3f}" if isinstance(drift, (int, float)) else "—"
        lines.append(
            f"  {row['iteration']:>4}  {str(row.get('guideline', '')):<6}"
            f"{r1:>8}{d_frict:>9}{f1s:>9}{d_f1:>8}{cov_s:>8}{rules_s:>8}{drift_s:>8}"
        )
        prev_r1, prev_f1 = r1, (f1 if f1 is not None else prev_f1)

    best = report.get("best_held_out_f1_iteration")
    if report.get("stop_reason"):
        lines.append(f"  stop: {report['stop_reason']}")
    if report.get("echo_chamber_flag"):
        lines.append(
            f"  ⚠ ECHO-CHAMBER flagged — recommend guideline from iteration {best} "
            f"(highest held-out F1)."
        )
    return "\n".join(lines)