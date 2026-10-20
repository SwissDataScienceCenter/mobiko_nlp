"""
Tests for stopping_rule.py — the dual stopping rule (spec §11.3).

Pure-function / no-LLM, no-network coverage: the decision logic, friction
parsing, held-out F1 (reusing the Layer-1 offset matching), and label-pair rule
coverage. Embedding-dependent paths (drift, semantic coverage) are not exercised
here — they are guarded and monitoring-only.

Run with:
    pytest src/multi_agent_annotation/tests/test_stopping_rule.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# stopping_rule lives in the loop/ subdir and uses flat sibling imports (it adds
# the package root + evaluation/ to sys.path on import). Put loop/ + the package
# root on the path so it imports the same way the loop runs it.
_PKG = Path(__file__).resolve().parent.parent        # …/multi_agent_annotation
for _p in (_PKG, _PKG / "loop", _PKG / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import stopping_rule as sr  # noqa: E402


# ───────────────────────── relative_reductions ─────────────────────────

def test_relative_reductions_basic():
    assert sr.relative_reductions([100, 60, 58, 57]) == pytest.approx(
        [0.4, 2 / 60, 1 / 58])


def test_relative_reductions_zero_edge():
    # prev==0 & stays 0 → 0.0 (converged); prev==0 & reappears → -1.0 (counts < eps)
    assert sr.relative_reductions([0, 0, 5]) == [0.0, -1.0]


def test_relative_reductions_single_point():
    assert sr.relative_reductions([10]) == []


# ───────────────────────── consecutive_tail_decreases ─────────────────────────

def test_tail_decreases_counts_run():
    assert sr.consecutive_tail_decreases([0.7, 0.65, 0.6, 0.55]) == 3


def test_tail_decreases_plateau_breaks_chain():
    # equality is NOT a decrease — a plateau near the ceiling must not trip the guard
    assert sr.consecutive_tail_decreases([0.7, 0.71, 0.71]) == 0


def test_tail_decreases_none_breaks_chain():
    assert sr.consecutive_tail_decreases([0.7, 0.6, None]) == 0
    assert sr.consecutive_tail_decreases([None, 0.6, 0.5]) == 1


# ───────────────────────── decide_stop: PRIMARY ─────────────────────────

def test_primary_converges_after_two_subeps_reductions():
    d = sr.decide_stop([100, 60, 58, 57], [None, None, None, None])
    assert d["stop"] and d["primary_converged"] and not d["echo_chamber"]
    assert "PRIMARY" in d["reason"]


def test_primary_not_triggered_while_friction_drops():
    # reductions 40%, 50% — both >> eps → keep going
    d = sr.decide_stop([100, 60, 30], [None, None, None])
    assert not d["stop"]


def test_primary_needs_enough_data():
    # only one reduction available → cannot satisfy patience=2
    d = sr.decide_stop([100, 98], [None, None])
    assert not d["stop"]


def test_primary_patience_one():
    d = sr.decide_stop([100, 98], [None, None], friction_patience=1)
    assert d["stop"] and d["primary_converged"]


# ───────────────────────── decide_stop: GUARD ─────────────────────────

def test_guard_overrides_primary_on_two_f1_decreases():
    # friction is plummeting (primary would happily continue), but held-out F1
    # falls twice → echo-chamber guard fires and OVERRIDES.
    d = sr.decide_stop([100, 50, 20, 5], [0.70, 0.72, 0.68, 0.60])
    assert d["stop"] and d["echo_chamber"] and not d["primary_converged"]
    assert "GUARD" in d["reason"]


def test_guard_not_tripped_by_single_decrease():
    d = sr.decide_stop([100, 50, 20], [0.70, 0.72, 0.68])
    assert not d["stop"]


def test_guard_disabled_without_f1():
    # all-None F1 series (no held-out split) → guard can never fire
    d = sr.decide_stop([100, 50, 20], [None, None, None])
    assert not d["echo_chamber"]


def test_best_f1_iteration():
    assert sr.best_f1_iteration([0.6, 0.72, 0.68, 0.60]) == 1
    assert sr.best_f1_iteration([None, None]) is None


# ───────────────────────── friction_from_report ─────────────────────────

def test_friction_from_report_parses_r1_and_rounds_rate():
    report = {
        "n_sentences": 2,
        "per_step": {
            "disagreements_by_round": {1: 10, 2: 3},
            "total_critic_disagreements": 13,
            "total_disagreements_all_steps": 15,
        },
    }
    records = [{"rounds_used": 2, "messages": []}, {"rounds_used": 1, "messages": []}]
    f = sr.friction_from_report(report, records)
    assert f["r1_disagreements"] == 10
    assert f["total_critic_disagreements"] == 13
    assert f["total_disagreements_all_steps"] == 15
    assert f["mean_rounds_used"] == pytest.approx(1.5)
    assert f["rounds_used_rate"] == pytest.approx(0.5)  # 1 of 2 sentences used round 2


def test_friction_from_report_tolerates_str_round_keys():
    # reloaded-from-JSON reports have string round keys
    report = {"n_sentences": 1, "per_step": {"disagreements_by_round": {"1": 7}}}
    f = sr.friction_from_report(report, [{"rounds_used": 1, "messages": []}])
    assert f["r1_disagreements"] == 7


# ───────────────────────── held-out F1 (Layer-1 offset matching) ─────────────────────────

def _write(path: Path, obj) -> Path:
    if isinstance(obj, list):
        path.write_text("\n".join(json.dumps(o) for o in obj), encoding="utf-8")
    else:
        path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_held_out_f1_perfect_match(tmp_path):
    agent = _write(tmp_path / "agent.jsonl", [
        {"sentence": "Alpha beta gamma.",
         "final_entities": [{"text": "Alpha", "entity_type": "CONCEPT", "start": 0, "end": 5}]},
    ])
    human = _write(tmp_path / "Expert.json", {
        "doc_id": "x",
        "sentences": [{"text": "Alpha beta gamma.",
                       "spans": [{"start_char": 0, "end_char": 5, "type": "CONCEPT", "text": "Alpha"}]}],
    })
    r = sr.compute_held_out_f1(agent, [human], mode="strict")
    assert r["f1"] == pytest.approx(1.0)
    assert r["n_eval_sentences"] == 1
    assert "Expert" in r["per_expert"]


def test_held_out_f1_type_mismatch_strict_vs_boundary(tmp_path):
    agent = _write(tmp_path / "agent.jsonl", [
        {"sentence": "Alpha beta.",
         "final_entities": [{"text": "Alpha", "entity_type": "ABIOTIC ENTITY", "start": 0, "end": 5}]},
    ])
    human = _write(tmp_path / "Expert.json", {
        "doc_id": "x",
        "sentences": [{"text": "Alpha beta.",
                       "spans": [{"start_char": 0, "end_char": 5, "type": "CONCEPT", "text": "Alpha"}]}],
    })
    # boundary: span detected → F1 = 1.0; strict: wrong type → F1 = 0.0
    assert sr.compute_held_out_f1(agent, [human], mode="boundary")["f1"] == pytest.approx(1.0)
    assert sr.compute_held_out_f1(agent, [human], mode="strict")["f1"] == pytest.approx(0.0)


def test_held_out_f1_none_when_no_overlap(tmp_path):
    agent = _write(tmp_path / "agent.jsonl", [
        {"sentence": "Totally different sentence.",
         "final_entities": [{"text": "X", "entity_type": "CONCEPT", "start": 0, "end": 1}]},
    ])
    human = _write(tmp_path / "Expert.json", {
        "doc_id": "x",
        "sentences": [{"text": "Alpha beta.",
                       "spans": [{"start_char": 0, "end_char": 5, "type": "CONCEPT", "text": "Alpha"}]}],
    })
    r = sr.compute_held_out_f1(agent, [human], mode="strict")
    assert r["f1"] is None and r["n_eval_sentences"] == 0


# ───────────────────────── rule coverage (§11.5b, label-pair) ─────────────────────────

def test_load_expert_rules_json_and_drops_malformed(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps([
        {"id": "r1", "labels": ["CONCEPT", "ABIOTIC ENTITY"], "rule": "measured → abiotic"},
        {"id": "bad", "labels": ["CONCEPT"]},                 # only one label → dropped
        {"labels": ["BIOTIC ENTITY", "ANTHROPOGENIC PROCESS"]},  # auto id
    ]), encoding="utf-8")
    rules = sr.load_expert_rules(p)
    assert len(rules) == 2
    assert rules[0]["pair"] == frozenset({"CONCEPT", "ABIOTIC ENTITY"})
    assert rules[1]["id"]  # auto-assigned


def test_load_expert_rules_csv(tmp_path):
    p = tmp_path / "rules.csv"
    p.write_text("label_a,label_b,rule\nCONCEPT,ABIOTIC ENTITY,measured thing → abiotic\n",
                 encoding="utf-8")
    rules = sr.load_expert_rules(p)
    assert len(rules) == 1
    assert rules[0]["pair"] == frozenset({"CONCEPT", "ABIOTIC ENTITY"})


def test_rule_coverage_pair_fraction():
    rules = [
        {"id": "r1", "pair": frozenset({"CONCEPT", "ABIOTIC ENTITY"}),
         "labels": ["ABIOTIC ENTITY", "CONCEPT"], "rule": "x"},
        {"id": "r2", "pair": frozenset({"BIOTIC ENTITY", "ANTHROPOGENIC PROCESS"}),
         "labels": ["ANTHROPOGENIC PROCESS", "BIOTIC ENTITY"], "rule": "y"},
    ]
    covered = [("concept", "abiotic entity", "if measured → abiotic")]  # case-insensitive
    cov = sr.rule_coverage(rules, covered, semantic=False)
    assert cov["pair_coverage"] == pytest.approx(0.5)
    assert cov["n_pair_matched"] == 1
    assert cov["semantic_coverage"] is None
    matched = {r["id"]: r["pair_matched"] for r in cov["per_rule"]}
    assert matched == {"r1": True, "r2": False}


# ───────────────────────── report assembly ─────────────────────────

def test_build_report_and_format_trajectory():
    per_iter = [
        {"iteration": 0, "guideline": "G0.md",
         "friction": {"r1_disagreements": 100},
         "held_out_f1": {"f1": 0.60}, "text_delta": {"rules_added": 0, "drift": None},
         "rule_coverage": {"pair_coverage": 0.0}},
        {"iteration": 1, "guideline": "G1.md",
         "friction": {"r1_disagreements": 60},
         "held_out_f1": {"f1": 0.70}, "text_delta": {"rules_added": 3, "drift": 0.12},
         "rule_coverage": {"pair_coverage": 0.3}},
    ]
    decision = sr.decide_stop([100, 60], [0.60, 0.70])
    report = sr.build_report(per_iter, decision)
    assert report["friction_r1_series"] == [100, 60]
    assert report["held_out_f1_series"] == [0.60, 0.70]
    assert report["rule_pair_coverage_series"] == [0.0, 0.3]
    assert report["best_held_out_f1_iteration"] == 1
    text = sr.format_trajectory(report)
    assert "held-F1" in text and "G0.md" in text and "G1.md" in text
