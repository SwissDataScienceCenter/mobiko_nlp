"""
Tests for evaluation/eval_relation_agent.py — the pipeline scored against human
relation gold.

Covers what separates this from the symmetric IAA metric: direction (agent =
prediction, human = gold, so P and R are not interchangeable), the scope rule
(a sentence the agent left empty is recall loss, not an excluded sentence), the
per-mode consensus, and the exclusion of gold hyper-relations the output format
cannot express.

Run with:
    pytest src/multi_agent_annotation/tests/test_relation_agent.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
for _p in (_PKG, _PKG / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_relation_agent as RA  # noqa: E402

SENT = "birds live in forests"          # birds 0-5, forests 13-20


def _agent_jsonl(tmp_path: Path, entities, relations, sentence=SENT) -> Path:
    rec = {"sentence": sentence,
           "final_entities": [{"text": t, "entity_type": ty, "start": s, "end": e}
                              for s, e, ty, t in entities],
           "final_relations": [{"relation": lab,
                                "e1": {"start": a[0], "end": a[1], "entity_type": a[2], "text": a[3]},
                                "e2": {"start": b[0], "end": b[1], "entity_type": b[2], "text": b[3]}}
                               for lab, a, b in relations]}
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf8")
    return p


def _gold(entities, relations, sentence=SENT) -> dict:
    return {" ".join(sentence.lower().split()): {
        "text": sentence,
        "spans": [{"start_char": s, "end_char": e, "type": ty, "text": t}
                  for s, e, ty, t in entities],
        "relations": relations}}


def _rel(lab, a, b, level=0, e1_kind="entity", e2_kind="entity"):
    return {"relation": lab, "level": level, "e1_kind": e1_kind, "e2_kind": e2_kind,
            "e1": {"start_char": a[0], "end_char": a[1], "type": a[2], "text": a[3]},
            "e2": {"start_char": b[0], "end_char": b[1], "type": b[2], "text": b[3]}}


BIRDS = (0, 5, "BIOTIC ENTITY", "birds")
FORESTS = (13, 20, "SPATIAL ENTITY", "forests")


def test_agent_jsonl_is_read_into_the_native_shape(tmp_path):
    p = _agent_jsonl(tmp_path, [BIRDS, FORESTS], [("LOCATED_IN", BIRDS, FORESTS)])
    agent = RA.load_agent(p)
    sent = agent[" ".join(SENT.split())]
    assert [(s["start_char"], s["end_char"]) for s in sent["spans"]] == [(0, 5), (13, 20)]
    assert sent["relations"][0]["relation"] == "LOCATED_IN"
    assert sent["relations"][0]["level"] == 0


def test_precision_and_recall_are_not_interchangeable(tmp_path):
    """The agent doubles up: same gold found, twice the predictions."""
    p = _agent_jsonl(tmp_path, [BIRDS, FORESTS],
                     [("LOCATED_IN", BIRDS, FORESTS), ("AFFECTS", BIRDS, FORESTS)])
    gold = _gold([BIRDS, FORESTS], [_rel("LOCATED_IN", BIRDS, FORESTS)])
    r = RA.score(RA.load_agent(p), gold, "raw", "boundary")
    t = r["triplets"]["boundary"]
    assert t["tp"] == 1 and t["n_pred"] == 2 and t["n_gold"] == 1
    assert t["precision"] == 0.5 and t["recall"] == 1.0


def test_an_empty_agent_sentence_is_recall_loss_not_an_excluded_sentence(tmp_path):
    p = _agent_jsonl(tmp_path, [], [])
    gold = _gold([BIRDS, FORESTS], [_rel("LOCATED_IN", BIRDS, FORESTS)])
    r = RA.score(RA.load_agent(p), gold, "raw", "boundary")
    assert r["scope"]["scored_relations"] == 1
    assert r["scope"]["agent_empty_relations"] == 1
    assert r["triplets"]["boundary"]["n_gold"] == 1
    assert r["triplets"]["boundary"]["recall"] == 0.0


def test_gold_hyper_relations_are_excluded_and_counted(tmp_path):
    p = _agent_jsonl(tmp_path, [BIRDS, FORESTS], [("LOCATED_IN", BIRDS, FORESTS)])
    hyper = _rel("AFFECTS", BIRDS, FORESTS, level=1, e2_kind="relation")
    gold = _gold([BIRDS, FORESTS], [_rel("LOCATED_IN", BIRDS, FORESTS), hyper])
    r = RA.score(RA.load_agent(p), gold, "raw", "boundary")
    assert r["scope"]["gold_hyper_excluded"] == 1
    assert r["triplets"]["boundary"]["n_gold"] == 1      # the hyper one is not a miss
    assert r["triplets"]["boundary"]["f1"] == 1.0


def test_consensus_entities_are_per_mode(tmp_path):
    """Same span, two types: consensus under boundary, not under strict."""
    other_type = (0, 5, "CONCEPT", "birds")
    a = _gold([BIRDS], [])
    b = _gold([other_type], [])
    cons = RA.build_consensus([a, b], "boundary", "raw")
    p = _agent_jsonl(tmp_path, [BIRDS], [])
    r = RA.score(RA.load_agent(p), cons, "raw", "boundary", gold_pair=(a, b))
    assert r["entities"]["boundary"]["n_gold"] == 1
    assert r["entities"]["strict"]["n_gold"] == 0


def test_consensus_relations_need_both_annotators_and_the_same_label(tmp_path):
    a = _gold([BIRDS, FORESTS], [_rel("LOCATED_IN", BIRDS, FORESTS)])
    b = _gold([BIRDS, FORESTS], [_rel("AFFECTS", BIRDS, FORESTS)])
    assert RA.build_consensus([a, b], "boundary", "raw")[
        " ".join(SENT.split())]["relations"] == []
    b2 = _gold([BIRDS, FORESTS], [_rel("LOCATED_IN", BIRDS, FORESTS)])
    assert len(RA.build_consensus([a, b2], "boundary", "raw")[
        " ".join(SENT.split())]["relations"]) == 1


def test_recall_ceiling_counts_gold_relations_the_agent_could_reach(tmp_path):
    """The agent found only one endpoint, so that gold relation is unreachable."""
    p = _agent_jsonl(tmp_path, [BIRDS], [])
    gold = _gold([BIRDS, FORESTS], [_rel("LOCATED_IN", BIRDS, FORESTS)])
    r = RA.score(RA.load_agent(p), gold, "raw", "boundary")
    assert r["argument_ceiling"]["boundary"]["recall_ceiling"] == 0.0


def test_offsetless_and_relation_typed_gold_spans_do_not_count_as_entities(tmp_path):
    p = _agent_jsonl(tmp_path, [BIRDS], [])
    gold = _gold([BIRDS], [])
    key = next(iter(gold))
    gold[key]["spans"] += [
        {"start_char": None, "end_char": None, "type": "BIOTIC ENTITY", "text": ""},
        {"start_char": 13, "end_char": 20, "type": "RELATION: LOCATED_IN", "text": "forests"},
    ]
    r = RA.score(RA.load_agent(p), gold, "raw", "boundary")
    assert r["entities"]["boundary"]["n_gold"] == 1
    assert r["entities"]["boundary"]["f1"] == 1.0
