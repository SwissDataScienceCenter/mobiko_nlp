"""
End-to-end test of the blind-pair orchestration, with canned model responses.

Drives annotate_sentence_blind through all five calls — Annotator A, Annotator B,
Critic blind, Critic revision, Adjudicator — by replacing _run_agent_turn, so the
control flow, the vote integration, the record assembly and the failure paths are
all exercised without a network call.

WHAT THIS IS GUARDING. Three properties the design depends on and which are easy
to break silently:

  call order        The Critic must annotate BEFORE it sees A and B. If the
                    revision were issued first, or the blind pass skipped, the
                    Critic would be the anchored reviewer the whole design exists
                    to replace, and nothing in the output would look wrong.

  vote membership   The majority vote must use A, B and critic_BLIND. Including
                    critic_revised would let one agent's view count twice and
                    turn a 2-1 "majority" into an agent agreeing with itself.

  both outputs      The majority vote and the Adjudicator must BOTH be recorded.
                    The vote is the baseline the Adjudicator has to beat; if the
                    pipeline kept only the adjudicated output, the comparison the
                    design was built for would be impossible after the fact.

    pytest src/multi_agent_annotation/tests/test_blind_pair_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
for _p in (_PKG, _PKG / "evaluation", _PKG.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import multi_agent_annotation_ag2 as M  # noqa: E402

SENT = "Centres of species richness correlate with areas of high temperatures."


def ann(entities):
    import json
    return json.dumps({"entities": entities, "relations": [],
                       "uncertain_cases": [], "reasoning": "ok"})


def ent(text, etype, conf=0.9):
    return {"text": text, "entity_type": etype, "confidence": conf}


def adj(entities, flagged=None):
    import json
    return json.dumps({"final_entities": entities, "final_relations": [],
                       "disagreement_resolutions": [],
                       "flagged_for_human_review": flagged or []})


class _Stub:
    def reset(self):
        pass


def make(responses):
    """A MultiAgentAnnotator with only what annotate_sentence_blind touches."""
    a = object.__new__(M.MultiAgentAnnotator)
    a.annotator = a.critic = a.adjudicator = _Stub()
    a.annotator_executor = a.critic_executor = a.adj_executor = _Stub()
    a.run_meta = {"deliberation_mode": "blind_pair"}
    a.context_radius = 1
    a._ctx_window = False          # no abstracts: sentence-only context
    a.calls = []

    def fake_turn(agent, executor, message):
        a.calls.append(message)
        text = responses[len(a.calls) - 1]
        return text, {"agent": "x", "content": text, "tool_calls": [],
                      "token_usage": {"prompt_tokens": 10,
                                      "completion_tokens": 5,
                                      "total_tokens": 15,
                                      "token_source": "api"},
                      "elapsed_s": 0.1, "mean_logprob": -0.2,
                      "mean_entropy": 0.3}
    a._run_agent_turn = fake_turn
    return a


AGREE = ann([ent("Centres", "SPATIAL ENTITY"),
             ent("species richness", "BIOTIC PROPERTY")])
DISAGREE = ann([ent("Centres", "CONCEPT"),
                ent("species richness", "BIOTIC PROPERTY")])
EXTRA = ann([ent("Centres", "SPATIAL ENTITY"),
             ent("species richness", "BIOTIC PROPERTY"),
             ent("areas", "SPATIAL ENTITY")])


# ── the five calls, in order ────────────────────────────────────────────────

def test_makes_exactly_five_calls():
    a = make([AGREE, AGREE, AGREE, AGREE,
              adj([ent("Centres", "SPATIAL ENTITY")])])
    a.annotate_sentence_blind(SENT)
    assert len(a.calls) == 5


def test_the_three_blind_calls_are_identical_tasks():
    """A, B and critic_blind differ by sampling alone. If their task messages
    differ, their disagreement measures the task."""
    a = make([AGREE, AGREE, AGREE, AGREE, adj([])])
    a.annotate_sentence_blind(SENT)
    assert a.calls[0] == a.calls[1] == a.calls[2]


def test_no_blind_call_mentions_another_annotation():
    a = make([AGREE, AGREE, AGREE, AGREE, adj([])])
    a.annotate_sentence_blind(SENT)
    for msg in a.calls[:3]:
        assert "Annotation A" not in msg
        assert "another annotator" not in msg


def test_the_critic_sees_the_peers_only_on_its_fourth_call():
    """The ordering that makes the Critic a third annotator rather than a
    reviewer. Reverse it and the design silently becomes the old one."""
    a = make([AGREE, DISAGREE, AGREE, AGREE, adj([])])
    a.annotate_sentence_blind(SENT)
    rev = a.calls[3]
    assert "another annotator, independent" in rev
    assert "Your annotation:" in rev


def test_the_adjudicator_is_called_with_the_contested_spans():
    a = make([EXTRA, AGREE, AGREE, AGREE, adj([])])
    a.annotate_sentence_blind(SENT)
    assert "Contested spans" in a.calls[4]
    assert "'areas'" in a.calls[4] or '"areas"' in a.calls[4]


# ── the record ──────────────────────────────────────────────────────────────

def test_all_four_annotations_are_recorded():
    a = make([AGREE, DISAGREE, AGREE, AGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    got = rec.blind_pair["annotations"]
    assert set(got) == {"annotator_a", "annotator_b",
                        "critic_blind", "critic_revised"}
    for r in got.values():
        assert r["parse_ok"] is True


def test_blind_and_revised_critic_are_kept_apart():
    """The delta between them IS the anchoring measurement; collapsing them into
    one field would make the mechanism claim unmeasurable."""
    a = make([AGREE, DISAGREE, DISAGREE, AGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    blind = rec.blind_pair["annotations"]["critic_blind"]["entities"]
    revised = rec.blind_pair["annotations"]["critic_revised"]["entities"]
    assert [e["entity_type"] for e in blind] != [e["entity_type"] for e in revised]


def test_the_vote_uses_the_blind_triple_not_the_revision():
    """critic_revised has seen A and B. Counting it would let one agent vote
    twice. Here A and B say SPATIAL ENTITY, critic_blind says CONCEPT, and the
    revision also says CONCEPT — if the revision were in the vote the type
    would be tied 2-2, and it must not be."""
    a = make([AGREE, AGREE, DISAGREE, DISAGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    centres = next(m for m in rec.blind_pair["majority"]["mentions"]
                   if m["text"] == "Centres")
    assert centres["entity_type"] == "SPATIAL ENTITY"
    assert centres["type_tied"] is False
    assert centres["types_proposed"] == {"SPATIAL ENTITY": 2, "CONCEPT": 1}


def test_both_majority_and_adjudicated_outputs_survive():
    a = make([EXTRA, AGREE, AGREE, AGREE,
              adj([ent("Centres", "SPATIAL ENTITY"),
                   ent("species richness", "BIOTIC PROPERTY"),
                   ent("areas", "SPATIAL ENTITY")])])
    rec = a.annotate_sentence_blind(SENT)
    assert len(rec.final_entities) == 3, "adjudicated output"
    assert len(rec.blind_pair["majority"]["entities"]) == 2, "majority output"
    assert rec.adjudication_status == "blind_pair"


def test_contested_mentions_carry_their_reasons():
    a = make([EXTRA, AGREE, DISAGREE, AGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    reasons = {m["text"]: m["contested_reasons"]
               for m in rec.blind_pair["majority"]["contested"]}
    assert reasons.get("areas") == ["detection"]
    assert "type" in reasons.get("Centres", [])


def test_anonymisation_maps_are_recorded():
    """Agents must not know who produced what; the experimenter must."""
    a = make([AGREE, AGREE, AGREE, AGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    anon = rec.blind_pair["anonymisation"]
    assert sorted(anon["critic_revision"].values()) == [0, 1]
    assert sorted(anon["adjudication"].values()) == [0, 1, 2]


def test_shuffle_is_reproducible_for_the_same_sentence():
    """Seeded from the sentence, not a counter, so the order does not depend on
    how many sentences came before it in the batch."""
    maps = []
    for _ in range(2):
        a = make([AGREE, AGREE, AGREE, AGREE, adj([])])
        maps.append(a.annotate_sentence_blind(SENT)
                    .blind_pair["anonymisation"]["adjudication"])
    assert maps[0] == maps[1]


def test_agreement_score_is_the_unanimous_fraction():
    a = make([EXTRA, AGREE, AGREE, AGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    sm = rec.blind_pair["majority"]["summary"]
    assert rec.agreement_score == pytest.approx(sm["unanimous"] / sm["mentions"])


def test_token_usage_is_per_role_and_totalled():
    a = make([AGREE, AGREE, AGREE, AGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    assert set(rec.token_usage) == {"annotator_a", "annotator_b", "critic_blind",
                                    "critic_revised", "adjudicator", "total"}
    assert rec.token_usage["total"]["completion_tokens"] == 25  # 5 calls x 5


# ── failure paths ───────────────────────────────────────────────────────────

def test_adjudicator_parse_failure_falls_back_to_the_vote():
    """A failed parse must not cost the sentence — and the fallback must be
    recorded, so those sentences can be excluded from the
    Adjudicator-vs-majority comparison rather than quietly counted as the
    Adjudicator agreeing with the vote."""
    a = make([AGREE, AGREE, AGREE, AGREE, "not json at all"])
    rec = a.annotate_sentence_blind(SENT)
    assert rec.adjudication_status == "blind_pair_adjudicator_parse_failed"
    assert rec.blind_pair["adjudicator_parse_ok"] is False
    assert len(rec.final_entities) == 2, "fell back to the majority output"


def test_an_annotator_parse_failure_is_recorded_not_fatal():
    a = make(["garbage", AGREE, AGREE, AGREE, adj([])])
    rec = a.annotate_sentence_blind(SENT)
    assert rec.blind_pair["annotations"]["annotator_a"]["parse_ok"] is False
    assert rec.blind_pair["annotations"]["annotator_b"]["parse_ok"] is True


def test_unresolvable_span_offsets_are_counted_not_dropped_silently():
    """A span whose surface form is not in the sentence is a hallucination and
    the strongest kind of false positive. It cannot be clustered, so it leaves
    the vote — but it must leave a trace."""
    a = make([ann([ent("peat bog", "ABIOTIC ENTITY")]), AGREE, AGREE, AGREE,
              adj([])])
    rec = a.annotate_sentence_blind(SENT)
    assert rec.blind_pair["unresolved_offsets"] == 1


def test_record_is_json_serialisable():
    import json
    a = make([EXTRA, DISAGREE, AGREE, AGREE, adj([ent("Centres", "CONCEPT")])])
    rec = a.annotate_sentence_blind(SENT)
    back = json.loads(rec.model_dump_json())
    assert "blind_pair" in back
    assert back["blind_pair"]["majority"]["rule"]["min_support"] == 2
