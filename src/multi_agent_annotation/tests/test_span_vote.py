"""
Tests for the majority vote and the definition of "contested".

The vote is the baseline the LLM Adjudicator has to beat. On the archived K=10
draws a majority rule settles 96.6% of type disagreements and the no-majority
case is 0.3% of spans, so if the Adjudicator cannot beat this it is cost and
opacity with no decision content. That makes the vote's correctness load-bearing
for a published comparison, not just plumbing.

Two things these tests are specifically about:

  detection first   35.7% of span observations come from only one or two of
                    three annotators, against 8.8% type disagreement. A vote
                    that resolved types and quietly dropped the rest would be
                    answering the minor question. Low-support mentions must
                    still be REPORTED, with their support count.

  no silent ties    A tie is the case where the vote has no answer. It must be
                    flagged, not rendered as a confident choice, and it must
                    never be resolved by the order the annotators were listed
                    in — that would make the result depend on argument order.

    pytest src/multi_agent_annotation/tests/test_span_vote.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from span_vote import cluster_mentions, iou, summarise, vote  # noqa: E402


def sp(text, etype, start, end):
    return {"text": text, "entity_type": etype, "start": start, "end": end}


# ── clustering ──────────────────────────────────────────────────────────────

def test_identical_spans_form_one_mention():
    a = [sp("forest", "BIOTIC ENTITY", 4, 10)]
    b = [sp("forest", "BIOTIC ENTITY", 4, 10)]
    assert len(cluster_mentions([a, b])) == 1


def test_overlapping_extents_are_the_same_mention():
    """"montane forest" and "forest" are one mention with an extent
    disagreement, not two separate entities."""
    a = [sp("montane forest", "SPATIAL ENTITY", 0, 14)]
    b = [sp("forest", "BIOTIC ENTITY", 8, 14)]
    assert iou((0, 14), (8, 14)) < 0.5
    # barely-overlapping spans stay separate at the default threshold
    assert len(cluster_mentions([a, b])) == 2
    # and merge when the overlap is substantial
    c = [sp("montane forest", "SPATIAL ENTITY", 0, 14)]
    d = [sp("montane forests", "SPATIAL ENTITY", 0, 15)]
    assert len(cluster_mentions([c, d])) == 1


def test_disjoint_spans_stay_separate():
    a = [sp("forest", "BIOTIC ENTITY", 0, 6)]
    b = [sp("rainfall", "ABIOTIC ENTITY", 20, 28)]
    assert len(cluster_mentions([a, b])) == 2


def test_spans_without_offsets_are_skipped():
    """Offsets are resolved by the harness; a span without them cannot be voted
    on and must not crash the vote or silently become position zero."""
    a = [{"text": "forest", "entity_type": "BIOTIC ENTITY"}]
    assert cluster_mentions([a]) == []


# ── detection: the dominant disagreement ────────────────────────────────────

def test_unanimous_span_is_in_the_majority_and_uncontested():
    s = sp("forest", "BIOTIC ENTITY", 0, 6)
    v = vote([[s], [s], [s]])
    assert len(v["entities"]) == 1
    assert v["mentions"][0]["support"] == 3
    assert v["mentions"][0]["contested_reasons"] == []
    assert v["contested"] == []


def test_two_of_three_is_included_but_contested():
    s = sp("forest", "BIOTIC ENTITY", 0, 6)
    v = vote([[s], [s], []])
    assert len(v["entities"]) == 1
    m = v["mentions"][0]
    assert m["support"] == 2 and m["in_majority"]
    assert m["contested_reasons"] == ["detection"]


def test_one_of_three_is_excluded_but_still_reported():
    """The single most common disagreement in this corpus. Dropping it from the
    output is correct; dropping it from the record would hide a third of the
    signal."""
    s = sp("forest", "BIOTIC ENTITY", 0, 6)
    v = vote([[s], [], []])
    assert v["entities"] == []
    assert len(v["mentions"]) == 1
    m = v["mentions"][0]
    assert m["support"] == 1 and not m["in_majority"]
    assert "detection" in m["contested_reasons"]
    assert v["contested"] == [m]


def test_min_support_is_configurable():
    s = sp("forest", "BIOTIC ENTITY", 0, 6)
    assert vote([[s], [], []], min_support=1)["entities"] != []
    assert vote([[s], [s], []], min_support=3)["entities"] == []


# ── type and extent ─────────────────────────────────────────────────────────

def test_type_majority_wins_and_is_recorded_as_contested():
    v = vote([[sp("forest", "BIOTIC ENTITY", 0, 6)],
              [sp("forest", "BIOTIC ENTITY", 0, 6)],
              [sp("forest", "SPATIAL ENTITY", 0, 6)]])
    m = v["mentions"][0]
    assert m["entity_type"] == "BIOTIC ENTITY"
    assert m["type_tied"] is False
    assert m["contested_reasons"] == ["type"]
    assert m["types_proposed"] == {"BIOTIC ENTITY": 2, "SPATIAL ENTITY": 1}


def test_three_way_type_split_is_flagged_as_tied():
    """0.3% of spans — the only case a majority rule genuinely cannot decide,
    and therefore the Adjudicator's irreducible job."""
    v = vote([[sp("forest", "BIOTIC ENTITY", 0, 6)],
              [sp("forest", "SPATIAL ENTITY", 0, 6)],
              [sp("forest", "CONCEPT", 0, 6)]])
    m = v["mentions"][0]
    assert m["type_tied"] is True
    assert m["support"] == 3
    assert "type" in m["contested_reasons"]


def test_tie_does_not_depend_on_annotator_order():
    """If the winner changed with argument order, the vote would be reporting
    the order it was called with."""
    x = sp("forest", "BIOTIC ENTITY", 0, 6)
    y = sp("forest", "SPATIAL ENTITY", 0, 6)
    a = vote([[x], [y]])["mentions"][0]["entity_type"]
    b = vote([[y], [x]])["mentions"][0]["entity_type"]
    assert a == b


def test_extent_disagreement_is_separate_from_type_disagreement():
    v = vote([[sp("montane forests", "SPATIAL ENTITY", 0, 15)],
              [sp("montane forest", "SPATIAL ENTITY", 0, 14)],
              [sp("montane forest", "SPATIAL ENTITY", 0, 14)]])
    m = v["mentions"][0]
    assert m["contested_reasons"] == ["extent"]
    assert m["extent"] == (0, 14), "the majority extent should win"
    assert m["type_agreement"] is True


def test_a_mention_can_be_contested_on_several_grounds_at_once():
    v = vote([[sp("montane forest", "SPATIAL ENTITY", 0, 14)],
              [sp("montane forests", "BIOTIC ENTITY", 0, 15)],
              []])
    assert set(v["mentions"][0]["contested_reasons"]) == {"detection", "extent", "type"}


# ── output shape ────────────────────────────────────────────────────────────

def test_voted_entities_carry_the_winning_extent_and_text():
    v = vote([[sp("montane forest", "SPATIAL ENTITY", 0, 14)],
              [sp("montane forest", "SPATIAL ENTITY", 0, 14)],
              [sp("forest", "SPATIAL ENTITY", 8, 14)]])
    e, = v["entities"]
    assert (e["start"], e["end"]) == (0, 14)
    assert e["text"] == "montane forest"


def test_the_rule_is_recorded_with_the_result():
    """A voted output whose rule is not stated cannot be reproduced or compared
    against a differently-parameterised vote."""
    v = vote([[sp("f", "BIOTIC ENTITY", 0, 1)]], n_annotators=3)
    assert v["rule"] == {"min_support": 2, "n_annotators": 3, "iou_threshold": 0.5}


def test_summarise_counts_each_disagreement_kind():
    v = vote([
        [sp("a", "BIOTIC ENTITY", 0, 1), sp("b", "SPATIAL ENTITY", 5, 6)],
        [sp("a", "BIOTIC ENTITY", 0, 1), sp("b", "CONCEPT", 5, 6)],
        [sp("a", "BIOTIC ENTITY", 0, 1)],
    ])
    s = summarise(v)
    assert s["mentions"] == 2
    assert s["unanimous"] == 1
    assert s["type_disagreement"] == 1
    assert s["detection_disagreement"] == 1


def test_empty_input_is_handled():
    v = vote([[], [], []])
    assert v["entities"] == [] and v["mentions"] == [] and v["contested"] == []
