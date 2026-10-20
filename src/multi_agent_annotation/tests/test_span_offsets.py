"""
Tests for span offset resolution and per-round offset recording.

WHY THIS EXISTS
Agent output carries span TEXT only. For months every span-level evaluation
decided "did a human mark this span?" with substring containment
(`a == b or a in b or b in a`), so a one-token agent span matched any human span
containing that token: "forest" matched "forest ecosystem", "tropical forest",
"forest fragmentation". Measured against real offsets, that understated the agent
false-positive rate by roughly half (6-12% -> 14-21%) and understated it most for
short spans, which is where the effect under study lives. It also manufactured a
spurious finding — an apparent "models hallucinate single-token spans" gradient
of 13/7/4/6% that is flat (18/17/12/18%) once offsets are used.

The harness already resolved offsets correctly for the Adjudicator's entities via
_fill_char_offsets; it simply never did so for the Annotator's or Critic's
per-round output, which existed only as raw JSON inside `messages`. These tests
lock in the shared primitive, the ambiguity accounting that lets analysis decline
to trust a guess, and the fact that round output is recorded at all.

    pytest src/multi_agent_annotation/tests/test_span_offsets.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
for _p in (_PKG, _PKG / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import multi_agent_annotation_ag2 as M  # noqa: E402


# ── the shared primitive ────────────────────────────────────────────────────

def test_unique_occurrence():
    assert M._span_occurrences("the boreal forest is cold", "boreal forest") == [(4, 17)]


def test_all_occurrences_are_returned():
    """Ambiguity must be visible, not silently resolved to the first hit."""
    occ = M._span_occurrences("the forest and the forest edge", "forest")
    assert occ == [(4, 10), (19, 25)]


def test_absent_span_returns_empty():
    assert M._span_occurrences("a short sentence", "orographic rainfall") == []


def test_offsets_slice_back_to_the_span():
    sent = "Species richness peaked in montane forest."
    (s, e), = M._span_occurrences(sent, "montane forest")
    assert sent[s:e] == "montane forest"


def test_case_insensitive_but_offsets_point_at_the_original():
    """The models lowercase their own text field; offsets must still be right."""
    sent = "Orographic rainfall may form on the windward side."
    (s, e), = M._span_occurrences(sent, "orographic rainfall")
    assert sent[s:e] == "Orographic rainfall"


def test_whitespace_run_is_tolerated():
    """A sentence stored with a newline where the agent saw a space still matches."""
    sent = "3.3 Mapping priority\nareas Overall, the maps agreed."
    (s, e), = M._span_occurrences(sent, "priority areas")
    assert sent[s:e] == "priority\nareas"


def test_whitespace_run_does_not_shift_later_offsets():
    """The old implementation searched a whitespace-COLLAPSED copy and stored the
    position as if it indexed the original, so any multi-character whitespace run
    shifted everything after it. This is the regression that guards against."""
    sent = "alpha   beta gamma"          # three spaces: collapsing shortens it
    (s, e), = M._span_occurrences(sent, "gamma")
    assert sent[s:e] == "gamma"


def test_empty_text_is_not_a_match():
    assert M._span_occurrences("anything at all", "") == []
    assert M._span_occurrences("anything at all", "   ") == []


def test_regex_metacharacters_in_span_text_are_literal():
    sent = "the ratio (a+b) was computed"
    (s, e), = M._span_occurrences(sent, "(a+b)")
    assert sent[s:e] == "(a+b)"


# ── ambiguity accounting ────────────────────────────────────────────────────

def test_locate_reports_unambiguous():
    got = M._locate("the boreal forest is cold", "boreal forest")
    assert got == {"start": 4, "end": 17, "n_candidates": 1}


def test_locate_flags_ambiguity_while_still_choosing():
    """Analysis needs both: a usable position AND the warning not to trust it."""
    got = M._locate("the forest and the forest edge", "forest")
    assert got["n_candidates"] == 2
    assert (got["start"], got["end"]) == (4, 10)


def test_locate_marks_absent_spans_rather_than_dropping_them():
    """n_candidates == 0 is a hallucinated span form — the strongest kind of
    false positive, and the last thing that should vanish from the record."""
    got = M._locate("a short sentence", "orographic rainfall")
    assert got == {"start": None, "end": None, "n_candidates": 0}


# ── per-round recording ─────────────────────────────────────────────────────

SENT = "Species richness peaked in montane forest near the forest edge."


class _Ann:
    def __init__(self, entities):
        self.entities = entities


class _Ent:
    def __init__(self, text, entity_type, confidence=None):
        self.text, self.entity_type, self.confidence = text, entity_type, confidence


class _Dis:
    def __init__(self, target, severity="major"):
        self.target = target
        self.annotator_label = "BIOTIC ENTITY"
        self.proposed_label = "SPATIAL ENTITY"
        self.severity = severity


class _Miss:
    def __init__(self, text):
        self.text, self.entity_type = text, "BIOTIC PROPERTY"


class _Agree:
    def __init__(self, target):
        self.target, self.label = target, "BIOTIC ENTITY"


class _Crit:
    def __init__(self, dis=(), miss=(), agr=()):
        self.disagreements = list(dis)
        self.missing_annotations = list(miss)
        self.agreements = list(agr)


def test_annotator_round_carries_offsets_and_confidence():
    ann, crit = M._resolve_round_spans(
        SENT, 0, _Ann([_Ent("montane forest", "SPATIAL ENTITY", 0.9)]), None)
    assert crit is None
    assert ann["round"] == 1
    e, = ann["entities"]
    assert e["confidence"] == 0.9
    assert SENT[e["start"]:e["end"]] == "montane forest"
    assert e["n_candidates"] == 1


def test_critic_acts_stay_separated():
    """Agreement, disagreement and missing-span are different assertions;
    pooling them has hidden effects before."""
    _, crit = M._resolve_round_spans(
        SENT, 1, None,
        _Crit(dis=[_Dis("montane forest")], miss=[_Miss("Species richness")],
              agr=[_Agree("forest edge")]))
    assert crit["round"] == 2
    assert len(crit["disagreements"]) == 1
    assert len(crit["missing_annotations"]) == 1
    assert len(crit["agreements"]) == 1
    assert crit["disagreements"][0]["severity"] == "major"


def test_ambiguous_round_span_is_flagged():
    ann, _ = M._resolve_round_spans(
        SENT, 0, _Ann([_Ent("forest", "BIOTIC ENTITY")]), None)
    assert ann["entities"][0]["n_candidates"] == 2


def test_neither_agent_parsed_yields_nothing():
    assert M._resolve_round_spans(SENT, 0, None, None) == (None, None)


def test_record_has_the_new_fields_and_they_default_empty():
    rec = M.DeliberationRecord(sentence=SENT)
    assert rec.annotator_rounds == []
    assert rec.critic_rounds == []


# ── regression lock against the archived runs ───────────────────────────────

ARCHIVE = Path("/home/katinska/mobiko_nlp/data/auto_annotated_clean_multi_run"
               "/multi_pass_deepseek_1")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archived run not present")
def test_primitive_reproduces_archived_adjudicator_offsets():
    """_fill_char_offsets was 100% correct on 17,149 archived entities before the
    primitive was extracted out of it. Extracting must not have changed that."""
    import json
    checked = 0
    for line in ARCHIVE.open():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        sent = rec.get("sentence", "")
        for e in rec.get("final_entities") or []:
            s, t, txt = e.get("start"), e.get("end"), e.get("text") or ""
            if not isinstance(s, int) or not isinstance(t, int):
                continue
            occ = M._span_occurrences(sent, txt)
            if not occ:
                continue
            assert any(o[0] == s for o in occ), (txt, s, occ)
            checked += 1
    assert checked > 500, f"only {checked} entities checked"
