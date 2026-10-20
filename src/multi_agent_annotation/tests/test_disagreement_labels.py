"""
Tests for the Critic's polymorphic disagreement labels and the removal sentinel.

Two defects these lock out, both found on a 5-sentence Kimi run:

1. `annotator_label` / `proposed_label` were validated with the ENTITY
   canonicaliser, but the Critic disputes entity types, relation labels and whole
   annotations through the same two strings. 78 off-schema warnings were reported
   on that run; 74 described correct Critic behaviour and none of the labels
   reached the output, so the number described nothing in the data.

2. `_disagreement_matches_relation` reads `proposed_label == "invalid"` as "delete
   this relation", and no prompt ever said so. The model emitted REMOVE x16,
   REMOVE_SPAN x6, none x4 and "IGNORE / not annotated" x3, and `invalid` zero
   times — so the structured deletion path never fired once. Removals only
   happened when the Annotator voluntarily acted on the prose explanation in a
   later round, which a single-round run would not have.

Run with:
    pytest src/multi_agent_annotation/tests/test_disagreement_labels.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "src" / "multi_agent_annotation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import src.multi_agent_annotation.multi_agent_annotation_ag2 as M  # noqa: E402


@pytest.fixture(autouse=True)
def _tool_state():
    schema = ast.literal_eval(
        (_REPO / "src/resources_updated/relation_schema_new.py").read_text().strip())
    M._init_tool_state(
        schema, [], {},
        relation_guide=M.load_relation_guide(M._DEFAULT_RELATION_GUIDE),
        guideline_search_backend="lexical")
    M.reset_off_schema_labels()
    yield
    M.reset_off_schema_labels()


def _proposed(value: str) -> str:
    return M.CriticDisagreement(target="x", proposed_label=value).proposed_label


# ------------------------------------------------------------------ #
#  The three things the field can legitimately hold                   #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("raw,expected", [
    ("BIOTIC_ENTITY", "BIOTIC ENTITY"),
    ("anthropogenic property", "ANTHROPOGENIC PROPERTY"),
])
def test_entity_types_are_canonicalised(raw, expected):
    assert _proposed(raw) == expected
    assert not M.off_schema_label_report()


@pytest.mark.parametrize("raw", ["HAS_PROPERTY", "ADDING_TO", "SUBSET_OF", "USES"])
def test_relation_names_are_accepted_not_flagged(raw):
    """Disputing a triplet's label is correct Critic behaviour, not an error."""
    assert _proposed(raw) == raw
    assert not M.off_schema_label_report(), \
        "a relation name in the disagreement field must not count as an off-schema entity"


@pytest.mark.parametrize("raw", [
    "REMOVE", "remove", "REMOVE_SPAN", "delete", "none",
    "IGNORE / not annotated", "invalid",
])
def test_removal_verdicts_map_onto_the_sentinel(raw):
    """Whatever the model calls deletion, it has to reach the one value the
    deliberation loop actually tests for."""
    assert _proposed(raw) == M.REMOVAL_SENTINEL
    assert not M.off_schema_label_report()


def test_genuine_garbage_is_still_reported():
    """The report must keep working for values that are none of the three kinds."""
    out = _proposed("pests HAS_PROPERTY distribution patterns")
    assert out == "PESTS HAS PROPERTY DISTRIBUTION PATTERNS"
    assert M.off_schema_label_report() == {"PESTS HAS PROPERTY DISTRIBUTION PATTERNS": 1}


# ------------------------------------------------------------------ #
#  The sentinel actually triggers deletion                            #
# ------------------------------------------------------------------ #

def _relation():
    return M.RelationFlat(relation="USES", e1_text="growers", e1_type="ANTHROPOGENIC ENTITY",
                          e2_text="fruits", e2_type="BIOTIC ENTITY")


@pytest.mark.parametrize("verdict", ["invalid", "REMOVE", "REMOVE_SPAN", "none"])
def test_removal_verdicts_reach_the_deletion_check(verdict):
    d = M.CriticDisagreement(target="growers USES fruits",
                             annotator_label="USES", proposed_label=verdict)
    assert M.MultiAgentAnnotator._disagreement_matches_relation(d, _relation()), \
        f"{verdict!r} must be read as a removal request"


def test_a_relabel_is_not_a_removal():
    """Proposing a different relation is a relabel, and must still match as a
    disagreement without being confused for deletion."""
    d = M.CriticDisagreement(target="growers USES fruits",
                             annotator_label="USES", proposed_label="PARTICIPATES_IN")
    assert d.proposed_label == "PARTICIPATES_IN"
    assert d.proposed_label != M.REMOVAL_SENTINEL


# ------------------------------------------------------------------ #
#  The prompts tell the Critic about it                               #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("builder", [
    "_critic_system_msg", "_critic_system_msg_strict", "_critic_system_msg_coldstart",
])
def test_critic_prompts_document_the_sentinel(builder):
    import prompts
    msg = getattr(prompts, builder)("GL", "ES", {}, relation_types=["AFFECTS", "CAUSES"])
    assert '"invalid"' in msg, "the removal sentinel must be documented"
    assert '"proposed_label": "invalid"' in msg, "and shown in the JSON example"
    assert "ENTITY TYPE" in msg and "RELATION NAME" in msg, \
        "the prompt must say which vocabulary the field takes"
