"""
The prompt contract for the blind-pair pipeline.

In that design all three agents — Annotator A, Annotator B and the Critic — are
independent annotators with BYTE-IDENTICAL system prompts. Only two things
differ: the Critic additionally receives the abstract, and the Critic gets a
second turn after the other two annotations are revealed. Everything else must
be the same, because the headline measurement is agent-agent agreement compared
against the 0.497 two-human ceiling, and that comparison is void if the two
agents were asked for different things.

TWO BUGS THIS FILE EXISTS TO PREVENT, both of which already happened:

1. _annotator_system_msg_solo accepted relation_search_mandatory and
   relation_types and forwarded NEITHER. Every single_pass run therefore
   rendered "No relation vocabulary is loaded for this run. Annotate entities
   only, and do not emit relations." while simultaneously keeping "All
   annotated entities must be linked to at least one relation. If an entity has
   no relations, it is likely a false positive and should be removed." Two
   contradictory instructions, in the arm used as the control for every
   deliberation claim, and absent from the arm it was controlling for.

2. The guideline is pasted in full (18KB total), so guideline_search is
   disabled — but a prompt that still instructs agents to call an unregistered
   tool makes them fail the call or improvise. Removing the references by hand
   across several prompt branches is easy to get half-right, so the absence is
   asserted rather than assumed.

    pytest src/multi_agent_annotation/tests/test_blind_pair_prompts.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
for _p in (_PKG, _PKG.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import prompts as P  # noqa: E402

RELATIONS = ["HAS_PROPERTY", "HAS_ROLE", "LOCATED_IN", "PART_OF", "USES"]


def blind(**kw):
    """The prompt every blind-pair agent receives."""
    return P._annotator_system_msg_solo(
        "GUIDELINE TEXT", "ENTITY SCHEMA", {},
        relation_types=RELATIONS, guideline_search_enabled=False, **kw)


# ── relations are loaded, requested, and required ───────────────────────────

def test_relation_vocabulary_is_rendered():
    out = blind()
    for r in RELATIONS:
        assert r in out, f"{r} missing from the prompt"


def test_the_no_vocabulary_branch_does_not_fire():
    """This sentence appearing is the bug: it told the solo arm to emit no
    relations while another rule required every entity to have one."""
    assert "No relation vocabulary is loaded" not in blind()


def test_relations_are_requested():
    out = blind()
    assert "Annotate every relation" in out or "emit relation" in out


def test_entities_must_be_linked_by_a_relation():
    assert "must be linked to at least one relation" in blind()


def test_solo_and_pipeline_prompts_agree_on_the_relation_vocabulary():
    """The solo variant is derived from the pipeline one so the two cannot
    drift; dropping the kwargs on the way through was exactly that drift."""
    solo = blind()
    pipe = P._annotator_system_msg("GUIDELINE TEXT", "ENTITY SCHEMA", {},
                                   relation_types=RELATIONS,
                                   guideline_search_enabled=False)
    for r in RELATIONS:
        assert (r in solo) == (r in pipe)


# ── no over-generation, no promised correction ──────────────────────────────

def test_no_over_generation_mandate():
    """Two blind annotators told to over-produce would disagree because they
    were instructed to, which makes agreement incomparable to the human
    ceiling: Mark and Davnah were not told to over-produce."""
    out = blind()
    assert "MAXIMUM COVERAGE" not in out
    assert "over-annotate" not in out
    assert "err on the side of inclusion" not in out
    assert "Prefer more entities over fewer" not in out


def test_no_promise_that_a_critic_will_correct_it():
    out = blind()
    assert "the Critic will filter" not in out
    assert "must stand on its own" in out


# ── the tool is gone, everywhere ────────────────────────────────────────────

def test_guideline_search_is_mentioned_nowhere_when_disabled():
    assert "guideline_search" not in blind()


def test_disabled_prompt_points_at_the_pasted_guideline_instead():
    assert "reproduced in full" in blind()


def test_guideline_search_still_present_when_enabled():
    """Guards the guard: the absence above must come from the flag, not from the
    tool having quietly disappeared from the prompt altogether."""
    out = P._annotator_system_msg_solo(
        "G", "S", {}, relation_types=RELATIONS, guideline_search_enabled=True)
    assert "guideline_search" in out


@pytest.mark.parametrize("mandatory", [True, False])
def test_mandatory_flag_cannot_resurrect_a_disabled_tool(mandatory):
    out = P._annotator_system_msg_solo(
        "G", "S", {}, relation_types=RELATIONS,
        guideline_search_mandatory=mandatory, guideline_search_enabled=False)
    assert "guideline_search" not in out


# ── the identity requirement ────────────────────────────────────────────────

def test_the_two_annotators_receive_an_identical_prompt():
    """A and B differ only by sampling. If anything in the prompt distinguishes
    them, their disagreement measures the prompt, not the model."""
    assert blind() == blind()


def test_the_relation_guide_really_has_a_vocabulary_to_load():
    """A silently empty relation guide puts the prompt straight back into the
    'no relation vocabulary' branch this file is about."""
    import multi_agent_annotation_ag2 as M
    if not M._DEFAULT_RELATION_GUIDE.exists():
        pytest.skip("relation guide not present")
    titles = [s["title"] for s in M.load_relation_guide(M._DEFAULT_RELATION_GUIDE)]
    assert len(titles) > 10, f"only {len(titles)} relations loaded"
    assert "No relation vocabulary is loaded" not in P._annotator_system_msg_solo(
        "G", "S", {}, relation_types=titles, guideline_search_enabled=False)


# ── the two framings must stay separated ───────────────────────────────────
#
# The standalone framing is now SELECTED (standalone=True picks _FRAMING_STANDALONE,
# _STEP1_STANDALONE, and drops _COVERAGE_PREFER_MORE) rather than regex-substituted
# out of the rendered pipeline prompt after the fact. That removes the failure mode
# these assertions were originally written for — a reworded paragraph silently
# un-matching, leaving the blind agents with the over-production mandate.
#
# They are kept, pinned from BOTH sides, because the separation still has to hold:
# the pipeline arm must KEEP this text (it is a real arm of the experiment and
# changing it changes the comparison) and the blind arm must not have it. A single
# constant accidentally shared between the two framings fails here.

PIPELINE_ONLY = [
    "It is far better to over-annotate than to miss entities or relations",
    "Your primary objective is MAXIMUM COVERAGE",
    "the Critic will filter errors later",
    "err on the side of inclusion",
    "Prefer more entities over fewer",
]


@pytest.mark.parametrize("text", PIPELINE_ONLY)
def test_pipeline_arm_keeps_the_over_generation_mandate(text):
    """The pipeline arm is a real arm of the experiment: it is SUPPOSED to tell
    the Annotator to over-produce because the Critic will filter. If this fails,
    the multi-arm prompt changed and is no longer comparable to the archived
    runs — which is a result-affecting change, not a cleanup."""
    pipe = P._annotator_system_msg("G", "S", {}, relation_types=RELATIONS,
                                   guideline_search_enabled=False)
    assert text in pipe, (
        f"the pipeline prompt no longer contains {text!r}; the solo variant's "
        f"removal of it has silently stopped doing anything")


@pytest.mark.parametrize("text", PIPELINE_ONLY)
def test_blind_prompt_has_none_of_it(text):
    assert text not in blind()


def test_the_standalone_flag_actually_selects_the_other_framing():
    """Guards against standalone=True being accepted and ignored — the failure
    mode a default-valued boolean parameter invites."""
    pipe = P._annotator_system_msg("G", "S", {}, relation_types=RELATIONS,
                                   guideline_search_enabled=False)
    solo = blind()
    assert solo != pipe
    assert "must stand on its own" in solo and "must stand on its own" not in pipe
    assert "identify the meaningful spans it contains" in solo
    # the one Coverage Rule that must SURVIVE the strip
    assert "must be linked to at least one relation" in solo


def test_coverage_rules_keeps_only_the_link_requirement():
    """The '## Coverage Rules' heading survives in the solo prompt but should now
    hold exactly one rule — the entity-must-link-to-a-relation requirement."""
    solo = blind()
    lines = solo.splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("## Coverage Rules"))
    j = next((k for k in range(i + 1, len(lines))
              if lines[k].strip().startswith("## ")), len(lines))
    bullets = [l.strip() for l in lines[i + 1:j] if l.strip().startswith("-")]
    assert len(bullets) == 1, f"expected one rule, got {bullets}"
    assert "must be linked to at least one relation" in bullets[0]
