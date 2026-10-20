"""
Tests for the turn-termination check — the bug that silently disabled round 2.

The deliberation loop ran for months with `rounds_used = 2` recorded on ~90% of
sentences while the round-2 Annotator turn made no LLM call at all: 0 tokens,
0.0 seconds, empty content, on every model in every run.

Cause: the revision prompt embeds the agent's OWN round-1 JSON, and the
executor's `is_termination_msg` asked only "does this content parse as my
output?". It fired on the prompt's arrival and ended the turn before the model
was invoked. Reconstructed from stored runs, 98 of 98 qwen prompts tripped it.

Role- or name-based gating cannot fix this: the prompt and the agent's real
reply reach the executor from the same sender. Prompts are therefore tagged with
TASK_PROMPT_SENTINEL and the check refuses to terminate on a tagged message.

These tests lock in both directions — a tagged prompt never ends a turn, and
every genuine completion still does.

Run with:
    pytest src/multi_agent_annotation/tests/test_round_termination.py -v
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

ANNOTATION_JSON = (
    '{"entities": [{"text": "centres", "entity_type": "SPATIAL ENTITY", '
    '"confidence": 0.9}], "relations": [], "uncertain_cases": [], '
    '"reasoning": "ok"}'
)
CRITIC_JSON = (
    '{"agreements": [{"target": "centres", "label": "SPATIAL ENTITY"}], '
    '"disagreements": [{"target": "areas", "annotator_label": "SPATIAL ENTITY", '
    '"proposed_label": "ABIOTIC ENTITY", "severity": "major"}], '
    '"missing_annotations": [], "reasoning": "ok"}'
)


@pytest.fixture
def annotator_check():
    return M._make_structured_termination_check(
        M.MultiAgentAnnotator._parse_annotator_output)


@pytest.fixture
def critic_check():
    return M._make_structured_termination_check(
        M.MultiAgentAnnotator._parse_critic_output)


# ── 1. a tagged prompt must never end a turn ────────────────────────────────

def test_revision_prompt_does_not_terminate(annotator_check):
    """The exact failure: a round-2 prompt carrying round-1's own annotation."""
    prompt = M.tag_task_prompt(
        M.MultiAgentAnnotator._build_annotator_revision_msg(
            "Some sentence.", ANNOTATION_JSON, CRITIC_JSON, None))
    assert "entities" in prompt, "prompt should embed the previous annotation"
    assert annotator_check({"content": prompt}) is False


def test_untagged_revision_prompt_would_terminate(annotator_check):
    """Guards the guard: without the tag this prompt still ends the turn, which
    is why tagging is load-bearing rather than decorative."""
    prompt = M.MultiAgentAnnotator._build_annotator_revision_msg(
        "Some sentence.", ANNOTATION_JSON, CRITIC_JSON, None)
    assert annotator_check({"content": prompt}) is True


def test_critic_revision_prompt_does_not_terminate(critic_check):
    """The Critic's round 2 embeds its own prior critique and failed the same way."""
    prompt = M.tag_task_prompt(
        M.MultiAgentAnnotator._build_critic_review_msg(
            "Some sentence.", ANNOTATION_JSON, prev_critic_text=CRITIC_JSON))
    assert critic_check({"content": prompt}) is False


def test_tagged_prompt_never_terminates_even_with_trailing_terminate(annotator_check):
    """A tagged prompt wins over a TERMINATE that happens to be inside it —
    otherwise quoting an agent's previous reply would re-open the hole."""
    prompt = M.tag_task_prompt(ANNOTATION_JSON + "\nTERMINATE")
    assert annotator_check({"content": prompt}) is False


# ── 2. genuine completions must still end the turn ──────────────────────────

@pytest.mark.parametrize("content", [
    ANNOTATION_JSON,
    ANNOTATION_JSON + "\nTERMINATE",
    "TERMINATE",
    "<think>reasoning</think>\n" + ANNOTATION_JSON,
])
def test_real_annotator_reply_terminates(annotator_check, content):
    assert annotator_check({"content": content}) is True


def test_real_critic_reply_terminates(critic_check):
    assert critic_check({"content": CRITIC_JSON}) is True


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_reply_does_not_terminate(annotator_check, content):
    """An empty turn is not a finished turn — it is the symptom this bug had."""
    assert annotator_check({"content": content}) is False


def test_prose_reply_does_not_terminate(annotator_check):
    assert annotator_check({"content": "I am thinking about it."}) is False


# ── 3. the sentinel itself ──────────────────────────────────────────────────

def test_tag_is_prepended_once_and_preserves_the_message():
    body = "Revise your annotation."
    tagged = M.tag_task_prompt(body)
    assert tagged.count(M.TASK_PROMPT_SENTINEL) == 1
    assert tagged.endswith(body)


def test_sentinel_is_a_comment_so_models_ignore_it():
    """It ships inside every prompt, so it must not read as an instruction."""
    assert M.TASK_PROMPT_SENTINEL.startswith("<!--")
    assert M.TASK_PROMPT_SENTINEL.endswith("-->")
