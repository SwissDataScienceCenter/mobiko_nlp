"""
Task messages for the blind-pair pipeline.

All three agents carry the same standalone annotator SYSTEM prompt, so every
difference between them lives in these task messages. Two properties have to
hold or the design collapses into the one it replaces:

  independence   Every agent sees the SAME context — the sentence plus its
                 immediate neighbours — and the Critic annotates BEFORE either
                 peer annotation is revealed, so its blind view is a genuine
                 third independent commitment and the delta after exposure is a
                 measurement of anchoring rather than an inference.

                 Context is uniform by a later design decision, replacing an
                 earlier version that gave the whole abstract to the Critic alone.
                 Two problems with that: it made the three votes in the majority
                 rule non-exchangeable (a 2-1 majority could be two less-informed
                 agents outvoting one better-informed one), and it put a context
                 confound on the Adjudicator-vs-vote comparison. The window is
                 +/-1 sentence because that is what Mark and Davnah had — they
                 annotated one sentence at a time and could navigate to the
                 previous and following ones — and the agent-agent agreement
                 number is only comparable to their 0.497 under their condition.

  no deference   The revision message must not tell the agent to defer. The
                 equivalent message in the review pipeline does exactly that —
                 the Critic "is usually right", applying its corrections "should
                 be your default", keeping your own label is "conceding" and
                 permitted "only when you can justify it". That is a second
                 source of conformity on top of the structural anchoring, and
                 the tests below pin its absence phrase by phrase.

    pytest src/multi_agent_annotation/tests/test_blind_pair_tasks.py -v
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
for _p in (_PKG, _PKG / "evaluation", _PKG.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import multi_agent_annotation_ag2 as M  # noqa: E402

A = M.MultiAgentAnnotator
SENT = "Centres of species richness correlate with areas of high temperatures."
ABSTRACT = ("Mountains host many species. Centres of species richness correlate "
            "with areas of high temperatures. The mechanism is unclear.")
CONTEXT = {"before": ["Mountains host many species."],
           "after": ["The mechanism is unclear."],
           "window": ABSTRACT, "n_before": 1, "n_after": 1}
# Schema-COMPLETE: _parse_annotator_output requires all four of entities,
# relations, uncertain_cases and reasoning. A fixture missing any of them does
# not parse as annotator output, which would make the termination tests below
# pass for the wrong reason — the message would be harmless because it is
# malformed, not because it is tagged.
ANN = ('{"entities": [{"text": "species richness", "entity_type": "BIOTIC PROPERTY", '
       '"confidence": 0.9}], "relations": [], "uncertain_cases": [], '
       '"reasoning": "ok"}')


# ── blind annotation task ───────────────────────────────────────────────────

def test_blind_task_carries_the_sentence():
    assert SENT in A._build_blind_annotation_msg(SENT)


def test_no_context_renders_the_sentence_alone():
    msg = A._build_blind_annotation_msg(SENT)
    assert "Surrounding sentences" not in msg
    assert "mechanism is unclear" not in msg


def test_every_agent_gets_the_same_window():
    """REPLACES an earlier pair of tests asserting that A and B got no context
    while the Critic got the whole abstract. That asymmetry was dropped, not
    worked around: it unbalanced the majority vote and confounded the
    Adjudicator comparison. Uniform context is now the requirement, so the
    tests assert the opposite of what they used to."""
    msg = A._build_blind_annotation_msg(SENT, CONTEXT)
    assert "Mountains host many species." in msg
    assert "The mechanism is unclear." in msg


def test_context_does_not_widen_the_annotation_target():
    """Context for resolving reference, not more text to annotate. Without this
    the neighbours' spans end up in the output and every comparison is off."""
    msg = A._build_blind_annotation_msg(SENT, CONTEXT)
    assert "Annotate ONLY the sentence quoted above" in msg
    assert "[THE SENTENCE TO ANNOTATE]" in msg


def test_neighbours_are_labelled_by_position():
    msg = A._build_blind_annotation_msg(SENT, CONTEXT)
    assert "[before]" in msg and "[after]" in msg


def test_blind_task_reveals_no_peer():
    """If the blind pass mentions other annotators at all it is not blind."""
    for msg in (A._build_blind_annotation_msg(SENT),
                A._build_blind_annotation_msg(SENT, CONTEXT)):
        low = msg.lower()
        for word in ("peer", "another annotator", "other annotator", "critic"):
            assert word not in low, f"{word!r} leaked into the blind task"


# ── anonymisation ───────────────────────────────────────────────────────────

def test_labels_are_neutral_and_mapping_is_recoverable():
    pairs, mapping = A._anonymise_annotations(["one", "two", "three"],
                                              random.Random(0))
    assert [lab for lab, _ in pairs] == ["Annotation A", "Annotation B",
                                         "Annotation C"]
    assert sorted(mapping.values()) == [0, 1, 2]
    for lab, txt in pairs:
        assert ["one", "two", "three"][mapping[lab]] == txt


def test_shuffle_actually_permutes_across_seeds():
    """A shuffle that always returns input order is not a shuffle; a stable order
    invites deference to whatever comes first."""
    seen = set()
    for seed in range(40):
        _, mapping = A._anonymise_annotations(["x", "y", "z"], random.Random(seed))
        seen.add(tuple(mapping[f"Annotation {c}"] for c in "ABC"))
    assert len(seen) > 1, "order never changed"
    assert (0, 1, 2) in seen or len(seen) > 2


def test_anonymisation_is_deterministic_for_a_seeded_rng():
    """Runs have to be reproducible; the mapping is stamped into the record."""
    a = A._anonymise_annotations(["p", "q"], random.Random(11))
    b = A._anonymise_annotations(["p", "q"], random.Random(11))
    assert a == b


def test_anonymisation_handles_a_single_annotation():
    pairs, mapping = A._anonymise_annotations(["only"], random.Random(1))
    assert pairs == [("Annotation A", "only")]
    assert mapping == {"Annotation A": 0}


# ── peer revision task: the negative requirements ───────────────────────────

DEFERENCE = [
    "usually right",
    "should be your default",
    "conceding",
    "careful reviewer",
    "only when you can justify it",
]


def revision():
    pairs, _ = A._anonymise_annotations([ANN, ANN], random.Random(3))
    return A._build_peer_revision_msg(SENT, ANN, pairs, CONTEXT)


@pytest.mark.parametrize("phrase", DEFERENCE)
def test_revision_message_does_not_instruct_deference(phrase):
    assert phrase not in revision()


@pytest.mark.parametrize("phrase", DEFERENCE)
def test_the_review_pipeline_message_still_does(phrase):
    """Guards the guard from both sides: these phrases must still be present in
    the OLD revision message. If they vanish, the review pipeline's prompt has
    changed and is no longer the thing the blind-pair design is contrasted
    against — and the assertions above would pass trivially."""
    old = A._build_annotator_revision_msg(SENT, ANN, '{"disagreements": []}', None)
    assert phrase in old


def test_revision_frames_peers_as_equals():
    msg = revision()
    assert "peers, not reviewers" in msg
    assert "no more and no less likely to be correct" in msg


def test_revision_treats_changing_and_keeping_symmetrically():
    assert "equally acceptable" in revision()


def test_revision_frames_disagreement_as_difficulty():
    assert "it is not evidence that you were wrong" in revision()


def test_revision_quotes_no_corpus_statistic():
    """Telling the model how often the human annotators disagree would make its
    behaviour partly a function of the number this pipeline is compared with."""
    msg = revision()
    for n in ("0.497", "49.7", "half", "50%"):
        assert n not in msg


def test_revision_states_the_peers_are_unidentified():
    assert "not told which annotator produced which" in revision()


def test_revision_includes_own_and_both_peers():
    msg = revision()
    assert "Your annotation:" in msg
    assert msg.count("another annotator, independent") == 2


def test_revision_asks_for_the_same_schema_in_full():
    msg = revision()
    assert "same JSON format" in msg
    assert "not only the ones" in msg


def test_revision_strips_terminate_from_embedded_annotations():
    """A trailing TERMINATE inside a quoted annotation would otherwise be read as
    this turn's own end-of-turn signal."""
    pairs, _ = A._anonymise_annotations([ANN + "\nTERMINATE"], random.Random(0))
    msg = A._build_peer_revision_msg(SENT, ANN + "\nTERMINATE", pairs)
    assert "TERMINATE" not in msg


# ── the termination hazard this message class caused before ─────────────────

def test_tagged_revision_message_does_not_end_the_turn():
    """The revision message embeds parseable annotator JSON, which is exactly
    what silently disabled round 2 for months: is_termination_msg fired on the
    prompt's arrival. _run_agent_turn tags every message it sends; this asserts
    the tag defeats the check for THIS message too."""
    check = M._make_structured_termination_check(A._parse_annotator_output)
    pairs, _ = A._anonymise_annotations([ANN, ANN], random.Random(0))
    msg = A._build_peer_revision_msg(SENT, ANN, pairs, CONTEXT)
    assert check({"content": M.tag_task_prompt(msg)}) is False


def test_untagged_revision_message_would_end_the_turn():
    """Guards the guard: the tag is load-bearing, not decorative."""
    check = M._make_structured_termination_check(A._parse_annotator_output)
    pairs, _ = A._anonymise_annotations([ANN, ANN], random.Random(0))
    msg = A._build_peer_revision_msg(SENT, ANN, pairs, CONTEXT)
    assert check({"content": msg}) is True


def test_passing_a_bare_abstract_string_is_rejected():
    """The parameter used to be an abstract string. Silently accepting one would
    be falsy-ish or raise somewhere unhelpful; dropping context for a whole run
    is the expensive failure."""
    with pytest.raises(TypeError, match="ContextWindow"):
        A._build_blind_annotation_msg(SENT, ABSTRACT)
    with pytest.raises(TypeError, match="ContextWindow"):
        A._build_peer_revision_msg(SENT, ANN, [("Annotation A", ANN)], ABSTRACT)
