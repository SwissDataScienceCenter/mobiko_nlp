"""
Tests for sentence -> abstract resolution.

The blind-pair pipeline's only genuine information asymmetry is that the Critic
sees the whole abstract and the two Annotators see one sentence. If resolution
fails silently, the Critic receives no context for those sentences and the
asymmetry becomes uneven in a way nobody declared — so the failure modes are
counted rather than swallowed.

    pytest src/multi_agent_annotation/tests/test_abstract_context.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from abstract_context import AbstractIndex, load_abstracts  # noqa: E402

A1 = "Mountains host many species. We find that richness correlates with relief."
A2 = "Crested gibbons are endemic to Indochina. Surveys were run in three sites."


def ix():
    return AbstractIndex(abstracts=[A1, A2])


def test_resolves_a_sentence_to_its_abstract():
    assert ix().context_for("We find that richness correlates with relief.") == A1


def test_resolves_into_the_second_abstract():
    assert ix().context_for("Surveys were run in three sites.") == A2


def test_strips_the_leaked_abstract_marker():
    """Six of the 155 grid sentences carry a literal '[abstract]' prefix from
    extraction. Without stripping it they match nothing, and the Critic would
    silently lose context on the first sentence of six abstracts."""
    i = ix()
    assert i.context_for("[abstract] Mountains host many species.") == A1
    assert i.stats()["unresolved"] == 0


def test_whitespace_differences_still_resolve():
    assert ix().context_for("We  find that richness\ncorrelates with relief.") == A1


def test_unresolved_is_counted_not_swallowed():
    i = ix()
    assert i.context_for("A sentence from no abstract at all.") is None
    assert i.stats()["unresolved"] == 1


def test_ambiguity_is_recorded_and_resolved_by_length_not_list_order():
    """A sentence shared verbatim by two abstracts must not resolve by whichever
    happens to come first — that is not a reason."""
    short, long = "Shared line here.", "Shared line here. Plus a great deal more text."
    i = AbstractIndex(abstracts=[short, long])
    assert i.context_for("Shared line here.") == long
    assert i.stats()["ambiguous"] == 1


def test_empty_sentence_resolves_to_nothing():
    i = ix()
    assert i.context_for("") is None
    assert i.context_for("   ") is None


def test_results_are_cached_without_double_counting_failures():
    i = ix()
    for _ in range(3):
        i.context_for("Not present anywhere.")
    assert i.stats()["unresolved"] == 1, "a repeated lookup must not re-record"


def test_missing_abstracts_file_yields_empty_not_an_exception():
    assert load_abstracts(Path("/nonexistent/abstracts.txt")) == []


# ── the real corpus ─────────────────────────────────────────────────────────

GRID_INPUT = Path("/home/katinska/mobiko_nlp/data/manually_labeled_last"
                  "/overlap_sentences.txt")


@pytest.mark.skipif(not GRID_INPUT.exists(), reason="grid input not present")
def test_every_grid_sentence_resolves():
    """Locks the measured state: 155/155, nothing unresolved, nothing ambiguous.
    A future change to either file that breaks the mapping shows up here rather
    than as a Critic quietly running without context."""
    i = AbstractIndex()
    if not i.abstracts:
        pytest.skip("abstracts.txt not present")
    sents = [l.strip() for l in GRID_INPUT.read_text().splitlines() if l.strip()]
    resolved = sum(1 for s in sents if i.context_for(s))
    assert resolved == len(sents), (
        f"{len(sents) - resolved} of {len(sents)} sentences lost context; "
        f"stats={i.stats()}")
    assert i.stats()["ambiguous"] == 0


# ── the matched context window ──────────────────────────────────────────────
#
# Mark and Davnah annotated one sentence at a time and could navigate to the
# previous and following sentences. So the agents' condition is a +/-1 sentence
# window: sentence-only handicaps them against the 0.497 ceiling, and the whole
# abstract makes them better informed than it. Measured on the grid input the
# window is ~121 tokens against ~2,700 for the whole abstract.

from abstract_context import ContextWindow  # noqa: E402

A = ("Mountains host many species. Centres of richness track temperature. "
     "The mechanism is unclear. Further work is needed.")


def cw():
    return ContextWindow(index=AbstractIndex(abstracts=[A]), radius=1)


def test_middle_sentence_gets_both_neighbours():
    w = cw().for_sentence("Centres of richness track temperature.")
    assert w["before"] == ["Mountains host many species."]
    assert w["after"] == ["The mechanism is unclear."]
    assert w["n_before"] == 1 and w["n_after"] == 1


def test_first_sentence_has_no_previous():
    """Faithful, not a failure: a human on the first sentence of an abstract
    also has nothing to navigate back to."""
    w = cw().for_sentence("Mountains host many species.")
    assert w["before"] == []
    assert w["after"] == ["Centres of richness track temperature."]


def test_last_sentence_has_no_following():
    w = cw().for_sentence("Further work is needed.")
    assert w["after"] == []
    assert w["before"] == ["The mechanism is unclear."]


def test_the_target_is_exact_even_though_neighbours_are_approximate():
    """Neighbours are cut on a terminator regex, which biodiversity text will
    occasionally cut badly. The target is located verbatim, so the thing being
    annotated is never approximate."""
    target = "Centres of richness track temperature."
    w = cw().for_sentence(target)
    assert target in w["window"]


def test_before_and_after_stay_separate_from_the_target():
    """A window that does not say which sentence is the target invites
    annotation of the neighbours, so the prompt needs them addressable."""
    w = cw().for_sentence("Centres of richness track temperature.")
    assert "before" in w and "after" in w
    assert "Mountains host many species." not in w["before"][0].replace(
        "Mountains host many species.", "", 1)


def test_radius_widens_the_window():
    w = ContextWindow(index=AbstractIndex(abstracts=[A]), radius=2).for_sentence(
        "The mechanism is unclear.")
    assert w["n_before"] == 2


def test_sentence_from_no_abstract_falls_back_to_itself():
    c = cw()
    w = c.for_sentence("A sentence from nowhere.")
    assert w["window"] == "A sentence from nowhere."
    assert w["n_before"] == 0 and w["n_after"] == 0
    assert c.stats()["no_context"] == 1


def test_one_sided_windows_are_counted():
    c = cw()
    c.for_sentence("Mountains host many species.")
    assert c.stats()["one_sided"] == 1


def test_leaked_abstract_marker_is_stripped_in_the_window_too():
    w = cw().for_sentence("[abstract] Mountains host many species.")
    assert "[abstract]" not in w["window"]
    assert w["n_after"] == 1


@pytest.mark.skipif(not GRID_INPUT.exists(), reason="grid input not present")
def test_grid_window_coverage_is_locked():
    """Measured state: 141 of 155 get both neighbours, 14 sit at an abstract
    boundary, none is left with no context at all."""
    c = ContextWindow(radius=1)
    if not c.index.abstracts:
        pytest.skip("abstracts.txt not present")
    sents = [l.strip() for l in GRID_INPUT.read_text().splitlines() if l.strip()]
    res = [c.for_sentence(s) for s in sents]
    both = sum(1 for r in res if r["n_before"] and r["n_after"])
    none = sum(1 for r in res if not r["n_before"] and not r["n_after"])
    assert none == 0, f"{none} sentences got no context at all"
    assert both == 141, f"both-neighbour coverage changed: {both} of {len(sents)}"


@pytest.mark.skipif(not GRID_INPUT.exists(), reason="grid input not present")
def test_the_window_is_far_cheaper_than_the_whole_abstract():
    """The reason context can be given to every agent rather than one."""
    c = ContextWindow(radius=1)
    if not c.index.abstracts:
        pytest.skip("abstracts.txt not present")
    sents = [l.strip() for l in GRID_INPUT.read_text().splitlines() if l.strip()]
    mean_window = sum(len(c.for_sentence(s)["window"]) for s in sents) / len(sents)
    mean_abstract = sum(len(c.index.context_for(s) or "") for s in sents) / len(sents)
    assert mean_window < mean_abstract / 5, (
        f"window {mean_window:.0f} chars vs abstract {mean_abstract:.0f}")
