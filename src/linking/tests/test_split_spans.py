"""
Tests for coordination splitting.

The operation exists because its sibling does not. Distributing a shared
modifier onto a neighbouring span was measured against 38 corrections the
annotators recorded in `concept_text` and reproduced 1 of them; splitting a
coordinated span gets 7 of 17 clearly right and declines the rest. What is
pinned here is that ratio's two halves: the splits that must happen, and the
traps that must not.
"""
from __future__ import annotations

import pytest

from src.linking.split_spans import SpanSplitter, split_span

splitter = SpanSplitter()
needs_trf = pytest.mark.skipif(
    not splitter.available or splitter.model != "en_core_web_trf",
    reason="en_core_web_trf absent; splitting is disabled by design without it")


def _split(sentence: str, phrase: str):
    start = sentence.index(phrase)
    got = splitter.split_all(sentence, [("k", start, start + len(phrase))])
    return got.get("k")


# ── splits that must happen ──────────────────────────────────────

@needs_trf
def test_elided_head_matches_the_annotators_own_split():
    """`eastern and western slopes` is the one case where the annotators
    recorded the split themselves, as two concept_text values on one span."""
    s = _split("Defaunation differed between the eastern and western slopes "
               "of the massif.", "eastern and western slopes")
    assert s and s.parts == ["eastern slopes", "western slopes"]


@needs_trf
@pytest.mark.parametrize("sentence,phrase,expected", [
    ("The area shows high and transient relief today.",
     "high and transient relief", ["high relief", "transient relief"]),
    ("It covers ecosystem goods and services broadly.",
     "ecosystem goods and services", ["ecosystem goods", "ecosystem services"]),
])
def test_coordinated_spans_split(sentence, phrase, expected):
    s = _split(sentence, phrase)
    assert s and s.parts == expected


@needs_trf
def test_a_conjunct_does_not_absorb_its_siblings():
    """spaCy hangs later conjuncts under the first, so `western` sits inside
    `eastern`'s subtree. Taking the subtree whole produced the part
    "eastern western slopes"."""
    s = _split("…the eastern and western slopes…", "eastern and western slopes")
    assert s and "western" not in s.parts[0]


@needs_trf
def test_shared_modifiers_are_kept_on_every_part():
    """Dropping a shared word silently changes meaning: the parts of
    "broadleaf evergreen and coniferous habitat mosaic" must still be a
    habitat mosaic."""
    s = _split("We mapped the broadleaf evergreen and coniferous habitat "
               "mosaic across the park.",
               "broadleaf evergreen and coniferous habitat mosaic")
    assert s and all("habitat" in p and "mosaic" in p for p in s.parts)


# ── traps that must not split ────────────────────────────────────

@needs_trf
@pytest.mark.parametrize("sentence,phrase", [
    # A range is one quantity, not two concepts.
    ("Sampling ran between 1100 and 1700m elevation.", "between 1100 and 1700m"),
    ("Traps were set between 25 and 40days after ice-off.",
     "between 25 and 40days after ice-off"),
    # Proper names. sm split this one into "Monitoring" + "Programme".
    ("Data came from the Monitoring and Assessment Programme.",
     "Monitoring and Assessment Programme"),
])
def test_traps_are_declined(sentence, phrase):
    assert _split(sentence, phrase) is None


def test_a_span_without_a_conjunction_is_never_split():
    if not splitter.available:
        pytest.skip("no parser")
    assert _split("We surveyed coniferous forest here.", "coniferous forest") is None


# ── availability is reported, never assumed ──────────────────────

def test_splitting_is_off_rather_than_degraded_without_trf():
    """sm turns parse disagreement into invented entities on this operation —
    "Da Nhim and Dran Protection Forests" became "Da Nhim" + "Da Forests". An
    entity minted from a mis-parse is not recoverable downstream, so the
    absence of trf disables splitting instead of downgrading it.
    """
    s = SpanSplitter(model="en_core_web_does_not_exist")
    assert not s.available
    assert s.error and "unavailable" in s.error
    assert s.split_all("Anything at all here.", [("k", 0, 8)]) == {}


def test_fallback_to_sm_is_opt_in_only():
    s = SpanSplitter(model="en_core_web_does_not_exist", allow_fallback=True)
    if s.available:
        assert s.model == "en_core_web_sm"


@needs_trf
def test_split_parts_become_separate_entities(tmp_path):
    """End to end: the container denotes nothing, the parts denote one thing each."""
    import json, subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    src = root / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
    if not src.exists():
        pytest.skip("human gold not present")
    out = tmp_path / "r.jsonl"
    r = subprocess.run([sys.executable, "-m", "src.linking.resolve_corpus",
                        "--input", str(src), "--out", str(out)],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]

    containers = [x for x in rows if x.get("split_into")]
    parts = [x for x in rows if x.get("part_index")]
    assert containers and parts

    for c in containers:
        assert c["entity_id"] is None, "a coordinated span is not one entity"
        mine = [p for p in parts if p["mention_id"].startswith(c["mention_id"] + ".")]
        assert len(mine) == c["split_into"]
        assert all(p["entity_id"] for p in mine)
        # The original text stays recoverable from every part.
        assert all(p["split_from"] == c["text"] for p in mine)

    slopes = [p for p in parts if p["text"] in ("eastern slopes", "western slopes")]
    if slopes:
        assert len({p["entity_id"] for p in slopes}) == 2, \
            "the two slopes must not collapse back into one entity"
