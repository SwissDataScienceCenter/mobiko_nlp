"""
Tests for canonicalize_entity_type() — the emit-time label canonicaliser.

Models drift on label FORMATTING (BIOTIC_ENTITY, "BIOTIC  ENTITY", lowercase)
far more than on label CONTENT. Left alone each variant scores 0 against gold:
across the eight scored runs 176 spans were malformed, worth up to 0.035 strict
F1, and enough to flip a human-parity verdict. It had been repaired by hand,
per file, three times before being fixed at emit time.

These tests lock in three properties:
  1. formatting variants collapse onto the schema label,
  2. genuinely underspecified labels are KEPT and reported, never guessed,
  3. the validator fires on every pydantic model that carries a label, so no
     construction path can bypass it.

Run with:
    pytest src/multi_agent_annotation/tests/test_entity_type_canonicalisation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent        # …/multi_agent_annotation
for _p in (_PKG, _PKG / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import multi_agent_annotation_ag2 as M  # noqa: E402

_REPO = _PKG.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── 1. the schema list this module validates against must be the real one ────

def test_fallback_list_matches_authoritative_schema():
    """_FALLBACK_ENTITY_TYPES is what the canonicaliser validates against when
    _ALL_ENTITY_TYPES is unset, so a typo there silently mislabels valid spans as
    off-schema. It previously read QUALITATIVE/QUANTITATIVE *ENTITY*, which the
    schema and the human gold (339 spans) type as *PROPERTY*."""
    from src.resources_updated.entity_schema import SCHEMA_BIODIV_LIST
    assert sorted(M._FALLBACK_ENTITY_TYPES) == sorted(SCHEMA_BIODIV_LIST)


# ── 2. formatting variants collapse ─────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("BIOTIC_ENTITY", "BIOTIC ENTITY"),                     # underscore (kimi, apertus)
    ("BIOTIC  ENTITY", "BIOTIC ENTITY"),                    # double space (the "_"->" " slip)
    ("ANTHROPOGENIC  PROPERTY", "ANTHROPOGENIC PROPERTY"),
    ("biotic entity", "BIOTIC ENTITY"),                     # case
    (" Biotic-Entity ", "BIOTIC ENTITY"),                   # hyphen + padding
    ("SPATIAL_PROPERTY", "SPATIAL PROPERTY"),
    ("QUANTITATIVEPROPERTY", "QUANTITATIVE PROPERTY"),       # missing separator
    ("BIOTIC COLLECTIVE ENTITY", "BIOTIC ENTITY"),           # v1 -> v2 rename
    ("ANTHROGENIC PROCESS", "ANTHROPOGENIC PROCESS"),        # explicit misspelling alias
    ("QUALITATIVE PROPERTY", "QUALITATIVE PROPERTY"),        # already valid, untouched
])
def test_formatting_variants_canonicalise(raw, expected):
    assert M.canonicalize_entity_type(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "   "])
def test_empty_input_yields_empty_string(raw):
    assert M.canonicalize_entity_type(raw) == ""


def test_every_schema_label_is_a_fixed_point():
    """Canonicalising a valid label must never change it."""
    for label in M._FALLBACK_ENTITY_TYPES:
        assert M.canonicalize_entity_type(label) == label


def test_single_word_labels_are_not_split():
    """The missing-separator repair must not mangle legitimate one-word labels."""
    assert M.canonicalize_entity_type("CONCEPT") == "CONCEPT"


def test_relation_pseudo_labels_pass_through():
    """RELATION:* spans are a different vocabulary and are not schema-checked."""
    assert M.canonicalize_entity_type("RELATION:CAUSAL") == "RELATION:CAUSAL"


# ── 3. unrecoverable labels are kept and reported, not guessed ──────────────

@pytest.mark.parametrize("raw", ["PROCESS", "ENTITY", "UNKNOWN"])
def test_underspecified_labels_are_kept_not_dropped(raw):
    """No modifier is recoverable from these, so guessing would risk assigning
    the WRONG label. Dropping would silently move recall and could orphan
    relations referencing the span. Keep + report is the only safe option."""
    M.reset_off_schema_labels()
    assert M.canonicalize_entity_type(raw) == raw
    assert raw in M.off_schema_label_report()


def test_off_schema_report_counts_and_resets():
    M.reset_off_schema_labels()
    assert M.off_schema_label_report() == {}
    for _ in range(3):
        M.canonicalize_entity_type("UNKNOWN")
    M.canonicalize_entity_type("PROCESS")
    assert M.off_schema_label_report() == {"UNKNOWN": 3, "PROCESS": 1}
    M.reset_off_schema_labels()
    assert M.off_schema_label_report() == {}


# ── 4. no construction path bypasses the validator ──────────────────────────

def test_entity_annotation_validator():
    assert M.EntityAnnotation(
        text="x", entity_type="BIOTIC_ENTITY").entity_type == "BIOTIC ENTITY"


def test_relation_endpoints_validator():
    rel = M.RelationFlat(relation="HAS_PROPERTY", e1_text="a",
                         e1_type="SPATIAL_ENTITY", e2_text="b",
                         e2_type="ABIOTIC  PROPERTY")
    assert rel.e1_type == "SPATIAL ENTITY"
    assert rel.e2_type == "ABIOTIC PROPERTY"
    # and again after conversion to the nested form actually written out
    nested = rel.to_relation_annotation()
    assert nested.e1.entity_type == "SPATIAL ENTITY"
    assert nested.e2.entity_type == "ABIOTIC PROPERTY"


def test_critic_missing_annotation_validator():
    assert M.CriticMissingAnnotation(
        text="x", entity_type="temporal_entity").entity_type == "TEMPORAL ENTITY"


def test_critic_disagreement_labels_validator():
    dis = M.CriticDisagreement(annotator_label="BIOTIC_ENTITY",
                               proposed_label="abiotic  process")
    assert dis.annotator_label == "BIOTIC ENTITY"
    assert dis.proposed_label == "ABIOTIC PROCESS"


def test_precedent_entry_validator():
    """A malformed label stored as a precedent would propagate to every later
    sentence that reuses it."""
    assert M.PrecedentEntry(
        span_text="x", entity_type="ABIOTIC_PROCESS").entity_type == "ABIOTIC PROCESS"


# ── 5. provenance stamping ──────────────────────────────────────────────────
# Three separate incidents motivated this: a raw run file edited 16 minutes
# after its reports were written; the hints on/off arm being unrecoverable from
# any run's output; and three byte-identical copies of one run existing under
# names that make contradictory claims about the relation-schema setting.

def test_record_carries_run_meta_through_serialisation():
    """Both write paths go through model_dump_json, so the field must survive it."""
    import json
    rec = M.DeliberationRecord(sentence="x", run_meta={"use_dependency_relation_hints": True})
    assert json.loads(rec.model_dump_json())["run_meta"]["use_dependency_relation_hints"] is True


def _fake_annotator(effective_hints, requested_hints, input_path=None):
    """Build run_meta without constructing agents (no LLM, no network)."""
    obj = M.MultiAgentAnnotator.__new__(M.MultiAgentAnnotator)
    obj.use_dependency_relation_hints = effective_hints
    obj._input_path = Path(input_path) if input_path else None
    obj.run_meta = obj._build_run_meta(
        annotator_model="qwen3-35B-vllm", critic_model="qwen3-35B-vllm",
        adjudicator_model="qwen3-35B-vllm", requested_hints=requested_hints,
        tool_choice="auto",
        annotator_temperature=0.3, max_rounds=3, strict_critic=False,
        cold_start=False, use_precedent_memory=False,
        dependency_model="en_core_web_trf")
    return obj


def test_run_meta_records_the_experimental_arm():
    meta = _fake_annotator(True, True).run_meta
    assert meta["use_dependency_relation_hints"] is True


def test_run_meta_distinguishes_requested_from_effective_hints():
    """Hints can be flipped by an env var or force-disabled when spaCy is
    missing. The EFFECTIVE value is the arm; recording only the requested one
    would mislabel the run."""
    meta = _fake_annotator(False, True).run_meta
    assert meta["hints_requested"] is True
    assert meta["use_dependency_relation_hints"] is False


def test_run_meta_hashes_the_input():
    gold = _PKG.parent.parent / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
    if not gold.exists():
        pytest.skip("gold file not present")
    meta = _fake_annotator(False, False, gold).run_meta
    assert meta["input_sha256"] and len(meta["input_sha256"]) == 64
    assert meta["input_path"].endswith("combined_M_D_Mark_postprocessed.jsonl")


def test_run_meta_records_decoding_params():
    """temperature/tool_choice determine whether logprob headroom exists, i.e.
    whether the RQ2 uncertainty signal is measurable at all."""
    meta = _fake_annotator(False, False).run_meta
    assert meta["annotator_temperature"] == 0.3
    assert meta["tool_choice"] == "auto"


def test_eval_stamp_provenance_hashes_and_labels():
    import sys as _s
    _s.path.insert(0, str(_PKG / "evaluation"))
    from eval_utils import stamp_provenance
    gold = _PKG.parent.parent / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
    if not gold.exists():
        pytest.skip("gold file not present")
    rep = stamp_provenance({}, gold)
    assert len(rep["provenance"]["agent_jsonl_sha256"]) == 64
    assert rep["provenance"]["agent_jsonl"].endswith("combined_M_D_Mark_postprocessed.jsonl")
    assert rep["provenance"]["scored_at"]
