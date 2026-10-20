"""
Tests for evaluation/eval_relation_iaa.py — the relation IAA metric.

Pure-function, no-LLM, no-network coverage of the parts that are easy to get
subtly wrong: the four argument-matching modes, the greedy ONE-TO-ONE alignment
(a relation must never be credited twice), direction handling, exclusion of
hyper-relations / offset-less relations, and the entity-ceiling decomposition.

Run with:
    pytest src/multi_agent_annotation/tests/test_relation_iaa.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent        # …/multi_agent_annotation
for _p in (_PKG, _PKG / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_relation_iaa as ri  # noqa: E402


# ───────────────────────── fixtures / builders ─────────────────────────

def _rel(s1, e1, t1, x1, label, s2, e2, t2, x2) -> ri.Rel:
    return ri.Rel(label, label,
                  ri.Arg(s1, e1, t1, x1), ri.Arg(s2, e2, t2, x2), "", "")


BIRDS_HAS_RICHNESS = _rel(0, 5, "BIOTIC ENTITY", "birds", "HAS_PROPERTY",
                          10, 20, "BIOTIC PROPERTY", "richness")
TREES_CAUSE_GROWTH = _rel(50, 55, "BIOTIC ENTITY", "trees", "CAUSES",
                          60, 70, "BIOTIC PROPERTY", "growth")


def _entity_relation(e1_offsets, e2_offsets, label="HAS_PROPERTY", level=0,
                     e1_kind="entity", e2_kind="entity") -> dict:
    (s1, e1), (s2, e2) = e1_offsets, e2_offsets
    return {
        "relation": label, "level": level,
        "e1_kind": e1_kind, "e2_kind": e2_kind,
        "e1": {"start_char": s1, "end_char": e1, "type": "T", "text": "a"},
        "e2": {"start_char": s2, "end_char": e2, "type": "T", "text": "b"},
    }


# ───────────────────────── argument matching ─────────────────────────

def test_arg_modes_agree_on_an_exact_match():
    a = ri.Arg(0, 5, "T", "x")
    for mode in ri.ARG_MODES:
        assert ri.arg_score(a, a, mode) is not None


def test_strict_needs_the_entity_type_boundary_does_not():
    a = ri.Arg(0, 5, "BIOTIC ENTITY", "birds")
    b = ri.Arg(0, 5, "ABIOTIC ENTITY", "birds")
    assert ri.arg_score(a, b, "strict") is None
    assert ri.arg_score(a, b, "boundary") is not None


def test_overlap_scores_by_intersection_over_union_and_rejects_disjoint():
    assert ri.arg_score(ri.Arg(0, 10, "T", "x"), ri.Arg(5, 15, "T", "x"),
                        "overlap") == pytest.approx(1 / 3)
    # touching but not overlapping
    assert ri.arg_score(ri.Arg(0, 5, "T", "x"), ri.Arg(5, 9, "T", "x"),
                        "overlap") is None


def test_text_mode_ignores_position_but_needs_a_surface_form():
    assert ri.arg_score(ri.Arg(0, 5, "T", "birds"), ri.Arg(80, 85, "T", "birds"),
                        "text") is not None
    assert ri.arg_score(ri.Arg(0, 5, "T", ""), ri.Arg(0, 5, "T", ""),
                        "text") is None


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        ri.arg_score(ri.Arg(0, 1, "T", "x"), ri.Arg(0, 1, "T", "x"), "fuzzy")


# ───────────────────────── relation matching ─────────────────────────

def test_labeled_matching_requires_the_same_label():
    other_label = _rel(0, 5, "BIOTIC ENTITY", "birds", "LOCATED_IN",
                       10, 20, "BIOTIC PROPERTY", "richness")
    assert ri.match_relations([BIRDS_HAS_RICHNESS], [other_label], "boundary") == []
    assert len(ri.match_relations([BIRDS_HAS_RICHNESS], [other_label], "boundary",
                                  labeled=False)) == 1


def test_direction_matters_unless_undirected_is_requested():
    swapped = _rel(10, 20, "BIOTIC PROPERTY", "richness", "HAS_PROPERTY",
                   0, 5, "BIOTIC ENTITY", "birds")
    assert ri.match_relations([BIRDS_HAS_RICHNESS], [swapped], "boundary") == []
    m = ri.match_relations([BIRDS_HAS_RICHNESS], [swapped], "boundary",
                           undirected=True)
    assert len(m) == 1 and m[0][2] is True     # flagged as a direction-only match


def test_same_direction_wins_over_a_swapped_candidate():
    """A symmetric relation must not be credited via the swapped alignment."""
    swapped = _rel(10, 20, "BIOTIC PROPERTY", "richness", "HAS_PROPERTY",
                   0, 5, "BIOTIC ENTITY", "birds")
    m = ri.match_relations([BIRDS_HAS_RICHNESS],
                           [BIRDS_HAS_RICHNESS, swapped], "boundary",
                           undirected=True)
    assert len(m) == 1 and m[0][2] is False


def test_matching_is_one_to_one():
    dup = BIRDS_HAS_RICHNESS
    assert len(ri.match_relations([dup, dup], [dup], "boundary")) == 1
    assert len(ri.match_relations([dup], [dup, dup], "boundary")) == 1
    assert len(ri.match_relations([dup, TREES_CAUSE_GROWTH],
                                 [dup, TREES_CAUSE_GROWTH], "boundary")) == 2


def test_offset_shift_matches_only_under_the_relaxed_modes():
    shifted = _rel(1, 6, "BIOTIC ENTITY", "birds", "HAS_PROPERTY",
                   10, 20, "BIOTIC PROPERTY", "richness")
    got = {mode: len(ri.match_relations([BIRDS_HAS_RICHNESS], [shifted], mode))
           for mode in ri.ARG_MODES}
    assert got == {"strict": 0, "boundary": 0, "normalized": 1, "overlap": 1, "text": 1}


# ───────────────────────── extraction ─────────────────────────

def test_hyper_and_offsetless_relations_are_excluded_and_counted():
    sent = {"text": "s", "spans": [], "relations": [
        _entity_relation((0, 5), (6, 9)),                                  # scorable
        _entity_relation((None, None), (6, 9), label="DEMONSTRATES",
                         level=1, e1_kind="relation"),                     # hyper
        _entity_relation((None, None), (6, 9), label="X"),                 # no offsets
    ]}
    rels, hyper, dropped = ri.sentence_relations(sent, "raw")
    assert (len(rels), hyper, dropped) == (1, 1, 1)


def test_label_views_map_only_what_they_declare():
    # every view canonicalises onto the GUIDE's name, so DURING (the page's
    # retired option) becomes OCCURS_DURING, not the other way round
    assert ri.apply_label_view("DURING", "raw") == "DURING"
    assert ri.apply_label_view("DURING", "migration") == "OCCURS_DURING"
    assert ri.apply_label_view("DURING", "synonyms") == "OCCURS_DURING"
    assert ri.apply_label_view("DURING", "families") == "TEMPORAL"
    assert ri.apply_label_view("OCCURS_DURING", "families") == "TEMPORAL"
    # an unknown label is never invented into a family
    assert ri.apply_label_view("MADE_UP_LABEL", "families") == "MADE_UP_LABEL"


def test_migration_view_undoes_only_the_page_renames():
    """It folds the retired page options onto their guide successors and
    touches nothing else — a typo stays a typo until the synonyms view."""
    assert ri.apply_label_view("IS_PART_OF", "migration") == "PART_OF"
    assert ri.apply_label_view("CONNTAINS", "migration") == "CONNTAINS"
    assert ri.apply_label_view("CONNTAINS", "synonyms") == "CONTAINS"
    # COMPARES_TO has no single successor in the guide, so it is left alone
    assert ri.apply_label_view("COMPARES_TO", "migration") == "COMPARES_TO"
    assert set(ri.MIGRATION_MAP) == {"IS_PART_OF", "DURING"}
    assert set(ri.MIGRATION_MAP.values()) <= ri.FIXED_RELATION_TYPES


def test_synonyms_view_includes_the_migration():
    """Stacked, not alternative: a renamed label folded by the migration must
    not be un-folded by the synonym pass."""
    assert ri.apply_label_view("IS_PART_OF", "synonyms") == "PART_OF"
    assert ri.apply_label_view("PART_OF", "synonyms") == "PART_OF"


# ───────────────────────── end to end ─────────────────────────

def _doc(spans, relations):
    return [{"text": "the sentence", "spans": spans, "relations": relations}]


SPANS_AB = [{"start_char": 0, "end_char": 5, "type": "T", "text": "a"},
            {"start_char": 6, "end_char": 9, "type": "T", "text": "b"}]


def test_identical_annotations_score_perfectly():
    doc = _doc(SPANS_AB, [_entity_relation((0, 5), (6, 9))])
    r = ri.compute(doc, doc, "A", "B")
    assert r["labeled_f1"]["boundary"]["f1"] == 1.0
    assert r["detection_f1"]["boundary"]["f1"] == 1.0
    assert r["labeling_given_detection"]["cohen_kappa"] == 1.0
    assert r["argument_ceiling"]["boundary"]["ceiling_f1"] == 1.0


def test_same_pair_different_label_separates_detection_from_labeling():
    a = _doc(SPANS_AB, [_entity_relation((0, 5), (6, 9), label="HAS_PROPERTY")])
    b = _doc(SPANS_AB, [_entity_relation((0, 5), (6, 9), label="LOCATED_IN")])
    r = ri.compute(a, b, "A", "B")
    assert r["detection_f1"]["boundary"]["f1"] == 1.0
    assert r["labeled_f1"]["boundary"]["f1"] == 0.0
    assert r["labeling_given_detection"]["observed_agreement"] == 0.0


def test_ceiling_is_zero_when_the_other_side_lacks_the_endpoint_span():
    a = _doc(SPANS_AB, [_entity_relation((0, 5), (6, 9))])
    b = _doc([{"start_char": 0, "end_char": 5, "type": "T", "text": "a"}],
             [_entity_relation((0, 5), (20, 25))])
    r = ri.compute(a, b, "A", "B")
    assert r["argument_ceiling"]["boundary"]["ceiling_f1"] == 0.0
    assert r["labeled_f1"]["boundary"]["f1"] == 0.0
    # a zero ceiling must not turn the decomposition into a ZeroDivisionError
    assert r["decomposition"]["pair_linking_given_matchable"] is None


def test_decomposition_multiplies_back_to_the_labeled_score():
    """entity ceiling × pair linking × labeling == labeled F1, exactly."""
    a = _doc(SPANS_AB + [{"start_char": 20, "end_char": 25, "type": "T", "text": "c"}],
             [_entity_relation((0, 5), (6, 9), label="HAS_PROPERTY"),
              _entity_relation((0, 5), (20, 25), label="CAUSES")])
    b = _doc(SPANS_AB,
             [_entity_relation((0, 5), (6, 9), label="LOCATED_IN"),
              _entity_relation((6, 9), (0, 5), label="CAUSES")])
    r = ri.compute(a, b, "A", "B")
    d = r["decomposition"]
    assert (d["entity_ceiling"] * d["pair_linking_given_matchable"]
            * d["labeling_given_co_detected"]) == pytest.approx(d["labeled_f1"])


def test_single_annotator_sentences_are_workload_not_disagreement():
    """A sentence only one annotator touched must not be scored by default."""
    a = _doc(SPANS_AB, [_entity_relation((0, 5), (6, 9))])
    b = _doc(SPANS_AB, [])
    r = ri.compute(a, b, "A", "B")
    assert r["coverage"]["doubly_annotated"] == 0
    assert r["coverage"]["A_only_annotated"] == 1
    assert r["labeled_f1"]["boundary"]["f1"] is None      # nothing in scope
    # …but --scope all-aligned counts it as a miss
    r_all = ri.compute(a, b, "A", "B", scope="all-aligned")
    assert r_all["coverage"]["scored_sentences"] == 1
    assert r_all["labeled_f1"]["boundary"]["f1"] == 0.0


def test_sentences_are_aligned_by_text_not_by_index():
    a = [{"text": "first", "spans": SPANS_AB,
          "relations": [_entity_relation((0, 5), (6, 9))]},
         {"text": "second", "spans": SPANS_AB, "relations": []}]
    b = [{"text": "second", "spans": SPANS_AB, "relations": []},
         {"text": "first", "spans": SPANS_AB,
          "relations": [_entity_relation((0, 5), (6, 9))]}]
    r = ri.compute(a, b, "A", "B")
    assert r["coverage"]["text_aligned"] == 2
    assert r["labeled_f1"]["boundary"]["f1"] == 1.0


# ──────────────── hyper-relations (relation as an argument) ────────────────

def _hyper_doc(inner_label="CAUSES", outer_label="AFFECTS",
               inner_e2=(60, 70)) -> dict:
    """One sentence: entity --outer--> ⟨entity --inner--> entity⟩."""
    inner = {
        "kind": "relation", "relation": inner_label,
        "e1": {"start_char": 50, "end_char": 55, "type": "T", "text": "trees"},
        "e2": {"start_char": inner_e2[0], "end_char": inner_e2[1],
               "type": "T", "text": "growth"},
    }
    return {
        "text": "s", "spans": [],
        "relations": [{
            "relation": outer_label, "level": 1,
            "e1_kind": "entity", "e2_kind": "relation",
            "e1": {"start_char": 0, "end_char": 5, "type": "T", "text": "birds"},
            "e2": inner,
        }],
    }


def test_hyper_relations_are_read_with_their_nested_argument():
    rels, dropped = ri.sentence_relations_with_hyper(_hyper_doc(), "raw")
    assert dropped == 0 and len(rels) == 1
    assert ri.is_hyper(rels[0])
    assert isinstance(rels[0].e2, ri.RelArg) and rels[0].e2.label == "CAUSES"
    # the flat reader still leaves it out, as every published number assumes
    flat, hyper, _ = ri.sentence_relations(_hyper_doc(), "raw")
    assert flat == [] and hyper == 1


def test_identical_hyper_relations_match_at_every_level():
    a, _ = ri.sentence_relations_with_hyper(_hyper_doc(), "raw")
    b, _ = ri.sentence_relations_with_hyper(_hyper_doc(), "raw")
    assert len(ri.match_relations(a, b, "boundary", labeled=True)) == 1


def test_a_nested_label_disagreement_is_a_labeled_miss_not_a_detection_miss():
    a, _ = ri.sentence_relations_with_hyper(_hyper_doc(inner_label="CAUSES"), "raw")
    b, _ = ri.sentence_relations_with_hyper(_hyper_doc(inner_label="AFFECTS"), "raw")
    assert len(ri.match_relations(a, b, "boundary", labeled=False)) == 1
    assert len(ri.match_relations(a, b, "boundary", labeled=True)) == 0


def test_a_nested_endpoint_shift_breaks_the_match_under_boundary():
    a, _ = ri.sentence_relations_with_hyper(_hyper_doc(), "raw")
    b, _ = ri.sentence_relations_with_hyper(_hyper_doc(inner_e2=(61, 70)), "raw")
    assert ri.match_relations(a, b, "boundary", labeled=False) == []
    assert len(ri.match_relations(a, b, "overlap", labeled=False)) == 1


def test_an_entity_never_matches_a_nested_relation():
    hyper, _ = ri.sentence_relations_with_hyper(_hyper_doc(), "raw")
    flat = [_rel(0, 5, "T", "birds", "AFFECTS", 50, 55, "T", "trees")]
    assert ri.match_relations(hyper, flat, "boundary", labeled=False) == []
    assert ri.arg_score(hyper[0].e2, ri.Arg(50, 55, "T", "trees"),
                        "boundary") is None


def test_hyper_ceiling_needs_every_leaf_entity_on_the_other_side():
    rel = ri.sentence_relations_with_hyper(_hyper_doc(), "raw")[0][0]
    leaves = [ri.Arg(0, 5, "T", "birds"), ri.Arg(50, 55, "T", "trees"),
              ri.Arg(60, 70, "T", "growth")]
    assert ri._endpoint_present(rel, leaves, "boundary")
    assert not ri._endpoint_present(rel, leaves[:-1], "boundary")


# ──────────────── the "custom label" marking tracks the guide ────────────────

def test_custom_marking_is_read_from_the_relation_guide():
    """The report calls a label 'custom' when it is outside the vocabulary the
    page offers and the pipeline serves. That set is the guide's, read at
    import — it used to be an inlined copy of the page's ORIGINAL nine options,
    which silently marked standard labels as annotator inventions."""
    vocab = ri.FIXED_RELATION_TYPES
    assert len(vocab) > 30, "the guide did not load; custom marking would be wrong"
    for standard in ("PART_OF", "SUBSET_OF", "LOCATED_IN", "HAS_VALUE", "OCCURS_DURING"):
        assert standard in vocab, f"{standard} is in the guide but marked custom"
    for retired in ("IS_PART_OF", "COMPARES_TO", "DURING", "OCCURS_IN", "HAS_STATE"):
        assert retired not in vocab, (
            f"{retired} left the guide; it must read as custom, not standard")


def test_the_guide_the_page_offers_and_the_report_scores_are_the_same_list():
    import csv as _csv
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parents[3]
    guide_csv = repo / "src" / "multi_agent_annotation" / "Relation schema.csv"
    if not guide_csv.is_file():
        import pytest as _pytest
        _pytest.skip("relation guide not present")
    with guide_csv.open(newline="", encoding="utf-8") as fh:
        names = {(r.get("Relation") or "").strip() for r in _csv.DictReader(fh)
                 if (r.get("Relation") or "").strip()}
    assert ri.FIXED_RELATION_TYPES == names


# ──────────── 'normalized' mode: punctuation and plural endings ────────────

def test_normalized_mode_ignores_a_trailing_full_stop():
    """Sentence 149: Mark's span includes the full stop, Davnah's does not."""
    mark = ri.Arg(224, 235, "T", "well-being.")
    davnah = ri.Arg(224, 234, "T", "well-being")
    assert ri.arg_score(mark, davnah, "boundary") is None
    assert ri.arg_score(mark, davnah, "normalized") is not None


def test_normalized_mode_ignores_a_plural_ending():
    """Sentences 146/147: 'assemblages' vs 'assemblage' at the same position."""
    assert ri.arg_score(ri.Arg(32, 43, "T", "assemblages"),
                        ri.Arg(32, 42, "T", "assemblage"), "normalized") is not None
    assert ri.arg_score(ri.Arg(10, 26, "T", "lower elevations"),
                        ri.Arg(10, 25, "T", "lower elevation"), "normalized") is not None


def test_normalized_mode_accepts_another_occurrence_of_the_same_phrase():
    """Sentence 140: 'glacier soil' is in the sentence three times."""
    other = ri.arg_score(ri.Arg(0, 7, "T", "beetles"),
                         ri.Arg(40, 46, "T", "beetle"), "normalized")
    same = ri.arg_score(ri.Arg(40, 47, "T", "beetles"),
                        ri.Arg(40, 46, "T", "beetle"), "normalized")
    assert other is not None and same is not None and same > other


def test_normalized_mode_still_needs_the_same_words():
    # an extra word is not punctuation or a plural ending
    assert ri.arg_score(ri.Arg(20, 37, "T", "from 750 to 850 m"),
                        ri.Arg(25, 37, "T", "750 to 850 m"), "normalized") is None


def test_normalized_sits_between_boundary_and_overlap():
    modes = ri.ARG_MODES
    assert modes.index("boundary") < modes.index("normalized") < modes.index("overlap")
    exact = ri.Arg(0, 5, "T", "birds")
    assert ri.arg_score(exact, exact, "normalized") is not None   # boundary ⊆ normalized
