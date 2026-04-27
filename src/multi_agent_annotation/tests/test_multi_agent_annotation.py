"""
Tests for multi_agent_annotation_ag2.py — pure-function / no-LLM coverage.

Run with:
    pytest src/multi_agent_annotation/tests/test_multi_agent_annotation.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.multi_agent_annotation.multi_agent_annotation_ag2 as mod
from src.multi_agent_annotation.multi_agent_annotation_ag2 import (
    AdjudicatorOutput,
    AnnotatorOutput,
    CriticDisagreement,
    CriticMissingAnnotation,
    CriticOutput,
    EntityAnnotation,
    RelationFlat,
    RelationAnnotation,
    DeliberationRecord,
    MultiAgentAnnotator,
    _init_tool_state,
    _critic_is_satisfied,
    _get_embedded_guideline,
    _is_final_terminate_msg,
    analyze_disagreements,
    build_llm_config,
    consistency_check,
    guideline_search,
    list_entity_types,
    load_schema,
    schema_lookup,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

SAMPLE_SCHEMA = {
    "HAS_PROPERTY": [
        ["BIOTIC ENTITY", "BIOTIC PROPERTY"],
        ["ABIOTIC ENTITY", "ABIOTIC PROPERTY"],
    ],
    "LOCATED_IN": [
        ["BIOTIC ENTITY", "SPATIAL ENTITY"],
        ["BIOTIC COLLECTIVE ENTITY", "SPATIAL ENTITY"],
    ],
}

SAMPLE_SEEDS = {
    "HAS_PROPERTY": [
        {
            "e1": {"text": "species", "type": "BIOTIC ENTITY"},
            "e2": {"text": "density", "type": "BIOTIC PROPERTY"},
            "sentence": "The species density was high in this region.",
        }
    ],
    "LOCATED_IN": [
        {
            "e1": {"text": "population", "type": "BIOTIC COLLECTIVE ENTITY"},
            "e2": {"text": "forest", "type": "SPATIAL ENTITY"},
            "sentence": "The population lives in a forest.",
        }
    ],
}

SAMPLE_GUIDELINE = [
    {"title": "Step 1 — Abstract/theoretical", "content": "Scientific concept → CONCEPT"},
    {"title": "Step 4 — Anthropogenic", "content": "Human system characteristic → ANTHROPOGENIC PROPERTY."},
    {"title": "Step 5 — Biotic", "content": "Single organism → BIOTIC ENTITY. habitat property → BIOTIC PROPERTY."},
    {"title": "Step 3 — Spatial", "content": "Place → SPATIAL ENTITY. spatial attribute → SPATIAL PROPERTY."},
]


@pytest.fixture(autouse=True)
def reset_tool_state():
    """Reset module-level state before each test."""
    _init_tool_state({}, [], {})
    yield
    _init_tool_state({}, [], {})


@pytest.fixture()
def initialized_state():
    _init_tool_state(SAMPLE_SCHEMA, SAMPLE_GUIDELINE, SAMPLE_SEEDS)


# ─────────────────────────────────────────────────────────────
# Data-structure tests
# ─────────────────────────────────────────────────────────────

class TestDataclasses:
    def test_entity_annotation_defaults(self):
        ea = EntityAnnotation(text="species", entity_type="BIOTIC ENTITY")
        assert ea.text == "species"
        assert ea.entity_type == "BIOTIC ENTITY"
        assert ea.start is None
        assert ea.confidence is None

    def test_entity_annotation_full(self):
        ea = EntityAnnotation(
            text="forest", entity_type="SPATIAL ENTITY",
            start=10, end=16, confidence=0.9, reasoning="place", guideline_step="Step 3"
        )
        assert ea.start == 10
        assert ea.end == 16
        assert ea.confidence == 0.9

    def test_relation_annotation(self):
        e1 = EntityAnnotation(text="bear", entity_type="BIOTIC ENTITY")
        e2 = EntityAnnotation(text="habitat", entity_type="SPATIAL ENTITY")
        rel = RelationAnnotation(relation="LOCATED_IN", e1=e1, e2=e2, confidence=0.8)
        assert rel.relation == "LOCATED_IN"
        assert rel.e1.text == "bear"
        assert rel.e2.text == "habitat"

    def test_deliberation_record_defaults(self):
        rec = DeliberationRecord(sentence="Test sentence.")
        assert rec.sentence == "Test sentence."
        assert rec.messages == []
        assert rec.final_entities == []
        assert rec.final_relations == []
        assert rec.agreement_score is None
        assert rec.rounds_used == 0

    def test_entity_annotation_asdict(self):
        ea = EntityAnnotation(text="oak", entity_type="BIOTIC ENTITY", confidence=0.95)
        d = ea.model_dump()
        assert d["text"] == "oak"
        assert d["entity_type"] == "BIOTIC ENTITY"
        assert d["confidence"] == 0.95


# ─────────────────────────────────────────────────────────────
# load_schema tests
# ─────────────────────────────────────────────────────────────

class TestLoadSchema:
    def test_load_json(self, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SAMPLE_SCHEMA))
        result = load_schema(schema_file)
        assert result == SAMPLE_SCHEMA

    def test_load_python_file_with_schema_var(self, tmp_path):
        py_file = tmp_path / "schema.py"
        py_file.write_text(f"SCHEMA = {repr(SAMPLE_SCHEMA)}\n")
        result = load_schema(py_file)
        assert result == SAMPLE_SCHEMA

    def test_load_python_file_with_seeds_var(self, tmp_path):
        py_file = tmp_path / "seeds.py"
        py_file.write_text(f"SEEDS = {repr(SAMPLE_SEEDS)}\n")
        result = load_schema(py_file)
        assert result == SAMPLE_SEEDS

    def test_load_python_bare_expression(self, tmp_path):
        py_file = tmp_path / "bare.py"
        py_file.write_text(repr({"REL": [["A", "B"]]}))
        result = load_schema(py_file)
        assert result == {"REL": [["A", "B"]]}

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("col1,col2\n")
        with pytest.raises(ValueError, match="Unsupported schema format"):
            load_schema(bad_file)

    def test_non_dict_python_raises(self, tmp_path):
        py_file = tmp_path / "list_schema.py"
        py_file.write_text("SCHEMA = [1, 2, 3]\n")
        with pytest.raises(ValueError, match="Could not read dict"):
            load_schema(py_file)


# ─────────────────────────────────────────────────────────────
# _init_tool_state and tool-function tests
# ─────────────────────────────────────────────────────────────

class TestInitToolState:
    def test_builds_type_pair_lookup(self):
        _init_tool_state(SAMPLE_SCHEMA, [], {})
        assert mod._TYPE_PAIR_TO_RELATIONS[("BIOTIC ENTITY", "BIOTIC PROPERTY")] == ["HAS_PROPERTY"]
        assert "LOCATED_IN" in mod._TYPE_PAIR_TO_RELATIONS[("BIOTIC ENTITY", "SPATIAL ENTITY")]

    def test_infers_entity_types(self):
        _init_tool_state(SAMPLE_SCHEMA, [], {})
        assert "BIOTIC ENTITY" in mod._ALL_ENTITY_TYPES
        assert "SPATIAL ENTITY" in mod._ALL_ENTITY_TYPES

    def test_override_entity_types_list(self):
        custom_types = ["TYPE_A", "TYPE_B"]
        _init_tool_state(SAMPLE_SCHEMA, [], {}, entity_types_list=custom_types)
        assert mod._ALL_ENTITY_TYPES == {"TYPE_A", "TYPE_B"}

    def test_guideline_sections_stored(self):
        _init_tool_state({}, SAMPLE_GUIDELINE, {})
        assert mod._GUIDELINE_SECTIONS == SAMPLE_GUIDELINE

    def test_seed_examples_stored(self):
        _init_tool_state({}, [], SAMPLE_SEEDS)
        assert mod._SEED_EXAMPLES == SAMPLE_SEEDS


class TestSchemaLookup:
    def test_forward_relation_found(self, initialized_state):
        result = json.loads(schema_lookup("BIOTIC ENTITY", "BIOTIC PROPERTY"))
        assert "HAS_PROPERTY" in result["valid_relations_forward"]
        assert result["e1_known"] is True
        assert result["e2_known"] is True

    def test_reverse_relation_found(self, initialized_state):
        result = json.loads(schema_lookup("BIOTIC PROPERTY", "BIOTIC ENTITY"))
        assert "HAS_PROPERTY" in result["valid_relations_reverse"]

    def test_unknown_types(self, initialized_state):
        result = json.loads(schema_lookup("UNKNOWN_TYPE", "ANOTHER_UNKNOWN"))
        assert result["e1_known"] is False
        assert result["e2_known"] is False
        assert result["valid_relations_forward"] == []

    def test_case_insensitive_lookup(self, initialized_state):
        result = json.loads(schema_lookup("biotic entity", "biotic property"))
        assert "HAS_PROPERTY" in result["valid_relations_forward"]

    def test_whitespace_stripped(self, initialized_state):
        result = json.loads(schema_lookup("  BIOTIC ENTITY  ", "  SPATIAL ENTITY  "))
        assert "LOCATED_IN" in result["valid_relations_forward"]

    def test_no_relation_between_unrelated_types(self, initialized_state):
        result = json.loads(schema_lookup("ABIOTIC ENTITY", "BIOTIC PROPERTY"))
        assert result["valid_relations_forward"] == []
        assert result["valid_relations_reverse"] == []


class TestGuidelineSearch:
    def test_matching_query(self, initialized_state):
        result = json.loads(guideline_search("biotic habitat property"))
        assert result["status"] == "matched"
        assert result["backend"] == "lexical"
        assert result["suggestion"] is None
        titles = [r["title"] for r in result["results"]]
        assert any("Biotic" in t or "biotic" in t.lower() for t in titles)

    def test_no_match_returns_explicit_status(self, initialized_state):
        result = json.loads(guideline_search("xyzzy frobnicator"))
        assert result["status"] == "no_match"
        assert result["backend"] == "lexical"
        assert result["results"] == []
        assert result["suggestion"] == "This concept may not be covered in the guideline"

    def test_returns_at_most_3_results(self, initialized_state):
        result = json.loads(guideline_search("entity property concept spatial biotic"))
        assert len(result["results"]) <= 3

    def test_title_tokens_weighted_higher(self, initialized_state):
        # "Step 1" appears in title only → should rank first
        result = json.loads(guideline_search("Abstract"))
        assert len(result["results"]) > 0

    def test_embedding_backend_can_match_semantic_section(self, monkeypatch):
        def fake_embed(texts):
            vectors = []
            for text in texts:
                if text == "food insecurity":
                    vectors.append([1.0, 0.0])
                elif "Anthropogenic" in text or "Human system" in text:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

        monkeypatch.setattr(mod, "_embed_guideline_texts", fake_embed)
        _init_tool_state(
            SAMPLE_SCHEMA,
            SAMPLE_GUIDELINE,
            SAMPLE_SEEDS,
            guideline_search_backend="embedding",
        )

        result = json.loads(guideline_search("food insecurity"))

        assert result["status"] == "matched"
        assert result["backend"] == "embedding"
        assert result["results"][0]["title"] == "Step 4 — Anthropogenic"

    def test_embedding_backend_falls_back_to_lexical(self, monkeypatch):
        def failing_embed(texts):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(mod, "_embed_guideline_texts", failing_embed)
        _init_tool_state(
            SAMPLE_SCHEMA,
            SAMPLE_GUIDELINE,
            SAMPLE_SEEDS,
            guideline_search_backend="embedding",
        )

        result = json.loads(guideline_search("biotic habitat property"))

        assert result["status"] == "matched"
        assert result["backend"] == "lexical"
        assert result["results"]


class TestConsistencyCheck:
    def test_exact_span_match(self, initialized_state):
        results = json.loads(consistency_check("species"))
        assert len(results) > 0
        assert results[0]["entity_text"] == "species"
        assert results[0]["entity_type"] == "BIOTIC ENTITY"

    def test_single_token_match(self, initialized_state):
        results = json.loads(consistency_check("population"))
        assert len(results) > 0
        assert results[0]["entity_text"] == "population"

    def test_single_token_does_not_match_phrase(self):
        seeds = {
            "HAS_PROPERTY": [
                {
                    "e1": {"text": "food insecurity", "type": "ANTHROPOGENIC PROPERTY"},
                    "e2": {"text": "dietary quality", "type": "BIOTIC PROPERTY"},
                    "sentence": "Food insecurity affects dietary quality.",
                }
            ]
        }
        _init_tool_state({}, [], seeds)
        results = json.loads(consistency_check("food"))
        assert results == []

    def test_single_token_does_not_match_substring_inside_token(self):
        seeds = {
            "HAS_PROPERTY": [
                {
                    "e1": {"text": "seafood", "type": "BIOTIC ENTITY"},
                    "e2": {"text": "availability", "type": "ANTHROPOGENIC PROPERTY"},
                    "sentence": "Seafood availability varies.",
                }
            ]
        }
        _init_tool_state({}, [], seeds)
        results = json.loads(consistency_check("food"))
        assert results == []

    def test_phrase_matches_near_variant_by_token_overlap(self):
        seeds = {
            "IS_AFFECTING": [
                {
                    "e1": {"text": "chronic food insecurity", "type": "ANTHROPOGENIC PROPERTY"},
                    "e2": {"text": "anxiety", "type": "BIOTIC PROPERTY"},
                    "sentence": "Chronic food insecurity is associated with anxiety.",
                }
            ]
        }
        _init_tool_state({}, [], seeds)
        results = json.loads(consistency_check("food insecurity"))
        assert len(results) == 1
        assert results[0]["entity_text"] == "chronic food insecurity"
        assert results[0]["matched_tokens"] == ["food", "insecurity"]
        assert results[0]["match_score"] >= 0.5

    def test_stopword_only_query_does_not_match(self, initialized_state):
        results = json.loads(consistency_check("of the"))
        assert results == []

    def test_no_match_returns_empty(self, initialized_state):
        results = json.loads(consistency_check("nonexistent_span_xyz"))
        assert results == []

    def test_returns_at_most_5_matches(self, initialized_state):
        # Build a schema with many seeds containing "organism"
        many_seeds = {
            f"REL_{i}": [
                {
                    "e1": {"text": "organism", "type": "BIOTIC ENTITY"},
                    "e2": {"text": "habitat", "type": "SPATIAL ENTITY"},
                    "sentence": f"Sentence {i}.",
                }
            ]
            for i in range(10)
        }
        _init_tool_state({}, [], many_seeds)
        results = json.loads(consistency_check("organism"))
        assert len(results) <= 5


class TestListEntityTypes:
    def test_returns_sorted_json_list(self, initialized_state):
        result = json.loads(list_entity_types())
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_contains_known_types(self, initialized_state):
        result = json.loads(list_entity_types())
        assert "BIOTIC ENTITY" in result
        assert "SPATIAL ENTITY" in result

    def test_empty_when_not_initialized(self):
        # reset_tool_state fixture already cleared state
        result = json.loads(list_entity_types())
        assert result == []


# ─────────────────────────────────────────────────────────────
# _get_embedded_guideline
# ─────────────────────────────────────────────────────────────

class TestEmbeddedGuideline:
    def test_returns_non_empty_list(self):
        sections = _get_embedded_guideline()
        assert isinstance(sections, list)
        assert len(sections) > 0

    def test_sections_have_required_keys(self):
        sections = _get_embedded_guideline()
        for sec in sections:
            assert "title" in sec
            assert "content" in sec


# ─────────────────────────────────────────────────────────────
# _critic_is_satisfied
# ─────────────────────────────────────────────────────────────

class TestCriticIsSatisfied:
    def test_terminate_keyword(self):
        assert _critic_is_satisfied({"content": "All good.\nTERMINATE"}) is True

    def test_terminate_in_middle_does_not_stop(self):
        assert _critic_is_satisfied({"content": "TERMINATE some trailing text"}) is False

    def test_embedded_terminate_does_not_stop_tool_executor(self):
        msg = {"content": "## Final review\nTERMINATE\n\nProduce the final annotation."}
        assert _is_final_terminate_msg(msg) is False

    def test_final_standalone_terminate_stops(self):
        msg = {"content": '{"ok": true}\nTERMINATE'}
        assert _is_final_terminate_msg(msg) is True

    def test_empty_disagreements_and_missing(self):
        payload = json.dumps({"disagreements": [], "missing_annotations": []})
        assert _critic_is_satisfied({"content": payload}) is True

    def test_non_empty_disagreements(self):
        payload = json.dumps({
            "disagreements": [{"target": "forest", "annotator_label": "X"}],
            "missing_annotations": [],
        })
        assert _critic_is_satisfied({"content": payload}) is False

    def test_non_empty_missing_annotations(self):
        payload = json.dumps({
            "disagreements": [],
            "missing_annotations": [{"text": "oak", "entity_type": "BIOTIC ENTITY"}],
        })
        assert _critic_is_satisfied({"content": payload}) is False

    def test_empty_content(self):
        assert _critic_is_satisfied({"content": ""}) is False

    def test_content_is_none(self):
        assert _critic_is_satisfied({"content": None}) is False

    def test_thinking_block_stripped(self):
        content = (
            "<think>Internal reasoning here.</think>\n"
            + json.dumps({"disagreements": [], "missing_annotations": []})
        )
        assert _critic_is_satisfied({"content": content}) is True

    def test_invalid_json_returns_false(self):
        assert _critic_is_satisfied({"content": "{ invalid json }"}) is False


# ─────────────────────────────────────────────────────────────
# _try_parse_json  (via MultiAgentAnnotator static method)
# ─────────────────────────────────────────────────────────────

class TestTryParseJson:
    def test_plain_json(self):
        text = '{"key": "value"}'
        result = MultiAgentAnnotator._try_parse_json(text)
        assert result == {"key": "value"}

    def test_returns_last_json_object(self):
        text = '{"first": 1} some text {"second": 2}'
        result = MultiAgentAnnotator._try_parse_json(text)
        assert result == {"second": 2}

    def test_strips_thinking_block(self):
        text = "<think>reasoning</think>\n{\"key\": 42}"
        result = MultiAgentAnnotator._try_parse_json(text)
        assert result == {"key": 42}

    def test_no_json_returns_none(self):
        result = MultiAgentAnnotator._try_parse_json("just plain text")
        assert result is None

    def test_empty_string_returns_none(self):
        result = MultiAgentAnnotator._try_parse_json("")
        assert result is None

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = MultiAgentAnnotator._try_parse_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_schema_key_parser_rejects_nested_relation_fragment(self):
        text = '''
        {
          "entities": [
            {"text": "oak", "entity_type": "BIOTIC ENTITY"}
          ],
          "relations": [
            {"relation": "LOCATED_IN", "e1_text": "oak", "e1_type": "BIOTIC ENTITY", "e2_text": "forest", "e2_type": "SPATIAL ENTITY"}
          ],
          "uncertain_cases": [
            "bad" (this parenthetical is outside the JSON string)
          ],
          "reasoning": "broken"
        }
        '''

        generic = MultiAgentAnnotator._try_parse_json(text)
        strict = MultiAgentAnnotator._try_parse_json_with_keys(
            text, mod.ANNOTATOR_REQUIRED_KEYS
        )

        assert generic is not None
        assert generic["relation"] == "LOCATED_IN"
        assert strict is None

    def test_malformed_uncertain_cases_does_not_parse_as_annotator_output(self):
        text = '''
        {
          "entities": [{"text": "oak", "entity_type": "BIOTIC ENTITY"}],
          "relations": [],
          "uncertain_cases": [
            "characterised" (could be a process nominalization but functions as a verb here),
            "more commonly (qualitative descriptor)"
          ],
          "reasoning": "broken"
        }
        '''

        assert MultiAgentAnnotator._parse_annotator_output(text) is None

    def test_valid_uncertain_cases_parses_as_annotator_output(self):
        text = json.dumps({
            "entities": [{"text": "oak", "entity_type": "BIOTIC ENTITY"}],
            "relations": [],
            "uncertain_cases": [
                "characterised (could be a process nominalization)"
            ],
            "reasoning": "valid",
        })

        parsed = MultiAgentAnnotator._parse_annotator_output(text)

        assert parsed is not None
        assert parsed.entities[0].text == "oak"


class TestDeliberationTurnBudget:
    def test_turn_budget_includes_tool_exchange_buffer(self):
        old_budget = 1 * 2 + 1
        new_budget = MultiAgentAnnotator._deliberation_max_turns(1)

        assert new_budget > old_budget
        assert new_budget >= 10

    def test_adjudicator_initial_budget_allows_tool_exchange(self):
        assert MultiAgentAnnotator._adjudicator_max_turns() > 3

    def test_empty_adjudicator_output_uses_generation_retry(self):
        assert MultiAgentAnnotator._adjudicator_retry_mode("") == "generate"
        assert MultiAgentAnnotator._adjudicator_retry_mode("   \n") == "generate"

    def test_malformed_adjudicator_output_uses_repair_retry(self):
        assert MultiAgentAnnotator._adjudicator_retry_mode("{not json") == "repair"


# ─────────────────────────────────────────────────────────────
# _extract_last_json (via MultiAgentAnnotator static method)
# ─────────────────────────────────────────────────────────────

class TestExtractLastJson:
    def test_finds_final_entities(self):
        history = [
            {"content": "some text"},
            {"content": '{"final_entities": [{"text": "oak", "entity_type": "BIOTIC ENTITY"}]}'},
        ]
        result = MultiAgentAnnotator._extract_last_json(history)
        assert result is not None
        assert result["final_entities"][0]["text"] == "oak"

    def test_finds_entities_key_as_fallback(self):
        history = [
            {"content": '{"entities": [{"text": "forest"}]}'},
        ]
        result = MultiAgentAnnotator._extract_last_json(history)
        assert result is not None

    def test_returns_none_when_no_entities_key(self):
        history = [
            {"content": '{"other_key": "value"}'},
        ]
        result = MultiAgentAnnotator._extract_last_json(history)
        assert result is None

    def test_empty_history_returns_none(self):
        assert MultiAgentAnnotator._extract_last_json([]) is None

    def test_walks_backward(self):
        history = [
            {"content": '{"final_entities": [{"text": "first"}]}'},
            {"content": "no json here"},
            {"content": '{"final_entities": [{"text": "last"}]}'},
        ]
        result = MultiAgentAnnotator._extract_last_json(history)
        assert result["final_entities"][0]["text"] == "last"

    def test_extract_last_json_with_keys_ignores_incomplete_output(self):
        history = [
            {"content": '{"final_entities": [{"text": "oak"}]}'},
            {"content": json.dumps({
                "final_entities": [{"text": "oak", "entity_type": "BIOTIC ENTITY"}],
                "final_relations": [],
                "disagreement_resolutions": [],
                "flagged_for_human_review": [],
            })},
        ]

        result = MultiAgentAnnotator._extract_last_json_with_keys(
            history, mod.ADJUDICATOR_REQUIRED_KEYS
        )

        assert result is not None
        assert "final_relations" in result


# ─────────────────────────────────────────────────────────────
# _constrain_adjudicator_output
# ─────────────────────────────────────────────────────────────

class TestConstrainAdjudicatorOutput:
    def test_missing_critic_keeps_annotator_output(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="BIOTIC PROCESS")
            ],
            relations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=[
                EntityAnnotation(
                    text="food insecurity",
                    entity_type="ANTHROPOGENIC ENTITY",
                )
            ],
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, None, adjudicator
        )

        assert result.status == "critic_missing_fallback"
        assert result.final_entities[0].entity_type == "BIOTIC PROCESS"
        assert result.audit["warnings"]

    def test_missing_adjudicator_keeps_annotator_output(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="BIOTIC PROCESS")
            ],
            relations=[],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(
                    target="food insecurity",
                    annotator_label="BIOTIC PROCESS",
                    proposed_label="ANTHROPOGENIC PROPERTY",
                )
            ],
            missing_annotations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, None
        )

        assert result.status == "adjudicator_parse_failed"
        assert result.final_entities[0].entity_type == "BIOTIC PROCESS"
        assert result.audit["warnings"]

    def test_consensus_entity_label_change_is_rejected(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="anxiety", entity_type="BIOTIC PROPERTY")
            ],
            relations=[],
        )
        critic = CriticOutput(disagreements=[], missing_annotations=[])
        adjudicator = AdjudicatorOutput(
            final_entities=[
                EntityAnnotation(
                    text="anxiety",
                    entity_type="ANTHROPOGENIC PROPERTY",
                )
            ],
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.final_entities[0].entity_type == "BIOTIC PROPERTY"
        assert any("anxiety" in item for item in result.audit["rejected_changes"])

    def test_critic_disagreement_authorizes_entity_label_change(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(
                    text="food insecurity",
                    entity_type="ANTHROPOGENIC PROPERTY",
                ),
                EntityAnnotation(
                    text="dietary quality",
                    entity_type="QUALITATIVE PROPERTY",
                ),
            ],
            relations=[
                RelationFlat(
                    relation="HAS_PROPERTY",
                    e1_text="food insecurity",
                    e1_type="ANTHROPOGENIC PROPERTY",
                    e2_text="dietary quality",
                    e2_type="QUALITATIVE PROPERTY",
                )
            ],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(
                    target="dietary quality",
                    annotator_label="QUALITATIVE PROPERTY",
                    proposed_label="ANTHROPOGENIC PROPERTY",
                )
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=[
                EntityAnnotation(
                    text="food insecurity",
                    entity_type="ANTHROPOGENIC PROPERTY",
                ),
                EntityAnnotation(
                    text="dietary quality",
                    entity_type="ANTHROPOGENIC PROPERTY",
                ),
            ],
            final_relations=[
                RelationFlat(
                    relation="HAS_PROPERTY",
                    e1_text="food insecurity",
                    e1_type="ANTHROPOGENIC PROPERTY",
                    e2_text="dietary quality",
                    e2_type="ANTHROPOGENIC PROPERTY",
                )
            ],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        by_text = {e.text: e for e in result.final_entities}
        assert by_text["dietary quality"].entity_type == "ANTHROPOGENIC PROPERTY"
        assert result.final_relations[0].e2_type == "ANTHROPOGENIC PROPERTY"
        assert any("dietary quality" in item for item in result.audit["allowed_changes"])

    def test_missing_annotation_authorizes_entity_addition(self):
        annotator = AnnotatorOutput(entities=[], relations=[])
        critic = CriticOutput(
            disagreements=[],
            missing_annotations=[
                CriticMissingAnnotation(
                    text="chronic compromises",
                    entity_type="ANTHROPOGENIC PROPERTY",
                )
            ],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=[
                EntityAnnotation(
                    text="chronic compromises",
                    entity_type="ANTHROPOGENIC PROPERTY",
                )
            ],
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert len(result.final_entities) == 1
        assert result.final_entities[0].text == "chronic compromises"

    def test_adjudicator_only_entity_addition_is_rejected(self):
        annotator = AnnotatorOutput(entities=[], relations=[])
        critic = CriticOutput(disagreements=[], missing_annotations=[])
        adjudicator = AdjudicatorOutput(
            final_entities=[
                EntityAnnotation(text="accessing food", entity_type="ANTHROPOGENIC PROCESS")
            ],
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.final_entities == []
        assert any("accessing food" in item for item in result.audit["rejected_changes"])

    def test_relation_change_requires_critic_disagreement(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="anxiety", entity_type="BIOTIC PROPERTY"),
                EntityAnnotation(text="accessing food", entity_type="BIOTIC PROCESS"),
            ],
            relations=[
                RelationFlat(
                    relation="IS_AFFECTING",
                    e1_text="anxiety",
                    e1_type="BIOTIC PROPERTY",
                    e2_text="accessing food",
                    e2_type="BIOTIC PROCESS",
                )
            ],
        )
        critic = CriticOutput(disagreements=[], missing_annotations=[])
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[
                RelationFlat(
                    relation="HAS_PROPERTY",
                    e1_text="anxiety",
                    e1_type="BIOTIC PROPERTY",
                    e2_text="accessing food",
                    e2_type="BIOTIC PROCESS",
                )
            ],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.final_relations[0].relation == "IS_AFFECTING"
        assert any("relation" in item for item in result.audit["rejected_changes"])

    def test_relation_disagreement_authorizes_relation_change(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="anxiety", entity_type="BIOTIC PROPERTY"),
                EntityAnnotation(text="accessing food", entity_type="BIOTIC PROCESS"),
            ],
            relations=[
                RelationFlat(
                    relation="IS_AFFECTING",
                    e1_text="anxiety",
                    e1_type="BIOTIC PROPERTY",
                    e2_text="accessing food",
                    e2_type="BIOTIC PROCESS",
                )
            ],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(target="anxiety IS_AFFECTING accessing food")
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[
                RelationFlat(
                    relation="HAS_PROPERTY",
                    e1_text="anxiety",
                    e1_type="BIOTIC PROPERTY",
                    e2_text="accessing food",
                    e2_type="BIOTIC PROCESS",
                )
            ],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.final_relations[0].relation == "HAS_PROPERTY"

    def test_disputed_relation_omitted_by_adjudicator_is_removed(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="ANTHROPOGENIC PROPERTY"),
                EntityAnnotation(text="anxiety", entity_type="BIOTIC PROPERTY"),
            ],
            relations=[
                RelationFlat(
                    relation="IS_AFFECTING",
                    e1_text="food insecurity",
                    e1_type="ANTHROPOGENIC PROPERTY",
                    e2_text="anxiety",
                    e2_type="BIOTIC PROPERTY",
                )
            ],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(
                    target="IS_AFFECTING: food insecurity -> anxiety",
                    annotator_label="IS_AFFECTING",
                    proposed_label="INVALID",
                )
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.final_relations == []
        assert any("removed" in item for item in result.audit["allowed_changes"])

    def test_relation_dispute_target_can_omit_relation_name(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="ANTHROPOGENIC PROPERTY"),
                EntityAnnotation(text="anxiety", entity_type="BIOTIC PROPERTY"),
            ],
            relations=[
                RelationFlat(
                    relation="IS_AFFECTING",
                    e1_text="food insecurity",
                    e1_type="ANTHROPOGENIC PROPERTY",
                    e2_text="anxiety",
                    e2_type="BIOTIC PROPERTY",
                )
            ],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(
                    target="food insecurity -> anxiety",
                    annotator_label="IS_AFFECTING",
                    proposed_label="INVALID",
                )
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.final_relations == []
        assert any("removed" in item for item in result.audit["allowed_changes"])

    def test_transcript_style_disputed_relations_reduce_to_adjudicator_subset(self):
        entities = [
            EntityAnnotation(text="high-income countries", entity_type="ANTHROPOGENIC ENTITY"),
            EntityAnnotation(text="high-income", entity_type="ANTHROPOGENIC PROPERTY"),
            EntityAnnotation(text="food insecurity", entity_type="ANTHROPOGENIC PROPERTY"),
            EntityAnnotation(text="dietary quality", entity_type="BIOTIC PROPERTY"),
            EntityAnnotation(text="anxiety", entity_type="BIOTIC PROPERTY"),
            EntityAnnotation(text="accessing food", entity_type="ANTHROPOGENIC PROCESS"),
            EntityAnnotation(text="food", entity_type="BIOTIC ENTITY"),
        ]
        annotator = AnnotatorOutput(
            entities=entities,
            relations=[
                RelationFlat(relation="LOCATED_IN", e1_text="food insecurity", e1_type="ANTHROPOGENIC PROPERTY", e2_text="high-income countries", e2_type="ANTHROPOGENIC ENTITY"),
                RelationFlat(relation="HAS_PROPERTY", e1_text="high-income countries", e1_type="ANTHROPOGENIC ENTITY", e2_text="high-income", e2_type="ANTHROPOGENIC PROPERTY"),
                RelationFlat(relation="IS_AFFECTING", e1_text="food insecurity", e1_type="ANTHROPOGENIC PROPERTY", e2_text="dietary quality", e2_type="BIOTIC PROPERTY"),
                RelationFlat(relation="IS_AFFECTING", e1_text="food insecurity", e1_type="ANTHROPOGENIC PROPERTY", e2_text="anxiety", e2_type="BIOTIC PROPERTY"),
                RelationFlat(relation="IS_AFFECTING", e1_text="accessing food", e1_type="ANTHROPOGENIC PROCESS", e2_text="anxiety", e2_type="BIOTIC PROPERTY"),
                RelationFlat(relation="LOCATED_IN", e1_text="accessing food", e1_type="ANTHROPOGENIC PROCESS", e2_text="food", e2_type="BIOTIC ENTITY"),
            ],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(target="high-income countries", annotator_label="ANTHROPOGENIC ENTITY", proposed_label="SPATIAL ENTITY"),
                CriticDisagreement(target="dietary quality", annotator_label="BIOTIC PROPERTY", proposed_label="ANTHROPOGENIC PROPERTY"),
                CriticDisagreement(target="food insecurity -> high-income countries", annotator_label="LOCATED_IN", proposed_label="INVALID"),
                CriticDisagreement(target="food insecurity -> dietary quality", annotator_label="IS_AFFECTING", proposed_label="INVALID"),
                CriticDisagreement(target="food insecurity -> anxiety", annotator_label="IS_AFFECTING", proposed_label="INVALID"),
                CriticDisagreement(target="accessing food -> food", annotator_label="LOCATED_IN", proposed_label="INVALID"),
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=[
                EntityAnnotation(text="high-income countries", entity_type="SPATIAL ENTITY"),
                EntityAnnotation(text="high-income", entity_type="ANTHROPOGENIC PROPERTY"),
                EntityAnnotation(text="food insecurity", entity_type="ANTHROPOGENIC PROPERTY"),
                EntityAnnotation(text="dietary quality", entity_type="ANTHROPOGENIC PROPERTY"),
                EntityAnnotation(text="anxiety", entity_type="BIOTIC PROPERTY"),
                EntityAnnotation(text="accessing food", entity_type="ANTHROPOGENIC PROCESS"),
                EntityAnnotation(text="food", entity_type="BIOTIC ENTITY"),
            ],
            final_relations=[
                RelationFlat(relation="HAS_PROPERTY", e1_text="high-income countries", e1_type="SPATIAL ENTITY", e2_text="high-income", e2_type="ANTHROPOGENIC PROPERTY"),
                RelationFlat(relation="IS_AFFECTING", e1_text="accessing food", e1_type="ANTHROPOGENIC PROCESS", e2_text="anxiety", e2_type="BIOTIC PROPERTY"),
            ],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        final_pairs = {
            (r.relation, r.e1_text, r.e2_text)
            for r in result.final_relations
        }
        assert final_pairs == {
            ("HAS_PROPERTY", "high-income countries", "high-income"),
            ("IS_AFFECTING", "accessing food", "anxiety"),
        }
        assert sum("removed" in item for item in result.audit["allowed_changes"]) == 4

    def test_uncertain_cases_are_flagged_even_when_adjudicator_omits_flags(self):
        uncertainty = "food insecurity: BIOTIC PROCESS vs BIOTIC PROPERTY"
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="BIOTIC PROCESS")
            ],
            relations=[],
            uncertain_cases=[uncertainty],
        )
        critic = CriticOutput(disagreements=[], missing_annotations=[])
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[],
            flagged_for_human_review=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.flagged_for_human_review == [uncertainty]
        assert result.audit["human_review_flags"] == [
            {"flag": uncertainty, "sources": ["annotator_uncertain_case"]}
        ]

    def test_critical_disagreement_without_guideline_reference_is_flagged(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="BIOTIC PROCESS")
            ],
            relations=[],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(
                    target="food insecurity",
                    annotator_label="BIOTIC PROCESS",
                    proposed_label="BIOTIC PROPERTY",
                    guideline_reference="n/a",
                    severity="critical",
                )
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.flagged_for_human_review == ["food insecurity"]
        assert result.audit["human_review_flags"] == [
            {"flag": "food insecurity", "sources": ["critic_critical_no_guideline"]}
        ]

    def test_critical_disagreement_with_guideline_reference_is_not_auto_flagged(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="BIOTIC PROCESS")
            ],
            relations=[],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(
                    target="food insecurity",
                    annotator_label="BIOTIC PROCESS",
                    proposed_label="BIOTIC PROPERTY",
                    guideline_reference="Step 5",
                    severity="critical",
                )
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.flagged_for_human_review == []
        assert result.audit["human_review_flags"] == []

    def test_human_review_flags_are_deduplicated_with_all_sources(self):
        annotator = AnnotatorOutput(
            entities=[
                EntityAnnotation(text="food insecurity", entity_type="BIOTIC PROCESS")
            ],
            relations=[],
            uncertain_cases=["food insecurity"],
        )
        critic = CriticOutput(
            disagreements=[
                CriticDisagreement(
                    target="food insecurity",
                    severity="critical",
                    guideline_reference="",
                )
            ],
            missing_annotations=[],
        )
        adjudicator = AdjudicatorOutput(
            final_entities=annotator.entities,
            final_relations=[],
            flagged_for_human_review=[" food   insecurity "],
        )

        result = MultiAgentAnnotator._constrain_adjudicator_output(
            annotator, critic, adjudicator
        )

        assert result.flagged_for_human_review == ["food insecurity"]
        assert result.audit["human_review_flags"] == [
            {
                "flag": "food insecurity",
                "sources": [
                    "annotator_uncertain_case",
                    "adjudicator_flag",
                    "critic_critical_no_guideline",
                ],
            }
        ]


# ─────────────────────────────────────────────────────────────
# analyze_disagreements
# ─────────────────────────────────────────────────────────────

class TestAnalyzeDisagreements:
    def _make_record(self, sentence, entities=None, relations=None, score=1.0, rounds=1):
        rec = DeliberationRecord(sentence=sentence)
        rec.final_entities = entities or []
        rec.final_relations = relations or []
        rec.agreement_score = score
        rec.rounds_used = rounds
        return rec

    def test_empty_records(self):
        stats = analyze_disagreements([])
        assert stats["total_sentences"] == 0
        assert stats["avg_agreement"] == 0
        assert stats["avg_rounds"] == 0

    def test_single_record_averages(self):
        rec = self._make_record("Test.", score=0.8, rounds=2)
        stats = analyze_disagreements([rec])
        assert stats["total_sentences"] == 1
        assert stats["avg_agreement"] == pytest.approx(0.8)
        assert stats["avg_rounds"] == pytest.approx(2.0)

    def test_entity_type_distribution(self):
        e1 = EntityAnnotation(text="oak", entity_type="BIOTIC ENTITY")
        e2 = EntityAnnotation(text="forest", entity_type="SPATIAL ENTITY")
        rec = self._make_record("Oak in forest.", entities=[e1, e2])
        stats = analyze_disagreements([rec])
        assert stats["entity_type_distribution"]["BIOTIC ENTITY"] == 1
        assert stats["entity_type_distribution"]["SPATIAL ENTITY"] == 1

    def test_relation_distribution(self):
        e1 = EntityAnnotation(text="bear", entity_type="BIOTIC ENTITY")
        e2 = EntityAnnotation(text="forest", entity_type="SPATIAL ENTITY")
        rel = RelationAnnotation(relation="LOCATED_IN", e1=e1, e2=e2)
        rec = self._make_record("Bear in forest.", relations=[rel])
        stats = analyze_disagreements([rec])
        assert stats["relation_distribution"]["LOCATED_IN"] == 1

    def test_multiple_records_averaging(self):
        r1 = self._make_record("S1", score=0.6, rounds=1)
        r2 = self._make_record("S2", score=1.0, rounds=3)
        stats = analyze_disagreements([r1, r2])
        assert stats["avg_agreement"] == pytest.approx(0.8)
        assert stats["avg_rounds"] == pytest.approx(2.0)

    def test_none_agreement_score_treated_as_zero(self):
        rec = self._make_record("S1")
        rec.agreement_score = None
        stats = analyze_disagreements([rec])
        assert stats["avg_agreement"] == pytest.approx(0.0)

    def test_flagged_for_review_includes_sentence_context(self):
        rec = self._make_record("Food insecurity affects anxiety.")
        rec.flagged_for_human_review = ["food insecurity"]

        stats = analyze_disagreements([rec])

        assert stats["flagged_for_review"] == [
            {
                "sentence": "Food insecurity affects anxiety.",
                "flag": "food insecurity",
            }
        ]


# ─────────────────────────────────────────────────────────────
# _append_jsonl
# ─────────────────────────────────────────────────────────────

class TestAppendJsonl:
    def test_appends_valid_jsonl(self, tmp_path):
        path = tmp_path / "output.jsonl"
        e = EntityAnnotation(text="oak", entity_type="BIOTIC ENTITY", confidence=0.9)
        rec = DeliberationRecord(sentence="Oak grows here.")
        rec.final_entities = [e]
        rec.agreement_score = 1.0
        rec.rounds_used = 1

        MultiAgentAnnotator._append_jsonl(rec, path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["sentence"] == "Oak grows here."
        assert obj["final_entities"][0]["text"] == "oak"
        assert obj["agreement_score"] == 1.0

    def test_appends_multiple_records(self, tmp_path):
        path = tmp_path / "output.jsonl"
        for i in range(3):
            rec = DeliberationRecord(sentence=f"Sentence {i}.")
            rec.agreement_score = 1.0
            rec.rounds_used = 0
            MultiAgentAnnotator._append_jsonl(rec, path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_output_is_valid_json_per_line(self, tmp_path):
        path = tmp_path / "output.jsonl"
        rec = DeliberationRecord(sentence="Test.")
        rec.agreement_score = 0.5
        rec.rounds_used = 2
        MultiAgentAnnotator._append_jsonl(rec, path)
        for line in path.read_text().strip().splitlines():
            obj = json.loads(line)  # must not raise
            assert "sentence" in obj

    def test_adjudication_audit_serializes(self, tmp_path):
        path = tmp_path / "output.jsonl"
        rec = DeliberationRecord(sentence="Test.")
        rec.adjudication_status = "constrained"
        rec.adjudication_audit = {
            "preserved_consensus": ["entity \"oak\": BIOTIC ENTITY"],
            "allowed_changes": [],
            "rejected_changes": [],
            "human_review_flags": [],
            "warnings": [],
        }

        MultiAgentAnnotator._append_jsonl(rec, path)

        obj = json.loads(path.read_text().strip())
        assert obj["adjudication_status"] == "constrained"
        assert obj["adjudication_audit"]["preserved_consensus"]

    def test_flagged_for_human_review_serializes(self, tmp_path):
        path = tmp_path / "output.jsonl"
        rec = DeliberationRecord(sentence="Food insecurity affects anxiety.")
        rec.flagged_for_human_review = ["food insecurity"]

        MultiAgentAnnotator._append_jsonl(rec, path)

        obj = json.loads(path.read_text().strip())
        assert obj["flagged_for_human_review"] == ["food insecurity"]


# ─────────────────────────────────────────────────────────────
# build_llm_config (error paths only — no real API calls)
# ─────────────────────────────────────────────────────────────

class TestBuildLlmConfig:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_llm_config("nonexistent-model-xyz")

    def test_known_model_with_empty_key_env_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPEN_WEB_UI_API_KEY", raising=False)
        # qwen3-32B has api_key=None and relies on env vars
        with pytest.raises(ValueError, match="API key required"):
            build_llm_config("qwen3-32B")

    def test_known_model_with_explicit_key_succeeds(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        config = build_llm_config("qwen3-32B")
        assert config is not None
