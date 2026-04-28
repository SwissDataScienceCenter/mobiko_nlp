"""
Multi-agent annotation system for biodiversity entity & relation extraction.
Built on AG2 (formerly AutoGen).

Install:
    pip install "ag2[openai]"

Architecture:
    Agent 1 (Annotator)  – labels entities/relations following the guideline.
    Agent 2 (Critic)     – reviews labels, finds violations, proposes fixes.
    Agent 3 (Adjudicator)– resolves disagreements, produces final labels.

All three agents share registered tools:
    - schema_lookup      : validate (entity_type, entity_type) → allowed relations
    - guideline_search   : retrieve relevant sections from the labelling guideline
    - consistency_check  : look up how similar spans were labelled in seed examples
    - list_entity_types  : list all valid entity types from the schema
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field, ValidationError

from autogen import (
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    register_function,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

class EntityAnnotation(BaseModel):
    text: str
    entity_type: str
    start: Optional[int] = None
    end: Optional[int] = None
    guideline_step: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class RelationAnnotation(BaseModel):
    relation: str
    e1: EntityAnnotation
    e2: EntityAnnotation
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class DeliberationRecord(BaseModel):
    sentence: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    final_entities: List[EntityAnnotation] = Field(default_factory=list)
    final_relations: List[RelationAnnotation] = Field(default_factory=list)
    flagged_for_human_review: List[str] = Field(default_factory=list)
    agreement_score: Optional[float] = None
    rounds_used: int = 0
    adjudication_status: Optional[str] = None
    adjudication_audit: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Agent output schemas
# ─────────────────────────────────────────────────────────────

class RelationFlat(BaseModel):
    """Flat relation format as output by LLM agents (e1_text/e1_type instead of nested model)."""
    relation: str
    e1_text: str
    e1_type: str
    e2_text: str
    e2_type: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    def to_relation_annotation(self) -> RelationAnnotation:
        return RelationAnnotation(
            relation=self.relation,
            e1=EntityAnnotation(text=self.e1_text, entity_type=self.e1_type),
            e2=EntityAnnotation(text=self.e2_text, entity_type=self.e2_type),
            confidence=self.confidence,
            reasoning=self.reasoning,
        )


class AnnotatorOutput(BaseModel):
    entities: List[EntityAnnotation] = Field(default_factory=list)
    relations: List[RelationFlat] = Field(default_factory=list)
    uncertain_cases: List[str] = Field(default_factory=list)
    reasoning: str = ""


class CriticDisagreement(BaseModel):
    target: str = ""
    annotator_label: str = ""
    proposed_label: str = ""
    guideline_reference: str = ""
    severity: str = ""
    explanation: str = ""


class CriticMissingAnnotation(BaseModel):
    text: str = ""
    entity_type: str = ""
    guideline_step: str = ""
    reasoning: str = ""


class AgreementItem(BaseModel):
    target: str = ""
    label: str = ""


class CriticOutput(BaseModel):
    agreements: List[AgreementItem] = Field(default_factory=list)
    disagreements: List[CriticDisagreement] = Field(default_factory=list)
    missing_annotations: List[CriticMissingAnnotation] = Field(default_factory=list)
    reasoning: str = ""


class DisagreementResolution(BaseModel):
    issue: str = ""
    decision: str = ""
    rationale: str = ""


class AdjudicatorOutput(BaseModel):
    final_entities: List[EntityAnnotation] = Field(default_factory=list)
    final_relations: List[RelationFlat] = Field(default_factory=list)
    disagreement_resolutions: List[DisagreementResolution] = Field(default_factory=list)
    flagged_for_human_review: List[str] = Field(default_factory=list)


class ConstrainedAdjudication(BaseModel):
    final_entities: List[EntityAnnotation] = Field(default_factory=list)
    final_relations: List[RelationFlat] = Field(default_factory=list)
    flagged_for_human_review: List[str] = Field(default_factory=list)
    status: str = "constrained"
    audit: Dict[str, Any] = Field(default_factory=dict)


LOW_CONFIDENCE_THRESHOLD = 0.7

ANNOTATOR_REQUIRED_KEYS = {"entities", "relations", "uncertain_cases", "reasoning"}
CRITIC_REQUIRED_KEYS = {"agreements", "disagreements", "missing_annotations", "reasoning"}
ADJUDICATOR_REQUIRED_KEYS = {
    "final_entities",
    "final_relations",
    "disagreement_resolutions",
    "flagged_for_human_review",
}


# ─────────────────────────────────────────────────────────────
# LLM config builders for SDSC endpoints
# ─────────────────────────────────────────────────────────────

MODEL_ENDPOINTS = {
    "qwen3-4B": {
        "base_url": "https://qwen3-4b-instruct.runai-mobiko-anisia.inference.compute.datascience.ch/v1",
        "api_key": "EMPTY",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
    },
    "qwen3-32B": {
        "base_url": "https://openwebui-runai-codev-llm.inference.compute.datascience.ch/api",
        "api_key": None,
        "model": "Qwen/Qwen3-32B-AWQ",
    },
    "qwen3-32B-vllm": {
        "base_url": "https://vllm-gateway-runai-codev-llm.inference.compute.datascience.ch/v1",
        "api_key": None,
        "model": "Qwen/Qwen3-32B-AWQ",
    },
    "gpt4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": None,
        "model": "gpt-4o",
    },
    "qwen3-35B-vllm": {
        "base_url": "https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1",
        "api_key": None,
        "model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    },
}


def build_llm_config(model_key: str, temperature: float = 0.3) -> LLMConfig:
    """
    Build an AG2 LLMConfig for one of the SDSC endpoints.
    AG2 wraps OpenAI-compatible APIs natively.
    """
    endpoint = MODEL_ENDPOINTS.get(model_key)
    if not endpoint:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_ENDPOINTS.keys())}")

    api_key = (
        endpoint["api_key"]
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_WEB_UI_API_KEY")
    )
    if not api_key:
        raise ValueError(f"API key required for {model_key}.")

    return LLMConfig(
        {
            "model": endpoint["model"],
            "base_url": endpoint["base_url"],
            "api_key": api_key,
            "api_type": "openai",
        },
        temperature=temperature,
    )


# ─────────────────────────────────────────────────────────────
# Schema / seeds / guideline loaders  (reused from vanilla)
# ─────────────────────────────────────────────────────────────

def load_schema(path: Path) -> Dict[str, List[List[str]]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf8") as f:
            return json.load(f)
    if suffix != ".py":
        raise ValueError(f"Unsupported schema format: {path}")
    text = path.read_text(encoding="utf8")
    tree = ast.parse(text, filename=str(path), mode="exec")
    obj = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.upper() in {"SCHEMA", "SEEDS"}:
                    obj = ast.literal_eval(node.value)
                    break
        if obj is not None:
            break
    if obj is None and len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        obj = ast.literal_eval(tree.body[0].value)
    if not isinstance(obj, dict):
        raise ValueError(f"Could not read dict from {path}.")
    return obj


load_seeds = load_schema  # same format


def load_decision_support(path: Path) -> List[Dict[str, str]]:
    """
    Parse Decision_support.csv — a table-based decision guide.
    Each row (LABEL, Question, Examples, Definition) becomes one section.
    Falls back to an empty list if the file is missing or unreadable.
    """
    try:
        if path.suffix.lower() == ".csv":
            import csv
            sections: List[Dict[str, str]] = []
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    label = (row.get("LABEL") or "").strip()
                    question = (row.get("Question") or "").strip()
                    examples = (row.get("Examples") or "").strip()
                    definition = (row.get("Definition") or "").strip()
                    if not label:
                        continue
                    content_parts = []
                    if question:
                        content_parts.append(f"Question: {question}")
                    if definition:
                        content_parts.append(f"Definition: {definition}")
                    if examples:
                        content_parts.append(f"Examples: {examples}")
                    sections.append({
                        "title": label,
                        "content": "\n".join(content_parts),
                        "source": "decision_support",
                    })
            return sections if sections else _get_embedded_guideline()
    except Exception:
        return _get_embedded_guideline()


# Section-boundary patterns for MoBiKo label guidance draft v2
_MOBIKO_V2_SECTION_STARTS = (
    re.compile(r"^Step \d"),
    re.compile(r"^[IVX]+\."),
    re.compile(r"^(Step 1: identifying spans|Step 2: Labelling spans|General rules|Handling difficult|Typical difficult|Rule|Needs further"
               r"|Species and Taxonomic|Ecological Attributes|Human research activities|System-Level|Polysemic terms|Polysemic Terms"
               r"|Typical Difficult Cases|Rule for Tiebreaker)", re.IGNORECASE),
)


def load_guideline_from_docx(path: Path) -> List[Dict[str, str]]:
    """
    Parse MoBiKo label guidance draft v2.docx — a narrative guideline with sections.
    Each recognised heading starts a new section.
    Falls back to the embedded guideline if python-docx is unavailable.
    """
    try:
        from docx import Document
        doc = Document(str(path))
        sections: List[Dict[str, str]] = []
        title, content = "Introduction", []

        def _is_section_boundary(text: str) -> bool:
            return any(pat.match(text) for pat in _MOBIKO_V2_SECTION_STARTS)

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Also treat bold short paragraphs as section headings
            is_heading = (
                _is_section_boundary(text)
                or para.style.name.startswith("Heading")
                or (len(text) < 80 and all(run.bold for run in para.runs if run.text.strip()))
            )
            if is_heading:
                if content:
                    sections.append({"title": title, "content": "\n".join(content),
                                     "source": "mobiko_v2"})
                title, content = text, []
            else:
                content.append(text)
        if content:
            sections.append({"title": title, "content": "\n".join(content),
                              "source": "mobiko_v2"})
        return sections if sections else _get_embedded_guideline()
    except ImportError:
        return _get_embedded_guideline()


def _get_embedded_guideline() -> List[Dict[str, str]]:
    return [
        {"title": "General rules for ontological roles",
         "content": "a. PROCESS: 'Is this something that happens?' b. PROPERTY: 'Is this something that something has?' c. Adjectival process nouns → default ABIOTIC PROPERTY."},
        {"title": "Step 1 — Abstract/theoretical", "content": "Scientific/mathematical concept → CONCEPT"},
        {"title": "Step 2 — Temporal", "content": "Time unit/interval → TEMPORAL ENTITY. Temporal attribute → TEMPORAL PROPERTY."},
        {"title": "Step 3 — Spatial", "content": "Place/spatial unit → SPATIAL ENTITY. Spatial attribute → SPATIAL PROPERTY."},
        {"title": "Step 4 — Anthropogenic", "content": "Human-created thing → ANTHROPOGENIC ENTITY. Human activity → ANTHROPOGENIC PROCESS. Human system characteristic → ANTHROPOGENIC PROPERTY."},
        {"title": "Step 5 — Biotic", "content": "Single organism → BIOTIC ENTITY. Group → BIOTIC COLLECTIVE ENTITY. Biological activity → BIOTIC PROCESS. Attribute → BIOTIC PROPERTY."},
        {"title": "Step 6 — Abiotic", "content": "Physical thing → ABIOTIC ENTITY. Environmental process → ABIOTIC PROCESS. Environmental attribute → ABIOTIC PROPERTY. Aggregated system → ABIOTIC COLLECTIVE ENTITY."},
        {"title": "Ambiguous cases", "content": "Look at the modified noun: population density → BIOTIC PROPERTY, soil density → ABIOTIC PROPERTY. Habitat → SPATIAL ENTITY, habitat quality → SPATIAL PROPERTY. Taxonomic names → BIOTIC COLLECTIVE ENTITY. Tiebreaker: choose the primary referent."},
    ]


# ─────────────────────────────────────────────────────────────
# Tool functions  (registered with AG2 agents)
# ─────────────────────────────────────────────────────────────
# AG2 tools are plain functions with Annotated type hints.
# We use module-level state set by the orchestrator before running.

_SCHEMA: Dict[str, List[List[str]]] = {}
_TYPE_PAIR_TO_RELATIONS: Dict[Tuple[str, str], List[str]] = {}
_ALL_ENTITY_TYPES: set = set()
_GUIDELINE_SECTIONS: List[Dict[str, str]] = []
_SEED_EXAMPLES: Dict[str, List[dict]] = {}
_GUIDELINE_SEARCH_BACKEND = "lexical"
_DEFAULT_GUIDELINE_SEARCH_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME = _DEFAULT_GUIDELINE_SEARCH_EMBEDDING_MODEL
_GUIDELINE_SECTION_EMBEDDINGS: Optional[List[List[float]]] = None
_GUIDELINE_EMBEDDING_MODEL = None
_GUIDELINE_EMBEDDING_MODEL_LOADED_NAME: Optional[str] = None
_GUIDELINE_EMBEDDING_ERROR: Optional[str] = None

_GUIDELINE_SEARCH_NO_MATCH_SUGGESTION = "This concept may not be covered in the guideline"
_GUIDELINE_EMBEDDING_MIN_SCORE = 0.25
_CONSISTENCY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "with",
}
_CONSISTENCY_MIN_TOKEN_OVERLAP = 0.5


_FALLBACK_ENTITY_TYPES = [
    "ABIOTIC ENTITY", "ABIOTIC PROCESS", "ABIOTIC PROPERTY",
    "ANTHROPOGENIC ENTITY", "ANTHROPOGENIC PROCESS", "ANTHROPOGENIC PROPERTY",
    "BIOTIC ENTITY", "BIOTIC PROCESS", "BIOTIC PROPERTY",
    "CONCEPT", "SPATIAL ENTITY", "SPATIAL PROPERTY", "TEMPORAL ENTITY", "TEMPORAL PROPERTY",
    "QUALITATIVE ENTITY", "QUANTITATIVE ENTITY"
]


def _init_tool_state(schema, guideline_sections, seed_examples,
                     entity_types_list: Optional[list] = None,
                     guideline_search_backend: Optional[str] = None,
                     guideline_search_embedding_model: Optional[str] = None):
    """Populate module-level state used by tool functions."""
    global _SCHEMA, _TYPE_PAIR_TO_RELATIONS, _ALL_ENTITY_TYPES
    global _GUIDELINE_SECTIONS, _SEED_EXAMPLES
    global _GUIDELINE_SEARCH_BACKEND, _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME
    global _GUIDELINE_SECTION_EMBEDDINGS, _GUIDELINE_EMBEDDING_ERROR

    _SCHEMA = schema
    _GUIDELINE_SECTIONS = guideline_sections
    _SEED_EXAMPLES = seed_examples
    _GUIDELINE_SEARCH_BACKEND = (
        guideline_search_backend
        or os.getenv("GUIDELINE_SEARCH_BACKEND")
        or "lexical"
    ).strip().lower()
    if _GUIDELINE_SEARCH_BACKEND not in {"lexical", "embedding"}:
        logger.warning(
            "Unknown GUIDELINE_SEARCH_BACKEND=%r; falling back to lexical",
            _GUIDELINE_SEARCH_BACKEND,
        )
        _GUIDELINE_SEARCH_BACKEND = "lexical"
    _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME = (
        guideline_search_embedding_model
        or os.getenv("GUIDELINE_SEARCH_EMBEDDING_MODEL")
        or _DEFAULT_GUIDELINE_SEARCH_EMBEDDING_MODEL
    )
    _GUIDELINE_SECTION_EMBEDDINGS = None
    _GUIDELINE_EMBEDDING_ERROR = None

    _TYPE_PAIR_TO_RELATIONS.clear()
    _ALL_ENTITY_TYPES.clear()
    for rel, pairs in schema.items():
        for t1, t2 in pairs:
            _TYPE_PAIR_TO_RELATIONS.setdefault((t1, t2), []).append(rel)
            _ALL_ENTITY_TYPES.add(t1)
            _ALL_ENTITY_TYPES.add(t2)

    # Override inferred types with the authoritative list if provided
    if entity_types_list:
        _ALL_ENTITY_TYPES.clear()
        _ALL_ENTITY_TYPES.update(entity_types_list)


def schema_lookup(
    e1_type: Annotated[str, "First entity type, e.g. 'BIOTIC ENTITY'"],
    e2_type: Annotated[str, "Second entity type, e.g. 'BIOTIC PROPERTY'"],
) -> str:
    """Check which relations are valid between two entity types in the MoBiKo schema."""
    e1 = e1_type.strip().upper()
    e2 = e2_type.strip().upper()
    fwd = _TYPE_PAIR_TO_RELATIONS.get((e1, e2), [])
    rev = _TYPE_PAIR_TO_RELATIONS.get((e2, e1), [])
    return json.dumps({
        "e1_type": e1, "e2_type": e2,
        "valid_relations_forward": fwd,
        "valid_relations_reverse": rev,
        "e1_known": e1 in _ALL_ENTITY_TYPES,
        "e2_known": e2 in _ALL_ENTITY_TYPES,
    })


def guideline_search(
    query: Annotated[str, "Keywords to search in the labelling guideline, e.g. 'habitat spatial property'"],
) -> str:
    """Search the MoBiKo labelling guideline for relevant rules and classification steps."""
    backend = _GUIDELINE_SEARCH_BACKEND
    if backend == "embedding":
        try:
            results = _guideline_search_embedding(query)
            if results is not None:
                return _format_guideline_search_response(query, "embedding", results)
        except Exception as exc:
            logger.warning("Embedding guideline_search failed; falling back to lexical: %s", exc)

    results = _guideline_search_lexical(query)
    return _format_guideline_search_response(query, "lexical", results)


def _format_guideline_search_response(
    query: str,
    backend: str,
    results: List[Dict[str, Any]],
) -> str:
    status = "matched" if results else "no_match"
    return json.dumps({
        "status": status,
        "backend": backend,
        "query": query,
        "results": results,
        "suggestion": None if results else _GUIDELINE_SEARCH_NO_MATCH_SUGGESTION,
    }, ensure_ascii=False)


def _guideline_search_lexical(query: str) -> List[Dict[str, Any]]:
    tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for sec in _GUIDELINE_SECTIONS:
        title_tokens = set(re.findall(r"\w+", sec["title"].lower()))
        content_tokens = set(re.findall(r"\w+", sec["content"].lower()))
        score = len(tokens & title_tokens) * 3 + len(tokens & content_tokens)
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:3]]


def _guideline_search_embedding(query: str) -> Optional[List[Dict[str, Any]]]:
    section_embeddings = _ensure_guideline_section_embeddings()
    if section_embeddings is None:
        return None

    query_embedding = _embed_guideline_texts([query])[0]
    scored = []
    for sec, sec_embedding in zip(_GUIDELINE_SECTIONS, section_embeddings):
        score = _cosine_similarity(query_embedding, sec_embedding)
        if score >= _GUIDELINE_EMBEDDING_MIN_SCORE:
            scored.append((score, sec))
    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:3]]


def _ensure_guideline_section_embeddings() -> Optional[List[List[float]]]:
    global _GUIDELINE_SECTION_EMBEDDINGS, _GUIDELINE_EMBEDDING_ERROR

    if _GUIDELINE_SECTION_EMBEDDINGS is not None:
        return _GUIDELINE_SECTION_EMBEDDINGS
    if not _GUIDELINE_SECTIONS:
        _GUIDELINE_SECTION_EMBEDDINGS = []
        return _GUIDELINE_SECTION_EMBEDDINGS
    if _GUIDELINE_EMBEDDING_ERROR:
        return None

    try:
        section_texts = [
            f"{sec.get('title', '')}\n{sec.get('content', '')}"
            for sec in _GUIDELINE_SECTIONS
        ]
        _GUIDELINE_SECTION_EMBEDDINGS = _embed_guideline_texts(section_texts)
        return _GUIDELINE_SECTION_EMBEDDINGS
    except Exception as exc:
        _GUIDELINE_EMBEDDING_ERROR = str(exc)
        logger.warning("Could not initialize guideline embeddings: %s", exc)
        return None


def _embed_guideline_texts(texts: List[str]) -> List[List[float]]:
    model = _get_guideline_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [_as_float_vector(vec) for vec in vectors]


def _get_guideline_embedding_model():
    global _GUIDELINE_EMBEDDING_MODEL, _GUIDELINE_EMBEDDING_MODEL_LOADED_NAME

    if (
        _GUIDELINE_EMBEDDING_MODEL is not None
        and _GUIDELINE_EMBEDDING_MODEL_LOADED_NAME == _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME
    ):
        return _GUIDELINE_EMBEDDING_MODEL

    from sentence_transformers import SentenceTransformer

    _GUIDELINE_EMBEDDING_MODEL = SentenceTransformer(_GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME)
    _GUIDELINE_EMBEDDING_MODEL_LOADED_NAME = _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME
    return _GUIDELINE_EMBEDDING_MODEL


def _as_float_vector(vec: Any) -> List[float]:
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _consistency_tokens(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [tok for tok in tokens if tok not in _CONSISTENCY_STOPWORDS]


def _consistency_match(query_tokens: List[str], candidate_text: str) -> Optional[Tuple[float, List[str]]]:
    candidate_tokens = _consistency_tokens(candidate_text)
    if not query_tokens or not candidate_tokens:
        return None

    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    overlap = query_set & candidate_set
    if not overlap:
        return None

    if len(query_set) == 1:
        if len(candidate_set) == 1 and query_set == candidate_set:
            return 1.0, sorted(overlap)
        return None

    score = len(overlap) / max(len(query_set), len(candidate_set))
    if score < _CONSISTENCY_MIN_TOKEN_OVERLAP:
        return None
    return score, sorted(overlap)


def consistency_check(
    span_text: Annotated[str, "The entity text to look up, e.g. 'species'"],
) -> str:
    """Find how similar spans were labelled in existing seed examples for consistency."""
    query_tokens = _consistency_tokens(span_text)
    matches = []
    for rel, examples in _SEED_EXAMPLES.items():
        for ex in examples:
            e1 = ex.get("e1", {})
            e2 = ex.get("e2", {})
            matched = None
            e1_match = _consistency_match(query_tokens, e1.get("text", ""))
            if e1_match:
                matched = ("e1", e1, e1_match)
            else:
                e2_match = _consistency_match(query_tokens, e2.get("text", ""))
                if e2_match:
                    matched = ("e2", e2, e2_match)
            if matched:
                role, ent, (score, matched_tokens) = matched
                matches.append({
                    "relation": rel, "matched_entity": role,
                    "entity_text": ent["text"], "entity_type": ent["type"],
                    "match_score": round(score, 3),
                    "query_tokens": query_tokens,
                    "matched_tokens": matched_tokens,
                    "sentence_snippet": ex["sentence"][:120],
                })
            if len(matches) >= 5:
                return json.dumps(matches, ensure_ascii=False)
    return json.dumps(matches, ensure_ascii=False)


def list_entity_types() -> str:
    """List all valid entity types defined in the MoBiKo schema."""
    return json.dumps(sorted(_ALL_ENTITY_TYPES))


# ─────────────────────────────────────────────────────────────
# Agent system prompts
# ─────────────────────────────────────────────────────────────

def _build_guideline_summary(sections: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"### {s['title']}\n{s['content']}" for s in sections)


def _annotator_system_msg(guideline: str, entity_schema: str, relation_schema: dict) -> str:
    return f"""\
You are Annotator, a biodiversity NLP expert. Your primary objective is MAXIMUM COVERAGE: identify \
and annotate every possible entity and every valid relation (triplet) in the given sentence. \
It is far better to over-annotate than to miss entities or relations — the Critic will filter errors later.

## Entity Type Schema
{entity_schema}

## Relation Schema
{relation_schema}

## Guideline Summary
{guideline}

## Available Tools
- list_entity_types   : retrieve the full list of valid entity types
- schema_lookup       : check which relations are valid for a pair of entity types
- guideline_search    : search the labelling guideline when a classification is unclear

## Process
1. Read the sentence carefully and identify ALL meaningful spans — err on the side of inclusion.
2. For each candidate span, call list_entity_types to confirm the type exists, then assign the best type \
   using Steps 1-6 (domain + ontological role) from the guideline.
3. For every pair of annotated entities, call schema_lookup to find valid relations. \
   Annotate ALL valid triplets (e1, relation, e2).
4. Call guideline_search when a classification is ambiguous.

## Coverage Rules
- Prefer more entities over fewer: if a span could plausibly be an entity, include it.
- Propose ALL relations schema_lookup returns as valid for a given entity-type pair.
- List ambiguous spans in "uncertain_cases" rather than dropping them.

## Output
Return a JSON object with exactly these fields:
{{
  "entities": [
    {{"text": "species richness", "entity_type": "BIOTIC PROPERTY", "guideline_step": "Step 5", "confidence": 0.9, "reasoning": "attribute of biotic entity"}}
  ],
  "relations": [
    {{"relation": "HAS_PROPERTY", "e1_text": "birds", "e1_type": "BIOTIC ENTITY", "e2_text": "species richness", "e2_type": "BIOTIC PROPERTY", "confidence": 0.85, "reasoning": "..."}}
  ],
  "uncertain_cases": ["optional span text and short explanation if ambiguous"],
  "reasoning": "brief overall reasoning"
}}

Output rules:
- Return JSON only, then end your message with TERMINATE on its own line.
- Do not include commentary, markdown, or <think> blocks.
- Keep every reasoning field brief and evidence-based.
- Every uncertain_cases item must be a complete JSON string. Put any explanation inside the quotes.
"""


def _critic_system_msg(guideline: str, entity_schema: str, relation_schema: dict) -> str:
    return f"""\
You are Critic, a rigorous QA reviewer for biodiversity annotations. \
Your objective is precision: scrutinise every label the Annotator proposes, \
challenge anything that is incorrect or ambiguous, and surface anything that was missed. \
Disagreement is expected and productive — correctness matters more than consensus.

## Entity Type Schema
{entity_schema}

## Guideline Summary
{guideline}

## Relation Schema
{relation_schema}

## Available Tools
- guideline_search   : retrieve the exact guideline rule that applies to a disputed span
- schema_lookup      : verify that a relation is valid for a given entity-type pair
- consistency_check  : compare a span against seed examples to detect labelling inconsistencies
- list_entity_types  : confirm entity type names

## Review Process
Start by checking any items the Annotator flagged as low-confidence (< {LOW_CONFIDENCE_THRESHOLD}) \
— these are the most likely to contain errors and deserve the closest scrutiny. \
Then work through the remaining annotation systematically in this order:

1. **Guideline violations** — for each entity label, call guideline_search with the span text \
   and its proposed type. Check whether the guideline’s step-by-step decision tree supports \
   the chosen category. Flag any label that contradicts the guideline rules.

2. **Category confusions** — look for common misclassifications:
   - BIOTIC PROPERTY vs ABIOTIC PROPERTY (check the modified noun, not the adjective)
   - SPATIAL ENTITY vs ABIOTIC ENTITY (place/unit of analysis vs physical object)
   - CONCEPT vs any concrete category (abstract theoretical construct vs real-world referent)
   - BIOTIC PROCESS vs ANTHROPOGENIC PROCESS (organism-driven vs human-driven activity)
   For each suspected confusion, call guideline_search to cite the relevant rule.

3. **Edge cases** — spans that sit on a categorical boundary. Use consistency_check to see \
   how similar spans were labelled in seed examples. If the seed label differs, flag it.

4. **Relation validity** — for every proposed triplet, call schema_lookup to confirm the \
   relation is valid for that entity-type pair. Flag invalid or missing relations.

5. **Missing spans** — re-read the original sentence. Identify any entity spans the \
   Annotator overlooked. For each, state the span text, the correct entity type, and cite \
   the guideline step that supports it.

After any tool calls return, you MUST produce the final review JSON. Do not stop after tool
results or ask for another turn.

## Output
Return a JSON object with exactly these fields:
{{
  "agreements": [{{"target": "span text", "label": "ENTITY_TYPE or RELATION"}}],
  "disagreements": [
    {{"target": "span text", "annotator_label": "WRONG_TYPE", "proposed_label": "CORRECT_TYPE", "guideline_reference": "Step 5", "severity": "major", "explanation": "reason"}}
  ],
  "missing_annotations": [
    {{"text": "missed span", "entity_type": "BIOTIC ENTITY", "guideline_step": "Step 5", "reasoning": "reason it should be annotated"}}
  ],
  "reasoning": "brief overall reasoning"
}}

Output rules:
- Return JSON only, then end your message with TERMINATE on its own line.
- Do not restate the sentence or the full annotation.
- Limit each disagreement to the minimal concrete correction needed.
"""


def _adjudicator_system_msg(guideline: str, entity_schema: str, relation_schema: dict) -> str:
    return f"""\
You are Adjudicator, the final decision-maker for biodiversity annotations.
You see the Annotator's labels and the Critic's review.

## Entity Type Schema
{entity_schema}

## Relation schema:
{relation_schema}

## Guideline Summary
{guideline}

## Decision Rules
1. Agreement between Annotator and Critic -> accept unchanged (high confidence).
2. You may only change Annotator labels that appear in the Critic's final
   "disagreements" list, or add spans that appear in "missing_annotations".
3. If the Critic did not dispute a span or relation, keep the Annotator's
   label exactly. Do not independently re-annotate accepted items.
4. Disagreement -> check guideline via tools, apply tiebreaker:
   "choose the category describing the primary referent in the sentence."
5. Genuine ambiguity -> flag for human review, pick the safer label.
6. Always copy Annotator "uncertain_cases" into "flagged_for_human_review".
7. If a Critic disagreement has severity "critical" and no clear
   guideline_reference, include that target in "flagged_for_human_review".

## Output

Return a JSON object with exactly these fields, then end your message with TERMINATE on its own line:
{{
  "final_entities": [
    {{"text": "species richness", "entity_type": "BIOTIC PROPERTY", "confidence": 0.9, "reasoning": "..."}}
  ],
  "final_relations": [
    {{"relation": "HAS_PROPERTY", "e1_text": "birds", "e1_type": "BIOTIC ENTITY", "e2_text": "species richness", "e2_type": "BIOTIC PROPERTY", "confidence": 0.9, "reasoning": "..."}}
  ],
  "disagreement_resolutions": [
    {{"issue": "span was labelled X", "decision": "correct label is Y", "rationale": "guideline step Z says..."}}
  ],
  "flagged_for_human_review": ["optional span text if genuinely ambiguous"]
}}

You must return this JSON right before the end of your message, and your message must end with "TERMINATE" on its own line.

Output rules:
- Return JSON only, then TERMINATE.
- Do not reproduce the prior transcript.

"""


# ─────────────────────────────────────────────────────────────
# Tool registration helper
# ─────────────────────────────────────────────────────────────

# Tools focused on proposing comprehensive annotations (entity + relation coverage)
ANNOTATOR_TOOL_FUNCTIONS = [
    (list_entity_types, "List all valid entity types from the schema."),
    (schema_lookup, "Check which relations are valid between two entity types."),
    (guideline_search, "Search the labelling guideline for relevant rules."),
]

# Tools focused on verifying and quality-checking annotations
CRITIC_TOOL_FUNCTIONS = [
    (schema_lookup, "Check which relations are valid between two entity types."),
    (guideline_search, "Search the labelling guideline for relevant rules."),
    (consistency_check, "Find how similar spans were labelled in seed examples."),
    (list_entity_types, "List all valid entity types from the schema."),
]


def _register_tools_on_agents(
    caller: ConversableAgent,
    executor: ConversableAgent,
    tool_functions: list,
):
    """Register the given tool functions: caller proposes, executor runs."""
    for func, desc in tool_functions:
        register_function(
            func,
            caller=caller,
            executor=executor,
            name=func.__name__,
            description=desc,
        )


# ─────────────────────────────────────────────────────────────
# Tool-call extraction from AG2 chat history
# ─────────────────────────────────────────────────────────────

def _extract_tool_calls_from_msg(msg: dict) -> List[Dict[str, Any]]:
    """
    Extract structured tool-call records from a single AG2 chat-history
    message.

    AG2 represents tool calls in two message types:
      1. The *proposal* from the LLM agent:
         {"role": "assistant", "tool_calls": [{"id": ..., "function": {"name": ..., "arguments": ...}}]}
      2. The *result* from the executor:
         {"role": "tool", "tool_call_id": ..., "content": "..."}

    This function only processes type 1 (proposals).  Results are attached
    to their proposals by `_pair_tool_results` below.
    """
    raw_calls = msg.get("tool_calls") or []
    extracted: List[Dict[str, Any]] = []
    for tc in raw_calls:
        func = tc.get("function") or {}
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")
        # Try to parse arguments as JSON for cleaner logging
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (json.JSONDecodeError, TypeError):
            args = args_raw
        extracted.append({
            "tool_call_id": tc.get("id", ""),
            "tool_name": name,
            "arguments": args,
            "result": None,   # filled in by _pair_tool_results
        })
    return extracted


def _pair_tool_results(
    chat_history: List[dict],
) -> Dict[str, str]:
    """
    Walk an AG2 chat_history and build a mapping
    tool_call_id → result_content for every tool-result message.
    """
    results: Dict[str, str] = {}
    for msg in chat_history:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                results[tc_id] = msg.get("content", "")
    return results


def _collect_messages_with_tools(
    chat_history: List[dict],
    skip_first: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert an AG2 chat_history into our record format, attaching
    tool_calls (with results) to the agent message that proposed them.

    Skips pure tool-result messages (role="tool") since their content is
    folded into the proposing message's tool_calls[].result field.
    """
    # First pass: collect all tool results keyed by call ID
    result_map = _pair_tool_results(chat_history)

    messages: List[Dict[str, Any]] = []
    for i, msg in enumerate(chat_history):
        if skip_first and i == 0:
            continue
        # Skip bare tool-result messages — they'll be paired with proposals
        if msg.get("role") == "tool":
            continue

        agent = msg.get("name", msg.get("role", "unknown"))
        content = msg.get("content", "") or ""

        # Extract and enrich tool calls
        tool_calls = _extract_tool_calls_from_msg(msg)
        for tc in tool_calls:
            tc_id = tc.get("tool_call_id", "")
            if tc_id in result_map:
                tc["result"] = result_map[tc_id]

        messages.append({
            "agent": agent,
            "content": content,
            "tool_calls": tool_calls,   # empty list if no tool calls
        })

    return messages


# ─────────────────────────────────────────────────────────────
# Termination helpers
# ─────────────────────────────────────────────────────────────

def _is_final_terminate_msg(msg: dict) -> bool:
    """True only when TERMINATE is the final standalone line."""
    content = msg.get("content", "") or ""
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    return bool(lines and lines[-1] == "TERMINATE")


def _critic_is_satisfied(msg: dict) -> bool:
    """
    Return True when the Critic's message signals no remaining issues.

    Checks for explicit TERMINATE first, then falls back to parsing the
    Critic's JSON: if both 'disagreements' and 'missing_annotations' are
    empty lists, the conversation can stop without the LLM saying TERMINATE.
    """
    content = msg.get("content", "") or ""
    if _is_final_terminate_msg(msg):
        return True
    # Strip thinking blocks before trying to parse JSON
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            parsed = json.loads(match.group())
            no_disagreements = len(parsed.get("disagreements", ["placeholder"])) == 0
            no_missing = len(parsed.get("missing_annotations", ["placeholder"])) == 0
            if no_disagreements and no_missing:
                return True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return False


# ─────────────────────────────────────────────────────────────
# Default document paths (relative to this file)
# ─────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_DECISION_SUPPORT = _THIS_DIR / "Decision_support.csv"
_DEFAULT_GUIDELINE = _THIS_DIR / "MoBiKo label guidance draft v2.docx"


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Deliberation diff helpers
# ─────────────────────────────────────────────────────────────

def _diff_annotator_rounds(
    prev: AnnotatorOutput,
    curr: AnnotatorOutput,
) -> str:
    """
    Compare two successive AnnotatorOutput objects and return a concise
    natural-language summary of what changed (added / removed / re-typed
    entities and relations).

    Returns an empty string if nothing changed.
    """
    changes: List[str] = []

    # ── Entity diffs ──────────────────────────────────────────
    prev_ents = {(e.text.lower(), e.entity_type) for e in prev.entities}
    curr_ents = {(e.text.lower(), e.entity_type) for e in curr.entities}

    # Entities present in prev but not curr (removed or re-typed)
    prev_texts = {text for text, _ in prev_ents}
    curr_texts = {text for text, _ in curr_ents}

    added_texts = curr_texts - prev_texts
    removed_texts = prev_texts - curr_texts

    # Entities whose text stayed but type changed
    shared_texts = prev_texts & curr_texts
    prev_by_text = {text: typ for text, typ in prev_ents}
    curr_by_text = {text: typ for text, typ in curr_ents}
    for text in sorted(shared_texts):
        old_type = prev_by_text.get(text)
        new_type = curr_by_text.get(text)
        if old_type and new_type and old_type != new_type:
            changes.append(f'Re-typed "{text}": {old_type} → {new_type}')

    if added_texts:
        added_details = []
        for text in sorted(added_texts):
            typ = curr_by_text.get(text, "?")
            added_details.append(f'"{text}" ({typ})')
        changes.append(f"Added entities: {', '.join(added_details)}")

    if removed_texts:
        removed_details = []
        for text in sorted(removed_texts):
            typ = prev_by_text.get(text, "?")
            removed_details.append(f'"{text}" ({typ})')
        changes.append(f"Removed entities: {', '.join(removed_details)}")

    # ── Relation diffs ────────────────────────────────────────
    def _rel_key(r: RelationFlat) -> tuple:
        return (r.e1_text.lower(), r.relation, r.e2_text.lower())

    prev_rels = {_rel_key(r) for r in prev.relations}
    curr_rels = {_rel_key(r) for r in curr.relations}

    added_rels = curr_rels - prev_rels
    removed_rels = prev_rels - curr_rels

    if added_rels:
        changes.append(
            f"Added {len(added_rels)} relation(s): "
            + "; ".join(f"({e1}, {rel}, {e2})" for e1, rel, e2 in sorted(added_rels))
        )
    if removed_rels:
        changes.append(
            f"Removed {len(removed_rels)} relation(s): "
            + "; ".join(f"({e1}, {rel}, {e2})" for e1, rel, e2 in sorted(removed_rels))
        )

    return " | ".join(changes)


class MultiAgentAnnotator:
    """
    AG2-based multi-agent annotation system.

    Parameters
    ----------
    annotator_model, critic_model, adjudicator_model : str
        Keys into MODEL_ENDPOINTS.
    schema_path : Path
        Relation schema file.
    decision_support_path : Path
        Decision support .docx (table-based decision guide for the Annotator).
        Defaults to the copy in src/multi_agent_annotation/.
    guideline_path : Path
        MoBiKo labelling guideline .docx (narrative, for Critic & Adjudicator).
        Defaults to the copy in src/multi_agent_annotation/.
    seeds_path : Path
        Seed examples for consistency checking.
    max_rounds : int
        Max Annotator↔Critic turns before adjudication.
    guideline_search_backend : str
        "lexical" by default; set to "embedding" to opt into SentenceTransformer retrieval.
    """

    def __init__(
        self,
        annotator_model: str = "qwen3-32B-vllm",
        critic_model: str = "qwen3-32B-vllm",
        adjudicator_model: str = "qwen3-32B-vllm",
        schema_path: Optional[Path] = None,
        decision_support_path: Optional[Path] = None,
        guideline_path: Optional[Path] = None,
        seeds_path: Optional[Path] = None,
        max_rounds: int = 2,
        entity_schema_str: Optional[str] = None,
        entity_types_list: Optional[list] = None,
        guideline_search_backend: Optional[str] = None,
        guideline_search_embedding_model: Optional[str] = None,
    ):
        self.max_rounds = max_rounds

        # ── Load resources ───────────────────────────────────
        relation_schema = load_schema(schema_path) if schema_path else {}
        seeds = load_seeds(seeds_path) if seeds_path else {}

        # Decision support doc → Annotator system prompt (compact decision table)
        ds_path = decision_support_path or _DEFAULT_DECISION_SUPPORT
        decision_support_sections = (
            load_decision_support(ds_path)
            if ds_path and ds_path.exists()
            else _get_embedded_guideline()
        )

        # MoBiKo v2 narrative → Critic & Adjudicator system prompts (edge cases, tiebreaker)
        gl_path = guideline_path or _DEFAULT_GUIDELINE
        guidance_sections = (
            load_guideline_from_docx(gl_path)
            if gl_path and gl_path.exists()
            else _get_embedded_guideline()
        )

        # guideline_search tool searches across both documents combined
        all_sections = decision_support_sections + guidance_sections
        _init_tool_state(relation_schema, all_sections, seeds,
                         entity_types_list=entity_types_list or _FALLBACK_ENTITY_TYPES,
                         guideline_search_backend=guideline_search_backend,
                         guideline_search_embedding_model=guideline_search_embedding_model)

        logger.info(
            f"Loaded decision support: {len(decision_support_sections)} sections | "
            f"guidance: {len(guidance_sections)} sections"
        )

        annotator_guideline = _build_guideline_summary(decision_support_sections)
        critic_guideline = _build_guideline_summary(guidance_sections)

        # Build entity schema string for system prompts
        if entity_schema_str is None:
            entity_schema_str = "\n".join(
                f"- {t}" for t in (entity_types_list or _FALLBACK_ENTITY_TYPES)
            )

        # ── Build LLM configs ────────────────────────────────
        annotator_llm = build_llm_config(annotator_model, temperature=0.2)
        critic_llm = build_llm_config(critic_model, temperature=0.3)
        adjudicator_llm = build_llm_config(adjudicator_model, temperature=0.1)

        # ── Create agents ────────────────────────────────────
        self.annotator = ConversableAgent(
            name="Annotator",
            system_message=_annotator_system_msg(annotator_guideline, entity_schema_str, relation_schema),
            llm_config=annotator_llm,
            human_input_mode="NEVER",
            is_termination_msg=_is_final_terminate_msg,
        )

        self.critic = ConversableAgent(
            name="Critic",
            system_message=_critic_system_msg(critic_guideline, entity_schema_str, relation_schema),
            llm_config=critic_llm,
            human_input_mode="NEVER",
            is_termination_msg=_is_final_terminate_msg,
        )

        self.adjudicator = ConversableAgent(
            name="Adjudicator",
            system_message=_adjudicator_system_msg(critic_guideline, entity_schema_str, relation_schema),
            llm_config=adjudicator_llm,
            human_input_mode="NEVER",
            is_termination_msg=_is_final_terminate_msg,
        )

        # A tool executor proxy (no LLM, just runs tool calls)
        self.tool_executor = ConversableAgent(
            name="ToolExecutor",
            llm_config=False,
            human_input_mode="NEVER",
            is_termination_msg=_is_final_terminate_msg,
        )

        # ── Register tools ───────────────────────────────────
        # All agents use the dedicated ToolExecutor proxy so tool results come
        # back as role="tool" messages and never appear as conversation turns
        # attributed to the counterpart agent.
        # Suppress the ag2 "Function X is being overridden" warning that fires
        # when overlapping tool sets (guideline_search, schema_lookup, etc.) are
        # registered on the same executor for different callers — all
        # registrations point to the same function objects, so there is no real
        # override risk.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Function '.*' is being overridden",
                category=UserWarning,
            )
            _register_tools_on_agents(self.annotator, self.tool_executor, ANNOTATOR_TOOL_FUNCTIONS)
            _register_tools_on_agents(self.critic, self.tool_executor, CRITIC_TOOL_FUNCTIONS)
            _register_tools_on_agents(self.adjudicator, self.tool_executor, CRITIC_TOOL_FUNCTIONS)

    def annotate_sentence(
        self,
        sentence: str,
        pre_identified_entities: Optional[List[dict]] = None,
    ) -> DeliberationRecord:
        """
        Run the full 3-agent deliberation on one sentence.

        Flow:
        1. Critic sends the task to the Annotator (initiates chat).
           Annotator generates its annotation once as the first reply.
           Critic reviews; says TERMINATE immediately if no critical issues.
           Otherwise Annotator revises and Critic re-reviews, up to max_rounds.
        2. Adjudicator receives the full Annotator↔Critic transcript
           and produces the final labels.
        """
        record = DeliberationRecord(sentence=sentence)

        # ── Build task message ────────────────────────────────
        task_msg = f'Annotate this sentence:\n\n"{sentence}"'
        if pre_identified_entities:
            task_msg += (
                f"\n\nPre-identified entities (verify types, find relations):\n"
                f"{json.dumps(pre_identified_entities, ensure_ascii=False, indent=2)}"
            )

        # ── Phase 1: per-round Annotator → Critic deliberation ──
        # Each round runs two separate agent↔ToolExecutor chats so tool
        # results arrive as role="tool" messages and never appear as
        # conversation turns attributed to the counterpart agent.
        logger.info("Phase 1: Annotator ↔ Critic deliberation")

        deliberation_messages: List[Dict[str, Any]] = []
        last_annotator_out: Optional[AnnotatorOutput] = None
        last_critic_out: Optional[CriticOutput] = None
        last_annotator_text = ""
        last_critic_text = ""

        for round_idx in range(self.max_rounds):
            # ── Annotator turn ────────────────────────────────
            if round_idx == 0:
                ann_msg = task_msg
            else:
                ann_msg = self._build_annotator_revision_msg(
                    sentence, last_annotator_text, last_critic_text,
                    pre_identified_entities,
                )
            ann_content, ann_record = self._run_agent_turn(self.annotator, ann_msg)
            deliberation_messages.append(ann_record)
            last_annotator_text = ann_content

            parsed_ann = self._parse_annotator_output(ann_content)
            if parsed_ann is not None:
                last_annotator_out = parsed_ann

            # ── Critic turn ───────────────────────────────────
            crit_msg = self._build_critic_review_msg(sentence, ann_content)
            crit_content, crit_record = self._run_agent_turn(self.critic, crit_msg)
            deliberation_messages.append(crit_record)
            last_critic_text = crit_content

            parsed_crit = self._parse_critic_output(crit_content)
            if parsed_crit is not None:
                last_critic_out = parsed_crit

            if (
                last_critic_out is not None
                and not last_critic_out.disagreements
                and not last_critic_out.missing_annotations
            ):
                break

        record.rounds_used = round_idx + 1
        record.messages = deliberation_messages

        if last_annotator_out is None:
            repair_messages, repaired_text = self._repair_agent_json(
                requester=self.tool_executor,
                producer=self.annotator,
                output_kind="Annotator",
                original_text=last_annotator_text,
                sentence=sentence,
            )
            if repair_messages:
                deliberation_messages.extend(repair_messages)
                last_annotator_out = self._parse_annotator_output(repaired_text)

        if last_annotator_out is None:
            record.adjudication_status = "annotator_parse_failed"
            record.adjudication_audit = self._base_adjudication_audit(
                "Annotator output missing, malformed, or schema-incomplete after one repair attempt."
            )
            record.agreement_score = None
            logger.info(
                "Done: 0 entities, 0 relations, agreement=None "
                "(annotator_parse_failed)"
            )
            return record

        if last_critic_out is None:
            repair_messages, repaired_text = self._repair_agent_json(
                requester=self.tool_executor,
                producer=self.critic,
                output_kind="Critic",
                original_text=last_critic_text,
                sentence=sentence,
                reference_annotation=last_annotator_out.model_dump_json(),
            )
            if repair_messages:
                deliberation_messages.extend(repair_messages)
                last_critic_out = self._parse_critic_output(repaired_text)

        # ── Phase 2: Adjudicator resolves ─────────────────────
        logger.info("Phase 2: Adjudicator resolving")

        # Build a condensed summary of the full deliberation trajectory:
        # final annotation + final critique + per-round dispute history.
        adjudicator_msg = self._build_adjudicator_summary(
            sentence, deliberation_messages
        )

        adj_result = self.adjudicator.initiate_chat(
            recipient=self.tool_executor,
            message=adjudicator_msg,
            max_turns=self._adjudicator_max_turns(),
        )

        # Extract final output from adjudicator's last message
        adj_output = self._extract_last_json_with_keys(
            adj_result.chat_history, ADJUDICATOR_REQUIRED_KEYS
        )

        # Collect adjudicator messages (including tool calls to/from ToolExecutor)
        adj_messages = _collect_messages_with_tools(
            adj_result.chat_history, skip_first=True  # skip the task message we sent
        )
        record.messages.extend(adj_messages)

        # ── Parse and constrain final annotations ─────────────
        parsed_adj = None
        if adj_output:
            try:
                parsed_adj = AdjudicatorOutput.model_validate(adj_output)
            except ValidationError as e:
                logger.warning(f"AdjudicatorOutput validation error: {e}")

        if parsed_adj is None:
            raw_adjudicator = self._last_agent_content(adj_messages, "Adjudicator")
            retry_mode = self._adjudicator_retry_mode(raw_adjudicator)
            if retry_mode == "generate":
                retry_messages, retry_text = self._retry_adjudicator_generation(
                    sentence, adjudicator_msg
                )
                if retry_messages:
                    record.messages.extend(retry_messages)
                    retry_raw = self._try_parse_json_with_keys(
                        retry_text, ADJUDICATOR_REQUIRED_KEYS
                    )
                    if retry_raw:
                        try:
                            parsed_adj = AdjudicatorOutput.model_validate(retry_raw)
                        except ValidationError as e:
                            logger.warning(f"Retried AdjudicatorOutput validation error: {e}")
            else:
                repair_messages, repaired_text = self._repair_agent_json(
                    requester=self.tool_executor,
                    producer=self.adjudicator,
                    output_kind="Adjudicator",
                    original_text=raw_adjudicator,
                    sentence=sentence,
                    reference_annotation=adjudicator_msg,
                )
                if repair_messages:
                    record.messages.extend(repair_messages)
                    repaired_raw = self._try_parse_json_with_keys(
                        repaired_text, ADJUDICATOR_REQUIRED_KEYS
                    )
                    if repaired_raw:
                        try:
                            parsed_adj = AdjudicatorOutput.model_validate(repaired_raw)
                        except ValidationError as e:
                            logger.warning(f"Repaired AdjudicatorOutput validation error: {e}")

        constrained = self._constrain_adjudicator_output(
            last_annotator_out, last_critic_out, parsed_adj
        )
        record.final_entities = constrained.final_entities
        record.final_relations = [
            r.to_relation_annotation() for r in constrained.final_relations
        ]
        record.flagged_for_human_review = constrained.flagged_for_human_review
        record.adjudication_status = constrained.status
        record.adjudication_audit = constrained.audit

        # ── Compute agreement ─────────────────────────────────
        # Use the last round only (earlier disputes may have been resolved).
        # Denominator = Annotator's proposed items (entities + relations).
        # Disputes  = Critic's explicit disagreements + relations whose endpoints
        #             are disputed (entity-type changes cascade to their relations)
        #             + missing_annotations.
        # Score     = (proposed − disagreed) / (proposed + missing).

        if last_annotator_out and last_critic_out:
            entity_dispute_targets = {
                MultiAgentAnnotator._normalize_annotation_text(d.target)
                for d in last_critic_out.disagreements
                if d.target
            }
            n_proposed = len(last_annotator_out.entities) + len(last_annotator_out.relations)
            n_disagreed = len(last_critic_out.disagreements)
            # Count relations whose endpoints are disputed but weren't flagged directly.
            n_implicated_relations = sum(
                1 for r in last_annotator_out.relations
                if (
                    MultiAgentAnnotator._normalize_annotation_text(r.e1_text) in entity_dispute_targets
                    or MultiAgentAnnotator._normalize_annotation_text(r.e2_text) in entity_dispute_targets
                )
                and not any(
                    MultiAgentAnnotator._disagreement_matches_relation(d, r)
                    for d in last_critic_out.disagreements
                )
            )
            n_disagreed += n_implicated_relations
            n_missing = len(last_critic_out.missing_annotations)
            n_agreed = max(n_proposed - n_disagreed, 0)
            total_considered = n_proposed + n_missing
            record.agreement_score = (
                n_agreed / total_considered if total_considered > 0 else 1.0
            )
        elif last_critic_out:
            # No parseable annotator output — fall back to disagreement count
            n_agree = len(last_critic_out.agreements)
            n_disagree = len(last_critic_out.disagreements)
            n_missing = len(last_critic_out.missing_annotations)
            total = n_agree + n_disagree + n_missing
            record.agreement_score = n_agree / total if total > 0 else 1.0
        else:
            record.agreement_score = None

        logger.info(
            f"Done: {len(record.final_entities)} entities, "
            f"{len(record.final_relations)} relations, "
            f"agreement={record.agreement_score if record.agreement_score is None else f'{record.agreement_score:.2f}'}"
        )
        return record

    def annotate_batch(
        self,
        sentences: List[str],
        pre_entities: Optional[List[Optional[List[dict]]]] = None,
        output_path: Optional[Path] = None,
    ) -> List[DeliberationRecord]:
        """Annotate a batch of sentences with JSONL output."""
        records = []
        ents_list = pre_entities or [None] * len(sentences)

        for i, (sent, ents) in enumerate(zip(sentences, ents_list)):
            logger.info(f"\n{'#'*60}\n  Sentence {i+1}/{len(sentences)}\n{'#'*60}")
            logger.info(f"  {sent[:100]}...")

            # Clear agent chat histories between sentences
            self.annotator.reset()
            self.critic.reset()
            self.adjudicator.reset()
            self.tool_executor.reset()

            record = self.annotate_sentence(sent, ents)
            records.append(record)

            if output_path:
                self._append_jsonl(record, output_path)

        return records

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict]:
        # Remove thinking blocks
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Scan for all valid top-level JSON objects and return the last one.
        # Using raw_decode avoids the greedy-regex pitfall where multiple JSON
        # blocks in one string (e.g. a transcript) are merged into invalid JSON.
        decoder = json.JSONDecoder()
        last_obj = None
        i = 0
        while i < len(cleaned):
            if cleaned[i] == "{":
                try:
                    obj, end = decoder.raw_decode(cleaned, i)
                    if isinstance(obj, dict):
                        last_obj = obj
                    i = end
                    continue
                except json.JSONDecodeError:
                    pass
            i += 1
        return last_obj

    @staticmethod
    def _try_parse_json_with_keys(
        text: str,
        required_keys: set[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Return the last parseable JSON object containing all required top-level
        keys. This prevents a malformed outer agent response from being reduced
        to a nested entity/relation fragment.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        decoder = json.JSONDecoder()
        last_obj = None
        i = 0
        while i < len(cleaned):
            if cleaned[i] == "{":
                try:
                    obj, end = decoder.raw_decode(cleaned, i)
                    if isinstance(obj, dict) and required_keys.issubset(obj.keys()):
                        last_obj = obj
                    i = end
                    continue
                except json.JSONDecodeError:
                    pass
            i += 1
        return last_obj

    @staticmethod
    def _extract_last_json(chat_history: list) -> Optional[Dict]:
        """Walk backward through chat history to find the last JSON output."""
        for msg in reversed(chat_history):
            content = msg.get("content", "")
            if not content:
                continue
            parsed = MultiAgentAnnotator._try_parse_json(content)
            if parsed and ("final_entities" in parsed or "entities" in parsed):
                return parsed
        return None

    @staticmethod
    def _extract_last_json_with_keys(
        chat_history: list,
        required_keys: set[str],
    ) -> Optional[Dict[str, Any]]:
        """Walk backward through chat history to find schema-complete JSON."""
        for msg in reversed(chat_history):
            content = msg.get("content", "")
            if not content:
                continue
            parsed = MultiAgentAnnotator._try_parse_json_with_keys(
                content, required_keys
            )
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_critic_output(text: str) -> Optional[CriticOutput]:
        raw = MultiAgentAnnotator._try_parse_json_with_keys(text, CRITIC_REQUIRED_KEYS)
        if raw is None:
            return None
        try:
            return CriticOutput.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"CriticOutput validation failed: {e}")
            return None

    @staticmethod
    def _parse_adjudicator_output(text: str) -> Optional[AdjudicatorOutput]:
        raw = MultiAgentAnnotator._try_parse_json_with_keys(
            text, ADJUDICATOR_REQUIRED_KEYS
        )
        if raw is None:
            return None
        try:
            return AdjudicatorOutput.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"AdjudicatorOutput validation failed: {e}")
            return None

    @staticmethod
    def _parse_annotator_output(text: str) -> Optional[AnnotatorOutput]:
        raw = MultiAgentAnnotator._try_parse_json_with_keys(
            text, ANNOTATOR_REQUIRED_KEYS
        )
        if raw is None:
            return None
        try:
            return AnnotatorOutput.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"AnnotatorOutput validation failed: {e}")
            return None

    @staticmethod
    def _last_deliberation_outputs(
        deliberation_messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[AnnotatorOutput], Optional[CriticOutput]]:
        """Return the final parseable Annotator and Critic outputs."""
        last_annotator_out = None
        last_critic_out = None
        for m in reversed(deliberation_messages):
            if m["agent"] == "Annotator" and last_annotator_out is None:
                last_annotator_out = MultiAgentAnnotator._parse_annotator_output(
                    m.get("content", "")
                )
            elif m["agent"] == "Critic" and last_critic_out is None:
                last_critic_out = MultiAgentAnnotator._parse_critic_output(
                    m.get("content", "")
                )
            if last_annotator_out is not None and last_critic_out is not None:
                break
        return last_annotator_out, last_critic_out

    @staticmethod
    def _last_agent_content(
        messages: List[Dict[str, Any]],
        agent_name: str,
    ) -> str:
        for m in reversed(messages):
            if m.get("agent") == agent_name:
                return m.get("content", "") or ""
        return ""

    @staticmethod
    def _agent_turn_max_turns() -> int:
        """Max turns for a single agent↔ToolExecutor chat (tool batches + output)."""
        return 10

    @staticmethod
    def _strip_terminate(text: str) -> str:
        stripped = text.rstrip()
        if stripped.endswith("TERMINATE"):
            stripped = stripped[: -len("TERMINATE")].rstrip()
        return stripped

    @staticmethod
    def _build_annotator_revision_msg(
        sentence: str,
        prev_annotation_text: str,
        critic_feedback_text: str,
        pre_identified_entities: Optional[List[dict]] = None,
    ) -> str:
        clean_ann = MultiAgentAnnotator._strip_terminate(prev_annotation_text)
        clean_crit = MultiAgentAnnotator._strip_terminate(critic_feedback_text)
        msg = (
            f"Revise your annotation for this sentence to address the Critic's feedback.\n\n"
            f'Sentence: "{sentence}"\n\n'
            f"Your previous annotation:\n{clean_ann}\n\n"
            f"Critic's feedback:\n{clean_crit}"
        )
        if pre_identified_entities:
            msg += (
                f"\n\nPre-identified entities (verify types, find relations):\n"
                f"{json.dumps(pre_identified_entities, ensure_ascii=False, indent=2)}"
            )
        return msg

    @staticmethod
    def _build_critic_review_msg(sentence: str, annotator_text: str) -> str:
        clean = MultiAgentAnnotator._strip_terminate(annotator_text)
        msg = (
            f"Review this annotation for the sentence below.\n\n"
            f'Sentence: "{sentence}"\n\n'
            f"Annotation:\n{clean}"
        )
        parsed = MultiAgentAnnotator._parse_annotator_output(clean)
        if parsed:
            low_conf: List[str] = []
            for e in parsed.entities:
                if e.confidence is not None and e.confidence < LOW_CONFIDENCE_THRESHOLD:
                    low_conf.append(
                        f'  - entity "{e.text}" ({e.entity_type}, conf={e.confidence:.2f})'
                    )
            for r in parsed.relations:
                if r.confidence is not None and r.confidence < LOW_CONFIDENCE_THRESHOLD:
                    low_conf.append(
                        f'  - relation "{r.e1_text} {r.relation} {r.e2_text}"'
                        f" (conf={r.confidence:.2f})"
                    )
            if low_conf:
                msg += (
                    f"\n\nLow-confidence items (< {LOW_CONFIDENCE_THRESHOLD}) — check these first:\n"
                    + "\n".join(low_conf)
                )
        return msg

    def _run_agent_turn(
        self,
        agent: ConversableAgent,
        message: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run one agent turn against ToolExecutor.

        Resets both agents so each round starts with a clean history and tool
        results arrive as role="tool" messages, not as counterpart conversation
        turns. All tool calls from the turn are folded into a single record
        message keyed by agent name.

        Returns (last_content, record_message).
        """
        agent.reset()
        self.tool_executor.reset()
        chat = agent.initiate_chat(
            recipient=self.tool_executor,
            message=message,
            max_turns=self._agent_turn_max_turns(),
        )
        all_msgs = _collect_messages_with_tools(chat.chat_history, skip_first=True)

        agent_name = agent.name
        last_content = ""
        for msg in reversed(all_msgs):
            if msg["agent"] == agent_name and msg["content"].strip():
                last_content = msg["content"]
                break

        all_tool_calls: List[Dict[str, Any]] = []
        for msg in all_msgs:
            if msg["agent"] == agent_name:
                all_tool_calls.extend(msg.get("tool_calls", []))

        return last_content, {
            "agent": agent_name,
            "content": last_content,
            "tool_calls": all_tool_calls,
        }

    @staticmethod
    def _adjudicator_max_turns() -> int:
        """Initial adjudication needs room for one tool exchange and final JSON."""
        return 6

    @staticmethod
    def _adjudicator_retry_mode(raw_adjudicator: str) -> str:
        return "repair" if raw_adjudicator.strip() else "generate"

    @staticmethod
    def _adjudicator_generation_retry_prompt(
        sentence: str,
        adjudicator_summary: str,
    ) -> str:
        return f"""\
You did not return the required final adjudication JSON.
Produce the final annotation now using the Annotator/Critic context below.
Return JSON only, with no markdown and no commentary.
End your message with TERMINATE on its own line.

Required JSON fields:
- final_entities
- final_relations
- disagreement_resolutions
- flagged_for_human_review

Sentence:
"{sentence}"

Annotator/Critic context:
{adjudicator_summary}
"""

    @staticmethod
    def _json_repair_prompt(
        output_kind: str,
        original_text: str,
        sentence: str,
        reference_annotation: Optional[str] = None,
    ) -> str:
        schemas = {
            "Annotator": """
Return exactly this JSON shape:
{
  "entities": [],
  "relations": [],
  "uncertain_cases": [],
  "reasoning": ""
}
""",
            "Critic": """
Return exactly this JSON shape:
{
  "agreements": [{"target": "span text", "label": "TYPE"}],
  "disagreements": [],
  "missing_annotations": [],
  "reasoning": ""
}
""",
            "Adjudicator": """
Return exactly this JSON shape:
{
  "final_entities": [],
  "final_relations": [],
  "disagreement_resolutions": [],
  "flagged_for_human_review": []
}
""",
        }
        reference = (
            f"\nReference annotation/review context:\n{reference_annotation}\n"
            if reference_annotation
            else ""
        )
        return f"""\
Your previous {output_kind} output was not parseable as the required JSON schema.
Repair syntax and schema only. Do not add new reasoning, labels, spans, or relations.
Return JSON only, with no markdown and no commentary.

Sentence:
"{sentence}"
{reference}
Required schema:
{schemas[output_kind]}
Previous output to repair:
{original_text}
"""

    def _repair_agent_json(
        self,
        requester: ConversableAgent,
        producer: ConversableAgent,
        output_kind: str,
        original_text: str,
        sentence: str,
        reference_annotation: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Ask the producing agent for one JSON-only repair response."""
        prompt = self._json_repair_prompt(
            output_kind,
            original_text,
            sentence,
            reference_annotation=reference_annotation,
        )
        try:
            repair_result = requester.initiate_chat(
                recipient=producer,
                message=prompt,
                max_turns=2,
            )
        except Exception as e:
            logger.warning(f"{output_kind} JSON repair failed to run: {e}")
            return [], ""

        repair_messages = _collect_messages_with_tools(
            repair_result.chat_history, skip_first=True
        )
        repaired_text = self._last_agent_content(repair_messages, output_kind)
        return repair_messages, repaired_text

    def _retry_adjudicator_generation(
        self,
        sentence: str,
        adjudicator_summary: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Ask Adjudicator to generate final JSON when no output was produced."""
        prompt = self._adjudicator_generation_retry_prompt(
            sentence, adjudicator_summary
        )
        try:
            retry_result = self.tool_executor.initiate_chat(
                recipient=self.adjudicator,
                message=prompt,
                max_turns=2,
            )
        except Exception as e:
            logger.warning(f"Adjudicator generation retry failed to run: {e}")
            return [], ""

        retry_messages = _collect_messages_with_tools(
            retry_result.chat_history, skip_first=True
        )
        retry_text = self._last_agent_content(retry_messages, "Adjudicator")
        return retry_messages, retry_text

    @staticmethod
    def _base_adjudication_audit(warning: str) -> Dict[str, Any]:
        return {
            "preserved_consensus": [],
            "allowed_changes": [],
            "rejected_changes": [],
            "human_review_flags": [],
            "warnings": [warning],
        }

    @staticmethod
    def _has_clear_guideline_reference(guideline_reference: str) -> bool:
        ref = re.sub(r"\s+", " ", (guideline_reference or "").strip()).lower()
        placeholder_refs = {
            "",
            "n/a",
            "na",
            "none",
            "null",
            "unknown",
            "unclear",
            "not cited",
            "not provided",
            "not specified",
            "no citation",
            "no guideline citation",
        }
        return ref not in placeholder_refs

    @staticmethod
    def _human_review_flags(
        annotator: AnnotatorOutput,
        critic: Optional[CriticOutput],
        adjudicator: Optional[AdjudicatorOutput],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        flags: List[str] = []
        provenance: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_flag(text: str, source: str) -> None:
            flag = re.sub(r"\s+", " ", (text or "").strip())
            if not flag:
                return
            key = flag.lower()
            if key in seen:
                for item in provenance:
                    if item["flag"].lower() == key and source not in item["sources"]:
                        item["sources"].append(source)
                return
            seen.add(key)
            flags.append(flag)
            provenance.append({"flag": flag, "sources": [source]})

        for case in annotator.uncertain_cases:
            add_flag(case, "annotator_uncertain_case")

        for entity in annotator.entities:
            if entity.confidence is not None and entity.confidence < LOW_CONFIDENCE_THRESHOLD:
                add_flag(
                    f'"{entity.text}" ({entity.entity_type},'
                    f" confidence={entity.confidence:.2f})",
                    "low_confidence",
                )
        for rel in annotator.relations:
            if rel.confidence is not None and rel.confidence < LOW_CONFIDENCE_THRESHOLD:
                add_flag(
                    f'"{rel.e1_text} {rel.relation} {rel.e2_text}"'
                    f" (confidence={rel.confidence:.2f})",
                    "low_confidence",
                )

        if adjudicator is not None:
            for flag in adjudicator.flagged_for_human_review:
                add_flag(flag, "adjudicator_flag")

        if critic is not None:
            for disagreement in critic.disagreements:
                if disagreement.severity.strip().lower() != "critical":
                    continue
                if MultiAgentAnnotator._has_clear_guideline_reference(
                    disagreement.guideline_reference
                ):
                    continue
                flag = disagreement.target or disagreement.explanation
                if flag:
                    add_flag(flag, "critic_critical_no_guideline")

        return flags, provenance

    @staticmethod
    def _normalize_annotation_text(text: str) -> str:
        """Normalize spans and free-text Critic targets for exact matching."""
        text = (text or "").strip().strip("\"'`")
        return re.sub(r"\s+", " ", text).lower()

    @staticmethod
    def _relation_slot(rel: RelationFlat) -> Tuple[str, str]:
        """Relation identity slot for replacing relation/type decisions."""
        return (
            MultiAgentAnnotator._normalize_annotation_text(rel.e1_text),
            MultiAgentAnnotator._normalize_annotation_text(rel.e2_text),
        )

    @staticmethod
    def _relation_key(rel: RelationFlat) -> Tuple[str, str, str, str, str]:
        return (
            MultiAgentAnnotator._normalize_annotation_text(rel.relation),
            MultiAgentAnnotator._normalize_annotation_text(rel.e1_text),
            MultiAgentAnnotator._normalize_annotation_text(rel.e1_type),
            MultiAgentAnnotator._normalize_annotation_text(rel.e2_text),
            MultiAgentAnnotator._normalize_annotation_text(rel.e2_type),
        )

    @staticmethod
    def _target_matches_relation(target: str, rel: RelationFlat) -> bool:
        target_norm = MultiAgentAnnotator._normalize_annotation_text(target)
        if not target_norm:
            return False

        e1 = MultiAgentAnnotator._normalize_annotation_text(rel.e1_text)
        e2 = MultiAgentAnnotator._normalize_annotation_text(rel.e2_text)
        relation = MultiAgentAnnotator._normalize_annotation_text(rel.relation)
        descriptors = {
            f"{e1} {relation} {e2}",
            f"{e1} -> {relation} -> {e2}",
            f"{relation}: {e1} -> {e2}",
        }
        if target_norm in descriptors:
            return True
        return e1 in target_norm and e2 in target_norm and relation in target_norm

    @staticmethod
    def _disagreement_matches_relation(
        disagreement: CriticDisagreement,
        rel: RelationFlat,
    ) -> bool:
        target_norm = MultiAgentAnnotator._normalize_annotation_text(
            disagreement.target
        ).replace("→", "->")
        if not target_norm:
            return False

        e1 = MultiAgentAnnotator._normalize_annotation_text(rel.e1_text)
        e2 = MultiAgentAnnotator._normalize_annotation_text(rel.e2_text)
        relation = MultiAgentAnnotator._normalize_annotation_text(rel.relation)
        annotator_label = MultiAgentAnnotator._normalize_annotation_text(
            disagreement.annotator_label
        )
        proposed_label = MultiAgentAnnotator._normalize_annotation_text(
            disagreement.proposed_label
        )

        if MultiAgentAnnotator._target_matches_relation(disagreement.target, rel):
            return True

        endpoints_match = e1 in target_norm and e2 in target_norm
        label_matches_relation = annotator_label == relation
        invalid_relation = proposed_label == "invalid" and label_matches_relation
        return endpoints_match and (label_matches_relation or invalid_relation)

    @staticmethod
    def _sync_relation_endpoint_types(
        rel: RelationFlat,
        final_entity_type_by_text: Dict[str, str],
    ) -> RelationFlat:
        updates: Dict[str, str] = {}
        e1_key = MultiAgentAnnotator._normalize_annotation_text(rel.e1_text)
        e2_key = MultiAgentAnnotator._normalize_annotation_text(rel.e2_text)
        if e1_key in final_entity_type_by_text:
            updates["e1_type"] = final_entity_type_by_text[e1_key]
        if e2_key in final_entity_type_by_text:
            updates["e2_type"] = final_entity_type_by_text[e2_key]
        return rel.model_copy(update=updates) if updates else rel

    @staticmethod
    def _constrain_adjudicator_output(
        annotator: AnnotatorOutput,
        critic: Optional[CriticOutput],
        adjudicator: Optional[AdjudicatorOutput],
    ) -> ConstrainedAdjudication:
        """
        Enforce the adjudicator's authority boundary:
        keep Annotator/Critic consensus, resolve only final Critic disputes,
        and add only final Critic missing annotations.
        """
        audit: Dict[str, Any] = {
            "preserved_consensus": [],
            "allowed_changes": [],
            "rejected_changes": [],
            "human_review_flags": [],
            "warnings": [],
        }
        human_review_flags, human_review_provenance = (
            MultiAgentAnnotator._human_review_flags(annotator, critic, adjudicator)
        )
        audit["human_review_flags"] = human_review_provenance

        if critic is None:
            audit["warnings"].append(
                "Critic output missing or unparseable; kept final Annotator output unchanged."
            )
            return ConstrainedAdjudication(
                final_entities=annotator.entities,
                final_relations=annotator.relations,
                flagged_for_human_review=human_review_flags,
                status="critic_missing_fallback",
                audit=audit,
            )

        if adjudicator is None:
            audit["warnings"].append(
                "Adjudicator output missing or unparseable; kept final Annotator output unchanged."
            )
            return ConstrainedAdjudication(
                final_entities=annotator.entities,
                final_relations=annotator.relations,
                flagged_for_human_review=human_review_flags,
                status="adjudicator_parse_failed",
                audit=audit,
            )

        disagreement_targets = {
            MultiAgentAnnotator._normalize_annotation_text(d.target)
            for d in critic.disagreements
            if d.target
        }
        missing_by_text = {
            MultiAgentAnnotator._normalize_annotation_text(m.text): m
            for m in critic.missing_annotations
            if m.text
        }

        annotator_entities_by_text = {
            MultiAgentAnnotator._normalize_annotation_text(e.text): e
            for e in annotator.entities
        }
        entity_order = [
            MultiAgentAnnotator._normalize_annotation_text(e.text)
            for e in annotator.entities
        ]
        final_entities_by_text = dict(annotator_entities_by_text)

        for key, entity in annotator_entities_by_text.items():
            if key not in disagreement_targets:
                audit["preserved_consensus"].append(
                    f'entity "{entity.text}": {entity.entity_type}'
                )

        for adj_entity in adjudicator.final_entities:
            key = MultiAgentAnnotator._normalize_annotation_text(adj_entity.text)
            original = annotator_entities_by_text.get(key)

            if original is not None:
                if key in disagreement_targets:
                    final_entities_by_text[key] = adj_entity
                    if original.entity_type != adj_entity.entity_type:
                        audit["allowed_changes"].append(
                            f'entity "{adj_entity.text}": '
                            f"{original.entity_type} -> {adj_entity.entity_type}"
                        )
                    continue

                if original.entity_type != adj_entity.entity_type:
                    audit["rejected_changes"].append(
                        f'entity "{adj_entity.text}": '
                        f"{original.entity_type} -> {adj_entity.entity_type} "
                        "(not in final Critic disagreements)"
                    )
                continue

            missing = missing_by_text.get(key)
            if missing is not None and adj_entity.entity_type == missing.entity_type:
                final_entities_by_text[key] = adj_entity
                entity_order.append(key)
                audit["allowed_changes"].append(
                    f'entity "{adj_entity.text}" added as {adj_entity.entity_type} '
                    "(final Critic missing annotation)"
                )
            elif missing is not None:
                audit["rejected_changes"].append(
                    f'entity "{adj_entity.text}" add as {adj_entity.entity_type} '
                    f"rejected; final Critic proposed {missing.entity_type}"
                )
            else:
                audit["rejected_changes"].append(
                    f'entity "{adj_entity.text}" add as {adj_entity.entity_type} '
                    "(not in final Critic missing annotations)"
                )

        final_entities = [final_entities_by_text[key] for key in entity_order]
        final_entity_type_by_text = {
            MultiAgentAnnotator._normalize_annotation_text(e.text): e.entity_type
            for e in final_entities
        }

        annotator_relations_by_slot = {
            MultiAgentAnnotator._relation_slot(r): r
            for r in annotator.relations
        }
        disputed_relation_slots = {
            MultiAgentAnnotator._relation_slot(r)
            for r in annotator.relations
            if any(
                MultiAgentAnnotator._disagreement_matches_relation(d, r)
                for d in critic.disagreements
            ) or (
                MultiAgentAnnotator._normalize_annotation_text(r.e1_text) in disagreement_targets
                or MultiAgentAnnotator._normalize_annotation_text(r.e2_text) in disagreement_targets
            )
        }

        relation_order = [
            MultiAgentAnnotator._relation_slot(r)
            for r in annotator.relations
            if MultiAgentAnnotator._relation_slot(r) not in disputed_relation_slots
        ]
        final_relations_by_slot = {
            MultiAgentAnnotator._relation_slot(r): MultiAgentAnnotator._sync_relation_endpoint_types(
                r, final_entity_type_by_text
            )
            for r in annotator.relations
            if MultiAgentAnnotator._relation_slot(r) not in disputed_relation_slots
        }

        for slot, relation in final_relations_by_slot.items():
            audit["preserved_consensus"].append(
                f'relation "{relation.e1_text} {relation.relation} {relation.e2_text}"'
            )

        for adj_relation in adjudicator.final_relations:
            synced_adj_relation = MultiAgentAnnotator._sync_relation_endpoint_types(
                adj_relation, final_entity_type_by_text
            )
            slot = MultiAgentAnnotator._relation_slot(synced_adj_relation)
            original = annotator_relations_by_slot.get(slot)
            current = final_relations_by_slot.get(slot)

            if original is not None:
                if slot in disputed_relation_slots:
                    final_relations_by_slot[slot] = synced_adj_relation
                    if slot not in relation_order:
                        relation_order.append(slot)
                    if (
                        current is not None
                        and MultiAgentAnnotator._relation_key(current)
                        != MultiAgentAnnotator._relation_key(synced_adj_relation)
                    ):
                        audit["allowed_changes"].append(
                            f'relation "{synced_adj_relation.e1_text} '
                            f"{synced_adj_relation.relation} "
                            f'{synced_adj_relation.e2_text}" changed '
                            "(final Critic disagreement)"
                        )
                    continue

                if (
                    current is not None
                    and MultiAgentAnnotator._relation_key(current)
                    != MultiAgentAnnotator._relation_key(synced_adj_relation)
                ):
                    audit["rejected_changes"].append(
                        f'relation "{synced_adj_relation.e1_text} '
                        f"{synced_adj_relation.relation} "
                        f'{synced_adj_relation.e2_text}" changed '
                        "(not in final Critic disagreements)"
                    )
                continue

            if any(
                MultiAgentAnnotator._disagreement_matches_relation(d, synced_adj_relation)
                for d in critic.disagreements
            ):
                final_relations_by_slot[slot] = synced_adj_relation
                relation_order.append(slot)
                audit["allowed_changes"].append(
                    f'relation "{synced_adj_relation.e1_text} '
                    f"{synced_adj_relation.relation} "
                    f'{synced_adj_relation.e2_text}" added '
                    "(final Critic disagreement)"
                )
            else:
                audit["rejected_changes"].append(
                    f'relation "{synced_adj_relation.e1_text} '
                    f"{synced_adj_relation.relation} "
                    f'{synced_adj_relation.e2_text}" added '
                    "(not in final Critic disagreements)"
                )

        for slot in disputed_relation_slots:
            if slot not in final_relations_by_slot:
                removed = annotator_relations_by_slot[slot]
                audit["allowed_changes"].append(
                    f'relation "{removed.e1_text} {removed.relation} '
                    f'{removed.e2_text}" removed '
                    "(final Critic disagreement)"
                )

        return ConstrainedAdjudication(
            final_entities=[final_entities_by_text[key] for key in entity_order],
            final_relations=[final_relations_by_slot[key] for key in relation_order],
            flagged_for_human_review=human_review_flags,
            status="constrained",
            audit=audit,
        )

    @staticmethod
    def _build_adjudicator_summary(
        sentence: str,
        deliberation_messages: List[Dict[str, Any]],
    ) -> str:
        """
        Build a condensed summary of the Annotator↔Critic deliberation
        for the Adjudicator, including:
        - The final annotation and final review (always present).
        - A per-round dispute trajectory (only for multi-round deliberations)
          showing what was challenged and what changed.
        """
        parse_a = MultiAgentAnnotator._parse_annotator_output
        parse_c = MultiAgentAnnotator._parse_critic_output

        # ── Separate messages by round ────────────────────────
        # Messages alternate: Annotator (round 1), Critic (round 1),
        # Annotator (round 2), Critic (round 2), ...
        annotator_msgs: List[Dict[str, Any]] = []
        critic_msgs: List[Dict[str, Any]] = []
        for m in deliberation_messages:
            if m["agent"] == "Annotator":
                annotator_msgs.append(m)
            elif m["agent"] == "Critic":
                critic_msgs.append(m)

        n_rounds = min(len(annotator_msgs), max(len(critic_msgs), 1))

        # ── Parse all structured outputs ──────────────────────
        annotator_outputs = [parse_a(m["content"]) for m in annotator_msgs]
        critic_outputs = [parse_c(m["content"]) for m in critic_msgs]

        # ── Build dispute trajectory (only if >1 round) ──────
        trajectory_lines: List[str] = []
        if n_rounds > 1:
            for r in range(n_rounds):
                round_lines = [f"### Round {r + 1}"]

                # Annotator summary for this round
                a_out = annotator_outputs[r] if r < len(annotator_outputs) else None
                if a_out:
                    n_ent = len(a_out.entities)
                    n_rel = len(a_out.relations)
                    round_lines.append(
                        f"Annotator proposed {n_ent} entities, {n_rel} relations."
                    )
                    # If round > 0, show what changed from previous round
                    if r > 0:
                        prev = annotator_outputs[r - 1]
                        if prev:
                            changes = _diff_annotator_rounds(prev, a_out)
                            if changes:
                                round_lines.append(f"Changes from round {r}: {changes}")
                            else:
                                round_lines.append("No changes from previous round.")

                # Critic summary for this round
                c_out = critic_outputs[r] if r < len(critic_outputs) else None
                if c_out:
                    n_agree = len(c_out.agreements)
                    n_disagree = len(c_out.disagreements)
                    n_missing = len(c_out.missing_annotations)
                    round_lines.append(
                        f"Critic: {n_agree} agreements, "
                        f"{n_disagree} disagreements, "
                        f"{n_missing} missing annotations."
                    )
                    # List each disagreement concisely
                    for d in c_out.disagreements:
                        sev = f" [{d.severity}]" if d.severity else ""
                        ref = f" (guideline: {d.guideline_reference})" if d.guideline_reference else ""
                        round_lines.append(
                            f"  - {d.target}: "
                            f"{d.annotator_label} → {d.proposed_label}{sev}{ref}"
                        )
                    for miss in c_out.missing_annotations:
                        round_lines.append(
                            f"  - MISSING: \"{miss.text}\" → {miss.entity_type}"
                        )

                trajectory_lines.append("\n".join(round_lines))

        # ── Build final-state sections ────────────────────────
        last_annotator_content = annotator_msgs[-1]["content"] if annotator_msgs else "(none)"
        last_critic_content = critic_msgs[-1]["content"] if critic_msgs else "(none)"

        # ── Assemble the full summary ─────────────────────────
        parts = [f'Sentence: "{sentence}"']

        if trajectory_lines:
            parts.append(
                "## Dispute trajectory\n" + "\n\n".join(trajectory_lines)
            )

        parts.append(f"## Final annotation (Annotator)\n{last_annotator_content}")
        parts.append(f"## Final review (Critic)\n{last_critic_content}")
        parts.append("Produce the final annotation.")

        return "\n\n".join(parts)

    @staticmethod
    def _append_jsonl(record: DeliberationRecord, path: Path):
        with path.open("a", encoding="utf8") as f:
            f.write(record.model_dump_json() + "\n")


# ─────────────────────────────────────────────────────────────
# Analysis  (same as vanilla)
# ─────────────────────────────────────────────────────────────

def analyze_disagreements(records: List[DeliberationRecord]) -> Dict[str, Any]:
    stats = {
        "total_sentences": len(records),
        "avg_agreement": 0.0,
        "avg_rounds": 0.0,
        "disagreement_patterns": {},
        "flagged_for_review": [],
        "entity_type_distribution": {},
        "relation_distribution": {},
    }
    scores, rounds = [], []
    for rec in records:
        scores.append(rec.agreement_score or 0)
        rounds.append(rec.rounds_used)
        for ent in rec.final_entities:
            stats["entity_type_distribution"][ent.entity_type] = (
                stats["entity_type_distribution"].get(ent.entity_type, 0) + 1
            )
        for rel in rec.final_relations:
            stats["relation_distribution"][rel.relation] = (
                stats["relation_distribution"].get(rel.relation, 0) + 1
            )
        for flag in rec.flagged_for_human_review:
            stats["flagged_for_review"].append(
                {"sentence": rec.sentence, "flag": flag}
            )
    stats["avg_agreement"] = sum(scores) / len(scores) if scores else 0
    stats["avg_rounds"] = sum(rounds) / len(rounds) if rounds else 0
    return stats


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AG2-based multi-agent annotation for biodiversity IE."
    )
    parser.add_argument("--sentences", type=str, nargs="+")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--guideline", type=Path, default=None)
    parser.add_argument("--seeds", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator-model", type=str, default="qwen3-32B-vllm")
    parser.add_argument("--critic-model", type=str, default="qwen3-32B-vllm")
    parser.add_argument("--adjudicator-model", type=str, default="qwen3-32B-vllm")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--guideline-search-backend",
        choices=["lexical", "embedding"],
        default=None,
        help="Optional guideline_search backend. Defaults to GUIDELINE_SEARCH_BACKEND or lexical.",
    )
    parser.add_argument(
        "--guideline-search-embedding-model",
        type=str,
        default=None,
        help="SentenceTransformer model name used when --guideline-search-backend=embedding.",
    )

    args = parser.parse_args()

    # Collect sentences
    sentences = []
    if args.sentences:
        sentences = args.sentences
    elif args.input_jsonl:
        with args.input_jsonl.open("r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    sentences.append(obj.get("text") or obj.get("sentence", ""))
    else:
        parser.error("Provide --sentences or --input-jsonl")

    # Clear output file
    args.output.write_text("")

    annotator = MultiAgentAnnotator(
        annotator_model=args.annotator_model,
        critic_model=args.critic_model,
        adjudicator_model=args.adjudicator_model,
        schema_path=args.schema,
        guideline_path=args.guideline,
        seeds_path=args.seeds,
        max_rounds=args.max_rounds,
        guideline_search_backend=args.guideline_search_backend,
        guideline_search_embedding_model=args.guideline_search_embedding_model,
    )

    records = annotator.annotate_batch(sentences, output_path=args.output)

    stats = analyze_disagreements(records)
    print(f"\n{'='*60}")
    print(f"  BATCH ANALYSIS")
    print(f"{'='*60}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
