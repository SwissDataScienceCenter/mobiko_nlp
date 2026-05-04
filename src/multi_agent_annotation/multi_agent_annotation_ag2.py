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
    - lookup_precedent   : look up adjudicated decisions from earlier sentences this batch
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
import time
import warnings
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field, ValidationError

from prompts import _annotator_system_msg, _critic_system_msg, _adjudicator_system_msg, _build_guideline_summary, LOW_CONFIDENCE_THRESHOLD

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
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    timing: Dict[str, Any] = Field(default_factory=dict)
    # Memory: which precedents the Annotator applied, which new ones were added
    precedents_applied: List[str] = Field(default_factory=list)   # span_text entries reused
    precedents_added: List[str] = Field(default_factory=list)     # new entries added to store


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


class PrecedentEntry(BaseModel):
    """A single adjudicated entity-type decision stored for cross-sentence reuse."""
    span_text: str                          # normalized span text
    entity_type: str                        # decided entity type
    rationale: str = ""                     # guideline rule / adjudicator reasoning
    confidence: float = 1.0
    source_sentence: str = ""              # sentence it came from (for traceability)
    times_applied: int = 0                 # incremented when the Annotator reuses it
    status: str = "authoritative"          # "authoritative" | "provisional"


class RelationPrecedent(BaseModel):
    """A single adjudicated relation-type decision stored for cross-sentence reuse."""
    e1_type: str
    relation: str
    e2_type: str
    rationale: str = ""
    confidence: float = 1.0
    source_sentence: str = ""
    times_applied: int = 0



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


# Asymmetric model allocation: Annotator/Critic handle high-volume grunt work
# on the cheaper local Qwen model; Adjudicator does the hard cross-transcript
# reasoning and benefits from a stronger model.
# Override at run-time without code changes: ADJUDICATOR_MODEL=gpt4o python ...
_DEFAULT_ADJUDICATOR_MODEL: str = os.getenv("ADJUDICATOR_MODEL", "qwen3-32B-vllm")


def build_llm_config(
    model_key: str,
    temperature: float = 0.3,
    timeout: int = 600,
) -> LLMConfig:
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
            "timeout": timeout,
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
            return sections
    except Exception:
        return []


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
    Returns an empty list if python-docx is unavailable or parsing fails.
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
        return sections
    except ImportError:
        return []



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
_PRECEDENT_STORE: Optional["PrecedentStore"] = None   # set per MultiAgentAnnotator instance
_GUIDELINE_SEARCH_BACKEND = "embedding"  # module-level default; override via arg or GUIDELINE_SEARCH_BACKEND env var
_DEFAULT_GUIDELINE_SEARCH_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME = _DEFAULT_GUIDELINE_SEARCH_EMBEDDING_MODEL
_GUIDELINE_SECTION_EMBEDDINGS: Optional[List[List[float]]] = None
_GUIDELINE_SECTIONS_HASH: Optional[str] = None  # hash of sections text; used to skip re-embedding
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


def _guideline_sections_hash(sections: List[Dict[str, str]]) -> str:
    import hashlib
    blob = json.dumps(sections, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode()).hexdigest()


def _init_tool_state(schema, guideline_sections, seed_examples,
                     entity_types_list: Optional[list] = None,
                     guideline_search_backend: Optional[str] = None,
                     guideline_search_embedding_model: Optional[str] = None,
                     precedent_store: Optional["PrecedentStore"] = None):
    """Populate module-level state used by tool functions."""
    global _SCHEMA, _TYPE_PAIR_TO_RELATIONS, _ALL_ENTITY_TYPES
    global _GUIDELINE_SECTIONS, _SEED_EXAMPLES, _PRECEDENT_STORE
    global _GUIDELINE_SEARCH_BACKEND, _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME
    global _GUIDELINE_SECTION_EMBEDDINGS, _GUIDELINE_SECTIONS_HASH, _GUIDELINE_EMBEDDING_ERROR

    _PRECEDENT_STORE = precedent_store

    _SCHEMA = schema
    _GUIDELINE_SECTIONS = guideline_sections
    _SEED_EXAMPLES = seed_examples
    _GUIDELINE_SEARCH_BACKEND = (
        guideline_search_backend
        or os.getenv("GUIDELINE_SEARCH_BACKEND")
        or _GUIDELINE_SEARCH_BACKEND  # preserve module-level default ("embedding")
    ).strip().lower()
    if _GUIDELINE_SEARCH_BACKEND not in {"lexical", "embedding"}:
        logger.warning(
            "Unknown GUIDELINE_SEARCH_BACKEND=%r; falling back to embedding",
            _GUIDELINE_SEARCH_BACKEND,
        )
        _GUIDELINE_SEARCH_BACKEND = "embedding"
    new_embedding_model = (
        guideline_search_embedding_model
        or os.getenv("GUIDELINE_SEARCH_EMBEDDING_MODEL")
        or _DEFAULT_GUIDELINE_SEARCH_EMBEDDING_MODEL
    )
    new_sections_hash = _guideline_sections_hash(guideline_sections)
    if (
        new_embedding_model != _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME
        or new_sections_hash != _GUIDELINE_SECTIONS_HASH
    ):
        _GUIDELINE_SECTION_EMBEDDINGS = None
        _GUIDELINE_EMBEDDING_ERROR = None
        _GUIDELINE_SECTIONS_HASH = new_sections_hash
    _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME = new_embedding_model

    # Pre-load model + section embeddings now so the first guideline_search call
    # doesn't stall mid-annotation with a cold-start download/encode.
    if _GUIDELINE_SEARCH_BACKEND == "embedding":
        _ensure_guideline_section_embeddings()

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
    query: Annotated[str, "Descriptive phrase including the span text AND its candidate entity type, e.g. 'habitat spatial property classification' or 'information concept vs anthropogenic entity'"],
) -> str:
    """Search the MoBiKo labelling guideline for relevant rules and classification steps."""
    backend = _GUIDELINE_SEARCH_BACKEND
    if backend == "embedding":
        try:
            results = _guideline_search_embedding(query)
            if results:  # non-empty list → return embedding results
                return _format_guideline_search_response(query, "embedding", results)
            # Empty results (no_match or short query) → fall through to lexical
            if results is None:
                logger.warning("Embedding search unavailable; falling back to lexical.")
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
    logger.info("Loaded embedding model: %s", _GUIDELINE_SEARCH_EMBEDDING_MODEL_NAME)
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


def lookup_precedent(
    span_text: Annotated[str, "Entity span to look up, e.g. 'elevation' or 'species richness'"],
) -> str:
    """
    Look up whether this entity span (or a similar one) has been annotated
    and adjudicated in an earlier sentence this session.

    Returns a list of established precedents: span text, entity type,
    rationale, and how many times it has been applied.
    If no precedent exists, returns an empty list.

    Use this BEFORE assigning a type to a span you are uncertain about,
    or to verify consistency with earlier decisions.
    """
    if _PRECEDENT_STORE is None:
        return json.dumps([])
    matches = _PRECEDENT_STORE.lookup(span_text)
    return json.dumps(
        [
            {
                "span_text": m.span_text,
                "entity_type": m.entity_type,
                "rationale": m.rationale,
                "confidence": m.confidence,
                "times_applied": m.times_applied,
                "status": m.status,
            }
            for m in matches
        ],
        ensure_ascii=False,
    )



# ─────────────────────────────────────────────────────────────
# Tool registration helper
# ─────────────────────────────────────────────────────────────

# Tools focused on proposing comprehensive annotations (entity + relation coverage)
ANNOTATOR_TOOL_FUNCTIONS = [
    (list_entity_types, "List all valid entity types from the schema."),
    (schema_lookup, "Check which relations are valid between two entity types."),
    (guideline_search, "Search the labelling guideline for relevant rules."),
    (lookup_precedent, "Look up how a span was annotated and adjudicated in earlier sentences."),
]

# Tools focused on verifying and quality-checking annotations
CRITIC_TOOL_FUNCTIONS = [
    (schema_lookup, "Check which relations are valid between two entity types."),
    (guideline_search, "Search the labelling guideline for relevant rules."),
    (lookup_precedent, "Look up how a span was adjudicated in earlier sentences this batch."),
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
_DEFAULT_GUIDELINE = _THIS_DIR / "MoBiKo label guidance draft v3.docx"


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


# ─────────────────────────────────────────────────────────────
# Precedent Store — cross-sentence annotation memory
# ─────────────────────────────────────────────────────────────

class PrecedentStore:
    """
    Stores adjudicated entity-type and relation-type decisions made in
    earlier sentences of the current batch, so the Annotator can apply
    them consistently instead of re-deliberating the same spans.

    Design principles
    -----------------
    * Only *authoritative* decisions are injected into agent context:
      those where the Critic did not dispute the label AND the Adjudicator
      had high confidence (≥ LOW_CONFIDENCE_THRESHOLD).
    * Low-confidence or critically-disputed decisions enter a *provisional*
      pool that is excluded from automatic reuse but is visible in logs.
    * The store is batch-scoped: it is reset between independent batches
      but persists across sentences within a single annotate_batch() call.
    * Error propagation is mitigated by the provisional pool and by
      tracking `times_applied` — a type that appears as an authoritative
      precedent but is consistently re-challenged by the Critic should
      be moved to provisional by the caller.
    """

    _STOP: set = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from",
        "in", "into", "is", "it", "its", "of", "on", "or", "the", "to", "with",
    }

    def __init__(self) -> None:
        self.entity_entries: List[PrecedentEntry] = []
        self.relation_entries: List[RelationPrecedent] = []
        self.provisional: List[PrecedentEntry] = []

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.lower().split())

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return [t for t in re.findall(r"\w+", text.lower()) if t not in cls._STOP]

    @classmethod
    def _overlap(cls, a: str, b: str) -> float:
        ta, tb = set(cls._tokens(a)), set(cls._tokens(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta), len(tb))

    # ── Writing ───────────────────────────────────────────────

    def add_entity(
        self,
        span_text: str,
        entity_type: str,
        rationale: str,
        confidence: float,
        source_sentence: str,
    ) -> str:
        """
        Add or update an entity precedent.
        Returns "authoritative", "provisional", or "updated" (if already existed).
        """
        norm = self._norm(span_text)

        # Update if the same span already exists
        for entry in self.entity_entries:
            if self._norm(entry.span_text) == norm:
                if entity_type == entry.entity_type:
                    entry.times_applied += 1
                    return "updated"
                else:
                    # Conflict: same span, different type — move to provisional
                    self.provisional.append(PrecedentEntry(
                        span_text=span_text,
                        entity_type=entity_type,
                        rationale=f"CONFLICT with existing {entry.entity_type}. {rationale}",
                        confidence=confidence,
                        source_sentence=source_sentence,
                        status="provisional",
                    ))
                    return "provisional"

        # New entry
        entry = PrecedentEntry(
            span_text=norm,
            entity_type=entity_type,
            rationale=rationale,
            confidence=confidence,
            source_sentence=source_sentence[:80],
            status="authoritative" if confidence >= LOW_CONFIDENCE_THRESHOLD else "provisional",
        )
        if entry.status == "authoritative":
            self.entity_entries.append(entry)
        else:
            self.provisional.append(entry)
        return entry.status

    def add_relation(
        self,
        e1_type: str,
        relation: str,
        e2_type: str,
        rationale: str,
        confidence: float,
        source_sentence: str,
    ) -> None:
        """Add a relation-type precedent (keyed on type triple, not span text)."""
        key = (e1_type.upper(), relation.upper(), e2_type.upper())
        for entry in self.relation_entries:
            if (entry.e1_type, entry.relation, entry.e2_type) == key:
                entry.times_applied += 1
                return
        self.relation_entries.append(RelationPrecedent(
            e1_type=key[0], relation=key[1], e2_type=key[2],
            rationale=rationale,
            confidence=confidence,
            source_sentence=source_sentence[:80],
        ))

    def add_from_adjudication(
        self,
        constrained: "ConstrainedAdjudication",
        source_sentence: str,
    ) -> List[str]:
        """
        Extract authoritative decisions from a ConstrainedAdjudication and
        populate the store.  Returns list of span_texts newly added as
        authoritative (excludes provisional and updated entries).
        """
        added: List[str] = []
        for ent in constrained.final_entities:
            # Always attempt to add — add_entity routes to provisional if
            # confidence is below threshold or there is a conflict.
            # Flagged items (genuinely ambiguous) are skipped.
            if ent.text in constrained.flagged_for_human_review:
                continue
            status = self.add_entity(
                span_text=ent.text,
                entity_type=ent.entity_type,
                rationale=ent.reasoning or "",
                confidence=ent.confidence if ent.confidence is not None else 1.0,
                source_sentence=source_sentence,
            )
            if status == "authoritative":
                added.append(ent.text)
        for rel in constrained.final_relations:
            if rel.confidence is None or rel.confidence >= LOW_CONFIDENCE_THRESHOLD:
                self.add_relation(
                    e1_type=rel.e1_type,
                    relation=rel.relation,
                    e2_type=rel.e2_type,
                    rationale=rel.reasoning or "",
                    confidence=rel.confidence or 1.0,
                    source_sentence=source_sentence,
                )
        return added

    # ── Reading ───────────────────────────────────────────────

    def lookup(self, span_text: str, threshold: float = 0.6) -> List[PrecedentEntry]:
        """
        Return authoritative precedents whose span_text overlaps with
        the query above `threshold`.  Sorted by overlap descending.
        """
        scored: List[Tuple[float, PrecedentEntry]] = []
        for entry in self.entity_entries:
            score = self._overlap(span_text, entry.span_text)
            if score >= threshold:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:5]]

    def to_context_block(self, max_entries: int = 20) -> str:
        """
        Produce a compact string for injection into the Annotator's
        task message.  Shows the most-applied authoritative precedents.

        Returns an empty string if the store has no entries yet.
        """
        if not self.entity_entries:
            return ""

        # Most frequently applied first, then by insertion order
        sorted_entries = sorted(
            self.entity_entries, key=lambda e: -e.times_applied
        )[:max_entries]

        lines = ["## Established Precedents (apply these consistently across sentences)",
                 "These decisions were adjudicated in earlier sentences this batch. "
                 "Apply them directly without re-deliberating — use lookup_precedent "
                 "if you need to verify a specific span.\n"]

        for e in sorted_entries:
            freq = f" (×{e.times_applied})" if e.times_applied > 0 else ""
            rationale = f"  ← {e.rationale}" if e.rationale else ""
            lines.append(f'- "{e.span_text}" → {e.entity_type}{freq}{rationale}')

        if self.relation_entries:
            lines.append("\nEstablished relation precedents:")
            for r in self.relation_entries[:10]:
                freq = f" (×{r.times_applied})" if r.times_applied > 0 else ""
                lines.append(f"- ({r.e1_type}, {r.relation}, {r.e2_type}){freq}")

        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        """Return a summary dict for logging."""
        return {
            "authoritative_entities": len(self.entity_entries),
            "provisional_entities": len(self.provisional),
            "relation_triples": len(self.relation_entries),
            "total_applications": sum(e.times_applied for e in self.entity_entries),
        }

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Atomically write the store to a JSON file.

        Uses write-to-temp-then-rename so a crash mid-write never corrupts
        the existing file.  The stored format is a plain dict so it is
        human-readable and easy to inspect or edit manually.
        """
        data = {
            "entity_entries":    [e.model_dump() for e in self.entity_entries],
            "relation_entries":  [r.model_dump() for r in self.relation_entries],
            "provisional":       [e.model_dump() for e in self.provisional],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)          # atomic on POSIX; best-effort on Windows
        logger.debug("Precedent store saved → %s  (%d authoritative, %d provisional)",
                     path, len(self.entity_entries), len(self.provisional))

    @classmethod
    def load(cls, path: Path) -> "PrecedentStore":
        """
        Load a store from a JSON file previously written by :meth:`save`.

        Returns an empty store if the file does not exist, so callers can
        always do ``store = PrecedentStore.load(path)`` unconditionally.
        """
        path = Path(path)
        store = cls()
        if not path.exists():
            logger.info("No precedent store found at %s — starting fresh.", path)
            return store
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            store.entity_entries   = [PrecedentEntry(**e)    for e in data.get("entity_entries",   [])]
            store.relation_entries = [RelationPrecedent(**r) for r in data.get("relation_entries", [])]
            store.provisional      = [PrecedentEntry(**e)    for e in data.get("provisional",      [])]
            logger.info(
                "Precedent store loaded from %s — %d authoritative, %d provisional, %d relations.",
                path, len(store.entity_entries), len(store.provisional), len(store.relation_entries),
            )
        except Exception as exc:
            logger.warning("Could not load precedent store from %s: %s — starting fresh.", path, exc)
        return store


class MultiAgentAnnotator:
    """
    AG2-based multi-agent annotation system.

    Parameters
    ----------
    annotator_model, critic_model : str
        Keys into MODEL_ENDPOINTS.  High-volume agents — a fast, cheaper
        model (e.g. "qwen3-35B-vllm") is appropriate here.
    adjudicator_model : str
        Key into MODEL_ENDPOINTS.  This agent does the hardest reasoning
        (cross-transcript adjudication + tiebreakers), so a stronger model
        like "gpt4o" is recommended.  Defaults to ``_DEFAULT_ADJUDICATOR_MODEL``
        which can be overridden with the ``ADJUDICATOR_MODEL`` env var.
    schema_path : Path
        Relation schema file.
    decision_support_path : Path
        Decision support .docx (table-based decision guide for the Annotator).
        Defaults to the copy in src/multi_agent_annotation/.
    guideline_path : Path
        MoBiKo labelling guideline .docx (narrative, for Critic & Adjudicator).
        Defaults to the copy in src/multi_agent_annotation/.
    seeds_path : Path
        Seed examples (used by the Annotator for initial annotation context).
    max_rounds : int
        Max Annotator↔Critic turns before adjudication.
    guideline_search_backend : str
        "lexical" by default; set to "embedding" to opt into SentenceTransformer retrieval.
    """

    def __init__(
        self,
        annotator_model: str = "qwen3-32B-vllm",
        critic_model: str = "qwen3-32B-vllm",
        adjudicator_model: str = _DEFAULT_ADJUDICATOR_MODEL,
        schema_path: Optional[Path] = None,
        decision_support_path: Optional[Path] = None,
        guideline_path: Optional[Path] = None,
        seeds_path: Optional[Path] = None,
        max_rounds: int = 2,
        entity_schema_str: Optional[str] = None,
        entity_types_list: Optional[list] = None,
        guideline_search_backend: Optional[str] = None,
        guideline_search_embedding_model: Optional[str] = None,
        precedent_store_path: Optional[Path] = None,
        use_precedent_memory: bool = True,
        request_timeout: int = 600,
    ):
        self.max_rounds = max_rounds
        self.use_precedent_memory = use_precedent_memory
        self.precedent_store_path = Path(precedent_store_path) if precedent_store_path else None

        # ── Load resources ───────────────────────────────────
        relation_schema = load_schema(schema_path) if schema_path else {}
        seeds = load_seeds(seeds_path) if seeds_path else {}

        # Decision support doc → Annotator system prompt (compact decision table)
        ds_path = decision_support_path or _DEFAULT_DECISION_SUPPORT
        decision_support_sections = (
            load_decision_support(ds_path)
            if ds_path and ds_path.exists()
            else []
        )

        # MoBiKo v2 narrative → Critic & Adjudicator system prompts (edge cases, tiebreaker)
        gl_path = guideline_path or _DEFAULT_GUIDELINE
        guidance_sections = (
            load_guideline_from_docx(gl_path)
            if gl_path and gl_path.exists()
            else []
        )

        # guideline_search tool searches across both documents combined
        all_sections = decision_support_sections + guidance_sections

        # ── Precedent store (persistent memory across batches) ──
        if use_precedent_memory:
            self.precedent_store: Optional[PrecedentStore] = (
                PrecedentStore.load(self.precedent_store_path)
                if self.precedent_store_path
                else PrecedentStore()
            )
        else:
            self.precedent_store = None
            logger.info("Precedent memory disabled.")

        _init_tool_state(relation_schema, all_sections, seeds,
                         entity_types_list=entity_types_list or _FALLBACK_ENTITY_TYPES,
                         guideline_search_backend=guideline_search_backend,
                         guideline_search_embedding_model=guideline_search_embedding_model,
                         precedent_store=self.precedent_store)

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
        annotator_llm = build_llm_config(annotator_model, temperature=0.2, timeout=request_timeout)
        critic_llm = build_llm_config(critic_model, temperature=0.3, timeout=request_timeout)
        adjudicator_llm = build_llm_config(adjudicator_model, temperature=0.1, timeout=request_timeout)
        logger.info(
            f"Models — annotator: {annotator_model} | "
            f"critic: {critic_model} | "
            f"adjudicator: {adjudicator_model}"
        )

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
        # Each agent gets its own executor so tool registrations never share
        # state across agents — shared executors caused AG2 to corrupt its
        # function_map when the same function name was registered for multiple
        # callers, producing content=None messages and a ValueError on send.
        def _make_executor(name: str) -> ConversableAgent:
            return ConversableAgent(
                name=name,
                llm_config=False,
                human_input_mode="NEVER",
                is_termination_msg=_is_final_terminate_msg,
            )

        self.annotator_executor  = _make_executor("AnnotatorExecutor")
        self.critic_executor     = _make_executor("CriticExecutor")
        self.adj_executor        = _make_executor("AdjudicatorExecutor")

        # ── Register tools ───────────────────────────────────
        if use_precedent_memory:
            annotator_tools = ANNOTATOR_TOOL_FUNCTIONS
            critic_tools = CRITIC_TOOL_FUNCTIONS
        else:
            annotator_tools = [t for t in ANNOTATOR_TOOL_FUNCTIONS if t[0] is not lookup_precedent]
            critic_tools = [t for t in CRITIC_TOOL_FUNCTIONS if t[0] is not lookup_precedent]
        _register_tools_on_agents(self.annotator,  self.annotator_executor, annotator_tools)
        _register_tools_on_agents(self.critic,     self.critic_executor,    critic_tools)
        _register_tools_on_agents(self.adjudicator, self.adj_executor,      critic_tools)

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
        _sentence_t0 = time.perf_counter()

        # ── Build task message ────────────────────────────────
        task_msg = f'Annotate this sentence:\n\n"{sentence}"'
        if pre_identified_entities:
            task_msg += (
                f"\n\nPre-identified entities (verify types, find relations):\n"
                f"{json.dumps(pre_identified_entities, ensure_ascii=False, indent=2)}"
            )

        # Inject precedent context if the store has entries from earlier sentences
        precedent_block = (
            self.precedent_store.to_context_block()
            if self.precedent_store is not None else ""
        )
        if precedent_block:
            task_msg += f"\n\n{precedent_block}"

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
        annotator_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        critic_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        round_timings: List[Dict[str, Any]] = []
        _phase1_t0 = time.perf_counter()

        for round_idx in range(self.max_rounds):
            # ── Annotator turn ────────────────────────────────
            if round_idx == 0:
                ann_msg = task_msg
            else:
                ann_msg = self._build_annotator_revision_msg(
                    sentence, last_annotator_text, last_critic_text,
                    pre_identified_entities,
                )
            logger.info("  [round %d] Annotator …", round_idx + 1)
            ann_content, ann_record = self._run_agent_turn(self.annotator, self.annotator_executor, ann_msg)
            deliberation_messages.append(ann_record)
            last_annotator_text = ann_content
            for k in annotator_tokens:
                annotator_tokens[k] += ann_record["token_usage"].get(k, 0)
            ann_elapsed = ann_record.get("elapsed_s", 0.0)
            logger.info("  [round %d] Annotator done in %.1fs", round_idx + 1, ann_elapsed)

            parsed_ann = self._parse_annotator_output(ann_content)
            if parsed_ann is not None:
                last_annotator_out = parsed_ann

            # ── Critic turn ───────────────────────────────────
            crit_msg = self._build_critic_review_msg(sentence, ann_content)
            logger.info("  [round %d] Critic …", round_idx + 1)
            crit_content, crit_record = self._run_agent_turn(self.critic, self.critic_executor, crit_msg)
            deliberation_messages.append(crit_record)
            last_critic_text = crit_content
            for k in critic_tokens:
                critic_tokens[k] += crit_record["token_usage"].get(k, 0)
            crit_elapsed = crit_record.get("elapsed_s", 0.0)
            logger.info("  [round %d] Critic done in %.1fs", round_idx + 1, crit_elapsed)

            round_timings.append({
                "round": round_idx + 1,
                "annotator_s": ann_elapsed,
                "critic_s": crit_elapsed,
                "round_total_s": round(ann_elapsed + crit_elapsed, 2),
            })

            parsed_crit = self._parse_critic_output(crit_content)
            if parsed_crit is not None:
                last_critic_out = parsed_crit

            if (
                last_critic_out is not None
                and not last_critic_out.disagreements
                and not last_critic_out.missing_annotations
            ):
                break

        phase1_s = round(time.perf_counter() - _phase1_t0, 2)
        record.rounds_used = round_idx + 1
        record.messages = deliberation_messages

        if last_annotator_out is None:
            repair_messages, repaired_text = self._repair_agent_json(
                requester=self.annotator_executor,
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
                requester=self.critic_executor,
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
        logger.info("Phase 2: Adjudicator resolving …")
        _phase2_t0 = time.perf_counter()

        # Build a condensed summary of the full deliberation trajectory:
        # final annotation + final critique + per-round dispute history.
        adjudicator_msg = self._build_adjudicator_summary(
            sentence, deliberation_messages
        )

        self.adjudicator.reset()
        self.adj_executor.reset()
        adj_result = self.adjudicator.initiate_chat(
            recipient=self.adj_executor,
            message=adjudicator_msg,
            max_turns=self._adjudicator_max_turns(),
        )
        phase2_s = round(time.perf_counter() - _phase2_t0, 2)
        logger.info("  Adjudicator done in %.1fs", phase2_s)
        adjudicator_tokens = self._sum_usage(getattr(adj_result, "cost", {}))

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
                    requester=self.adj_executor,
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

        # ── Update precedent store from this sentence ─────────
        if self.precedent_store is not None and constrained.status not in ("annotator_parse_failed",):
            added_spans = self.precedent_store.add_from_adjudication(
                constrained, source_sentence=sentence
            )
            record.precedents_added = added_spans
            store_stats = self.precedent_store.stats()
            logger.info(
                f"Precedent store: {store_stats['authoritative_entities']} authoritative, "
                f"{store_stats['provisional_entities']} provisional, "
                f"{store_stats['total_applications']} total applications"
            )
            if self.precedent_store_path:
                self.precedent_store.save(self.precedent_store_path)

        # Track which precedents the Annotator applied in this sentence
        if self.precedent_store is not None and precedent_block and last_annotator_out:
            for entry in self.precedent_store.entity_entries:
                for ent in last_annotator_out.entities:
                    if self.precedent_store._overlap(ent.text, entry.span_text) >= 0.6:
                        if ent.entity_type == entry.entity_type:
                            if entry.span_text not in record.precedents_applied:
                                record.precedents_applied.append(entry.span_text)
                                entry.times_applied += 1

        # ── Post-process: flag relations between overlapping spans ─
        overlap_flags = self._find_overlap_relation_flags(
            record.final_entities, record.final_relations
        )
        if overlap_flags:
            seen_flags = {f.lower() for f in record.flagged_for_human_review}
            for flag in overlap_flags:
                if flag.lower() not in seen_flags:
                    record.flagged_for_human_review.append(flag)
                    seen_flags.add(flag.lower())

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

        total_tokens = {
            k: annotator_tokens[k] + critic_tokens[k] + adjudicator_tokens.get(k, 0)
            for k in annotator_tokens
        }
        record.token_usage = {
            "annotator": annotator_tokens,
            "critic": critic_tokens,
            "adjudicator": adjudicator_tokens,
            "total": total_tokens,
        }

        sentence_s = round(time.perf_counter() - _sentence_t0, 2)
        record.timing = {
            "total_s": sentence_s,
            "phase1_s": phase1_s,
            "phase2_s": phase2_s,
            "rounds": round_timings,
        }

        logger.info(
            f"Done in {sentence_s:.1f}s (phase1={phase1_s:.1f}s, phase2={phase2_s:.1f}s) | "
            f"{len(record.final_entities)} entities, "
            f"{len(record.final_relations)} relations, "
            f"agreement={record.agreement_score if record.agreement_score is None else f'{record.agreement_score:.2f}'} | "
            f"tokens — annotator: {annotator_tokens['total_tokens']} "
            f"(prompt {annotator_tokens['prompt_tokens']} + completion {annotator_tokens['completion_tokens']}), "
            f"critic: {critic_tokens['total_tokens']} "
            f"(prompt {critic_tokens['prompt_tokens']} + completion {critic_tokens['completion_tokens']}), "
            f"adjudicator: {adjudicator_tokens.get('total_tokens', 0)} "
            f"(prompt {adjudicator_tokens.get('prompt_tokens', 0)} + completion {adjudicator_tokens.get('completion_tokens', 0)}), "
            f"total: {total_tokens['total_tokens']}"
        )
        return record

    def annotate_batch(
        self,
        sentences: List[str],
        pre_entities: Optional[List[Optional[List[dict]]]] = None,
        output_path: Optional[Path] = None,
        resume: bool = False,
        max_retries: int = 2,
    ) -> List[DeliberationRecord]:
        """Annotate a batch of sentences with JSONL output.

        When ``resume=True`` and ``output_path`` points to an existing file,
        sentences whose text already appears in that file are skipped and their
        records are prepended to the returned list.  The output file is left
        intact; new records are appended.  When ``resume=False`` (default) the
        output file is cleared before the first write.
        """
        # ── Resume: load already-completed records ────────────
        done_sentences: set[str] = set()
        records: List[DeliberationRecord] = []
        if resume and output_path and Path(output_path).exists():
            for line in Path(output_path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = DeliberationRecord.model_validate_json(line)
                    done_sentences.add(rec.sentence)
                    records.append(rec)
                except Exception:
                    pass
            if done_sentences:
                logger.info(
                    "Resume: %d sentence(s) already done — skipping.", len(done_sentences)
                )
        elif output_path:
            Path(output_path).write_text("", encoding="utf-8")

        ents_list = pre_entities or [None] * len(sentences)
        _batch_t0 = time.perf_counter()
        sentence_timings: List[Dict[str, Any]] = []

        for i, (sent, ents) in enumerate(zip(sentences, ents_list)):
            if sent in done_sentences:
                logger.info(f"  [skip] Sentence {i+1}/{len(sentences)} already in output.")
                continue

            logger.info(f"\n{'#'*60}\n  Sentence {i+1}/{len(sentences)}\n{'#'*60}")
            logger.info(f"  {sent[:100]}...")

            record = None
            last_exc: Optional[Exception] = None
            _sent_t0 = time.perf_counter()
            for attempt in range(1, max_retries + 1):
                # Clear agent chat histories before each attempt; preserve the
                # precedent store so decisions from earlier sentences carry over.
                self.annotator.reset()
                self.critic.reset()
                self.adjudicator.reset()
                self.annotator_executor.reset()
                self.critic_executor.reset()
                self.adj_executor.reset()

                try:
                    record = self.annotate_sentence(sent, ents)
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Sentence %d/%d failed on attempt %d/%d: %s",
                        i + 1, len(sentences), attempt, max_retries, exc,
                    )
                    if attempt < max_retries:
                        logger.info("Retrying…")

            sent_elapsed = round(time.perf_counter() - _sent_t0, 2)

            if record is None:
                logger.error(
                    "Sentence %d/%d skipped after %d failed attempt(s) — "
                    "it will be retried when you resume. Last error: %s",
                    i + 1, len(sentences), max_retries, last_exc,
                )
                sentence_timings.append({
                    "index": i + 1,
                    "preview": sent[:60],
                    "elapsed_s": sent_elapsed,
                    "status": "failed",
                })
                continue  # not written to output, so --resume will retry it

            sentence_timings.append({
                "index": i + 1,
                "preview": sent[:60],
                "elapsed_s": sent_elapsed,
                "status": "ok",
            })
            records.append(record)

            if output_path:
                self._append_jsonl(record, output_path)

        # ── Batch timing summary ──────────────────────────────
        batch_elapsed = round(time.perf_counter() - _batch_t0, 2)
        if sentence_timings:
            logger.info("\n%s\n  TIMING SUMMARY\n%s", "=" * 60, "=" * 60)
            for st in sentence_timings:
                status_tag = "" if st["status"] == "ok" else "  [FAILED]"
                logger.info(
                    "  [%d/%d] %s…  %.1fs%s",
                    st["index"], len(sentences), st["preview"], st["elapsed_s"], status_tag,
                )
            ok_times = [st["elapsed_s"] for st in sentence_timings if st["status"] == "ok"]
            if ok_times:
                logger.info(
                    "  avg: %.1fs | min: %.1fs | max: %.1fs | total batch: %.1fs",
                    sum(ok_times) / len(ok_times), min(ok_times), max(ok_times), batch_elapsed,
                )

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

    @staticmethod
    def _sum_usage(cost: Dict[str, Any]) -> Dict[str, int]:
        """Sum prompt/completion tokens from an ag2 CostDict."""
        block = (cost or {}).get("usage_including_cached_inference", {})
        prompt = sum(v.get("prompt_tokens", 0) for v in block.values() if isinstance(v, dict))
        completion = sum(v.get("completion_tokens", 0) for v in block.values() if isinstance(v, dict))
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _run_agent_turn(
        self,
        agent: ConversableAgent,
        executor: ConversableAgent,
        message: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run one agent turn against its dedicated executor.

        Resets both agents so each round starts with a clean history and tool
        results arrive as role="tool" messages, not as counterpart conversation
        turns. All tool calls from the turn are folded into a single record
        message keyed by agent name.

        Returns (last_content, record_message).  record_message includes a
        ``token_usage`` key with prompt/completion/total counts for the turn.
        """
        agent.reset()
        executor.reset()
        _t0 = time.perf_counter()
        chat = agent.initiate_chat(
            recipient=executor,
            message=message,
            max_turns=self._agent_turn_max_turns(),
        )
        elapsed_s = time.perf_counter() - _t0
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
            "token_usage": self._sum_usage(getattr(chat, "cost", {})),
            "elapsed_s": round(elapsed_s, 2),
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
            retry_result = self.adj_executor.initiate_chat(
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
    def _remove_nested_entities(
        entities: List[EntityAnnotation],
        relations: List[RelationFlat],
        audit: Dict[str, Any],
    ) -> Tuple[List[EntityAnnotation], List[RelationFlat]]:
        """
        Drop entities whose text is strictly contained within a longer co-annotated
        entity span (e.g. "limited" inside "limited information").  Relations that
        reference a dropped entity are also removed.

        When offsets are available the containment check uses character positions;
        otherwise it falls back to substring matching.
        """
        def _is_contained(short: EntityAnnotation, long: EntityAnnotation) -> bool:
            ts, tl = short.text.lower().strip(), long.text.lower().strip()
            if ts == tl:
                return False
            if all(x is not None for x in (short.start, short.end, long.start, long.end)):
                return long.start <= short.start and short.end <= long.end  # type: ignore[operator]
            return ts in tl

        dropped: set[str] = set()
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if _is_contained(e1, e2):
                    k = MultiAgentAnnotator._normalize_annotation_text(e1.text)
                    if k not in dropped:
                        dropped.add(k)
                        audit["warnings"].append(
                            f'entity "{e1.text}" removed (nested inside "{e2.text}")'
                        )
                elif _is_contained(e2, e1):
                    k = MultiAgentAnnotator._normalize_annotation_text(e2.text)
                    if k not in dropped:
                        dropped.add(k)
                        audit["warnings"].append(
                            f'entity "{e2.text}" removed (nested inside "{e1.text}")'
                        )

        if not dropped:
            return entities, relations

        clean_entities = [
            e for e in entities
            if MultiAgentAnnotator._normalize_annotation_text(e.text) not in dropped
        ]
        clean_relations = [
            r for r in relations
            if MultiAgentAnnotator._normalize_annotation_text(r.e1_text) not in dropped
            and MultiAgentAnnotator._normalize_annotation_text(r.e2_text) not in dropped
        ]
        for r in relations:
            e1k = MultiAgentAnnotator._normalize_annotation_text(r.e1_text)
            e2k = MultiAgentAnnotator._normalize_annotation_text(r.e2_text)
            if e1k in dropped or e2k in dropped:
                audit["warnings"].append(
                    f'relation "{r.e1_text} {r.relation} {r.e2_text}" removed '
                    "(endpoint was a nested entity)"
                )
        return clean_entities, clean_relations

    @staticmethod
    def _find_overlap_relation_flags(
        final_entities: List[EntityAnnotation],
        final_relations: List[RelationAnnotation],
    ) -> List[str]:
        """
        Return flag strings for every relation whose two endpoints have
        overlapping or nested character spans.

        Uses character offsets when both entities have them; falls back to
        text-containment (one span text is a substring of the other) when
        offsets are absent.
        """
        def _overlaps(a: EntityAnnotation, b: EntityAnnotation) -> bool:
            if all(x is not None for x in (a.start, a.end, b.start, b.end)):
                return a.start < b.end and b.start < a.end  # type: ignore[operator]
            ta = a.text.lower().strip()
            tb = b.text.lower().strip()
            return ta != tb and (ta in tb or tb in ta)

        overlapping: set[tuple[str, str]] = set()
        for i, e1 in enumerate(final_entities):
            for e2 in final_entities[i + 1:]:
                if _overlaps(e1, e2):
                    key = (
                        min(e1.text.lower(), e2.text.lower()),
                        max(e1.text.lower(), e2.text.lower()),
                    )
                    overlapping.add(key)

        flags = []
        for rel in final_relations:
            t1, t2 = rel.e1.text.lower(), rel.e2.text.lower()
            if (min(t1, t2), max(t1, t2)) in overlapping:
                flags.append(
                    f'Relation between overlapping spans: '
                    f'"{rel.e1.text}" -{rel.relation}-> "{rel.e2.text}"'
                )
        return flags

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

        # Drop disputed entities the Adjudicator chose not to re-include.
        # This handles span-split corrections (e.g. "warmer lakes" → "lakes")
        # where the Adjudicator replaces the compound span with a different-text
        # entity instead of relabelling the same span.
        adj_entity_keys = {
            MultiAgentAnnotator._normalize_annotation_text(e.text)
            for e in adjudicator.final_entities
        }
        for disputed_key in list(disagreement_targets):
            if disputed_key in final_entities_by_text and disputed_key not in adj_entity_keys:
                entity = final_entities_by_text.pop(disputed_key)
                entity_order = [k for k in entity_order if k != disputed_key]
                audit["allowed_changes"].append(
                    f'entity "{entity.text}" removed '
                    "(disputed, Adjudicator replaced with different span)"
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

            e1_key = MultiAgentAnnotator._normalize_annotation_text(synced_adj_relation.e1_text)
            e2_key = MultiAgentAnnotator._normalize_annotation_text(synced_adj_relation.e2_text)
            # Allow relations produced from span-splitting: both endpoints must be
            # valid final entities and at least one must come from critic missing_annotations.
            split_span_relation = (
                e1_key in final_entity_type_by_text
                and e2_key in final_entity_type_by_text
                and (e1_key in missing_by_text or e2_key in missing_by_text)
            )
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
            elif split_span_relation:
                final_relations_by_slot[slot] = synced_adj_relation
                relation_order.append(slot)
                audit["allowed_changes"].append(
                    f'relation "{synced_adj_relation.e1_text} '
                    f"{synced_adj_relation.relation} "
                    f'{synced_adj_relation.e2_text}" added '
                    "(split-span relation from Critic missing annotations)"
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

        out_entities = [final_entities_by_text[key] for key in entity_order]
        out_relations = [final_relations_by_slot[key] for key in relation_order]

        # Drop entities whose span is strictly nested inside a longer co-annotated
        # entity, and remove any relation that references a dropped entity.
        out_entities, out_relations = MultiAgentAnnotator._remove_nested_entities(
            out_entities, out_relations, audit
        )

        return ConstrainedAdjudication(
            final_entities=out_entities,
            final_relations=out_relations,
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
        "precedents": {
            "total_applied": sum(len(r.precedents_applied) for r in records),
            "total_added": sum(len(r.precedents_added) for r in records),
            "per_sentence_applied": [
                {"sentence": r.sentence[:60], "applied": r.precedents_applied}
                for r in records if r.precedents_applied
            ],
        },
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
        default="embeding",
        help="Optional guideline_search backend. Defaults to GUIDELINE_SEARCH_BACKEND or lexical.",
    )
    parser.add_argument(
        "--guideline-search-embedding-model",
        type=str,
        default=None,
        help="SentenceTransformer model name used when --guideline-search-backend=embedding.",
    )
    parser.add_argument(
        "--precedent-store",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file for the persistent precedent store. "
            "Loaded on startup if it exists; updated after every sentence. "
            "Omit to use a fresh in-memory store that is discarded after the run."
        ),
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
        precedent_store_path=args.precedent_store,
    )

    records = annotator.annotate_batch(sentences, output_path=args.output)

    stats = analyze_disagreements(records)
    print(f"\n{'='*60}")
    print(f"  BATCH ANALYSIS")
    print(f"{'='*60}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()