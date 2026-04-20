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
import os
import re
import sys
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
    agreement_score: Optional[float] = None
    rounds_used: int = 0


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


class CriticOutput(BaseModel):
    agreements: List[str] = Field(default_factory=list)
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
        "model_name": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
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


_FALLBACK_ENTITY_TYPES = [
    "ABIOTIC COLLECTIVE ENTITY", "ABIOTIC ENTITY", "ABIOTIC PROCESS", "ABIOTIC PROPERTY",
    "ANTHROPOGENIC ENTITY", "ANTHROPOGENIC PROCESS", "ANTHROPOGENIC PROPERTY",
    "BIOTIC COLLECTIVE ENTITY", "BIOTIC ENTITY", "BIOTIC PROCESS", "BIOTIC PROPERTY",
    "CONCEPT", "SPATIAL ENTITY", "SPATIAL PROPERTY", "TEMPORAL ENTITY", "TEMPORAL PROPERTY",
]


def _init_tool_state(schema, guideline_sections, seed_examples,
                     entity_types_list: Optional[list] = None):
    """Populate module-level state used by tool functions."""
    global _SCHEMA, _TYPE_PAIR_TO_RELATIONS, _ALL_ENTITY_TYPES
    global _GUIDELINE_SECTIONS, _SEED_EXAMPLES

    _SCHEMA = schema
    _GUIDELINE_SECTIONS = guideline_sections
    _SEED_EXAMPLES = seed_examples

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
    tokens = set(query.lower().split())
    scored = []
    for sec in _GUIDELINE_SECTIONS:
        title_tokens = set(sec["title"].lower().split())
        content_tokens = set(sec["content"].lower().split())
        score = len(tokens & title_tokens) * 3 + len(tokens & content_tokens)
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: -x[0])
    return json.dumps([s[1] for s in scored[:3]], ensure_ascii=False)


def consistency_check(
    span_text: Annotated[str, "The entity text to look up, e.g. 'species'"],
) -> str:
    """Find how similar spans were labelled in existing seed examples for consistency."""
    span_lower = span_text.lower().strip()
    matches = []
    for rel, examples in _SEED_EXAMPLES.items():
        for ex in examples:
            e1t = ex.get("e1", {}).get("text", "").lower()
            e2t = ex.get("e2", {}).get("text", "").lower()
            matched = None
            if span_lower in e1t or e1t in span_lower:
                matched = ("e1", ex["e1"])
            elif span_lower in e2t or e2t in span_lower:
                matched = ("e2", ex["e2"])
            if matched:
                role, ent = matched
                matches.append({
                    "relation": rel, "matched_entity": role,
                    "entity_text": ent["text"], "entity_type": ent["type"],
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
Return a JSON object conforming exactly to this schema:
{json.dumps(AnnotatorOutput.model_json_schema(), indent=2)}

Output rules:
- Return JSON only. Do not include commentary, markdown, or <think> blocks.
- Keep every reasoning field brief and evidence-based.
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
Work through the annotation systematically in this order:

1. **Guideline violations** — for each entity label, call guideline_search with the span text \
   and its proposed type. Check whether the guideline’s step-by-step decision tree supports \
   the chosen category. Flag any label that contradicts the guideline rules.

2. **Category confusions** — look for common misclassifications:
   - BIOTIC PROPERTY vs ABIOTIC PROPERTY (check the modified noun, not the adjective)
   - BIOTIC ENTITY vs BIOTIC COLLECTIVE ENTITY (individual/taxon vs assemblage)
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

## Output
Return a JSON object conforming exactly to this schema:
{json.dumps(CriticOutput.model_json_schema(), indent=2)}

TERMINATION RULE: If your "disagreements" list is empty and "missing_annotations" is empty,
you MUST end your entire response with the word TERMINATE on its own line. Do not ask follow-up
questions or summarise — just output the JSON and then TERMINATE.

Output rules:
- Return JSON only, optionally followed by TERMINATE on its own line.
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
1. Agreement between Annotator and Critic → accept (high confidence).
2. Disagreement → check guideline via tools, apply tiebreaker:
   "choose the category describing the primary referent in the sentence."
3. Genuine ambiguity → flag for human review, pick the safer label.

## Output

You have to return a JSON object conforming exactly to this schema:
{json.dumps(AdjudicatorOutput.model_json_schema(), indent=2)}

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

def _critic_is_satisfied(msg: dict) -> bool:
    """
    Return True when the Critic's message signals no remaining issues.

    Checks for explicit TERMINATE first, then falls back to parsing the
    Critic's JSON: if both 'disagreements' and 'missing_annotations' are
    empty lists, the conversation can stop without the LLM saying TERMINATE.
    """
    content = msg.get("content", "") or ""
    if "TERMINATE" in content:
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
                         entity_types_list=entity_types_list or _FALLBACK_ENTITY_TYPES)

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
            # Annotator receives Critic messages: stop when Critic is satisfied,
            # whether or not it remembered to write "TERMINATE".
            is_termination_msg=_critic_is_satisfied,
        )

        self.critic = ConversableAgent(
            name="Critic",
            system_message=_critic_system_msg(critic_guideline, entity_schema_str, relation_schema),
            llm_config=critic_llm,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
        )

        self.adjudicator = ConversableAgent(
            name="Adjudicator",
            system_message=_adjudicator_system_msg(critic_guideline, entity_schema_str, relation_schema),
            llm_config=adjudicator_llm,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
        )

        # A tool executor proxy (no LLM, just runs tool calls)
        self.tool_executor = ConversableAgent(
            name="ToolExecutor",
            llm_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
        )

        # ── Register tools ───────────────────────────────────
        # Annotator: coverage-focused tools (entity schema, relation schema, guideline)
        _register_tools_on_agents(self.annotator, self.tool_executor, ANNOTATOR_TOOL_FUNCTIONS)
        # Critic: QA-focused tools (adds consistency_check against seed examples)
        _register_tools_on_agents(self.critic, self.tool_executor, CRITIC_TOOL_FUNCTIONS)
        # Adjudicator: same QA set as the Critic
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

        # ── Phase 1: Annotator ↔ Critic deliberation ─────────
        # The Critic initiates by forwarding the task to the Annotator.
        # This way the Annotator produces its annotation exactly once (as its
        # first reply), avoiding the duplication that arose when Phase 1a ran a
        # separate chat and then the same annotation text was re-sent as the
        # opening message of the deliberation.
        #
        # Turn counts (Critic is initiator):
        #   Turn 1 : Annotator generates annotation
        #   Turn 2 : Critic reviews  → says TERMINATE if no critical issues
        #   Turn 3+: Annotator / Critic exchange until TERMINATE or max_turns
        logger.info("Phase 1: Annotator ↔ Critic deliberation")

        # max_turns = 1 (annotation) + max_rounds review cycles × 2 (critic+annotator)
        deliberation_max_turns = self.max_rounds * 2 + 1

        chat_result = self.critic.initiate_chat(
            recipient=self.annotator,
            message=task_msg,
            max_turns=deliberation_max_turns,
        )

        # Collect messages from the deliberation (skip the initiating task_msg
        # from the Critic — it is just a relay and adds no annotation content).
        # Tool calls and their results are attached to the proposing message.
        deliberation_messages = _collect_messages_with_tools(
            chat_result.chat_history, skip_first=True
        )
        record.messages = deliberation_messages
        # Each deliberation "round" = one Annotator turn + one Critic turn
        record.rounds_used = len(deliberation_messages) // 2

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
            max_turns=3,  # tool_executor turn + adjudicator reply + up to 1 extra tool round
        )

        # Extract final output from adjudicator's last message
        adj_output = self._extract_last_json(adj_result.chat_history)

        # Collect adjudicator messages (including tool calls to/from ToolExecutor)
        adj_messages = _collect_messages_with_tools(
            adj_result.chat_history, skip_first=True  # skip the task message we sent
        )
        record.messages.extend(adj_messages)

        # ── Parse final annotations ───────────────────────────
        if adj_output:
            try:
                parsed_adj = AdjudicatorOutput.model_validate(adj_output)
                record.final_entities = parsed_adj.final_entities
                record.final_relations = [r.to_relation_annotation() for r in parsed_adj.final_relations]
            except ValidationError as e:
                logger.warning(f"AdjudicatorOutput validation error: {e}")

        # ── Compute agreement ─────────────────────────────────
        # Use the last round only (earlier disputes may have been resolved).
        # Denominator = Annotator's proposed items (entities + relations).
        # Disputes  = Critic's disagreements + missing_annotations.
        # Score     = (proposed − disagreed) / (proposed + missing).
        #
        # This avoids the granularity mismatch of the free-text
        # "agreements" list (which might pack multiple items into one
        # string) and correctly penalises for missing annotations.

        # Find the last Annotator and last Critic messages
        last_annotator_out = None
        last_critic_out = None
        for m in reversed(deliberation_messages):
            if m["agent"] == "Annotator" and last_annotator_out is None:
                last_annotator_out = self._parse_annotator_output(m.get("content", ""))
            elif m["agent"] == "Critic" and last_critic_out is None:
                last_critic_out = self._parse_critic_output(m.get("content", ""))
            if last_annotator_out is not None and last_critic_out is not None:
                break

        if last_annotator_out and last_critic_out:
            n_proposed = len(last_annotator_out.entities) + len(last_annotator_out.relations)
            n_disagreed = len(last_critic_out.disagreements)
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
    def _parse_critic_output(text: str) -> Optional[CriticOutput]:
        raw = MultiAgentAnnotator._try_parse_json(text)
        if raw is None:
            return None
        try:
            return CriticOutput.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"CriticOutput validation failed: {e}")
            return None

    @staticmethod
    def _parse_adjudicator_output(text: str) -> Optional[AdjudicatorOutput]:
        raw = MultiAgentAnnotator._try_parse_json(text)
        if raw is None:
            return None
        try:
            return AdjudicatorOutput.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"AdjudicatorOutput validation failed: {e}")
            return None

    @staticmethod
    def _parse_annotator_output(text: str) -> Optional[AnnotatorOutput]:
        raw = MultiAgentAnnotator._try_parse_json(text)
        if raw is None:
            return None
        try:
            return AnnotatorOutput.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"AnnotatorOutput validation failed: {e}")
            return None

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
    )

    records = annotator.annotate_batch(sentences, output_path=args.output)

    stats = analyze_disagreements(records)
    print(f"\n{'='*60}")
    print(f"  BATCH ANALYSIS")
    print(f"{'='*60}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
