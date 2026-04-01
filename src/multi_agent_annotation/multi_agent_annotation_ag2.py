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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, Any

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
# Data structures  (unchanged from vanilla version)
# ─────────────────────────────────────────────────────────────

@dataclass
class EntityAnnotation:
    text: str
    entity_type: str
    start: Optional[int] = None
    end: Optional[int] = None
    guideline_step: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


@dataclass
class RelationAnnotation:
    relation: str
    e1: EntityAnnotation
    e2: EntityAnnotation
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


@dataclass
class DeliberationRecord:
    sentence: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    final_entities: List[EntityAnnotation] = field(default_factory=list)
    final_relations: List[RelationAnnotation] = field(default_factory=list)
    agreement_score: Optional[float] = None
    rounds_used: int = 0


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


def load_guideline_from_docx(path: Path) -> List[Dict[str, str]]:
    try:
        from docx import Document
        doc = Document(str(path))
        sections, title, content = [], "Introduction", []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if (re.match(r"^Step \d", text) or re.match(r"^[IVX]+\.", text)
                    or text.startswith("General rules") or text.startswith("Handling difficult")
                    or text.startswith("Typical difficult") or text.startswith("Rule that may")
                    or text.startswith("Needs further")):
                if content:
                    sections.append({"title": title, "content": "\n".join(content)})
                title, content = text, []
            else:
                content.append(text)
        if content:
            sections.append({"title": title, "content": "\n".join(content)})
        return sections
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
Return a JSON object with keys:
- "entities": [{{"text","entity_type","guideline_step","confidence"}}]
- "relations": [{{"relation","e1_text","e1_type","e2_text","e2_type","confidence","reasoning"}}]
- "uncertain_cases": [descriptions of ambiguous spans you kept but flagged]
- "reasoning": your step-by-step thought process
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
Return a JSON object with keys:
- "agreements": [list of correct annotations]
- "disagreements": [{{"target","annotator_label","proposed_label","guideline_reference","severity","explanation"}}]
- "missing_annotations": [{{"text","entity_type","guideline_step","reasoning"}}]
- "reasoning": your detailed step-by-step review

TERMINATION RULE: If your "disagreements" list is empty and "missing_annotations" is empty,
you MUST end your entire response with the word TERMINATE on its own line. Do not ask follow-up
questions or summarise — just output the JSON and then TERMINATE.
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
Return a JSON object with keys:
- "final_entities": [{{"text","entity_type","confidence","source","guideline_step"}}]
- "final_relations": [{{"relation","e1_text","e1_type","e2_text","e2_type","confidence","reasoning"}}]
- "disagreement_resolutions": [{{"issue","decision","rationale"}}]
- "flagged_for_human_review": [genuinely ambiguous cases]

End with "TERMINATE".
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
# Orchestrator
# ─────────────────────────────────────────────────────────────

class MultiAgentAnnotator:
    """
    AG2-based multi-agent annotation system.

    Parameters
    ----------
    annotator_model, critic_model, adjudicator_model : str
        Keys into MODEL_ENDPOINTS.
    schema_path : Path
        Relation schema file.
    guideline_path : Path
        MoBiKo labelling guideline .docx.
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
        guideline_path: Optional[Path] = None,
        seeds_path: Optional[Path] = None,
        max_rounds: int = 2,
        entity_schema_str: Optional[str] = None,
        entity_types_list: Optional[list] = None,
    ):
        self.max_rounds = max_rounds

        # Load resources
        relation_schema = load_schema(schema_path) if schema_path else {}
        seeds = load_seeds(seeds_path) if seeds_path else {}
        sections = (load_guideline_from_docx(guideline_path)
                    if guideline_path and guideline_path.exists()
                    else _get_embedded_guideline())

        # Populate tool state (use canonical entity type list if provided)
        _init_tool_state(relation_schema, sections, seeds,
                         entity_types_list=entity_types_list or _FALLBACK_ENTITY_TYPES)

        guideline_summary = _build_guideline_summary(sections)

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
            system_message=_annotator_system_msg(guideline_summary, entity_schema_str, relation_schema),
            llm_config=annotator_llm,
            human_input_mode="NEVER",
            # Annotator receives Critic messages: stop when Critic is satisfied,
            # whether or not it remembered to write "TERMINATE".
            is_termination_msg=_critic_is_satisfied,
        )

        self.critic = ConversableAgent(
            name="Critic",
            system_message=_critic_system_msg(guideline_summary, entity_schema_str, relation_schema),
            llm_config=critic_llm,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
        )

        self.adjudicator = ConversableAgent(
            name="Adjudicator",
            system_message=_adjudicator_system_msg(guideline_summary, entity_schema_str, relation_schema),
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
        deliberation_messages = []
        for i, msg in enumerate(chat_result.chat_history):
            if i == 0:
                # This is the Critic's initiating task message; skip it.
                continue
            deliberation_messages.append({
                "agent": msg.get("name", msg.get("role", "unknown")),
                "content": msg.get("content", ""),
            })
        record.messages = deliberation_messages
        # Each deliberation "round" = one Annotator turn + one Critic turn
        record.rounds_used = len(deliberation_messages) // 2

        # ── Phase 2: Adjudicator resolves ─────────────────────
        logger.info("Phase 2: Adjudicator resolving")

        # Build adjudicator input from the full transcript
        transcript = "\n\n".join(
            f"**{m['agent']}**: {m['content']}" for m in deliberation_messages
        )
        adjudicator_msg = (
            f'Sentence: "{sentence}"\n\n'
            f"## Deliberation transcript\n{transcript}\n\n"
            f"Produce the final annotation."
        )

        adj_result = self.adjudicator.initiate_chat(
            recipient=self.tool_executor,
            message=adjudicator_msg,
            max_turns=3,  # tool_executor turn + adjudicator reply + up to 1 extra tool round
        )

        # Extract final output from adjudicator's last message
        adj_output = self._extract_last_json(adj_result.chat_history)
        record.messages.append({
            "agent": "Adjudicator",
            "content": adj_result.chat_history[-1].get("content", "") if adj_result.chat_history else "",
        })

        # ── Parse final annotations ───────────────────────────
        if adj_output:
            for ent in adj_output.get("final_entities", []):
                record.final_entities.append(EntityAnnotation(
                    text=ent.get("text", ""),
                    entity_type=ent.get("entity_type", ""),
                    confidence=ent.get("confidence"),
                    guideline_step=ent.get("guideline_step"),
                ))
            for rel in adj_output.get("final_relations", []):
                e1 = EntityAnnotation(text=rel.get("e1_text", ""), entity_type=rel.get("e1_type", ""))
                e2 = EntityAnnotation(text=rel.get("e2_text", ""), entity_type=rel.get("e2_type", ""))
                record.final_relations.append(RelationAnnotation(
                    relation=rel.get("relation", ""),
                    e1=e1, e2=e2,
                    confidence=rel.get("confidence"),
                    reasoning=rel.get("reasoning"),
                ))

        # ── Compute agreement ─────────────────────────────────
        # Parse critic messages to count agreements vs disagreements
        n_agree, n_disagree = 0, 0
        for msg in deliberation_messages:
            content = msg.get("content", "")
            if msg["agent"] == "Critic":
                parsed = self._try_parse_json(content)
                if parsed:
                    n_agree += len(parsed.get("agreements", []))
                    n_disagree += len(parsed.get("disagreements", []))
        total = n_agree + n_disagree
        record.agreement_score = n_agree / total if total > 0 else 1.0

        logger.info(
            f"Done: {len(record.final_entities)} entities, "
            f"{len(record.final_relations)} relations, "
            f"agreement={record.agreement_score:.2f}"
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
    def _append_jsonl(record: DeliberationRecord, path: Path):
        obj = {
            "sentence": record.sentence,
            "final_entities": [asdict(e) for e in record.final_entities],
            "final_relations": [asdict(r) for r in record.final_relations],
            "agreement_score": record.agreement_score,
            "rounds_used": record.rounds_used,
            "messages": record.messages,
        }
        with path.open("a", encoding="utf8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
