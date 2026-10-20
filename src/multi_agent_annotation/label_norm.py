"""
Single source of truth for text and entity-label normalisation.

Both halves of the paper are scored from the same run files through two
different stacks — RQ1 via ``evaluation/eval_layer1_output``, RQ2 via
``evaluation/eval_layer2_correlation`` — and until this module existed each one
carried its own copy of the sentence normaliser and its own alias table, while
the real formatting canonicaliser lived a third place again, in
``multi_agent_annotation_ag2``. Three copies agreed by luck rather than by
construction; an audit found the label paths had already diverged, with RQ1
scoring canonicalised labels and RQ2 scoring them as shipped.

Everything now imports from here. Stdlib only, no package imports, so the
pipeline, the loop modules and the eval scripts can all take it without
dragging anything heavy behind it.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ───────────────────────────── text ─────────────────────────────

def normalize_text(text: Optional[str]) -> str:
    """Lowercase, collapse whitespace. The key for sentences and span text.

    Previously defined three times over: eval_layer1_output._normalize,
    eval_layer2_correlation._normalize and deliberation_history.norm. They were
    byte-identical, which is exactly the state in which one of them gets
    "improved" alone.

    Non-string input becomes "". Callers feed this raw model-generated JSON: a
    Critic in one run emitted a relation OBJECT where a span string belongs
    (3 fields in 2 of 155 records), which raised AttributeError deep inside the
    analysis and took the whole run out of the comparison. One malformed field
    should cost that field, not the run — the span simply fails to match, which
    is the correct outcome for a target that names nothing.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


# ──────────────────────────── labels ────────────────────────────

_ENTITY_TYPE_ALIASES: Dict[str, str] = {
    # v1 -> v2 schema renames.
    "BIOTIC COLLECTIVE ENTITY": "BIOTIC ENTITY",
    "ABIOTIC COLLECTIVE ENTITY": "ABIOTIC ENTITY",
    # Observed misspellings. Listed EXPLICITLY rather than fuzzy-matched: a
    # near-miss repair that guesses would risk assigning the wrong label, which
    # is worse than scoring 0 on a visibly broken one.
    "ANTHROGENIC ENTITY": "ANTHROPOGENIC ENTITY",
    "ANTHROGENIC PROCESS": "ANTHROPOGENIC PROCESS",
    "ANTHROGENIC PROPERTY": "ANTHROPOGENIC PROPERTY",
}

# The schema list canonicalize_entity_type validates against when the
# orchestrator has not populated _ALL_ENTITY_TYPES (i.e. in every eval script).
# QUALITATIVE/QUANTITATIVE are PROPERTY, not ENTITY: the human gold types 339
# spans that way, and an earlier typo here flagged every one as off-schema.
_FALLBACK_ENTITY_TYPES: List[str] = [
    "ABIOTIC ENTITY", "ABIOTIC PROCESS", "ABIOTIC PROPERTY",
    "ANTHROPOGENIC ENTITY", "ANTHROPOGENIC PROCESS", "ANTHROPOGENIC PROPERTY",
    "BIOTIC ENTITY", "BIOTIC PROCESS", "BIOTIC PROPERTY",
    "CONCEPT", "SPATIAL ENTITY", "SPATIAL PROPERTY", "TEMPORAL ENTITY", "TEMPORAL PROPERTY",
    "QUALITATIVE PROPERTY", "QUANTITATIVE PROPERTY",
]

# Populated in place by the orchestrator at run time. Mutated with
# .clear()/.update(), never rebound, so importers share the one set.
_ALL_ENTITY_TYPES: set = set()

# Labels that survive canonicalisation but are still not in the schema, counted
# per process so a run can report them instead of hiding them.
_OFF_SCHEMA_LABELS: Dict[str, int] = {}

# The relation vocabulary, populated in place by the orchestrator from the
# relation guide (Relation schema.csv). Mutated with .clear()/.update(), never
# rebound, so importers share the one set. Empty until a run loads the guide,
# and canonicalize_relation_type then validates against nothing rather than
# against a stale hard-coded list.
_ALL_RELATION_TYPES: set = set()

# Same bookkeeping as _OFF_SCHEMA_LABELS, for relation labels.
_OFF_SCHEMA_RELATIONS: Dict[str, int] = {}

RELATION_PREFIX = "RELATION"


# Punctuation is anything that is neither a word character nor whitespace.
_SPAN_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _singular(token: str) -> str:
    """Strip an English plural ending from one token.

    Deliberately crude: it only has to map the singular and plural of one word
    onto the SAME string, and it is applied identically to both annotators, so
    an odd stem ("species" -> "specy") is harmless — both sides get it. The
    guards keep short tokens ("m", "is") and -ss words ("biomass") intact.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"                      # studies -> study
    if token.endswith(("sses", "xes", "ches", "shes", "zes")):
        return token[:-2]                            # processes -> process
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]                            # assemblages -> assemblage
    return token


def canonical_span_text(text: Optional[str]) -> str:
    """The surface form of a span with the differences annotators asked to
    ignore taken out (Mark and Davnah, 2026-10): punctuation dropped, so
    "well-being." equals "well-being", and every token reduced to its singular,
    so "assemblages" equals "assemblage". Punctuation becomes a space before
    whitespace is collapsed, so "well-being" and "well being" also agree.

    Used by the 'normalized' endpoint-matching mode, which additionally requires
    the two spans to overlap in the sentence — this is a test of "the same
    mention, trimmed or inflected differently", never of two mentions that
    merely share words.
    """
    if not isinstance(text, str):
        return ""
    words = _SPAN_PUNCT.sub(" ", text.lower()).split()
    return " ".join(_singular(w) for w in words)


def canonicalize_entity_type(raw: Optional[str], record_off_schema: bool = True) -> str:
    """Normalise a label's FORMATTING. Never changes which label was meant.

    "BIOTIC_ENTITY", "biotic  entity", " Biotic-Entity " -> "BIOTIC ENTITY"

    A label that is STILL not in the schema after canonicalisation is returned
    unchanged and counted in ``_OFF_SCHEMA_LABELS`` — deliberately not dropped.
    Dropping spans would silently move recall, could orphan relations that
    reference them, and the human gold itself carries a handful of off-schema
    labels, so "off-schema" does not imply "wrong". Surface it, don't delete it.
    """
    if not raw:
        return ""
    text = str(raw).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip().upper()
    if not text:
        return ""
    text = _ENTITY_TYPE_ALIASES.get(text, text)
    # Relation pseudo-labels (e.g. "RELATION:CAUSAL") are a different vocabulary
    # and are not validated against the entity schema.
    if text.startswith(RELATION_PREFIX):
        return text
    valid = _ALL_ENTITY_TYPES or set(_FALLBACK_ENTITY_TYPES)

    # Missing separator: "QUANTITATIVEPROPERTY" -> "QUANTITATIVE PROPERTY". Only
    # applied when inserting one space yields EXACTLY ONE schema label, so it is
    # a deterministic repair rather than a guess.
    if text not in valid and " " not in text:
        split = [f"{text[:i]} {text[i:]}" for i in range(1, len(text))
                 if f"{text[:i]} {text[i:]}" in valid]
        if len(split) == 1:
            logger.info("Recovered missing separator: %r -> %r", text, split[0])
            text = split[0]

    if text not in valid:
        if not record_off_schema:
            # A probe, not an emitted label: callers that are only asking "is this
            # an entity type?" (the Critic's polymorphic disagreement fields) must
            # not inflate the run's off-schema report.
            return text
        _OFF_SCHEMA_LABELS[text] = _OFF_SCHEMA_LABELS.get(text, 0) + 1
        if _OFF_SCHEMA_LABELS[text] == 1:  # log each novel label once per process
            logger.warning(
                "Off-schema entity type %r (canonicalised from %r) — kept, not "
                "dropped. Check the prompt or add it to the schema.", text, raw
            )
    return text


def normalize_label(raw: Optional[str], canon: bool = True) -> str:
    """Label key for scoring.

    canon=True   full formatting canonicalisation (the paper's primary
                 convention, and what the pipeline applies at emit time).
    canon=False  case/whitespace and the alias table only — the labels as the
                 model shipped them, modulo the v1->v2 rename. This is what
                 ``--as-is`` reports, and it exists so the formatting repair is
                 visible rather than assumed.
    """
    if canon:
        return canonicalize_entity_type(raw)
    upper = (raw or "").strip().upper()
    return _ENTITY_TYPE_ALIASES.get(upper, upper)


def off_schema_label_report() -> Dict[str, int]:
    """Off-schema labels seen so far this process, most frequent first."""
    return dict(sorted(_OFF_SCHEMA_LABELS.items(), key=lambda kv: -kv[1]))


def reset_off_schema_labels() -> None:
    _OFF_SCHEMA_LABELS.clear()


def canonicalize_relation_type(raw: Optional[str], record_off_schema: bool = True) -> str:
    """Normalise a RELATION label's formatting. Never changes which one was meant.

    "has property", "Has-Property", "HAS  PROPERTY" -> "HAS_PROPERTY".

    Relation names are underscore-joined (HAS_PROPERTY, LOCATED_IN), the opposite
    convention to entity types, which are space-joined. That is the only real
    difference from canonicalize_entity_type; the reasoning is identical, and so
    is the refusal to drop: a relation whose label is still unknown after
    canonicalisation is returned unchanged and counted in _OFF_SCHEMA_RELATIONS.
    Dropping it would remove a triplet a human might have accepted, and the
    relation vocabulary is still being revised.
    """
    if not raw:
        return ""
    text = str(raw).replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip().upper()
    if not text:
        return ""
    text = text.replace(" ", "_")

    if not _ALL_RELATION_TYPES:
        # No vocabulary loaded (eval scripts, or a corpus with no relations):
        # normalise formatting and validate against nothing.
        return text

    if text not in _ALL_RELATION_TYPES:
        if not record_off_schema:
            # Probes, not emitted labels: relation_search normalises an arbitrary
            # user query to try an exact-name hit, and the guide loader normalises
            # its own rows. Counting either would fill the run's off-schema report
            # with things no agent ever annotated.
            return text
        _OFF_SCHEMA_RELATIONS[text] = _OFF_SCHEMA_RELATIONS.get(text, 0) + 1
        if _OFF_SCHEMA_RELATIONS[text] == 1:  # log each novel label once
            logger.warning(
                "Off-schema relation %r (canonicalised from %r) — kept, not "
                "dropped. The relation guide is the closed vocabulary; check "
                "that the agent called relation_search before emitting this.",
                text, raw
            )
    return text


def off_schema_relation_report() -> Dict[str, int]:
    """Off-schema relation labels seen so far this process, most frequent first."""
    return dict(sorted(_OFF_SCHEMA_RELATIONS.items(), key=lambda kv: -kv[1]))


def reset_off_schema_relations() -> None:
    _OFF_SCHEMA_RELATIONS.clear()
