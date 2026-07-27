"""
Guideline amender — closed-loop guideline building (RQ-C / spec 11.4).

Turns Layer-4 confusion patterns into concrete, operational guideline
amendments. For each recurring (annotator_label → critic_label) confusion, an
LLM is asked to produce:

    {
      "original_rule":      the G_i text this concerns (or "NONE — gap"),
      "target_section":     the G_i heading the rule belongs under (or "NEW"),
      "proposed_amendment": the new/edited rule text,
      "decision_test":      a MANDATORY concrete, sentence-level test,
      "rationale":          why, citing confusion frequency + examples
    }

The single most important design choice (per the spec): an amendment is only
accepted if its ``decision_test`` is *operational* — a checkable if-then rule
that names the competing labels and points at an observable cue. Vague additions
("consider the context", "use judgment") are rejected and redrafted, because
vague rules do not reduce friction; operational tests do.

INTEGRATION (how G_{i+1} is produced). Accepted amendments are **not** dumped
into a dated appendix. Each one is routed to the section of G_i it concerns
(via its verbatim ``original_rule``, its ``target_section``, or a heading naming
one of the competing labels), and that section is then **rewritten** by the model
so the new rule is woven into the prose: superseded sentences are edited or
replaced, overlapping rules merged, contradictions resolved. Sections no
amendment touches are copied through byte-identically — the rewrite can never
silently drop or reword an untouched part of the guideline, because only the
bodies of routed sections are ever regenerated. Amendments that match no section
land in a single stable "Disambiguation rules" section that is itself rewritten
on later iterations (so it never becomes a pile of dated blocks).

A rewritten section is accepted only if it keeps the existing content (word-count
floor), still states each amendment's decision test as a checkable rule naming
the competing labels, and carries no changelog phrasing. On repeated failure the
section falls back to a deterministic in-section merge; ``--integration append``
restores the old append-only behaviour.

The original guideline is NEVER modified. The amender writes a new versioned
guideline (G_{i+1}), an amendments JSON, and a human-reviewable Markdown doc.

Input:
  --layer34-json   : output/eval_reports/multi_agent_layer34_deliberation.json
                     (provides the confusion patterns + counts; has NO examples)
  --deliberations  : the deliberation JSONL(s) — mined for concrete examples,
                     since the Layer-4 JSON stores counts only.
  --guideline      : current G_i (.md), default MoBiKo_label_guidance_v3.md

Usage:
  python guideline_amender.py \
      --layer34-json ./output/eval_reports/multi_agent_layer34_deliberation.json \
      --deliberations ./data/auto_annotated/datademo_manually_labeled1.jsonl \
      --model qwen3-35B-vllm --top-k 8 --out-dir output/guideline_amendments
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv optional — fall back to exported env vars
    load_dotenv = None

_THIS_DIR = Path(__file__).resolve().parent          # …/multi_agent_annotation/loop
_PKG_ROOT = _THIS_DIR.parent                         # …/multi_agent_annotation (shared core)
_SRC = _PKG_ROOT.parent                              # …/src (for resources_updated)
_REPO_ROOT = _SRC.parent                             # repo root
# Make the shared core (multi_agent_annotation_ag2, deliberation_history) at the
# package root and resources_updated under src/ importable before we use them.
for _p in (_SRC, _PKG_ROOT, _PKG_ROOT / "loop", _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Source of truth for endpoints (gpt4o / qwen3-35B-vllm / swissai-*). Importing
# the pipeline module is heavier than ideal but avoids duplicating endpoint URLs
# that drift.
from multi_agent_annotation_ag2 import MODEL_ENDPOINTS
from deliberation_history import reconstruct_timeline
from resources_updated.entity_schema import SCHEMA_BIODIV_LIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_GUIDELINE = _PKG_ROOT / "MoBiKo_label_guidance_v4.md"

# Load .env (OPENAI_API_KEY / OPEN_WEB_UI_API_KEY) so the amender picks up keys
# the same way as the rest of the pipeline, without manually exporting them.
if load_dotenv:
    load_dotenv()                                          # search cwd and parents
    load_dotenv(_REPO_ROOT / ".env", override=False)       # repo-root .env regardless of cwd

# The closed set of valid entity types (normalised), used to positively
# recognise entity-type confusions rather than treating them as a catch-all.
_ENTITY_TYPES = {re.sub(r"\s+", " ", t.strip().upper()) for t in SCHEMA_BIODIV_LIST}

# Critic "labels" that mean "this should not have been annotated / the relation
# is not valid" rather than a competing entity type.
_SCOPE_LABELS = {"IGNORE", "IGNORED", "NOT ANNOTATED", "DROP", "REMOVE"}
_RELATION_INVALID_LABELS = {
    "INVALID", "INVALID_RELATION", "INVALID RELATION", "NO_RELATION",
    "NO RELATION", "RELATION INVALID PER SCHEMA",
}


# ─────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────

def _norm_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().upper())


def _looks_like_relation(label: str) -> bool:
    """Relations are SCREAMING_SNAKE (IS_AFFECTING, LOCATED_IN, HAS_PROPERTY)."""
    return "_" in label and " " not in label.strip()


def classify_pattern(annotator: str, critic: str) -> str:
    """
    One of: 'entity_type' | 'relation_validity' | 'annotation_scope'.

    Determines which decision_test rubric to apply: entity-type confusions get a
    typing test, relation issues a schema/validity test, scope issues an
    is-this-annotatable test.
    """
    a, c = _norm_label(annotator), _norm_label(critic)
    if c in _SCOPE_LABELS:
        return "annotation_scope"
    if c in _RELATION_INVALID_LABELS or _looks_like_relation(a) or _looks_like_relation(c):
        return "relation_validity"
    # Entity-type confusions are between two members of the closed entity-type
    # set. If both labels are valid types this is unambiguously entity_type; we
    # still fall back to entity_type otherwise (best effort for off-schema labels).
    if a in _ENTITY_TYPES and c in _ENTITY_TYPES:
        return "entity_type"
    return "entity_type"


# ─────────────────────────────────────────────────────────────
# Inputs: patterns + examples
# ─────────────────────────────────────────────────────────────

def load_confusion_patterns(layer34_json: Path, top_k: int) -> List[Dict[str, Any]]:
    """Read top_confusions_all_rounds → [{annotator, critic, count}], top_k of them."""
    data = json.loads(Path(layer34_json).read_text(encoding="utf-8"))
    confusions = data.get("top_confusions_all_rounds", [])
    return [
        {
            "annotator": _norm_label(c["annotator"]),
            "critic": _norm_label(c["critic"]),
            "count": int(c.get("count", 0)),
        }
        for c in confusions[:top_k]
    ]


def _load_records(paths: List[Path]) -> List[dict]:
    records: List[dict] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def collect_examples(
    records: List[dict],
    annotator: str,
    critic: str,
    max_examples: int,
) -> List[Dict[str, str]]:
    """
    Mine the deliberation records for concrete instances of the
    (annotator → critic) confusion: the spans where the Critic disputed an
    ``annotator_label == annotator`` with ``proposed_label == critic``.

    Returns up to ``max_examples`` {sentence, span, severity, explanation}.
    """
    a, c = _norm_label(annotator), _norm_label(critic)
    examples: List[Dict[str, str]] = []
    seen: set = set()
    for rec in records:
        sentence = (rec.get("sentence") or "").strip()
        tl = reconstruct_timeline(rec)
        for cr in tl["critic_rounds"]:
            for d in cr["disagreements"]:
                if _norm_label(d.get("annotator_label", "")) != a:
                    continue
                if _norm_label(d.get("proposed_label", "")) != c:
                    continue
                span = (d.get("target") or "").strip()
                key = (sentence, span)
                if key in seen:
                    continue
                seen.add(key)
                examples.append({
                    "sentence": sentence,
                    "span": span,
                    "severity": (d.get("severity") or "").strip(),
                    "explanation": (d.get("explanation") or "").strip(),
                })
                if len(examples) >= max_examples:
                    return examples
    return examples


# ─────────────────────────────────────────────────────────────
# Guideline structure: split into sections, route amendments, re-render
# ─────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

# Where amendments that match no existing section go. Deliberately undated and
# stable: on later iterations it is just another section, so its rules get
# rewritten in place instead of stacking up as one dated block per iteration.
_FALLBACK_HEADING = "## Disambiguation rules"


def split_sections(md: str) -> List[Dict[str, Any]]:
    """Split a Markdown guideline into ``{heading, level, title, body}`` sections.

    The heading line is kept verbatim so re-rendering preserves it exactly. Text
    before the first heading becomes a leading section with ``heading == ""``.
    Headings inside fenced code blocks are not treated as headings.
    """
    raw: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {"heading": "", "level": 0, "title": "", "lines": []}
    in_fence = False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            raw.append(cur)
            cur = {"heading": line.rstrip(), "level": len(m.group(1)),
                   "title": m.group(2).strip(), "lines": []}
        else:
            cur["lines"].append(line)
    raw.append(cur)

    sections: List[Dict[str, Any]] = []
    for s in raw:
        body = "\n".join(s["lines"]).strip("\n")
        if not s["heading"] and not body.strip():
            continue                       # empty preamble — nothing to keep
        sections.append({"heading": s["heading"], "level": s["level"],
                         "title": s["title"], "body": body})
    return sections


def render_sections(sections: List[Dict[str, Any]]) -> str:
    """Inverse of :func:`split_sections` (blank lines around headings normalised)."""
    out: List[str] = []
    for s in sections:
        if s["heading"]:
            out.append(s["heading"])
            out.append("")
        body = (s["body"] or "").strip("\n")
        if body:
            out.append(body)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _norm_heading(title: str) -> str:
    """Normalise a heading/label for matching: no ``#``, numbering, escapes, case."""
    t = re.sub(r"^#+\s*", "", (title or "").strip())
    t = t.replace("\\", "")
    t = re.sub(r"^\d+(\.\d+)*[.)]?\s*", "", t)        # "4.1 ", "5. ", "3) "
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())        # drops *emphasis*, `code`, —
    return re.sub(r"\s+", " ", t).strip()


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def find_section_index(sections: List[Dict[str, Any]], needle: str) -> Optional[int]:
    """Index of the section whose heading matches ``needle`` (exact, then containment)."""
    key = _norm_heading(needle)
    if not key:
        return None
    for i, s in enumerate(sections):
        if s["heading"] and _norm_heading(s["title"]) == key:
            return i
    for i, s in enumerate(sections):
        if not s["heading"]:
            continue
        title = _norm_heading(s["title"])
        # Containment, but never on a stub ("Case A") that would swallow anything.
        if len(title) >= 4 and len(key) >= 4 and (key in title or title in key):
            return i
    return None


def route_amendment(
    sections: List[Dict[str, Any]],
    amendment: Dict[str, Any],
) -> Optional[int]:
    """Decide which section of G_i an accepted amendment belongs in.

    Strongest signal first: the verbatim ``original_rule`` text (whichever
    section actually contains it), then the model's ``target_section`` heading,
    then a heading that names one of the two competing labels. ``None`` means
    "no home in this guideline" — the caller sends it to the fallback section.
    """
    original = (amendment.get("original_rule") or "").strip()
    if original and "NONE" not in original.upper() and len(original) >= 25:
        needle = _norm_ws(original)
        for i, s in enumerate(sections):
            if needle in _norm_ws(s["body"]):
                return i

    target = (amendment.get("target_section") or "").strip()
    if target and target.upper() not in {"NEW", "NONE", "N/A", "GAP"}:
        idx = find_section_index(sections, target)
        if idx is not None:
            return idx

    left, sep, right = (amendment.get("pattern") or "").partition("→")
    for label in ([left, right] if sep else []):
        label = label.strip()
        if not label:
            continue
        idx = find_section_index(sections, label)
        if idx is not None:
            return idx
    return None


def ensure_fallback_section(sections: List[Dict[str, Any]]) -> int:
    """Index of the stable fallback section, appending it if not present."""
    idx = find_section_index(sections, _FALLBACK_HEADING)
    if idx is not None:
        return idx
    sections.append({"heading": _FALLBACK_HEADING, "level": 2,
                     "title": _FALLBACK_HEADING.lstrip("# ").strip(), "body": ""})
    return len(sections) - 1


# ─────────────────────────────────────────────────────────────
# Prompting
# ─────────────────────────────────────────────────────────────

_DECISION_TEST_RUBRIC = {
    "entity_type": (
        "A concrete typing test that distinguishes the two ENTITY TYPES via an "
        "observable property of the span, phrased as checkable questions mapping "
        "to a label, e.g. \"Can you point to it on a map? → SPATIAL ENTITY; is it "
        "a physical substance/material? → ABIOTIC ENTITY\"."
    ),
    "relation_validity": (
        "A concrete test for WHEN the relation holds and is schema-valid, phrased "
        "as checkable conditions mapping to a verdict, e.g. \"Does the sentence "
        "state that X physically changes/impacts Y? → IS_AFFECTING is valid; "
        "otherwise → no relation\"."
    ),
    "annotation_scope": (
        "A concrete test for WHETHER the span should be annotated at all, phrased "
        "as checkable questions, e.g. \"Is it a bare numeric measurement with no "
        "ecological referent? → do NOT annotate; does it name a measured ecological "
        "attribute? → annotate as the property type\"."
    ),
}

_SYSTEM_MSG = (
    "You are a guideline engineer for a biodiversity entity/relation annotation "
    "scheme (MoBiKo). You improve the labelling guideline by turning observed "
    "annotator-vs-critic confusions into precise, operational rules. You return "
    "ONLY a single JSON object and nothing else."
)


def build_amendment_prompt(
    pattern: Dict[str, Any],
    examples: List[Dict[str, str]],
    guideline_text: str,
    pclass: str,
    redraft_reason: Optional[str] = None,
) -> List[Dict[str, str]]:
    examples_block = "\n".join(
        f'  {i+1}. sentence: "{ex["sentence"]}"\n'
        f'     span: "{ex["span"]}"  severity: {ex["severity"] or "?"}\n'
        f'     critic note: {ex["explanation"] or "(none)"}'
        for i, ex in enumerate(examples)
    ) or "  (no concrete examples were found for this pattern)"

    # For entity-type confusions, pin the model to the closed set of valid types
    # so the decision_test can only route to labels that exist in the schema.
    allowed_block = ""
    if pclass == "entity_type":
        allowed_block = (
            "\n\nThe ONLY valid entity types are (use these exact labels, never invent new ones):\n"
            + ", ".join(SCHEMA_BIODIV_LIST)
        )

    redraft = ""
    if redraft_reason:
        redraft = (
            "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED. Reason: "
            f"{redraft_reason}\nFix the decision_test so it is fully operational."
        )

    # The headings of G_i, so target_section can only name a section that exists.
    headings = [s["heading"] for s in split_sections(guideline_text) if s["heading"]]
    headings_block = "\n".join(f"  {h}" for h in headings) or "  (the guideline has no headings)"

    user = f"""\
The annotation pipeline shows a recurring confusion:
  Annotator labelled it: {pattern['annotator']}
  Critic proposed:        {pattern['critic']}
  Frequency:              {pattern['count']}× across all rounds
  Pattern class:          {pclass}

Concrete examples from the deliberations:
{examples_block}

Current guideline (G_i):
\"\"\"
{guideline_text}
\"\"\"

Its section headings are:
{headings_block}

Produce ONE guideline amendment that would prevent this confusion. Return a
single JSON object with EXACTLY these keys:
{{
  "original_rule": "the exact G_i text this amendment concerns, copied VERBATIM (one sentence or short passage), or 'NONE — gap' if no rule covers it",
  "target_section": "the heading from the list above under which this rule belongs (copy it exactly), or 'NEW' if none fits",
  "proposed_amendment": "the rewritten rule text — see INTEGRATION below",
  "decision_test": "<MANDATORY> {_DECISION_TEST_RUBRIC[pclass]}",
  "illustrative_example": "ONE short PARAPHRASED example sentence with the span marked in [brackets] and its label, demonstrating the decision_test — see GENERALISATION below",
  "rationale": "why this helps, citing the {pattern['count']}x frequency and the examples above"
}}

The decision_test is the most important field. It MUST be a concrete, sentence-
level, checkable rule that names the competing labels and points at an observable
cue. Do NOT use vague phrasing like "consider the context" or "use judgment".{allowed_block}

INTEGRATION: your amendment will be MERGED INTO the guideline text, not appended
to the end of it. proposed_amendment must therefore read as guideline prose that
can stand in the place of original_rule — a self-contained rule in the guideline's
own voice. Do NOT write it as a note about a change ("the rule should now say…",
"in addition to the above…", "this amends section X"), and do NOT restate the
surrounding section. When original_rule is real G_i text, proposed_amendment is
its REPLACEMENT and must keep whatever of it is still correct.

GENERALISATION (critical): proposed_amendment and decision_test become permanent
guideline text that will be used to re-annotate the SAME corpus. They MUST be
general rules — do NOT quote, paraphrase, or embed the example sentences/spans
above in them; those examples are evidence for YOU only. The illustrative_example
DOES go into the guideline, so it must be a PARAPHRASE — reword/invent a fresh
sentence that conveys the same point but is NOT identical to any example above
(change the wording, entities, and numbers). The guideline must never reveal how
to label any specific corpus sentence.{redraft}"""

    return [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user", "content": user},
    ]


# ─────────────────────────────────────────────────────────────
# decision_test operationality gate (the core of the spec)
# ─────────────────────────────────────────────────────────────

_VAGUE_PHRASES = (
    "depends on context", "depends on the context", "use judgment",
    "use your judgment", "as appropriate", "case by case", "case-by-case",
    "it varies", "in general", "consider the context", "consider context",
)


def validate_decision_test(
    decision_test: str,
    labels: List[str],
) -> Tuple[bool, str]:
    """
    Accept only operational decision tests. Returns (ok, reason_if_not).

    An operational test must:
      * exist and be non-trivial,
      * contain an if-then / "→ LABEL" decision structure,
      * name at least one competing label,
      * pose an observable, checkable cue (a question or a concrete predicate),
      * not be dominated by vague hedging.
    """
    t = (decision_test or "").strip()
    if len(t) < 15:
        return False, "decision_test is absent or too short to be operational"

    has_rule = ("→" in t) or ("->" in t) or bool(re.search(r"\bthen\b", t, re.I))
    if not has_rule:
        return False, "decision_test has no concrete if-then / '→ LABEL' decision rule"

    label_tokens = {
        tok.upper()
        for lbl in labels
        for tok in re.findall(r"[A-Za-z_]+", lbl or "")
        if len(tok) > 2
    }
    if label_tokens and not any(tok in t.upper() for tok in label_tokens):
        return False, "decision_test does not name the competing label(s)"

    has_cue = ("?" in t) or bool(
        re.search(r"\b(is|are|can|does|do|has|have|appears|refers|names|describes|contains)\b", t, re.I)
    )
    if not has_cue:
        return False, "decision_test poses no observable, checkable cue (phrase it as a question)"

    low = t.lower()
    if any(v in low for v in _VAGUE_PHRASES) and "?" not in t:
        return False, "decision_test relies on vague hedging instead of an operational cue"

    return True, ""


# ─────────────────────────────────────────────────────────────
# Integration: rewrite the affected sections of G_i into G_{i+1}
# ─────────────────────────────────────────────────────────────

_REWRITE_SYSTEM_MSG = (
    "You are a guideline engineer for a biodiversity entity/relation annotation "
    "scheme (MoBiKo). You revise ONE section of the labelling guideline so that "
    "approved rule changes are integrated into its text. You preserve everything "
    "the changes do not supersede, and you return only the rewritten section "
    "between the requested markers."
)

_SECTION_START = "<<<SECTION>>>"
_SECTION_END = "<<<END>>>"

# Phrases that betray a changelog/patch-note voice rather than guideline prose.
_CHANGELOG_MARKERS = (
    "auto-amendment", "auto-amendments", "changelog", "this amendment",
    "the amendment", "previous version", "previously, the guideline",
    "as amended", "revision history", "was updated to", "has been updated",
    "new rule:", "added rule", "updated rule",
)


def _amendment_block(amendments: List[Dict[str, Any]]) -> str:
    """Render the approved changes for one section as the rewrite prompt's payload."""
    out: List[str] = []
    for i, a in enumerate(amendments, 1):
        original = (a.get("original_rule") or "").strip()
        supersedes = (
            "no existing rule covers this — it is a gap to fill"
            if (not original or "NONE" in original.upper())
            else f'"{original}"'
        )
        out.append(
            f"{i}. Concerns the confusion {a.get('pattern', '?')} "
            f"(seen {a.get('count', '?')}×).\n"
            f"   Supersedes: {supersedes}\n"
            f"   Rule to state: {(a.get('proposed_amendment') or '').strip()}\n"
            f"   Decision test (MUST survive, keep the → arrows): "
            f"{(a.get('decision_test') or '').strip()}"
            + (f"\n   Illustrative example: {a['illustrative_example'].strip()}"
               if (a.get("illustrative_example") or "").strip() else "")
        )
    return "\n".join(out)


def build_rewrite_prompt(
    heading: str,
    body: str,
    amendments: List[Dict[str, Any]],
    redraft_reason: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Prompt to rewrite ONE section so the approved changes are woven into it."""
    current = body.strip() or "(this section is currently empty)"
    redraft = ""
    if redraft_reason:
        redraft = (
            "\n\nYOUR PREVIOUS REWRITE WAS REJECTED. Reason: "
            f"{redraft_reason}\nProduce a corrected rewrite of the same section."
        )

    user = f"""\
Rewrite ONE section of the MoBiKo labelling guideline so that the approved
changes below are part of its text.

SECTION HEADING (it stays as it is — do not output it, do not rename it):
{heading or "(the section before the first heading)"}

CURRENT SECTION TEXT:
\"\"\"
{current}
\"\"\"

APPROVED CHANGES TO INTEGRATE ({len(amendments)}):
{_amendment_block(amendments)}

Rewrite the section so that:
  * each change is woven into the prose where it belongs — EDIT or REPLACE the
    sentences it supersedes; never leave the old wording standing and restate
    the new rule after it;
  * every decision test appears in the section as a checkable if-then rule that
    names the competing labels and keeps its "→ LABEL" arrows;
  * anything in the current text that contradicts a change is rewritten, not
    kept alongside it; overlapping or duplicated rules are merged into one;
  * ALL existing content the changes do not supersede is preserved — keep it
    verbatim or near-verbatim, do not summarise, shorten, reorder for its own
    sake, or drop unrelated rules, examples, or sub-headings;
  * sub-headings (####, lists, tables) inside the section are kept and reused;
  * the result reads as one coherent guideline section written in the
    guideline's own voice.

Never write it as a patch note: no "Amendment", "New rule", "Update", "(added
2026-…)", no changelog, no mention of annotators, critics, confusions, counts,
frequencies, or of the fact that anything was changed. A reader must not be able
to tell which sentences are new.

Rules must be GENERAL: do not quote or paraphrase any sentence from the annotated
corpus, and keep any example illustrative and invented.

Output ONLY the rewritten section body — markdown, WITHOUT the heading line —
between the markers, and nothing else:

{_SECTION_START}
…rewritten section body…
{_SECTION_END}{redraft}"""

    return [
        {"role": "system", "content": _REWRITE_SYSTEM_MSG},
        {"role": "user", "content": user},
    ]


def _extract_marked_block(text: str) -> Optional[str]:
    """Pull the rewritten body out of ``<<<SECTION>>> … <<<END>>>``.

    Falls back to a fenced code block when the model drops the markers; returns
    ``None`` when neither is present (the caller then redrafts).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start = cleaned.find(_SECTION_START)
    if start != -1:
        rest = cleaned[start + len(_SECTION_START):]
        end = rest.find(_SECTION_END)
        block = rest if end == -1 else rest[:end]
        block = block.strip("\n").strip()
        if block:
            return block
    m = re.search(r"```(?:markdown|md)?\n(.*?)```", cleaned, flags=re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip("\n").strip()
    return None


def _strip_repeated_heading(body: str, heading: str) -> str:
    """Drop the heading line if the model re-emitted it at the top of the body."""
    if not heading:
        return body
    lines = body.splitlines()
    if lines and _norm_heading(lines[0]) == _norm_heading(heading):
        return "\n".join(lines[1:]).strip("\n")
    return body


def validate_rewritten_section(
    new_body: str,
    old_body: str,
    amendments: List[Dict[str, Any]],
    *,
    min_retention: float = 0.8,
) -> Tuple[bool, str]:
    """Accept a rewritten section only if it integrated without losing content.

    Checks, in order: non-trivial output · no content deletion (word count must
    stay at ``min_retention`` of the original, since integration adds rules) ·
    every amendment's decision test present as a checkable rule naming its
    labels · no changelog phrasing.
    """
    body = (new_body or "").strip()
    if len(body) < 40:
        return False, "rewritten section is empty or too short"

    old_words = len(re.findall(r"\w+", old_body or ""))
    new_words = len(re.findall(r"\w+", body))
    if old_words and new_words < min_retention * old_words:
        return False, (
            f"rewrite dropped existing content ({new_words} words vs {old_words} "
            "before) — keep everything the changes do not supersede"
        )

    for a in amendments:
        labels = [p.strip() for p in (a.get("pattern") or "").split("→")]
        ok, why = validate_decision_test(body, [l for l in labels if l])
        if not ok:
            return False, f"for {a.get('pattern', '?')}: {why}"

    low = body.lower()
    for marker in _CHANGELOG_MARKERS:
        if marker in low:
            return False, (
                f'rewrite reads as a changelog (contains "{marker}") — write it '
                "as plain guideline prose"
            )
    return True, ""


def _merge_amendment_into_body(body: str, a: Dict[str, Any]) -> str:
    """Deterministic fallback merge: add one rule to the END of its own section.

    Used only when the model fails to produce an acceptable rewrite. Still far
    better than the old global appendix — the rule lands in the section it
    concerns, with no dated wrapper — but it is not integrated into the prose,
    so ``rewrite_log`` records it for review.
    """
    parts = [body.rstrip()] if body.strip() else []
    parts.append((a.get("proposed_amendment") or "").strip())
    test = (a.get("decision_test") or "").strip()
    if test:
        parts.append(f"**Decision test:** {test}")
    example = (a.get("illustrative_example") or "").strip()
    if example:
        parts.append(f"**Example:** {example}")
    return "\n\n".join(p for p in parts if p)


def rewrite_guideline(
    guideline_text: str,
    accepted: List[Dict[str, Any]],
    *,
    generate_fn,
    max_retries: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    """Produce G_{i+1} by rewriting the sections the accepted amendments touch.

    ``generate_fn(messages) -> str`` is injected (same contract as
    :func:`amend_pattern`) so this is testable without hitting a model. Sections
    with no routed amendment are copied through untouched, so the structure of
    G_i and every unrelated rule survive by construction.

    Returns ``(new_guideline_text, log)`` where ``log`` reports, per section, the
    routed patterns and whether it was rewritten or fell back to a merge.
    """
    sections = split_sections(guideline_text)
    if not accepted:
        return render_sections(sections), {
            "sections": [], "n_sections": 0, "n_rewritten": 0, "n_fallback": 0,
        }

    routed: Dict[int, List[Dict[str, Any]]] = {}
    n_fallback_section = 0
    for a in accepted:
        idx = route_amendment(sections, a)
        if idx is None:
            idx = ensure_fallback_section(sections)
            n_fallback_section += 1
        routed.setdefault(idx, []).append(a)

    log: Dict[str, Any] = {
        "sections": [], "n_sections": len(routed), "n_rewritten": 0,
        "n_fallback": 0, "n_unrouted_amendments": n_fallback_section,
    }

    for idx in sorted(routed):
        section = sections[idx]
        ams = routed[idx]
        old_body = section["body"]
        entry = {
            "heading": section["heading"] or "(preamble)",
            "patterns": [a.get("pattern", "?") for a in ams],
            "status": "fallback_merge",
            "attempts": 0,
            "reason": "",
        }

        reason: Optional[str] = None
        for attempt in range(1, max_retries + 2):     # 1 rewrite + max_retries redrafts
            messages = build_rewrite_prompt(section["heading"], old_body, ams,
                                            redraft_reason=reason)
            raw = generate_fn(messages)
            candidate = _extract_marked_block(raw)
            entry["attempts"] = attempt
            if candidate is None:
                reason = (f"output did not contain the {_SECTION_START} … "
                          f"{_SECTION_END} block")
                entry["reason"] = reason
                continue
            candidate = _strip_repeated_heading(candidate, section["heading"])
            ok, why = validate_rewritten_section(candidate, old_body, ams)
            if ok:
                section["body"] = candidate
                entry.update({"status": "rewritten", "reason": ""})
                log["n_rewritten"] += 1
                break
            reason = why
            entry["reason"] = why
        else:
            body = old_body
            for a in ams:
                body = _merge_amendment_into_body(body, a)
            section["body"] = body
            log["n_fallback"] += 1
            logger.warning("  integration fell back to a merge for %s (%s)",
                           entry["heading"], entry["reason"])

        log["sections"].append(entry)

    return render_sections(sections), log


# ─────────────────────────────────────────────────────────────
# Post-hoc leakage verification (closed-loop safety net)
# ─────────────────────────────────────────────────────────────

def _corpus_ngram_index(corpus_sentences: List[str], n: int) -> Dict[Tuple[str, ...], str]:
    """{word n-gram → the sentence it came from} over the annotated corpus."""
    index: Dict[Tuple[str, ...], str] = {}
    for s in corpus_sentences:
        words = re.findall(r"\w+", (s or "").lower())
        for i in range(len(words) - n + 1):
            index.setdefault(tuple(words[i:i + n]), s)
    return index


def verify_no_corpus_leak(
    amendments: List[Dict[str, Any]],
    corpus_sentences: List[str],
    n: int = 7,
) -> List[Dict[str, str]]:
    """
    Verify that no guideline-bound amendment text reproduces a verbatim chunk of
    any corpus sentence. Checks only the text that is written INTO the guideline
    (proposed_amendment + decision_test + illustrative_example of accepted
    amendments), so G_i's own content is never flagged.

    A leak = a contiguous run of ``n`` words shared between an amendment and a
    corpus sentence (n-gram match tolerates paraphrase while catching real
    copying; short common phrases below n words are ignored).

    Returns a list of {pattern, ngram, source_sentence}; empty == clean.
    """
    corpus_ngrams = _corpus_ngram_index(corpus_sentences, n)

    leaks: List[Dict[str, str]] = []
    for a in amendments:
        if a.get("status") != "accepted":
            continue
        text = " ".join(filter(None, [
            a.get("proposed_amendment", ""),
            a.get("decision_test", ""),
            a.get("illustrative_example", ""),
        ]))
        words = re.findall(r"\w+", text.lower())
        for i in range(len(words) - n + 1):
            ng = tuple(words[i:i + n])
            if ng in corpus_ngrams:
                leaks.append({
                    "pattern": a.get("pattern", "?"),
                    "ngram": " ".join(ng),
                    "source_sentence": corpus_ngrams[ng],
                })
                break  # one report per amendment is enough
    return leaks


def verify_rewrite_no_corpus_leak(
    old_guideline: str,
    new_guideline: str,
    corpus_sentences: List[str],
    n: int = 7,
) -> List[Dict[str, str]]:
    """Leak-check the prose the integration pass actually wrote.

    Only sections whose body CHANGED are checked, and within them only n-grams
    that are NOT already somewhere in G_i — so text the rewrite merely carried
    over (already vetted when it entered the guideline) can never be re-flagged,
    while wording the rewrite invented is still caught.
    """
    old_bodies: Dict[str, str] = {}
    for s in split_sections(old_guideline):
        old_bodies[s["heading"]] = s["body"]
    old_ngrams = set(_corpus_ngram_index([old_guideline], n))

    corpus_ngrams = _corpus_ngram_index(corpus_sentences, n)
    leaks: List[Dict[str, str]] = []
    for s in split_sections(new_guideline):
        body = s["body"]
        if not body.strip() or _norm_ws(old_bodies.get(s["heading"], "")) == _norm_ws(body):
            continue
        words = re.findall(r"\w+", body.lower())
        for i in range(len(words) - n + 1):
            ng = tuple(words[i:i + n])
            if ng in corpus_ngrams and ng not in old_ngrams:
                leaks.append({
                    "pattern": f"rewritten section {s['heading'] or '(preamble)'}",
                    "ngram": " ".join(ng),
                    "source_sentence": corpus_ngrams[ng],
                })
                break
    return leaks


# ─────────────────────────────────────────────────────────────
# Model call
# ─────────────────────────────────────────────────────────────

def _make_client(model_key: str):
    import openai
    endpoint = MODEL_ENDPOINTS.get(model_key)
    if not endpoint:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_ENDPOINTS)}")
    print('ENDPOINT', endpoint)
    api_key = endpoint.get("api_key") or os.getenv(endpoint.get("api_key_env", ""))


    if not api_key:
        raise ValueError(f"API key required for {model_key} (set OPENAI_API_KEY / OPEN_WEB_UI_API_KEY).")
    client = openai.OpenAI(base_url=endpoint["base_url"], api_key=api_key)
    return client, endpoint["model"]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Lenient JSON extraction: strip <think> blocks, take the last brace span."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def generate_amendment(client, model: str, messages: List[Dict[str, str]]) -> str:
    """One streamed chat completion → accumulated raw text. Isolated for mockability.

    The on-cluster gateway requires ``stream=True`` (mirrors the main pipeline),
    so we consume the chunk stream and concatenate the content deltas.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        stream=True,
        timeout=600,
    )
    parts: List[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            parts.append(delta.content)
    return "".join(parts)


# ─────────────────────────────────────────────────────────────
# Per-pattern amendment with reject-and-redraft
# ─────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"original_rule", "proposed_amendment", "decision_test", "rationale"}


def amend_pattern(
    pattern: Dict[str, Any],
    examples: List[Dict[str, str]],
    guideline_text: str,
    *,
    generate_fn,
    max_redrafts: int = 2,
) -> Dict[str, Any]:
    """
    Produce one validated amendment for a confusion pattern, redrafting while the
    decision_test fails the operationality gate. ``generate_fn(messages) -> str``
    is injected so this is testable without hitting a model.

    Returns the amendment dict augmented with bookkeeping: ``pattern``, ``count``,
    ``pattern_class``, ``n_examples``, ``status`` (accepted|rejected|malformed),
    ``attempts``, and ``reject_reason`` when not accepted.
    """
    pclass = classify_pattern(pattern["annotator"], pattern["critic"])
    labels = [pattern["annotator"], pattern["critic"]]
    reason: Optional[str] = None
    last: Dict[str, Any] = {}

    for attempt in range(1, max_redrafts + 2):  # 1 initial + max_redrafts redrafts
        messages = build_amendment_prompt(
            pattern, examples, guideline_text, pclass, redraft_reason=reason
        )
        raw = generate_fn(messages)
        parsed = _extract_json(raw)
        if not parsed or not _REQUIRED_KEYS.issubset(parsed):
            reason = "output was not valid JSON with the four required keys"
            last = {"status": "malformed", "reject_reason": reason, "raw": raw[:500]}
            continue

        ok, why = validate_decision_test(parsed.get("decision_test", ""), labels)
        last = {
            "original_rule": parsed["original_rule"],
            "target_section": (parsed.get("target_section") or "").strip(),
            "proposed_amendment": parsed["proposed_amendment"],
            "decision_test": parsed["decision_test"],
            "illustrative_example": (parsed.get("illustrative_example") or "").strip(),
            "rationale": parsed["rationale"],
        }
        if ok:
            last["status"] = "accepted"
            last["attempts"] = attempt
            break
        reason = why
        last["status"] = "rejected"
        last["reject_reason"] = why
        last["attempts"] = attempt

    last.update({
        "pattern": f"{pattern['annotator']} → {pattern['critic']}",
        "count": pattern["count"],
        "pattern_class": pclass,
        "n_examples": len(examples),
    })
    return last


# ─────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────

def _next_version_path(guideline: Path, out_dir: Path) -> Path:
    """Bump MoBiKo_..._v3.md → v4.md (else <stem>_amended.md) inside out_dir."""
    stem = guideline.stem
    m = re.search(r"^(.*_v)(\d+)$", stem)
    new_stem = f"{m.group(1)}{int(m.group(2)) + 1}" if m else f"{stem}_amended"
    return out_dir / f"{new_stem}{guideline.suffix}"


def append_amendments(guideline_text: str, accepted: List[Dict[str, Any]], today: str) -> str:
    """LEGACY integration (``--integration append``): a dated appendix on G_i.

    Kept as an escape hatch / baseline for the rewrite pass — it never touches
    G_i's own text, at the cost of a guideline that grows one dated block per
    iteration instead of staying coherent.
    """
    appended = [guideline_text.rstrip(), "", "", f"## Auto-amendments ({today})", ""]
    for a in accepted:
        appended.append(f"### {a['pattern']}  (seen {a['count']}×)")
        appended.append(a["proposed_amendment"].strip())
        appended.append("")
        appended.append(f"**Decision test:** {a['decision_test'].strip()}")
        if a.get("illustrative_example"):
            appended.append("")
            appended.append(f"**Example (paraphrased):** {a['illustrative_example'].strip()}")
        appended.append("")
    return "\n".join(appended)


def write_outputs(
    amendments: List[Dict[str, Any]],
    guideline_text: str,
    guideline_path: Path,
    out_dir: Path,
    today: str,
    new_guideline_text: Optional[str] = None,
    rewrite_log: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write amendments.json, the next guideline version, and the review doc.

    ``new_guideline_text`` is G_{i+1} as produced by :func:`rewrite_guideline`;
    when it is ``None`` the legacy append-only integration is used instead.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted = [a for a in amendments if a.get("status") == "accepted"]

    # 1) machine-readable amendments
    json_path = out_dir / "amendments.json"
    json_path.write_text(
        json.dumps(amendments, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2) new versioned guideline (G_i on disk is untouched): the integrated
    #    rewrite, or — without one — G_i plus a dated appendix.
    new_guideline = _next_version_path(guideline_path, out_dir)
    text = new_guideline_text if new_guideline_text is not None \
        else append_amendments(guideline_text, accepted, today)
    new_guideline.write_text(text, encoding="utf-8")

    # 3) human-reviewable Markdown
    md_path = out_dir / "amendments_review.md"
    mode = "rewritten in place" if new_guideline_text is not None else "appended (legacy)"
    lines = [f"# Guideline amendment proposals ({today})", "",
             f"Base guideline: `{guideline_path.name}`  →  new version: `{new_guideline.name}`",
             f"Accepted: {len(accepted)} / {len(amendments)} patterns  ·  integration: {mode}", ""]
    if rewrite_log and rewrite_log.get("sections"):
        lines += [f"## Sections rewritten ({rewrite_log['n_rewritten']} rewritten, "
                  f"{rewrite_log['n_fallback']} fallback merge)", ""]
        for s in rewrite_log["sections"]:
            mark = "✅" if s["status"] == "rewritten" else "⚠️"
            lines.append(f"- {mark} `{s['heading']}` ← {', '.join(s['patterns'])}"
                         + (f"  _(fallback: {s['reason']})_" if s["status"] != "rewritten" else ""))
        lines.append("")
    for a in amendments:
        status = a.get("status", "?")
        mark = {"accepted": "✅", "rejected": "❌", "malformed": "⚠️"}.get(status, "•")
        lines.append(f"## {mark} {a['pattern']}  ({a['count']}×, class={a['pattern_class']}, {a['n_examples']} ex.)")
        if status == "accepted":
            lines += [
                f"- **Concerns:** {a['original_rule']}",
                f"- **Target section:** {a.get('target_section') or '(routed by label)'}",
                f"- **Amendment:** {a['proposed_amendment']}",
                f"- **Decision test:** {a['decision_test']}",
                f"- **Example (paraphrased):** {a.get('illustrative_example') or '(none)'}",
                f"- **Rationale:** {a['rationale']}",
            ]
        else:
            lines.append(f"- _{status}_ after {a.get('attempts', '?')} attempt(s): {a.get('reject_reason', '')}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": json_path, "guideline": new_guideline, "review": md_path}


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop guideline amender (spec 11.4).")
    parser.add_argument("--layer34-json", type=Path, required=True,
                        help="Layer-4 report JSON with top_confusions_all_rounds.")
    parser.add_argument("--deliberations", type=Path, nargs="+", required=True,
                        help="Deliberation JSONL file(s) — mined for confusion examples.")
    parser.add_argument("--guideline", type=Path, default=_DEFAULT_GUIDELINE,
                        help="Current guideline G_i (.md). Never modified.")
    parser.add_argument("--model", choices=list(MODEL_ENDPOINTS), default="qwen3-35B-vllm",
                        help="Amender model. gpt4o sends examples to OpenAI, swissai-* to CSCS; "
                             "qwen3-35B-vllm stays on-cluster.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top confusions to amend.")
    parser.add_argument("--examples-per-pattern", type=int, default=5)
    parser.add_argument("--max-redrafts", type=int, default=2,
                        help="Redraft attempts when the decision_test fails the operationality gate.")
    parser.add_argument("--integration", choices=["rewrite", "append"], default="rewrite",
                        help="How accepted amendments become G_{i+1}: 'rewrite' integrates each "
                             "rule into the section it concerns (default); 'append' is the legacy "
                             "dated appendix.")
    parser.add_argument("--max-rewrite-retries", type=int, default=1,
                        help="Redraft attempts per section when a rewrite drops content, loses a "
                             "decision test, or reads as a changelog.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/guideline_amendments"))
    parser.add_argument("--leak-ngram", type=int, default=7,
                        help="Word n-gram length for post-hoc corpus-leak verification of the amended guideline.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show patterns + mined examples and exit (no model calls).")
    args = parser.parse_args()

    today = datetime.date.today().isoformat()
    patterns = load_confusion_patterns(args.layer34_json, args.top_k)
    records = _load_records(args.deliberations)
    guideline_text = args.guideline.read_text(encoding="utf-8")
    logger.info("Loaded %d confusion pattern(s), %d deliberation record(s).",
                len(patterns), len(records))

    enriched = [
        (p, collect_examples(records, p["annotator"], p["critic"], args.examples_per_pattern))
        for p in patterns
    ]

    if args.dry_run:
        for p, ex in enriched:
            cls = classify_pattern(p["annotator"], p["critic"])
            print(f"\n{p['annotator']} → {p['critic']}  ({p['count']}×, class={cls}, {len(ex)} example(s))")
            for e in ex:
                print(f'  - "{e["span"]}"  in: {e["sentence"][:90]}')
        return

    client, model_name = _make_client(args.model)
    logger.info("Amender model: %s (%s)", args.model, model_name)
    gen = lambda messages: generate_amendment(client, model_name, messages)

    amendments: List[Dict[str, Any]] = []
    for p, ex in enriched:
        logger.info("Amending: %s → %s (%d×, %d examples)…",
                    p["annotator"], p["critic"], p["count"], len(ex))
        a = amend_pattern(p, ex, guideline_text, generate_fn=gen, max_redrafts=args.max_redrafts)
        logger.info("  → %s (%d attempt(s))", a["status"], a.get("attempts", 0))
        amendments.append(a)

    # Integrate the accepted rules INTO the guideline (rewrite the sections they
    # concern) rather than appending them; --integration append restores the old
    # behaviour by leaving new_text as None.
    accepted = [a for a in amendments if a.get("status") == "accepted"]
    new_text: Optional[str] = None
    rewrite_log: Optional[Dict[str, Any]] = None
    if args.integration == "rewrite" and accepted:
        logger.info("Integrating %d accepted amendment(s) into the guideline…", len(accepted))
        new_text, rewrite_log = rewrite_guideline(
            guideline_text, accepted, generate_fn=gen, max_retries=args.max_rewrite_retries)
        logger.info("  %d section(s) rewritten, %d fell back to a merge.",
                    rewrite_log["n_rewritten"], rewrite_log["n_fallback"])

    paths = write_outputs(amendments, guideline_text, args.guideline, args.out_dir, today,
                          new_guideline_text=new_text, rewrite_log=rewrite_log)
    n_ok = len(accepted)
    logger.info("Done: %d/%d accepted.\n  amendments: %s\n  new guideline: %s\n  review: %s",
                n_ok, len(amendments), paths["json"], paths["guideline"], paths["review"])

    # Programmatic safety net: the amended guideline must not contain verbatim
    # corpus text (else the closed loop would leak labels into G_{i+1}). Both the
    # drafted rules and the prose the rewrite produced are checked.
    corpus_sentences = [(r.get("sentence") or "") for r in records]
    leaks = verify_no_corpus_leak(amendments, corpus_sentences, n=args.leak_ngram)
    if new_text is not None:
        leaks += verify_rewrite_no_corpus_leak(guideline_text, new_text, corpus_sentences,
                                               n=args.leak_ngram)
    if leaks:
        logger.error("CORPUS LEAK DETECTED in amended guideline (%d) — NOT safe for the closed loop:", len(leaks))
        for lk in leaks:
            logger.error('  [%s] "%s…" ← from: "%s"', lk["pattern"], lk["ngram"], lk["source_sentence"][:80])
        raise SystemExit(2)
    logger.info("Leak check passed: no %d-gram from the corpus appears in the amended guideline.", args.leak_ngram)


if __name__ == "__main__":
    main()