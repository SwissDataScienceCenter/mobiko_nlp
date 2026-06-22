"""
Guideline amender — closed-loop guideline building (RQ-C / spec 11.4).

Turns Layer-4 confusion patterns into concrete, operational guideline
amendments. For each recurring (annotator_label → critic_label) confusion, an
LLM is asked to produce:

    {
      "original_rule":      the G_i text this concerns (or "NONE — gap"),
      "proposed_amendment": the new/edited rule text,
      "decision_test":      a MANDATORY concrete, sentence-level test,
      "rationale":          why, citing confusion frequency + examples
    }

The single most important design choice (per the spec): an amendment is only
accepted if its ``decision_test`` is *operational* — a checkable if-then rule
that names the competing labels and points at an observable cue. Vague additions
("consider the context", "use judgment") are rejected and redrafted, because
vague rules do not reduce friction; operational tests do.

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

# Source of truth for endpoints (gpt4o / qwen3-35B-vllm). Importing the pipeline
# module is heavier than ideal but avoids duplicating endpoint URLs that drift.
from multi_agent_annotation_ag2 import MODEL_ENDPOINTS
from deliberation_history import reconstruct_timeline

# The canonical, closed set of entity types. It lives under src/ (one level up
# from this package dir), which the flat sibling-imports above don't put on the
# path — so add it explicitly before importing the schema.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resources_updated.entity_schema import SCHEMA_BIODIV_LIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_GUIDELINE = _THIS_DIR / "MoBiKo_label_guidance_v3.md"

# Load .env (OPENAI_API_KEY / OPEN_WEB_UI_API_KEY) so the amender picks up keys
# the same way as the rest of the pipeline, without manually exporting them.
if load_dotenv:
    load_dotenv()                                          # search cwd and parents
    load_dotenv(_THIS_DIR.parents[1] / ".env", override=False)  # repo-root .env regardless of cwd

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

Produce ONE guideline amendment that would prevent this confusion. Return a
single JSON object with EXACTLY these keys:
{{
  "original_rule": "the exact G_i text this amendment concerns, or 'NONE — gap' if no rule covers it",
  "proposed_amendment": "the new or edited rule text to add to the guideline",
  "decision_test": "<MANDATORY> {_DECISION_TEST_RUBRIC[pclass]}",
  "illustrative_example": "ONE short PARAPHRASED example sentence with the span marked in [brackets] and its label, demonstrating the decision_test — see GENERALISATION below",
  "rationale": "why this helps, citing the {pattern['count']}x frequency and the examples above"
}}

The decision_test is the most important field. It MUST be a concrete, sentence-
level, checkable rule that names the competing labels and points at an observable
cue. Do NOT use vague phrasing like "consider the context" or "use judgment".{allowed_block}

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
# Post-hoc leakage verification (closed-loop safety net)
# ─────────────────────────────────────────────────────────────

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
    corpus_ngrams: Dict[Tuple[str, ...], str] = {}
    for s in corpus_sentences:
        words = re.findall(r"\w+", (s or "").lower())
        for i in range(len(words) - n + 1):
            corpus_ngrams.setdefault(tuple(words[i:i + n]), s)

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


# ─────────────────────────────────────────────────────────────
# Model call
# ─────────────────────────────────────────────────────────────

def _make_client(model_key: str):
    import openai
    endpoint = MODEL_ENDPOINTS.get(model_key)
    if not endpoint:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_ENDPOINTS)}")
    api_key = (
        endpoint["api_key"]
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_WEB_UI_API_KEY")
    )
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


def write_outputs(
    amendments: List[Dict[str, Any]],
    guideline_text: str,
    guideline_path: Path,
    out_dir: Path,
    today: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted = [a for a in amendments if a.get("status") == "accepted"]

    # 1) machine-readable amendments
    json_path = out_dir / "amendments.json"
    json_path.write_text(
        json.dumps(amendments, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2) new versioned guideline (G_i untouched) — accepted amendments appended
    #    as a clearly-marked section so the whole doc remains a usable guideline.
    new_guideline = _next_version_path(guideline_path, out_dir)
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
    new_guideline.write_text("\n".join(appended), encoding="utf-8")

    # 3) human-reviewable Markdown
    md_path = out_dir / "amendments_review.md"
    lines = [f"# Guideline amendment proposals ({today})", "",
             f"Base guideline: `{guideline_path.name}`  →  new version: `{new_guideline.name}`",
             f"Accepted: {len(accepted)} / {len(amendments)} patterns", ""]
    for a in amendments:
        status = a.get("status", "?")
        mark = {"accepted": "✅", "rejected": "❌", "malformed": "⚠️"}.get(status, "•")
        lines.append(f"## {mark} {a['pattern']}  ({a['count']}×, class={a['pattern_class']}, {a['n_examples']} ex.)")
        if status == "accepted":
            lines += [
                f"- **Concerns:** {a['original_rule']}",
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
    parser.add_argument("--model", choices=["gpt4o", "qwen3-35B-vllm"], default="qwen3-35B-vllm",
                        help="Amender model. gpt4o sends examples to OpenAI; qwen3-35B-vllm stays on-cluster.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top confusions to amend.")
    parser.add_argument("--examples-per-pattern", type=int, default=5)
    parser.add_argument("--max-redrafts", type=int, default=2,
                        help="Redraft attempts when the decision_test fails the operationality gate.")
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

    paths = write_outputs(amendments, guideline_text, args.guideline, args.out_dir, today)
    n_ok = sum(1 for a in amendments if a["status"] == "accepted")
    logger.info("Done: %d/%d accepted.\n  amendments: %s\n  new guideline: %s\n  review: %s",
                n_ok, len(amendments), paths["json"], paths["guideline"], paths["review"])

    # Programmatic safety net: the amended guideline must not contain verbatim
    # corpus text (else the closed loop would leak labels into G_{i+1}).
    corpus_sentences = [(r.get("sentence") or "") for r in records]
    leaks = verify_no_corpus_leak(amendments, corpus_sentences, n=args.leak_ngram)
    if leaks:
        logger.error("CORPUS LEAK DETECTED in amended guideline (%d) — NOT safe for the closed loop:", len(leaks))
        for lk in leaks:
            logger.error('  [%s] "%s…" ← from: "%s"', lk["pattern"], lk["ngram"], lk["source_sentence"][:80])
        raise SystemExit(2)
    logger.info("Leak check passed: no %d-gram from the corpus appears in the amended guideline.", args.leak_ngram)


if __name__ == "__main__":
    main()