"""
Dependency-parser relation-candidate net (Option A) — an OPTIONAL recall aid.

Given a sentence and the entities the Annotator proposed, this finds pairs of
entities that are syntactically connected (short shortest-dependency-path
between their head tokens) but that currently have NO relation between them,
and surfaces them to the Critic as candidates to check. The parser only
*proposes* candidate pairs plus a weak relation hint; the LLM agents +
schema_lookup make the final call.

Design notes
------------
* Parser = candidate GENERATOR, not a relation classifier. AFFECTS-vs-CAUSES
  style decisions are left to the agents; we only say "these two spans are
  syntactically linked, and here is a soft hint".
* Reuses the shortest-dependency-path idea from
  ``src/relation_extraction/baselines/relation_baseline_dep.py`` but drops that
  baseline's type-bucket classifier (it targets the OLD schema).
* Graceful degradation: if spaCy or the model is unavailable the hinter reports
  ``available == False`` and returns nothing — the pipeline runs unchanged.
* spaCy ``en_core_web_*`` models use the ClearNLP / OntoNotes dependency scheme
  (prep / pobj / nsubj / dobj / poss / compound / conj / appos), NOT Universal
  Dependencies — the cue heuristics below are written against THAT scheme.

Entities/relations are accessed by duck-typing so this module stays decoupled
from the pipeline's pydantic models: an entity needs ``text`` and
``entity_type`` (``start``/``end`` optional); a relation needs ``e1_text`` and
``e2_text``. Both attribute objects and plain dicts work.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Cue vocabularies (spaCy ClearNLP dep + coarse POS lemmas)
# ─────────────────────────────────────────────────────────────
# The connecting preposition / verb along the dependency path is a weak prior
# for which of the 7 MoBiKo relations might hold. These are HINTS only.

_LOC_PREPS = {
    "in", "at", "on", "within", "across", "near", "throughout", "along",
    "around", "inside", "among", "amongst", "over", "beneath", "under",
    "above", "between",
}
_CAUSAL_VERBS = {
    "cause", "lead", "result", "produce", "generate", "induce", "trigger",
    "yield",
}
_EFFECT_VERBS = {
    "affect", "influence", "reduce", "increase", "decrease", "alter", "impact",
    "hinder", "contribute", "enhance", "limit", "threaten", "change", "modify",
    "regulate", "control", "shape", "drive", "damage", "improve", "promote",
    "suppress", "facilitate", "disrupt", "diminish",
}
_COMPARE_VERBS = {"compare", "exceed", "differ", "resemble", "outnumber", "surpass"}
_POSSESS_VERBS = {"have", "contain", "include", "comprise", "possess", "consist", "encompass"}

# Path made only of these edges ⇒ the two entities are coordinate siblings /
# in apposition, not (usually) related to each other directly.
_COORD_EDGES = {"conj", "cc", "punct", "appos", "preconj"}


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict or as an attribute; None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _norm_nospace(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


# ─────────────────────────────────────────────────────────────
# Dependency-path helpers (adapted from relation_baseline_dep.py)
# ─────────────────────────────────────────────────────────────

def _shortest_dep_path(t1, t2):
    """
    Shortest dependency path between two spaCy tokens.
    Returns (path_tokens: List[token], length: int|None, lca_token|None).
    ``path_tokens`` is [t1, …, lca, …, t2] and includes the LCA exactly once.

    NOTE: compare tokens with ``==`` / ``!=``, never ``is``. spaCy builds a
    fresh ``Token`` wrapper on every ``.head`` access, so identity checks fail
    even for the same underlying token (root check via ``is`` would spin
    forever). ``==`` compares (doc, index) and is what the loops below rely on.
    """
    if t1 == t2:
        return [t1], 0, t1

    anc1, cur, d = {}, t1, 0
    while cur is not None:
        anc1[cur] = d
        d += 1
        if cur.head == cur:
            break
        cur = cur.head

    anc2, cur, d = {}, t2, 0
    while cur is not None:
        anc2[cur] = d
        d += 1
        if cur.head == cur:
            break
        cur = cur.head

    lca, best = None, None
    for tok, d1 in anc1.items():
        if tok in anc2:
            total = d1 + anc2[tok]
            if best is None or total < best:
                best, lca = total, tok
    if lca is None:
        return [], None, None

    path1, cur = [], t1
    while cur != lca:
        path1.append(cur)
        cur = cur.head
    path1.append(lca)

    path2, cur = [], t2
    while cur != lca:
        path2.append(cur)
        cur = cur.head

    return path1 + list(reversed(path2)), best, lca


class DependencyRelationHinter:
    """
    Lazy-loading wrapper around a spaCy dependency parser that emits
    missing-relation candidate pairs. Construct once and reuse.
    """

    def __init__(
        self,
        model: str = "en_core_web_trf",
        max_dep_distance: int = 4,
        max_candidates: int = 12,
        fallback_models: Tuple[str, ...] = ("en_core_web_sm",),
    ) -> None:
        self.model_name = model
        self.max_dep_distance = max_dep_distance
        self.max_candidates = max_candidates
        self._fallback_models = tuple(m for m in fallback_models if m and m != model)
        self._nlp = None
        self.loaded_model: Optional[str] = None
        self.load_error: Optional[str] = None
        self._doc_key: Optional[str] = None
        self._doc = None
        self._load()

    # ── model loading ────────────────────────────────────────
    def _load(self) -> None:
        try:
            import spacy
        except Exception as exc:  # spaCy not installed
            self.load_error = f"spaCy import failed: {exc}"
            logger.warning("DependencyRelationHinter disabled — %s", self.load_error)
            return
        for name in (self.model_name, *self._fallback_models):
            try:
                # We only need the parser (+ tagger for POS); NER/lemmatizer stay
                # on because en_core_web_trf shares one transformer anyway.
                self._nlp = spacy.load(name)
                self.loaded_model = name
                if name != self.model_name:
                    logger.warning(
                        "Requested spaCy model %r unavailable; using fallback %r.",
                        self.model_name, name,
                    )
                logger.info("DependencyRelationHinter loaded spaCy model: %s", name)
                return
            except Exception as exc:
                self.load_error = f"could not load spaCy model {name!r}: {exc}"
                continue
        logger.warning("DependencyRelationHinter disabled — %s", self.load_error)

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def _parse(self, sentence: str):
        """Parse with a 1-entry cache so re-reviews of the same sentence reuse it."""
        if self._doc_key == sentence and self._doc is not None:
            return self._doc
        self._doc = self._nlp(sentence)
        self._doc_key = sentence
        return self._doc

    # ── entity span → head token ─────────────────────────────
    def _head_token(self, doc, ent):
        """
        Map an entity to its syntactic head token. Uses char offsets when the
        entity carries them; otherwise matches on concatenated token text
        (offset-independent — the Annotator's entities have no offsets yet at
        Critic time).
        """
        start, end = _get(ent, "start"), _get(ent, "end")
        if start is not None and end is not None:
            span = doc.char_span(int(start), int(end), alignment_mode="expand")
            if span is not None:
                return span.root

        target = _norm_nospace(_get(ent, "text") or "")
        if not target:
            return None
        n = len(doc)
        for i in range(n):
            acc = ""
            for j in range(i, min(i + 12, n)):
                acc += doc[j].text.lower()
                acc_ns = _norm_nospace(acc)
                if acc_ns == target:
                    return doc[i:j + 1].root
                if len(acc_ns) > len(target):
                    break
        return None

    # ── relation-type hint from the connecting path ──────────
    @staticmethod
    def _is_coordination(path_tokens, lca) -> bool:
        # ``!=`` not ``is not`` — see note in _shortest_dep_path about spaCy
        # rebuilding Token wrappers on each access.
        edges = [t.dep_ for t in path_tokens if t != lca]
        return bool(edges) and all(e in _COORD_EDGES for e in edges)

    @staticmethod
    def _relation_hint(path_tokens, coordinated: bool):
        """Return (cue_str, suggested_relation|None, note|None)."""
        if coordinated:
            return ("coordination (conj/appos)", "COMPARES_TO",
                    "coordinated items — consider COMPARES_TO, or a shared "
                    "relation of each to a third entity")

        lemmas = [t.lemma_.lower() for t in path_tokens]
        preps = [t.lemma_.lower() for t in path_tokens if t.dep_ == "prep"]
        verbs = [t.lemma_.lower() for t in path_tokens if t.pos_ == "VERB"]
        deps = {t.dep_ for t in path_tokens}

        if ("than" in preps or any(v in _COMPARE_VERBS for v in verbs)
                or "versus" in lemmas or "vs" in lemmas):
            return ("comparative", "COMPARES_TO", None)
        for v in verbs:
            if v in _CAUSAL_VERBS:
                return (f"verb '{v}'", "CAUSES", None)
        if "due" in lemmas and "to" in preps:
            return ("'due to'", "CAUSES", None)
        for v in verbs:
            if v in _EFFECT_VERBS:
                return (f"verb '{v}'", "AFFECTS", None)
        for p in preps:
            if p in _LOC_PREPS:
                return (f"prep '{p}'", "LOCATED_IN", None)
        for v in verbs:
            if v in _POSSESS_VERBS:
                return (f"verb '{v}'", "HAS_PROPERTY or IS_PART_OF", None)
        if "of" in preps:
            return ("prep 'of'", "HAS_PROPERTY or IS_PART_OF", None)
        if "poss" in deps:
            return ("possessive", "HAS_PROPERTY or IS_PART_OF", None)
        if "compound" in deps:
            return ("compound", "HAS_PROPERTY or IS_PART_OF", None)

        if verbs:
            return (f"verb '{verbs[0]}'", None, None)
        if preps:
            return (f"prep '{preps[0]}'", None, None)
        return ("direct dependency", None, None)

    @staticmethod
    def _suggest_order(ei, hi, ej, hj, suggested_rel: Optional[str], lca):
        """
        Best-effort (e1, e2) ordering — a SUGGESTION only.
        Governor-first by default; subject-first for verb-mediated effect/cause.
        Returns (e1, e2).
        """
        subj_deps = {"nsubj", "nsubjpass", "agent", "csubj"}
        if suggested_rel in {"AFFECTS", "CAUSES"}:
            # subject side is the agent → e1
            if hi.dep_ in subj_deps and hj.dep_ not in subj_deps:
                return ei, ej
            if hj.dep_ in subj_deps and hi.dep_ not in subj_deps:
                return ej, ei
        # governor (ancestor) first  (``==`` not ``is``; see _shortest_dep_path)
        if hi == lca:
            return ei, ej
        if hj == lca:
            return ej, ei
        return ei, ej

    # ── main entry point ─────────────────────────────────────
    def find_missing_candidates(
        self,
        sentence: str,
        entities: List[Any],
        existing_relations: Optional[List[Any]] = None,
    ) -> List[dict]:
        """
        Return candidate relation pairs (entities syntactically close but with
        no relation between them), sorted by dependency distance ascending and
        capped at ``max_candidates``.
        """
        if not self.available or not entities or len(entities) < 2:
            return []

        existing_relations = existing_relations or []
        linked = set()
        for r in existing_relations:
            e1, e2 = _norm(_get(r, "e1_text")), _norm(_get(r, "e2_text"))
            if e1 and e2:
                linked.add(frozenset((e1, e2)))

        doc = self._parse(sentence)
        heads = [self._head_token(doc, e) for e in entities]

        candidates: List[dict] = []
        seen_pairs = set()
        for i in range(len(entities)):
            hi = heads[i]
            if hi is None:
                continue
            for j in range(i + 1, len(entities)):
                hj = heads[j]
                if hj is None:
                    continue
                ei, ej = entities[i], entities[j]
                ti, tj = _norm(_get(ei, "text")), _norm(_get(ej, "text"))
                if not ti or not tj or ti == tj:
                    continue
                pair_key = frozenset((ti, tj))
                if pair_key in linked or pair_key in seen_pairs:
                    continue

                path, dist, lca = _shortest_dep_path(hi, hj)
                if dist is None or dist == 0 or dist > self.max_dep_distance:
                    continue

                coordinated = self._is_coordination(path, lca)
                cue, suggested, note = self._relation_hint(path, coordinated)
                e1, e2 = self._suggest_order(ei, hi, ej, hj, suggested, lca)

                seen_pairs.add(pair_key)
                candidates.append({
                    "e1_text": _get(e1, "text"),
                    "e1_type": _get(e1, "entity_type"),
                    "e2_text": _get(e2, "text"),
                    "e2_type": _get(e2, "entity_type"),
                    "dep_distance": dist,
                    "cue": cue,
                    "suggested_relation": suggested,
                    "note": note,
                    "dep_path": [t.text for t in path],
                })

        candidates.sort(key=lambda c: c["dep_distance"])
        return candidates[: self.max_candidates]

    # ── render for the Critic prompt ─────────────────────────
    def format_block(self, candidates: List[dict]) -> str:
        if not candidates:
            return ""
        header = (
            f"Dependency-parser relation candidates (model: {self.loaded_model}).\n"
            f"These entity pairs are syntactically connected (dependency path "
            f"≤ {self.max_dep_distance}) but have NO relation between them in the "
            f"current annotation. This is a RECALL aid: the parser only proposes "
            f"pairs, it does NOT decide relations. For EACH pair, check whether a "
            f"schema-valid relation actually holds:\n"
            f"  - if yes, flag it as a missing relation (correct type + direction, "
            f"confirmed with schema_lookup);\n"
            f"  - if no, ignore it.\n"
            f"The \"suggests\" value is a syntax-only guess — do not trust it over "
            f"the guideline/schema. Ignore any pair that is not a genuine relation."
        )
        lines = [header, ""]
        for n, c in enumerate(candidates, 1):
            arrow = "→"
            head = (
                f'  {n}. "{c["e1_text"]}" ({c["e1_type"]}) {arrow} '
                f'"{c["e2_text"]}" ({c["e2_type"]})'
            )
            bits = [f'cue: {c["cue"]}']
            if c["suggested_relation"]:
                bits.append(f'suggests: {c["suggested_relation"]}')
            bits.append(f'dep-dist: {c["dep_distance"]}')
            detail = "       " + " | ".join(bits)
            if c.get("note"):
                detail += f'\n       note: {c["note"]}'
            lines.append(head)
            lines.append(detail)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Standalone smoke test:  python dependency_relations.py
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Smoke-test the relation-candidate net.")
    ap.add_argument("--model", default="en_core_web_trf")
    ap.add_argument("--max-dep-distance", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=12)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    hinter = DependencyRelationHinter(
        model=args.model,
        max_dep_distance=args.max_dep_distance,
        max_candidates=args.max_candidates,
    )
    print("available:", hinter.available, "| model:", hinter.loaded_model,
          "| error:", hinter.load_error)
    if not hinter.available:
        raise SystemExit(1)

    sentence = (
        "However, the limited information on the effects of overexploitation on "
        "the current status and community composition of wildlife hinders "
        "effective conservation efforts."
    )
    entities = [
        {"text": "overexploitation", "entity_type": "ANTHROPOGENIC PROCESS"},
        {"text": "status", "entity_type": "BIOTIC PROPERTY"},
        {"text": "community composition", "entity_type": "BIOTIC PROPERTY"},
        {"text": "wildlife", "entity_type": "BIOTIC ENTITY"},
        {"text": "conservation efforts", "entity_type": "ANTHROPOGENIC PROCESS"},
    ]
    relations = [
        {"e1_text": "wildlife", "e2_text": "status"},
    ]
    cands = hinter.find_missing_candidates(sentence, entities, relations)
    print(f"\n{len(cands)} candidate(s):\n")
    print(hinter.format_block(cands))
    print("\nraw:", json.dumps(cands, ensure_ascii=False, indent=2))