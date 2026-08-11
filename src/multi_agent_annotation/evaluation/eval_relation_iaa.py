#!/usr/bin/env python3
"""
Relation inter-annotator agreement (IAA) — the relation counterpart of
scripts/human_iaa_report.py.

Input is one merged relation file per annotator, i.e. the output of
scripts/merge_relation_annotations.py (native doc format:
{doc_id, sentences:[{text, spans, relations}]}). Use the MERGED file, never a
single re_*.jsonl snapshot: the relation page saves one snapshot per session and
a single snapshot holds only that session's relations (Davnah: 29 in her last
snapshot, 569 across all of them).

Why this is not just "F1 over triples"
--------------------------------------
Each annotator's relation input carried THEIR OWN entity spans, so the two
annotators do not share an entity layer (Mark vs Davnah: strict entity F1 0.48).
A relation can therefore disagree for two very different reasons:

    (a) the endpoints are not the same spans   → an ENTITY-layer disagreement
    (b) the endpoints are the same but the label differs → a RELATION disagreement

Mixing them yields an uninterpretable number. This report separates them:

  1. Coverage / scope          — what is even comparable, what is workload split.
  2. Argument (entity) ceiling — the share of each annotator's relations whose
                                 BOTH endpoints exist in the other's entity set.
                                 Relation agreement cannot exceed this; it is the
                                 attainable maximum, not a result.
  3. Pair detection (unlabeled) — did they link the same two entities at all?
  4. Full agreement (labeled)   — …with the same relation label?
  5. Labeling agreement given a co-detected pair — observed agreement + Cohen's
                                 kappa. This is the chance-corrected number that
                                 is NOT inflated by the huge implicit NONE class,
                                 because it conditions on both annotators having
                                 linked the pair.
  6. Per-label F1, label confusions, direction diagnostics, vocabulary overlap.

Argument matching runs at four strictnesses, reported side by side, because the
right one depends on the question being asked:

    strict    endpoints equal on (start_char, end_char, type)
    boundary  endpoints equal on (start_char, end_char)        [default primary]
    overlap   endpoint character ranges intersect
    text      endpoint normalised surface form equal (position-agnostic)

Matching is a greedy one-to-one alignment per sentence, so one relation can
never be credited twice. "F1" is symmetric — 2·agree / (n_A + n_B) — which is
the standard pairwise agreement score, with a bootstrap CI resampling sentences.

Label views (--label-view) — the two annotators invented different custom
labels for overlapping ideas (Mark OCCURS_DURING vs Davnah DURING), so raw label
agreement understates conceptual agreement while an aggressive mapping would
manufacture it. All three views are explicit and the active mapping is printed:

    raw       (default) labels as annotated — the honest baseline
    synonyms  near-certain lexical equivalents unified
    families  coarse semantic families (negated variants join their positive base)

Hyper-relations (a relation whose argument is another relation) have no char
offsets and are excluded from the scored metrics; they are counted and reported
separately (Mark 88, Davnah 0 — so they are unmatchable in that pair anyway;
Mark's use of them grew steadily, so re-check this count rather than citing it).

Usage
-----
    python eval_relation_iaa.py \
        --a data/aug_runs/Mark_relation_input_relation/Mark/relations_merged.jsonl \
        --b data/aug_runs/Davnah_relation_input_relation/Davnah/relations_merged.jsonl \
        --name-a Mark --name-b Davnah \
        [--label-view synonyms] [--primary-mode boundary] \
        [--output output/eval_reports/relation_iaa.json]

    # three or more annotators → every pair is reported
    python eval_relation_iaa.py \
        --annotator Mark=.../Mark/relations_merged.jsonl \
        --annotator Davnah=.../Davnah/relations_merged.jsonl \
        --annotator Third=.../Third/relations_merged.jsonl

Mark and Davnah are the only annotators with enough relation coverage to score
(93 doubly-annotated sentences as of 2026-08-11; the report prints the current
count). Anyone with a handful of sentences produces CIs that span zero — the
report says so, but do not put those pairs in a write-up.

Labels the annotators invent keep arriving, and a label absent from FAMILY_MAP
stays a family of its own — silently scored as disagreeing with everything. The
report's per-label table shows every label, so check for newcomers there before
trusting the families view: as of 2026-08-11 ADDING_TO, IS_INSTANCE_OF,
SEPARATES and SUBSET_OF are unmapped, and whether the subsumption labels
(IS_INSTANCE_OF, SUBSET_OF) belong in PART_WHOLE is an ontological decision for
the annotators, not a mapping to make quietly — it is worth 0.06 kappa.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Resolve flat sibling imports however this module is launched (mirrors
# eval_layer1_output.py).
_PKG_ROOT = Path(__file__).resolve().parent.parent   # …/multi_agent_annotation
for _p in (_PKG_ROOT, _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_BOOT = 5000
SEED = 42

ARG_MODES: Tuple[str, ...] = ("strict", "boundary", "overlap", "text")

# Mirrors relation_annotation_page.RELATION_TYPES (and the merge script): labels
# outside this set were typed in by the annotator as a custom relation.
FIXED_RELATION_TYPES = {
    "HAS_PROPERTY", "IS_PART_OF", "LOCATED_IN", "AFFECTS", "HAS_PROCESS",
    "COMPARES_TO", "RELATED_TO", "CAUSES", "DURING",
}

# --label-view synonyms: only labels whose equivalence is a naming difference,
# not a semantic judgement. Deliberately short — anything arguable belongs in
# the families view, where the coarsening is visible.
SYNONYM_MAP: Dict[str, str] = {
    "OCCURS_DURING": "DURING",
    "IS_EQUAL_TO":   "IS_THE_SAME_AS",
    "COMPARATIVE":   "COMPARES_TO",
}

# --label-view families: coarse semantic grouping. A negated label joins its
# positive base (AFFECTS_NOT → CAUSAL), so this view answers "do they agree on
# the KIND of link?" and NOT "do they agree on polarity".
FAMILY_MAP: Dict[str, str] = {
    # property / description
    "HAS_PROPERTY": "PROPERTY", "DESCRIPTIVE": "PROPERTY",
    "INDICATIVE": "PROPERTY", "IS_SUBJECT_OF": "PROPERTY",
    # part–whole
    "IS_PART_OF": "PART_WHOLE",
    # process
    "HAS_PROCESS": "PROCESS",
    # spatial
    "LOCATED_IN": "SPATIAL", "LOCATED_BY": "SPATIAL", "IS_ABSENT_FROM": "SPATIAL",
    # causal / influence
    "CAUSES": "CAUSAL", "AFFECTS": "CAUSAL", "AFFECTS_NOT": "CAUSAL",
    "ENABLING": "CAUSAL", "DEMONSTRATES": "CAUSAL",
    # association / correlation
    "RELATED_TO": "ASSOCIATION", "ASSOCIATIVE": "ASSOCIATION",
    "CORRELATE_WITH": "ASSOCIATION", "POSITIVELY_RELATED_TO": "ASSOCIATION",
    "NEGATIVELY_RELATED_TO": "ASSOCIATION", "VARIES_WITH": "ASSOCIATION",
    "INCREASES_WITH": "ASSOCIATION", "INTERACTS_WITH": "ASSOCIATION",
    "INTERACTS_NOT_WITH": "ASSOCIATION", "DEPENDENCY": "ASSOCIATION",
    # comparison / identity
    "COMPARES_TO": "COMPARISON", "COMPARATIVE": "COMPARISON",
    "IS_EQUAL_TO": "COMPARISON", "IS_THE_SAME_AS": "COMPARISON",
    # temporal
    "DURING": "TEMPORAL", "OCCURS_DURING": "TEMPORAL",
}

LABEL_VIEWS = ("raw", "synonyms", "families")


# ───────────────────────── data model ─────────────────────────

@dataclass(frozen=True)
class Arg:
    """One relation endpoint: an entity span."""
    start: Optional[int]
    end: Optional[int]
    type: str
    text: str

    @property
    def has_offsets(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass(frozen=True)
class Rel:
    """A scored relation instance: entity --label--> entity, both with offsets."""
    label: str          # after the active label view
    raw_label: str      # as annotated
    e1: Arg
    e2: Arg
    note: str
    prop: str


def _norm_text(t: str) -> str:
    return " ".join((t or "").lower().split())


def _norm_label(label: str) -> str:
    return (label or "").strip().upper().replace(" ", "_")


def apply_label_view(label: str, view: str) -> str:
    if view == "synonyms":
        return SYNONYM_MAP.get(label, label)
    if view == "families":
        return FAMILY_MAP.get(label, label)
    return label


# ───────────────────────── loading ─────────────────────────

def load_doc(path: Path) -> List[dict]:
    """
    Sentences of a native relation doc. Accepts the merged single-line JSONL and
    a pretty-printed native JSON; multi-line JSONL docs are concatenated.
    """
    raw = path.read_text(encoding="utf8").strip()
    if not raw:
        raise ValueError(f"{path} is empty")
    try:
        doc = json.loads(raw)
        docs = [doc] if isinstance(doc, dict) else list(doc)
    except json.JSONDecodeError:
        docs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    sentences: List[dict] = []
    for d in docs:
        sentences.extend(d.get("sentences", []))
    return sentences


def entity_args(sent: dict) -> List[Arg]:
    """The annotator's entity layer for this sentence (endpoint candidates)."""
    out = []
    for sp in sent.get("spans", []) or []:
        out.append(Arg(sp.get("start_char"), sp.get("end_char"),
                       _norm_label(sp.get("type", "")), _norm_text(sp.get("text", ""))))
    return out


def _arg_of(raw: Any) -> Optional[Arg]:
    if not isinstance(raw, dict):
        return None
    return Arg(raw.get("start_char"), raw.get("end_char"),
               _norm_label(raw.get("type", "")), _norm_text(raw.get("text", "")))


def sentence_relations(sent: dict, view: str) -> Tuple[List[Rel], int, int]:
    """
    (scorable relations, hyper-relation count, dropped-for-missing-offsets count).

    A relation is scorable when both endpoints are entities with char offsets —
    hyper-relations (level > 0, an argument that is itself a relation) have no
    offsets and cannot be aligned against another annotator's spans.
    """
    rels: List[Rel] = []
    hyper = 0
    dropped = 0
    for r in sent.get("relations", []) or []:
        if r.get("level", 0) or r.get("e1_kind") == "relation" or r.get("e2_kind") == "relation":
            hyper += 1
            continue
        e1, e2 = _arg_of(r.get("e1")), _arg_of(r.get("e2"))
        if e1 is None or e2 is None or not e1.has_offsets or not e2.has_offsets:
            dropped += 1
            continue
        raw = _norm_label(r.get("relation", ""))
        rels.append(Rel(
            label=apply_label_view(raw, view),
            raw_label=raw,
            e1=e1, e2=e2,
            note=(r.get("note") or "").strip(),
            prop=(r.get("property") or "").strip(),
        ))
    return rels, hyper, dropped


# ───────────────────────── argument matching ─────────────────────────

def arg_score(a: Arg, b: Arg, mode: str) -> Optional[float]:
    """
    Score for identifying endpoint `a` with endpoint `b` under `mode`.
    None = not the same endpoint. Higher = better (used for tie-breaking).
    """
    if mode == "strict":
        return 1.0 if (a.start == b.start and a.end == b.end and a.type == b.type) else None
    if mode == "boundary":
        return 1.0 if (a.start == b.start and a.end == b.end) else None
    if mode == "text":
        return 1.0 if (a.text and a.text == b.text) else None
    if mode == "overlap":
        if not (a.has_offsets and b.has_offsets):
            return None
        lo, hi = max(a.start, b.start), min(a.end, b.end)
        if hi <= lo:
            return None
        union = max(a.end, b.end) - min(a.start, b.start)
        return (hi - lo) / union if union else 1.0
    raise ValueError(f"unknown argument-match mode: {mode}")


def relation_compat(
    ra: Rel, rb: Rel, mode: str, labeled: bool, undirected: bool,
) -> Optional[Tuple[float, bool]]:
    """
    (score, swapped) if `ra` and `rb` are the same relation instance, else None.
    `swapped` means they only align with the arguments exchanged, i.e. the two
    annotators disagree on direction.
    """
    if labeled and ra.label != rb.label:
        return None
    s1, s2 = arg_score(ra.e1, rb.e1, mode), arg_score(ra.e2, rb.e2, mode)
    if s1 is not None and s2 is not None:
        return (s1 + s2, False)
    if undirected:
        r1, r2 = arg_score(ra.e1, rb.e2, mode), arg_score(ra.e2, rb.e1, mode)
        if r1 is not None and r2 is not None:
            return (r1 + r2 - 1e-6, True)   # tie-break: prefer a same-direction match
    return None


def match_relations(
    rels_a: Sequence[Rel], rels_b: Sequence[Rel], mode: str,
    labeled: bool = True, undirected: bool = False,
) -> List[Tuple[int, int, bool]]:
    """
    Greedy one-to-one alignment between two annotators' relations in ONE
    sentence. Returns [(index_a, index_b, swapped)].

    Greedy on a descending score is exact for the equality-based modes (strict /
    boundary / text), where every candidate scores the same and the alignment is
    a multiset intersection; for `overlap` it is a well-behaved approximation of
    maximum-weight matching at these sizes (≤ ~30 relations per sentence).
    """
    cands: List[Tuple[float, bool, int, int]] = []
    for i, ra in enumerate(rels_a):
        for j, rb in enumerate(rels_b):
            got = relation_compat(ra, rb, mode, labeled, undirected)
            if got is not None:
                score, swapped = got
                cands.append((score, swapped, i, j))
    # deterministic: best score first, then stable by index
    cands.sort(key=lambda c: (-c[0], c[2], c[3]))
    used_a: set = set()
    used_b: set = set()
    out: List[Tuple[int, int, bool]] = []
    for score, swapped, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, swapped))
    return out


# ───────────────────────── statistics ─────────────────────────

def micro_f1(tp: int, n_a: int, n_b: int) -> Optional[float]:
    """Symmetric pairwise agreement: 2·agree / (n_A + n_B)."""
    return (2 * tp / (n_a + n_b)) if (n_a + n_b) else None


def bootstrap_f1_ci(per_sentence: List[Tuple[int, int, int]]) -> Tuple[float, float]:
    """Percentile CI for micro-F1, resampling SENTENCES with replacement."""
    rng = random.Random(SEED)
    n = len(per_sentence)
    if n == 0:
        return (float("nan"), float("nan"))
    vals: List[float] = []
    for _ in range(N_BOOT):
        tp = na = nb = 0
        for _ in range(n):
            t, a, b = per_sentence[rng.randrange(n)]
            tp += t
            na += a
            nb += b
        f1 = micro_f1(tp, na, nb)
        if f1 is not None:
            vals.append(f1)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[max(0, int(0.975 * len(vals)) - 1)])


def cohen_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    """Cohen's kappa over (label_a, label_b) decisions."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[t] / n) * (cb[t] / n) for t in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ───────────────────────── main computation ─────────────────────────

def _align_sentences(
    sents_a: List[dict], sents_b: List[dict],
) -> List[Tuple[dict, dict]]:
    """
    Pair sentences by normalised text. Repeated texts are paired in order, so a
    duplicated sentence is not collapsed into one unit.
    """
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for s in sents_b:
        buckets[_norm_text(s.get("text", ""))].append(s)
    cursor: Counter = Counter()
    pairs: List[Tuple[dict, dict]] = []
    for sa in sents_a:
        key = _norm_text(sa.get("text", ""))
        idx = cursor[key]
        if idx < len(buckets.get(key, [])):
            pairs.append((sa, buckets[key][idx]))
            cursor[key] += 1
    return pairs


def compute(
    sents_a: List[dict], sents_b: List[dict], name_a: str, name_b: str,
    view: str = "raw", primary_mode: str = "boundary",
    scope: str = "both-annotated",
) -> Dict[str, Any]:
    aligned = _align_sentences(sents_a, sents_b)

    # ── per-sentence relation lists + hyper/offset bookkeeping ──
    # A scored row carries both relation lists AND both entity layers, since the
    # ceiling needs each annotator's relations tested against the other's spans.
    rows: List[Tuple[List[Rel], List[Rel]]] = []
    all_rows: List[Tuple[List[Rel], List[Rel], List[Arg], List[Arg]]] = []
    hyper_a = hyper_b = dropped_a = dropped_b = 0
    a_only = b_only = 0
    for sa, sb in aligned:
        ra, ha, da = sentence_relations(sa, view)
        rb, hb, db = sentence_relations(sb, view)
        hyper_a += ha
        hyper_b += hb
        dropped_a += da
        dropped_b += db
        rows.append((ra, rb))
        all_rows.append((ra, rb, entity_args(sa), entity_args(sb)))
        if ra and rb:
            pass
        elif ra and not rb:
            a_only += 1
        elif rb and not ra:
            b_only += 1

    if scope == "all-aligned":
        scoped = [row for row in all_rows if row[0] or row[1]]
    else:
        scoped = [row for row in all_rows if row[0] and row[1]]

    rel_a_total = sum(len(ra) for ra, _ in rows)
    rel_b_total = sum(len(rb) for _, rb in rows)
    n_a = sum(len(row[0]) for row in scoped)
    n_b = sum(len(row[1]) for row in scoped)

    result: Dict[str, Any] = {
        "annotators": [name_a, name_b],
        "label_view": view,
        "primary_mode": primary_mode,
        "scope": scope,
        "coverage": {
            "sentences_a": len(sents_a),
            "sentences_b": len(sents_b),
            "text_aligned": len(aligned),
            f"{name_a}_sentences_with_relations": sum(1 for ra, _ in rows if ra),
            f"{name_b}_sentences_with_relations": sum(1 for _, rb in rows if rb),
            "doubly_annotated": sum(1 for ra, rb in rows if ra and rb),
            f"{name_a}_only_annotated": a_only,
            f"{name_b}_only_annotated": b_only,
            f"{name_a}_relations_total": rel_a_total,
            f"{name_b}_relations_total": rel_b_total,
            f"{name_a}_relations_in_scope": n_a,
            f"{name_b}_relations_in_scope": n_b,
            f"{name_a}_hyper_relations_excluded": hyper_a,
            f"{name_b}_hyper_relations_excluded": hyper_b,
            f"{name_a}_no_offset_excluded": dropped_a,
            f"{name_b}_no_offset_excluded": dropped_b,
            "scored_sentences": len(scoped),
        },
    }

    # ── 2. argument (entity) ceiling ──
    ceiling: Dict[str, Any] = {}
    for mode in ARG_MODES:
        ok_a = ok_b = 0
        for ra, rb, ents_a, ents_b in scoped:
            for r in ra:
                if _endpoint_present(r, ents_b, mode):
                    ok_a += 1
            for r in rb:
                if _endpoint_present(r, ents_a, mode):
                    ok_b += 1
        ceiling[mode] = {
            f"{name_a}_matchable": ok_a,
            f"{name_b}_matchable": ok_b,
            f"{name_a}_matchable_frac": (ok_a / n_a) if n_a else None,
            f"{name_b}_matchable_frac": (ok_b / n_b) if n_b else None,
            "ceiling_f1": micro_f1(min(ok_a, ok_b), n_a, n_b),
        }
    result["argument_ceiling"] = ceiling

    # ── 3/4. detection (unlabeled) and full (labeled) agreement ──
    for tag, labeled in (("detection_f1", False), ("labeled_f1", True)):
        per_mode: Dict[str, Any] = {}
        for mode in ARG_MODES:
            per_sent: List[Tuple[int, int, int]] = []
            tp = 0
            for ra, rb, _ea, _eb in scoped:
                m = match_relations(ra, rb, mode, labeled=labeled)
                tp += len(m)
                per_sent.append((len(m), len(ra), len(rb)))
            lo, hi = bootstrap_f1_ci(per_sent)
            per_mode[mode] = {
                "f1": micro_f1(tp, n_a, n_b), "tp": tp, "n_a": n_a, "n_b": n_b,
                "ci95": [lo, hi],
            }
        result[tag] = per_mode

    # ── 5. labeling agreement conditioned on a co-detected pair ──
    label_pairs: List[Tuple[str, str]] = []
    raw_label_pairs: List[Tuple[str, str]] = []
    confusion: Counter = Counter()
    note_pairs = 0
    for ra, rb, _ea, _eb in scoped:
        for i, j, _sw in match_relations(ra, rb, primary_mode, labeled=False):
            label_pairs.append((ra[i].label, rb[j].label))
            raw_label_pairs.append((ra[i].raw_label, rb[j].raw_label))
            confusion[(ra[i].label, rb[j].label)] += 1
            if ra[i].note or rb[j].note:
                note_pairs += 1
    n_pairs = len(label_pairs)
    agree_n = sum(1 for a, b in label_pairs if a == b)
    result["labeling_given_detection"] = {
        "co_detected_pairs": n_pairs,
        "observed_agreement": (agree_n / n_pairs) if n_pairs else None,
        "cohen_kappa": cohen_kappa(label_pairs),
        "cohen_kappa_raw_labels": cohen_kappa(raw_label_pairs),
        "distinct_labels_a": len({a for a, _ in label_pairs}),
        "distinct_labels_b": len({b for _, b in label_pairs}),
        "co_detected_pairs_with_a_note": note_pairs,
    }
    result["_confusion"] = confusion

    # ── 6. per-label F1 under the primary mode ──
    per_label: Dict[str, Dict[str, Any]] = {}
    tp_by_label: Counter = Counter()
    for ra, rb, _ea, _eb in scoped:
        for i, j, _sw in match_relations(ra, rb, primary_mode, labeled=True):
            tp_by_label[ra[i].label] += 1
    count_a = Counter(r.label for row in scoped for r in row[0])
    count_b = Counter(r.label for row in scoped for r in row[1])
    for label in sorted(set(count_a) | set(count_b)):
        per_label[label] = {
            "f1": micro_f1(tp_by_label[label], count_a[label], count_b[label]),
            "tp": tp_by_label[label],
            "n_a": count_a[label],
            "n_b": count_b[label],
            "custom": label not in FIXED_RELATION_TYPES and view != "families",
        }
    result["per_label"] = per_label

    # ── 7. direction diagnostics ──
    tp_dir = tp_undir = swap_only = swap_same_label = 0
    for ra, rb, _ea, _eb in scoped:
        tp_dir += len(match_relations(ra, rb, primary_mode, labeled=False))
        for i, j, sw in match_relations(ra, rb, primary_mode, labeled=False,
                                        undirected=True):
            tp_undir += 1
            if sw:
                swap_only += 1
                if ra[i].label == rb[j].label:
                    swap_same_label += 1
    result["direction"] = {
        "pairs_same_direction": tp_dir,
        "pairs_any_direction": tp_undir,
        "direction_only_disagreements": swap_only,
        "direction_only_same_label": swap_same_label,
    }

    # ── 8. where the disagreement actually lives ──
    # Exact factorisation of the labeled score under the primary mode:
    #   labeled F1 = ceiling × (detection F1 / ceiling) × (labeled F1 / detection F1)
    # i.e. entity layer × which pairs to link × which label to give the pair.
    ceil_f1 = ceiling[primary_mode]["ceiling_f1"]
    det_f1 = result["detection_f1"][primary_mode]["f1"]
    lab_f1 = result["labeled_f1"][primary_mode]["f1"]
    result["decomposition"] = {
        "entity_ceiling": ceil_f1,
        "pair_linking_given_matchable": (det_f1 / ceil_f1) if (ceil_f1 or 0) > 0 else None,
        "labeling_given_co_detected": (lab_f1 / det_f1) if (det_f1 or 0) > 0 else None,
        "labeled_f1": lab_f1,
    }

    # ── 9. vocabulary comparison (raw labels, whole files) ──
    raw_a = Counter(r.raw_label for ra, _ in rows for r in ra)
    raw_b = Counter(r.raw_label for _, rb in rows for r in rb)
    result["vocabulary"] = {
        f"{name_a}_labels": dict(raw_a.most_common()),
        f"{name_b}_labels": dict(raw_b.most_common()),
        "shared": sorted(set(raw_a) & set(raw_b)),
        f"only_{name_a}": sorted(set(raw_a) - set(raw_b)),
        f"only_{name_b}": sorted(set(raw_b) - set(raw_a)),
        f"{name_a}_custom": sorted(l for l in raw_a if l not in FIXED_RELATION_TYPES),
        f"{name_b}_custom": sorted(l for l in raw_b if l not in FIXED_RELATION_TYPES),
    }

    # which view mappings actually fired on this data
    fired: Dict[str, str] = {}
    if view != "raw":
        table = SYNONYM_MAP if view == "synonyms" else FAMILY_MAP
        for label in set(raw_a) | set(raw_b):
            if label in table:
                fired[label] = table[label]
    result["label_view_mappings_applied"] = dict(sorted(fired.items()))

    return result


def _endpoint_present(r: Rel, ents: List[Arg], mode: str) -> bool:
    """Both endpoints of `r` exist in `ents` under `mode` (the ceiling test)."""
    return (any(arg_score(r.e1, e, mode) is not None for e in ents)
            and any(arg_score(r.e2, e, mode) is not None for e in ents))


# ───────────────────────── reporting ─────────────────────────

def print_report(r: Dict[str, Any]) -> None:
    a, b = r["annotators"]
    cov = r["coverage"]
    mode = r["primary_mode"]

    print(f"\n{'=' * 74}")
    print(f"  RELATION INTER-ANNOTATOR AGREEMENT — {a} vs {b}")
    print(f"{'=' * 74}")
    print(f"  label view: {r['label_view']}   primary argument match: {mode}"
          f"   scope: {r['scope']}")

    print("\n  Coverage")
    print(f"    Sentences ({a}/{b}):          {cov['sentences_a']} / {cov['sentences_b']}"
          f"   text-aligned: {cov['text_aligned']}")
    print(f"    With ≥1 relation:              {cov[f'{a}_sentences_with_relations']}"
          f" / {cov[f'{b}_sentences_with_relations']}")
    print(f"    Doubly annotated:              {cov['doubly_annotated']}")
    print(f"    Sentences scored:              {cov['scored_sentences']}"
          f"   (scope: {r['scope']})")
    print(f"    {a}-only / {b}-only:  {cov[f'{a}_only_annotated']}"
          f" / {cov[f'{b}_only_annotated']}     (workload division — never scored)")
    print(f"    Relations total:               {cov[f'{a}_relations_total']}"
          f" / {cov[f'{b}_relations_total']}")
    print(f"    Relations in scope:            {cov[f'{a}_relations_in_scope']}"
          f" / {cov[f'{b}_relations_in_scope']}")
    if cov[f"{a}_hyper_relations_excluded"] or cov[f"{b}_hyper_relations_excluded"]:
        print(f"    Hyper-relations excluded:      {cov[f'{a}_hyper_relations_excluded']}"
              f" / {cov[f'{b}_hyper_relations_excluded']}   (no offsets — unscorable)")
    if cov[f"{a}_no_offset_excluded"] or cov[f"{b}_no_offset_excluded"]:
        print(f"    Missing-offset excluded:       {cov[f'{a}_no_offset_excluded']}"
              f" / {cov[f'{b}_no_offset_excluded']}")

    print("\n  Argument (entity) ceiling — share of relations whose BOTH endpoints")
    print("  exist in the other annotator's entity layer. Agreement cannot exceed this.")
    for m in ARG_MODES:
        c = r["argument_ceiling"][m]
        fa, fb = c[f"{a}_matchable_frac"], c[f"{b}_matchable_frac"]
        ceil = c["ceiling_f1"]
        print(f"    {m:<9} {a}: {_pct(fa)}  {b}: {_pct(fb)}"
              f"   → ceiling F1 ≤ {_num(ceil)}")

    def block(title: str, key: str) -> None:
        print(f"\n  {title}")
        for m in ARG_MODES:
            d = r[key][m]
            star = "  ← primary" if m == mode else ""
            ci = d["ci95"]
            print(f"    {m:<9} F1={_num(d['f1'])}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
                  f"   [agree={d['tp']:3d} | {a}={d['n_a']} | {b}={d['n_b']}]{star}")

    block("Pair detection agreement (unlabeled: same two entities linked?)", "detection_f1")
    block("Full relation agreement (labeled: same pair AND same label)", "labeled_f1")

    dec = r["decomposition"]
    print(f"\n  Where the disagreement lives (argument match: {mode})")
    print(f"    entity ceiling            {_num(dec['entity_ceiling'])}"
          "   ← do the endpoint spans even exist on both sides?")
    print(f"  × pair linking | matchable  {_num(dec['pair_linking_given_matchable'])}"
          "   ← of the linkable pairs, both chose to link?")
    print(f"  × labeling | co-detected    {_num(dec['labeling_given_co_detected'])}"
          "   ← same label for a pair both linked?")
    print(f"  = labeled F1                {_num(dec['labeled_f1'])}")

    lg = r["labeling_given_detection"]
    print("\n  Labeling agreement GIVEN a co-detected pair "
          f"(argument match: {mode})")
    print(f"    Co-detected pairs:             {lg['co_detected_pairs']}")
    print(f"    Observed label agreement:      {_num(lg['observed_agreement'])}")
    print(f"    Cohen's kappa:                 {_num(lg['cohen_kappa'])}"
          f"   (raw labels: {_num(lg['cohen_kappa_raw_labels'])})")
    print(f"    Distinct labels used ({a}/{b}): "
          f"{lg['distinct_labels_a']} / {lg['distinct_labels_b']}")

    d = r["direction"]
    print("\n  Direction")
    print(f"    Pairs matched same direction:  {d['pairs_same_direction']}")
    print(f"    Pairs matched either way:      {d['pairs_any_direction']}")
    print(f"    Direction-only disagreements:  {d['direction_only_disagreements']}"
          f"   (of which same label: {d['direction_only_same_label']})")

    print(f"\n  Per relation label (labeled F1, argument match: {mode}, by support)")
    rows = sorted(r["per_label"].items(), key=lambda kv: -(kv[1]["n_a"] + kv[1]["n_b"]))
    for label, m in rows:
        tag = "  ← custom" if m["custom"] else ""
        print(f"    {label:<24} F1={_num(m['f1'])}  agree={m['tp']:3d}"
              f"  {a}={m['n_a']:3d}  {b}={m['n_b']:3d}{tag}")

    diffs = sorted(((k, v) for k, v in r["_confusion"].items() if k[0] != k[1]),
                   key=lambda x: -x[1])
    if diffs:
        print(f"\n  Top label confusions on co-detected pairs ({a} → {b})")
        for (la, lb), c in diffs[:15]:
            print(f"    {la:<24} → {lb:<24} ×{c}")

    v = r["vocabulary"]
    print("\n  Label vocabulary")
    print(f"    Shared labels ({len(v['shared'])}): {', '.join(v['shared']) or '—'}")
    print(f"    Only {a} ({len(v[f'only_{a}'])}): {', '.join(v[f'only_{a}']) or '—'}")
    print(f"    Only {b} ({len(v[f'only_{b}'])}): {', '.join(v[f'only_{b}']) or '—'}")
    if r["label_view_mappings_applied"]:
        print(f"    View '{r['label_view']}' mappings applied to this data:")
        for src, dst in r["label_view_mappings_applied"].items():
            print(f"      {src:<24} → {dst}")

    _print_caveats(r)


def _print_caveats(r: Dict[str, Any]) -> None:
    a, b = r["annotators"]
    cov = r["coverage"]
    notes: List[str] = []
    if cov["scored_sentences"] < 20:
        notes.append(
            f"only {cov['scored_sentences']} doubly-annotated sentence(s) — the CIs are wide "
            "and these numbers are indicative, not publishable.")
    ceil_primary = r["argument_ceiling"][r["primary_mode"]]["ceiling_f1"]
    labeled = r["labeled_f1"][r["primary_mode"]]["f1"]
    if ceil_primary is not None and ceil_primary < 0.9:
        notes.append(
            f"the entity layers differ: at most {ceil_primary:.3f} F1 is reachable under "
            f"'{r['primary_mode']}' argument matching, so a low labeled F1 "
            f"({_num(labeled)}) is mostly an ENTITY-layer disagreement, not a relation one — "
            "read the kappa given co-detected pairs instead.")
    if r["vocabulary"][f"only_{a}"] or r["vocabulary"][f"only_{b}"]:
        notes.append(
            "the annotators used non-overlapping custom labels; compare --label-view raw "
            "with synonyms/families to see how much of the label disagreement is naming.")
    if cov[f"{a}_hyper_relations_excluded"] != cov[f"{b}_hyper_relations_excluded"]:
        notes.append(
            f"hyper-relation use is asymmetric ({cov[f'{a}_hyper_relations_excluded']} vs "
            f"{cov[f'{b}_hyper_relations_excluded']}) — that disagreement is invisible here "
            "because hyper-relations are excluded from every metric.")
    if notes:
        print("\n  Read this before quoting any number")
        for n in notes:
            print(f"    - {n}")
    print()


def _num(x: Optional[float]) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else " N/A "


def _pct(x: Optional[float]) -> str:
    return f"{100 * x:5.1f}%" if isinstance(x, (int, float)) and x == x else "  N/A"


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Relation IAA between annotators (merged relation files).")
    ap.add_argument("--a", type=Path, help="Annotator A merged relation file")
    ap.add_argument("--b", type=Path, help="Annotator B merged relation file")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--annotator", action="append", default=[], metavar="NAME=PATH",
                    help="Repeatable; with 3+ annotators every pair is reported.")
    ap.add_argument("--label-view", choices=LABEL_VIEWS, default="raw",
                    help="raw (default) = labels as annotated; synonyms = unify "
                         "near-certain naming variants; families = coarse semantic "
                         "families. The active mapping is printed.")
    ap.add_argument("--primary-mode", choices=ARG_MODES, default="boundary",
                    help="Argument matching used for the kappa, per-label F1 and "
                         "confusions (all four modes are always reported for F1).")
    ap.add_argument("--scope", choices=["both-annotated", "all-aligned"],
                    default="both-annotated",
                    help="both-annotated (default): score only sentences where BOTH "
                         "annotated ≥1 relation. all-aligned: also count sentences only "
                         "one of them touched (turns workload division into disagreement).")
    ap.add_argument("--output", type=Path, default=None, help="Write the report as JSON")
    args = ap.parse_args()

    annotators: List[Tuple[str, Path]] = []
    for spec in args.annotator:
        if "=" not in spec:
            ap.error(f"--annotator expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        annotators.append((name, Path(path)))
    if args.a and args.b:
        annotators = [
            (args.name_a or _stem_name(args.a), args.a),
            (args.name_b or _stem_name(args.b), args.b),
        ] + annotators
    if len(annotators) < 2:
        ap.error("give --a/--b or at least two --annotator NAME=PATH")

    loaded = [(name, load_doc(path)) for name, path in annotators]
    reports = []
    for (na, sa), (nb, sb) in combinations(loaded, 2):
        rep = compute(sa, sb, na, nb, view=args.label_view,
                      primary_mode=args.primary_mode, scope=args.scope)
        print_report(rep)
        rep.pop("_confusion", None)
        reports.append(rep)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = reports[0] if len(reports) == 1 else {"pairs": reports}
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf8")
        print(f"  JSON written to {args.output}\n")


def _stem_name(path: Path) -> str:
    """'…/aug_runs/X_relation_input_relation/Mark/relations_merged.jsonl' → 'Mark'."""
    parent = path.parent.name
    return parent if parent and parent not in {".", "/"} else path.stem


if __name__ == "__main__":
    main()