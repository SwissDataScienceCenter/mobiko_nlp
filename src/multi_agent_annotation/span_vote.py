"""
Majority vote over independent annotations, and what counts as contested.

WHY A VOTE AT ALL. Simulating three independent annotators from the archived
K=10 draws (Qwen3.6-35B, 4,312 span observations, 20 random triples per
sentence) gives:

    unanimous on type                   55.5%
    2-1 type majority                    8.5%
    3-way type split                     0.3%
    produced by only 1 or 2 of the 3    35.7%

So a majority rule settles 96.6% of type disagreements, and the genuinely
undecidable case — no majority at all — is 0.3% of spans. An LLM Adjudicator is
structurally the anchored reviewer this whole design exists to avoid: it never
commits before exposure, it is the same model, and it carries none of the
independent authority a human adjudicator would. It therefore needs a baseline
it has to beat, and a free deterministic rule is that baseline. Both outputs are
produced for every sentence so the comparison can be made against human gold.

THE REAL DISAGREEMENT IS DETECTION, not typing: 35.7% against 8.8%. A vote that
only resolved types would be answering the minor question, so the rule below
decides existence first, then extent, then type.

SPAN IDENTITY. Three annotators do not produce identical offsets, so spans are
clustered into mentions by offset overlap (IoU >= 0.5 by default) rather than
compared by exact position. Within a mention the three decisions are recorded
separately — who produced it, which extent each chose, which type each assigned
— because those are three different kinds of disagreement and pooling them is
how the earlier analyses lost the detection signal.

TIES. Resolved by a documented, deterministic order, never by dict or list
order: see `_pick`. A tie is also recorded as a tie, so analysis can see where
the vote had no real answer rather than inferring it from a confident-looking
output.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

Span = Tuple[int, int]


def iou(a: Span, b: Span) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _pick(counts: Counter) -> Tuple[Any, bool]:
    """Most common value, and whether the top count was tied.

    Ties break on the value's own sort order, which is arbitrary but stable and
    independent of the order the annotators happened to be listed in. Returning
    the tie flag matters more than which side wins: a tie is the case where the
    vote has no answer, and it must not look like a decision.
    """
    if not counts:
        return None, False
    top = max(counts.values())
    winners = sorted([v for v, c in counts.items() if c == top], key=str)
    return winners[0], len(winners) > 1


def cluster_mentions(annotations: Sequence[List[Dict[str, Any]]],
                     threshold: float = 0.5) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Group spans across annotators into mentions.

    `annotations` is one list of span dicts per annotator; each dict needs
    `start`, `end` and `entity_type`. Returns one list per mention of
    (annotator_index, span dict).

    Spans are added to the first mention they overlap at >= threshold IoU,
    scanning annotators in order. That is greedy and therefore order-dependent
    in the rare case of a span overlapping two existing mentions equally; the
    alternative (full clustering) buys precision this data does not have, since
    the three annotators are the same model and their extents mostly coincide.
    """
    mentions: List[List[Tuple[int, Dict[str, Any]]]] = []
    for ai, spans in enumerate(annotations):
        for sp in spans:
            s, e = sp.get("start"), sp.get("end")
            if not isinstance(s, int) or not isinstance(e, int):
                continue
            placed = False
            for m in mentions:
                if any(iou((s, e), (o["start"], o["end"])) >= threshold
                       for _, o in m):
                    m.append((ai, sp))
                    placed = True
                    break
            if not placed:
                mentions.append([(ai, sp)])
    return mentions


def vote(annotations: Sequence[List[Dict[str, Any]]],
         n_annotators: Optional[int] = None,
         threshold: float = 0.5,
         min_support: Optional[int] = None) -> Dict[str, Any]:
    """Majority vote over independent annotations of one sentence.

    min_support defaults to a simple majority of annotators (2 of 3). A mention
    produced by fewer than that is NOT in the voted output, but it is still
    reported in `mentions` with its support count, because a span one annotator
    saw and two did not is the single most common disagreement in this corpus
    and dropping it silently would hide a third of the signal.

    Returns:
      entities   the voted annotation: [{text, entity_type, start, end}]
      mentions   every mention with its full decision record
      contested  mentions the vote could not settle unanimously — the set an
                 Adjudicator is asked about, and the set on which the two
                 procedures are compared
    """
    n = n_annotators if n_annotators is not None else len(annotations)
    need = min_support if min_support is not None else (n // 2 + 1)
    out_entities: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for m in cluster_mentions(annotations, threshold):
        producers = sorted({ai for ai, _ in m})
        extents = Counter((sp["start"], sp["end"]) for _, sp in m)
        types = Counter(sp.get("entity_type") for _, sp in m
                        if sp.get("entity_type"))
        extent, extent_tied = _pick(extents)
        etype, type_tied = _pick(types)
        text = next((sp.get("text") for _, sp in m
                     if (sp["start"], sp["end"]) == extent), None)

        rec = {
            "support": len(producers),
            "n_annotators": n,
            "producers": producers,
            "extent": extent,
            "extent_agreement": len(extents) == 1,
            "extent_tied": extent_tied,
            "entity_type": etype,
            "type_agreement": len(types) <= 1,
            "type_tied": type_tied,
            "types_proposed": dict(types),
            "text": text,
            # Why this mention is contested, if it is. Kept as explicit reasons
            # rather than one boolean: detection, extent and type disagreement
            # are different problems and the paper reports them separately.
            "contested_reasons": [],
            "in_majority": False,
        }
        if len(producers) < n:
            rec["contested_reasons"].append("detection")
        if len(extents) > 1:
            rec["contested_reasons"].append("extent")
        if len(types) > 1:
            rec["contested_reasons"].append("type")

        if len(producers) >= need:
            rec["in_majority"] = True
            out_entities.append({"text": text, "entity_type": etype,
                                 "start": extent[0], "end": extent[1]})
        records.append(rec)

    return {
        "entities": out_entities,
        "mentions": records,
        "contested": [r for r in records if r["contested_reasons"]],
        "rule": {"min_support": need, "n_annotators": n,
                 "iou_threshold": threshold},
    }


def summarise(voted: Dict[str, Any]) -> Dict[str, int]:
    """Counts for one sentence, for aggregation across a run."""
    ms = voted["mentions"]
    return {
        "mentions": len(ms),
        "unanimous": sum(1 for m in ms if not m["contested_reasons"]),
        "detection_disagreement": sum(1 for m in ms
                                      if "detection" in m["contested_reasons"]),
        "extent_disagreement": sum(1 for m in ms
                                   if "extent" in m["contested_reasons"]),
        "type_disagreement": sum(1 for m in ms
                                 if "type" in m["contested_reasons"]),
        "type_tied": sum(1 for m in ms if m["type_tied"]),
        "in_majority": sum(1 for m in ms if m["in_majority"]),
    }
