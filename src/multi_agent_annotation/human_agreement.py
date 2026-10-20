"""
Human disagreement over N annotators, grouped by POSITION rather than by text.

WHAT IS WRONG WITH THE EXISTING MEASURE. eval_layer2_correlation.
compute_human_span_disagreements keys spans by normalised text, so two
annotators marking the same mention at different extents become two unrelated
entities, each internally unanimous. The CoNLL-MTurk case that makes it obvious:

    a12  'America'           (10-17)  LOC
    a24  'Corporate America' (0-17)   MISC
    a33  'America'           (10-17)  LOC
    a46  'America'           (10-17)  LOC

Text keying yields {"america": LOC x3, present 3/5} and {"corporate america":
MISC x1, present 1/5}. Each group agrees with itself, so type_disagreed is False
for both and the LOC-vs-MISC conflict — the actual disagreement — disappears.

split_buckets (which produces the 0.497 ceiling) misses it differently: (10,17)
!= (0,17), so the pair lands in detection_only, "one annotator found a span the
other missed". Also no type disagreement recorded.

MEASURED CONSEQUENCE. On MoBiKo's 154 doubly-annotated sentences, type
disagreements found: 128 text-keyed, 122 exact-offset, 180 clustered here — ~45%
more, plus 168 extent disagreements neither of the others can represent, over 159
mentions where the annotators used different wording for one mention. On
CoNLL-MTurk (400 sentences x 5): 154 text-keyed vs 176 clustered, and 123 extent
disagreements.

WHY IT MATTERS. "Where do humans disagree" is the RQ2 target — the variable the
agents' signals are tested against. Understating it by 45% and losing extent
disagreement entirely means that correlation was measured against a damaged
target. It also inflated the detection-vs-typing ratio: a third of what was
counted as detection disagreement is extent disagreement on a shared mention.

A SIBLING, NOT A REPLACEMENT. The text-keyed function stays so previously
reported numbers remain reproducible and the two can be compared directly.

The 0.497 ceiling is NOT affected: strict span F1 with exact matching is a
legitimate scoring convention, which simply counts an extent mismatch as a miss.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from label_norm import canonicalize_entity_type, normalize_text
from span_vote import cluster_mentions, iou, same_mention

Span = Tuple[int, int]


def _spans_of(annotator_spans: List[dict]) -> List[Dict[str, Any]]:
    """Normalise one annotator's spans into what cluster_mentions expects.

    Accepts both human shapes (start_char/end_char, `type`) and agent shapes
    (start/end, `entity_type`) so the same clustering serves both sides.
    """
    out = []
    for s in annotator_spans:
        start = s.get("start_char", s.get("start"))
        end = s.get("end_char", s.get("end"))
        etype = canonicalize_entity_type(s.get("type") or s.get("entity_type"))
        if isinstance(start, int) and isinstance(end, int) and etype:
            out.append({"text": s.get("text", ""), "entity_type": etype,
                        "start": start, "end": end})
    return out


def mention_disagreements(
    annotator_spans: Sequence[List[dict]],
    threshold: float = 0.5,
    nested: bool = False,
) -> List[Dict[str, Any]]:
    """One record per MENTION, across N annotators.

    Each record carries the three disagreements separately, because they are
    different phenomena and pooling them is how the detection signal was lost:

      presence   how many annotators marked this mention at all
      extent     whether those who marked it chose the same span
      type       whether those who marked it assigned the same label

    `graded_*` fields mirror eval_layer2_correlation.graded_disagreement: binary
    flags have base rates that climb with annotator count (presence
    disagreement runs 0% to 100% across 1..8 annotators on CoNLL-MTurk and is
    degenerate at the top), so a measure meant to be compared across corpora
    with different N has to be graded.
    """
    n = len(annotator_spans)
    lists = [_spans_of(a) for a in annotator_spans]
    out: List[Dict[str, Any]] = []
    for m in cluster_mentions(lists, threshold, nested):
        producers = sorted({ai for ai, _ in m})
        extents = Counter((sp["start"], sp["end"]) for _, sp in m)
        types = Counter(sp["entity_type"] for _, sp in m)
        modal_extent, _ = extents.most_common(1)[0]
        modal_type, modal_type_n = types.most_common(1)[0]
        text = next((sp["text"] for _, sp in m
                     if (sp["start"], sp["end"]) == modal_extent), "")
        marked = len(producers)
        out.append({
            "extent": modal_extent,
            "text": text,
            "norm_text": normalize_text(text),
            "entity_type": modal_type,
            "types": dict(types),
            "extents": {f"{a}-{b}": c for (a, b), c in extents.items()},
            "present_in": marked,
            "total_annotators": n,
            "presence_disagreed": marked < n,
            "extent_disagreed": len(extents) > 1,
            "type_disagreed": len(types) > 1,
            # 1 - modal share among those who marked it; 0 when unanimous.
            "graded_type": 1.0 - (modal_type_n / sum(types.values())),
            "graded_presence": 1.0 - (marked / n) if n else 0.0,
            "graded_extent": 1.0 - (extents[modal_extent] / sum(extents.values())),
        })
    return out


def match(mentions: Sequence[Dict[str, Any]], span: Span,
          threshold: float = 0.5,
          nested: bool = False) -> Optional[Dict[str, Any]]:
    """The mention an agent span at `span` refers to, or None.

    Matched by overlap, not by text: an agent that marked a real mention at a
    slightly different extent is talking about that mention, and scoring it as
    a miss is the error this module exists to remove. Best overlap wins, so a
    span straddling two mentions attaches to the nearer one rather than the
    first in list order.
    """
    best, best_iou = None, 0.0
    for mm in mentions:
        if not same_mention(span, mm["extent"], threshold, nested):
            continue
        v = iou(span, mm["extent"])
        if v > best_iou or best is None:
            best, best_iou = mm, v
    return best
