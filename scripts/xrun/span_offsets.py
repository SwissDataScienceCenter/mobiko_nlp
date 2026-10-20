#!/usr/bin/env python
"""
Resolve agent span TEXT to character offsets, so span matching can stop relying
on substring containment.

Why this exists. The Annotator emits only {"text": ..., "entity_type": ...} with
no offsets, so every span-level comparison in evaluation/ falls back to
deliberation_history.overlaps, which is `a == b or a in b or b in a`. That makes
a one-token agent span match ANY human span containing that token: "forest"
matches "forest ecosystem", "tropical forest", "forest fragmentation". Short
spans therefore get inflated match rates and understated false-positive rates,
and the bias varies systematically with span length.

The human files DO carry start_char/end_char, and their raw sentence text is
byte-identical between the two annotators across all 377 shared sentences, so
there is one coordinate system to resolve into. Agent sentence strings differ
from it in 7 of 155 cases, always a newline rendered as a space, which preserves
positions; the whitespace-tolerant pattern below absorbs that anyway.

Resolution outcomes, measured over 3487 annotator spans:
    unique      91.5%   one occurrence in the sentence
    ambiguous    5.6%   two or more occurrences, cannot tell which was meant
    absent       2.9%   the surface string is not in the sentence at all

AMBIGUITY POLICY. For an ambiguous span, every occurrence is a candidate and the
span counts as matched if ANY candidate matches a human span. That is deliberately
charitable: it biases AGAINST finding a false positive, so it cannot manufacture
the effect it is used to test.

ABSENT spans are kept, not dropped. A span whose text does not occur in the
sentence is a surface-form hallucination and the strongest kind of false
positive; silently dropping 2.9% of spans would remove exactly the cases the
analysis is about. They match nothing by construction.

MATCH POLICIES, weakest to strictest:
    overlap   any character overlap with a human span
    iou50     intersection-over-union >= 0.5 with a human span
    exact     identical (start, end)

iou50 is the default for "did a human mark this mention". `exact` conflates a
mis-delimited real entity with an invented one, which is the distinction the
false-positive analysis depends on; `overlap` lets a long agent span collect any
human span it straddles. Both are reported as bounds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("src/multi_agent_annotation",
           "src/multi_agent_annotation/evaluation"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

Span = Tuple[int, int]


def _pattern(text: str) -> re.Pattern:
    """Literal match, tolerant of any whitespace run standing in for another."""
    parts = [re.escape(p) for p in text.split()]
    return re.compile(r"\s+".join(parts) if parts else re.escape(text))


def resolve(span_text: str, sentence: str) -> Tuple[List[Span], str]:
    """-> (candidate offsets, status in {unique, ambiguous, absent})."""
    t = (span_text or "").strip()
    if not t:
        return [], "absent"
    hits = [(m.start(), m.end()) for m in _pattern(t).finditer(sentence)]
    if not hits:
        return [], "absent"
    return hits, ("unique" if len(hits) == 1 else "ambiguous")


def iou(a: Span, b: Span) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def matches(cands: List[Span], human: List[Span], policy: str = "iou50") -> bool:
    """Does ANY candidate offset match ANY human span under `policy`?"""
    for c in cands:
        for h in human:
            if policy == "exact":
                if c == h:
                    return True
            elif policy == "iou50":
                if iou(c, h) >= 0.5:
                    return True
            else:  # overlap
                if min(c[1], h[1]) > max(c[0], h[0]):
                    return True
    return False


def human_offsets(spans) -> List[Span]:
    return [(s["start_char"], s["end_char"]) for s in spans
            if s.get("start_char") is not None and s.get("end_char") is not None]


def raw_sentences(path: Path) -> Dict[str, str]:
    """norm(sentence) -> the RAW sentence text human offsets index into."""
    import json
    from label_norm import normalize_text as _n
    doc = json.loads(Path(path).read_text(encoding="utf8"))
    return {_n(s["text"]): s["text"] for s in doc.get("sentences", [])}
