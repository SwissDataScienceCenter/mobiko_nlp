"""
Split a span that covers a coordination into the concepts it actually names.

    "eastern and western slopes"  ->  "eastern slopes" + "western slopes"
    "high and transient relief"   ->  "high relief"    + "transient relief"

Why this and not modifier distribution
--------------------------------------
The plan originally paired this with the inverse operation — pushing a shared
modifier from one conjunct onto a *neighbouring span*, so that `annual rainfall`
in "areas of high temperatures, annual rainfall and topographic relief" becomes
"high annual rainfall". That was measured against 38 real corrections recorded
by the annotators in `concept_text` and it reproduced **1 of 38** with
`en_core_web_sm` and 2 with `en_core_web_trf`: almost none of those corrections
are coordination at all, they are spans drawn too narrow and abbreviated
binomials. Distribution was dropped.

Splitting is the half that works. On the 17 distinct coordinated spans in the
gold it gets 7 clearly right — including `eastern and western slopes`, which
matches the annotators' own recorded split exactly — and correctly declines the
traps: number ranges ("between 1100 and 1700m"), and proper names (IPBES,
"Monitoring and Assessment Programme").

Why this requires en_core_web_trf
---------------------------------
Not for speed — sm parses the corpus in ~1s, trf in ~15s on CPU. For safety.
The two models disagree on the attachment of 13.7% of spans, and on this
operation sm turns disagreement into garbage while trf declines:

    Da Nhim and Dran Protection Forests   sm: "Da Nhim" + "Da Forests"   trf: declines
    Monitoring and Assessment Programme   sm: "Monitoring" + "Programme" trf: declines
    broadleaf evergreen and coniferous…   sm: "broadleaf mosaic"         trf: "coniferous mosaic"

Splitting creates entities, and an entity minted from a mis-parse is not
recoverable by anything downstream. So when only sm is available the splitter
does nothing and says so, rather than guessing — which keeps the no-GPU tier
free of this failure mode instead of exposed to it.

Nothing is destroyed
--------------------
A split does not replace the mention. The original span keeps its offsets and
its row; the parts are emitted alongside it carrying ``split_from``, so the
original text is always recoverable and a run without the splitter produces a
strict subset of the rows rather than different ones.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

PREFERRED_MODEL = "en_core_web_trf"

# "between 1100 and 1700m", "between 25 and 40days after ice-off" — a range is
# one quantity, not two concepts.
_RANGE = re.compile(r"\bbetween\b", re.I)

# A span carrying several capitalised tokens is a name, not a coordination of
# two concepts: "Monitoring and Assessment Programme" (AMAP),
# "Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem
# Services" (IPBES). sm split the first of those into "Monitoring" + "Programme".
_MIN_CAPS_FOR_PROPER_NAME = 2

_MIN_PART_TOKENS = 1
_CONJ_WORDS = frozenset({"and", "or", "&"})


@dataclass
class Split:
    parts: List[str]
    pattern: str          # "shared_premodifier" | "elided_head"
    model: str


def _looks_like_proper_name(text: str) -> bool:
    caps = [t for t in text.split()
            if t[:1].isupper() and t.lower() not in _CONJ_WORDS]
    return len(caps) >= _MIN_CAPS_FOR_PROPER_NAME


def _ok(part: str) -> bool:
    toks = [t for t in part.split() if t]
    return (len(toks) >= _MIN_PART_TOKENS
            and not all(t.lower() in _CONJ_WORDS for t in toks)
            and any(c.isalpha() for c in part))


def split_span(span, model: str = PREFERRED_MODEL) -> Optional[Split]:
    """Split a parsed spaCy span, or return None to leave it alone.

    ``span`` must be a spaCy Span parsed IN ITS SENTENCE — a span parsed on its
    own loses the attachments this reads.
    """
    text = span.text
    if not any(w.lower() in _CONJ_WORDS for w in text.split()):
        return None
    if _RANGE.search(text) or _looks_like_proper_name(text):
        return None

    root = span.root

    # "huge forests and mountains": the heads are conjoined and share a
    # premodifier that sits on the first of them.
    conjs = [c for c in root.children if c.dep_ == "conj" and c in span]
    if conjs:
        shared = [c.text for c in root.children
                  if c.dep_ in ("amod", "compound", "nummod")
                  and c.i < root.i and c in span]
        parts = ([" ".join(shared + [root.text])]
                 + [" ".join(shared + [c.text]) for c in conjs])
        return Split(parts, "shared_premodifier", model) if all(map(_ok, parts)) else None

    # "eastern and western slopes": the head is stated once and the MODIFIERS
    # are conjoined, so the head is what gets shared out.
    for mod in [c for c in root.children if c.dep_ in ("amod", "compound") and c in span]:
        sub = [c for c in mod.children if c.dep_ == "conj" and c in span]
        if not sub:
            continue
        # The shared tail is the head plus every OTHER modifier of it, not just
        # the head token: "broadleaf evergreen and coniferous habitat mosaic"
        # must keep `habitat`, or the parts silently change meaning.
        coordinated = {t.i for m in [mod] + sub for t in m.subtree}
        coordinated |= {mod.i} | {m.i for m in sub}
        tail = sorted({t for t in root.subtree
                       if t in span and t.i not in coordinated
                       and t.lower_ not in _CONJ_WORDS
                       and t.pos_ not in ("PUNCT", "DET")}, key=lambda t: t.i)
        tail_text = " ".join(t.text for t in tail)

        def _own(m) -> set:
            """m's own words: its subtree minus the conjuncts hanging off it.

            spaCy hangs later conjuncts under the first one, so `western` sits
            inside `eastern`'s subtree. Taking the subtree whole made the first
            part "eastern western slopes".
            """
            drop = {t.i for c in m.children if c.dep_ == "conj" for t in c.subtree}
            return {t for t in m.subtree if t.i not in drop}

        def _part(m) -> str:
            head_words = " ".join(
                t.text for t in sorted(_own(m), key=lambda t: t.i)
                if t in span and t.lower_ not in _CONJ_WORDS
                and t.pos_ not in ("PUNCT",))
            return f"{head_words} {tail_text}".strip()

        parts = [_part(m) for m in [mod] + sub]
        return Split(parts, "elided_head", model) if all(map(_ok, parts)) else None
    return None


class SpanSplitter:
    """Lazy-loading splitter. Reports whether it is actually able to run."""

    def __init__(self, model: str = PREFERRED_MODEL, allow_fallback: bool = False):
        self.requested = model
        self.model: Optional[str] = None
        self.error: Optional[str] = None
        self._nlp = None
        self._load(model, allow_fallback)

    def _load(self, model: str, allow_fallback: bool) -> None:
        try:
            import spacy
        except Exception as exc:
            self.error = f"spaCy not importable: {exc}"
            return
        try:
            self._nlp = spacy.load(model)
            self.model = model
            return
        except Exception as exc:
            self.error = f"{model} unavailable: {exc}"
        if allow_fallback:
            # Deliberately opt-in. See the module docstring: sm turns parse
            # disagreement into invented entities on this operation.
            try:
                self._nlp = spacy.load("en_core_web_sm")
                self.model = "en_core_web_sm"
                logger.warning("splitting with en_core_web_sm at the caller's "
                               "request; it mis-splits proper names")
            except Exception as exc:
                self.error = f"{self.error}; en_core_web_sm also unavailable: {exc}"
        if self._nlp is None:
            logger.warning("span splitting is OFF — %s. Install with: "
                           "python -m spacy download %s", self.error, model)

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def split_all(self, sentence: str,
                  spans: Sequence[tuple]) -> dict:
        """``spans`` is [(key, start_char, end_char)]; returns key -> Split."""
        if not self.available or not sentence:
            return {}
        doc = self._nlp(sentence)
        out = {}
        for key, start, end in spans:
            if start is None or end is None:
                continue
            sp = doc.char_span(int(start), int(end), alignment_mode="expand")
            if sp is None:
                continue
            s = split_span(sp, self.model or "")
            if s:
                out[key] = s
        return out
