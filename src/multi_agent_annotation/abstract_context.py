"""
Map each annotation sentence back to the abstract it came from.

WHY. The blind-pair pipeline gives the Critic the whole abstract while the two
Annotators see only the sentence. That is the one genuine information asymmetry
in the design: the Annotators and the Critic then know different things, rather
than reading the same guideline rendered two ways, which is what the old
decision-table/narrative-guideline split amounted to.

The grid input (data/manually_labeled_last/overlap_sentences.txt) is a flat
sentence list with no document grouping, so the grouping is recovered by
matching each sentence into data/abstracts/abstracts.txt. Measured on the
155-sentence grid input: 15 abstracts, every sentence matching exactly one.

COST. The abstracts average 4,231 characters, but that is the wrong figure to
budget with: long abstracts contribute more sentences, so the mean context a
Critic turn actually receives is 10,730 characters, about 2,700 tokens. That is
added to every Critic call, twice (its blind pass and its revision).

Six of those sentences carry a literal "[abstract]" prefix — the document
delimiter leaked into the sentence text during extraction — which is why
_strip_marker exists. Without it those six match nothing, and a Critic silently
receiving no context on the first sentence of six abstracts would be an
asymmetry nobody declared.

Matching is on normalised text via label_norm, the same normaliser the rest of
the pipeline uses, so a sentence that differs from its abstract only in
whitespace still resolves.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from label_norm import normalize_text as _norm

ABSTRACT_DELIMITER = re.compile(r"\[abstract\]")
_MARKER = re.compile(r"^\s*\[abstract\]\s*")

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ABSTRACTS = _THIS_DIR.parents[1] / "data" / "abstracts" / "abstracts.txt"


def _strip_marker(text: str) -> str:
    return _MARKER.sub("", text or "").strip()


def load_abstracts(path: Optional[Path] = None) -> List[str]:
    """The abstracts, split on the [abstract] delimiter. [] if unreadable."""
    p = Path(path) if path else DEFAULT_ABSTRACTS
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return []
    return [d.strip() for d in ABSTRACT_DELIMITER.split(raw) if d.strip()]


class AbstractIndex:
    """Sentence -> containing abstract.

    Resolution is reported, never guessed silently: `unresolved` and `ambiguous`
    are counted so a run can state how many sentences the Critic actually
    received context for instead of leaving it to be inferred.
    """

    def __init__(self, abstracts: Optional[List[str]] = None,
                 path: Optional[Path] = None):
        self.abstracts = abstracts if abstracts is not None else load_abstracts(path)
        self._normed = [_norm(a) for a in self.abstracts]
        self._cache: Dict[str, Optional[int]] = {}
        self.unresolved: List[str] = []
        self.ambiguous: List[str] = []

    def index_of(self, sentence: str) -> Optional[int]:
        key = _norm(_strip_marker(sentence))
        if key in self._cache:
            return self._cache[key]
        hits = [i for i, a in enumerate(self._normed) if key and key in a]
        if len(hits) == 1:
            idx: Optional[int] = hits[0]
        elif not hits:
            idx = None
            self.unresolved.append(sentence[:80])
        else:
            # Pick the longest abstract containing it, and record that a choice
            # was made. Two abstracts sharing a sentence verbatim would
            # otherwise resolve by list order, which is not a reason.
            idx = max(hits, key=lambda i: len(self.abstracts[i]))
            self.ambiguous.append(sentence[:80])
        self._cache[key] = idx
        return idx

    def context_for(self, sentence: str) -> Optional[str]:
        idx = self.index_of(sentence)
        return self.abstracts[idx] if idx is not None else None

    def stats(self) -> Dict[str, int]:
        return {"abstracts": len(self.abstracts),
                "unresolved": len(self.unresolved),
                "ambiguous": len(self.ambiguous)}


# ─────────────────────────── matched context window ───────────────────────────
#
# WHY A WINDOW RATHER THAN THE WHOLE ABSTRACT. The headline measurement is
# agent-agent agreement against the two-human ceiling of 0.497, and that
# comparison is only valid if the agents annotate under the humans' condition.
# Mark and Davnah annotated one sentence at a time and could navigate to the
# previous and following sentences. So neither extreme matches them: a
# sentence-only agent is handicapped relative to a human who could look at the
# neighbours, and a whole-abstract agent is better informed than the ceiling it
# is being compared against.
#
# The window is given to EVERY agent — both Annotators, the Critic and the
# Adjudicator. An earlier version of this design gave the abstract to the Critic
# alone as a deliberate information asymmetry; that made the three votes in the
# majority rule non-exchangeable, so a 2-1 majority could mean "two less-informed
# agents outvote one better-informed agent". Uniform context removes that, and
# leaves sampling as the only thing distinguishing the annotators.
#
# ONLY THE NEIGHBOURS ARE APPROXIMATE. The target sentence is located verbatim,
# so the annotation target is always exact; the surrounding sentences are cut on
# a terminator regex, which biodiversity text (abbreviations, decimals, binomial
# names) will occasionally cut badly. A slightly over- or under-trimmed
# neighbour costs nothing, unlike a mis-identified target.

_TERMINATOR = re.compile(r"(?<=[.!?])\s+")


def _sentence_bounds(abstract: str) -> List[Tuple[int, int]]:
    """Character spans of the abstract's sentences, split on terminators."""
    bounds, start = [], 0
    for m in _TERMINATOR.finditer(abstract):
        bounds.append((start, m.start()))
        start = m.end()
    if start < len(abstract):
        bounds.append((start, len(abstract)))
    return bounds


class ContextWindow:
    """The matched-condition context for a sentence: itself plus n neighbours."""

    def __init__(self, index: Optional[AbstractIndex] = None, radius: int = 1):
        self.index = index or AbstractIndex()
        self.radius = radius
        self.no_context: List[str] = []
        self.one_sided: List[str] = []

    def for_sentence(self, sentence: str) -> Dict[str, Any]:
        """-> {before, after, window, n_before, n_after}.

        `window` is the renderable block: neighbours plus the target, in reading
        order. `before`/`after` are kept separately so a prompt can mark which
        sentence is the one being annotated — a window that does not say which
        sentence is the target invites annotation of the neighbours.
        """
        abstract = self.index.context_for(sentence)
        target = _strip_marker(sentence)
        empty = {"before": [], "after": [], "window": target,
                 "n_before": 0, "n_after": 0}
        if not abstract:
            self.no_context.append(sentence[:80])
            return empty

        bounds = _sentence_bounds(abstract)
        key = _norm(target)
        hit = next((i for i, (s, e) in enumerate(bounds)
                    if key and key in _norm(abstract[s:e])), None)
        if hit is None:
            # The target spans a terminator the splitter put inside it (a decimal
            # or an abbreviation). Fall back to a character window around the
            # verbatim match so context is still provided.
            occ = abstract.find(target)
            if occ < 0:
                self.no_context.append(sentence[:80])
                return empty
            before = [abstract[max(0, occ - 400):occ].strip()] if occ else []
            tail = abstract[occ + len(target):occ + len(target) + 400].strip()
            after = [tail] if tail else []
        else:
            lo, hi = max(0, hit - self.radius), min(len(bounds), hit + self.radius + 1)
            before = [abstract[s:e].strip() for s, e in bounds[lo:hit]]
            after = [abstract[s:e].strip() for s, e in bounds[hit + 1:hi]]

        before = [b for b in before if b]
        after = [a for a in after if a]
        if not before or not after:
            self.one_sided.append(sentence[:80])
        return {"before": before, "after": after,
                "window": " ".join(before + [target] + after).strip(),
                "n_before": len(before), "n_after": len(after)}

    def stats(self) -> Dict[str, int]:
        return {"no_context": len(self.no_context),
                "one_sided": len(self.one_sided)}
