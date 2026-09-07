"""
WordNet as negative evidence. Never as a source of links.

WordNet's biodiversity coverage is thin and it was frozen in 2006 — `species
richness`, `defaunation` and `permafrost thaw` are simply absent. So absence
from WordNet is never evidence of anything, and nothing here ever *creates* a
link. It only ever declines one.

What it is used for
-------------------
**Deciding whether a mention is generic.** GBIF registers ordinary English
words as vernacular names on particular species: "fish" is recorded against
*Gadus morhua*, "beetle" against *Biocrypta curtipennis*. Matching is perfectly
correct at the string level and completely wrong at the semantic one, and no
amount of string evidence can tell the difference — the failing mention and the
succeeding one are structurally identical. Compare, from the corpus:

    fish                     -> Gadus morhua           species   WRONG
    annamite striped rabbit  -> Nesolagus timminsi      species   RIGHT

Both are single species-rank matches reached through a vernacular alias. What
separates them is a fact about the *mention*, not the candidate: "fish" names a
superordinate category and "annamite striped rabbit" names one animal.

WordNet's hyponym closure — the transitive count of "kinds of X" — measures
exactly that:

    mammals 1169   birds 871   fish 617   insects 454   bacteria 87
    beetle    62   bats  40    trout   5   dung beetle 3
    chironomidae, pinus kesiya, annamite striped rabbit,
    large-antlered muntjac, collared laughingthrush   all exactly 0

The useful property is NOT that generic and specific terms occupy disjoint
ranges — they do not. Checked across all 225 unique BIOTIC ENTITY forms in the
human gold, `tetrapods` and `leafhoppers` are plainly generic and score 1. What
does hold, checked over the whole population where this guard can fail open, is
narrower and is the thing actually relied on:

    every mention that reaches a species-rank concept scores exactly 0,
    and all 32 such links in the gold are correct

(Pinus kesiya, Nesolagus timminsi, gaur -> Bos frontalis, Atlas cedar ->
Cedrus atlantica, black-shanked douc -> Pygathrix nigripes, and so on.) The
sub-threshold generics never reach species rank in the first place: `tetrapods`
is NIL, `leafhoppers` resolves to Cicadellidae at family rank, `predator` to an
ECOCORE concept with no rank at all.

Note what this does NOT say. Being generic is not itself disqualifying:
`insects` -> Insecta and `mammals` -> Mammalia are right, and both mentions are
maximally generic. Genericness only matters *against a specific candidate*. The
rule the matcher applies is the conjunction — a generic mention must not be
resolved to a species — and it is the conjunction that makes every case above
come out correctly.

Note also that plain membership would have been useless here: `chironomidae`,
`culicidae` and `ephemeroptera` are all WordNet nouns. It is the shape of the
hyponym tree that carries the signal, not the presence of an entry.
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Set at 1, i.e. "has any hyponym at all". Every one of the 32 mentions in the
# gold that reaches a species-rank concept scores exactly 0, so a nonzero count
# is free to block: the strictest setting costs nothing measurable and is the
# safe side of the residual risk below. A higher value was tried first on the
# mistaken belief that generic terms had a floor around 3; they do not.
#
# Residual risk, unmitigated and worth stating: a term that is generic but
# carries NO WordNet hyponyms — because WordNet does not know it, or records it
# as a leaf — and which GBIF happens to hold as a species-level vernacular,
# would still slip through. Nothing in this corpus does, but the corpus is 154
# sentences. Widening it is a job for a bigger sample, not a bigger threshold.
GENERIC_HYPONYM_THRESHOLD = 1

# Ranks at which a concept denotes one particular organism rather than a group.
SPECIFIC_RANKS = frozenset({"species", "subspecies", "variety", "form"})

_wn = None
_available: Optional[bool] = None


def available() -> bool:
    """Whether the WordNet corpus is actually loadable.

    Resolved once and reported, never assumed. A guard that silently does
    nothing because its data is missing would turn every blocked link back into
    a confident wrong answer, with no trace in the run record — the same failure
    mode `hints_effective` exists to prevent elsewhere in this repo.
    """
    global _wn, _available
    if _available is not None:
        return _available
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test", pos=wn.NOUN)   # forces the lazy corpus load
        _wn = wn
        _available = True
    except Exception as exc:
        logger.warning("WordNet unavailable (%s); the genericness guard is OFF. "
                       "Install with: python -m nltk.downloader wordnet", exc)
        _available = False
    return _available


@functools.lru_cache(maxsize=100_000)
def hyponym_count(term: str) -> int:
    """Size of the transitive hyponym closure, over the term's noun senses.

    The maximum across senses, not the sum: a word with one broad sense is
    generic even if its other senses are narrow. "fish" is generic because of
    the animal sense, whatever the others do.
    """
    if not term or not available():
        return 0
    key = term.strip().replace(" ", "_")
    if not key:
        return 0
    best = 0
    for syn in _wn.synsets(key, pos=_wn.NOUN):
        best = max(best, sum(1 for _ in syn.closure(lambda s: s.hyponyms())))
    return best


def is_generic(term: str, threshold: int = GENERIC_HYPONYM_THRESHOLD) -> bool:
    """Does this mention name a category rather than a particular thing?"""
    return hyponym_count(term) >= threshold


def blocks_link(term: str, candidate_rank: Optional[str],
                matched_scope: Optional[str] = None) -> bool:
    """Should this mention be refused a link to this candidate?

    The conjunction described in the module docstring: a generic mention
    resolved to a concept denoting one particular organism. Returns False when
    WordNet is unavailable, so a missing corpus degrades to the old behaviour
    rather than silently rejecting everything.
    """
    if not available():
        return False
    if (candidate_rank or "").strip().lower() not in SPECIFIC_RANKS:
        return False
    return is_generic(term)
