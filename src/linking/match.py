"""
Tier A matching: decide, per mention, which authority concept it denotes — or
decline to decide.

The outcome is one of four, and keeping them apart is the whole point of this
module:

    LINKED      one concept survived the policy
    AMBIGUOUS   candidates were found and none could be chosen between
    NIL         routed and searched, nothing matched
    UNROUTED    no authority serves this entity type at all

An earlier version of this code did ``LIMIT 1`` and reported the first row. That
is not a simplification of the above, it is a different and worse thing: it
silently converts every ambiguity into a confident answer. Measured on the human
gold, it made `diptera` (16 mentions, the most frequent link of its type)
resolve to *Saxifraga*, a plant genus that happens to carry *Diptera* as an
internal name, and `fish` resolve to *Gadus morhua*. NIL and AMBIGUOUS must also
stay apart from each other: "nothing in the vocabulary" and "several things and
we cannot tell which" call for completely different fixes, and averaging them
into one "unlinked" rate hides both.

The policy, in order
--------------------
1. **Route by the mention's OWN entity type.** No authority serves it ->
   UNROUTED, which is not a miss. One type per resolution: entity type is part
   of entity identity, so `biodiversity` as BIOTIC PROPERTY and `biodiversity`
   as BIOTIC ENTITY are two entities, not one entity two annotators disagreed
   about.
2. **Look up every candidate** on identity-bearing scopes (label, exact,
   vernacular). Never one.
3. **Refuse to bind a common noun to a named individual.** "the lake" is not
   unresolvable, it is under-specified: it denotes a particular lake named
   earlier in the text. That is a coreference question, and answering it by
   spelling match produced `lake` -> a hamlet called "Lake", `landscape` ->
   "Landscape", `center` -> "Zou Department".
4. **Refuse to choose across authorities.** Candidates from two different
   sources are two different kinds of claim and there is no principled
   tiebreak between them, so they are AMBIGUOUS. Getting this wrong is not
   hypothetical: rank preference is defined only inside GBIF, so ranking a
   GBIF genus (5) against an ENVO concept with no rank (99) made *every* GBIF
   hit beat *every* ontology hit. That silently turned `biota` from the
   correct ENVO:06105012 into *Platycladus*, a conifer genus formerly named
   *Biota*.
5. **Prefer the highest taxonomic rank**, within one source. This settles the
   homonyms:
   `diptera` = Saxifraga(genus) vs Diptera(**order**); `bats` =
   Chiroptera(**order**) vs Vespertilionidae(family); `bacteria` =
   Bacteria(genus) vs Bacteria(**kingdom**). A mention that matches two ranks of
   the same name almost always means the broader one in running prose.
6. **Then prefer the stronger alias scope**, label > exact > vernacular — but
   only for class-naming sources. Two places are not a name and its synonym.
7. **Refuse a generic mention resolved to one species.** See
   ``wordnet_guard`` — this is what stops `fish`, `beetle` and `dung beetle`
   without also stopping `insects` -> Insecta or
   `annamite striped rabbit` -> *Nesolagus timminsi*.
8. **Anything still tied is AMBIGUOUS.** Handed up a tier, not guessed at.

Steps 3-7 are deterministic and offline. No GPU, no API, no threshold.

On the candidate cap
--------------------
The lookup asks for up to ``CANDIDATE_CAP`` rows and the SQL carries no ORDER
BY, so a truncated set is an ARBITRARY subset, not a best-N. Preferring the
highest rank out of an arbitrary subset is a coin flip wearing a policy's
clothes: `beetle` has 65 candidates and an earlier cap of 50 returned the right
answer only because Coleoptera happened to fall inside the window. The cap is
therefore set well above any real fan-out, and hitting it is reported as
AMBIGUOUS rather than decided on partial evidence.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.linking.dictionary import CONFIDENT_SCOPES, Dictionary, Match
from src.linking.mention_norm import normalize_key
from src.linking import wordnet_guard

# Broadest first. Anything unrecognised (an OBO namespace, a GeoNames feature
# code, None) sorts last, so rank preference is inert outside GBIF rather than
# arbitrary.
RANK_ORDER: Dict[str, int] = {
    "kingdom": 0, "phylum": 1, "class": 2, "order": 3, "family": 4,
    "genus": 5, "species": 6, "subspecies": 7, "variety": 8, "form": 9,
}
_RANK_UNKNOWN = 99

SCOPE_ORDER: Dict[str, int] = {"label": 0, "exact": 1, "vernacular": 2}
_SCOPE_UNKNOWN = 99

# Sources whose concepts are INDIVIDUALS — one particular place — rather than
# classes. GeoNames holds 1.38M named features and some are named with ordinary
# English words ("Lake", "Landscape", "Highland"), so a bare common noun will
# match one and look like a confident answer.
#
# The deeper point is that such a mention is not unresolvable, it is
# UNDER-SPECIFIED: "the lake" denotes a particular lake named earlier in the
# text. Its referent is a coreference question, not a lookup question, and this
# module has no access to the antecedent. So it is reported as ambiguous with
# that reason rather than silently bound to whichever feature shares its
# spelling. `add_coref_annotations.py` already resolves anaphora over these
# documents; feeding those antecedents in is the principled fix, and is exactly
# what this outcome is flagging as needed.
INDIVIDUAL_SOURCES = frozenset({"geonames"})

# Far above any observed fan-out (the widest in the human gold is `beetle` at
# 65). Saturating it means the candidate set could not be enumerated, which is
# reported rather than decided on. See "On the candidate cap".
CANDIDATE_CAP = 1000


class Outcome(str, enum.Enum):
    LINKED = "linked"
    AMBIGUOUS = "ambiguous"
    NIL = "nil"
    UNROUTED = "unrouted"


@dataclass
class Resolution:
    """What the matcher concluded, and why.

    ``reason`` is always populated for a non-LINKED outcome. A decision that
    cannot explain itself cannot be audited, and the unlinked set is the input
    to NIL clustering later, where "ambiguous, 10 candidate species" and
    "nothing in the vocabulary" have to be told apart.
    """
    text: str
    norm: str
    entity_type: str
    outcome: Outcome
    match: Optional[Match] = None
    candidates: List[Match] = field(default_factory=list)
    reason: str = ""
    decided_by: str = "tier_a"

    @property
    def uri(self) -> Optional[str]:
        return self.match.uri if self.match else None


def _best_by(matches: Sequence[Match], key) -> List[Match]:
    """All matches tying for the best value of ``key``. Ties are preserved."""
    best = min(key(m) for m in matches)
    return [m for m in matches if key(m) == best]


def _rank_key(m: Match) -> int:
    return RANK_ORDER.get((m.rank or "").strip().lower(), _RANK_UNKNOWN)


def _scope_key(m: Match) -> int:
    return SCOPE_ORDER.get((m.scope or "").strip().lower(), _SCOPE_UNKNOWN)


def _has_proper_name_token(text: str) -> bool:
    """Does the surface form carry a capitalised word?

    Deliberately reads the ORIGINAL text, not the normalised key, because
    casefolding is what destroys this evidence. `southern Africa` passes on
    *Africa*, so the test is "any token", not "the first token".
    """
    return any(tok[:1].isupper() for tok in (text or "").split())


def resolve(dictionary: Dictionary, text: str, entity_types,
            scopes: Sequence[str] = CONFIDENT_SCOPES,
            limit: int = CANDIDATE_CAP) -> Resolution:
    """Resolve one mention.

    ``text`` must be the mention's SURFACE form, exactly as it appears in the
    document — not a normalised or lowercased key. Capitalisation is evidence
    here: it is what separates "the landscape" from the place named Landscape,
    and casefolding it away is irreversible. Normalisation happens inside.

    ``entity_types`` should normally be ONE type — the type this mention was
    given. Entity type is part of entity identity (decision, 7 Sep): the same
    string under two ontological roles is two entities, so there is nothing to
    reconcile and nothing to union.

    Passing several is an analysis affordance only, for asking "what would this
    resolve to under any of these types". It deliberately crosses entity
    identities and it leaks across ontological roles, so it must not be used on
    the resolution path. Measured over the human gold, unioning changes 135
    mentions and the changes are wrong in kind, not just in degree:

        mountains     as SPATIAL ENTITY   nil      -> ENVO:00000264
        habitat       as SPATIAL ENTITY   nil      -> ENVO:01000739
        location      as SPATIAL ENTITY   nil      -> PATO:0000140  (a property)
        distribution  as CONCEPT          unrouted -> PATO:0000060

    An earlier version of this docstring recommended the union, on the grounds
    that `lakes` is typed ABIOTIC ENTITY 22 times and SPATIAL ENTITY 14 times in
    the gold. That reasoning was wrong: those are 36 mentions of two different
    entities, not 36 mentions of one entity with a disputed type.
    """
    if isinstance(entity_types, str):
        entity_types = [entity_types]
    types = [t for t in entity_types if t]

    sources = sorted({s for t in types for s in dictionary.sources_for(t)})
    norm = normalize_key(text)
    label = "|".join(sorted(set(types)))

    if not sources:
        return Resolution(text, norm, label, Outcome.UNROUTED,
                          reason="no authority serves this entity type")
    if not norm:
        return Resolution(text, norm, label, Outcome.NIL, reason="empty after normalisation")

    candidates = dictionary.lookup(norm, sources, scopes, limit=limit)
    if not candidates:
        return Resolution(text, norm, label, Outcome.NIL,
                          reason=f"no match in {','.join(sources)}")

    # One concept may be reached by several aliases; keep its strongest.
    by_uri: Dict[str, Match] = {}
    for m in candidates:
        cur = by_uri.get(m.uri)
        if cur is None or _scope_key(m) < _scope_key(cur):
            by_uri[m.uri] = m
    distinct = list(by_uri.values())

    if len(candidates) >= limit:
        return Resolution(text, norm, label, Outcome.AMBIGUOUS, candidates=distinct,
                          reason=f"candidate set hit the {limit}-row cap; the "
                                 f"returned subset is arbitrary, so no preference "
                                 f"can be applied to it")

    # Two authorities answering is two different kinds of claim, and there is no
    # principled tiebreak between them — rank is defined only inside GBIF, so
    # comparing across sources silently ranks every taxon above every ontology
    # concept. See the module docstring.
    sources_answering = {m.source for m in distinct}
    if len(sources_answering) > 1:
        return Resolution(text, norm, label, Outcome.AMBIGUOUS, candidates=distinct,
                          reason=f"candidates in {len(sources_answering)} "
                                 f"authorities ({', '.join(sorted(sources_answering))}); "
                                 f"no basis for choosing between them")

    only_source = next(iter(sources_answering))
    names_individuals = only_source in INDIVIDUAL_SOURCES

    survivors = distinct
    if len(survivors) > 1:
        survivors = _best_by(survivors, _rank_key)
    # The label-over-synonym tiebreak says a concept's own name is a stronger
    # claim than its synonym. True for an ontology term; meaningless for two
    # PLACES, which are not a name and a synonym but different things. Applying
    # it there made `lake` resolve to an obscure feature officially named
    # "Lake", beating every "Lake County", for no reason at all.
    if len(survivors) > 1 and not names_individuals:
        survivors = _best_by(survivors, _scope_key)

    if len(survivors) > 1:
        return Resolution(text, norm, label, Outcome.AMBIGUOUS,
                          candidates=distinct,
                          reason=f"{len(survivors)} concepts tie after rank and "
                                 f"scope preference")

    winner = survivors[0]

    if names_individuals and not _has_proper_name_token(text):
        return Resolution(
            text, norm, label, Outcome.AMBIGUOUS, candidates=distinct,
            reason="common-noun mention matched a named individual place; its "
                   "referent depends on coreference, which is not resolved here")

    if wordnet_guard.blocks_link(text, winner.rank, winner.scope):
        return Resolution(
            text, norm, label, Outcome.AMBIGUOUS, candidates=distinct,
            reason=f"generic mention (WordNet hyponyms="
                   f"{wordnet_guard.hyponym_count(text)}) resolved to a "
                   f"{winner.rank}-rank concept; refusing to pick one organism")

    return Resolution(text, norm, label, Outcome.LINKED, match=winner,
                      candidates=distinct,
                      reason="" if len(distinct) == 1 else
                             f"chosen from {len(distinct)} by rank/scope preference")
