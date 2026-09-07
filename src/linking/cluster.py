"""
Give project-local identities to mentions that no authority could resolve.

Two thirds of this corpus grounds to nothing — the ecology vocabulary
(`species richness`, `biodiversity`, `abundance`, `presence`) is absent from
ENVO, PATO, GO, ECOCORE, OBA and GBIF alike. Those mentions still have to become
entities or they are not in the graph at all, so they get identities minted
here instead of borrowed from an authority.

Why this is not a similarity problem
------------------------------------
Clustering unresolved mentions is normally approximate: embed, threshold, and
hope antonyms do not collide. None of that is needed here, and the reason is the
granularity decision rather than any cleverness.

Node granularity is **modifier-preserving**: `high temperatures` and
`temperature` are deliberately different entities, joined later by a
broader/narrower edge rather than merged. So the only thing left for clustering
to do is recognise *lexical variants of the same phrase* — plural, case,
punctuation, word order. That is an exact-key problem:

    key = (entity type, sorted multiset of lemmatised content words)

Measured over the human gold, this collapses 2126 unresolved mentions to 1197
entities, and every one of the 73 clusters that merges more than one surface
form is a genuine variant:

    broadleaf evergreen forest / evergreen broadleaf forests / broadleaf evergreen forests
    Bidoup—Nui Ba National Park / Bidoup Nui Ba National Park      (em dash)
    Annamites / Annamite / Annamites:                              (plural, punctuation)
    populations / population        dispersal / dispersals

Sorting the words is what handles word order; lemmatising handles plurals; the
normaliser handles case, dashes and diacritics. Nothing merges across a modifier
difference, so `bird species`, `bat species` and `species` stay three entities,
as do `temperature`, `high temperatures` and `peak water temperature`.

Entity type is part of the key because type is part of entity identity
(decision, 7 Sep): `biodiversity` as BIOTIC PROPERTY and as BIOTIC ENTITY are
two entities.

Why ids are content-derived rather than sequential
--------------------------------------------------
``mobiko:e<hash>`` is a digest of the key itself, so the same phrase gets the
same identity in every run and across corpora, independent of iteration order or
of which documents happened to be processed. A sequential counter would renumber
every entity whenever a document was added, which would silently break any
downstream join — including the relation mapping that is meant to attach later
by ``uid``.

What is deliberately NOT clustered
----------------------------------
AMBIGUOUS mentions. Those have candidate concepts and were declined on purpose;
minting a project-local entity for them would assert that all eight uses of
"the lake" are one lake, which is exactly the guess the matcher refused to make.
They stay unresolved until a tier with sentence context, or resolved
coreference, can decide.
"""
from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.linking.mention_norm import normalize_key

logger = logging.getLogger(__name__)

ID_PREFIX = "mobiko:e"
ID_LEN = 12

# Determiners only. Quantifiers ("all", "many", "several") are NOT stripped:
# the schema types them as QUANTITATIVE PROPERTY in their own right, so treating
# "all species" as "species" would discard a distinction the annotators drew.
_DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those"})

_wn = None


def _lemma(token: str) -> str:
    """Noun lemma, falling back to the token itself.

    Morphy handles the irregular plurals a naive rule breaks on (`taxa`,
    `larvae`, `genera`) but not all of them — `fungi` and `algae` come back
    unchanged. Left as-is rather than patched here: a wrong lemma would merge
    two entities, and an unmerged variant is the recoverable error.
    """
    global _wn
    if _wn is None:
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test", pos=wn.NOUN)
            _wn = wn
        except Exception as exc:
            logger.warning("WordNet unavailable (%s); plural variants will not "
                           "merge and clusters will be over-split", exc)
            _wn = False
    if not _wn:
        return token
    return _wn.morphy(token, _wn.NOUN) or token


def content_key(text: str, entity_type: str) -> Tuple[str, Tuple[str, ...]]:
    """The identity key: (entity type, sorted lemmatised content words)."""
    tokens = [t for t in normalize_key(text).replace("-", " ").split() if t]
    while tokens and tokens[0] in _DETERMINERS:
        tokens = tokens[1:]
    return ((entity_type or "").strip().upper(),
            tuple(sorted(_lemma(t) for t in tokens)))


def entity_id(key: Tuple[str, Tuple[str, ...]]) -> str:
    """Stable id for a key. Same phrase, same id, in any run or corpus."""
    payload = f"{key[0]}|{'+'.join(key[1])}".encode("utf-8")
    return ID_PREFIX + hashlib.sha256(payload).hexdigest()[:ID_LEN]


@dataclass
class Cluster:
    entity_id: str
    entity_type: str
    key: Tuple[str, ...]
    label: str                              # most frequent surface form
    surface_forms: Counter = field(default_factory=Counter)
    mention_uids: List[str] = field(default_factory=list)
    doc_ids: set = field(default_factory=set)

    @property
    def n_mentions(self) -> int:
        return sum(self.surface_forms.values())

    @property
    def doc_frequency(self) -> int:
        return len(self.doc_ids)


def build_clusters(mentions: Iterable[Tuple[str, str, str, str]]) -> Dict[str, Cluster]:
    """Cluster unresolved mentions.

    ``mentions`` yields (uid, surface text, entity type, doc id). Returns
    entity_id -> Cluster.
    """
    clusters: Dict[str, Cluster] = {}
    for uid, text, etype, doc_id in mentions:
        key = content_key(text, etype)
        if not key[1]:
            continue
        eid = entity_id(key)
        c = clusters.get(eid)
        if c is None:
            c = clusters[eid] = Cluster(entity_id=eid, entity_type=key[0],
                                        key=key[1], label=text)
        c.surface_forms[text] += 1
        c.mention_uids.append(uid)
        if doc_id:
            c.doc_ids.add(doc_id)
    for c in clusters.values():
        # Most frequent surface form, ties broken alphabetically so the label is
        # reproducible rather than dependent on insertion order.
        c.label = min(c.surface_forms.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return clusters


def rank_by_document_frequency(clusters: Dict[str, Cluster]) -> List[Cluster]:
    """Clusters ordered as an ontology-gap report.

    Document frequency rather than mention count: a phrase used ten times in one
    abstract is that author's habit, whereas one used once in ten papers is
    domain vocabulary the ontologies are missing. This ranking is the artefact
    worth taking back to ENVO and GMBA.
    """
    return sorted(clusters.values(),
                  key=lambda c: (-c.doc_frequency, -c.n_mentions, c.entity_id))
