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

Identity is the FULL entity label; the ontology link is what is shared
----------------------------------------------------------------------
The label schema has two layers (MoBiKo_label_guidance_v4, "Step 2"): an
ontological **role** — entity, process, property, concept — and a **domain** —
abiotic, biotic, anthropogenic, spatial, temporal, quantitative, qualitative.
"BIOTIC ENTITY" is the pair.

Identity keys on the **pair**. `habitat` typed ABIOTIC ENTITY, BIOTIC ENTITY and
SPATIAL ENTITY is three nodes, one per label — and all three carry the same
``authority_uri``, ENVO:01000739, so the ontological entity they share is
explicit rather than implied (decision, 7 Sep, superseding the role-only key
tried earlier the same day). In RDF that is three nodes with `skos:exactMatch`
to one ENVO term.

The role/domain split is still computed and exposed (``entity_role``,
``entity_domain``), because the distinction is real and analyses want it; it
just does not decide identity. Keying on the role alone merged 213 mentions that
the domain distinguishes, and the merge was not recoverable afterwards, whereas
keeping them apart and sharing an authority_uri is recoverable in both
directions.

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

# The ontological roles of the schema. A label's LAST word carries the role;
# everything before it is the domain. CONCEPT takes no domain.
ROLES = frozenset({"ENTITY", "PROCESS", "PROPERTY", "CONCEPT"})

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


# Trimmed from the head token before lemmatising. `normalize_key` strips
# punctuation only from the ends of the WHOLE string, so "Annamites:" arrives
# with the colon still attached to the word Morphy has to recognise.
_HEAD_TRIM = " \t\n.,;:!?()[]{}\"'`"


def singular_form(text: str) -> Optional[str]:
    """The mention with its HEAD noun singularised, or None if it is unchanged.

    A surface form, not a key — `match.resolve` must be given surface text.
    **The capitalisation is preserved deliberately.** It is what the proper-name
    rule reads, and folding it here would hide the exact failure morphology was
    demoted for: `landscapes` has to arrive as lowercase `landscape` so that it
    can still be refused against the settlement named Landscape.

    Only the head is touched. Lemmatising every token is right for the identity
    key, where the words are an unordered multiset, and wrong here, where the
    result has to be a phrase an authority might actually list.
    """
    tokens = (text or "").split()
    if not tokens:
        return None
    head = tokens[-1].strip(_HEAD_TRIM)
    if not head:
        return None
    lemma = _lemma(head.lower())
    if lemma == head.lower():
        return None
    if head[:1].isupper():
        lemma = lemma[:1].upper() + lemma[1:]
    return " ".join(tokens[:-1] + [lemma])


def split_entity_type(entity_type: str) -> Tuple[str, Optional[str]]:
    """(role, domain) for a canonicalised label. "BIOTIC ENTITY" -> ("ENTITY", "BIOTIC").

    An off-schema label whose last word is not a known role is returned whole as
    the role, with no domain. That keeps it distinct rather than guessing which
    role was meant — the same choice `label_norm.canonicalize_entity_type` makes
    when it surfaces an unknown label instead of repairing it. The gold carries
    a handful: NEGATION, CHANGE, QUANTITATIVE CHANGE, COMPARATIVE PROPERTY.
    """
    label = (entity_type or "").strip().upper()
    if not label:
        return "", None
    head, _, last = label.rpartition(" ")
    if last in ROLES:
        return last, (head or None)
    return label, None


def content_key(text: str, entity_type: str) -> Tuple[str, Tuple[str, ...]]:
    """The identity key: (full entity label, sorted lemmatised content words)."""
    tokens = [t for t in normalize_key(text).replace("-", " ").split() if t]
    while tokens and tokens[0] in _DETERMINERS:
        tokens = tokens[1:]
    return ((entity_type or "").strip().upper(),
            tuple(sorted(_lemma(t) for t in tokens)))


def surface_key(text: str) -> Tuple[str, ...]:
    """The words alone, with no label. Groups a phrase across its labels.

    `habitat` under three labels is three entities by design, but an analysis
    that wants "every habitat node" needs a handle on them. For grounded
    entities the shared ``authority_uri`` already provides one; this is the
    equivalent for entities no authority covers.
    """
    return content_key(text, "")[1]


def entity_id(key: Tuple[str, Tuple[str, ...]]) -> str:
    """Stable id for a key. Same phrase, same id, in any run or corpus."""
    payload = f"{key[0]}|{'+'.join(key[1])}".encode("utf-8")
    return ID_PREFIX + hashlib.sha256(payload).hexdigest()[:ID_LEN]


@dataclass
class Cluster:
    entity_id: str
    role: str
    key: Tuple[str, ...]
    label: str                              # most frequent surface form
    entity_types: Counter = field(default_factory=Counter)   # every label seen
    surface_forms: Counter = field(default_factory=Counter)
    mention_uids: List[str] = field(default_factory=list)
    doc_ids: set = field(default_factory=set)

    @property
    def n_mentions(self) -> int:
        return sum(self.surface_forms.values())

    @property
    def doc_frequency(self) -> int:
        return len(self.doc_ids)

    @property
    def entity_type(self) -> str:
        """The cluster's label. Identity keys on it, so there is exactly one.

        Read off ``entity_types`` rather than stored separately because the key
        already carries it; the counter is kept for the mention counts. Ties are
        broken alphabetically for the same reason ``label`` breaks them that
        way — the gap report has to be byte-reproducible.
        """
        if not self.entity_types:
            return ""
        return min(self.entity_types.items(), key=lambda kv: (-kv[1], kv[0]))[0]


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
            c = clusters[eid] = Cluster(entity_id=eid, role=split_entity_type(key[0])[0],
                                        key=key[1], label=text)
        c.entity_types[(etype or "").strip().upper()] += 1
        c.surface_forms[text] += 1
        c.mention_uids.append(uid)
        if doc_id:
            c.doc_ids.add(doc_id)
    for c in clusters.values():
        # Most frequent surface form, ties broken alphabetically so the label is
        # reproducible rather than dependent on insertion order.
        c.label = min(c.surface_forms.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return clusters


class _Union:
    """Minimal union-find over hashable signatures."""

    def __init__(self) -> None:
        self.parent: Dict[object, object] = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Order by repr so the representative does not depend on insertion
            # order — the table has to be byte-reproducible.
            lo, hi = sorted((ra, rb), key=repr)
            self.parent[hi] = lo


def link_signatures(signatures: Iterable[Tuple[object, object]]) -> Dict[object, object]:
    """Group signatures that co-occur on the same mention.

    A mention can carry two identity signatures: its content key, and the
    authority concept it grounded to. Either alone is incomplete.

      * Content key alone loses what grounding merges. `Vietnam` and `Viet Nam`
        have different words and are one place; `Diptera` and `flies` likewise.
      * Grounding alone loses what the key merges, and splits on an accident.
        `mountains` typed ABIOTIC ENTITY grounds to ENVO while `mountains`
        typed SPATIAL ENTITY does not, so the same word in the same role became
        two entities — 26 mentions of the gold did this.

    Treating both as evidence of one identity and taking connected components
    uses each where it is strong. Returns signature -> component representative.
    """
    u = _Union()
    for a, b in signatures:
        u.union(a, a)
        if b is not None:
            u.union(a, b)
    return {sig: u.find(sig) for sig in list(u.parent)}


def rank_by_document_frequency(clusters: Dict[str, Cluster]) -> List[Cluster]:
    """Clusters ordered as an ontology-gap report.

    Document frequency rather than mention count: a phrase used ten times in one
    abstract is that author's habit, whereas one used once in ten papers is
    domain vocabulary the ontologies are missing. This ranking is the artefact
    worth taking back to ENVO and GMBA.
    """
    return sorted(clusters.values(),
                  key=lambda c: (-c.doc_frequency, -c.n_mentions, c.entity_id))
