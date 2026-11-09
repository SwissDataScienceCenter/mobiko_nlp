"""
Build the knowledge graph: relations from the annotation files, joined to
entities from the resolution table.

Where this sits
---------------
    run JSONL (final_entities + final_relations)
      -> scripts/agent_jsonl_to_relation_input.py   spans with uids, relations by uid
      -> src.linking.resolve_corpus                  mention -> entity, linked to authorities
      -> THIS                                        nodes.jsonl + edges.jsonl
      -> src.graph.load_neo4j / src.graph.render_html

It does not matter whether the run came from the single-pass annotator or the
full Annotator/Critic/Adjudicator pipeline: both write ``final_entities`` and
``final_relations``, and that is all the converter reads. Human annotation
files in the same shape work too.

The join
--------
A relation names its arguments by span ``uid``; the resolution table says which
entity each span is. The key is ``(annotation file name, uid)``, because that is
what ``resolve_corpus`` records (``source_file``, ``uid``) and because a uid is
only unique inside one file. Two annotation files with the same name would make
the key ambiguous, so that is refused rather than guessed.

Concepts and instances: what a relation attaches to
----------------------------------------------------
Linking answers *which concept* a mention is: `elevation` is PATO:0001687 in
every paper. It does not answer *which thing's* elevation. On the
second-iteration run one concept-level `elevation` node collected values from
three different sites in one paper (2743 m, 2456 m, 2601 m) and correlations
from five others, so a two-hop path could join facts no paper stated together.

So relations attach to the **referent** of each mention, one of:

* **global entity** — a mention grounded to a named individual or a taxon
  (GeoNames, GBIF, WDPA). Kilimanjaro is Kilimanjaro in every paper, so the
  ``Entity`` node itself is the referent and papers meet there.
* **instance** — everything else: properties, processes, values, and generic
  nouns ("the sites"). The referent is an ``Instance`` scoped to the document,
  joined to its concept by ``INSTANCE_OF``. Cross-paper questions go through
  that edge, explicitly.
* **bearer-scoped instance** — a mention that is the object of a bearer
  relation (``HAS_PROPERTY``: "Monduli forest reserve HAS_PROPERTY elevation")
  is scoped by its bearer too: *elevation of Monduli forest reserve*, in that
  document. The document stays in the key even then, because two papers
  measure the elevation of one mountain at different points. A mention with
  two or more bearers ("sites A and B differ in elevation") stays at document
  scope rather than be assigned to one arbitrarily; the count is reported.
  Which relations mark a bearer is a flag, because the relation guide is still
  changing.
* **unresolved** — the matcher found candidates and refused to choose
  (`resolve_corpus` docstring, "Why ambiguous mentions get no id"). It becomes
  an ``Unresolved`` node scoped to that one mention: the triplet is kept, the
  identity claim is not made.

* **coreference** — when the annotation file carries ``coref`` links
  (scripts/add_coref_annotations.py), an anaphor that is itself an annotated
  span takes its antecedent's referent: "the mountain" in sentence 9 becomes
  the node for *Mt Kenya* from sentence 4. This is the layer coreference
  belongs to. It was rejected for linking because it would merge CONCEPTS
  across labels ("the habitat" / "coniferous forest"); here it only says which
  thing a paper is talking about, within that paper. Links are accepted
  strictly (see `load_coref`), because a wrong one joins two different
  things; every redirect is recorded on the edge evidence and in the node's
  sentences, and ``--no-coref`` turns it off.

* **shared quality** — a QUALITATIVE PROPERTY mention ("high",
  "significant", "relatively constant") points straight at its shared concept
  node, with no per-paper node (decided 3 Nov 2026 with Luis): "what is high,
  in which papers" is one hop, and each edge still carries the papers and
  sentences it came from, so clicking it shows where the triplet appeared. A
  qualitative mention that is the SUBJECT of a relation ("high ASSOCIATED_WITH
  …") keeps its per-paper node, so a shared quality node only ever receives
  edges and cannot join two papers' facts into one path. These concept nodes
  carry ``value_concept``; ``--no-shared-qualities`` turns this off.

* **ambiguous place** — a place name with several GeoNames referents
  ("Kilimanjaro": the mountain and the region; "Arusha": region and district;
  "Kenya": the country and the mountain) becomes one shared ``AmbiguousPlace``
  node, ``place:kilimanjaro``, joined by ``MAY_BE`` to an Entity for each
  candidate (decided 3 Nov 2026). The mention's relations attach to it, so
  nothing is claimed about which one the paper meant, and a question can still
  reach every Kilimanjaro. ``MAY_BE`` lists the papers that also name that
  candidate outright. Common nouns ("lowlands", "National Park") stay
  Unresolved; ``--no-ambiguous-places`` turns this off.

A relation on a coordination ("eastern and western slopes") is distributed
over the parts, and every resulting triplet is marked ``distributed``.

Relation properties
-------------------
A relation may carry a ``property`` — the annotation page's qualifier on the
relation itself: "Negation" (it does NOT hold), "Negative_polarity" (it holds
in the negative direction), "Exclusive", "Dominance", or a custom one. The
property is part of an edge's identity, so "elevation AFFECTS richness" and the
same triplet under "Negation" are two edges, never one with their evidence
pooled. A negated edge also gets its own graph type, ``NOT_AFFECTS``
(`graph_relation_type`), so a query for AFFECTS cannot return a finding the
paper denies. And a negated bearer relation scopes nothing: "species richness
HAS_PROPERTY heightened [Negation]" says richness is NOT heightened, so it
cannot make "heightened (species richness)".

Two relation instances between the same two referents become ONE edge carrying
both as evidence. ``--scope concept`` restores the concept-only graph (every
mention's referent is its entity), for comparison.

Values are measurements, not entities
-------------------------------------
A mention whose type is a PROPERTY and whose whole text is a value expression
("2743 m", "6–10 µm", "25%", "eight", "p < .001", a coordinate) becomes a
``Measurement`` node of its own, with the number parsed (src/graph/values.py):
``value`` or ``low``/``high``, ``unit``, ``comparator``, ``uncertainty``, and the
raw text. One per mention and never shared — as concepts, 34 values ("25%",
"26", ".001") bridged unrelated papers on second_iteration, and as text none of
them could be compared. Mentions typed as entities ("Five sites", "three
species") stay entities: they are things that happen to be counted.

A relation from a thing to a number is HAS_VALUE (decided with the annotators,
3 Nov 2026); one annotated as HAS_PROPERTY whose object parses as a number or
range is read as HAS_VALUE, with ``relation_as_annotated`` kept on the
evidence. A statistic is not a value of anything, so it is left alone.

Statements carry no test statistics (decided with the annotators, 3 Nov 2026):
whether a finding is significant is enough, and it is already in the relation —
a result reported as not significant is annotated with the property Negation.
The values stay readable in the evidence sentence. Attaching the sentence's
p-values to each claim gave every claim all of them ("soil bulk density
NOT_AFFECTS rodent abundance" listed p = 0.001, 0.01 and 0.06; only 0.06 was
its own).

Node kinds
----------
* ``Entity`` — one per ``entity_id`` in the table: the concept layer.
* ``Measurement`` — one per value mention.
* ``Instance`` — one per (document, entity, bearer) referent.
* ``Unresolved`` — one per ambiguous mention that a relation uses.
* ``AmbiguousPlace`` — one per ambiguous place name, ``MAY_BE`` each candidate.
* ``OntologyTerm`` — one per ``authority_uri``, joined from each Entity grounded
  to it by ``EXACT_MATCH``. This is how `habitat` typed ABIOTIC, BIOTIC and
  SPATIAL ENTITY stays three nodes while visibly being one ENVO term — the
  identity model in UNIFICATION_RULES.md, rendered rather than merged.

Output is deterministic: rows are totally ordered and nothing time-dependent is
written outside the ``.meta.json`` sidecar, so two runs over the same input are
byte-identical.

Usage
-----
    python -m src.graph.build_graph \\
        --annotations output/kg/second_iteration/annotations.json \\
        --resolution  output/kg/second_iteration/resolution.jsonl \\
        --out-dir     output/kg/second_iteration
"""
from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multi_agent_annotation"))
from label_norm import canonicalize_entity_type, canonicalize_relation_property  # noqa: E402

from src.linking.resolve_corpus import DOC_BOUNDARY_RE, _code_version, _iter_records
from src.graph import papers as paper_meta
from src.graph.text_store import sentence_id, short_doc, write_store
from src.graph.values import parse_value

EXACT_MATCH = "EXACT_MATCH"
INSTANCE_OF = "INSTANCE_OF"
MAY_BE = "MAY_BE"
# A GMBA range inside its parent range: an authority's fact, not a paper's.
WITHIN = "WITHIN"
STRUCTURAL = (EXACT_MATCH, INSTANCE_OF, MAY_BE, WITHIN)

SCOPES = ("bearer", "document", "concept")
# The subject of these relations is the bearer of the object: "lake
# HAS_PROPERTY temperature" (relation guide). Measured on second_iteration: 576
# of 577 HAS_PROPERTY relations point that way.
DEFAULT_BEARER_RELATIONS = ("HAS_PROPERTY",)
# Authorities whose concepts are named individuals or taxa, the same in every
# paper. ENVO, PATO, GO and ECOCORE terms are classes, used per paper.
GLOBAL_SOURCES = frozenset({"geonames", "gbif", "wdpa", "gmba"})

_OBO_URI = re.compile(r"/obo/([A-Za-z]+)_(\w+)$")
_GBIF_URI = re.compile(r"gbif\.org/species/(\d+)")
_GEONAMES_URI = re.compile(r"geonames\.org/(\d+)")
_WDPA_URI = re.compile(r"protectedplanet\.net/(\d+)")
_GMBA_URI = re.compile(r"^urn:gmba:v2:(\d+)$")


def curie(uri: Optional[str]) -> Optional[str]:
    """The short form `compile_db` stores, derived from the URI alone.

    ``http://purl.obolibrary.org/obo/ENVO_01000739`` -> ``ENVO:01000739``,
    ``https://www.gbif.org/species/1457`` -> ``gbif:1457``,
    ``https://sws.geonames.org/149590/`` -> ``geonames:149590``.
    An unrecognised URI is returned unchanged rather than guessed at.
    """
    if not uri:
        return None
    for rx, prefix in ((_GBIF_URI, "gbif"), (_GEONAMES_URI, "geonames"),
                       (_WDPA_URI, "wdpa"), (_GMBA_URI, "gmba")):
        m = rx.search(uri)
        if m:
            return f"{prefix}:{m.group(1)}"
    m = _OBO_URI.search(uri)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return uri


def relation_type(label: str) -> str:
    """A relation label as a graph edge type: upper case, ``[A-Z0-9_]`` only.

    The converter already canonicalises, so this is a no-op on every label in
    the relation guide; it exists because Neo4j relationship types cannot be
    parameterised and are spliced into Cypher, so the character set has to be
    guaranteed here rather than assumed.
    """
    t = re.sub(r"[^A-Z0-9_]+", "_", (label or "").strip().upper()).strip("_")
    if not t:
        return "RELATED_TO"
    return t if not t[0].isdigit() else f"R_{t}"


NEGATION = "Negation"


def graph_relation_type(relation: str, prop: str = "") -> str:
    """The edge type a graph store uses: ``AFFECTS``, or ``NOT_AFFECTS`` when negated.

    Only Negation changes the type, because only Negation changes whether the
    relation holds; the other properties qualify a relation that does, and are
    carried as an attribute.
    """
    t = relation_type(relation)
    return f"NOT_{t}" if prop == NEGATION else t


def is_pronoun(text: str) -> bool:
    """A pronoun takes its antecedent's identity and type (decision, 27 Oct):
    annotated only to carry a relation, it is never a concept of its own."""
    return _norm(text) in _PRONOUNS


# Authorities that name places. An ambiguous mention whose candidates all come
# from one of these is a place name with several referents.
PLACE_SOURCES = frozenset({"geonames", "wdpa", "gmba"})


def place_id(key: str) -> str:
    return f"place:{key}"


@lru_cache(maxsize=None)
def _is_common_noun(text: str) -> bool:
    """WordNet knows the phrase only as a class ("national park"), never as a
    named individual ("Kenya", "Kilimanjaro"). Unknown to WordNet ("Arusha",
    "Moshi") is not evidence of anything, so it is not a common noun."""
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(re.sub(r"\s+", "_", text.strip()), pos="n")
    except Exception:       # WordNet missing: fall back to capitalisation alone
        return False
    return bool(synsets) and not any(s.instance_hypernyms() for s in synsets)


def ambiguous_place_key(row: dict) -> Optional[str]:
    """The meta-node key for an ambiguous place name, or None.

    "Kilimanjaro" ties between the mountain and the region; it gets one shared
    ``AmbiguousPlace`` node linked by ``MAY_BE`` to each candidate (decision,
    3 Nov 2026). Only a proper name qualifies: its candidates are all places,
    it has a capitalised word, and it is not a common noun spelt with
    capitals ("National Park"). Lower-case "lowlands" or "site" stays an
    Unresolved mention, because it names whatever the paper means by it.
    """
    cands = row.get("candidates") or []
    if (row["entity_id"] is not None or row.get("outcome") != "ambiguous"
            or row.get("split_into") or len(cands) < 2
            or any(c.get("source") not in PLACE_SOURCES for c in cands)
            or is_pronoun(row["text"])
            or not any(tok[:1].isupper() for tok in row["text"].split())
            or _is_common_noun(row["text"])):
        return None
    return _norm(row["text"]) or None


def mention_key(row: dict) -> str:
    """A mention's id without the input file name: ``<doc_id>:<uid>#<n>``.

    ``mention_id`` starts with the annotation file's stem, which says nothing
    about the paper. Node ids built from it would change when a file is
    renamed or a batch is split differently, so per-mention nodes take the
    paper instead. Two runs over one paper mint the same uid for the same span,
    so they share the node, as they share instances.
    """
    return f"{row['doc_id']}:{row['mention_id'].split(':', 1)[-1]}"


def measurement_id(key: str) -> str:
    return f"value:{key}"


def value_mentions(rows: Iterable[dict]) -> Dict[str, dict]:
    """mention_id -> parsed value, for the mentions that are values.

    A PROPERTY whose whole text parses, or a test statistic whatever its type.
    """
    out: Dict[str, dict] = {}
    for r in rows:
        if r.get("split_into"):
            continue
        parsed = parse_value(r["text"])
        if parsed and (r["entity_role"] == "PROPERTY" or parsed["kind"] == "statistic"):
            out[r["mention_id"]] = parsed
    return out


def unresolved_id(key: str) -> str:
    return f"unresolved:{key}"


def is_global(row: dict) -> bool:
    c = curie(row.get("authority_uri"))
    return bool(c) and c.split(":", 1)[0] in GLOBAL_SOURCES


def instance_id(doc_id: str, entity_id: str, bearer: Optional[str]) -> str:
    payload = f"{doc_id}|{entity_id}|{bearer or ''}".encode("utf-8")
    return "inst:" + hashlib.sha256(payload).hexdigest()[:16]


# ------------------------------------------------------------------ #
#  Loading                                                            #
# ------------------------------------------------------------------ #

def load_resolution(path: Path) -> Dict[Tuple[str, str], List[dict]]:
    """Resolution rows grouped by ``(source_file, uid)``."""
    by_key: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                by_key[(row["source_file"], row["uid"])].append(row)
    return by_key


def load_sentence_table(paths: List[Path], stats: Optional[Counter] = None
                        ) -> Dict[str, Tuple[str, int, str]]:
    """Every sentence of the annotation files: id -> (doc_id, index, text).

    The id is ``<doc_id>/<sent_index>`` with the resolution table's doc_id,
    which adds a ``#NN`` sub-document suffix at each abstract boundary
    (``resolve_corpus.iter_spans``); ``sent_index`` counts within the file's
    record. Two runs over one paper give the same ids; when their text
    differs, the first is kept and the clash counted.
    """
    out: Dict[str, Tuple[str, int, str]] = {}
    for path in paths:
        for obj in _iter_records(path):
            base = obj.get("doc_id") or path.stem
            sentences = obj.get("sentences")
            blocks = sentences if isinstance(sentences, list) else [obj]
            sub_doc = 0
            for sent_i, s in enumerate(blocks):
                if not isinstance(s, dict):
                    continue
                text = s.get("text") or s.get("sentence") or ""
                if DOC_BOUNDARY_RE.match(text):
                    sub_doc += 1
                doc_id = f"{base}#{sub_doc:02d}"
                sid = sentence_id(doc_id, sent_i)
                if sid in out and out[sid][2] != text and stats is not None:
                    stats["sentence_text_conflicts"] += 1
                out.setdefault(sid, (doc_id, sent_i, text))
    return out


def _mention_record(row: dict, node_id: Optional[str], concept_id: Optional[str],
                    antecedent: Optional[dict]) -> dict:
    return {
        "key": mention_key(row), "node_id": node_id, "concept_id": concept_id,
        "sentence_id": sentence_id(row["doc_id"], row["sent_index"]),
        "doc_id": row["doc_id"], "sent_index": row["sent_index"],
        "start_char": row.get("start_char"), "end_char": row.get("end_char"),
        "text": row["text"], "entity_type": row["entity_type"],
        "part_index": row.get("part_index"), "split_from": row.get("split_from"),
        "refers_to_text": antecedent["text"] if antecedent else None,
        "refers_to_sent": antecedent["sent_index"] if antecedent else None,
    }


def iter_relations(path: Path) -> Iterator[dict]:
    """One dict per relation in an annotation file, with its sentence context.

    ``doc_id`` here is the file's own document id. The resolution table also
    splits concatenated abstracts into ``doc#NN`` sub-documents; evidence
    carries the table's form, taken from the endpoint rows, not this one.
    ``_sentence_id`` is the relation's own sentence in ``sentences.sqlite``,
    counted the same way as ``load_sentence_table``.
    """
    for obj in _iter_records(path):
        base = obj.get("doc_id") or path.stem
        sentences = obj.get("sentences")
        blocks = sentences if isinstance(sentences, list) else [obj]
        sub_doc = 0
        for sent_i, s in enumerate(blocks):
            if not isinstance(s, dict):
                continue
            if DOC_BOUNDARY_RE.match(s.get("text") or s.get("sentence") or ""):
                sub_doc += 1
            for rel in s.get("relations") or []:
                if isinstance(rel, dict):
                    yield {**rel, "_sent_index": sent_i,
                           "_sentence_id": sentence_id(f"{base}#{sub_doc:02d}", sent_i)}


# ------------------------------------------------------------------ #
#  Endpoint resolution                                                #
# ------------------------------------------------------------------ #

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_DETERMINER = re.compile(r"(the|this|these|that|those|its|their|such|a|an)\s+", re.I)
_PREPOSITION = re.compile(r"\s+(of|in|on|at|between|from|for|with|across|along|within|"
                          r"under|over|among|to|by|during)\s+", re.I)
_COORDINATION = re.compile(r"\s(and|or)\s", re.I)
_PRONOUNS = frozenset({"it", "its", "they", "them", "their", "theirs", "these",
                       "those", "this", "that", "which", "who", "whom", "whose",
                       "he", "him", "his", "she", "her", "hers", "we", "us", "our"})
_CONTRASTIVE = re.compile(r"\b(other|another|remaining)\b", re.I)
_COUNTED = re.compile(r"(\d+|two|three|four|five|six|seven|eight|nine|ten|both|several|"
                      r"many|all)\b", re.I)


def _norm(text: str) -> str:
    """Lower-cased, whitespace-collapsed, one leading determiner dropped."""
    t = re.sub(r"\s+", " ", text or "").strip().lower()
    m = _DETERMINER.match(t)
    return t[m.end():] if m else t


def _align_coref(link: dict, sentence: str, here: List[dict], there: Optional[List[dict]]
                 ) -> Tuple[Optional[Tuple[dict, dict]], str]:
    """-> ((anaphor row, antecedent row), "") or (None, reason it was rejected).

    Strict on purpose, because a wrong link joins two different things.
    Measured on second_iteration (215 links): the anaphor must be an annotated
    span, either exactly or after a determiner ("the coniferous forest" over
    the span "coniferous forest"); most anaphors are pronouns, which are not.
    The antecedent is located by text, since the file gives only its sentence:
    its whole text, or its text up to the first preposition ("the western
    slope *of Mt Kenya*" -> `western slope`), must be an annotated span.
    Rejected:

      * contrastive anaphors — "other soil samples" is not "the soil samples";
      * coordinated antecedents — "shrews and rodents" is a group, not either;
      * an anaphor that is the antecedent's own modifier — "buckets" against
        "rodents in buckets" would otherwise resolve to the rodents;
      * a counted set as antecedent of something not counted the same way —
        "Community assemblage one (C1)" is a member of "two major community
        assemblages", not the pair;
      * a role mismatch — a transect (ENTITY) cannot be a gradient (PROPERTY).
        Not checked for a pronoun: it takes its antecedent's type, so there is
        nothing to compare ("They had direct effects on pollination" reaches the
        traits it refers to whatever "They" was typed).
    """
    a0, a1 = link.get("start_char"), link.get("end_char")
    anaphor = None
    for r in here:
        s0, s1 = r.get("start_char"), r.get("end_char")
        if s0 is None or s1 is None:
            continue
        if (s0, s1) == (a0, a1) or (s1 == a1 and a0 is not None and a0 < s0
                                    and _DETERMINER.fullmatch(sentence[a0:s0])):
            anaphor = r
            break
    if anaphor is None:
        return None, "anaphor_not_a_span"
    if _CONTRASTIVE.search(link.get("text") or ""):
        return None, "contrastive_anaphor"
    if there is None:
        return None, "antecedent_sentence_unknown"
    full = _norm(link.get("antecedent") or "")
    parts = _PREPOSITION.split(full, maxsplit=1)
    head = parts[0].strip()
    if _COORDINATION.search(f" {head} "):
        return None, "coordinated_antecedent"
    if len(parts) == 3 and _norm(anaphor["text"]) in parts[2]:
        return None, "anaphor_is_antecedent_modifier"
    counted = _COUNTED.match(head)
    if counted and not _norm(link.get("text") or "").startswith(counted.group(0)):
        return None, "counted_set_antecedent"
    cands = [r for r in there if r["mention_id"] != anaphor["mention_id"]
             and _norm(r["text"]) in (full, head)]
    cands.sort(key=lambda r: (_norm(r["text"]) != full, r.get("start_char") or 0))
    if not cands:
        return None, "antecedent_not_a_span"
    if (cands[0]["entity_role"] != anaphor["entity_role"]
            and _norm(anaphor["text"]) not in _PRONOUNS):
        return None, "role_mismatch"
    return (anaphor, cands[0]), ""


def load_coref(paths: List[Path], rows: List[dict], min_confidence: str, stats: Counter
               ) -> Dict[str, Tuple[dict, dict]]:
    """anaphor mention_id -> (antecedent row, the link), from the files' ``coref`` keys.

    ``antecedent_offset`` is a SENTENCE offset relative to the anaphor's
    sentence (0 = same sentence, -1 = the one before), not a character
    position; ``antecedent_sentence`` is the annotation page's own numbering.
    """
    by_sent: Dict[Tuple[str, str, int], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("part_index") is None and not r.get("split_into"):
            by_sent[(r["source_file"], r["doc_id"].rsplit("#", 1)[0], r["sent_index"])].append(r)
    floor = CONFIDENCE_RANK[min_confidence]
    out: Dict[str, Tuple[dict, dict]] = {}
    for path in paths:
        for obj in _iter_records(path):
            base = obj.get("doc_id") or path.stem
            sentences = obj.get("sentences")
            blocks = sentences if isinstance(sentences, list) else [obj]
            for si, s in enumerate(blocks):
                if not isinstance(s, dict):
                    continue
                for link in s.get("coref") or []:
                    stats["coref_links_in"] += 1
                    if CONFIDENCE_RANK.get(str(link.get("confidence")).lower(), 1) < floor:
                        stats["coref_rejected_below_min_confidence"] += 1
                        continue
                    off = link.get("antecedent_offset")
                    there = (by_sent.get((path.name, base, si + off), [])
                             if isinstance(off, int) else None)
                    pair, reason = _align_coref(link, s.get("text") or "",
                                                by_sent.get((path.name, base, si), []), there)
                    if pair is None:
                        stats[f"coref_rejected_{reason}"] += 1
                        continue
                    anaphor, antecedent = pair
                    if anaphor["mention_id"] in out:
                        stats["coref_rejected_duplicate_anaphor"] += 1
                        continue
                    out[anaphor["mention_id"]] = (antecedent, link)
                    stats["coref_accepted"] += 1
    return out


def endpoint_rows(rows: List[dict], want_type: Optional[str]) -> Tuple[str, List[dict]]:
    """-> (kind, mention rows) for one relation argument.

    ``rows`` are every resolution row sharing the argument's uid. Usually one;
    a coordination adds its parts, and the human files reuse a uid for a span
    recorded twice — once with two different types (`359f7971e0`). Among
    reused top-level rows the one whose type matches the relation's own record
    of the argument wins, else the first occurrence.
    """
    top = sorted((r for r in rows if r.get("part_index") is None),
                 key=lambda r: r["uid_occurrence"])
    if not top:
        return "missing", []
    row = next((r for r in top if want_type and r["entity_type"] == want_type), top[0])
    if row.get("split_into"):
        parts = sorted((r for r in rows if r.get("part_index") is not None
                        and r["mention_id"].startswith(row["mention_id"] + ".")),
                       key=lambda r: r["part_index"])
        return "distributed", parts
    if row["entity_id"] is None:
        return "ambiguous", [row]
    return "resolved", [row]


class Referents:
    """Which node each mention's relations attach to. See the module docstring."""

    def __init__(self, scope: str, bearers: Dict[str, Dict[str, dict]],
                 stats: Counter, coref: Optional[Dict[str, Tuple[dict, dict]]] = None,
                 values: Optional[Dict[str, dict]] = None,
                 shared_qualities: bool = False,
                 subject_mentions: Optional[set] = None,
                 ambiguous_places: bool = False):
        self.scope = scope
        self.values = values or {}      # value mention_id -> parse_value() output
        # QUALITATIVE PROPERTY mentions go to their shared concept, except those
        # that are some relation's subject (see the module docstring).
        self.shared_qualities = shared_qualities
        self.subject_mentions = subject_mentions or set()
        self.value_concepts: set = set()
        self.ambiguous_places = ambiguous_places
        self.places: Dict[str, List[dict]] = defaultdict(list)   # place id -> rows
        self.measurements: Dict[str, dict] = {}
        self.bearers = bearers          # mention_id -> {bearer mention_id: row}
        self.coref = coref or {}        # anaphor mention_id -> (antecedent row, link)
        self.stats = stats
        self.instances: Dict[str, dict] = {}
        # Referent id -> [(anaphor row, antecedent row)], for the node's sentences.
        self.via_coref: Dict[str, List[Tuple[dict, dict]]] = defaultdict(list)
        self.redirected: Dict[str, dict] = {}   # anaphor mention_id -> antecedent row
        self._memo: Dict[str, str] = {}
        self._active: set = set()

    def of(self, row: dict) -> str:
        mid = row["mention_id"]
        if mid in self._memo:
            return self._memo[mid]
        if mid in self.values:
            ref = measurement_id(mention_key(row))
            self.measurements.setdefault(ref, {"row": row, "parsed": self.values[mid]})
            self._memo[mid] = ref
            return ref
        if self.scope != "concept" and mid in self.coref:
            ante = self.coref[mid][0]
            if ante["mention_id"] in self._active:
                self.stats["coref_cycles_ignored"] += 1
            else:
                self._active.add(mid)
                ref = self.of(ante)
                self._active.discard(mid)
                self._memo[mid] = ref
                self.redirected[mid] = ante
                self.via_coref[ref].append((row, ante))
                return ref
        place = ambiguous_place_key(row) if self.ambiguous_places else None
        if place:
            ref = place_id(place)
            self.places[ref].append(row)
        elif row["entity_id"] is None or is_pronoun(row["text"]):
            # A pronoun reaching here has no resolved antecedent, so it names
            # nothing yet: kept for its relations, never a shared concept "they".
            ref = unresolved_id(mention_key(row))
        elif self.scope == "concept" or is_global(row):
            ref = row["entity_id"]
        elif (self.shared_qualities and row["entity_type"] == "QUALITATIVE PROPERTY"
              and mid not in self.subject_mentions):
            ref = row["entity_id"]
            self.value_concepts.add(ref)
            self.stats["qualities_to_shared_node"] += 1
        else:
            ref = self._instance(row)
        self._memo[mid] = ref
        return ref

    def _instance(self, row: dict) -> str:
        mid = row["mention_id"]
        bearer = None
        if self.scope == "bearer" and self.bearers.get(mid):
            self._active.add(mid)
            refs = set()
            for b_row in self.bearers[mid].values():
                if b_row["mention_id"] in self._active:
                    self.stats["bearer_cycles_ignored"] += 1
                    continue
                refs.add(self.of(b_row))
            self._active.discard(mid)
            if len(refs) == 1:
                bearer = refs.pop()
            elif len(refs) > 1:
                self.stats["mentions_with_several_bearers"] += 1
        iid = instance_id(row["doc_id"], row["entity_id"], bearer)
        inst = self.instances.setdefault(iid, {
            "id": iid, "kind": "Instance", "entity_id": row["entity_id"],
            "doc_id": row["doc_id"], "bearer_id": bearer,
            "entity_type": row["entity_type"], "role": row["entity_role"],
            "domain": row["entity_domain"], "_surface": Counter(), "_mentions": {}})
        if mid not in inst["_mentions"]:
            inst["_mentions"][mid] = row
            inst["_surface"][row["text"]] += 1
        return iid


# ------------------------------------------------------------------ #
#  Graph assembly                                                     #
# ------------------------------------------------------------------ #

def _forms(counter: Counter) -> List[str]:
    return [f"{f} ({n})" for f, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]


def _commonest(counter: Counter) -> str:
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def _edge_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def build(annotation_paths: List[Path], resolution: Dict[Tuple[str, str], List[dict]],
          scope: str = "bearer",
          bearer_relations: FrozenSet[str] = frozenset(DEFAULT_BEARER_RELATIONS),
          coref: bool = True, coref_min_confidence: str = "high",
          measurements: bool = True, shared_qualities: bool = True,
          ambiguous_places: bool = True, store: Optional[Path] = None,
          papers_dir: Optional[Path] = None
          ) -> Tuple[List[dict], List[dict], Counter]:
    """-> (nodes, edges, stats).

    Sentence text is not on the nodes and edges: it goes to ``store``
    (``sentences.sqlite``, see ``text_store``) when one is given, and is
    otherwise only read. Edges carry ``sentence_id`` in their evidence.

    ``papers_dir`` is the directory the annotated sentences were fetched into
    (``SOURCE_DIR``: ``<pmcid>.txt``, ``manifest.csv``, ``raw/``). With it,
    every statement carries its section, every edge its papers, years,
    journals and sections, and each paper is a ``Paper`` node.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, not {scope!r}")
    names = Counter(p.name for p in annotation_paths)
    clash = sorted(n for n, k in names.items() if k > 1)
    if clash:
        raise ValueError(f"annotation files share a name, so (file, uid) is ambiguous: "
                         f"{clash}. Rename them before resolving.")
    bearer_relations = frozenset(relation_type(r) for r in bearer_relations)
    stats: Counter = Counter()
    sentences = load_sentence_table(annotation_paths, stats)
    paper_info: Dict[str, dict] = {}
    sections: Dict[Tuple[str, int], Tuple[str, Optional[str]]] = {}
    if papers_dir is not None:
        paper_info = paper_meta.papers(papers_dir)
        found: dict = {}
        sections = paper_meta.sentence_sections(
            papers_dir, found,
            expected={(paper_meta.paper_of(d), i): t for d, i, t in sentences.values()})
        stats.update(found)

    all_rows: List[dict] = [r for rows in resolution.values() for r in rows]
    values = value_mentions(all_rows) if measurements else {}

    # The concept layer: every entity id in the table, connected or not. A
    # value mention is a Measurement, not a mention of a concept "2743 m".
    ent: Dict[str, dict] = {}
    terms: Dict[str, dict] = {}
    for r in all_rows:
        eid = r["entity_id"]
        if eid is None or r["mention_id"] in values or is_pronoun(r["text"]):
            continue
        e = ent.setdefault(eid, {
            "id": eid, "kind": "Entity",
            "entity_type": r["entity_type"], "role": r["entity_role"],
            "domain": r["entity_domain"],
            "grounded": bool(r["authority_uri"]), "global": is_global(r),
            "authority_uri": r["authority_uri"], "curie": curie(r["authority_uri"]),
            "authority_label": None, "source": None,
            "_surface": Counter(), "_docs": set(), "_label": None, "_rows": [],
            "_aliases": set(),
        })
        e["_aliases"].update(r.get("entity_aliases") or ())
        e["_surface"][r["text"]] += 1
        e["_rows"].append(r)
        if r.get("gmba") and "_gmba" not in e:
            e["_gmba"] = r["gmba"]
        e["_docs"].add(r["doc_id"])
        # A row that linked on its own carries the authority's naming; a
        # row that reached the entity by chaining does not.
        if r["outcome"] == "linked" and r["entity_label"] and not e["authority_label"]:
            e["authority_label"] = r["entity_label"]
            e["source"] = r["entity_source"]
        if r["outcome"] != "linked" and r["entity_label"] and not e["_label"]:
            e["_label"] = r["entity_label"]
        if r["authority_uri"]:
            t = terms.setdefault(r["authority_uri"], {
                "id": r["authority_uri"], "kind": "OntologyTerm",
                "curie": curie(r["authority_uri"]), "label": None, "source": None})
            if r["outcome"] == "linked" and r["entity_label"] and not t["label"]:
                t["label"] = r["entity_label"]
                t["source"] = r["entity_source"]

    # Pass 1 — every relation instance with its argument mentions.
    instances_in: List[dict] = []
    for path in annotation_paths:
        for rel in iter_relations(path):
            stats["relations_in"] += 1
            sides = []
            for side in ("e1", "e2"):
                want = canonicalize_entity_type(((rel.get(side) or {}).get("type")) or "")
                kind, rows = endpoint_rows(
                    resolution.get((path.name, rel.get(f"{side}_uid")), []), want or None)
                stats[f"endpoint_{kind}"] += 1
                sides.append((kind, rows))
            if any(kind == "missing" for kind, _ in sides):
                stats["relations_dropped_missing_endpoint"] += 1
                continue
            prop = canonicalize_relation_property(rel.get("property"))
            stats[f"relations_property_{prop or 'none'}"] += 1
            instances_in.append({"rel": rel, "file": path.name,
                                 "label": relation_type(rel.get("relation", "")),
                                 "property": prop, "sides": sides})

    # Pass 2 — who bears what. Mention-level, so one paper's three elevations
    # stay three when each names its own site.
    bearers: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for inst in instances_in:
        if inst["label"] not in bearer_relations:
            continue
        if inst["property"] == NEGATION:
            stats["bearer_relations_negated_ignored"] += 1
            continue
        for s_row in inst["sides"][0][1]:
            for o_row in inst["sides"][1][1]:
                if s_row["mention_id"] != o_row["mention_id"]:
                    bearers[o_row["mention_id"]][s_row["mention_id"]] = s_row
    links = (load_coref(annotation_paths, all_rows, coref_min_confidence, stats)
             if coref and scope != "concept" else {})
    subject_mentions = {row["mention_id"] for inst in instances_in
                        for row in inst["sides"][0][1]}
    refs = Referents(scope, bearers, stats, links, values,
                     shared_qualities=shared_qualities and scope != "concept",
                     subject_mentions=subject_mentions,
                     ambiguous_places=ambiguous_places)
    # Every mention gets a referent, so an instance exists for each entity in
    # each paper that mentions it, related or not.
    for r in sorted(all_rows, key=lambda r: r["mention_id"]):
        if (r["entity_id"] is not None or r["mention_id"] in values
                or (ambiguous_places and ambiguous_place_key(r))):
            refs.of(r)

    # Ambiguous place names: each candidate is an Entity, so a query reaches
    # "Kilimanjaro" -> MAY_BE -> Mount Kilimanjaro and everything said about
    # it. A candidate no mention linked to gets a node of its own.
    for pid, rows in refs.places.items():
        for c in rows[0]["candidates"]:
            eid = f"{rows[0]['entity_type']}@{c['uri']}"
            if eid not in ent:
                stats["nodes_entity_candidate_only"] += 1
                ent[eid] = {
                    "id": eid, "kind": "Entity", "entity_type": rows[0]["entity_type"],
                    "role": rows[0]["entity_role"], "domain": rows[0]["entity_domain"],
                    "grounded": True, "global": True,
                    "authority_uri": c["uri"], "curie": c.get("curie") or curie(c["uri"]),
                    "authority_label": c.get("label"), "source": c.get("source"),
                    "candidate_only": True,
                    "_surface": Counter(), "_docs": set(), "_label": c.get("label"),
                    "_rows": []}
            terms.setdefault(c["uri"], {
                "id": c["uri"], "kind": "OntologyTerm",
                "curie": c.get("curie") or curie(c["uri"]),
                "label": c.get("label"), "source": c.get("source")})

    # Pass 3 — aggregate relation instances into edges between referents.
    unresolved: Dict[str, dict] = {}
    triplets: Dict[Tuple[str, str, str, str], dict] = {}
    for inst in instances_in:
        rel, label, sides = inst["rel"], inst["label"], inst["sides"]
        prop = inst["property"]
        distributed = any(kind == "distributed" for kind, _ in sides)
        agent = rel.get("agent") or {}
        for s_row in sides[0][1]:
            for o_row in sides[1][1]:
                s_id, o_id = refs.of(s_row), refs.of(o_row)
                edge_label = label
                if (label == "HAS_PROPERTY" and o_id in refs.measurements
                        and refs.measurements[o_id]["parsed"]["kind"] != "statistic"):
                    edge_label = "HAS_VALUE"
                    stats["has_property_to_a_number_read_as_has_value"] += 1
                via = [f"{r['text']} → {refs.redirected[r['mention_id']]['text']}"
                       for r in (s_row, o_row) if r["mention_id"] in refs.redirected]
                stats["triplet_endpoints_via_coref"] += len(via)
                for nid, row in ((s_id, s_row), (o_id, o_row)):
                    if nid.startswith("unresolved:") and nid not in unresolved:
                        unresolved[nid] = {
                            "id": nid, "kind": "Unresolved",
                            "entity_type": row["entity_type"],
                            "role": row["entity_role"], "domain": row["entity_domain"],
                            "name": row["text"], "mention_id": row["mention_id"],
                            "doc_id": row["doc_id"],
                            "reason": row["reason"] or (
                                "pronoun; its antecedent is not resolved"
                                if is_pronoun(row["text"]) else None),
                            "n_candidates": row["n_candidates"], "_row": row}
                t = triplets.setdefault((s_id, edge_label, o_id, prop), {
                    "source": s_id, "target": o_id, "relation": edge_label,
                    "property": prop, "_evidence": []})
                t["_evidence"].append({
                    "relation_uid": rel.get("uid"),
                    "annotator": rel.get("annotator"),
                    "doc_id": s_row["doc_id"],
                    "sent_index": rel["_sent_index"],
                    "sentence_id": rel["_sentence_id"],
                    "e1_text": s_row["text"], "e2_text": o_row["text"],
                    "confidence": agent.get("confidence"),
                    "distributed": distributed,
                    "coref": via or None,
                    "relation_as_annotated": label if edge_label != label else None,
                    "file": inst["file"],
                })
                stats["triplet_instances"] += 1
                stats["triplet_instances_distributed"] += int(distributed)
                stats["self_loops"] += int(s_id == o_id)

    degree: Counter = Counter()
    for (s, _, o, _) in triplets:
        degree[s] += 1
        degree[o] += 1

    nodes: List[dict] = []
    n_instances: Counter = Counter(i["entity_id"] for i in refs.instances.values())
    # GMBA: a mountain's range (its own link, or a crosswalk from GeoNames) and
    # the ranges that contain it, as ontology terms joined by WITHIN.
    gmba_of: Dict[str, dict] = {}
    within: Dict[str, str] = {}
    for e in ent.values():
        g = e.pop("_gmba", None)
        if not g:
            continue
        gmba_of[e["id"]] = g
        chain = [{"uri": g["uri"], "curie": g["curie"], "label": g["label"]}] + g["path"]
        for node in chain:
            # GMBA's top levels are mountain systems: "Africa" there is the
            # mountains of Africa, not the continent GeoNames links to.
            label = (f"{node['label']} (GMBA mountain region)"
                     if node.get("feature") == "(Sub)-Continent" else node["label"])
            terms.setdefault(node["uri"], {"id": node["uri"], "kind": "OntologyTerm",
                                           "curie": node["curie"], "label": label,
                                           "source": "gmba"})
        if g.get("wikidata"):
            terms[g["uri"]]["wikidata"] = g["wikidata"]
        for child, parent in zip(chain, chain[1:]):
            within[child["uri"]] = parent["uri"]
    stats["entities_with_gmba_range"] = len(gmba_of)
    stats["entities_gmba_crosswalk"] = sum(1 for g in gmba_of.values() if g["crosswalk"])

    for e in ent.values():
        surface = e.pop("_surface")
        docs = e.pop("_docs")
        local_label = e.pop("_label")
        aliases = e.pop("_aliases", None)
        n_sentences = len({(r["doc_id"], r["sent_index"]) for r in e.pop("_rows")}
                          | {(a["doc_id"], a["sent_index"])
                             for a, _ in refs.via_coref.get(e["id"], ())})
        e.update({
            # The text's own wording reads best in a graph built from text; the
            # authority's name is kept beside it (`insects` / Insecta).
            "name": local_label or (_commonest(surface) if surface else e["id"]),
            "n_mentions": sum(surface.values()), "n_docs": len(docs),
            "docs": sorted(docs), "surface_forms": _forms(surface),
            "n_instances": n_instances[e["id"]],
            # Ids this concept had, or would have, in a corpus where it is not
            # merged with the rest of its component (see resolve_corpus).
            "aliases": sorted(aliases) if aliases else None,
            "value_concept": e["id"] in refs.value_concepts or None,
            "n_sentences": n_sentences,
            "degree": degree[e["id"]],
        })
        nodes.append(e)

    # Instance names read "elevation of Monduli forest reserve"; the bearer is
    # named without its document, which the instance states once.
    plain: Dict[str, str] = {eid: e["name"] for eid, e in ent.items()}
    plain.update({uid: u["name"] for uid, u in unresolved.items()})
    plain.update({pid: _commonest(Counter(r["text"] for r in rows))
                  for pid, rows in refs.places.items()})
    plain.update({mid: m["row"]["text"] for mid, m in refs.measurements.items()})

    def plain_name(iid: str, depth: int = 0) -> str:
        if iid in plain:
            return plain[iid]
        i = refs.instances[iid]
        base = ent[i["entity_id"]]["name"]
        if i["bearer_id"] and depth < 3:
            bearer = plain_name(i["bearer_id"], depth + 1)
            # "altitude of Udzungwa mountains", but "significant (increase)":
            # a qualitative value is not "of" its bearer in English.
            base = (f"{base} ({bearer})" if i["entity_type"] == "QUALITATIVE PROPERTY"
                    else f"{base} of {bearer}")
        plain[iid] = base
        return base

    for i in refs.instances.values():
        surface = i.pop("_surface")
        mentions = i.pop("_mentions")
        i.update({
            "base_name": ent[i["entity_id"]]["name"],
            "name": f"{plain_name(i['id'])} · {short_doc(i['doc_id'])}",
            "n_mentions": len(mentions), "surface_forms": _forms(surface),
            "n_coref_mentions": len(refs.via_coref.get(i["id"], ())),
            "docs": [i["doc_id"]], "degree": degree[i["id"]],
        })
        nodes.append(i)
    for mid, m in refs.measurements.items():
        row, parsed = m["row"], m["parsed"]
        nodes.append({
            "id": mid, "kind": "Measurement",
            "name": f"{row['text']} · {short_doc(row['doc_id'])}",
            "entity_type": row["entity_type"], "role": row["entity_role"],
            "domain": row["entity_domain"], "mention_id": row["mention_id"],
            "doc_id": row["doc_id"], "docs": [row["doc_id"]],
            "value_kind": parsed["kind"],
            **{k: v for k, v in parsed.items() if k not in ("kind", "raw")},
            "raw": parsed["raw"],
            "degree": degree[mid],
        })
    for pid, rows in refs.places.items():
        docs = sorted({r["doc_id"] for r in rows})
        nodes.append({
            "id": pid, "kind": "AmbiguousPlace", "global": True,
            "name": plain[pid], "entity_type": rows[0]["entity_type"],
            "role": rows[0]["entity_role"], "domain": rows[0]["entity_domain"],
            "n_mentions": len(rows), "n_docs": len(docs), "docs": docs,
            "surface_forms": _forms(Counter(r["text"] for r in rows)),
            "candidates": [{"curie": c.get("curie") or curie(c["uri"]),
                            "label": c.get("label"), "feature": c.get("rank")}
                           for c in rows[0]["candidates"]],
            "reason": rows[0]["reason"],
            "degree": degree[pid],
        })
    for u in unresolved.values():
        u["degree"] = degree[u["id"]]
        u.pop("_row")
        nodes.append(u)
    grounded_to: Counter = Counter(e["authority_uri"] for e in ent.values()
                                   if e["authority_uri"])
    grounded_to.update(g["uri"] for eid, g in gmba_of.items()
                       if g["uri"] != ent[eid]["authority_uri"])
    for t in terms.values():
        t["name"] = t["label"] or t["curie"]
        t["n_entities"] = grounded_to[t["id"]]
        nodes.append(t)
    used = {paper_meta.paper_of(d) for _, (d, _, _) in sentences.items()}
    for pmcid in sorted(used & set(paper_info)):
        rec = paper_info[pmcid]
        nodes.append({"id": f"paper:{pmcid}", "kind": "Paper",
                      "name": rec["citation"] or pmcid, **rec})
    nodes.sort(key=lambda n: (n["kind"], n["id"]))

    edges: List[dict] = []
    for t in triplets.values():
        ev = sorted(t.pop("_evidence"), key=lambda x: (x["file"], x["doc_id"],
                                                      x["sent_index"], x["relation_uid"] or ""))
        for x in ev:
            x["section"], x["subsection"] = sections.get(
                (paper_meta.paper_of(x["doc_id"]), x["sent_index"]), (None, None))
        pubs = [paper_info.get(paper_meta.paper_of(x["doc_id"])) for x in ev]
        t.update({
            # The property joins the id only when there is one, so an edge
            # without one keeps the id it had before properties existed.
            "id": _edge_id(t["source"], t["relation"], t["target"],
                           *([t["property"]] if t["property"] else [])),
            "graph_type": graph_relation_type(t["relation"], t["property"]),
            "n_evidence": len(ev),
            "docs": sorted({x["doc_id"] for x in ev}),
            "annotators": sorted({x["annotator"] for x in ev if x["annotator"]}),
            "confidence_mean": _mean(x["confidence"] for x in ev
                                     if isinstance(x["confidence"], (int, float))),
            "distributed": all(x["distributed"] for x in ev),
            "via_coref": any(x["coref"] for x in ev),
            # where the statements come from, for filtering without the store
            "papers": sorted({paper_meta.paper_of(x["doc_id"]) for x in ev}),
            "years": sorted({p["year"] for p in pubs if p and p["year"]}) or None,
            "journals": sorted({p["journal"] for p in pubs if p and p["journal"]}) or None,
            "sections": sorted({x["section"] for x in ev if x["section"]}) or None,
            "evidence": ev,
        })
        edges.append(t)
    for i in refs.instances.values():
        edges.append({"id": _edge_id(i["id"], INSTANCE_OF), "source": i["id"],
                      "target": i["entity_id"], "relation": INSTANCE_OF,
                      "n_evidence": i["n_mentions"]})
    for pid, rows in refs.places.items():
        docs = {r["doc_id"] for r in rows}
        for c in rows[0]["candidates"]:
            eid = f"{rows[0]['entity_type']}@{c['uri']}"
            edges.append({
                "id": _edge_id(pid, MAY_BE, eid), "source": pid, "target": eid,
                "relation": MAY_BE, "n_evidence": len(rows),
                "feature": c.get("rank"),
                # Papers that say the bare name AND name this candidate
                # outright: the first place to look when disambiguating.
                "docs_naming_candidate": sorted(docs & set(ent[eid]["docs"])) or None,
            })
    for e in ent.values():
        if e["authority_uri"]:
            edges.append({"id": _edge_id(e["id"], EXACT_MATCH), "source": e["id"],
                          "target": e["authority_uri"], "relation": EXACT_MATCH,
                          "n_evidence": e["n_mentions"]})
        g = gmba_of.get(e["id"])
        if g and g["uri"] != e["authority_uri"]:
            edges.append({"id": _edge_id(e["id"], EXACT_MATCH, g["uri"]), "source": e["id"],
                          "target": g["uri"], "relation": EXACT_MATCH,
                          "n_evidence": e["n_mentions"], "crosswalk": "gmba"})
    for child, parent in within.items():
        edges.append({"id": _edge_id(child, WITHIN, parent), "source": child,
                      "target": parent, "relation": WITHIN, "n_evidence": 1})
    edges.sort(key=lambda x: (x["relation"] in STRUCTURAL, x["relation"] == EXACT_MATCH,
                              x["source"], x["relation"], x["target"],
                              x.get("property") or ""))

    stats["nodes_entity"] = len(ent)
    stats["nodes_entity_global"] = sum(1 for e in ent.values() if e["global"])
    stats["nodes_entity_global_connected"] = sum(
        1 for e in ent.values() if e["global"] and degree[e["id"]])
    stats["nodes_instance"] = len(refs.instances)
    stats["nodes_instance_connected"] = sum(1 for i in refs.instances if degree[i])
    stats["nodes_instance_bearer_scoped"] = sum(
        1 for i in refs.instances.values() if i["bearer_id"])
    stats["nodes_unresolved"] = len(unresolved)
    stats["nodes_ambiguous_place"] = len(refs.places)
    stats["ambiguous_place_mentions"] = sum(len(r) for r in refs.places.values())
    stats["nodes_measurement"] = len(refs.measurements)
    stats["nodes_value_concept"] = len(refs.value_concepts)
    for m in refs.measurements.values():
        stats[f"measurements_{m['parsed']['kind']}"] += 1
    stats["nodes_ontology_term"] = len(terms)
    stats["edges_semantic"] = len(triplets)
    for t in triplets.values():
        stats[f"edges_property_{t['property'] or 'none'}"] += 1
    stats["edges_instance_of"] = len(refs.instances)
    stats["edges_exact_match"] = sum(1 for e in ent.values() if e["authority_uri"])
    stats["nodes_paper"] = sum(1 for n in nodes if n["kind"] == "Paper")
    sem = [x for t in triplets.values() for x in t["evidence"]]
    stats["evidence_with_paper"] = sum(1 for x in sem
                                       if paper_meta.paper_of(x["doc_id"]) in paper_info)
    stats["evidence_with_section"] = sum(1 for x in sem if x["section"])
    stats["evidence_with_subsection"] = sum(1 for x in sem if x["subsection"])

    if store is not None:
        # One row per mention, saying which node draws it and which concept it
        # names; two runs over one paper give one row (same key, same node).
        mentions: Dict[str, dict] = {}
        for r in sorted(all_rows, key=lambda r: r["mention_id"]):
            mid, eid = r["mention_id"], r["entity_id"]
            concept = (eid if eid in ent and mid not in values and not is_pronoun(r["text"])
                       else None)
            rec = _mention_record(r, refs._memo.get(mid), concept, refs.redirected.get(mid))
            if rec["key"] in mentions:
                stats["mentions_shared_across_files"] += 1
                continue
            mentions[rec["key"]] = rec
        evidence = [{"edge_id": e["id"], **{k: x.get(k) for k in (
                        "sentence_id", "doc_id", "relation_uid", "annotator",
                        "confidence", "file")},
                     "distributed": int(bool(x.get("distributed"))),
                     "coref": "; ".join(x["coref"]) if x.get("coref") else None}
                    for e in edges for x in e.get("evidence") or ()]
        write_store(store, sentences, mentions.values(), evidence,
                    sections={sid: sections.get((paper_meta.paper_of(d), i), (None, None))
                              for sid, (d, i, _) in sentences.items()},
                    papers=[paper_info[p] for p in sorted(used & set(paper_info))],
                    meta={"generated_at": datetime.now(timezone.utc)
                          .isoformat(timespec="seconds"),
                          "annotations": " ".join(str(p) for p in annotation_paths)})
        stats["store_sentences"] = len(sentences)
        stats["store_mentions"] = len(mentions)
        stats["store_evidence"] = len(evidence)
    return nodes, edges, stats


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_triplets_tsv(path: Path, nodes: List[dict], edges: List[dict]) -> None:
    """One readable row per semantic edge — what a domain expert skims.

    The curie columns are the subject's and object's CONCEPT link, so an
    instance shows the ontology term of the entity it is an instance of.
    """
    by_id = {n["id"]: n for n in nodes}

    def concept(n):
        return by_id.get(n.get("entity_id"), n) if n["kind"] == "Instance" else n

    with path.open("w", encoding="utf-8") as fh:
        fh.write("n_evidence\tdocs\tsubject\tsubject_type\tsubject_curie\trelation"
                 "\tproperty\tobject\tobject_type\tobject_curie\tconfidence_mean"
                 "\tdistributed\tsubject_id\tobject_id\n")
        for e in sorted((e for e in edges if e["relation"] not in STRUCTURAL),
                        key=lambda e: (-e["n_evidence"], e["source"], e["relation"],
                                       e["target"], e["property"])):
            s, o = by_id[e["source"]], by_id[e["target"]]
            fh.write("\t".join(str(v) if v is not None else "" for v in (
                e["n_evidence"], ",".join(short_doc(d) for d in e["docs"]),
                s["name"], s.get("entity_type"), concept(s).get("curie"), e["relation"],
                e["property"],
                o["name"], o.get("entity_type"), concept(o).get("curie"),
                e["confidence_mean"], e["distributed"], e["source"], e["target"])) + "\n")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, nargs="+", required=True,
                    help="the annotation file(s) given to resolve_corpus --input")
    ap.add_argument("--resolution", type=Path, required=True,
                    help="resolve_corpus output (resolution.jsonl)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scope", choices=SCOPES, default="bearer",
                    help="what a relation attaches to: per-document instances scoped "
                         "by their bearer (default), per-document instances, or the "
                         "shared concept")
    ap.add_argument("--bearer-relations", nargs="+",
                    default=list(DEFAULT_BEARER_RELATIONS), metavar="REL",
                    help="relations whose subject bears their object "
                         "(default: %(default)s)")
    ap.add_argument("--no-shared-qualities", action="store_true",
                    help="give QUALITATIVE PROPERTY mentions per-paper nodes instead "
                         "of pointing relations at their shared concept")
    ap.add_argument("--no-ambiguous-places", action="store_true",
                    help="keep ambiguous place names as per-mention Unresolved nodes "
                         "instead of one AmbiguousPlace node linked to its candidates")
    ap.add_argument("--papers", type=Path, default=None, metavar="SOURCE_DIR",
                    help="directory the sentences were fetched into (<pmcid>.txt, "
                         "manifest.csv, raw/): adds Paper nodes and each "
                         "statement's paper, journal, year and section")
    ap.add_argument("--no-measurements", action="store_true",
                    help="keep values as ordinary entity nodes instead of parsed "
                         "Measurement nodes")
    ap.add_argument("--no-coref", action="store_true",
                    help="ignore the annotation files' coreference links")
    ap.add_argument("--coref-min-confidence", choices=list(CONFIDENCE_RANK),
                    default="high", help="lowest coreference confidence used "
                                         "(default: %(default)s)")
    args = ap.parse_args(argv)

    for p in [*args.annotations, args.resolution]:
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 1
    resolution = load_resolution(args.resolution)
    files_in_table = {k[0] for k in resolution}
    absent = sorted(p.name for p in args.annotations if p.name not in files_in_table)
    if absent:
        print(f"not in the resolution table (resolve them first): {absent}",
              file=sys.stderr)
        return 1
    try:
        nodes, edges, stats = build(args.annotations, resolution, scope=args.scope,
                                    bearer_relations=frozenset(args.bearer_relations),
                                    coref=not args.no_coref,
                                    coref_min_confidence=args.coref_min_confidence,
                                    measurements=not args.no_measurements,
                                    shared_qualities=not args.no_shared_qualities,
                                    ambiguous_places=not args.no_ambiguous_places,
                                    store=args.out_dir / "sentences.sqlite",
                                    papers_dir=args.papers)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "nodes.jsonl", nodes)
    _write_jsonl(args.out_dir / "edges.jsonl", edges)
    write_triplets_tsv(args.out_dir / "triplets.tsv", nodes, edges)
    res_meta = args.resolution.with_suffix(".meta.json")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "annotations": [str(p) for p in args.annotations],
        "resolution": str(args.resolution),
        "resolution_meta": (json.loads(res_meta.read_text(encoding="utf-8"))
                            if res_meta.exists() else None),
        "scope": args.scope,
        "shared_qualities": not args.no_shared_qualities,
        "ambiguous_places": not args.no_ambiguous_places,
        "bearer_relations": sorted(relation_type(r) for r in args.bearer_relations),
        "global_sources": sorted(GLOBAL_SOURCES),
        "coref": None if args.no_coref else {"min_confidence": args.coref_min_confidence},
        "papers": str(args.papers) if args.papers else None,
        "counts": dict(sorted(stats.items())),
        "relation_types": dict(Counter(e["relation"] for e in edges
                                       if e["relation"] not in STRUCTURAL).most_common()),
        "relation_properties": dict(Counter(e["property"] for e in edges
                                            if e["relation"] not in STRUCTURAL
                                            and e["property"]).most_common()),
    }
    (args.out_dir / "graph.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    n_ep = sum(v for k, v in stats.items() if k.startswith("endpoint_"))
    print(f"wrote {args.out_dir}/{{nodes,edges}}.jsonl, sentences.sqlite, triplets.tsv, graph.meta.json "
          f"(scope: {args.scope})")
    print(f"  relations in         {stats['relations_in']:,}  "
          f"({stats['relations_dropped_missing_endpoint']} dropped: endpoint not in table)")
    print(f"  endpoints            {n_ep:,}: {stats['endpoint_resolved']:,} resolved, "
          f"{stats['endpoint_ambiguous']:,} ambiguous, "
          f"{stats['endpoint_distributed']:,} coordinations")
    print(f"  concepts (Entity)    {stats['nodes_entity']:,}, of which "
          f"{stats['nodes_entity_global']:,} global (GeoNames/GBIF/WDPA/GMBA; "
          f"{stats['nodes_entity_global_connected']:,} in a triplet)")
    print(f"  instances            {stats['nodes_instance']:,} "
          f"({stats['nodes_instance_connected']:,} in a triplet; "
          f"{stats['nodes_instance_bearer_scoped']:,} scoped by a bearer; "
          f"{stats['mentions_with_several_bearers']} mention(s) with several bearers "
          f"kept at document scope)")
    if stats["coref_links_in"]:
        rejected = {k.removeprefix("coref_rejected_"): v for k, v in stats.items()
                    if k.startswith("coref_rejected_")}
        print(f"  coreference          {stats['coref_accepted']} of "
              f"{stats['coref_links_in']} links used; "
              f"{stats['triplet_endpoints_via_coref']} relation ends redirected")
        print("                       rejected: " + ", ".join(
            f"{k} {v}" for k, v in sorted(rejected.items(), key=lambda kv: -kv[1])))
    if stats["nodes_measurement"]:
        kinds = {k.removeprefix("measurements_"): v for k, v in stats.items()
                 if k.startswith("measurements_")}
        print(f"  measurements         {stats['nodes_measurement']:,} ("
              + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
              + ")")
    if stats["nodes_value_concept"]:
        print(f"  shared qualities     {stats['qualities_to_shared_node']:,} mentions point "
              f"straight at {stats['nodes_value_concept']:,} shared quality nodes")
    print(f"  unresolved nodes     {stats['nodes_unresolved']:,}")
    print(f"  ambiguous places     {stats['nodes_ambiguous_place']:,} "
          f"({stats['ambiguous_place_mentions']:,} mentions)")
    print(f"  ontology terms       {stats['nodes_ontology_term']:,}")
    props = {k.removeprefix("edges_property_"): v for k, v in stats.items()
             if k.startswith("edges_property_") and k != "edges_property_none"}
    if props:
        print("  relation properties  " + ", ".join(
            f"{k} {v}" for k, v in sorted(props.items(), key=lambda kv: -kv[1]))
            + f" edges ({stats['bearer_relations_negated_ignored']} negated bearer "
              f"relation(s) not used for scoping)")
    print(f"  triplet edges        {stats['edges_semantic']:,} from "
          f"{stats['triplet_instances']:,} instances "
          f"({stats['self_loops']} self-loop instances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
