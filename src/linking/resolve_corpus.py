"""
Produce the resolution table: one row per mention, saying which entity it is.

This is the deliverable. Everything else in `src/linking` either builds the
vocabulary it consults or measures how well it does; this is the file that other
work joins against.

The contract
------------
One JSONL row per mention. **Two mentions denote the same entity if and only if
they carry the same ``entity_id``.**

Rows are keyed by ``mention_id``, not by ``uid``. The annotation files carry a
``uid`` per span, but it is not unique — not across annotators, and not even
within one file. In Mark's file alone 15 uids are reused: mostly a span recorded
twice identically, and once (`359f7971e0`, "species" at sentence 159) the same
span annotated BIOTIC ENTITY and TEMPORAL PROPERTY, an unresolved type conflict.

Keying on uid would silently drop rows or merge two entities. So ``mention_id``
is ``<file>:<uid>#<occurrence>``, and ``uid`` is kept on every row because that
is what a relation file joins against inside its own annotator's data. The reuse
count is reported in the sidecar rather than repaired here: these are the
annotators' files, and quietly collapsing their duplicates would change span
counts that other analyses depend on. That is the whole de-duplication story, and it
is a plain equality test rather than a similarity score anyone has to threshold.

An ``entity_id`` is either

  * a type-scoped authority identity — ``SPATIAL ENTITY@https://sws.geonames.org/1562822/``
    — when the mention grounded, or
  * a project-local id — ``mobiko:e<hash>`` — when nothing in any authority
    matched, or
  * ``null``, when the matcher found candidates and deliberately declined to
    choose between them.

Nothing is ever deleted or rewritten. Every mention keeps its own row with its
offsets and provenance, and a merge is expressed by two rows sharing an
``entity_id``. A bad merge is therefore reversible, which is the property that
made modifier-preserving granularity the safe default in the first place.

Why the entity id carries the full label
----------------------------------------
The label schema has two layers: an ontological **role** (entity, process,
property, concept) and a **domain** (abiotic, biotic, spatial, …). Identity is
the **pair**, so `habitat` typed ABIOTIC / BIOTIC / SPATIAL ENTITY is three
nodes — and all three carry the same ``authority_uri``, ENVO:01000739. The
ontological entity they share is stated explicitly rather than by merging them.

    entity_id      ABIOTIC ENTITY@…/ENVO_01000739
                   BIOTIC ENTITY@…/ENVO_01000739
                   SPATIAL ENTITY@…/ENVO_01000739
    authority_uri  http://purl.obolibrary.org/obo/ENVO_01000739   (all three)

In RDF that is three nodes with `skos:exactMatch` to one ENVO term. Both
directions stay recoverable: a consumer wanting the labels apart has them, and
one wanting them together groups on `authority_uri`.

``surface_key`` is the same handle for entities no authority covers — the words
with the label stripped — so `species` under three labels can still be gathered
without being merged.

Why relations are untouched
---------------------------
Relations stay exactly where they are, in the input files. When their mapping is
built later it is a join on ``uid`` against this table — no rework on either
side. That is the reason ``uid`` is the key and not, say, a normalised string.

Coordinated spans
-----------------
A span covering a coordination denotes more than one thing, so it is emitted as
a container row with no ``entity_id`` plus one row per part. "eastern and
western slopes" is not an entity; "eastern slopes" and "western slopes" are.
Parts carry ``split_from`` and a ``mention_id`` suffixed ``.1``, ``.2``, so the
original span is always recoverable and a run with splitting disabled produces a
strict subset of the rows rather than different ones.

Why ambiguous mentions get no id
--------------------------------
Because the matcher declined on purpose. Minting a project-local entity for the
eight uses of "the lake" would assert they are one lake — the exact guess that
was refused. They are carried in the table, with their candidates, awaiting a
tier that has sentence context or resolved coreference.

Usage
-----
    python -m src.linking.resolve_corpus --out output/resolution.jsonl
    python -m src.linking.resolve_corpus --input run.jsonl --out out.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from src.linking.authorities.registry import ARTIFACT_DIR
from src.linking.cluster import (build_clusters, content_key, entity_id,
                                 link_signatures, rank_by_document_frequency,
                                 singular_form, split_entity_type, surface_key)
from src.linking.dictionary import Dictionary
from src.linking.split_spans import SpanSplitter
from src.linking.match import Outcome, resolve
from src.linking import wordnet_guard
from src.linking.mention_norm import NORMALIZER_VERSION, normalize_key

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multi_agent_annotation"))
from label_norm import canonicalize_entity_type  # noqa: E402

logger = logging.getLogger(__name__)

# A new source abstract starts here. The annotation files concatenate 15
# abstracts into one pseudo-document with doc_id "abstracts", so taking that
# field at face value makes document frequency identically 1 and destroys the
# only signal that separates domain vocabulary from one author's habit. The
# same boundary, and the same regex, are used by
# scripts/add_coref_annotations.py, which notes that "[results]"/"[methods]"
# are sections WITHIN an abstract and must not split.
import re as _re
DOC_BOUNDARY_RE = _re.compile(r"^\s*\[abstract\]", _re.I)

DEFAULT_INPUTS = [
    Path("data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"),
    Path("data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl"),
]

# Outcomes that get a project-local entity. AMBIGUOUS is excluded on purpose —
# see the module docstring.
CLUSTERED_OUTCOMES = (Outcome.NIL, Outcome.UNROUTED)


def _synthetic_uid(doc_id: str, sent_i: int, start, end, text: str) -> str:
    """A stable uid for input that carries none.

    The human files stamp ``uid`` on every span, but agent output does not — its
    entities have only text, type, confidence and reasoning, with no offsets
    either. Deriving one from the position and the text keeps the table keyed
    uniformly, and keeps it reproducible: the same input yields the same uid.
    """
    payload = f"{doc_id}|{sent_i}|{start}|{end}|{text}".encode("utf-8")
    return "syn" + hashlib.sha256(payload).hexdigest()[:10]


def _iter_records(path: Path) -> Iterator[dict]:
    """Records from a file that is either JSONL or one whole JSON document.

    Both shapes are in use and they are not distinguishable by extension: the
    postprocessed gold is JSONL, while the relation-annotation inputs are a
    single pretty-printed object — and those are the only files that carry
    ``concept_text``, so a JSONL-only reader silently cannot see it.
    """
    text = path.read_text(encoding="utf-8")
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        yield from (whole if isinstance(whole, list) else [whole])
        return
    for line_i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s: unparseable line %d, skipped", path, line_i)


_TOKEN_TRIM = " \t\n.,;:!?()[]{}\"'`"


def _tokens(text: str) -> List[str]:
    """Words, with punctuation trimmed from each token's edges.

    Trimming per token matters: `normalize_key` only strips punctuation from the
    ends of the whole string, so a sentence ending "...topographic relief."
    yields the token `relief.`, and a naive containment test then reports
    `relief` as missing from a sentence that plainly contains it. That bug
    rejected most of the valid concept_texts on the first attempt.
    """
    return [t for t in (tok.strip(_TOKEN_TRIM)
                        for tok in normalize_key(text).replace("-", " ").split())
            if t]


def validate_concept_text(concept: str, sentence: str,
                          document: Optional[str] = None) -> Optional[str]:
    """Which scope supports this ``concept_text``: "sentence", "document", or None.

    Token containment, not substring. A substring test would reject the valid
    cases: "annual rainfall" carries concept_text "high annual rainfall" because
    `high` distributes across a conjunction, and that phrase is not contiguous
    in "areas of high temperatures, annual rainfall and topographic relief".
    Every token IS present, so containment accepts it.

    The document is checked when the sentence does not support the value,
    because the most valuable corrections are exactly the ones that reach beyond
    it: the annotators expand abbreviated binomials, turning the span `frater`
    into `Myotis frater` and `javanicus` into `Pipistrellus javanicus` using a
    genus named earlier in the abstract. Those are worth honouring — a full
    binomial is what GBIF can match, and a bare epithet is not.

    Both scopes still reject the corrupt values, and on this data almost every
    rejection is one artefact rather than a judgement call: 190 of the 195
    rejected values are the annotation tool's placeholder, the literal token
    "x" prefixed to the span text ("x three", "x phylogenetic", "x Bacillota").
    The remaining five are genuine corruptions — "Helarctos malayanus" recorded
    as "Helarctos Malayans", "predator presence" as "predator changed".
    """
    if not concept:
        return None
    want = _tokens(concept)
    if not want:
        return None
    if sentence and all(t in set(_tokens(sentence)) for t in want):
        return "sentence"
    if document and all(t in set(_tokens(document)) for t in want):
        return "document"
    return None


def iter_spans(path: Path) -> Iterator[dict]:
    """Yield one dict per span, from either file shape in use."""
    seen_uids: Counter = Counter()
    for obj in _iter_records(path):
        base_doc = obj.get("doc_id") or path.stem
        sentences = obj.get("sentences")
        blocks = sentences if isinstance(sentences, list) else [obj]
        document = " ".join((b.get("text") or b.get("sentence") or "")
                            for b in blocks if isinstance(b, dict))
        sub_doc = 0
        for sent_i, s in enumerate(blocks):
            if not isinstance(s, dict):
                continue
            sentence = s.get("text") or s.get("sentence") or ""
            if DOC_BOUNDARY_RE.match(sentence):
                sub_doc += 1
            doc_id = f"{base_doc}#{sub_doc:02d}"
            for sp in (s.get("spans") or s.get("entities") or []):
                if not isinstance(sp, dict):
                    continue
                text = (sp.get("text") or "").strip()
                if not text:
                    continue
                start, end = sp.get("start_char"), sp.get("end_char")
                uid = sp.get("uid") or _synthetic_uid(doc_id, sent_i, start, end, text)
                seen_uids[uid] += 1
                occurrence = seen_uids[uid]
                concept = sp.get("concept_text")
                scope = validate_concept_text(concept, sentence, document) if concept else None
                concept_ok = scope is not None
                yield {
                    "mention_id": f"{path.stem}:{uid}#{occurrence}",
                    "uid_occurrence": occurrence,
                    "source_file": path.name,
                    "uid": uid,
                    "doc_id": doc_id,
                    "sent_index": sent_i,
                    "sentence": sentence,
                    "start_char": start,
                    "end_char": end,
                    "text": text,
                    "entity_type": canonicalize_entity_type(
                        sp.get("type") or sp.get("entity_type") or ""),
                    # Optional, honoured when present AND supported by its
                    # sentence. The canonical form is what identity and
                    # matching both use; concept_source records which it was,
                    # so a run can be read without guessing.
                    "concept_text": concept,
                    "concept_text_scope": scope,
                    "canonical_text": concept if concept_ok else text,
                    "concept_source": (f"concept_text:{scope}" if concept_ok
                                       else ("concept_text_rejected" if concept
                                             else "surface")),
                }


def _try_singular(dictionary, sp: dict, nil_result):
    """One more attempt at a mention nothing matched, using its singular.

    Third in the canonical-form precedence, after `concept_text` and the
    surface form, and reached only when the surface form matched nothing — so
    it can add grounding but never take an answer away.

    **Only a LINK is accepted.** The singular is a hypothesis about what the
    plural denotes; a hypothesis that grounds cleanly is evidence, while one
    that comes back ambiguous says only that the singular is ambiguous, which
    is a fact about a form the annotator did not write. Adopting that would
    strip 15 mentions of an entity_id they currently have — `landscapes` would
    inherit `landscape`'s refusal — and buy nothing. They stay NIL and keep
    their project-local identity.

    Identity does not move either way: `content_key` lemmatises every token, so
    the plural and the singular already share a key.
    """
    alt_text = singular_form(sp["canonical_text"])
    if not alt_text:
        return nil_result, sp
    alt = resolve(dictionary, alt_text, sp["entity_type"])
    if alt.outcome is not Outcome.LINKED:
        return nil_result, sp
    return alt, {**sp, "canonical_text": alt_text, "concept_source": "singular"}


def _code_version() -> Optional[str]:
    """Git sha of the tree that produced this table, or None, visibly."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=Path(__file__).resolve().parent,
                           capture_output=True, text=True, timeout=5)
        sha = r.stdout.strip()
        if not sha:
            return None
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, timeout=10)
        return f"{sha}-dirty" if st.stdout.strip() else sha
    except Exception:
        return None


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ARTIFACT_DIR)
    ap.add_argument("--input", type=Path, action="append", dest="inputs")
    ap.add_argument("--out", type=Path, default=Path("output/resolution.jsonl"))
    ap.add_argument("--no-split", action="store_true",
                    help="skip coordination splitting even when the parser is present")
    ap.add_argument("--no-singular", action="store_true",
                    help="do not retry an unmatched mention with its head noun "
                         "singularised (`glaciations` -> `glaciation`)")
    ap.add_argument("--gap-report", type=Path, default=None,
                    help="also write the unresolved clusters ranked by document "
                         "frequency: the measured ontology coverage gap")
    ap.add_argument("--linked-report", type=Path, default=None,
                    help="also write one row per GROUNDED entity — the mirror of "
                         "--gap-report, and what a precision check reads")
    ap.add_argument("--entity-report", type=Path, default=None,
                    help="also write one row per entity, grounded and project-local "
                         "together, with a status column")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        d = Dictionary.open(args.dir)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    inputs = args.inputs or DEFAULT_INPUTS
    for p in inputs:
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 1

    splitter = None if args.no_split else SpanSplitter()
    if splitter and not splitter.available:
        print(f"  note: coordination splitting is OFF — {splitter.error}")

    # Pass 0 — collect, then split coordinated spans into their parts.
    raw: List[dict] = []
    for path in inputs:
        raw.extend(iter_spans(path))

    n_split = 0
    if splitter and splitter.available:
        by_sentence: Dict[str, list] = {}
        for i, sp in enumerate(raw):
            by_sentence.setdefault(sp["sentence"], []).append((i, sp["start_char"],
                                                              sp["end_char"]))
        expanded: List[dict] = []
        for sp in raw:
            sp["split_from"] = None
            sp["part_index"] = None
        for sentence, group in by_sentence.items():
            for i, sres in splitter.split_all(sentence, group).items():
                raw[i]["_split"] = sres
        for sp in raw:
            sres = sp.pop("_split", None)
            expanded.append(sp)
            if not sres:
                continue
            n_split += 1
            sp["split_into"] = len(sres.parts)
            for k, part in enumerate(sres.parts, 1):
                child = dict(sp)
                child.pop("split_into", None)
                child.update({
                    "mention_id": f"{sp['mention_id']}.{k}",
                    "text": part,
                    "canonical_text": part,
                    "concept_source": f"split:{sres.pattern}",
                    "split_from": sp["text"],
                    "part_index": k,
                })
                expanded.append(child)
        raw = expanded

    # Pass 1 — resolve every mention against the authorities.
    rows: List[dict] = []
    stats: Counter = Counter()
    reused_uids = 0
    n_singular = 0
    concept_stats: Counter = Counter()
    if True:
        for sp in raw:
            if sp.get("part_index") is None and sp["uid_occurrence"] > 1:
                reused_uids += 1
            # canonical_text, not the raw span: where an annotator recorded a
            # concept_text and the sentence supports it, that is what the
            # mention denotes. The surface form is kept on the row regardless.
            r = resolve(d, sp["canonical_text"], sp["entity_type"])
            if r.outcome is Outcome.NIL and not args.no_singular:
                r, sp = _try_singular(d, sp, r)
                n_singular += int(sp["concept_source"] == "singular")
            concept_stats[sp["concept_source"]] += 1
            stats[r.outcome.value] += 1
            rows.append({**sp, "_res": r})

    # Pass 2 — decide identity. A mention carries up to two signatures, its
    # content key and the concept it grounded to, and both are evidence of the
    # same identity; connected components over them use each where it is strong.
    sigs = []
    for r in rows:
        res = r["_res"]
        if res.outcome is Outcome.AMBIGUOUS or r.get("split_into"):
            # A container row for a coordination denotes no single entity; its
            # parts do.
            r["_sig"] = None
            continue
        # Both signatures carry the FULL label: joining on the role alone would
        # merge the three habitat nodes back together through their shared URI.
        key_sig = ("key", content_key(r["canonical_text"], r["entity_type"]))
        uri_sig = (("uri", r["entity_type"], res.match.uri)
                   if res.outcome is Outcome.LINKED else None)
        r["_sig"] = key_sig
        sigs.append((key_sig, uri_sig))
    comp = link_signatures(sigs)

    # Name each component: its authority URI where it has one, else its key.
    comp_uri: Dict[object, str] = {}
    comp_key: Dict[object, tuple] = {}
    for r in rows:
        if r["_sig"] is None:
            continue
        c = comp[r["_sig"]]
        comp_key.setdefault(c, r["_sig"][1])
        if r["_res"].outcome is Outcome.LINKED:
            uri = r["_res"].match.uri
            # Deterministic when a component somehow spans two authorities.
            if c not in comp_uri or uri < comp_uri[c]:
                comp_uri[c] = uri

    def _entity_of(row) -> Optional[str]:
        if row["_sig"] is None:
            return None
        c = comp[row["_sig"]]
        if c in comp_uri:
            return f"{row['entity_type']}@{comp_uri[c]}"
        return entity_id(comp_key[c])

    to_cluster = [(r["mention_id"], r["canonical_text"], r["entity_type"], r["doc_id"])
                  for r in rows if r["_res"].outcome in CLUSTERED_OUTCOMES]
    clusters = build_clusters(to_cluster)

    # Every label any annotator gave this entity, so the merge discards nothing.
    entity_labels: Dict[str, Counter] = {}
    for r in rows:
        eid = _entity_of(r)
        if eid:
            entity_labels.setdefault(eid, Counter())[r["entity_type"]] += 1

    # Counted directly rather than derived from len(clusters). A content key
    # that chained into a grounded component is named by its authority URI, so
    # it is no longer a project-local entity — subtracting the cluster count
    # moved 28 entities to the wrong side of the split.
    grounded_entities: set = set()
    local_entities: set = set()
    # One record per entity, grounded or not. The three reports are views of
    # this: --linked-report is what the authorities caught (a precision check),
    # --gap-report what they missed (the coverage gap), --entity-report both.
    entities: Dict[str, dict] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            res = r["_res"]
            role, domain = split_entity_type(r["entity_type"])
            eid = _entity_of({**r, "_res": res, "_sig": r.get("_sig")})
            authority_uri = (comp_uri.get(comp[r["_sig"]])
                             if r.get("_sig") is not None else None)
            source = (res.match.source if res.outcome is Outcome.LINKED
                      else ("authority" if authority_uri else
                            ("cluster" if eid else None)))
            if eid:
                (grounded_entities if authority_uri else local_entities).add(eid)
            if eid:
                e = entities.setdefault(eid, {
                    "entity_type": r["entity_type"], "role": role, "domain": domain,
                    "uri": authority_uri, "grounded": bool(authority_uri),
                    "curie": None, "label": None, "source": None,
                    "scope": None, "surface": Counter(), "docs": set(),
                    "via_singular": 0,
                })
                e["surface"][r["text"]] += 1
                e["docs"].add(r["doc_id"])
                e["via_singular"] += int(r["concept_source"] == "singular")
                # A mention that reached this entity by chaining carries no
                # Match of its own; take the naming details from one that did.
                if res.outcome is Outcome.LINKED and res.match and not e["curie"]:
                    e["curie"] = res.match.curie or res.match.uri
                    e["label"] = res.match.label
                    e["source"] = res.match.source
                    e["scope"] = res.match.scope
            fh.write(json.dumps({
                "mention_id": r["mention_id"],
                "source_file": r["source_file"],
                "uid": r["uid"],
                "uid_occurrence": r["uid_occurrence"],
                "split_from": r.get("split_from"),
                "part_index": r.get("part_index"),
                "split_into": r.get("split_into"),
                "doc_id": r["doc_id"],
                "sent_index": r["sent_index"],
                "start_char": r["start_char"],
                "end_char": r["end_char"],
                "text": r["text"],
                "concept_text": r["concept_text"],
                "canonical_text": r["canonical_text"],
                "concept_source": r["concept_source"],
                "entity_type": r["entity_type"],
                "entity_role": role,
                "entity_domain": domain,
                "surface_key": "+".join(surface_key(r["canonical_text"])) or None,
                "entity_types": sorted(entity_labels.get(eid, ())) or None,
                "entity_id": eid,
                "authority_uri": authority_uri,
                "entity_label": res.match.label if res.match else (
                    clusters[eid].label if eid and eid in clusters else None),
                "entity_source": source,
                "outcome": res.outcome.value,
                "decided_by": res.decided_by if res.outcome is Outcome.LINKED else (
                    "cluster" if eid else None),
                "reason": res.reason or None,
                "n_candidates": len(res.candidates) or None,
            }, ensure_ascii=False) + "\n")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "normalizer_version": NORMALIZER_VERSION,
        "wordnet_available": wordnet_guard.available(),
        "inputs": [str(p) for p in inputs],
        "dictionary": d.provenance(),
        "counts": dict(stats),
        "n_mentions": len(rows),
        "reused_uids_in_input": reused_uids,
        "spans_split": n_split,
        "grounded_via_singular": n_singular,
        "splitter_model": (splitter.model if splitter and splitter.available else None),
        "concept_text": dict(concept_stats),
        "n_entities": len(grounded_entities) + len(local_entities),
        "n_entities_from_authorities": len(grounded_entities),
        "n_entities_project_local": len(local_entities),
        # Content keys minted for unresolved mentions. Larger than
        # n_entities_project_local by however many chained into a grounded
        # component; the gap report lists only the ones that stayed unresolved.
        "n_unresolved_clusters": len(clusters),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")

    print(f"\nwrote {args.out}  ({len(rows):,} mentions)")
    print(f"      {meta_path}")
    if n_split:
        print(f"  coordination: {n_split} spans split into parts "
              f"({splitter.model})")
    if n_singular:
        print(f"  singular:     {n_singular:,} mention(s) grounded via their "
              f"singular after matching nothing")
    used = sum(v for k, v in concept_stats.items() if k.startswith("concept_text:"))
    rejected = concept_stats["concept_text_rejected"]
    if used or rejected:
        print(f"  concept_text: {used:,} honoured, {rejected:,} rejected "
              f"(tokens absent from the sentence and the document)")
    for k in ("linked", "ambiguous", "nil", "unrouted"):
        print(f"  {k:10s} {stats[k]:6,}  ({100*stats[k]/max(len(rows),1):4.1f}%)")
    print(f"\n  distinct entities: {len(grounded_entities) + len(local_entities):,}  "
          f"({len(grounded_entities):,} from authorities, "
          f"{len(local_entities):,} project-local)")
    print(f"  {len(rows) - stats['ambiguous']:,} mentions carry an entity_id; "
          f"{stats['ambiguous']:,} deliberately left unresolved")

    if args.gap_report:
        # Only the clusters that stayed unresolved. A key that chained into a
        # grounded component IS covered by an authority, so listing it here
        # would overstate the gap — and its minted id appears nowhere in the
        # table, since the component is named by its URI instead.
        ranked = [c for c in rank_by_document_frequency(clusters)
                  if c.entity_id in local_entities]
        args.gap_report.parent.mkdir(parents=True, exist_ok=True)
        with args.gap_report.open("w", encoding="utf-8") as fh:
            fh.write("doc_freq\tmentions\tentity_type\tlabel\tentity_id\tsurface_forms\n")
            for c in ranked:
                forms = "; ".join(f"{f}({n})" for f, n in c.surface_forms.most_common())
                fh.write(f"{c.doc_frequency}\t{c.n_mentions}\t{c.entity_type}\t"
                         f"{c.label}\t{c.entity_id}\t{forms}\n")
        grounded_after_chaining = len(clusters) - len(ranked)
        print(f"      {args.gap_report}  ({len(ranked):,} unresolved entities"
              + (f"; {grounded_after_chaining} more chained into a grounded one"
                 if grounded_after_chaining else "") + ")")

    # Document frequency first, then mentions, then the id: the same total
    # ordering for every report, so they read as one set and each is
    # byte-reproducible.
    def _ranked(items):
        return sorted(items, key=lambda kv: (-len(kv[1]["docs"]),
                                             -sum(kv[1]["surface"].values()), kv[0]))

    if args.entity_report:
        args.entity_report.parent.mkdir(parents=True, exist_ok=True)
        with args.entity_report.open("w", encoding="utf-8") as fh:
            fh.write("status\tdoc_freq\tmentions\tentity_type\trole\tdomain\tconcept"
                     "\tcurie\tsource\tscope\tvia_singular\tentity_id\tauthority_uri"
                     "\tsurface_forms\n")
            for eid, e in _ranked(entities.items()):
                forms = "; ".join(f"{f}({n})" for f, n in e["surface"].most_common())
                # A project-local entity has no authority label; its cluster
                # label — the commonest surface form — is the readable name.
                name = e["label"] or (clusters[eid].label if eid in clusters else "")
                fh.write(f"{'grounded' if e['grounded'] else 'project_local'}\t"
                         f"{len(e['docs'])}\t{sum(e['surface'].values())}\t"
                         f"{e['entity_type']}\t{e['role']}\t{e['domain'] or ''}\t{name}\t"
                         f"{e['curie'] or ''}\t{e['source'] or ''}\t{e['scope'] or ''}\t"
                         f"{e['via_singular']}\t{eid}\t{e['uri'] or ''}\t{forms}\n")
        n_g = sum(1 for e in entities.values() if e["grounded"])
        print(f"      {args.entity_report}  ({len(entities):,} entities: "
              f"{n_g:,} grounded, {len(entities)-n_g:,} project-local)")

    if args.linked_report:
        order = _ranked([(k, v) for k, v in entities.items() if v["grounded"]])
        args.linked_report.parent.mkdir(parents=True, exist_ok=True)
        with args.linked_report.open("w", encoding="utf-8") as fh:
            fh.write("doc_freq\tmentions\tentity_type\tconcept\tcurie\tsource"
                     "\tscope\tvia_singular\tentity_id\tauthority_uri"
                     "\tsurface_forms\n")
            for eid, e in order:
                forms = "; ".join(f"{f}({n})" for f, n in e["surface"].most_common())
                fh.write(f"{len(e['docs'])}\t{sum(e['surface'].values())}\t"
                         f"{e['entity_type']}\t{e['label'] or ''}\t{e['curie'] or ''}\t"
                         f"{e['source'] or ''}\t{e['scope'] or ''}\t"
                         f"{e['via_singular']}\t{eid}\t{e['uri']}\t{forms}\n")
        by_source = Counter(e["source"] for _, e in order)
        print(f"      {args.linked_report}  ({len(order):,} grounded entities: "
              + ", ".join(f"{n} {s}" for s, n in by_source.most_common()) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
