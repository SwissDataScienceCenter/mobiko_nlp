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

Why the entity id is type-scoped even when it grounds
-----------------------------------------------------
Entity type is part of entity identity (decision, 7 Sep), so ``biodiversity`` as
BIOTIC PROPERTY and as BIOTIC ENTITY are two entities. Grounding must not quietly
undo that. `habitat` is typed ABIOTIC ENTITY in one sentence and BIOTIC ENTITY in
another and both link to ENVO:01000739; on the raw URI they would merge, in
contradiction of the rule. Ten pairs in the gold do this — `habitat`, `size`,
`climate`, `grassland`, `length`, `rhizosphere`, `orientation`, `temperate`,
`sensitive`, `variability` — and every one is an annotator disagreement about
domain or role, not about which concept is meant.

So identity is type-scoped and grounding is not: ``entity_id`` carries the type,
while ``authority_uri`` carries the bare external URI. Two entities may share an
authority_uri, which an RDF export renders as two nodes with `skos:exactMatch`
to one ENVO term — legitimate, and visible to a human who may want to merge
them. Splitting is the reversible error; merging is not.

Why relations are untouched
---------------------------
Relations stay exactly where they are, in the input files. When their mapping is
built later it is a join on ``uid`` against this table — no rework on either
side. That is the reason ``uid`` is the key and not, say, a normalised string.

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
from src.linking.cluster import build_clusters, rank_by_document_frequency
from src.linking.dictionary import Dictionary
from src.linking.match import Outcome, resolve
from src.linking import wordnet_guard
from src.linking.mention_norm import NORMALIZER_VERSION

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


def iter_spans(path: Path) -> Iterator[dict]:
    """Yield one dict per span, from either file shape in use."""
    seen_uids: Counter = Counter()
    with path.open(encoding="utf-8") as fh:
        for line_i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s: unparseable line %d, skipped", path, line_i)
                continue
            base_doc = obj.get("doc_id") or path.stem
            sentences = obj.get("sentences")
            blocks = sentences if isinstance(sentences, list) else [obj]
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
                        # Optional, and honoured when present — see the plan.
                        "concept_text": sp.get("concept_text"),
                    }


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
    ap.add_argument("--gap-report", type=Path, default=None,
                    help="also write the unresolved clusters ranked by document "
                         "frequency: the measured ontology coverage gap")
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

    # Pass 1 — resolve every mention against the authorities.
    rows: List[dict] = []
    stats: Counter = Counter()
    reused_uids = 0
    for path in inputs:
        for sp in iter_spans(path):
            if sp["uid_occurrence"] > 1:
                reused_uids += 1
            r = resolve(d, sp["text"], sp["entity_type"])
            stats[r.outcome.value] += 1
            rows.append({**sp, "_res": r})

    # Pass 2 — mint project-local ids for everything that grounded to nothing.
    to_cluster = [(r["mention_id"], r["text"], r["entity_type"], r["doc_id"])
                  for r in rows if r["_res"].outcome in CLUSTERED_OUTCOMES]
    clusters = build_clusters(to_cluster)
    mention_to_entity = {mid: c.entity_id
                         for c in clusters.values() for mid in c.mention_uids}

    distinct_entities: set = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            res = r.pop("_res")
            authority_uri = res.match.uri if res.match else None
            if res.outcome is Outcome.LINKED:
                # Type-scoped: grounding must not undo type-as-identity.
                eid = f"{r['entity_type']}@{authority_uri}"
                source = res.match.source
            elif res.outcome in CLUSTERED_OUTCOMES:
                eid, source = mention_to_entity.get(r["mention_id"]), "cluster"
            else:
                eid, source = None, None
            if eid:
                distinct_entities.add(eid)
            fh.write(json.dumps({
                "mention_id": r["mention_id"],
                "source_file": r["source_file"],
                "uid": r["uid"],
                "uid_occurrence": r["uid_occurrence"],
                "doc_id": r["doc_id"],
                "sent_index": r["sent_index"],
                "start_char": r["start_char"],
                "end_char": r["end_char"],
                "text": r["text"],
                "concept_text": r["concept_text"],
                "entity_type": r["entity_type"],
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
        "n_entities": len(distinct_entities),
        "n_entities_from_authorities": len(distinct_entities) - len(clusters),
        "n_entities_project_local": len(clusters),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")

    print(f"\nwrote {args.out}  ({len(rows):,} mentions)")
    print(f"      {meta_path}")
    for k in ("linked", "ambiguous", "nil", "unrouted"):
        print(f"  {k:10s} {stats[k]:6,}  ({100*stats[k]/max(len(rows),1):4.1f}%)")
    print(f"\n  distinct entities: {len(distinct_entities):,}  "
          f"({len(distinct_entities) - len(clusters):,} from authorities, "
          f"{len(clusters):,} project-local)")
    print(f"  {len(rows) - stats['ambiguous']:,} mentions carry an entity_id; "
          f"{stats['ambiguous']:,} deliberately left unresolved")

    if args.gap_report:
        ranked = rank_by_document_frequency(clusters)
        args.gap_report.parent.mkdir(parents=True, exist_ok=True)
        with args.gap_report.open("w", encoding="utf-8") as fh:
            fh.write("doc_freq\tmentions\tentity_type\tlabel\tentity_id\tsurface_forms\n")
            for c in ranked:
                forms = "; ".join(f"{f}({n})" for f, n in c.surface_forms.most_common())
                fh.write(f"{c.doc_frequency}\t{c.n_mentions}\t{c.entity_type}\t"
                         f"{c.label}\t{c.entity_id}\t{forms}\n")
        print(f"      {args.gap_report}  ({len(ranked):,} unresolved entities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
