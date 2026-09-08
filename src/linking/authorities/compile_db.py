"""
Compile authority snapshots into the SQLite artifacts the pipeline reads.

This is the build-machine half of the split. Collaborators never run it: they
download the .sqlite files it produces and match against those, offline.

Why SQLite, and why FTS5
------------------------
FTS5 ships BM25 inside SQLite, and ``sqlite3`` is in the standard library. That
one choice removes faiss, a separate BM25 implementation and any search server
from the required path, so Tier A needs no GPU, no service and no extra install.

Why one artifact per group rather than one file
-----------------------------------------------
Forced by size, and confirmed by measurement. The four ontologies together are
69 MB; GBIF's backbone is 7.7M names and GeoNames 13.5M places, and one file
holding all three lands near 4 GB. Because each entity type routes to exactly
one group, a collaborator who never grounds SPATIAL ENTITY has no use for
GeoNames. So the ontologies are required and the other two are opt-in.

Three things the schema is deliberate about
-------------------------------------------
**Synonym scope is preserved, never flattened.** OBO distinguishes
``hasExactSynonym`` from ``hasBroadSynonym`` / ``hasNarrowSynonym`` /
``hasRelatedSynonym``. Collapsing them into "alias" is the hypernym merge this
design exists to prevent: a BROAD synonym names a concept the term is *subsumed
by*, not equal to. ``alias.scope`` is recorded so the matcher can require
label-or-exact for a confident link and treat looser scopes as candidates only.

**Source comes from each term's own URI, not the file it arrived in.** ENVO
inlines imported terms from CHEBI, PATO, UBERON and others. Labelling those
"envo" would misattribute provenance and double-load PATO. A term native to its
own file wins (INSERT OR REPLACE); an inlined import only fills a gap
(INSERT OR IGNORE), so precedence holds regardless of processing order.

**Obsolete terms are kept and flagged, not dropped.** An obsolete ENVO term
still tells you what a mention meant. It must not be a link target, but
"matched only an obsolete term" is a distinguishable, informative outcome, and
that cannot be reported if the row was never loaded.

Normaliser coupling
-------------------
``alias.norm`` is produced by ``mention_norm.normalize_key``, the same function
the matcher later runs over mentions. If the two diverge, exact match silently
under-fires — no error, just a lower link rate that reads as a property of the
data. ``NORMALIZER_VERSION`` is written into ``build_meta`` so the matcher can
refuse a stale artifact loudly instead.

Usage
-----
    python -m src.linking.authorities.compile_db                  # ontologies
    python -m src.linking.authorities.compile_db --group gbif
    python -m src.linking.authorities.compile_db --group geonames
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from src.linking.authorities.registry import (
    AUTHORITIES, GROUPS, UNIVERSAL_TYPE, Authority, artifact_path, file_sha256,
    load_manifest,
)
from src.linking.mention_norm import NORMALIZER_VERSION, normalize_display, normalize_key

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2"
BATCH = 100_000

_OBO_PREFIX = "http://purl.obolibrary.org/obo/"

_SCOPE = {
    "hasExactSynonym": "exact",
    "hasNarrowSynonym": "narrow",
    "hasBroadSynonym": "broad",
    "hasRelatedSynonym": "related",
}

CONFIDENT_SCOPES = ("label", "exact", "vernacular")

# ── Upstream scope errors ────────────────────────────────────────
#
# Corrections to synonym scopes that are wrong in the released ontology, keyed
# by (concept URI, normalised alias). Data repair, deliberately not a matching
# heuristic: each entry names two rows in one release and changes nothing else.
# The matcher cannot do this job, because a mis-scoped synonym is
# string-identical to a correct one.
#
# ENVO ships `mountain` and `seamount` with their scopes swapped for this pair:
#
#     ENVO:00000081  mountain   hasRelatedSynonym  "mountains"
#     ENVO:00000264  seamount   hasExactSynonym    "mountains"
#
# `related` is not an identity scope, so the correct concept is excluded and the
# undersea one wins every time: all 14 mentions of `mountains` in the corpus
# resolved to *seamount*, in a corpus about mountain biodiversity. Singular
# `mountain` was unaffected, which is what makes it a data bug rather than a
# policy one.
#
# Keep this table tiny and report upstream. Corrections that no longer fire are
# listed in `build_meta` so an ontology upgrade shows which entries went stale.
OBO_SCOPE_CORRECTIONS = {
    (f"{_OBO_PREFIX}ENVO_00000264", "mountains"): "related",
    (f"{_OBO_PREFIX}ENVO_00000081", "mountains"): "exact",
}

# ── GBIF filters ─────────────────────────────────────────────────
#
# Measured over the 7,746,724-row backbone:
#   species 4,994,322 | unranked 1,275,874 | genus 547,518 | variety 421,934
#   subspecies 380,481 | form 87,936 | family 34,356 | order 3,092 | ...
#
# ``unranked`` is dropped because it is not names: it is 1.3M UNITE fungal
# sequence identifiers ("SH1076374.09FU"). ``doubtful`` (300,247) is dropped
# because GBIF itself does not vouch for it. variety and form are dropped as
# below the resolution any ecology abstract writes at.
GBIF_RANKS = {"kingdom", "phylum", "class", "order", "family", "genus",
              "species", "subspecies"}
GBIF_SYNONYM_STATUS = {"synonym", "homotypic synonym", "heterotypic synonym",
                       "proparte synonym"}
GBIF_VERNACULAR_LANGS = {"en"}

# ── GeoNames filters ─────────────────────────────────────────────
#
# Not only a size cut — a precision one. GeoNames' long tail is 504k hills,
# 179k localities and 225k fourth-order admin units, an unusually rich source of
# collisions with ordinary English words. For a corpus of 154 sentences that is
# a false-positive generator, so the kept set is the feature codes that ecology
# writing actually names: ranges, plateaus, parks, reserves, regions, countries.
GEONAMES_KEEP = {
    "A": {"PCLI", "PCL", "PCLD", "PCLF", "PCLIX", "PCLS", "ADM1", "ADM2", "TERR", "ZN"},
    "T": {"MTS", "MT", "PK", "PKS", "PLAT", "PLATS", "RDGE", "VAL", "VALS", "PASS",
          "UPLD", "HLLS", "ISL", "ISLS", "CAPE", "CNYN", "DSRT", "PEN", "MASS",
          "RANGE", "SLP", "GRGE"},
    "L": {"PRK", "RESV", "RESN", "RESW", "RESF", "RESH", "AREA", "RGN", "CONT", "RESA"},
    "V": {"FRST", "FRSTS", "GRSLD", "SCRB", "HTH", "MDW", "TUND", "GROVE"},
}
GEONAMES_URI = "https://sws.geonames.org/{}/"
GBIF_URI = "https://www.gbif.org/species/{}"

SCHEMA = """
PRAGMA journal_mode = OFF;
PRAGMA synchronous  = OFF;

CREATE TABLE concept (
    uri        TEXT PRIMARY KEY,
    curie      TEXT,
    source     TEXT NOT NULL,
    label      TEXT NOT NULL,
    definition TEXT,
    namespace  TEXT,
    obsolete   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE alias (
    uri    TEXT NOT NULL,
    name   TEXT NOT NULL,
    norm   TEXT NOT NULL,
    scope  TEXT NOT NULL,
    source TEXT NOT NULL
);

-- The type gate, carried in the artifact so the matcher can route without
-- importing the Python registry.
CREATE TABLE source_serves (source TEXT NOT NULL, entity_type TEXT NOT NULL);

CREATE TABLE build_meta (key TEXT PRIMARY KEY, value TEXT);
"""

INDEXES = """
CREATE INDEX alias_norm_idx  ON alias(norm);
CREATE INDEX alias_uri_idx   ON alias(uri);
CREATE INDEX concept_src_idx ON concept(source);
CREATE INDEX serves_type_idx ON source_serves(entity_type);
CREATE VIRTUAL TABLE alias_fts USING fts5(name, uri UNINDEXED, tokenize='unicode61');
"""

# A parser yields ("concept", row) and ("alias", row) tuples, streamed rather
# than accumulated: GBIF alone would put ~4M concepts in memory at once.
Rec = Tuple[str, tuple]


# ── OBO Graphs ───────────────────────────────────────────────────

def _split_uri(uri: str) -> Tuple[Optional[str], Optional[str]]:
    """(source prefix, CURIE) for an OBO PURL, else (None, None).

    Non-OBO ids appear in these files (``go#goslim_agr`` and similar internal
    annotation properties). They are not concepts, so they are skipped rather
    than coerced into something that looks like one.
    """
    if not uri.startswith(_OBO_PREFIX):
        return None, None
    tail = uri[len(_OBO_PREFIX):]
    if "_" not in tail or "#" in tail or "/" in tail:
        return None, None
    prefix, local = tail.split("_", 1)
    if not prefix.isalnum():
        return None, None
    return prefix.lower(), f"{prefix}:{local}"


_CORRECTION_STAT = "scope_correction"


def _correction_stat(uri: str, norm: str, was: str, now: str) -> str:
    return f"{_CORRECTION_STAT}:{uri.rsplit('/', 1)[-1]}|{norm}|{was}->{now}"


def stale_corrections(all_stats: dict) -> List[str]:
    """Entries of ``OBO_SCOPE_CORRECTIONS`` that no correction fired for.

    A correction that stops firing means the release fixed it upstream, or the
    concept was retired — either way the entry is now dead weight pretending to
    be doing something. Surfaced at build time rather than discovered later.
    """
    fired = {k.split(":", 1)[1].rsplit("|", 1)[0]
             for s in all_stats.values() for k in s
             if k.startswith(f"{_CORRECTION_STAT}:")}
    return sorted(f"{uri.rsplit('/', 1)[-1]}|{norm}"
                  for uri, norm in OBO_SCOPE_CORRECTIONS
                  if f"{uri.rsplit('/', 1)[-1]}|{norm}" not in fired)


def parse_obograph(auth: Authority, stats: Counter) -> Iterator[Rec]:
    with auth.path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    native = auth.key.split("_", 1)[0]

    for graph in doc.get("graphs") or []:
        for node in graph.get("nodes") or []:
            stats["rows"] += 1
            if node.get("type") != "CLASS":
                continue
            uri = node.get("id") or ""
            source, curie = _split_uri(uri)
            if not source:
                continue
            label = normalize_display(node.get("lbl"))
            if not label:
                continue

            meta = node.get("meta") or {}
            ns = None
            for bpv in (meta.get("basicPropertyValues") or []):
                if str(bpv.get("pred", "")).endswith("hasOBONamespace"):
                    ns = bpv.get("val")
                    break
            # GO ships three ontologies in one file. Filter only the file's own
            # terms: an imported CHEBI term carries no GO namespace and would be
            # discarded by a blanket test.
            if auth.namespace_filter and source == native and ns != auth.namespace_filter:
                stats["skipped_namespace"] += 1
                continue

            obsolete = bool(meta.get("deprecated"))
            stats["concepts"] += 1
            stats["obsolete"] += int(obsolete)
            yield ("concept", (uri, curie, source, label,
                               (meta.get("definition") or {}).get("val") or None,
                               ns, int(obsolete), int(source == native)))
            yield ("alias", (uri, label, normalize_key(label), "label", source))
            for syn in (meta.get("synonyms") or []):
                scope = _SCOPE.get(str(syn.get("pred", "")).split("#")[-1])
                val = normalize_display(syn.get("val"))
                if scope and val:
                    norm = normalize_key(val)
                    corrected = OBO_SCOPE_CORRECTIONS.get((uri, norm))
                    if corrected and corrected != scope:
                        # Counted in stats like everything else, so build_meta
                        # can name the corrections that actually fired.
                        stats[_correction_stat(uri, norm, scope, corrected)] += 1
                        scope = corrected
                    stats[f"syn_{scope}"] += 1
                    yield ("alias", (uri, val, norm, scope, source))


# ── GBIF Darwin Core Archive ─────────────────────────────────────

def parse_gbif(auth: Authority, stats: Counter) -> Iterator[Rec]:
    """Taxon.tsv then VernacularName.tsv, streamed straight out of the zip.

    The taxonomic-synonym handling is the point of this parser, not a detail.
    GBIF marks 3.3M names as synonyms of an accepted taxon; each becomes an
    ALIAS on the accepted concept rather than a concept of its own. That single
    rule is what makes an old genus placement and its current name resolve to
    one node without any similarity threshold — the same mechanism that puts
    "Bornean Colugo" and "Galeopterus variegatus" on one URI.

    Synonyms pointing at an accepted taxon we filtered out become orphan
    aliases; they are removed in one SQL pass at the end rather than by holding
    4M accepted ids in memory to check against.
    """
    z = zipfile.ZipFile(auth.path)

    with z.open("Taxon.tsv") as fh:
        t = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        hdr = t.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        i_id, i_acc = ix["taxonID"], ix["acceptedNameUsageID"]
        i_sci, i_can = ix["scientificName"], ix["canonicalName"]
        i_rank, i_status = ix["taxonRank"], ix["taxonomicStatus"]

        for line in t:
            p = line.rstrip("\n").split("\t")
            if len(p) <= i_status:
                continue
            stats["rows"] += 1
            if p[i_rank] not in GBIF_RANKS:
                stats["skipped_rank"] += 1
                continue
            status = p[i_status]
            canonical = normalize_display(p[i_can]) or normalize_display(p[i_sci])
            if not canonical:
                continue

            if status == "accepted":
                uri = GBIF_URI.format(p[i_id])
                stats["concepts"] += 1
                yield ("concept", (uri, f"gbif:{p[i_id]}", "gbif", canonical,
                                   None, p[i_rank], 0, 1))
                yield ("alias", (uri, canonical, normalize_key(canonical),
                                 "label", "gbif"))
                sci = normalize_display(p[i_sci])
                if sci and sci != canonical:
                    # scientificName carries the authorship string; it is the
                    # same name, so it is an exact alias.
                    yield ("alias", (uri, sci, normalize_key(sci), "exact", "gbif"))
            elif status in GBIF_SYNONYM_STATUS and p[i_acc]:
                stats["synonyms"] += 1
                yield ("alias", (GBIF_URI.format(p[i_acc]), canonical,
                                 normalize_key(canonical), "exact", "gbif"))
            else:
                stats["skipped_status"] += 1

    with z.open("VernacularName.tsv") as fh:
        t = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        hdr = t.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        i_id, i_name, i_lang = ix["taxonID"], ix["vernacularName"], ix["language"]
        for line in t:
            p = line.rstrip("\n").split("\t")
            if len(p) <= i_lang or p[i_lang] not in GBIF_VERNACULAR_LANGS:
                continue
            name = normalize_display(p[i_name])
            if not name:
                continue
            stats["vernacular"] += 1
            yield ("alias", (GBIF_URI.format(p[i_id]), name,
                             normalize_key(name), "vernacular", "gbif"))


# ── GeoNames dump ────────────────────────────────────────────────

def parse_geonames(auth: Authority, stats: Counter) -> Iterator[Rec]:
    """allCountries.txt: 19 tab-separated columns, no header.

    ``alternatenames`` (column 3) is a comma-separated list of name variants for
    the same place, so those are exact aliases rather than related ones — they
    denote the identical feature, just in another language or transliteration.
    The column does also carry the occasional URL and bare numeric code, which
    are skipped.
    """
    z = zipfile.ZipFile(auth.path)
    with z.open("allCountries.txt") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 19:
                continue
            stats["rows"] += 1
            if p[7] not in GEONAMES_KEEP.get(p[6], ()):
                continue
            name = normalize_display(p[1])
            if not name:
                continue
            uri = GEONAMES_URI.format(p[0])
            stats["concepts"] += 1
            yield ("concept", (uri, f"geonames:{p[0]}", "geonames", name,
                               None, f"{p[6]}.{p[7]}", 0, 1))
            yield ("alias", (uri, name, normalize_key(name), "label", "geonames"))

            seen = {normalize_key(name)}
            for raw in [p[2]] + p[3].split(","):
                alt = normalize_display(raw)
                if not alt or "://" in alt or alt.isdigit():
                    continue
                key = normalize_key(alt)
                if key and key not in seen:
                    seen.add(key)
                    stats["alt_names"] += 1
                    yield ("alias", (uri, alt, key, "exact", "geonames"))


PARSERS = {"obograph": parse_obograph, "gbif_dwca": parse_gbif,
           "geonames_dump": parse_geonames}


# ── build ────────────────────────────────────────────────────────

def build(group: str, out: Optional[Path] = None) -> int:
    keys = GROUPS[group]
    out = out or artifact_path(group)

    missing = [k for k in keys if not AUTHORITIES[k].path.exists()]
    if missing:
        print(f"snapshots not fetched: {', '.join(missing)}\n"
              f"run: python -m src.linking.authorities.fetch "
              f"--key {' --key '.join(missing)}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    all_stats = {}

    try:
        conn.executescript(SCHEMA)
        for key in keys:
            auth = AUTHORITIES[key]
            parser = PARSERS.get(auth.fmt)
            if parser is None:
                print(f"  {key:10s} no parser for format {auth.fmt!r}", file=sys.stderr)
                return 1
            stats: Counter = Counter()
            print(f"  parsing {key} ...", file=sys.stderr)

            cbuf: List[tuple] = []
            abuf: List[tuple] = []
            for kind, row in parser(auth, stats):
                if kind == "concept":
                    cbuf.append(row)
                    if len(cbuf) >= BATCH:
                        _flush_concepts(conn, cbuf)
                else:
                    abuf.append(row)
                    if len(abuf) >= BATCH:
                        conn.executemany(
                            "INSERT INTO alias(uri,name,norm,scope,source) VALUES (?,?,?,?,?)",
                            abuf)
                        abuf.clear()
            _flush_concepts(conn, cbuf)
            if abuf:
                conn.executemany(
                    "INSERT INTO alias(uri,name,norm,scope,source) VALUES (?,?,?,?,?)",
                    abuf)
            conn.commit()
            all_stats[key] = dict(stats)
            print(f"  {key:10s} {stats['concepts']:>9,} concepts from "
                  f"{stats['rows']:>10,} rows", file=sys.stderr)

        # A universal authority is written once against the "*" marker rather
        # than enumerated: the schema's off-schema labels (NEGATION, CHANGE…)
        # cannot be listed in advance, and they must route too.
        serves_rows = []
        for k in keys:
            src = AUTHORITIES[k].key.split("_", 1)[0]
            if AUTHORITIES[k].universal:
                serves_rows.append((src, UNIVERSAL_TYPE))
            serves_rows += [(src, et) for et in AUTHORITIES[k].serves]
        conn.executemany(
            "INSERT INTO source_serves(source,entity_type) VALUES (?,?)", serves_rows)

        # Aliases whose concept was filtered out (a GO molecular_function term,
        # a GBIF synonym of a dropped taxon) point at nothing.
        orphans = conn.execute(
            "DELETE FROM alias WHERE uri NOT IN (SELECT uri FROM concept)").rowcount
        conn.commit()

        # De-duplicate. The same alias legitimately arrives more than once — a
        # PATO term inlined into ENVO carries its synonyms in both files, and a
        # GBIF taxon can hold several vernacular rows that normalise alike.
        # Done as a rebuild rather than a UNIQUE constraint because the implicit
        # index would slow the multi-million-row bulk load it exists to protect.
        before = conn.execute("SELECT COUNT(*) FROM alias").fetchone()[0]
        conn.executescript("""
            CREATE TABLE alias_dedup AS
                SELECT uri, name, norm, scope, source FROM alias
                GROUP BY uri, norm, scope;
            DROP TABLE alias;
            ALTER TABLE alias_dedup RENAME TO alias;
        """)
        dupes = before - conn.execute("SELECT COUNT(*) FROM alias").fetchone()[0]
        conn.commit()

        print("  indexing ...", file=sys.stderr)
        conn.executescript(INDEXES)
        # Populate FTS after the bulk load: incremental index maintenance during
        # a multi-million-row insert is dramatically slower than one build.
        conn.execute("INSERT INTO alias_fts(name, uri) SELECT name, uri FROM alias")

        n_c = conn.execute("SELECT COUNT(*) FROM concept").fetchone()[0]
        n_a = conn.execute("SELECT COUNT(*) FROM alias").fetchone()[0]
        manifest = load_manifest()
        meta = {
            "group": group,
            "schema_version": SCHEMA_VERSION,
            "normalizer_version": NORMALIZER_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "authorities": json.dumps(
                {k: {"pinned_version": AUTHORITIES[k].version,
                     "payload_version": (manifest.get(k) or {}).get("payload_version"),
                     "sha256": (manifest.get(k) or {}).get("sha256")}
                 for k in keys}, sort_keys=True),
            "parse_stats": json.dumps(all_stats, sort_keys=True),
            "scope_corrections": json.dumps(
                {"declared": sorted(f"{u.rsplit('/', 1)[-1]}|{n}"
                                    for u, n in OBO_SCOPE_CORRECTIONS),
                 "stale": stale_corrections(all_stats)}, sort_keys=True),
            "n_concepts": str(n_c),
            "n_aliases": str(n_a),
            "orphan_aliases_dropped": str(orphans),
            "duplicate_aliases_dropped": str(dupes),
        }
        conn.executemany("INSERT INTO build_meta(key,value) VALUES (?,?)",
                         sorted(meta.items()))
        conn.commit()
    finally:
        conn.close()

    tmp.replace(out)
    print(f"\n  wrote {out}  ({out.stat().st_size/1e6:,.1f} MB)")
    fired = sum(v for s in all_stats.values() for k, v in s.items()
                if k.startswith(f"{_CORRECTION_STAT}:"))
    if fired:
        print(f"  {fired} upstream synonym scope(s) corrected")
    stale = stale_corrections(all_stats)
    if stale and any(AUTHORITIES[k].fmt == "obograph" for k in keys):
        print(f"  WARNING scope correction(s) never fired, so they are now dead "
              f"entries or the concept moved: {', '.join(stale)}", file=sys.stderr)
    print(f"  {n_c:,} concepts, {n_a:,} aliases"
          f"{f', {orphans:,} orphan' if orphans else ''}"
          f"{f', {dupes:,} duplicate' if dupes else ''}"
          f"{' aliases dropped' if (orphans or dupes) else ''}")
    print(f"  sha256 {file_sha256(out)}")
    return 0


def _flush_concepts(conn: sqlite3.Connection, buf: List[tuple]) -> None:
    """Native terms overwrite; inlined imports only fill gaps.

    Two statements rather than one because the precedence has to hold whichever
    order the files are read in — see the module docstring.
    """
    if not buf:
        return
    native = [r[:7] for r in buf if r[7]]
    imported = [r[:7] for r in buf if not r[7]]
    sql = ("INSERT OR {} INTO concept"
           "(uri,curie,source,label,definition,namespace,obsolete)"
           " VALUES (?,?,?,?,?,?,?)")
    if native:
        conn.executemany(sql.format("REPLACE"), native)
    if imported:
        conn.executemany(sql.format("IGNORE"), imported)
    buf.clear()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="ontologies", choices=sorted(GROUPS),
                    help="which artifact to build (default: ontologies)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return build(args.group, out=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
