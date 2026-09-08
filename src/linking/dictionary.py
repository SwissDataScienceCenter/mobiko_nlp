"""
Read access to the compiled authority artifacts.

The dictionary is physically several SQLite files — ``mobiko_ontologies``,
``mobiko_gbif``, ``mobiko_geonames`` — because one file holding all of them
lands near 4 GB and most users need only the first. This module hides that:
callers ask the dictionary a question, not a particular file.

The artifacts are ATTACHed to a single connection, so a lookup is one SQL
statement over a generated UNION rather than a Python loop issuing N queries.
That matters at Phase 3 volumes, where every mention is looked up several times
under different scope sets.

Two guards worth knowing about
------------------------------
**Normaliser version.** ``alias.norm`` was written by
``mention_norm.normalize_key`` at build time and mentions are normalised by the
same function at match time. If those diverge, exact match silently under-fires
— no error, just a lower link rate that reads as a property of the data. Any
artifact built by a different normaliser version is refused at open, loudly.

**Missing is not empty.** An absent artifact means a whole entity type cannot be
grounded — SPATIAL ENTITY scores zero without GeoNames. ``present`` and
``missing_groups`` are exposed so a report can say "not attempted" rather than
"attempted and missed", which are very different facts about the pipeline.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.linking.authorities.registry import (ARTIFACT_DIR, GROUPS,
                                               UNIVERSAL_TYPE, artifact_path)
from src.linking.mention_norm import NORMALIZER_VERSION

logger = logging.getLogger(__name__)

# Scopes that assert identity. narrow/broad/related do not: a BROAD synonym
# names a concept the term is subsumed by, which is the hypernym merge the
# whole design exists to avoid.
CONFIDENT_SCOPES: Tuple[str, ...] = ("label", "exact", "vernacular")
LOOSE_SCOPES: Tuple[str, ...] = ("narrow", "broad", "related")


class StaleArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class Match:
    uri: str
    curie: Optional[str]
    source: str
    label: str
    scope: str
    matched_name: str
    obsolete: bool
    group: str
    # For GBIF this is the taxonomic rank ("order", "species"); for the OBO
    # ontologies the OBO namespace; for GeoNames the feature code. Carried on
    # the Match because rank is what breaks a homonym tie, and re-querying for
    # it per candidate would issue a round trip inside the hot loop.
    rank: Optional[str] = None


class Dictionary:
    """Query surface over whichever compiled artifacts are on disk."""

    def __init__(self, paths: Dict[str, Path]):
        self.paths = paths
        self._conn = sqlite3.connect(":memory:")
        self._schemas: List[Tuple[str, str]] = []   # (sqlite schema name, group)

        for group, path in paths.items():
            schema = f"g_{group}"
            self._conn.execute("ATTACH DATABASE ? AS " + schema, (str(path),))
            built = dict(self._conn.execute(
                f"SELECT key, value FROM {schema}.build_meta"))
            if built.get("normalizer_version") != NORMALIZER_VERSION:
                raise StaleArtifactError(
                    f"{path} was built with normalizer "
                    f"v{built.get('normalizer_version')} but this code is "
                    f"v{NORMALIZER_VERSION}. Rebuild it, or the dictionary and "
                    f"the mentions normalise differently and exact match "
                    f"silently under-fires.")
            self._schemas.append((schema, group))

        self._routing: Dict[str, Set[str]] = defaultdict(set)
        self._universal: Set[str] = set()
        self._source_group: Dict[str, str] = {}
        for schema, group in self._schemas:
            for source, etype in self._conn.execute(
                    f"SELECT source, entity_type FROM {schema}.source_serves"):
                if etype == UNIVERSAL_TYPE:
                    self._universal.add(source)
                else:
                    self._routing[etype].add(source)
                self._source_group[source] = group

    # ── construction ─────────────────────────────────────────────

    @classmethod
    def open(cls, directory: Path = ARTIFACT_DIR,
             groups: Optional[Iterable[str]] = None) -> "Dictionary":
        wanted = list(groups) if groups is not None else list(GROUPS)
        found = {}
        for g in wanted:
            p = artifact_path(g) if directory == ARTIFACT_DIR else directory / f"mobiko_{g}.sqlite"
            if p.exists():
                found[g] = p
        if not found:
            raise FileNotFoundError(
                f"no compiled artifacts in {directory}; run "
                f"python -m src.linking.authorities.compile_db")
        return cls(found)

    @property
    def present(self) -> List[str]:
        return [g for _, g in self._schemas]

    @property
    def missing_groups(self) -> List[str]:
        return [g for g in GROUPS if g not in self.present]

    def sources_for(self, entity_type: str) -> Set[str]:
        """Which sources may answer for a canonicalised entity type.

        Universal authorities (the concept ontologies) answer for every type;
        the name authorities answer only for the types they are gated to. See
        ``registry.Authority`` for why the two are treated differently.

        Empty means the type is not routable with the artifacts present — which
        the caller must report as "not attempted", not as a miss.
        """
        gated = self._routing.get((entity_type or "").strip().upper(), set())
        return gated | self._universal

    # ── lookup ───────────────────────────────────────────────────

    def _union(self, body: str) -> str:
        return " UNION ALL ".join(body.format(s=schema, g=group)
                                  for schema, group in self._schemas)

    def lookup(self, norm: str, sources: Sequence[str],
               scopes: Sequence[str] = CONFIDENT_SCOPES,
               include_obsolete: bool = False,
               limit: int = 10) -> List[Match]:
        """Exact lookup of a normalised key, gated by source and alias scope."""
        if not norm or not sources:
            return []
        src_ph = ",".join("?" * len(sources))
        sc_ph = ",".join("?" * len(scopes))
        obs = "" if include_obsolete else " AND c.obsolete = 0"
        sql = self._union(
            f"""SELECT c.uri, c.curie, c.source, c.label, a.scope, a.name,
                       c.obsolete, '{{g}}', c.namespace
                FROM {{s}}.alias a JOIN {{s}}.concept c USING(uri)
                WHERE a.norm = ? AND a.scope IN ({sc_ph})
                  AND c.source IN ({src_ph}){obs}""")
        params: List = []
        for _ in self._schemas:
            params += [norm, *scopes, *sources]
        rows = self._conn.execute(f"{sql} LIMIT {int(limit)}", params).fetchall()
        return [Match(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6]), r[7], r[8])
                for r in rows]

    def search(self, text: str, sources: Sequence[str],
               limit: int = 10) -> List[Match]:
        """BM25 candidate generation over alias names.

        Candidates only — never a decision. Ranking by string overlap cannot
        tell a synonym from an antonym, which is the entire reason the matcher
        keeps retrieval and adjudication apart.
        """
        terms = [t for t in text.split() if t]
        if not terms or not sources:
            return []
        query = " ".join(f'"{t}"' for t in terms)
        src_ph = ",".join("?" * len(sources))
        sql = self._union(
            f"""SELECT c.uri, c.curie, c.source, c.label, a.scope, a.name,
                       c.obsolete, '{{g}}', c.namespace, bm25(alias_fts) AS score
                FROM {{s}}.alias_fts
                JOIN {{s}}.concept c ON c.uri = alias_fts.uri
                JOIN {{s}}.alias a ON a.uri = alias_fts.uri
                                  AND a.name = alias_fts.name
                WHERE alias_fts MATCH ? AND c.obsolete = 0
                  AND c.source IN ({src_ph})""")
        params: List = []
        for _ in self._schemas:
            params += [query, *sources]
        rows = self._conn.execute(
            f"SELECT * FROM ({sql}) ORDER BY score LIMIT {int(limit)}", params).fetchall()
        return [Match(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6]), r[7], r[8])
                for r in rows]

    # ── provenance ───────────────────────────────────────────────

    def provenance(self) -> Dict[str, dict]:
        """What to stamp into ``run_meta``: the vocabulary this run actually had."""
        out: Dict[str, dict] = {}
        for schema, group in self._schemas:
            meta = dict(self._conn.execute(
                f"SELECT key, value FROM {schema}.build_meta"))
            out[group] = {
                "path": str(self.paths[group]),
                "built_at": meta.get("built_at"),
                "schema_version": meta.get("schema_version"),
                "normalizer_version": meta.get("normalizer_version"),
                "n_concepts": int(meta.get("n_concepts") or 0),
                "n_aliases": int(meta.get("n_aliases") or 0),
                "authorities": json.loads(meta.get("authorities") or "{}"),
            }
        out["_missing_groups"] = {"groups": self.missing_groups}
        return out

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Dictionary":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
