"""
Which external authorities we ground mentions against, and exactly which
snapshot of each one a given run used.

Why a registry rather than URLs scattered through the fetch code
----------------------------------------------------------------
Every number this pipeline produces is conditional on the vocabulary it matched
against. ENVO ships a release roughly monthly and GBIF rebuilds its backbone
several times a year, so "we linked 62% of ABIOTIC ENTITY mentions" is not a
fact about the pipeline unless the vocabulary is named. A run that cannot say
which ENVO it used is not reproducible, and re-deriving that after the fact is
the wound `paper/TODO.md` documents for the annotation runs (4 of 16 carry
`run_meta`).

So: authorities are declared here with an explicit pinned version, fetched once,
hashed, and recorded in a manifest. The manifest — not the URL, and not the file
mtime — is what gets stamped into `run_meta`.

Why versions are pinned in this file rather than tracking "current"
-------------------------------------------------------------------
The OBO PURLs (`purl.obolibrary.org/obo/envo.obo`) redirect to the project's
**master branch**, not to a release. That is a moving target which can be
mid-edit: the ENVO master OBO on the day this was written fails to parse at all,
on an escaped comma inside an xref URL. Pinning to a release tag makes the fetch
reproducible and makes an upgrade a deliberate, reviewable edit to this file.

Why OBO Graphs JSON rather than .obo
-------------------------------------
Three reasons, in order of weight:

1. It parses. The `.obo` serialisation goes through fastobo, which rejects the
   current ENVO release outright.
2. It carries **synonym scope** — `hasExactSynonym` versus `hasBroadSynonym` /
   `hasNarrowSynonym` / `hasRelatedSynonym`. That distinction is load-bearing
   here: treating a BROAD synonym as an exact alias is precisely the hypernym
   merge ("wolf" onto "canid") this whole design exists to prevent. ENVO alone
   ships 551 broad and 486 narrow synonyms that must not be matched as equals.
3. It is stdlib `json`, so the required path gains no parser dependency.

Nothing here runs at match time. Fetching is a build-machine operation; the
pipeline reads the compiled SQLite artifact and never touches the network.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Snapshots are large and re-fetchable, so they live outside git. The manifest
# beside them is small and IS tracked — it is the provenance record.
SNAPSHOT_DIR = Path("data/authorities")
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"

# Compiled artifacts, one per group. Splitting them is forced by size: the
# ontologies together are 69 MB, but GBIF's backbone alone is 7.7M names and
# GeoNames 13.5M places, and a single file carrying all three lands near 4 GB.
# Since a given entity type routes to exactly one group, a collaborator who
# never grounds SPATIAL ENTITY has no reason to hold GeoNames. So: the
# ontologies are required, taxa and places are opt-in, and the matcher reads
# whichever artifacts are present.
ARTIFACT_DIR = SNAPSHOT_DIR


def artifact_path(group: str) -> Path:
    return ARTIFACT_DIR / f"mobiko_{group}.sqlite"


# Kept for callers that just want the required one.
ARTIFACT_PATH = artifact_path("ontologies")


@dataclass(frozen=True)
class Authority:
    """One external vocabulary we can ground against.

    ``serves`` lists the canonicalised MoBiKo entity types this authority may
    answer for, and ``universal`` overrides it to mean "any type at all".

    The distinction that decides which an authority gets is **what kind of thing
    it contains**, and it was settled by measurement (7 Sep):

      * **Name authorities must be gated.** GBIF and GeoNames hold millions of
        proper names that collide with ordinary English. Ungated, `data` and
        `areas` resolve to genera, `large` to a butterfly, `extinct` to the
        Labrador Duck, and `SDG` to Sundargarh district. Worse, the collisions
        trigger the cross-authority rule and destroy 46 *correct* links —
        `erosion`, `mammal`, `lake`, `Argentina` — by making them ambiguous.
      * **Concept ontologies can be universal.** ENVO and PATO hold concepts
        rather than names, so a spurious match is rare and the ambiguity rules
        catch it. Opening them to every type gained 51 links, lost **zero**,
        added **zero** ambiguity, and took `unrouted` from 647 mentions to 0.

    ``serves`` is kept on the universal authorities as documentation of the
    types they were designed around; it no longer restricts them.

    ``version`` is the pinned release. Changing it is how the vocabulary is
    upgraded, and doing so invalidates the compiled artifact.
    """

    key: str
    name: str
    url: str
    version: str
    fmt: str                      # "obograph" | "gbif_dwca" | "geonames_dump"
    group: str                    # which compiled artifact this lands in
    filename: str
    serves: tuple                 # canonicalised MoBiKo entity types
    licence: str
    size_hint: str
    # GO ships three ontologies in one file; only biological_process is relevant
    # to BIOTIC PROCESS. None means "take everything".
    namespace_filter: Optional[str] = None
    universal: bool = False
    notes: str = ""

    @property
    def path(self) -> Path:
        return SNAPSHOT_DIR / self.filename


# ── The authorities ──────────────────────────────────────────────
#
# Ordered small-to-large deliberately: the three ontologies are tens of MB and
# prove the fetch->compile path end to end in minutes. GBIF and GeoNames are
# each >1GB and are fetched only once that path is known to work.

ENVO_VERSION = "v2026-06-26"
PATO_VERSION = "v2025-05-14"
GO_VERSION = "2026-08-05"
ECOCORE_VERSION = "v2026-08-31"

AUTHORITIES: Dict[str, Authority] = {
    a.key: a for a in [
        Authority(
            key="envo",
            universal=True,
            name="Environment Ontology",
            url=f"https://raw.githubusercontent.com/EnvironmentOntology/envo/{ENVO_VERSION}/envo.json",
            version=ENVO_VERSION,
            fmt="obograph",
            group="ontologies",
            filename="envo.json",
            serves=("ABIOTIC ENTITY", "ABIOTIC PROCESS", "ABIOTIC PROPERTY",
                    "BIOTIC ENTITY", "BIOTIC PROCESS", "BIOTIC PROPERTY",
                    "ANTHROPOGENIC ENTITY", "ANTHROPOGENIC PROCESS",
                    "ANTHROPOGENIC PROPERTY"),
            licence="CC BY 4.0",
            size_hint="~7 MB",
            notes="The widest-coverage authority for this corpus: most mentions "
                  "are generic environmental terms, not taxa. Note it INLINES "
                  "imported terms from CHEBI, PATO, UBERON and others, so the "
                  "compiler assigns source from each term's own URI prefix "
                  "rather than from the file it arrived in.",
        ),
        Authority(
            key="pato",
            universal=True,
            name="Phenotype And Trait Ontology",
            url=f"https://raw.githubusercontent.com/pato-ontology/pato/{PATO_VERSION}/pato.json",
            version=PATO_VERSION,
            fmt="obograph",
            group="ontologies",
            filename="pato.json",
            serves=("ABIOTIC PROPERTY", "BIOTIC PROPERTY",
                    "ANTHROPOGENIC PROPERTY", "SPATIAL PROPERTY",
                    "QUALITATIVE PROPERTY"),
            licence="CC BY 3.0",
            size_hint="~3 MB",
            notes="Qualities and traits. Covers the PROPERTY half of the schema, "
                  "which ENVO handles only patchily.",
        ),
        Authority(
            key="go_bp",
            name="Gene Ontology (biological process)",
            url=f"https://release.geneontology.org/{GO_VERSION}/ontology/go-basic.json",
            version=GO_VERSION,
            fmt="obograph",
            group="ontologies",
            filename="go-basic.json",
            serves=("BIOTIC PROCESS",),
            licence="CC BY 4.0",
            size_hint="~110 MB",
            namespace_filter="biological_process",
            notes="go-basic is the release carrying only is_a/part_of, which is "
                  "what we want; the full GO adds relation types we do not use. "
                  "GO's PURL serves an unpinned 'current', so this points at the "
                  "dated release archive instead.",
        ),
        Authority(
            key="ecocore",
            name="Ecological Core Ontology",
            url=f"https://raw.githubusercontent.com/EcologicalSemantics/ecocore/{ECOCORE_VERSION}/ecocore.json",
            version=ECOCORE_VERSION,
            fmt="obograph",
            group="ontologies",
            filename="ecocore.json",
            serves=("BIOTIC ENTITY", "BIOTIC PROCESS", "BIOTIC PROPERTY"),
            licence="CC BY 4.0",
            size_hint="~10 MB",
            notes="Ecological roles and trophic vocabulary. Added because ENVO, "
                  "PATO and GO between them cover the physical environment well "
                  "and ecology barely at all: ecocore is where 'predator' lives "
                  "(11 occurrences in the gold), though it still does not carry "
                  "'biodiversity', 'species richness', 'diversity' or "
                  "'abundance' — see the coverage note in the plan.",
        ),
        Authority(
            key="gbif",
            name="GBIF Backbone Taxonomy",
            url="https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip",
            version="current",
            fmt="gbif_dwca",
            group="gbif",
            filename="gbif-backbone.zip",
            serves=("BIOTIC ENTITY",),
            licence="CC BY 4.0",
            size_hint=">1 GB — confirm before fetching",
            notes="Taxon.tsv plus VernacularName.tsv. The vernacular file is what "
                  "makes 'Bornean Colugo' and 'Galeopterus variegatus' land on "
                  "one URI, i.e. it IS the de-duplication, not a nicety. GBIF "
                  "publishes no dated backbone URL, so the sha256 of the "
                  "snapshot is the only version this one has.",
        ),
        Authority(
            key="geonames",
            name="GeoNames",
            url="https://download.geonames.org/export/dump/allCountries.zip",
            version="current",
            fmt="geonames_dump",
            group="geonames",
            filename="geonames-allCountries.zip",
            serves=("SPATIAL ENTITY",),
            licence="CC BY 4.0",
            size_hint=">1 GB uncompressed — confirm before fetching",
            notes="Filter to feature classes T (mountains), L (parks/areas), "
                  "H (water) and A (admin). Shipping all 12M rows would dwarf "
                  "everything else in the artifact for no gain on this corpus. "
                  "Like GBIF, versioned only by snapshot hash.",
        ),
    ]
}

# Small enough to fetch without asking; see the module docstring on ordering.
SMALL_KEYS = ("envo", "pato", "go_bp", "ecocore")

# group -> the authority keys compiled into it
GROUPS: Dict[str, tuple] = {
    "ontologies": ("envo", "pato", "go_bp", "ecocore"),
    "gbif": ("gbif",),
    "geonames": ("geonames",),
}
LARGE_KEYS = ("gbif", "geonames")


# ── hashing and versions ─────────────────────────────────────────

def file_sha256(path: Path, _chunk: int = 1 << 20) -> Optional[str]:
    """Hash a snapshot. Chunked because these files do not fit comfortably in RAM.

    Mirrors ``multi_agent_annotation_ag2._file_sha256`` in intent — tie a result
    to exact bytes rather than to a path that may be overwritten later.
    """
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(_chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception as exc:
        logger.warning("could not hash %s: %s", path, exc)
        return None


def payload_version(auth: Authority) -> Optional[str]:
    """The version the downloaded bytes claim for themselves.

    Read back out of the file rather than trusted from the registry, so a
    mis-pinned URL is caught: if ``Authority.version`` says v2026-06-26 and the
    payload says something else, the two disagree in the manifest instead of
    agreeing by assertion. Returns None when unavailable, which is recorded as
    None — an unknown version must be VISIBLE, not indistinguishable from a
    field nobody wrote.
    """
    if auth.fmt != "obograph":
        return None
    try:
        with auth.path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        graphs = doc.get("graphs") or []
        if not graphs:
            return None
        return ((graphs[0].get("meta") or {}).get("version")) or None
    except Exception as exc:
        logger.warning("could not read version from %s: %s", auth.path, exc)
        return None


# ── manifest ─────────────────────────────────────────────────────

@dataclass
class SnapshotRecord:
    key: str
    url: str
    filename: str
    pinned_version: str
    payload_version: Optional[str]
    fetched_at: str
    bytes: int
    sha256: Optional[str]
    licence: str


def record_snapshot(auth: Authority) -> SnapshotRecord:
    """Build the provenance record for an already-downloaded snapshot."""
    path = auth.path
    return SnapshotRecord(
        key=auth.key,
        url=auth.url,
        filename=auth.filename,
        pinned_version=auth.version,
        payload_version=payload_version(auth),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        bytes=path.stat().st_size if path.exists() else 0,
        sha256=file_sha256(path),
        licence=auth.licence,
    )


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("manifest at %s unreadable (%s); treating as empty", path, exc)
        return {}


def update_manifest(record: SnapshotRecord, path: Path = MANIFEST_PATH) -> None:
    """Merge one record in, leaving the other authorities' entries alone.

    Merge rather than overwrite so fetching ENVO alone does not erase the
    provenance of a GBIF snapshot that took an hour to download.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(path)
    manifest[record.key] = asdict(record)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def manifest_provenance(keys: Optional[List[str]] = None,
                        path: Path = MANIFEST_PATH) -> Dict[str, dict]:
    """The subset of the manifest to stamp into a run's ``run_meta``.

    Authorities missing from the manifest appear with nulls rather than being
    omitted: "we did not use ENVO" and "we used an ENVO we cannot name" must not
    look the same in a result file.
    """
    manifest = load_manifest(path)
    wanted = keys if keys is not None else list(AUTHORITIES)
    return {
        k: {
            "pinned_version": (manifest.get(k) or {}).get("pinned_version"),
            "payload_version": (manifest.get(k) or {}).get("payload_version"),
            "sha256": (manifest.get(k) or {}).get("sha256"),
        }
        for k in wanted
    }


# Marker written into ``source_serves`` for a universal authority.
UNIVERSAL_TYPE = "*"


def authorities_for(entity_type: str) -> List[Authority]:
    """Which authorities may answer for a canonicalised entity type.

    Callers pass a type already through ``label_norm.canonicalize_entity_type``.
    An unknown type returns [] — nothing is grounded, which is the safe outcome:
    a mention we cannot route is a NIL, not a guess.
    """
    et = (entity_type or "").strip().upper()
    return [a for a in AUTHORITIES.values() if a.universal or et in a.serves]
