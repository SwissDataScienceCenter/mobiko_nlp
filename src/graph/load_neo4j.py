"""
Load a built graph (``nodes.jsonl`` + ``edges.jsonl``) into Neo4j.

Start the database first (``make neo4j-up``); credentials come from the
environment or the repo's ``.env``: ``NEO4J_URI``, ``NEO4J_USER``,
``NEO4J_PASSWORD``.

Several graphs share one database
---------------------------------
Every node carries ``graph`` (default: the name of ``--in-dir``), and its
primary key is ``key = "<graph>|<id>"`` (edges carry ``key`` too). Loading
touches that graph and nothing else, so runs over different models or corpora
can sit side by side and be compared in one query.

Loading is a sync, not a reload: the loader reads the keys and content hashes
(``sync_hash``) already in Neo4j, writes only nodes and edges that are new or
changed (``MERGE`` in batches of 1,000), then deletes what is no longer in the
files. The old graph stays whole while the new one is written, re-loading an
unchanged graph writes nothing, and a load that fails half way is completed
by running it again. ``--full`` deletes the graph first and writes everything.

Schema in Neo4j
---------------
    (:KGNode:Entity:<TypeLabel>)       a concept, one per entity_id, e.g. :BioticEntity
    (:KGNode:Instance:<TypeLabel>)     that concept in one paper (and of one bearer)
    (:KGNode:Unresolved:<TypeLabel>)   an ambiguous mention used by a relation
    (:KGNode:AmbiguousPlace:<TypeLabel>) a place name with several referents
    (:KGNode:Measurement:<TypeLabel>)  a value: value|low,high, unit, comparator, raw
    (:KGNode:OntologyTerm)             one per authority URI
    (a)-[:<RELATION> {n_evidence, docs, sentence_ids, property, ...}]->(b)
    (a)-[:NOT_<RELATION> {...}]->(b)   a relation the sentence says does NOT hold
    (:Instance)-[:INSTANCE_OF]->(:Entity)
    (:Entity)-[:EXACT_MATCH]->(:OntologyTerm)
    (:AmbiguousPlace)-[:MAY_BE {feature, docs_naming_candidate}]->(:Entity)
    (:OntologyTerm)-[:WITHIN]->(:OntologyTerm)   GMBA range inside its parent range
        MATCH (r:OntologyTerm {name:'Eastern Arc Mountains'})<-[:WITHIN*0..]-(t)<-[:EXACT_MATCH]-(e)
        RETURN e.name                            -- every mountain in the Eastern Arc
        MATCH (p:AmbiguousPlace {name:'Kilimanjaro'})-[:MAY_BE]->(e)-[r]-(x)
        RETURN e.name, type(r), x.name   -- every Kilimanjaro and what is said of it

Relations join Instances, except where an end is a named place or taxon
(GeoNames, GBIF, WDPA): that Entity is shared by every paper and relations attach to
it directly. See src/graph/build_graph.py, "Concepts and instances".

Neo4j properties cannot hold maps, so the per-instance evidence is stored as
parallel lists (``relation_uids``, ``evidence_docs``) plus ``sentence_ids``;
the full records stay in ``edges.jsonl``.

No text is loaded: Neo4j is slow with long strings, and the same sentence
used to be copied onto every node and edge it touched. Each sentence is
stored once in ``sentences.sqlite`` next to ``nodes.jsonl``
(``src/graph/text_store.py``), with the mentions of every node and the
evidence of every edge:

    sqlite3 output/kg/second_iteration/sentences.sqlite \
      "SELECT text FROM sentence WHERE id = 'PMC6920850#00/37'"
    python -c "from src.graph.text_store import TextStore; \
      print(TextStore('output/kg/second_iteration/sentences.sqlite').contexts('<node id>'))"

Usage
-----
    python -m src.graph.load_neo4j --in-dir output/kg/second_iteration
    python -m src.graph.load_neo4j --in-dir output/kg/second_iteration --graph kimi_si
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from src.graph.build_graph import graph_relation_type

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH = 1000


def type_label(entity_type: str | None) -> str | None:
    """``BIOTIC ENTITY`` -> ``BioticEntity``: a node label, so the browser colours by type."""
    words = re.findall(r"[A-Za-z0-9]+", entity_type or "")
    label = "".join(w.capitalize() for w in words)
    if not label:
        return None
    return label if not label[0].isdigit() else f"T{label}"


def node_labels(node: dict) -> List[str]:
    labels = ["KGNode", node["kind"]]
    if node.get("value_concept"):
        # A shared quality ("high") that relations reach directly:
        # MATCH (x)-[r]->(q:Quality {name:'high'}) RETURN x.name, r.docs
        labels.append("Quality")
    t = type_label(node.get("entity_type"))
    if t and t != node["kind"]:
        labels.append(t)
    return labels


def node_props(node: dict, graph: str) -> dict:
    props = {k: v for k, v in node.items() if v is not None and k != "kind"}
    if props.get("candidates"):
        # Neo4j lists cannot hold maps: "geonames:157452 Kilimanjaro (T.MT)".
        props["candidates"] = [f"{c['curie']} {c['label']} ({c['feature']})"
                               for c in props["candidates"]]
    props.update({"graph": graph, "key": f"{graph}|{node['id']}"})
    return props


def edge_props(edge: dict) -> dict:
    props = {k: v for k, v in edge.items()
             if v is not None and k not in ("source", "target", "evidence")}
    ev = edge.get("evidence") or []
    if ev:
        # Ids, not text: the sentences are in sentences.sqlite (text_store).
        props["sentence_ids"] = sorted({x["sentence_id"] for x in ev})
        props["relation_uids"] = [x["relation_uid"] or "" for x in ev]
        props["evidence_docs"] = [x["doc_id"] for x in ev]
        # aligned with evidence_docs; "" where the section is unknown
        props["evidence_sections"] = [
            " / ".join(v for v in (x.get("section"), x.get("subsection")) if v)
            for x in ev]
    return props


def _load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$", line)
        if m and m.group(1).startswith("NEO4J_") and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip("\"'")


def _batches(rows: List[dict]):
    for i in range(0, len(rows), BATCH):
        yield rows[i:i + BATCH]


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                   default=str).encode("utf-8")).hexdigest()[:16]


def plan(graph: str, nodes: List[dict], edges: List[dict]) -> Tuple[dict, dict]:
    """What the graph should hold: key -> row, each with a content hash.

    The hash covers everything written (labels, type, endpoints, properties),
    so an unchanged row is recognised without comparing properties in Neo4j.
    """
    want_nodes = {}
    for node in nodes:
        props = node_props(node, graph)
        labels = node_labels(node)
        props["sync_hash"] = _digest([labels, props])
        want_nodes[props["key"]] = {"key": props["key"], "labels": labels, "props": props}
    want_edges = {}
    for e in edges:
        key = f"{graph}|{e['id']}"
        rtype = graph_relation_type(e["relation"], e.get("property") or "")
        s, o = f"{graph}|{e['source']}", f"{graph}|{e['target']}"
        props = {**edge_props(e), "key": key}
        props["sync_hash"] = _digest([rtype, s, o, props])
        want_edges[key] = {"key": key, "type": rtype, "s": s, "o": o, "props": props}
    return want_nodes, want_edges


# Nodes: one query for every label set. Labels a node no longer has (its kind
# or type changed) are removed, so a re-load never leaves a stale label.
_UPSERT_NODES = """
UNWIND $rows AS r
MERGE (n:KGNode {key: r.key})
SET n = r.props
SET n:$(r.labels)
WITH n, r
REMOVE n:$([l IN labels(n) WHERE NOT l IN r.labels])
"""
# Edges: a changed edge is replaced in the same transaction as its new
# version is created, so no reader sees it missing.
_WRITE_EDGES = """
UNWIND $rows AS r
OPTIONAL MATCH ()-[old]->() WHERE elementId(old) = r.old
DELETE old
WITH r
MATCH (a:KGNode {key: r.s})
MATCH (b:KGNode {key: r.o})
CREATE (a)-[x:$(r.type)]->(b)
SET x = r.props
"""
_DELETE_EDGES = """
UNWIND $ids AS id
MATCH ()-[x]->() WHERE elementId(x) = id
DELETE x
"""
_DELETE_NODES = """
UNWIND $keys AS k
MATCH (n:KGNode {key: k})
DETACH DELETE n
"""


def _write(session, cypher: str, **params) -> None:
    """One batch, one transaction, retried by the driver on transient errors."""
    session.execute_write(lambda tx: tx.run(cypher, **params).consume())


def load(driver, graph: str, nodes: List[dict], edges: List[dict], database: str | None,
         full: bool = False) -> Dict[str, int]:
    """Bring graph ``graph`` in Neo4j to exactly these nodes and edges.

    By default a sync: read the keys and hashes already there, write only
    what is new or changed, then delete what is gone. The old graph stays
    whole and queryable while the new one is written, and every batch is
    idempotent, so a load that fails half way is finished by running it again.
    Re-loading an unchanged graph writes nothing. ``full`` deletes the graph
    first and writes everything (the behaviour before 3 Nov 2026).
    """
    want_nodes, want_edges = plan(graph, nodes, edges)
    out = Counter()
    with driver.session(database=database) as s:
        s.run("CREATE CONSTRAINT kgnode_key IF NOT EXISTS "
              "FOR (n:KGNode) REQUIRE n.key IS UNIQUE")
        s.run("CREATE INDEX kgnode_graph IF NOT EXISTS FOR (n:KGNode) ON (n.graph)")
        # Range queries over parsed values ("elevations above 2000 m").
        s.run("CREATE INDEX measurement_value IF NOT EXISTS FOR (n:Measurement) ON (n.value)")
        if full:
            while True:
                n = s.execute_write(lambda tx: tx.run(
                    "MATCH (n:KGNode {graph: $g}) WITH n LIMIT 5000 "
                    "DETACH DELETE n RETURN count(*) AS n", g=graph).single()["n"])
                out["nodes_deleted"] += n
                if not n:
                    break

        have_nodes = {r["k"]: r["h"] for r in s.run(
            "MATCH (n:KGNode {graph: $g}) RETURN n.key AS k, n.sync_hash AS h", g=graph)}
        # An edge without a key (loaded before the sync) is replaced once.
        have_edges: Dict[str, Tuple[str, str]] = {}
        stale_edges: List[str] = []
        for r in s.run("MATCH (a:KGNode {graph: $g})-[x]->() "
                       "RETURN x.key AS k, x.sync_hash AS h, elementId(x) AS id", g=graph):
            if r["k"] in want_edges and r["k"] not in have_edges:
                have_edges[r["k"]] = (r["h"], r["id"])
            else:
                stale_edges.append(r["id"])

        node_rows = [w for k, w in want_nodes.items()
                     if have_nodes.get(k) != w["props"]["sync_hash"]]
        out["nodes_created"] = sum(1 for w in node_rows if w["key"] not in have_nodes)
        out["nodes_updated"] = len(node_rows) - out["nodes_created"]
        out["nodes_unchanged"] = len(want_nodes) - len(node_rows)
        for chunk in _batches(node_rows):
            _write(s, _UPSERT_NODES, rows=chunk)

        edge_rows = []
        for k, w in want_edges.items():
            h, eid = have_edges.get(k, (None, None))
            if h != w["props"]["sync_hash"]:
                edge_rows.append({**w, "old": eid})
        out["edges_created"] = sum(1 for w in edge_rows if w["old"] is None)
        out["edges_updated"] = len(edge_rows) - out["edges_created"]
        out["edges_unchanged"] = len(want_edges) - len(edge_rows)
        for chunk in _batches(edge_rows):
            _write(s, _WRITE_EDGES, rows=chunk)

        out["edges_deleted"] = len(stale_edges)
        for chunk in _batches(stale_edges):
            _write(s, _DELETE_EDGES, ids=chunk)
        gone = [k for k in have_nodes if k not in want_nodes]
        out["nodes_deleted"] += len(gone)
        for chunk in _batches(gone):
            _write(s, _DELETE_NODES, keys=chunk)

        counts = s.run("MATCH (n:KGNode {graph: $g}) "
                       "OPTIONAL MATCH (n)-[r]->() RETURN count(DISTINCT n) AS nodes, "
                       "count(r) AS edges", g=graph).single()
    out["nodes"], out["edges"] = counts["nodes"], counts["edges"]
    return dict(out)


def _read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="directory holding nodes.jsonl and edges.jsonl")
    ap.add_argument("--graph", default=None,
                    help="graph name stamped on every node (default: --in-dir's name)")
    ap.add_argument("--database", default=None,
                    help="Neo4j database (default: the server's default, `neo4j`)")
    ap.add_argument("--full", action="store_true",
                    help="delete the graph first and write everything, instead of "
                         "writing only what changed")
    args = ap.parse_args(argv)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("the `neo4j` driver is not installed: pip install 'neo4j>=5.20,<6'",
              file=sys.stderr)
        return 1
    _load_dotenv()
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        print("NEO4J_PASSWORD is not set (environment or .env)", file=sys.stderr)
        return 1

    graph = args.graph or args.in_dir.resolve().name
    nodes = _read_jsonl(args.in_dir / "nodes.jsonl")
    edges = _read_jsonl(args.in_dir / "edges.jsonl")
    ids = {n["id"] for n in nodes}
    dangling = sum(1 for e in edges if e["source"] not in ids or e["target"] not in ids)
    dup = len(nodes) - len(ids) + len(edges) - len({e["id"] for e in edges})
    if dup:
        print(f"{dup} duplicate node or edge id(s) — rebuild the graph", file=sys.stderr)
        return 1
    if dangling:
        print(f"{dangling} edge(s) reference a node not in nodes.jsonl — rebuild the "
              f"graph; refusing to load a partial one", file=sys.stderr)
        return 1

    # The first sync asks for `sync_hash` before any node has it; that is expected.
    with GraphDatabase.driver(uri, auth=(user, password),
                              notifications_disabled_categories=["UNRECOGNIZED"]) as driver:
        driver.verify_connectivity()
        got = load(driver, graph, nodes, edges, args.database, full=args.full)

    want_edges = len(edges)
    print(f"loaded graph '{graph}' into {uri}: {got['nodes']:,} nodes, {got['edges']:,} edges"
          + (" (full reload)" if args.full else ""))
    for what in ("nodes", "edges"):
        print(f"  {what}: {got.get(what + '_created', 0):,} created, "
              f"{got.get(what + '_updated', 0):,} updated, {got.get(what + '_deleted', 0):,} deleted, "
              f"{got.get(what + '_unchanged', 0):,} unchanged")
    if got["nodes"] != len(nodes) or got["edges"] != want_edges:
        print(f"  MISMATCH: files hold {len(nodes):,} nodes, {want_edges:,} edges",
              file=sys.stderr)
        return 1
    print(f"\n  browse: http://localhost:7474   (user {user})\n"
          f"  try:    MATCH p=(a:KGNode {{graph:'{graph}'}})-[r]->(b) "
          f"WHERE NOT type(r) IN ['INSTANCE_OF','EXACT_MATCH','MAY_BE','WITHIN'] AND r.n_evidence >= 2 "
          f"RETURN p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
