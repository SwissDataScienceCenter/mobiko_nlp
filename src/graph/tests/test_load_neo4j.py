"""
The Neo4j loader syncs: it writes only what changed and deletes what is gone.

The ``plan`` tests need nothing. The sync tests run against the local server
(``make neo4j-up``, password in .env) on a throwaway graph, and are skipped
when it is not reachable.
"""
from __future__ import annotations

import copy
import os
import uuid

import pytest

from src.graph import load_neo4j as L

NODES = [{"id": "a", "kind": "Entity", "name": "forest", "entity_type": "SPATIAL ENTITY"},
         {"id": "b", "kind": "Instance", "name": "canopy · X", "entity_type": "BIOTIC ENTITY"},
         {"id": "c", "kind": "Entity", "name": "canopy", "entity_type": "BIOTIC ENTITY"}]
EDGES = [{"id": "e1", "source": "a", "target": "b", "relation": "HAS_PART", "property": "",
          "n_evidence": 1, "evidence": [{"sentence_id": "X#00/0", "relation_uid": "r1",
                                         "doc_id": "X#00"}]},
         {"id": "e2", "source": "b", "target": "c", "relation": "INSTANCE_OF", "n_evidence": 1}]


def test_plan_hash_follows_content_labels_and_endpoints():
    n1, e1 = L.plan("g", NODES, EDGES)
    n2, e2 = L.plan("g", copy.deepcopy(NODES), copy.deepcopy(EDGES))
    assert {k: v["props"]["sync_hash"] for k, v in n1.items()} == \
           {k: v["props"]["sync_hash"] for k, v in n2.items()}
    changed = copy.deepcopy(NODES)
    changed[0]["entity_type"] = "ABIOTIC ENTITY"          # a label change only
    n3, _ = L.plan("g", changed, EDGES)
    assert n3["g|a"]["props"]["sync_hash"] != n1["g|a"]["props"]["sync_hash"]
    moved = copy.deepcopy(EDGES)
    moved[0]["target"] = "c"                              # same id, new endpoint
    _, e3 = L.plan("g", NODES, moved)
    assert e3["g|e1"]["props"]["sync_hash"] != e1["g|e1"]["props"]["sync_hash"]
    assert e1["g|e1"]["props"]["sentence_ids"] == ["X#00/0"]
    assert "evidence" not in e1["g|e1"]["props"]


@pytest.fixture
def driver():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        pytest.skip("neo4j driver absent")
    L._load_dotenv()
    if not os.environ.get("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set")
    d = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                             auth=(os.environ.get("NEO4J_USER", "neo4j"),
                                   os.environ["NEO4J_PASSWORD"]))
    try:
        d.verify_connectivity()
    except Exception as exc:                              # server not running
        d.close()
        pytest.skip(f"Neo4j not reachable: {exc}")
    yield d
    d.close()


def _state(d, g):
    with d.session() as s:
        nodes = {r["k"]: (r["name"], sorted(r["l"])) for r in s.run(
            "MATCH (n:KGNode {graph:$g}) RETURN n.key AS k, n.name AS name, labels(n) AS l", g=g)}
        edges = sorted((r["a"], r["t"], r["b"], r["n"]) for r in s.run(
            "MATCH (a:KGNode {graph:$g})-[x]->(b) RETURN a.key AS a, type(x) AS t, "
            "b.key AS b, x.n_evidence AS n", g=g))
    return nodes, edges


def test_sync_writes_only_changes_and_deletes_what_is_gone(driver):
    g = f"zz_test_{uuid.uuid4().hex[:8]}"
    try:
        first = L.load(driver, g, NODES, EDGES, None)
        assert (first["nodes_created"], first["edges_created"]) == (3, 2)
        again = L.load(driver, g, NODES, EDGES, None)
        assert (again.get("nodes_created", 0), again.get("nodes_updated", 0),
                again.get("edges_created", 0), again.get("edges_updated", 0)) == (0, 0, 0, 0)

        nodes, edges = copy.deepcopy(NODES), copy.deepcopy(EDGES)
        nodes[0]["entity_type"] = "ABIOTIC ENTITY"        # label must follow
        edges[0]["n_evidence"] = 2                        # edge replaced, not doubled
        nodes = nodes[:2]                                 # c gone, and its edge with it
        edges = edges[:1]
        got = L.load(driver, g, nodes, edges, None)
        assert got["nodes_updated"] == 1 and got["edges_updated"] == 1
        assert got["nodes_deleted"] == 1 and got["edges_deleted"] == 1
        ns, es = _state(driver, g)
        assert ns[f"{g}|a"][1] == ["AbioticEntity", "Entity", "KGNode"]
        assert es == [(f"{g}|a", "HAS_PART", f"{g}|b", 2)]
        assert set(ns) == {f"{g}|a", f"{g}|b"}
    finally:
        with driver.session() as s:
            s.run("MATCH (n:KGNode {graph:$g}) DETACH DELETE n", g=g)
