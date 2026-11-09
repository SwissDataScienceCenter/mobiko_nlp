"""
Tests for the graph builder.

The properties pinned here are what make the graph mean what it says:

  1. A triplet is a join on uid: two sentences stating the same relation
     between the same two entities are ONE edge with two pieces of evidence.
  2. An ambiguous mention keeps its triplet but gets a node of its own. Two
     ambiguous "the lake" mentions must not become one node — that is the
     identity claim the matcher refused to make.
  3. A relation on a coordination is distributed over its parts, and says so.
  4. Entities sharing an authority URI stay separate nodes, joined through one
     OntologyTerm — the identity model of UNIFICATION_RULES.md, not a merge.
  5. Output is byte-identical across runs.
  6. A relation attaches to the mention's referent, not to the shared concept:
     `elevation` in two papers is two instances of one concept, a named place
     is one node for every paper, and a property with one named bearer is
     scoped by it ("altitude of Udzungwa mountains").
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.graph.build_graph import (mention_key, EXACT_MATCH, INSTANCE_OF, MAY_BE, STRUCTURAL, build,
                                   ambiguous_place_key,
                                   curie, graph_relation_type, relation_type,
                                   write_triplets_tsv)
from src.graph.load_neo4j import node_labels, type_label
from src.graph.render_html import render
from src.graph.text_store import TextStore

ROOT = Path(__file__).resolve().parents[3]
ENVO_HABITAT = "http://purl.obolibrary.org/obo/ENVO_01000739"
PATO_ELEVATION = "http://purl.obolibrary.org/obo/PATO_0001687"
GEONAMES_KILI = "https://sws.geonames.org/157452/"


def _row(uid, text, etype="BIOTIC ENTITY", eid=None, uri=None, outcome="nil",
         occurrence=1, part=None, split_into=None, mention_id=None, file="ann.json",
         doc="D#00"):
    role = etype.split()[-1]
    return {
        "mention_id": mention_id or f"ann:{uid}#{occurrence}", "source_file": file,
        "uid": uid, "uid_occurrence": occurrence, "part_index": part,
        "split_into": split_into, "doc_id": doc, "text": text,
        "entity_type": etype, "entity_role": role, "entity_domain": etype.split()[0],
        "entity_id": eid if eid is not None else (
            None if outcome == "ambiguous" or split_into else f"mobiko:{text}"),
        "authority_uri": uri, "entity_label": text, "entity_source": "envo" if uri else "cluster",
        "outcome": outcome, "reason": "tied" if outcome == "ambiguous" else None,
        "n_candidates": 3 if outcome == "ambiguous" else None,
        "sent_index": 0, "start_char": None, "end_char": None, "split_from": None,
    }


def _rel(uid, e1, e2, label="HAS_PROPERTY", e1_type="BIOTIC ENTITY", e2_type="BIOTIC ENTITY"):
    return {"uid": uid, "relation": label, "e1_uid": e1, "e2_uid": e2,
            "e1": {"type": e1_type}, "e2": {"type": e2_type},
            "annotator": "agent:test", "agent": {"confidence": 0.8}}


def _setup(tmp_path, rows, sentences, name="ann.json", doc="D"):
    path = tmp_path / name
    path.write_text(json.dumps({"doc_id": doc, "sentences": sentences}), encoding="utf-8")
    table = {}
    for r in rows:
        table.setdefault((r["source_file"], r["uid"]), []).append(r)
    return path, table


def _semantic(edges):
    return [e for e in edges if e["relation"] not in STRUCTURAL]


def _names(nodes, edges):
    by_id = {n["id"]: n["name"] for n in nodes}
    return sorted((by_id[e["source"]], e["relation"], by_id[e["target"]])
                  for e in _semantic(edges))


# ── 1. same triplet in two sentences is one edge ─────────────────

def test_repeated_triplet_aggregates_into_one_edge(tmp_path):
    rows = [_row("a1", "forest"), _row("b1", "canopy"),
            _row("a2", "forests"), _row("b2", "canopy")]
    for r in rows:   # `forest` and `forests` are one entity, as resolve_corpus would say
        r["entity_id"] = f"mobiko:{r['text'].rstrip('s')}"
    path, table = _setup(tmp_path, rows, [
        {"text": "s1", "relations": [_rel("r1", "a1", "b1")]},
        {"text": "s2", "relations": [_rel("r2", "a2", "b2")]}])
    nodes, edges, stats = build([path], table, store=tmp_path / "s.sqlite")
    sem = _semantic(edges)
    assert len(sem) == 1
    assert sem[0]["n_evidence"] == 2
    assert [x["sentence_id"] for x in sem[0]["evidence"]] == ["D#00/0", "D#00/1"]
    with TextStore(tmp_path / "s.sqlite") as st:
        assert [x["sentence"] for x in st.evidence(sem[0]["id"])] == ["s1", "s2"]
    assert stats["triplet_instances"] == 2


# ── 2. ambiguous mentions keep their triplet, not a shared identity ─

def test_ambiguous_mentions_get_one_node_each(tmp_path):
    rows = [_row("l1", "the lake", "SPATIAL ENTITY", outcome="ambiguous"),
            _row("l2", "the lake", "SPATIAL ENTITY", outcome="ambiguous"),
            _row("t1", "trout")]
    path, table = _setup(tmp_path, rows, [
        {"text": "s1", "relations": [_rel("r1", "t1", "l1", "LOCATED_IN", e2_type="SPATIAL ENTITY")]},
        {"text": "s2", "relations": [_rel("r2", "t1", "l2", "LOCATED_IN", e2_type="SPATIAL ENTITY")]}])
    nodes, edges, stats = build([path], table)
    unresolved = [n for n in nodes if n["kind"] == "Unresolved"]
    assert len(unresolved) == 2
    assert len(_semantic(edges)) == 2
    assert stats["endpoint_ambiguous"] == 2


# ── 3. coordinations distribute, visibly ─────────────────────────

def test_relation_on_coordination_is_distributed_over_parts(tmp_path):
    rows = [_row("c1", "eastern and western slopes", "SPATIAL ENTITY", split_into=2),
            _row("c1", "eastern slopes", "SPATIAL ENTITY", part=1, mention_id="ann:c1#1.1"),
            _row("c1", "western slopes", "SPATIAL ENTITY", part=2, mention_id="ann:c1#1.2"),
            _row("p1", "steep", "SPATIAL PROPERTY")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r1", "c1", "p1", e1_type="SPATIAL ENTITY", e2_type="SPATIAL PROPERTY")]}])
    nodes, edges, stats = build([path], table, scope="concept")
    sem = _semantic(edges)
    assert sorted(e["source"] for e in sem) == ["mobiko:eastern slopes",
                                                "mobiko:western slopes"]
    assert all(e["distributed"] for e in sem)
    assert not any(n["id"] == "mobiko:eastern and western slopes" for n in nodes)


# ── 4. one ontology term, several entities ───────────────────────

def test_shared_authority_joins_through_a_term_without_merging(tmp_path):
    rows = [_row("h1", "habitat", "ABIOTIC ENTITY", eid=f"ABIOTIC ENTITY@{ENVO_HABITAT}",
                 uri=ENVO_HABITAT, outcome="linked"),
            _row("h2", "habitat", "BIOTIC ENTITY", eid=f"BIOTIC ENTITY@{ENVO_HABITAT}",
                 uri=ENVO_HABITAT, outcome="linked")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": []}])
    nodes, edges, _ = build([path], table)
    entities = [n for n in nodes if n["kind"] == "Entity"]
    terms = [n for n in nodes if n["kind"] == "OntologyTerm"]
    assert len(entities) == 2 and len(terms) == 1
    assert terms[0]["curie"] == "ENVO:01000739" and terms[0]["n_entities"] == 2
    assert sorted(e["source"] for e in edges if e["relation"] == EXACT_MATCH) == \
        sorted(n["id"] for n in entities)


# ── 6. referents: papers apart, places shared, bearers split ─────

def _elevation(uid, doc, **kw):
    return _row(uid, "elevation", "SPATIAL PROPERTY", eid=f"SPATIAL PROPERTY@{PATO_ELEVATION}",
                uri=PATO_ELEVATION, outcome="linked", doc=doc, **kw)


def _place(uid, text, doc, uri=None, **kw):
    if uri:
        return _row(uid, text, "SPATIAL ENTITY", eid=f"SPATIAL ENTITY@{uri}", uri=uri,
                    outcome="linked", doc=doc, **kw)
    return _row(uid, text, "SPATIAL ENTITY", doc=doc, **kw)


def test_one_concept_in_two_papers_is_two_instances(tmp_path):
    rows = [_elevation("x1", "X#00"), _row("x2", "species richness", "BIOTIC PROPERTY", doc="X#00"),
            _elevation("y1", "Y#00"), _row("y2", "2743 m", "QUANTITATIVE PROPERTY", doc="Y#00")]
    path, table = _setup(tmp_path, rows, [
        {"text": "x", "relations": [_rel("r1", "x2", "x1", "VARIES_WITH",
                                         "BIOTIC PROPERTY", "SPATIAL PROPERTY")]},
        {"text": "y", "relations": [_rel("r2", "y1", "y2", "HAS_VALUE",
                                         "SPATIAL PROPERTY", "QUANTITATIVE PROPERTY")]}])
    nodes, edges, _ = build([path], table)
    by_id = {n["id"]: n for n in nodes}
    elevations = [n for n in nodes if n["kind"] == "Instance" and n["base_name"] == "elevation"]
    assert sorted(n["name"] for n in elevations) == ["elevation · X", "elevation · Y"]
    # No path joins paper X's correlation to paper Y's value...
    for e in _semantic(edges):
        assert by_id[e["source"]]["doc_id"] == by_id[e["target"]]["doc_id"]
    # ...but both are explicitly instances of the one concept.
    concept = f"SPATIAL PROPERTY@{PATO_ELEVATION}"
    assert sorted(e["source"] for e in edges if e["relation"] == INSTANCE_OF
                  and e["target"] == concept) == sorted(n["id"] for n in elevations)


def test_a_named_place_is_one_node_for_every_paper(tmp_path):
    rows = [_place("x1", "Mount Kilimanjaro", "X#00", GEONAMES_KILI), _elevation("x2", "X#00"),
            _place("y1", "Kilimanjaro", "Y#00", GEONAMES_KILI), _elevation("y2", "Y#00")]
    path, table = _setup(tmp_path, rows, [
        {"text": "x", "relations": [_rel("r1", "x1", "x2", "HAS_PROPERTY", "SPATIAL ENTITY",
                                         "SPATIAL PROPERTY")]},
        {"text": "y", "relations": [_rel("r2", "y1", "y2", "HAS_PROPERTY", "SPATIAL ENTITY",
                                         "SPATIAL PROPERTY")]}])
    nodes, edges, _ = build([path], table)
    assert {e["source"] for e in _semantic(edges)} == {f"SPATIAL ENTITY@{GEONAMES_KILI}"}
    # The mountain is shared, its elevation is still measured per paper.
    assert len({e["target"] for e in _semantic(edges)}) == 2
    assert not any(n["kind"] == "Instance" and n["entity_id"].endswith(GEONAMES_KILI)
                   for n in nodes)


def test_one_bearer_scopes_a_property_within_a_paper(tmp_path):
    rows = [_place("a", "Udzungwa mountains", "X#00"), _row("p1", "altitude", "SPATIAL PROPERTY", doc="X#00"),
            _row("v1", "1800 m", "QUANTITATIVE PROPERTY", doc="X#00"),
            _place("b", "cloud forests", "X#00"), _row("p2", "altitude", "SPATIAL PROPERTY", doc="X#00"),
            _row("v2", "2600 m", "QUANTITATIVE PROPERTY", doc="X#00")]
    path, table = _setup(tmp_path, rows, [
        {"text": "s1", "relations": [
            _rel("r1", "a", "p1", "HAS_PROPERTY", "SPATIAL ENTITY", "SPATIAL PROPERTY"),
            _rel("r2", "p1", "v1", "HAS_VALUE", "SPATIAL PROPERTY", "QUANTITATIVE PROPERTY")]},
        {"text": "s2", "relations": [
            _rel("r3", "b", "p2", "HAS_PROPERTY", "SPATIAL ENTITY", "SPATIAL PROPERTY"),
            _rel("r4", "p2", "v2", "HAS_VALUE", "SPATIAL PROPERTY", "QUANTITATIVE PROPERTY")]}])
    nodes, edges, stats = build([path], table)
    assert ("altitude of Udzungwa mountains · X", "HAS_VALUE", "1800 m · X") in _names(nodes, edges)
    assert ("altitude of cloud forests · X", "HAS_VALUE", "2600 m · X") in _names(nodes, edges)
    assert stats["nodes_instance_bearer_scoped"] == 2
    # Without bearer scoping the two altitudes, and so both values, are one node.
    nodes, edges, _ = build([path], table, scope="document")
    assert ("altitude · X", "HAS_VALUE", "1800 m · X") in _names(nodes, edges)
    assert ("altitude · X", "HAS_VALUE", "2600 m · X") in _names(nodes, edges)


def test_several_bearers_stay_at_document_scope(tmp_path):
    """One implied `elevation` for two sites, as PMC6920850 sentence 77 has it."""
    rows = [_place("a", "Monduli forest reserve", "X#00"), _place("b", "Marangu route", "X#00"),
            _elevation("e", "X#00")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r1", "a", "e", "HAS_PROPERTY", "SPATIAL ENTITY", "SPATIAL PROPERTY"),
        _rel("r2", "b", "e", "HAS_PROPERTY", "SPATIAL ENTITY", "SPATIAL PROPERTY")]}])
    nodes, edges, stats = build([path], table)
    assert stats["mentions_with_several_bearers"] == 1
    assert {o for _, _, o in _names(nodes, edges)} == {"elevation · X"}


def test_bearer_relations_are_configurable(tmp_path):
    rows = [_place("a", "lake", "X#00"), _row("p", "temperature", "ABIOTIC PROPERTY", doc="X#00")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r1", "a", "p", "CHARACTERISED_BY", "SPATIAL ENTITY", "ABIOTIC PROPERTY")]}])
    nodes, edges, _ = build([path], table)
    assert _names(nodes, edges) == [("lake · X", "CHARACTERISED_BY", "temperature · X")]
    nodes, edges, _ = build([path], table, bearer_relations=frozenset({"characterised by"}))
    assert _names(nodes, edges) == [("lake · X", "CHARACTERISED_BY", "temperature of lake · X")]


def test_a_qualitative_value_is_named_with_its_bearer_in_brackets(tmp_path):
    rows = [_row("a", "increase", "BIOTIC PROCESS", doc="X#00"),
            _row("q", "significant", "QUALITATIVE PROPERTY", doc="X#00")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r1", "a", "q", "HAS_PROPERTY", "BIOTIC PROCESS", "QUALITATIVE PROPERTY")]}])
    nodes, edges, _ = build([path], table, shared_qualities=False)  # the rule under test; shared qualities would bypass it
    assert _names(nodes, edges) == [("increase · X", "HAS_PROPERTY", "significant (increase) · X")]


# ── sentences on nodes ───────────────────────────────────────────

def test_nodes_carry_their_sentences_with_the_mention_marked(tmp_path):
    rows = [_row("a", "lake", "SPATIAL ENTITY", doc="X#00"),
            _row("v", "elevation", "SPATIAL PROPERTY", doc="X#00")]
    rows[0].update(start_char=4, end_char=8)   # rows[1] has no offsets: implied
    path, table = _setup(tmp_path, rows, [{"text": "The lake lies at 2000 m.",
        "relations": [_rel("r1", "a", "v", "HAS_PROPERTY", "SPATIAL ENTITY",
                           "SPATIAL PROPERTY")]}], doc="X")
    nodes, edges, _ = build([path], table, store=tmp_path / "s.sqlite")
    by_name = {n["name"]: n for n in nodes}
    # No text on the nodes: it is in the store, once.
    assert not any("sentences" in n for n in nodes)
    assert by_name["lake"]["n_sentences"] == 1
    with TextStore(tmp_path / "s.sqlite") as st:
        ctx = lambda name: st.contexts(by_name[name]["id"],
                                       concept=by_name[name]["kind"] == "Entity")
        assert ctx("lake · X") == ["X s0: The «lake» lies at 2000 m."]
        assert ctx("lake") == ctx("lake · X")
        assert ctx("elevation of lake · X") == [
            "X s0: The lake lies at 2000 m.  [implied: elevation]"]
        assert st.search("lake")[0]["id"] == "X#00/0"
        assert st.con.execute("SELECT count(*) FROM sentence").fetchone()[0] == 1


def test_html_stores_each_sentence_once(tmp_path):
    from src.graph.render_html import _compact
    rows = [_row("a", "lake", "SPATIAL ENTITY", doc="X#00"),
            _row("b", "shore", "SPATIAL ENTITY", doc="X#00")]
    rows[0].update(start_char=4, end_char=8)
    rows[1].update(start_char=13, end_char=18)
    path, table = _setup(tmp_path, rows, [{"text": "The lake has shore birds.",
        "relations": [_rel("r1", "a", "b", "PART_OF", "SPATIAL ENTITY", "SPATIAL ENTITY")]}],
        doc="X")
    nodes, edges, _ = build([path], table, store=tmp_path / "s.sqlite")
    with TextStore(tmp_path / "s.sqlite") as st:
        data = _compact(nodes, edges, st)
    # Three nodes per mention side and one edge use one sentence, embedded once.
    assert data["sentences"] == [["X s0", "The lake has shore birds."]]
    assert sorted(data["mentions"]) == [[0, 4, 8, ""], [0, 13, 18, ""]]
    assert all(n.get("s") for n in data["nodes"] if n["kind"] in ("Entity", "Instance"))
    assert [e["s"] for e in data["edges"] if e["relation"] == "PART_OF"] == [[0]]


# ── coreference: an anaphor takes its antecedent's referent ─────

S0 = "We sampled the western slope of Mt Kenya."
S1 = "The mountain has high rainfall."


def _coref_case(tmp_path, link, anaphor=("a", "mountain", "ABIOTIC ENTITY", 4, 12),
                **build_kw):
    """S0 holds the antecedents, S1 an anaphor with a relation on it."""
    uid, text, etype, s0, s1 = anaphor
    rows = [_row("k", "Mt Kenya", "SPATIAL ENTITY", doc="X#00"),
            _row("w", "western slope", "SPATIAL PROPERTY", doc="X#00"),
            _row(uid, text, etype, doc="X#00"),
            _row("r", "high rainfall", "ABIOTIC PROPERTY", doc="X#00")]
    rows[0].update(start_char=32, end_char=40)
    rows[1].update(start_char=15, end_char=28)
    rows[2].update(sent_index=1, start_char=s0, end_char=s1)
    rows[3].update(sent_index=1, start_char=17, end_char=30)
    path, table = _setup(tmp_path, rows, [
        {"text": S0, "relations": []},
        {"text": S1, "coref": [link], "relations": [
            _rel("r1", uid, "r", "HAS_PROPERTY", etype, "ABIOTIC PROPERTY")]}], doc="X")
    return build([path], table, **build_kw)


def _link(antecedent, text="The mountain", start=0, end=12, confidence="high"):
    return {"start_char": start, "end_char": end, "text": text, "antecedent": antecedent,
            "kind": "definite_np", "confidence": confidence, "antecedent_offset": -1}


def test_anaphor_attaches_to_its_antecedent(tmp_path):
    nodes, edges, stats = _coref_case(tmp_path, _link("Mt Kenya"), store=tmp_path / "s.sqlite")
    assert _names(nodes, edges) == [
        ("Mt Kenya · X", "HAS_PROPERTY", "high rainfall of Mt Kenya · X")]
    assert _semantic(edges)[0]["evidence"][0]["coref"] == ["mountain → Mt Kenya"]
    kenya = next(n for n in nodes if n["name"] == "Mt Kenya · X")
    with TextStore(tmp_path / "s.sqlite") as st:
        assert any("«mountain»" in x and "refers to «Mt Kenya»" in x
                   for x in st.contexts(kenya["id"]))
    assert stats["coref_accepted"] == 1


def test_antecedent_found_by_its_head_before_a_preposition(tmp_path):
    nodes, edges, _ = _coref_case(
        tmp_path, _link("the western slope of Mt Kenya", "The slope", 0, 9),
        anaphor=("a", "slope", "SPATIAL PROPERTY", 4, 9))
    assert _names(nodes, edges)[0][0] == "western slope · X"


@pytest.mark.parametrize("link,anaphor,reason", [
    (_link("Mt Kenya", "it", 0, 2), None, "anaphor_not_a_span"),
    (_link("Mt Kenya", confidence="medium"), None, "below_min_confidence"),
    (_link("the western slope of Mt Kenya"), None, "role_mismatch"),
    (_link("Mt Kenya and Kilimanjaro"), None, "coordinated_antecedent"),
    (_link("two mountains"), None, "counted_set_antecedent"),
    (_link("Mt Kenya", "other mountain", 4, 18), ("a", "other mountain", "ABIOTIC ENTITY", 4, 18),
     "contrastive_anaphor"),
])
def test_unsafe_coreference_is_rejected_and_counted(tmp_path, link, anaphor, reason):
    kw = {"anaphor": anaphor} if anaphor else {}
    nodes, edges, stats = _coref_case(tmp_path, link, **kw)
    assert stats[f"coref_rejected_{reason}"] == 1
    assert not any(x["coref"] for e in _semantic(edges) for x in e["evidence"])


def test_a_pronoun_reaches_its_antecedent_whatever_its_type(tmp_path):
    """They had direct effects on pollination: "They" is only an anchor."""
    nodes, edges, stats = _coref_case(
        tmp_path, _link("Mt Kenya", "They", 0, 4, ),
        anaphor=("a", "They", "ABIOTIC PROPERTY", 0, 4))
    assert stats["coref_accepted"] == 1
    assert _names(nodes, edges)[0][0] == "Mt Kenya · X"
    assert not any(n["kind"] == "Entity" and n["name"] == "They" for n in nodes)


def test_an_unresolved_pronoun_is_never_a_concept(tmp_path):
    nodes, edges, _ = _coref_case(tmp_path, _link("Mt Kenya", "They", 0, 4, confidence="low"),
                                  anaphor=("a", "They", "ABIOTIC ENTITY", 0, 4))
    assert not any(n["kind"] == "Entity" and n["name"] == "They" for n in nodes)
    they = [n for n in nodes if n["kind"] == "Unresolved" and n["name"] == "They"]
    assert len(they) == 1 and "pronoun" in they[0]["reason"]


def test_coreference_can_be_turned_off_and_never_touches_concepts(tmp_path):
    for kw in ({"coref": False}, {"scope": "concept"}):
        _, edges, stats = _coref_case(tmp_path, _link("Mt Kenya"), **kw)
        assert not stats["coref_accepted"]
        assert not any(x["coref"] for e in _semantic(edges) for x in e["evidence"])


# ── relation properties ──────────────────────────────────────────

def _prop_rel(uid, e1, e2, label, prop, e1_type="SPATIAL PROPERTY", e2_type="BIOTIC PROPERTY"):
    return {**_rel(uid, e1, e2, label, e1_type, e2_type), "property": prop}


def test_a_negated_statement_is_its_own_edge_and_its_own_type(tmp_path):
    rows = [_row("a", "elevation", "SPATIAL PROPERTY"), _row("b", "richness", "BIOTIC PROPERTY"),
            _row("c", "elevation", "SPATIAL PROPERTY"), _row("d", "richness", "BIOTIC PROPERTY")]
    for r in rows:
        r["entity_id"] = f"mobiko:{r['text']}"
    path, table = _setup(tmp_path, rows, [
        {"text": "s1", "relations": [_prop_rel("r1", "a", "b", "AFFECTS", "")]},
        {"text": "s2", "relations": [_prop_rel("r2", "c", "d", "AFFECTS", "negation")]}])
    _, edges, stats = build([path], table)
    sem = sorted(_semantic(edges), key=lambda e: e["property"])
    assert [(e["property"], e["graph_type"], e["n_evidence"]) for e in sem] == [
        ("", "AFFECTS", 1), ("Negation", "NOT_AFFECTS", 1)]
    assert stats["edges_property_Negation"] == 1


def test_other_properties_keep_the_relation_type():
    assert graph_relation_type("affects", "Negative_polarity") == "AFFECTS"
    assert graph_relation_type("AFFECTS", "Dominance") == "AFFECTS"
    assert graph_relation_type("AFFECTS", "Negation") == "NOT_AFFECTS"
    assert graph_relation_type("AFFECTS") == "AFFECTS"


def test_a_negated_bearer_relation_scopes_nothing(tmp_path):
    rows = [_row("a", "species richness", "BIOTIC PROPERTY", doc="X#00"),
            _row("q", "heightened", "QUALITATIVE PROPERTY", doc="X#00")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _prop_rel("r1", "a", "q", "HAS_PROPERTY", "Negation",
                  "BIOTIC PROPERTY", "QUALITATIVE PROPERTY")]}])
    nodes, edges, stats = build([path], table, shared_qualities=False)  # the rule under test; shared qualities would bypass it
    assert _names(nodes, edges) == [("species richness · X", "HAS_PROPERTY", "heightened · X")]
    assert stats["bearer_relations_negated_ignored"] == 1


def test_triplets_tsv_shows_the_property(tmp_path):
    rows = [_row("a", "elevation", "SPATIAL PROPERTY"), _row("b", "richness", "BIOTIC PROPERTY")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _prop_rel("r1", "a", "b", "AFFECTS", "Negation")]}])
    nodes, edges, _ = build([path], table)
    out = tmp_path / "t.tsv"
    write_triplets_tsv(out, nodes, edges)
    header, row = out.read_text().splitlines()[:2]
    assert dict(zip(header.split("\t"), row.split("\t")))["property"] == "Negation"


# ── shared qualities (decided 3 Nov 2026 with Luis) ─────────────

def _quality_case(tmp_path, **kw):
    """Two papers each say something is "high"; in X, "high" is also a subject."""
    rows = [_row("a", "correlation values", "QUANTITATIVE PROPERTY", doc="X#00"),
            _row("q1", "high", "QUALITATIVE PROPERTY", eid="mobiko:high", doc="X#00"),
            _row("b", "elevation", "SPATIAL PROPERTY", doc="Y#00"),
            _row("q2", "high", "QUALITATIVE PROPERTY", eid="mobiko:high", doc="Y#00"),
            _row("q3", "high", "QUALITATIVE PROPERTY", eid="mobiko:high", doc="X#00"),
            _row("c", "conservation return", "ANTHROPOGENIC PROPERTY", doc="X#00")]
    for r, si in zip(rows, (0, 0, 1, 1, 2, 2)):
        r["sent_index"] = si
    path, table = _setup(tmp_path, rows, [
        {"text": "s0", "relations": [_rel("r1", "a", "q1", "HAS_PROPERTY",
                                          "QUANTITATIVE PROPERTY", "QUALITATIVE PROPERTY")]},
        {"text": "s1", "relations": [_rel("r2", "b", "q2", "HAS_PROPERTY",
                                          "SPATIAL PROPERTY", "QUALITATIVE PROPERTY")]},
        {"text": "s2", "relations": [_rel("r3", "q3", "c", "ASSOCIATED_WITH",
                                          "QUALITATIVE PROPERTY", "ANTHROPOGENIC PROPERTY")]}])
    return build([path], table, **kw)


def test_qualities_point_straight_at_the_shared_node(tmp_path):
    nodes, edges, stats = _quality_case(tmp_path)
    into_high = [e for e in _semantic(edges) if e["target"] == "mobiko:high"]
    assert len(into_high) == 2
    assert sorted(d for e in into_high for d in e["docs"]) == ["X#00", "Y#00"]
    high = next(n for n in nodes if n["id"] == "mobiko:high")
    assert high["value_concept"] is True
    assert stats["qualities_to_shared_node"] == 2


def test_a_quality_that_is_a_subject_keeps_its_per_paper_node(tmp_path):
    nodes, edges, _ = _quality_case(tmp_path)
    subject = [n for n in nodes if n["kind"] == "Instance" and n.get("base_name") == "high"]
    assert len(subject) == 1 and subject[0]["doc_id"] == "X#00"
    # ... so nothing but structure leaves the shared node: it bridges no papers
    assert not [e for e in _semantic(edges) if e["source"] == "mobiko:high"]


def test_shared_qualities_can_be_turned_off(tmp_path):
    nodes, edges, _ = _quality_case(tmp_path, shared_qualities=False)
    assert not [e for e in _semantic(edges) if e["target"] == "mobiko:high"]
    assert len([n for n in nodes if n["kind"] == "Instance" and n.get("base_name") == "high"]) == 3


def test_a_shared_quality_gets_its_neo4j_label():
    assert "Quality" in node_labels({"kind": "Entity", "entity_type": "QUALITATIVE PROPERTY",
                                     "value_concept": True})


# ── reused uids, name clashes, endpoint not in the table ─────────

def test_reused_uid_picks_the_occurrence_whose_type_the_relation_names(tmp_path):
    rows = [_row("u1", "species", "BIOTIC ENTITY", eid="mobiko:species-be"),
            _row("u1", "species", "TEMPORAL PROPERTY", eid="mobiko:species-tp", occurrence=2),
            _row("x1", "rich")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r1", "u1", "x1", e1_type="TEMPORAL_PROPERTY")]}])
    _, edges, _ = build([path], table, scope="concept")
    assert _semantic(edges)[0]["source"] == "mobiko:species-tp"


def test_two_annotation_files_with_one_name_are_refused(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1, table = _setup(tmp_path / "a", [], [])
    p2, _ = _setup(tmp_path / "b", [], [])
    with pytest.raises(ValueError, match="share a name"):
        build([p1, p2], table)


def test_relation_with_unknown_endpoint_is_dropped_and_counted(tmp_path):
    path, table = _setup(tmp_path, [_row("a1", "forest")], [
        {"text": "s", "relations": [_rel("r1", "a1", "nope")]}])
    _, edges, stats = build([path], table)
    assert not _semantic(edges)
    assert stats["relations_dropped_missing_endpoint"] == 1


# ── small pure helpers ───────────────────────────────────────────

@pytest.mark.parametrize("uri,want", [
    (ENVO_HABITAT, "ENVO:01000739"),
    ("http://purl.obolibrary.org/obo/PATO_0001687", "PATO:0001687"),
    ("https://www.gbif.org/species/1457", "gbif:1457"),
    ("https://sws.geonames.org/149590/", "geonames:149590"),
    ("urn:x:unknown", "urn:x:unknown"),
    (None, None),
])
def test_curie(uri, want):
    assert curie(uri) == want


@pytest.mark.parametrize("label,want", [
    ("HAS_PROPERTY", "HAS_PROPERTY"), ("located in", "LOCATED_IN"),
    ("part-of", "PART_OF"), ("", "RELATED_TO"), ("3D_SHAPE", "R_3D_SHAPE"),
    ("X`) DETACH DELETE n //", "X_DETACH_DELETE_N"),
])
def test_relation_type_is_safe_to_splice_into_cypher(label, want):
    assert relation_type(label) == want


def test_node_labels_colour_by_type():
    assert type_label("BIOTIC ENTITY") == "BioticEntity"
    assert node_labels({"kind": "Entity", "entity_type": "CONCEPT"}) == \
        ["KGNode", "Entity", "Concept"]
    assert node_labels({"kind": "OntologyTerm"}) == ["KGNode", "OntologyTerm"]


def test_html_cannot_be_broken_out_of_by_a_sentence():
    page = render([{"id": "a", "kind": "Entity", "name": "x"}],
                  [{"source": "a", "target": "a", "relation": "R", "n_evidence": 1,
                    "docs": [], "evidence": [{"sentence": "</script><b>"}]}], "t")
    assert "</script><b>" not in page


# ── 5. determinism, across processes ─────────────────────────────

def test_cli_output_is_byte_identical_across_runs(tmp_path):
    rows = [_row("a1", "forest"), _row("b1", "canopy"), _row("c1", "lake",
            "SPATIAL ENTITY", outcome="ambiguous")]
    path, _ = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r1", "a1", "b1"), _rel("r2", "a1", "c1", "LOCATED_IN",
                                     e2_type="SPATIAL ENTITY")]}])
    res = tmp_path / "resolution.jsonl"
    res.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    outs = []
    for i, seed in enumerate(("0", "1")):
        out = tmp_path / f"out{i}"
        subprocess.run([sys.executable, "-m", "src.graph.build_graph",
                        "--annotations", str(path), "--resolution", str(res),
                        "--out-dir", str(out)],
                       cwd=ROOT, check=True, capture_output=True,
                       env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"})
        outs.append([(out / f).read_bytes() for f in
                     ("nodes.jsonl", "edges.jsonl", "triplets.tsv")])
    assert outs[0] == outs[1]


# ── ambiguous place names become one meta-node (decided 3 Nov 2026) ─

_MT = "https://sws.geonames.org/157452/"
_REGION = "https://sws.geonames.org/157449/"
_KILI = [{"uri": _MT, "curie": "geonames:157452", "label": "Kilimanjaro",
          "source": "geonames", "scope": "label", "rank": "T.MT"},
         {"uri": _REGION, "curie": "geonames:157449", "label": "Kilimanjaro Region",
          "source": "geonames", "scope": "label", "rank": "A.ADM1"}]


def _ambiguous(uid, text, cands, doc="D#00", file="ann.json"):
    r = _row(uid, text, "SPATIAL ENTITY", outcome="ambiguous", doc=doc, file=file,
             mention_id=f"{file}:{uid}#1")
    r["candidates"] = cands
    return r


def _two_papers_say_kilimanjaro(tmp_path, **kw):
    rows_a = [_ambiguous("k", "Kilimanjaro", _KILI, doc="A#00", file="a.json"),
              _row("f", "forest", "BIOTIC ENTITY", file="a.json", doc="A#00",
                   mention_id="a.json:f#1"),
              _row("m", "Mount Kilimanjaro", "SPATIAL ENTITY", eid=f"SPATIAL ENTITY@{_MT}",
                   uri=_MT, outcome="linked", file="a.json", doc="A#00")]
    rows_b = [_ambiguous("k", "Kilimanjaro", _KILI, doc="B#00", file="b.json"),
              _row("c", "coffee", "BIOTIC ENTITY", file="b.json", doc="B#00",
                   mention_id="b.json:c#1")]
    for r in rows_a:
        r["entity_source"] = "geonames" if r["authority_uri"] else r["entity_source"]
    pa, ta = _setup(tmp_path, rows_a, [{"text": "Forest on Kilimanjaro.", "relations": [
        _rel("r1", "f", "k", "LOCATED_IN", "BIOTIC ENTITY", "SPATIAL ENTITY")]}],
        name="a.json", doc="A")
    pb, tb = _setup(tmp_path, rows_b, [{"text": "Coffee in Kilimanjaro.", "relations": [
        _rel("r2", "c", "k", "LOCATED_IN", "BIOTIC ENTITY", "SPATIAL ENTITY")]}],
        name="b.json", doc="B")
    return build([pa, pb], {**ta, **tb}, **kw)


def test_ambiguous_place_name_is_one_shared_meta_node(tmp_path):
    nodes, edges, stats = _two_papers_say_kilimanjaro(tmp_path)
    places = [n for n in nodes if n["kind"] == "AmbiguousPlace"]
    assert [p["id"] for p in places] == ["place:kilimanjaro"]
    assert places[0]["docs"] == ["A#00", "B#00"] and places[0]["n_mentions"] == 2
    assert not [n for n in nodes if n["kind"] == "Unresolved"]
    # both papers' relations reach the same node
    names = {n["id"]: n.get("base_name") for n in nodes}
    assert sorted(names[e["source"]] for e in _semantic(edges)
                  if e["target"] == "place:kilimanjaro") == ["coffee", "forest"]


def test_meta_node_may_be_each_candidate(tmp_path):
    nodes, edges, _ = _two_papers_say_kilimanjaro(tmp_path)
    by_id = {n["id"]: n for n in nodes}
    may = {e["target"]: e for e in edges if e["relation"] == "MAY_BE"}
    assert set(may) == {f"SPATIAL ENTITY@{_MT}", f"SPATIAL ENTITY@{_REGION}"}
    # the mountain is linked by paper A; the region exists only as a candidate
    assert not by_id[f"SPATIAL ENTITY@{_MT}"].get("candidate_only")
    assert by_id[f"SPATIAL ENTITY@{_REGION}"]["candidate_only"]
    assert by_id[f"SPATIAL ENTITY@{_REGION}"]["name"] == "Kilimanjaro Region"
    assert may[f"SPATIAL ENTITY@{_MT}"]["docs_naming_candidate"] == ["A#00"]
    assert may[f"SPATIAL ENTITY@{_REGION}"]["docs_naming_candidate"] is None
    assert may[f"SPATIAL ENTITY@{_MT}"]["feature"] == "T.MT"
    assert _REGION in by_id           # its ontology term, joined by EXACT_MATCH
    assert MAY_BE in STRUCTURAL


@pytest.mark.parametrize("text", ["lowlands", "National Park", "they"])
def test_common_nouns_stay_unresolved(text):
    r = _ambiguous("x", text, _KILI)
    assert ambiguous_place_key(r) is None


def test_non_place_candidates_are_not_a_place():
    r = _ambiguous("x", "Cyanobacteria", [dict(_KILI[0], source="gbif"), _KILI[1]])
    assert ambiguous_place_key(r) is None
    assert ambiguous_place_key(_ambiguous("x", "Kilimanjaro", _KILI)) == "kilimanjaro"
    assert ambiguous_place_key(_ambiguous("x", "Kilimanjaro", _KILI[:1])) is None


def test_flag_off_keeps_unresolved_nodes(tmp_path):
    nodes, edges, _ = _two_papers_say_kilimanjaro(tmp_path, ambiguous_places=False)
    assert not [n for n in nodes if n["kind"] == "AmbiguousPlace"]
    assert len([n for n in nodes if n["kind"] == "Unresolved"]) == 2
    assert not [e for e in edges if e["relation"] == "MAY_BE"]


def test_neo4j_flattens_candidates():
    from src.graph.load_neo4j import node_props
    props = node_props({"id": "place:kilimanjaro", "kind": "AmbiguousPlace",
                        "candidates": [{"curie": "geonames:157452", "label": "Kilimanjaro",
                                        "feature": "T.MT"}]}, "g")
    assert props["candidates"] == ["geonames:157452 Kilimanjaro (T.MT)"]


# ── GMBA ranges: crosswalk and WITHIN (decided 3 Nov 2026) ───────

def test_gmba_crosswalk_and_hierarchy(tmp_path):
    from src.graph.build_graph import WITHIN
    uri = "https://sws.geonames.org/179640/"
    row = _row("t", "Taita Hills", "SPATIAL ENTITY", eid=f"SPATIAL ENTITY@{uri}", uri=uri,
               outcome="linked")
    row["gmba"] = {"uri": "urn:gmba:v2:11972", "curie": "gmba:11972", "label": "Taita Hills",
                   "wikidata": None, "crosswalk": True,
                   "path": [{"uri": "urn:gmba:v2:12149", "curie": "gmba:12149",
                             "label": "Eastern Arc Mountains", "feature": "x"},
                            {"uri": "urn:gmba:v2:12154", "curie": "gmba:12154",
                             "label": "Africa", "feature": "(Sub)-Continent"}]}
    path, table = _setup(tmp_path, [row], [{"text": "Taita Hills.", "relations": []}])
    nodes, edges, stats = build([path], table)
    by_id = {n["id"]: n for n in nodes}
    assert {(e["source"], e["target"]) for e in edges if e["relation"] == EXACT_MATCH} == {
        (f"SPATIAL ENTITY@{uri}", uri), (f"SPATIAL ENTITY@{uri}", "urn:gmba:v2:11972")}
    assert [(e["source"], e["target"]) for e in edges if e["relation"] == WITHIN] == [
        ("urn:gmba:v2:11972", "urn:gmba:v2:12149"), ("urn:gmba:v2:12149", "urn:gmba:v2:12154")]
    assert by_id["urn:gmba:v2:12154"]["name"] == "Africa (GMBA mountain region)"
    assert by_id["urn:gmba:v2:11972"]["curie"] == "gmba:11972"
    assert WITHIN in STRUCTURAL and stats["entities_gmba_crosswalk"] == 1


# ── stable ids: file names and merges ────────────────────────────

def _file_named(tmp_path, stem):
    rows = [_row("a1", "forest", mention_id=f"{stem}:a1#1", file=f"{stem}.json"),
            _row("v1", "2743 m", etype="SPATIAL PROPERTY",
                 mention_id=f"{stem}:v1#1", file=f"{stem}.json"),
            _row("u1", "the lake", etype="ABIOTIC ENTITY", outcome="ambiguous",
                 mention_id=f"{stem}:u1#1", file=f"{stem}.json")]
    path, table = _setup(tmp_path, rows, [{"text": "s1", "relations": [
        _rel("r1", "a1", "v1", e2_type="SPATIAL PROPERTY"),
        _rel("r2", "u1", "a1", label="PART_OF", e1_type="ABIOTIC ENTITY")]}],
        name=f"{stem}.json")
    nodes, _, _ = build([path], table)
    return {n["id"] for n in nodes}


def test_node_ids_do_not_depend_on_the_input_file_name(tmp_path):
    first = _file_named(tmp_path, "batch_01")
    second = _file_named(tmp_path, "kimi_run_renamed")
    assert first == second
    assert any(i.startswith("value:D#00:") for i in first)
    assert any(i.startswith("unresolved:D#00:") for i in first)
    assert mention_key(_row("x", "x", mention_id="some_file:x#2", doc="P#01")) == "P#01:x#2"


def test_entity_nodes_keep_the_ids_they_had_before_a_merge(tmp_path):
    rows = [_row("a1", "insects", eid="BIOTIC ENTITY@gbif:216", uri="gbif:216",
                 outcome="linked"),
            _row("b1", "canopy")]
    rows[0]["entity_aliases"] = ["mobiko:e-insects", "mobiko:e-insect"]
    rows[1]["entity_aliases"] = None
    path, table = _setup(tmp_path, rows, [{"text": "s1", "relations": [_rel("r1", "a1", "b1")]}])
    nodes, _, _ = build([path], table)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["BIOTIC ENTITY@gbif:216"]["aliases"] == ["mobiko:e-insect", "mobiko:e-insects"]
    assert by_id["mobiko:canopy"]["aliases"] is None
