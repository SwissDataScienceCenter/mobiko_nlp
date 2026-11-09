"""
Values as measurements: parsed into typed fields, one node per mention.

  1. The parser reads what the corpus actually writes — ranges with en dashes,
     "between X and Y", "±", percentages, number words, "1.4 million",
     statistics, coordinates — and refuses what is not a value ("1-septate",
     "R2 values", a long phrase with a size in it).
  2. In the graph a value is a Measurement of its own: never INSTANCE_OF a
     shared "25%", so two papers mentioning 25% are not neighbours.
  3. Statements carry no p-values: significance is the relation's property
     (not significant = Negation), and the values stay in the sentence.
"""
from __future__ import annotations

import pytest

from src.graph.build_graph import STRUCTURAL, build
from src.graph.tests.test_build_graph import _rel, _row, _setup
from src.graph.values import parse_value


@pytest.mark.parametrize("text,want", [
    ("2743 m", {"kind": "number", "value": 2743.0, "unit": "m", "comparator": "="}),
    ("2,730 m a.s.l.", {"value": 2730.0, "unit": "m a.s.l."}),
    ("6–10 µm", {"kind": "range", "low": 6.0, "high": 10.0, "unit": "µm"}),
    ("between 0.16 and 0.58", {"kind": "range", "low": 0.16, "high": 0.58}),
    ("2329 to 4657 m", {"kind": "range", "low": 2329.0, "high": 4657.0, "unit": "m"}),
    ("0.151 ± 0.183", {"value": 0.151, "uncertainty": 0.183}),
    ("−1.475 ± 0.151", {"value": -1.475, "uncertainty": 0.151}),
    ("25%", {"kind": "percentage", "value": 25.0, "unit": "%"}),
    ("less than 10%", {"kind": "percentage", "value": 10.0, "comparator": "<"}),
    ("approximately 10%", {"value": 10.0, "comparator": "~"}),
    (">0.97", {"value": 0.97, "comparator": ">"}),
    ("eight", {"value": 8.0, "unit": ""}),
    ("four species", {"value": 4.0, "unit": "species"}),
    ("1.4 million", {"value": 1400000.0, "unit": ""}),
    ("p < .001", {"kind": "statistic", "stat": "p", "comparator": "<", "value": 0.001}),
    ("P = 0.002", {"kind": "statistic", "stat": "p", "value": 0.002}),
    ("R 2 = 0.50", {"kind": "statistic", "stat": "R2", "value": 0.5}),
])
def test_parse(text, want):
    got = parse_value(text)
    assert got is not None and got["raw"] == text
    for k, v in want.items():
        assert got[k] == v, (k, got)


def test_coordinate_with_double_apostrophe_seconds():
    got = parse_value("03°10’00’’ S, 37°21’40’’ E")
    assert got["kind"] == "coordinate"
    assert got["lat"] == pytest.approx(-3.166667) and got["lon"] == pytest.approx(37.361111)


def test_coordinate():
    got = parse_value("3°14’47.71” S 36°29’22.96” E")
    assert got["kind"] == "coordinate"
    assert got["lat"] == pytest.approx(-3.246586) and got["lon"] == pytest.approx(36.489711)


@pytest.mark.parametrize("text", ["high", "lowest", "1-septate", "R2 values", "km2", "SGT 438b",
                                  "1.1.–1.4 Mg ha−1 year−1",
                                  "2,900 m elevational gradient extending across five zones",
                                  "", None])
def test_not_a_value(text):
    assert parse_value(text) is None


# ── in the graph ─────────────────────────────────────────────────

def test_values_are_measurements_not_shared_concepts(tmp_path):
    rows = [_row("x1", "cover", "ABIOTIC PROPERTY", doc="X#00"),
            _row("x2", "25%", "QUANTITATIVE PROPERTY", doc="X#00"),
            _row("y1", "loss", "BIOTIC PROPERTY", doc="Y#00"),
            _row("y2", "25%", "QUANTITATIVE PROPERTY", doc="Y#00")]
    for r in rows:
        r["entity_id"] = f"mobiko:{r['text']}"
    path, table = _setup(tmp_path, rows, [
        {"text": "Cover was 25% (p = .01).", "relations": [
            _rel("r1", "x1", "x2", "HAS_VALUE", "ABIOTIC PROPERTY", "QUANTITATIVE PROPERTY")]},
        {"text": "Loss was 25%.", "relations": [
            _rel("r2", "y1", "y2", "HAS_VALUE", "BIOTIC PROPERTY", "QUANTITATIVE PROPERTY")]}])
    nodes, edges, stats = build([path], table, store=tmp_path / "s.sqlite")
    ms = [n for n in nodes if n["kind"] == "Measurement"]
    assert len(ms) == 2 and {m["doc_id"] for m in ms} == {"X#00", "Y#00"}
    assert all(m["value"] == 25.0 and m["unit"] == "%" for m in ms)
    # no concept "25%", and nothing points a measurement at one
    assert not any(n["kind"] == "Entity" and n["name"] == "25%" for n in nodes)
    assert not any(e["relation"] == "INSTANCE_OF" and e["source"].startswith("value:")
                   for e in edges)
    sem = {e["source"]: e for e in edges if e["relation"] not in STRUCTURAL}
    x_edge = next(e for e in sem.values() if "X#00" in e["docs"])
    assert "p_values" not in x_edge and "statistics" not in x_edge
    from src.graph.text_store import TextStore
    with TextStore(tmp_path / "s.sqlite") as st:
        assert "p = .01" in st.sentence(x_edge["evidence"][0]["sentence_id"])
    assert stats["nodes_measurement"] == 2


def test_a_counted_entity_stays_an_entity(tmp_path):
    rows = [_row("a", "Five sites", "SPATIAL ENTITY"), _row("b", "Tanzania", "SPATIAL ENTITY")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r", "a", "b", "LOCATED_IN", "SPATIAL ENTITY", "SPATIAL ENTITY")]}])
    nodes, _, _ = build([path], table)
    assert not any(n["kind"] == "Measurement" for n in nodes)


def test_measurements_can_be_turned_off(tmp_path):
    rows = [_row("a", "cover", "ABIOTIC PROPERTY"), _row("b", "25%", "QUANTITATIVE PROPERTY")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r", "a", "b", "HAS_VALUE", "ABIOTIC PROPERTY", "QUANTITATIVE PROPERTY")]}])
    nodes, _, _ = build([path], table, measurements=False)
    assert not any(n["kind"] == "Measurement" for n in nodes)


def test_has_property_to_a_number_is_read_as_has_value(tmp_path):
    """Decided 3 Nov 2026: a thing links to its number by HAS_VALUE."""
    rows = [_row("a", "species", "BIOTIC ENTITY"), _row("b", "28", "QUANTITATIVE PROPERTY"),
            _row("c", "richness", "BIOTIC PROPERTY"), _row("d", "p = .03", "QUANTITATIVE PROPERTY")]
    path, table = _setup(tmp_path, rows, [{"text": "s", "relations": [
        _rel("r1", "a", "b", "HAS_PROPERTY", "BIOTIC ENTITY", "QUANTITATIVE PROPERTY"),
        _rel("r2", "c", "d", "HAS_PROPERTY", "BIOTIC PROPERTY", "QUANTITATIVE PROPERTY")]}])
    _, edges, stats = build([path], table)
    by_target = {e["target"]: e for e in edges if e["relation"] not in STRUCTURAL}
    count = next(e for t, e in by_target.items() if t.endswith(":b#1"))
    stat = next(e for t, e in by_target.items() if t.endswith(":d#1"))
    assert count["relation"] == "HAS_VALUE"
    assert count["evidence"][0]["relation_as_annotated"] == "HAS_PROPERTY"
    assert stat["relation"] == "HAS_PROPERTY"           # a statistic is not a value
    assert stats["has_property_to_a_number_read_as_has_value"] == 1
