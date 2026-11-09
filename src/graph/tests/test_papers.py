"""
Paper, journal, year and section for every statement (decided 9 Nov 2026).

Pinned here:
  1. A BioC record gives the paper's citation, journal and year.
  2. A sentence's section follows the manifest's abstract count, and a Results
     sentence takes the heading of the subsection its passage sits in,
     including a number passage joined to its title ("2.1. Sites").
  3. A text file that no longer matches the manifest, or a line that differs
     from the annotated sentence, gets no section rather than a wrong one.
  4. The graph: a Paper node per paper, the section on every statement, the
     papers / years / journals / sections on every edge, and both in the store.

Run with:
    pytest src/graph/tests/test_papers.py -v
"""
from __future__ import annotations

import json

from src.graph import papers as P
from src.graph.build_graph import build
from src.graph.load_neo4j import edge_props
from src.graph.text_store import TextStore
from src.graph.tests.test_build_graph import _rel, _row, _semantic, _setup

ABSTRACT = ["Abstract Bats live on the mountain."]
RESULTS = ["Richness fell with elevation.", "Soils were acidic.", "Sites were dry."]


def _passage(field, cid, text, tag=None):
    return {"infons": {"field": field, "content_id": cid, "tag": tag}, "text": text}


def _source(tmp_path, lines=None, n_total=None):
    lines = lines if lines is not None else ABSTRACT + RESULTS
    d = tmp_path / "src"
    (d / "raw").mkdir(parents=True)
    (d / "PMC1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "manifest.csv").write_text(
        "pmcid,n_abstract,n_results,n_total,results_section_title,doi\n"
        f"PMC1,1,3,{n_total or 4},Results,10.1/x\n", encoding="utf-8")
    article = {"id": "PMC1", "infons": {
        "pmcid": "PMC1", "title": "Bats  on\nmountains", "journal": "Ecology and Evolution",
        "pubyear": "2024", "doi": "10.1/x", "pmid": "42",
        "authors": "Ann Snider;Bo Helgen;Cy Young"}, "passages": [
        _passage("abstract", "0", ABSTRACT[0]),
        _passage("section_title", "3.2", "Methods"),
        _passage("text", "3.2", "Sites were dry.", "p"),       # same words, not Results
        _passage("section_title", "3.3", "3. Results"),
        _passage("text", "3.3", RESULTS[0], "p"),
        _passage("section_title", "3.3.1", "3.1."),
        _passage("section_title", "3.3.1", "Soils"),
        _passage("text", "3.3.1", RESULTS[1], "p"),
        _passage("section_title", "3.3.2", "Climate"),
        _passage("text", "3.3.2", RESULTS[2], "p"),
    ]}
    (d / "raw" / "PMC1.json").write_text(
        json.dumps({"sibils_article_set": [article]}), encoding="utf-8")
    return d


def test_paper_record_from_bioc(tmp_path):
    rec = P.papers(_source(tmp_path))["PMC1"]
    assert rec["citation"] == "Snider et al. 2024"
    assert (rec["title"], rec["journal"], rec["year"]) == (
        "Bats on mountains", "Ecology and Evolution", 2024)


def test_sections_and_subsections(tmp_path):
    s = P.sentence_sections(_source(tmp_path))
    assert s[("PMC1", 0)] == ("Abstract", None)
    assert s[("PMC1", 1)] == ("Results", None)             # directly under Results
    assert s[("PMC1", 2)] == ("Results", "3.1. Soils")      # number joined to title
    assert s[("PMC1", 3)] == ("Results", "Climate")         # not the Methods passage


def test_no_section_when_the_file_or_line_does_not_match(tmp_path):
    stats = {}
    assert P.sentence_sections(_source(tmp_path, n_total=5), stats) == {}
    assert stats["papers_without_sections"] == 1
    stats = {}
    s = P.sentence_sections(_source(tmp_path / "b"), stats,
                            expected={("PMC1", 2): "Something else."})
    assert ("PMC1", 2) not in s and stats["section_text_mismatch"] == 1


def test_graph_carries_paper_and_section(tmp_path):
    src = _source(tmp_path)
    rows = [_row("a", "richness", doc="PMC1#00"), _row("b", "elevation", doc="PMC1#00")]
    for r in rows:
        r["sent_index"] = 1
    sentences = [{"text": t} for t in ABSTRACT + RESULTS]
    sentences[1]["relations"] = [_rel("r1", "a", "b", label="DECREASES_WITH")]
    path, table = _setup(tmp_path, rows, sentences, doc="PMC1")
    store = tmp_path / "s.sqlite"
    nodes, edges, stats = build([path], table, store=store, papers_dir=src)

    paper = [n for n in nodes if n["kind"] == "Paper"]
    assert [(n["id"], n["name"]) for n in paper] == [("paper:PMC1", "Snider et al. 2024")]
    [e] = _semantic(edges)
    assert (e["papers"], e["years"], e["journals"], e["sections"]) == (
        ["PMC1"], [2024], ["Ecology and Evolution"], ["Results"])
    assert e["evidence"][0]["section"] == "Results"
    assert edge_props(e)["evidence_sections"] == ["Results"]
    assert stats["evidence_with_section"] == 1

    with TextStore(store) as ts:
        [ev] = ts.evidence(e["id"])
        assert (ev["citation"], ev["journal"], ev["year"], ev["section"]) == (
            "Snider et al. 2024", "Ecology and Evolution", 2024, "Results")
        assert ts.paper("PMC1#00")["title"] == "Bats on mountains"


def test_without_papers_dir_nothing_changes(tmp_path):
    rows = [_row("a", "richness"), _row("b", "elevation")]
    path, table = _setup(tmp_path, rows, [{"text": "x", "relations": [_rel("r1", "a", "b")]}])
    nodes, edges, _ = build([path], table)
    [e] = _semantic(edges)
    assert e["papers"] == ["D"] and e["years"] is None and e["sections"] is None
    assert not [n for n in nodes if n["kind"] == "Paper"]
