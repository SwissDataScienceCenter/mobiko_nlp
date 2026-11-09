"""
The graph's text, stored once: ``sentences.sqlite`` next to ``nodes.jsonl``.

Neo4j reads a property whole, and long strings sit in a separate store, so
text copied onto nodes and edges is slow and large: on second_iteration the
nodes held each sentence 13 times over, and a shared concept such as
"elevation" kept every sentence of every paper in one list (920 sentences,
210 KB, in a 240-paper test). The graph now holds references, and the text is
here, once (decided 3 Nov 2026):

    sentence(id, doc_id, sent_index, text,        every sentence of the input,
             section, subsection)                  with its section (Abstract /
                                                   Results) and Results subsection
    paper(pmcid, title, journal, year, ...)        one row per paper (``papers``)
    sentence_fts                                   FTS5 keyword index over it
    mention(key, node_id, concept_id, sentence_id, start_char, end_char,
            text, entity_type, part_index, split_from,
            refers_to_text, refers_to_sent)       where each mention is
    evidence(edge_id, sentence_id, doc_id, relation_uid, annotator,
             confidence, distributed, coref, file) one row per statement of an edge

Ids:
    sentence  ``<doc_id>/<sent_index>``, e.g. ``PMC6920850#00/37``
    mention   ``mention_key``: ``<doc_id>:<uid>#<n>`` (no file name)
    node_id   the node the mention is drawn as: an instance, a measurement,
              a shared concept, a place meta-node, or an unresolved node
    concept_id the concept (Entity) the mention names, when it names one

A node's sentences are its mentions: ``node_id = ?`` for any node, and for a
concept also ``concept_id = ?`` (its mentions in every paper). An edge's are
``evidence.edge_id = ?``. ``TextStore`` answers both, rendered the way the
viewer shows them, and ``search`` runs a keyword query. RAG joins on the same
ids: a Cypher result gives node and edge ids, the store gives the text, and a
text hit gives a sentence id that ``evidence`` and ``mention`` lead back into
the graph.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCHEMA = """
CREATE TABLE sentence (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
                       sent_index INTEGER NOT NULL, text TEXT NOT NULL,
                       section TEXT, subsection TEXT);
CREATE TABLE paper (pmcid TEXT PRIMARY KEY, citation TEXT, title TEXT,
                    journal TEXT, year INTEGER, publication_date TEXT,
                    doi TEXT, pmid TEXT, n_authors INTEGER);
CREATE TABLE mention (key TEXT PRIMARY KEY, node_id TEXT, concept_id TEXT,
                      sentence_id TEXT NOT NULL, doc_id TEXT NOT NULL,
                      sent_index INTEGER NOT NULL, start_char INTEGER,
                      end_char INTEGER, text TEXT NOT NULL, entity_type TEXT,
                      part_index INTEGER, split_from TEXT,
                      refers_to_text TEXT, refers_to_sent INTEGER);
CREATE TABLE evidence (edge_id TEXT NOT NULL, sentence_id TEXT NOT NULL,
                       doc_id TEXT NOT NULL, relation_uid TEXT, annotator TEXT,
                       confidence REAL, distributed INTEGER, coref TEXT, file TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX mention_node ON mention(node_id);
CREATE INDEX mention_concept ON mention(concept_id);
CREATE INDEX mention_sentence ON mention(sentence_id);
CREATE INDEX evidence_edge ON evidence(edge_id);
CREATE INDEX evidence_sentence ON evidence(sentence_id);
"""
FTS = ("CREATE VIRTUAL TABLE sentence_fts USING fts5("
       "text, content='sentence', content_rowid='rowid')")


def sentence_id(doc_id: str, sent_index: int) -> str:
    return f"{doc_id}/{sent_index}"


def short_doc(doc_id: str) -> str:
    """``PMC6920850#00`` -> ``PMC6920850``; a real sub-document keeps its suffix."""
    return doc_id[:-3] if doc_id.endswith("#00") else doc_id


def write_store(path: Path, sentences: Dict[str, Tuple[str, int, str]],
                mentions: Iterable[dict], evidence: Iterable[dict],
                meta: Optional[dict] = None,
                sections: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
                papers: Iterable[dict] = ()) -> None:
    """Write the store from scratch. ``sentences`` is id -> (doc_id, index, text);
    ``sections`` is id -> (section, subsection); ``papers`` are ``papers.paper_record``s."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        sections = sections or {}
        con.executemany("INSERT INTO sentence VALUES (?,?,?,?,?,?)",
                        ((sid, d, i, t, *sections.get(sid, (None, None)))
                         for sid, (d, i, t) in sorted(sentences.items())))
        con.executemany(
            "INSERT INTO paper VALUES (:pmcid,:citation,:title,:journal,:year,"
            ":publication_date,:doi,:pmid,:n_authors)", papers)
        con.executemany(
            "INSERT INTO mention VALUES (:key,:node_id,:concept_id,:sentence_id,:doc_id,"
            ":sent_index,:start_char,:end_char,:text,:entity_type,:part_index,"
            ":split_from,:refers_to_text,:refers_to_sent)", mentions)
        con.executemany(
            "INSERT INTO evidence VALUES (:edge_id,:sentence_id,:doc_id,:relation_uid,"
            ":annotator,:confidence,:distributed,:coref,:file)", evidence)
        con.executemany("INSERT INTO meta VALUES (?,?)",
                        [(k, str(v)) for k, v in (meta or {}).items()])
        try:
            con.execute(FTS)
            con.execute("INSERT INTO sentence_fts(sentence_fts) VALUES ('rebuild')")
        except sqlite3.OperationalError:     # an SQLite built without FTS5
            pass
        con.commit()
    finally:
        con.close()
    tmp.replace(path)


def mention_parts(m: dict, sentence: str) -> Tuple[bool, str]:
    """(whether to mark the mention in its sentence, the note after it).

    The mention is marked when its offsets check out. A part of a split
    coordination says what it was split from. A mention with no usable
    offsets is one the annotator inferred rather than read ("elevation" for a
    bare "2456 m"): it is said to be implied, not marked somewhere it is not.
    An anaphor says what coreference resolved it to.
    """
    start, end = m.get("start_char"), m.get("end_char")
    marked = (m.get("part_index") is None and isinstance(start, int)
              and isinstance(end, int) and 0 <= start < end <= len(sentence))
    if m.get("part_index") is not None:
        note = f"  [part of «{m['split_from']}»: {m['text']}]"
    elif not marked:
        note = f"  [implied: {m['text']}]"
    else:
        note = ""
    if m.get("refers_to_text") is not None:
        note += f"  [coreference: refers to «{m['refers_to_text']}», s{m['refers_to_sent']}]"
    return marked, note


def render_mention(m: dict, sentence: str) -> str:
    """``PMC6920850 s77: … at «moderate elevations» in …`` for one mention."""
    marked, note = mention_parts(m, sentence)
    text = (f"{sentence[:m['start_char']]}«{sentence[m['start_char']:m['end_char']]}»"
            f"{sentence[m['end_char']:]}" if marked else sentence)
    return f"{short_doc(m['doc_id'])} s{m['sent_index']}: {text}{note}"


class TextStore:
    """Read access to ``sentences.sqlite``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.con.row_factory = sqlite3.Row

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def sentence(self, sid: str) -> Optional[str]:
        row = self.con.execute("SELECT text FROM sentence WHERE id = ?", (sid,)).fetchone()
        return row["text"] if row else None

    def mentions(self, node_id: str, concept: bool = False) -> List[dict]:
        """A node's mentions in reading order; a concept's in every paper."""
        sql = ("SELECT m.*, s.text AS sentence FROM mention m "
               "JOIN sentence s ON s.id = m.sentence_id WHERE m.node_id = ?")
        args: tuple = (node_id,)
        if concept:
            sql += " OR m.concept_id = ?"
            args = (node_id, node_id)
        rows = [dict(r) for r in self.con.execute(sql, args)]
        rows.sort(key=lambda r: (r["doc_id"], r["sent_index"],
                                 r["start_char"] if r["start_char"] is not None else -1,
                                 r["key"]))
        return rows

    def contexts(self, node_id: str, concept: bool = False) -> List[str]:
        """The node's sentences as the viewer shows them, without repeats."""
        out: Dict[str, None] = {}
        for m in self.mentions(node_id, concept):
            out.setdefault(render_mention(m, m["sentence"]), None)
        return list(out)

    def evidence(self, edge_id: str) -> List[dict]:
        """An edge's statements with sentence, section and paper (when known)."""
        return [dict(r) for r in self.con.execute(
            "SELECT e.*, s.text AS sentence, s.section, s.subsection, "
            "p.citation, p.title, p.journal, p.year, p.doi FROM evidence e "
            "JOIN sentence s ON s.id = e.sentence_id "
            "LEFT JOIN paper p ON p.pmcid = substr(e.doc_id, 1, "
            "CASE instr(e.doc_id, '#') WHEN 0 THEN length(e.doc_id) "
            "ELSE instr(e.doc_id, '#') - 1 END) "
            "WHERE e.edge_id = ? "
            "ORDER BY e.doc_id, s.sent_index, e.relation_uid", (edge_id,))]

    def paper(self, pmcid: str) -> Optional[dict]:
        row = self.con.execute("SELECT * FROM paper WHERE pmcid = ?",
                               (pmcid.split("#", 1)[0],)).fetchone()
        return dict(row) if row else None

    def search(self, query: str, limit: int = 20) -> List[dict]:
        """Keyword search (FTS5 syntax), best first: id, doc_id, text."""
        return [dict(r) for r in self.con.execute(
            "SELECT s.id, s.doc_id, s.sent_index, s.text FROM sentence_fts f "
            "JOIN sentence s ON s.rowid = f.rowid WHERE sentence_fts MATCH ? "
            "ORDER BY rank LIMIT ?", (query, limit))]
