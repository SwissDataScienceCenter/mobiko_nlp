"""
Where each statement comes from: the paper, its journal and year, and the
section of the sentence (decided 9 Nov 2026).

The annotated sentences are written by ``scripts/fetch_bioc_abstract_results.py``
into ``<data_dir>/<pmcid>.txt``, one sentence per line: the abstract, then the
Results section. The same run keeps the BioC record in ``<data_dir>/raw/`` and
the per-paper counts in ``manifest.csv``. This module reads those three back:

    papers(data_dir)            pmcid -> title, journal, year, doi, pmid, ...
    sentence_sections(data_dir) (pmcid, line) -> (section, subsection)

``section`` is "Abstract" or "Results" (the line's position against the
manifest's ``n_abstract``). ``subsection`` is the heading of the Results
subsection the sentence's BioC passage sits in ("List of Crustose Calicioid
Species in Tanzania"), or None for the abstract and for a passage directly
under "Results". A paper whose text file no longer matches the manifest's
count gets no sections rather than wrong ones.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ABSTRACT, RESULTS = "Abstract", "Results"
# "Results", "3. Results", "RESULTS AND DISCUSSION" (as the fetch script finds it)
_RESULTS_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?results\b", re.I)
# a heading that is only a number: "2.1." (BioC gives it as its own passage)
_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s*$")


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _article(raw: Path) -> Optional[dict]:
    try:
        payload = json.loads(raw.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    articles = payload.get("sibils_article_set") or []
    return articles[0] if articles else None


def _int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def paper_record(article: dict) -> dict:
    """The fields the graph keeps for a paper, from a BioC record's infons."""
    inf = article.get("infons") or {}
    authors = [a.strip() for a in (inf.get("authors") or "").split(";") if a.strip()]
    first = authors[0].split()[-1] if authors else None
    year = _int(inf.get("pubyear"))
    return {
        "pmcid": inf.get("pmcid") or article.get("id"),
        "title": _clean(inf.get("title")) or None,
        "journal": _clean(inf.get("journal")) or None,
        "year": year,
        "publication_date": inf.get("publication_date") or None,
        "doi": inf.get("doi") or None,
        "pmid": inf.get("pmid") or None,
        "n_authors": len(authors),
        # "Snider et al. 2024": how a reader names the paper
        "citation": (f"{first}{' et al.' if len(authors) > 2 else ''} {year or ''}".strip()
                     if first else None),
    }


def papers(data_dir: Path) -> Dict[str, dict]:
    """pmcid -> paper record, for every BioC record in ``<data_dir>/raw``."""
    out: Dict[str, dict] = {}
    for raw in sorted((Path(data_dir) / "raw").glob("*.json")):
        article = _article(raw)
        if article:
            rec = paper_record(article)
            out[rec["pmcid"] or raw.stem] = rec
    return out


def _headings(passages: List[dict]) -> Dict[str, str]:
    """content_id -> heading, joining a number passage to its title ("2.1. Sampling")."""
    out: Dict[str, str] = {}
    for p in passages:
        inf = p.get("infons") or {}
        if inf.get("field") != "section_title":
            continue
        cid, text = inf.get("content_id") or "", _clean(p.get("text"))
        if not text:
            continue
        out[cid] = f"{out[cid]} {text}" if cid in out else text
    return out


def _results_prefix(headings: Dict[str, str]) -> Optional[str]:
    """The top-level body section titled Results (body sections are "3.x")."""
    for cid, text in headings.items():
        parts = cid.split(".")
        if len(parts) == 2 and parts[0] == "3" and _RESULTS_RE.match(text):
            return cid
    return None


def _subsection(cid: str, prefix: str, headings: Dict[str, str]) -> Optional[str]:
    """The deepest heading between the Results section and the passage."""
    parts = cid.split(".")
    for n in range(len(parts), len(prefix.split(".")), -1):
        h = headings.get(".".join(parts[:n]))
        if h and not _NUMBER_RE.match(h):
            return h
    return None


def _results_passages(article: dict) -> List[Tuple[str, Optional[str]]]:
    """(text, subsection) per Results paragraph, in the fetch script's order."""
    passages = article.get("passages") or []
    headings = _headings(passages)
    prefix = _results_prefix(headings)
    if not prefix:
        return []
    out = []
    for p in passages:
        inf = p.get("infons") or {}
        cid = inf.get("content_id") or ""
        if inf.get("field") == "text" and inf.get("tag") == "p" and \
                (cid == prefix or cid.startswith(prefix + ".")):
            text = _clean(p.get("text"))
            if text:
                out.append((text, _subsection(cid, prefix, headings)))
    return out


def _manifest(data_dir: Path) -> Dict[str, dict]:
    path = Path(data_dir) / "manifest.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["pmcid"]: r for r in csv.DictReader(fh)}


def sentence_sections(data_dir: Path, stats: Optional[dict] = None,
                      expected: Optional[Dict[Tuple[str, int], str]] = None
                      ) -> Dict[Tuple[str, int], Tuple[str, Optional[str]]]:
    """(pmcid, line index) -> (section, subsection) for every annotated sentence.

    A Results sentence takes the subsection of the passage it starts with; the
    fetch script may have rejoined passages cut at an abbreviation, so a
    sentence can span several. One that matches no passage keeps its section
    and gets no subsection (counted in ``stats["subsection_unmatched"]``).
    ``expected`` is the annotated text per (pmcid, index); a line that differs
    from it gets no section (``stats["section_text_mismatch"]``), since the
    index then points at a different sentence.
    """
    data_dir = Path(data_dir)
    out: Dict[Tuple[str, int], Tuple[str, Optional[str]]] = {}
    for pmcid, row in _manifest(data_dir).items():
        txt = data_dir / f"{pmcid}.txt"
        n_abs, n_total = _int(row.get("n_abstract")), _int(row.get("n_total"))
        if not txt.exists() or n_abs is None:
            continue
        lines = txt.read_text(encoding="utf-8").splitlines()
        if n_total is not None and len(lines) != n_total:
            if stats is not None:
                stats["papers_without_sections"] = stats.get("papers_without_sections", 0) + 1
            continue
        article = _article(data_dir / "raw" / f"{pmcid}.json")
        results = _results_passages(article) if article else []
        j = 0
        for i, line in enumerate(lines):
            if expected is not None and (pmcid, i) in expected \
                    and _clean(expected[(pmcid, i)]) != _clean(line):
                if stats is not None:
                    stats["section_text_mismatch"] = stats.get("section_text_mismatch", 0) + 1
                continue
            if i < n_abs:
                out[(pmcid, i)] = (ABSTRACT, None)
                continue
            sent, sub = _clean(line), None
            for k in range(j, len(results)):
                if sent.startswith(results[k][0]) or results[k][0].startswith(sent):
                    sub, j = results[k][1], k + 1
                    break
            else:
                if stats is not None:
                    stats["subsection_unmatched"] = stats.get("subsection_unmatched", 0) + 1
            out[(pmcid, i)] = (RESULTS, sub)
    return out


def paper_of(doc_id: str) -> str:
    """``PMC6920850#00`` -> ``PMC6920850``."""
    return doc_id.split("#", 1)[0]
