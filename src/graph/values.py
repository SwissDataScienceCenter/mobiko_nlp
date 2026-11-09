"""
Parse an annotated value — "2743 m", "6–10 µm", "p < .001", "eight" — into
typed fields a graph can query.

Why this exists
---------------
Values used to be graph entities like any other: `2743 m` was a node with an
INSTANCE_OF edge to a shared concept "2743 m". That made them text, not numbers
(no "sites above 2000 m"), and made every paper mentioning "25%" a neighbour of
every other through one shared node — 34 such value concepts on second_iteration,
each a false bridge for a graph-RAG retriever expanding neighbourhoods. A parsed
value becomes a Measurement record instead (src/graph/build_graph.py): one per
mention, never shared, with ``value`` / ``low`` / ``high`` / ``unit`` /
``comparator`` and the raw text kept beside them.

`parse_value` returns None for anything that is not a value expression — "high",
"lowest", "1-septate" — and those stay qualitative properties. It refuses
rather than guesses: a long phrase that merely contains a number ("2,900 m
elevational gradient") is not a value, it is a thing with a size.

"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_TRANSLATE = str.maketrans({"−": "-", "–": "-", "—": "-", "‒": "-", "˂": "<", "˃": ">",
                            "’": "'", "′": "'", "”": '"', "″": '"', " ": " "})

_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)"

_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
          "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "dozen": 12}

# Word comparators in front of a number. "~" means approximately.
_COMPARATOR_WORDS = [
    (r"less than|lower than|below|under|fewer than", "<"),
    (r"more than|greater than|higher than|above|over|exceeding", ">"),
    (r"up to|at most|no more than", "<="),
    (r"at least|no less than", ">="),
    (r"approximately|approx\.?|about|around|roughly|nearly|almost|c\.|ca\.|circa|~", "~"),
]
_COMPARATOR_SYMBOL = {"<": "<", ">": ">", "<=": "<=", ">=": ">=", "≤": "<=", "≥": ">=",
                      "=": "=", "~": "~"}

# Statistic names as they appear in results sections.
_STAT = (r"(?P<name>ΔAICc?|AICc?|BIC|r2adj|R\s?2adj|R\s?²|R\s?2|r\s?2|R|r|rM|rs|ρ|τ|"
         r"p|P|t|F(?:\s?\d+\s?,\s?\d+)?|χ\s?2|χ²|chi2|G|H|U|V|W|Z|z|df|edf|n|N|ED)")
_STAT_RE = re.compile(rf"^{_STAT}\s*(?P<cmp><=|>=|≤|≥|=|<|>)\s*(?P<num>{_NUM})\s*(?P<rest>.*)$")

_DMS = re.compile(r"""(?P<d>\d{1,3})\s*°\s*(?:(?P<m>\d{1,2}(?:\.\d+)?)\s*'?\s*)?"""
                  r"""(?:(?P<s>\d{1,2}(?:\.\d+)?)\s*(?:"|'')?\s*)?(?P<h>[NSEW])\b""")

# "1.4 million" is 1,400,000, not 1.4 of a unit called "million".
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
# One spelling per statistic, so a query for "p" also finds "P".
_STAT_CANON = {"P": "p", "R²": "R2", "χ²": "χ2", "chi2": "χ2", "N": "n", "R2adj": "r2adj"}

# A unit tail longer than this is a description, not a unit.
_MAX_UNIT_WORDS = 4


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _clean_unit(rest: str) -> str:
    """Trailing punctuation off, except the closing dot of an abbreviation
    ("m a.s.l."), which is part of the unit rather than the sentence."""
    unit = rest.strip().strip(" ,;:()[]")
    if unit.endswith(".") and "." not in unit[:-1]:
        unit = unit[:-1]
    return unit.strip()


def _scaled(value: float, unit: str):
    """(value, unit) with a leading "thousand"/"million"/"billion" folded in."""
    head, _, tail = unit.partition(" ")
    if head.lower() in _SCALE:
        return value * _SCALE[head.lower()], tail.strip()
    return value, unit


def _stat_name(name: str) -> str:
    name = re.sub(r"\s+", "", name)
    return _STAT_CANON.get(name, name)


def _plausible_unit(unit: str) -> bool:
    if not unit:
        return True
    if len(unit.split()) > _MAX_UNIT_WORDS:
        return False
    # A unit starts with a letter or a unit sign; ".–1.4 Mg" is a malformed
    # range ("1.1.–1.4"), not 1.1 of something.
    if not re.match(r"[A-Za-z%°µ×]", unit):
        return False
    # "1-septate": a number welded to a word is a term, not a quantity.
    return not unit.startswith("-")


def _coordinate(text: str) -> Optional[Dict]:
    hits = list(_DMS.finditer(text))
    if not hits or _DMS.sub("", text).strip(" ,;"):
        return None
    out: Dict = {"kind": "coordinate", "unit": "decimal degrees"}
    for h in hits:
        deg = float(h["d"]) + float(h["m"] or 0) / 60 + float(h["s"] or 0) / 3600
        if h["h"] in "SW":
            deg = -deg
        out["lat" if h["h"] in "NS" else "lon"] = round(deg, 6)
    return out


def parse_value(text: Optional[str]) -> Optional[Dict]:
    """Typed fields for a value expression, or None when ``text`` is not one.

    Keys present depend on the kind: ``kind`` (number | range | percentage |
    statistic | coordinate), ``value`` or ``low``/``high``, ``unit``,
    ``comparator`` ("=", "<", ">", "<=", ">=", "~"), ``uncertainty`` (the "±"
    term), ``stat`` (for a statistic), ``lat``/``lon`` (for a coordinate).
    ``raw`` is always the text as annotated.
    """
    if not text or not isinstance(text, str):
        return None
    raw = text
    t = text.translate(_TRANSLATE).strip()
    if not t:
        return None

    coord = _coordinate(t)
    if coord:
        return {**coord, "raw": raw}

    m = _STAT_RE.match(t)
    if m and _plausible_unit(_clean_unit(m["rest"])) and not _clean_unit(m["rest"]):
        return {"kind": "statistic", "stat": _stat_name(m["name"]),
                "comparator": _COMPARATOR_SYMBOL[m["cmp"]], "value": _num(m["num"]),
                "raw": raw}

    comparator = "="
    for words, sym in _COMPARATOR_WORDS:
        mm = re.match(rf"^(?:{words})\s*", t, re.I)
        if mm:
            comparator, t = sym, t[mm.end():]
            break
    else:
        mm = re.match(r"^(<=|>=|≤|≥|<|>|~)\s*", t)
        if mm:
            comparator, t = _COMPARATOR_SYMBOL[mm[1]], t[mm.end():]

    mm = re.match(rf"^(?:between\s+)?({_NUM})\s*(?:-|to|and)\s*({_NUM})\s*(.*)$", t, re.I)
    if mm and _plausible_unit(_clean_unit(mm[3])):
        unit = _clean_unit(mm[3])
        return {"kind": "percentage" if unit.startswith("%") else "range",
                "low": _num(mm[1]), "high": _num(mm[2]),
                "unit": unit, "comparator": comparator, "raw": raw}

    mm = re.match(rf"^({_NUM})\s*(?:±|\+/-|\+-)\s*({_NUM})\s*(.*)$", t)
    if mm and _plausible_unit(_clean_unit(mm[3])):
        return {"kind": "number", "value": _num(mm[1]), "uncertainty": _num(mm[2]),
                "unit": _clean_unit(mm[3]), "comparator": comparator, "raw": raw}

    mm = re.match(rf"^({_NUM})\s*(.*)$", t)
    if mm and _plausible_unit(_clean_unit(mm[2])):
        value, unit = _scaled(_num(mm[1]), _clean_unit(mm[2]))
        kind = "percentage" if unit.startswith("%") else "number"
        return {"kind": kind, "value": value,
                "unit": "%" if kind == "percentage" else unit,
                "comparator": comparator, "raw": raw}

    mm = re.match(r"^([A-Za-z]+)\b\s*(.*)$", t)
    if mm and mm[1].lower() in _WORDS and _plausible_unit(_clean_unit(mm[2])):
        return {"kind": "number", "value": float(_WORDS[mm[1].lower()]),
                "unit": _clean_unit(mm[2]), "comparator": comparator, "raw": raw}
    return None
