"""
Single source of truth for MENTION-string normalisation.

Companion to ``multi_agent_annotation/label_norm.py``, which normalises entity
*labels* ("BIOTIC_ENTITY" -> "BIOTIC ENTITY"). This module normalises the entity
*strings* themselves ("Lakes " -> "lakes"), which is a different job with a
different failure mode.

Why the version constant matters
--------------------------------
The authority dictionary is built by running this normaliser over a few hundred
thousand ontology labels, and matching then runs it over mentions. If the two
sides are normalised differently, exact match silently under-fires: no error, no
warning, just a lower link rate that reads as a property of the data. Changing
the rules here therefore invalidates every compiled artifact.

``NORMALIZER_VERSION`` is stamped into the artifact at build time and checked at
match time, so a stale dictionary is a loud failure instead of a quiet one. Bump
it whenever ``normalize_key`` changes behaviour.

Scope note
----------
This is the minimal version, written alongside the dictionary compiler so there
is one copy from the start rather than two that agree by luck — the exact
situation ``label_norm``'s docstring describes. Morphy lemmatisation, taxonomic
authorship stripping and acronym expansion arrive in Phase 2 and each will need
a version bump and a rebuild.
"""
from __future__ import annotations

import re
import unicodedata

# Bump on ANY behaviour change to normalize_key. See module docstring.
NORMALIZER_VERSION = "2"

# Unicode punctuation that is semantically whitespace in a name. Ontology labels
# and OCR'd text disagree constantly about hyphen width and quote style, and the
# difference is never meaningful: "Baetic-Rifian" vs "Baetic–Rifian" is one name.
_DASHES = "‐‑‒–—―−"
_QUOTES = "‘’‚‛“”„‟«»"

_TRANSLATE = {ord(c): "-" for c in _DASHES}
_TRANSLATE.update({ord(c): "'" for c in _QUOTES})

# Latin letters whose diacritic is not a combining mark, so NFKD leaves them
# alone. Each needs saying explicitly.
_LATIN_EXTRA = {
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ß": "ss",
    "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ı": "i",
}
_EXTRA_TABLE = str.maketrans(_LATIN_EXTRA)

_WS = re.compile(r"\s+")
# Strip punctuation only at the edges. Interior punctuation is load-bearing:
# "Baetic-Rifian" and "3,4-dichlorophenol" must not lose it.
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def normalize_key(text: str | None) -> str:
    """The lookup key. Applied identically to authority names and to mentions.

    Diacritics are folded, which is the one lossy step worth justifying.
    Biodiversity writing is full of names that appear both ways — Donana and
    Do\u00f1ana, Bialowieza and Bia\u0142owie\u017ca, Phuoc Binh and Ph\u01b0\u1edbc B\u00ecnh — and
    without folding those are simply different strings, so a mention in an
    English abstract never reaches the accented authority record. GBIF's own
    name matcher folds, and GeoNames ships an ``asciiname`` column for exactly
    this reason. The unfolded form is never destroyed: it is kept by
    ``normalize_display`` and stored as the alias name.

    NFKD (not NFKC) so that accents decompose into combining marks that can be
    dropped; the letters that carry no combining mark are handled by
    ``_LATIN_EXTRA`` first, because NFKD does not touch \u0142 or \u00f8.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).translate(_TRANSLATE)
    s = s.translate(_EXTRA_TABLE)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _WS.sub(" ", s).strip()
    s = _EDGE_PUNCT.sub("", s)
    return s.casefold()


def normalize_display(text: str | None) -> str:
    """Whitespace-tidied but otherwise untouched. What a human should read.

    Kept separate from the lookup key on purpose: the key is lossy by design,
    and a node whose only surviving name is its casefolded key is worse to read
    and impossible to cite.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).translate(_TRANSLATE)
    return _WS.sub(" ", s).strip()
