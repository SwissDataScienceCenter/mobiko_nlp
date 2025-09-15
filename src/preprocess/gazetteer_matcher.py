from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable, Any
import argparse
import json
import sys
from pathlib import Path

try:
    import ahocorasick
    HAVE_AHO = True
except Exception:
    HAVE_AHO = False

try:
    import hyperscan as hs
    HAVE_HS = True
except Exception:
    HAVE_HS = False

# -----------------------------
# Data model
# -----------------------------
@dataclass
class Rule:
    rule_id: int
    label: str
    pattern: str
    pattern_type: str = "regex"  # "phrase" or "regex"
    phrase_key: Optional[str] = None
    re_flags: int = 0
    re_obj: Optional[re.Pattern] = None

# Escapes Python's 're' understands (single-letter and common forms)
_PY_ESC_ONE = set(list("AbBdDsSwWZGAfnrtv"))  # \A \b \B \d \D \s \S \w \W \Z \G and \a \f \n \r \t \v
# NOTE: \u and \x start multi-char escapes; we won't treat them as unknown if followed by hex/digits


# Map common PCRE/other-dialect escapes to Python-compatible fragments
_DIALECT_MAP = {
    r"\h": r"[ \t]",                # horizontal whitespace
    r"\H": r"[^\t ]",               # non-horizontal whitespace (rough)
    r"\v": r"[\n\r\f\v]",           # vertical whitespace
    r"\V": r"[^\n\r\f\v]",
    r"\R": r"(?:\r\n|[\n\r\f\v])",  # any newline (approx)
    r"\l": r"[a-z]",                # lowercase letter
    r"\u": r"[A-Z]",                # uppercase letter
}

_HEX = "0123456789abcdefABCDEF"

def _looks_like_valid_unicode_escape(pat: str, i: int) -> bool:
    # \xHH or \uHHHH or \UHHHHHHHH
    if i+1 >= len(pat): return False
    c = pat[i+1]
    if c == "x" and i+3 < len(pat) and all(ch in _HEX for ch in pat[i+2:i+4]):
        return True
    if c == "u" and i+5 < len(pat) and all(ch in _HEX for ch in pat[i+2:i+6]):
        return True
    if c == "U" and i+9 < len(pat) and all(ch in _HEX for ch in pat[i+2:i+10]):
        return True
    return False


def _sanitize_pattern_for_python(pattern: str) -> str:
    """
    1) Translate a few non-Python escapes via _DIALECT_MAP.
    2) Any remaining unknown single-letter escapes -> make the backslash literal (e.g., '\q' -> '\\q').
    """
    # First pass: direct replacements (handle \h, \v, \R, \l, \u, etc.)
    for k, v in _DIALECT_MAP.items():
        if k in pattern:
            pattern = pattern.replace(k, v)

    # Second pass: scan for backslashes and fix unknown single-letter escapes
    out = []
    i = 0
    L = len(pattern)
    while i < L:
        ch = pattern[i]
        if ch != "\\":
            out.append(ch); i += 1; continue

        # Backslash at last char -> escape it to be literal
        if i == L - 1:
            out.append("\\\\"); i += 1; continue

        nxt = pattern[i+1]

        # valid unicode/hex escape? keep as-is (and consume appropriate length later by re engine)
        if _looks_like_valid_unicode_escape(pattern, i):
            out.append("\\"); i += 1; continue

        # numeric backref like \1 .. \9 ? keep as-is
        if nxt.isdigit():
            out.append("\\"); i += 1; continue

        # Known 1-letter Python escapes or bracket/paren etc. Keep as-is
        if (nxt in _PY_ESC_ONE) or (nxt in {r"\\", ".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "(", ")", "|"}):
            out.append("\\"); i += 1; continue

        # Otherwise unknown escape -> make backslash literal
        out.append("\\\\")
        i += 1
    return "".join(out)


def safe_compile_regex(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern)
    except re.error:
        fixed = _sanitize_pattern_for_python(pattern)
        try:
            return re.compile(fixed)
        except re.error as e2:
            # propagate with context
            raise RuntimeError(f"Failed to compile pattern even after sanitizing.\n"
                               f"Original: {pattern}\nSanitized: {fixed}\nError: {e2}") from e2



META_RE = re.compile(r'[.^$*+?{}\[\]()/\\|]')  # any true regex metachar (backslash too)


def parse_flags(flags_str: Optional[str]) -> int:
    """
    Very simple flag parser. Your sample uses things like 'aA' (looks like 'case insensitive').
    We'll default to IGNORECASE if 'aA' present. Extend as needed.
    """
    if not flags_str:
        return 0
    f = 0
    if "aA" in flags_str or "i" in flags_str:
        f |= re.IGNORECASE
    return f


def unwrap_word_boundary_literal(pat: str) -> Optional[str]:
    """
    If pattern is exactly a word-boundary wrapped literal (e.g., r'\bPuig\b'),
    return the inner literal ("Puig"). Otherwise, None.
    Conditions:
      - must start with \b and end with \b
      - inner must NOT contain regex metacharacters
      - no backslashes inside
    """
    if not pat.startswith(r"\b") or not pat.endswith(r"\b"):
        return None
    inner = pat[2:-2]
    # reject if any regex meta inside
    if META_RE.search(inner):
        return None
    return inner


def classify_rule(pat: str) -> Tuple[str, Optional[str], int]:
    """
    Returns (pattern_type, phrase_key, re_flags).
    pattern_type: "phrase" or "regex"
    phrase_key: if phrase, the normalized key (case-insensitive handled by lower())
    """
    literal = unwrap_word_boundary_literal(pat)
    if literal is not None:
        # Treat as phrase; we’ll do case-insensitive search by lower-casing both sides
        phrase_key = literal.lower()
        return "phrase", phrase_key
    return "regex", None


def _infer_type_from_path(p: str) -> str:
    # file name stem → Type (e.g., "mountain.csv" -> "Mountain")
    stem = Path(p).stem.replace("_", " ").strip()
    stem = stem.split('-')[-1].strip()
    return stem[:1].upper() + stem[1:]

# -----------------------------
# Loader
# -----------------------------

def load_gaz_rules_from_csv(
    path: str,
    pattern_col: str = "regexsearch",
    delimiter: str = ",",
    force_label: Optional[str] = None,   # write label with table name/type
) -> List[Rule]:

    rules: List[Rule] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rid_local = 0
        for row in reader:
            pattern = (row.get(pattern_col) or "").strip()
            if not pattern:
                continue
            # if you want to use table name as the type, set force_label
            label = force_label
            if not label:
                # fall back to filename stem if still empty
                label = _infer_type_from_path(path)
            ptype, phrase_key = classify_rule(pattern)
            re_obj = None
            if ptype == "regex":
                re_obj = safe_compile_regex(pattern)
            rules.append(Rule(
                rule_id=rid_local, label=label,
                pattern=pattern, pattern_type=ptype, re_obj=re_obj, phrase_key=phrase_key
            ))
            rid_local += 1
    return rules


def load_gaz_rules_from_dir(
    dir_path: str,
    pattern_col: str = "regexsearch",
    delimiter: str = ",",
) -> List[Rule]:

    """Walk a directory, load every CSV/TSV. Table filename is used as entity type."""
    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Gazetteer dir not found: {dir_path}")

    files = sorted(base.glob("*.csv")) + sorted(base.glob("*.tsv"))
    if not files:
        raise RuntimeError(f"No CSV/TSV gazetteer files found in {dir_path}")

    all_rules: List[Rule] = []
    rid_base = 0
    for file_path in files:
        rules = load_gaz_rules_from_csv(
            file_path,
            pattern_col=pattern_col,
            force_label=_infer_type_from_path(file_path),
            delimiter=delimiter,
        )
        for r in rules:
            r.rule_id = rid_base + r.rule_id  # unique across files
        rid_base += len(rules)
        all_rules.extend(rules)
    return all_rules


# -----------------------------
# Matcher
# -----------------------------

class GazetteerMatcher:
    """
    Phrase rules: Aho–Corasick (lowercased), with word-boundary recheck.
    Regex rules: Hyperscan multi-pattern if available, else Python 're'.
    """

    def __init__(self, rules: List[Rule], use_languages: Optional[Iterable[str]] = None):
        self.rules: List[Rule] = []
        self.rules = rules
        self._phrase_len_by_id = {}  # rule_id -> phrase length

        # Build phrase matcher
        self.phrase_rules = [r for r in rules if r.pattern_type == "phrase"]
        self.regex_rules = [r for r in rules if r.pattern_type == "regex"]


        # --- phrase backend
        self._phrase = None
        if self.phrase_rules and HAVE_AHO:
            ac = ahocorasick.Automaton()
            for r in self.phrase_rules:
                if r.phrase_key:
                    ac.add_word(r.phrase_key, (r.rule_id, r.label))  # 2-tuple payload (safe)
                    self._phrase_len_by_id[r.rule_id] = len(r.phrase_key)
            ac.make_automaton()
            self._phrase = ("aho", ac)
        else:
            self._phrase = ("scan", None)  # fallback: substring scan


        # --- regex backend
        self._hs_db = None
        self._hs_id2rule = None
        if self.regex_rules and HAVE_HS:
            # Build Hyperscan db
            exprs = []
            ids   = []
            for idx, r in enumerate(self.regex_rules):
                # Hyperscan needs bytes; we use UTF-8 and scan on the original text bytes
                exprs.append(r.pattern.encode("utf-8"))
                ids.append(idx)
            db = hs.Database()
            db.compile(expressions=exprs, ids=ids, elements=len(exprs))
            self._hs_db = db
            self._hs_id2rule = {i: r for i, r in enumerate(self.regex_rules)}


    def match(self, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        # --- phrases

        if self.phrase_rules:
            tl = text.lower()
            mode, ac = self._phrase

            if mode == "aho":
                for end_idx, payload in ac.iter(tl):
                    try:
                        rid, label, plen = payload  # 3-tuple case
                    except ValueError:
                        rid, label = payload  # 2-tuple case
                        plen = self._phrase_len_by_id.get(rid)
                        if not plen:
                            # last resort: skip (shouldn't happen if you filled _phrase_len_by_id in __init__)
                            continue

                    s = end_idx - plen + 1
                    e = end_idx + 1
                    if _is_word_boundary(tl, s, e):
                        out.append({"start_char": s, "end_char": e, "text": text[s:e],
                                    "type": label, "source": "phrase", "rule_id": rid})

            else:
                # simple scan fallback: iterate each key once; still fast for ~10k
                # to avoid O(N*M) worst-case, bucket by first char
                buckets: Dict[str, List[Tuple[Rule, str]]] = {}
                for r in self.phrase_rules:
                    if not r.phrase_key:
                        continue
                    buckets.setdefault(r.phrase_key[0], []).append((r, r.phrase_key))
                seen_first = set(ch for ch in tl)  # chars present in text
                for ch in seen_first:
                    for r, key in buckets.get(ch, []):
                        start = 0
                        while True:
                            idx = tl.find(key, start)
                            if idx < 0:
                                break
                            j = idx + len(key)
                            if _is_word_boundary(tl, idx, j):
                                out.append({"start_char": idx, "end_char": j, "text": text[idx:j],
                                            "type": r.label, "source": "phrase", "rule_id": r.rule_id})
                            start = idx + 1

        # --- regex
        if self.regex_rules:
            if self._hs_db is not None:
                # Hyperscan scan on bytes; maintain byte→char map
                b = text.encode("utf-8")
                # build byte->char offsets (cheap)
                byte2char: List[int] = []
                ch_idx = 0
                for ch in text:
                    enc = ch.encode("utf-8")
                    for _ in enc:
                        byte2char.append(ch_idx)
                    ch_idx += 1
                matches = []

                def on_match(id, from_, to_, flags, context):
                    matches.append((id, from_, to_))
                    return 0

                self._hs_db.scan(b, match_event_handler=on_match)
                for ridx, bf, bt in matches:
                    r = self._hs_id2rule[ridx]
                    cs = byte2char[bf] if bf < len(byte2char) else len(text)
                    ce = byte2char[bt-1] + 1 if bt-1 < len(byte2char) else len(text)
                    if ce > cs:
                        out.append({"start_char": cs, "end_char": ce, "text": text[cs:ce],
                                    "type": r.label, "source": "regex", "rule_id": r.rule_id})
            else:
                # Python re fallback
                # Light prefilter: only run a regex if at least one of its first literal chars is in text
                seen = set(text.lower())
                for r in self.regex_rules:
                    # Heuristic: skip obviously impossible scans if pattern starts with \bX and X not present
                    pat = r.pattern
                    first_lit = None
                    if pat.startswith(r"\b") and len(pat) > 2 and pat[2:3].isalpha():
                        first_lit = pat[2].lower()
                    if first_lit is not None and first_lit not in seen:
                        continue
                    for m in r.re_obj.finditer(text):
                        s, e = m.span()
                        if e > s:
                            out.append({"start_char": s, "end_char": e, "text": text[s:e],
                                        "type": r.label, "source": "regex", "rule_id": r.rule_id})
        return out

    def match_and_resolve(self, text: str) -> List[Dict[str, Any]]:
        return resolve_overlaps(self.match(text))


# -----------------------------
# Overlap resolver
# -----------------------------

def resolve_overlaps(
    cands: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Resolve overlapping candidates with a simple, deterministic policy:
      1) Prefer longer spans (end - start).
      2) If equal length, prefer earlier start offset.
      3) If still tied, prefer earlier end offset.
    Keep only non-overlapping spans. No type or rule priority is considered.
    """
    def span_len(c): return c["end_char"] - c["start_char"]

    # Sort once by our preference; greedy select afterward
    cands_sorted = sorted(
        cands,
        key=lambda c: (-(span_len(c)), c["start_char"], c["end_char"])
    )

    kept: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []

    for c in cands_sorted:
        s, e = c["start_char"], c["end_char"]
        if not any(_overlaps((s, e), z) for z in occupied):
            kept.append(c)
            occupied.append((s, e))

    # Optional: deduplicate identical spans with identical labels (keep first)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for c in kept:
        k = (c["start_char"], c["end_char"], c.get("label"))
        if k in seen:
            continue
        seen.add(k)
        deduped.append(c)

    # Return ordered by span start for readability
    deduped.sort(key=lambda c: (c["start_char"], c["end_char"]))
    return deduped


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _is_word_boundary(text_lower: str, s: int, e: int) -> bool:
    """
    ASCII-ish word boundary check good enough for EN/ES/CAT typical data.
    If you need full Unicode boundaries, replace with regex \b on both sides via slices.
    """
    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"
    left_ok  = (s == 0) or (not is_word_char(text_lower[s - 1]))
    right_ok = (e == len(text_lower)) or (not is_word_char(text_lower[e]))
    return left_ok and right_ok



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gazetteer+regex matcher")
    parser.add_argument("--rules", required=True,
                        help="Path to rules CSV/TSV (e.g., rules.tsv)")
    parser.add_argument("--delimiter", default="\t",
                        help="CSV delimiter. Default: TAB")
    parser.add_argument("--pattern_col", default="regexsearch",
                        help="Column name containing the regex pattern (e.g., '\\bPuig\\b')")
    parser.add_argument("--flags_col", default="flags",
                        help="Column name for regex flags (only i/m/s/x recognized). Use '-' to disable.")
    parser.add_argument("--keyword", default="keyword",
                        help="Word/phrase column to match")
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve overlaps (longest-match-wins).")
    parser.add_argument("--out", default="-",
                        help="Where to write JSONL matches. '-' = stdout (default).")

    args = parser.parse_args()

    # Load rules
    rules = load_gaz_rules_from_csv(
        path=args.rules,
        keyword=args.keyword,
        pattern_col=args.pattern_col,
        flags_col=args.flags_col,
        delimiter=args.delimiter,
    )

    matcher = GazetteerMatcher(rules)

    # Load text
    if args.text:
        text = Path(args.text).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    # Run
    matches = matcher.match_and_resolve(text) if args.resolve else matcher.match(text)

    # Emit JSONL
    if args.out == "-":
        out_f = sys.stdout
        close_out = False
    else:
        out_f = open(args.out, "w", encoding="utf-8")
        close_out = True

    for m in matches:
        out_f.write(json.dumps(m, ensure_ascii=False) + "\n")

    if close_out:
        out_f.close()
