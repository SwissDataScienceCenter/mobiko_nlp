"""
decision_table.py — apply guideline amendments to the Annotator's decision table.

The reconstruction loop drafts ONE amendment per confusion (guideline_amender),
whose ``decision_test`` is an operational, checkable if-then cue naming the
competing labels. That is exactly the content the decision table's ``Question``
column wants, so we apply the SAME amendments to BOTH artifacts each iteration:

  * G_i → G_{i+1}: the amender appends a narrative rule section (markdown).
  * D_i → D_{i+1}: here — we inject each accepted decision_test into the Question
    column of the rows for the two labels it disambiguates.

Append-only (mirrors the guideline's "append accepted only"): an injected test
is added to a row's Question unless already present (substring dedup). The cold
start (D0) has a blank Question for every type, so iteration 1 fills them in.

Rows are matched to labels case-insensitively / whitespace-normalised. Labels
with no matching row (e.g. relation/scope labels under --patterns all) are
skipped — only existing entity-type rows are amended.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_DEFAULT_COLUMNS = ["LABEL", "Question", "Examples", "Definition", "Comment"]


def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip()).upper()


def load_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read the decision-table CSV → (fieldnames, rows)."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or _DEFAULT_COLUMNS)
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def write_table(fieldnames: Sequence[str], rows: Sequence[Dict[str, str]], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            # Restrict to known columns; default any missing to "".
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _inject(row: Dict[str, str], decision_test: str) -> bool:
    """Append ``decision_test`` to a row's Question unless already present.

    Returns True if the row changed.
    """
    test = (decision_test or "").strip()
    if not test:
        return False
    current = (row.get("Question") or "").strip()
    if test in current:                      # substring dedup — already there
        return False
    row["Question"] = f"{current}\n{test}".strip() if current else test
    return True


def apply_amendments(
    rows: List[Dict[str, str]],
    accepted: Sequence[Tuple[str, str, str]],
) -> int:
    """Inject each accepted decision_test into the rows for its two labels.

    ``accepted`` is a sequence of (annotator_label, critic_label, decision_test).
    Mutates ``rows`` in place; returns the number of (row, test) injections made.
    """
    by_label: Dict[str, Dict[str, str]] = {_norm(r.get("LABEL", "")): r for r in rows}
    n = 0
    for annotator_label, critic_label, decision_test in accepted:
        for lbl in (annotator_label, critic_label):
            row = by_label.get(_norm(lbl))
            if row is not None and _inject(row, decision_test):
                n += 1
    return n


def write_amended_table(
    in_path: Path,
    out_path: Path,
    accepted: Sequence[Tuple[str, str, str]],
) -> int:
    """Load D_i, apply accepted amendments, write D_{i+1}; return injection count."""
    fieldnames, rows = load_table(in_path)
    n = apply_amendments(rows, accepted)
    write_table(fieldnames, rows, out_path)
    return n