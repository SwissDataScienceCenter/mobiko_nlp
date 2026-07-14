"""
cold_start_init.py — generate the cold-start artifacts G0 and D0 (RQ-D, spec 11.1).

The cold start is, by definition:

    "entity type names + one-line definitions ONLY. No disambiguation rules, no
     decision trees, no worked examples, no 'handling ambiguous cases' section."

This script materialises that scaffold from the canonical schema so the cold
start is reproducible and reviewable (the Appendix-H artifact), for BOTH inputs
the multi-agent system reads:

  * G0.md  — the narrative guideline (Critic & Adjudicator): a heading + one-line
             definition per type, nothing more.
  * D0.csv — the decision table (Annotator): one row per type with the LABEL and
             one-line Definition; the Question (decision cues) and Examples
             (worked examples) columns are deliberately BLANK — those carry the
             expert disambiguation knowledge the loop must reconstruct.

The one-line definitions come from ``SCHEMA_BIODIV_SHORT``; the trailing "e.g. …"
illustrations are stripped so no worked examples leak into the cold start.

Usage:
  python cold_start_init.py --out-dir ./output/reconstruction_run1/coldstart
  # → writes G0.md and D0.csv there.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List

_THIS_DIR = Path(__file__).resolve().parent          # …/multi_agent_annotation/loop
_SRC = _THIS_DIR.parent.parent                        # …/src (for resources_updated)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from resources_updated.entity_schema import SCHEMA_BIODIV_SHORT, SCHEMA_BIODIV_LIST

# Decision-table columns, matching the existing Decision_support.csv schema.
_DECISION_COLUMNS = ["LABEL", "Question", "Examples", "Definition", "Comment"]

# CONCEPT is absent from SCHEMA_BIODIV_SHORT; supply a clean one-line definition.
_FALLBACK_DEFINITIONS = {
    "CONCEPT": "An abstract or theoretical construct used in analysis or discourse.",
}


def _strip_examples(definition: str) -> str:
    """Drop any trailing 'e.g. …' illustration and tidy punctuation.

    Spec 11.1 forbids worked examples in the cold start, so 'A non-living object,
    e.g., rock, glacier' becomes 'A non-living object'.
    """
    d = re.split(r"[,(]?\s*e\.g\.", definition, maxsplit=1)[0]
    d = d.strip().rstrip(",(.) ").strip()
    if d and not d.endswith("."):
        d += "."
    return d


def parse_one_line_definitions() -> Dict[str, str]:
    """Parse SCHEMA_BIODIV_SHORT → {TYPE (upper): one-line definition}.

    Every type in SCHEMA_BIODIV_LIST is covered; types missing from the SHORT
    block (CONCEPT) fall back to ``_FALLBACK_DEFINITIONS``.
    """
    defs: Dict[str, str] = {}
    line_re = re.compile(r"^(.+?)\s*\(DEFINITION:\s*(.*)$", re.IGNORECASE)
    for raw in SCHEMA_BIODIV_SHORT.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = line_re.match(line)
        if not m:
            continue
        label = re.sub(r"\s+", " ", m.group(1).strip()).upper()
        defs[label] = _strip_examples(m.group(2))

    out: Dict[str, str] = {}
    for t in SCHEMA_BIODIV_LIST:
        key = re.sub(r"\s+", " ", t.strip()).upper()
        out[key] = defs.get(key) or _FALLBACK_DEFINITIONS.get(key, "")
        if not out[key]:
            raise ValueError(f"No one-line definition available for type {key!r}.")
    return out


def build_g0_markdown(definitions: Dict[str, str], today: str = "") -> str:
    """A minimal cold-start guideline: a heading + one-line def per type."""
    header = "# MoBiKo label guidance — G0 (cold start)\n"
    note = (
        "_Cold-start scaffold: entity type names and one-line definitions only. "
        "No disambiguation rules, decision trees, or worked examples — these are "
        "what the reconstruction loop must rediscover._\n"
    )
    if today:
        note += f"\n_Generated {today}._\n"
    lines: List[str] = [header, note, "\n## Entity types\n"]
    for t in SCHEMA_BIODIV_LIST:
        key = re.sub(r"\s+", " ", t.strip()).upper()
        lines.append(f"### {t}\n")
        lines.append(f"{definitions[key]}\n")
    return "\n".join(lines).rstrip() + "\n"


def build_d0_rows(definitions: Dict[str, str]) -> List[Dict[str, str]]:
    """One decision-table row per type: LABEL + Definition only (Question/Examples blank)."""
    rows: List[Dict[str, str]] = []
    for t in SCHEMA_BIODIV_LIST:
        key = re.sub(r"\s+", " ", t.strip()).upper()
        rows.append({
            "LABEL": t.lower(),          # match the existing CSV's lowercase convention
            "Question": "",              # decision cues — to be reconstructed
            "Examples": "",              # worked examples — none at cold start
            "Definition": definitions[key],
            "Comment": "",
        })
    return rows


def write_d0_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_DECISION_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def generate(out_dir: Path, today: str = "") -> Dict[str, Path]:
    """Write G0.md and D0.csv into ``out_dir``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    definitions = parse_one_line_definitions()
    g0 = out_dir / "G0.md"
    d0 = out_dir / "D0.csv"
    g0.write_text(build_g0_markdown(definitions, today=today), encoding="utf-8")
    write_d0_csv(build_d0_rows(definitions), d0)
    return {"g0": g0, "d0": d0}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate cold-start G0.md + D0.csv (spec 11.1).")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    paths = generate(args.out_dir)
    print(f"Wrote cold-start guideline:     {paths['g0']}")
    print(f"Wrote cold-start decision table: {paths['d0']}")


if __name__ == "__main__":
    main()