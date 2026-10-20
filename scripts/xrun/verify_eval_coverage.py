#!/usr/bin/env python
"""
Verify that every run in data/auto_annotated_clean was actually evaluated.

Checking that a report FILE exists is not enough — a stale report from an
earlier version of the same run, or one copied between directories, would pass
that test. Every eval stamps `provenance.agent_jsonl_sha256` into its own
output, so the real check is that the hash the report recorded matches the hash
of the file on disk right now. Anything else means the report describes
different bytes than the run it is filed under.

    python scripts/xrun/verify_eval_coverage.py
    python scripts/xrun/verify_eval_coverage.py --quiet   # summary only

Exit status is non-zero if any run is missing an eval or carries a stale hash,
so this can gate an analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
DATA = ROOT / "data" / "auto_annotated_clean"
REPORTS = ROOT / "output" / "eval_reports_clean"

# The eight evals run_evals.sh drives, by report stem. eval_relation_iaa is not
# here on purpose: it compares two ANNOTATOR files for relation agreement and
# takes no --agent-jsonl, and the human relation annotation is unfinished.
STEMS = ["layer1", "layer2", "layer34", "history",
         "guideline", "diffmodel", "diffsplit", "logprob"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in DATA.iterdir() if p.is_file())
    problems: list[str] = []
    by_hash: dict[str, list[str]] = {}

    if not args.quiet:
        print(f"{'run':42s} {'recs':>5s} {'evals':>6s} {'hash ok':>8s}  note")

    for f in files:
        digest = sha256(f)
        by_hash.setdefault(digest, []).append(f.name)
        recs = sum(1 for line in f.open() if line.strip().startswith("{"))
        d = REPORTS / f.name

        present, matched, notes = 0, 0, []
        for stem in STEMS:
            rp = d / f"{stem}.json"
            if not rp.exists():
                notes.append(f"missing {stem}")
                continue
            present += 1
            try:
                rec = json.loads(rp.read_text()).get("provenance", {}).get(
                    "agent_jsonl_sha256")
            except Exception:
                notes.append(f"unreadable {stem}")
                continue
            if rec == digest:
                matched += 1
            else:
                notes.append(f"STALE {stem}")

        ok = present == len(STEMS) and matched == len(STEMS)
        if not ok:
            problems.append(f"{f.name}: {', '.join(notes) or 'incomplete'}")
        if not args.quiet:
            print(f"{f.name[:42]:42s} {recs:5d} {present:>3d}/{len(STEMS):<2d} "
                  f"{matched:>4d}/{len(STEMS):<3d} {'; '.join(notes)}")

    dupes = {h: v for h, v in by_hash.items() if len(v) > 1}
    print(f"\nruns on disk: {len(files)}   fully evaluated and hash-matched: "
          f"{len(files) - len(problems)}")
    if dupes:
        print("\nbyte-identical files (same sha256 — same content, same results):")
        for v in dupes.values():
            print(f"  {'  ==  '.join(v)}")
        print("  Evaluating both is harmless; POOLING both would fake a replicate")
        print("  and shrink the standard errors. clean_grid.CLEAN lists one of each.")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nevery run has all 8 evals, and every report's recorded input hash")
    print("matches the file on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
