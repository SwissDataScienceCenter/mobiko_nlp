#!/usr/bin/env python3
"""
prune_failed_records.py
-----------------------
Delete ENGINEERING failures from a run's JSONL so `--resume` re-attempts those
sentences.

Why this script has to exist
----------------------------
The pipeline has two failure modes that behave oppositely on resume
(multi_agent_annotation_ag2.py):

  - a hard exception writes NO record, so `--resume` retries the sentence
    by itself (:2950);
  - a parse failure WRITES a record with zero entities and status
    `annotator_parse_failed` (:2620). Resume decides what is done by sentence
    text present in the output file (:2875, :2904), so such a record marks the
    sentence permanently done. Resume will never retry it.

Deleting those records is therefore the only way to make resume try again.

What it does NOT delete, and why that matters
---------------------------------------------
"Zero entities" is not the same thing as "failed". A record can be empty
because the pipeline ran to completion and the model returned nothing — status
`constrained`, not a parse failure. That is a genuine recall miss and a real
result: the human gold for those sentences has entities.

Deleting them and re-running would resample a model failure until it scores
better, which inflates the run. So selection is by STATUS, not by emptiness.
`--include-empty` overrides this and is deliberately loud about it.

deepseek_dep (2026-08-11) is the worked example: 28 empty records, of which 24
are `annotator_parse_failed` (delete these) and 4 are `constrained` with gold
entities present in both annotators' files (keep these).

Backups
-------
A run JSONL is a write-once record — earlier runs in this programme were lost
by in-place edits. A timestamped backup is always written next to the original
and the script refuses to clobber an existing one.

Usage
-----
    # see what would go, change nothing
    python scripts/prune_failed_records.py <run.jsonl> --dry-run

    # delete annotator parse failures (default) and back up
    python scripts/prune_failed_records.py <run.jsonl>

    # also drop adjudicator parse failures
    python scripts/prune_failed_records.py <run.jsonl> \
        --status annotator_parse_failed adjudicator_parse_failed

Re-run afterwards with the SAME output path and `--resume`; only the deleted
sentences are re-attempted. Fix the cause first (the annotator gets exactly one
repair attempt, :2608-2618) or the same sentences fail the same way.
"""

import argparse
import datetime
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Statuses that mean "the pipeline broke", as opposed to "the model answered
# and the answer was empty".
DEFAULT_DROP_STATUSES = ("annotator_parse_failed",)


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            sys.exit(f"error: {path}:{n} is not valid JSON ({exc}). "
                     f"Refusing to rewrite a file that cannot be read back.")
    return records


def n_entities(rec: Dict) -> int:
    return len(rec.get("final_entities") or [])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete engineering-failure records from a run JSONL so "
                    "--resume re-attempts those sentences.")
    ap.add_argument("run_jsonl", type=Path, help="The run's output JSONL")
    ap.add_argument("--status", nargs="+", default=list(DEFAULT_DROP_STATUSES),
                    help=f"adjudication_status values to delete "
                         f"(default: {' '.join(DEFAULT_DROP_STATUSES)})")
    ap.add_argument("--include-empty", action="store_true",
                    help="ALSO delete every zero-entity record whatever its "
                         "status. Re-running those resamples genuine recall "
                         "misses — only for a run whose empties are all known "
                         "to be defects")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be deleted and exit")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip the backup copy (not recommended)")
    args = ap.parse_args()

    path: Path = args.run_jsonl
    if not path.is_file():
        sys.exit(f"error: no such file: {path}")

    records = load_records(path)
    drop_statuses = set(args.status)

    keep: List[Dict] = []
    dropped: List[Dict] = []
    for rec in records:
        status = rec.get("adjudication_status")
        is_target = status in drop_statuses or (args.include_empty
                                               and n_entities(rec) == 0)
        (dropped if is_target else keep).append(rec)

    print(f"{path}")
    print(f"  records:            {len(records)}")
    print(f"  unique sentences:   {len({r.get('sentence') for r in records})}")
    print(f"  status breakdown:   "
          f"{dict(Counter(r.get('adjudication_status') for r in records))}")
    print(f"  zero-entity:        {sum(1 for r in records if n_entities(r) == 0)}")
    print()
    print(f"  to delete:          {len(dropped)}  "
          f"({', '.join(sorted(drop_statuses))}"
          f"{' + any zero-entity' if args.include_empty else ''})")
    print(f"  to keep:            {len(keep)}")

    # An empty record the run is keeping is a real result, not an oversight —
    # name it so nobody has to wonder whether the prune missed something.
    kept_empty = [r for r in keep if n_entities(r) == 0]
    if kept_empty:
        print()
        print(f"  KEPT {len(kept_empty)} zero-entity record(s) — the pipeline "
              f"completed on these, so they are recall misses, not defects:")
        for rec in kept_empty:
            print(f"    [{rec.get('adjudication_status')}] "
                  f"{(rec.get('sentence') or '')[:72]}")

    if not dropped:
        print("\nNothing to delete.")
        return

    if args.dry_run:
        print("\n--dry-run: no changes written.")
        return

    if not args.no_backup:
        stamp = datetime.date.today().isoformat()
        backup = path.with_name(f"{path.name}.pre_prune_{stamp}.bak")
        if backup.exists():
            sys.exit(f"error: backup already exists: {backup}\n"
                     f"       Move it aside first — a run JSONL is a "
                     f"write-once record and this would overwrite the only "
                     f"copy of the pre-prune state.")
        shutil.copy2(path, backup)
        print(f"\n  backup:             {backup}")

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in keep:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)

    print(f"  wrote:              {path}  ({len(keep)} record(s))")
    print()
    print(f"Re-run with the same --output path and --resume to re-attempt the "
          f"{len(dropped)} deleted sentence(s).")
    print("Fix the repair path first, or they will fail the same way.")


if __name__ == "__main__":
    main()
