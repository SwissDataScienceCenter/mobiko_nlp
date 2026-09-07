"""
How a corpus resolves against the compiled dictionary, broken out by entity type.

This is the plan's decision instrument: "report linked vs NIL per entity type —
a low hit rate is a finding, and it arrives early enough to change the plan."

Reading the output
------------------
Four outcomes, deliberately not collapsed into one rate:

``unrouted``   no authority serves the entity type. Not attempted, not missed.
               CONCEPT and the TEMPORAL/QUANTITATIVE types read 0% by
               construction, and pricing the remaining work depends on not
               confusing that with failure.
``nil``        routed, searched, nothing in the vocabulary. This is the real
               ontology coverage gap, and its frequency ranking is the input to
               NIL clustering.
``ambiguous``  candidates found, none choosable. A different problem needing a
               different fix — a reranker, not a bigger vocabulary.
``linked``     one concept survived the policy.

It measures the Tier A floor: exact and alias lookup with deterministic
disambiguation, no fuzzy matching, no morphology, no retrieval, no LLM. Every
later tier is scored as an improvement on this.

Usage
-----
    python -m src.linking.coverage_report
    python -m src.linking.coverage_report --input run.jsonl --top 40
    python -m src.linking.coverage_report --show-ambiguous
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

from src.linking.authorities.registry import ARTIFACT_DIR
from src.linking.dictionary import Dictionary
from src.linking.match import Outcome, resolve
from src.linking import wordnet_guard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multi_agent_annotation"))
from label_norm import canonicalize_entity_type  # noqa: E402

DEFAULT_INPUTS = [
    Path("data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"),
    Path("data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl"),
]


def iter_mentions(path: Path) -> Iterator[Tuple[str, str]]:
    """(text, raw type) from an annotation file.

    Handles both shapes in use: records holding a ``sentences`` list and flat
    per-sentence records; span type lives under ``type`` in the human files and
    ``entity_type`` in agent output.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sentences = obj.get("sentences")
            blocks = sentences if isinstance(sentences, list) else [obj]
            for s in blocks:
                if not isinstance(s, dict):
                    continue
                for sp in (s.get("spans") or s.get("entities") or []):
                    if not isinstance(sp, dict):
                        continue
                    text = (sp.get("text") or "").strip()
                    if text:
                        yield text, (sp.get("type") or sp.get("entity_type") or "")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ARTIFACT_DIR)
    ap.add_argument("--input", type=Path, action="append", dest="inputs")
    ap.add_argument("--top", type=int, default=20,
                    help="how many NIL mentions to list (0 to suppress)")
    ap.add_argument("--show-ambiguous", action="store_true",
                    help="list the ambiguous mentions and why each was declined")
    args = ap.parse_args(argv)

    try:
        d = Dictionary.open(args.dir)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"artifacts : {', '.join(d.present)}"
          + (f"   MISSING: {', '.join(d.missing_groups)}" if d.missing_groups else ""))
    print(f"wordnet   : {'available' if wordnet_guard.available() else 'UNAVAILABLE — '
                          'genericness guard is OFF'}")
    print()

    inputs = args.inputs or DEFAULT_INPUTS
    counts: collections.Counter = collections.Counter()
    for path in inputs:
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 1
        for text, raw_type in iter_mentions(path):
            counts[(text, canonicalize_entity_type(raw_type))] += 1

    agg: collections.Counter = collections.Counter()
    nils: collections.Counter = collections.Counter()
    ambiguous: List[Tuple[int, str, str, str]] = []

    for (text, etype), n in counts.items():
        r = resolve(d, text, etype)
        agg[(etype, "m")] += n
        agg[(etype, "u")] += 1
        agg[(etype, r.outcome.value)] += n
        if r.outcome is Outcome.NIL:
            nils[(r.norm, etype)] += n
        elif r.outcome is Outcome.AMBIGUOUS:
            ambiguous.append((n, r.norm, etype, r.reason))

    # Total ordering. Sorting a set by count alone leaves ties to set iteration
    # order, which moves with PYTHONHASHSEED: the five one-mention types
    # reshuffled between runs, so the report was not byte-reproducible even
    # though every number in it was.
    types = sorted({t for t, _ in agg}, key=lambda t: (-agg[(t, "m")], t))
    cols = ("linked", "ambiguous", "nil", "unrouted")
    print(f"{'entity type':24s} {'ment':>6s} {'uniq':>5s} "
          + "".join(f"{c:>15s}" for c in cols))
    print("-" * 86)
    tot: collections.Counter = collections.Counter()
    for t in types:
        m = agg[(t, "m")]
        tot["m"] += m
        tot["u"] += agg[(t, "u")]
        row = f"{t or '(untyped)':24s} {m:6d} {agg[(t,'u')]:5d} "
        for c in cols:
            v = agg[(t, c)]
            tot[c] += v
            row += f"{v:8d} ({100*v/m:4.1f}%)" if v else f"{'-':>15s}"
        print(row)
    print("-" * 86)
    row = f"{'TOTAL':24s} {tot['m']:6d} {tot['u']:5d} "
    for c in cols:
        row += f"{tot[c]:8d} ({100*tot[c]/tot['m']:4.1f}%)"
    print(row)

    routed = tot["m"] - tot["unrouted"]
    if routed:
        print(f"\nof the {routed} routed mentions: {100*tot['linked']/routed:.1f}% linked, "
              f"{100*tot['ambiguous']/routed:.1f}% ambiguous, "
              f"{100*tot['nil']/routed:.1f}% NIL")

    if args.show_ambiguous and ambiguous:
        print(f"\n=== ambiguous ({len(ambiguous)} forms, "
              f"{sum(a[0] for a in ambiguous)} mentions) ===")
        for n, norm, etype, reason in sorted(
                ambiguous, key=lambda a: (-a[0], a[1], a[2]))[:25]:
            print(f"  {n:4d}  {norm[:30]:30s} {etype[:18]:18s} {reason[:60]}")

    if args.top:
        print(f"\n=== top {args.top} NIL, by mention frequency ===")
        print("(this ranking IS the ontology coverage gap; it is the input to "
              "NIL clustering)")
        for (norm, etype), n in sorted(
                nils.items(), key=lambda kv: (-kv[1], kv[0]))[:args.top]:
            print(f"  {n:4d}  {norm[:36]:36s} {etype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
