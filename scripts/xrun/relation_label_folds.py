#!/usr/bin/env python3
"""
What the still-unmapped custom relation labels cost Mark–Davnah agreement —
input for an annotator decision, not a scoring rule.

Prints the label-view ladder (raw → migration → synonyms → families), then folds
ONE candidate label at a time into an existing family on top of the `families`
view and reports the kappa given a co-detected pair, then all folds together.

The folds are NOT applied to FAMILY_MAP: folding two labels together is a claim
about what they MEAN, which is the annotators' call, not the metric's. Re-run
after any change to the merged files.

    python scripts/xrun/relation_label_folds.py \
        -o output/eval_reports/relation_iaa/label_fold_sensitivity.txt
"""

from __future__ import annotations

import argparse
import datetime
import io
import sys
from pathlib import Path
from typing import Dict, Tuple

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("src/multi_agent_annotation", "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(_ROOT / _p))

import eval_relation_iaa as RI  # noqa: E402

MARK = _ROOT / "data/aug_runs/Mark_relation_input_relation/Mark/relations_merged.jsonl"
DAVNAH = _ROOT / "data/aug_runs/Davnah_relation_input_relation/Davnah/relations_merged.jsonl"

# label → family it would join. Candidates only; see the module docstring.
# Already decided and applied in the merged files (merge_relation_annotations.py
# --relabel), so not candidates: OCCURS_IN → LOCATED_IN, IS_PART_OF → PART_OF.
CANDIDATES: Dict[str, str] = {
    "SUBSET_OF": "PART_WHOLE",
    "IS_INSTANCE_OF": "PART_WHOLE", "CONTAINS": "PART_WHOLE",
    "CONNTAINS": "PART_WHOLE",
    "HAS_STATE": "PROPERTY", "HAS_VALUE": "PROPERTY",
    "INDICATES": "ASSOCIATION", "ASSOCIATED_WITH": "ASSOCIATION",
    "DEPENDS_ON": "ASSOCIATION", "REQUIRES": "ASSOCIATION",
    "GREATER_THAN": "COMPARISON", "RESPONDS_TO": "CAUSAL", "ENABLES": "CAUSAL",
    "STARTS": "TEMPORAL", "ENDS": "TEMPORAL",
    "HAS_BEGINNING": "TEMPORAL", "HAS_END": "TEMPORAL",
}


def report(sa, sb, name_a: str, name_b: str, mode: str) -> str:
    out = io.StringIO()
    p = lambda *a: print(*a, file=out)  # noqa: E731

    def score(view: str = "families") -> Tuple[float, float, float, dict]:
        r = RI.compute(sa, sb, name_a, name_b, view=view, primary_mode=mode)
        lg = r["labeling_given_detection"]
        return lg["cohen_kappa"], lg["observed_agreement"], r["decomposition"]["labeled_f1"], r

    p(f"LABEL-FOLD SENSITIVITY — {name_a} vs {name_b} relation labels")
    p("generated", datetime.date.today().isoformat())
    p(f"""
Labeling agreement conditioned on a co-detected pair (argument match: {mode}).
The view ladder first, then what each still-unmapped custom label would be worth
if it were folded into an existing family. The folds below are NOT in FAMILY_MAP:
they are ontological decisions for the annotators. 'all together' is the ceiling
if every one of them is accepted.
""")
    p("  view ladder")
    for v in RI.LABEL_VIEWS:
        kk, obs, f1, r = score(v)
        p(f"    {v:<11} kappa={kk:.3f}  observed={obs:.3f}  triplet F1={f1:.3f}")

    saved = dict(RI.FAMILY_MAP)
    try:
        base_k = score()[0]
        p(f"\n  candidate folds, each on top of the families view (kappa={base_k:.3f})\n")
        rows = []
        for lab, fam in CANDIDATES.items():
            RI.FAMILY_MAP.clear(); RI.FAMILY_MAP.update(saved); RI.FAMILY_MAP[lab] = fam
            kk = score()[0]
            rows.append((kk - base_k, lab, fam, kk))
        for d, lab, fam, kk in sorted(rows, reverse=True):
            p(f"  {lab:>16} → {fam:<11}  kappa={kk:.3f}  ({d:+.3f})")
        RI.FAMILY_MAP.clear(); RI.FAMILY_MAP.update(saved); RI.FAMILY_MAP.update(CANDIDATES)
        kk, obs, f1, _ = score()
        p(f"\n  all together:  kappa={kk:.3f}  observed={obs:.3f}  triplet F1={f1:.3f}")
    finally:
        RI.FAMILY_MAP.clear(); RI.FAMILY_MAP.update(saved)

    p("""
Reading: OCCURS_IN -> LOCATED_IN and IS_PART_OF -> PART_OF are already applied
in the merged files. What is left is mainly whether the subsumption labels
(SUBSET_OF, IS_INSTANCE_OF, CONTAINS) belong with PART_OF. Settle that in the
guideline and the relation vocabulary stops being the bottleneck; the entity
layer still is.""")
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=Path, default=MARK)
    ap.add_argument("--b", type=Path, default=DAVNAH)
    ap.add_argument("--name-a", default="Mark")
    ap.add_argument("--name-b", default="Davnah")
    ap.add_argument("--primary-mode", choices=RI.ARG_MODES, default="normalized")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    text = report(RI.load_doc(args.a), RI.load_doc(args.b),
                  args.name_a, args.name_b, args.primary_mode)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
