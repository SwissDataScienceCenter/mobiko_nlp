#!/usr/bin/env python
"""
Every run in a multi-agent output directory, scored against the human relation
gold on entities, relation pairs and triplets — one report per run plus the
matrix that makes the runs comparable.

Driver only: all scoring lives in
src/multi_agent_annotation/evaluation/eval_relation_agent.py, which in turn uses
the matching engine of eval_relation_iaa.py, so a number in the matrix means
exactly what the same-named number means in the human IAA report.

The matrix carries the human ceiling as its last row (Mark vs Davnah on the same
sentences, symmetric F1). Read every agent row against it, not against 1.0: the
two annotators score 0.607 on entities and 0.192 on triplets, so a pipeline at
0.19 triplet F1 is at the human ceiling, not failing.

    python scripts/xrun/relation_agent_grid.py
    python scripts/xrun/relation_agent_grid.py --label-view families
    python scripts/xrun/relation_agent_grid.py --runs multi_pass_kimi_1 multi_pass_kimi_2
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path("/home/katinska/mobiko_nlp")
for _p in ("src/multi_agent_annotation", "src/multi_agent_annotation/evaluation"):
    sys.path.insert(0, str(ROOT / _p))

import eval_relation_agent as RA      # noqa: E402
import eval_relation_iaa as RI        # noqa: E402

RUN_DIR = ROOT / "data/auto_annotated_clean_multi_run"
GOLD = {
    "Mark": ROOT / "data/aug_runs/Mark_relation_input_relation/Mark/relations_merged.jsonl",
    "Davnah": ROOT / "data/aug_runs/Davnah_relation_input_relation/Davnah/relations_merged.jsonl",
}
OUT = ROOT / "output/eval_reports/relation_agent"


def run_meta(path: Path) -> dict:
    with path.open(encoding="utf8") as f:
        for line in f:
            if line.strip().startswith("{"):
                return json.loads(line).get("run_meta") or {}
    return {}


def _n(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--runs", nargs="*", default=None, help="Run file names (default: all)")
    ap.add_argument("--label-view", choices=RI.LABEL_VIEWS, default="raw")
    ap.add_argument("--primary-mode", choices=RI.ARG_MODES, default="boundary")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    golds = {name: RA.load_gold(p) for name, p in GOLD.items()}
    names = list(golds)
    consensus_name = f"Consensus ({'∩'.join(names)})"
    consensus = RA.build_consensus([golds[n] for n in names],
                                   args.primary_mode, args.label_view)

    run_paths = sorted(p for p in args.run_dir.iterdir() if p.is_file())
    if args.runs:
        wanted = set(args.runs)
        run_paths = [p for p in run_paths if p.name in wanted]
    if not run_paths:
        sys.exit(f"no runs found in {args.run_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in run_paths:
        agent = RA.load_agent(path)
        refs = [(n, golds[n], None) for n in names]
        refs.append((consensus_name, consensus, (golds[names[0]], golds[names[1]])))
        results = {n: RA.score(agent, g, args.label_view, args.primary_mode, gold_pair=pair)
                   for n, g, pair in refs}

        buf = io.StringIO()
        with redirect_stdout(buf):
            RA.print_report(path.name, results, args.label_view, args.primary_mode)
        text = buf.getvalue()
        print(text, end="")
        (args.out_dir / f"{path.name}.{args.label_view}.txt").write_text(text, encoding="utf8")
        (args.out_dir / f"{path.name}.{args.label_view}.json").write_text(
            json.dumps({"run": path.name, "run_meta": run_meta(path),
                        "label_view": args.label_view, "primary_mode": args.primary_mode,
                        "references": {
                            k: dict({kk: vv for kk, vv in v.items()
                                     if not kk.startswith("_")},
                                    label_confusion={f"{a} -> {b}": c for (a, b), c
                                                     in v["_confusion"].most_common()})
                            for k, v in results.items()}},
                       indent=2, ensure_ascii=False), encoding="utf8")

        meta = run_meta(path)
        rows.append({
            "run": path.name,
            "model": meta.get("annotator_model", "?"),
            "critic": meta.get("critic_model", "?"),
            "n": results[names[0]]["scope"]["covered"],
            "res": results,
        })

    # ── the matrix ──
    lines = []
    lines.append("=" * 118)
    lines.append(f"  MULTI-AGENT PIPELINE vs HUMAN RELATION GOLD — all runs"
                 f"   (label view: {args.label_view}, argument match: {args.primary_mode})")
    lines.append("=" * 118)
    lines.append("  F1 against each gold. 'ceil' = share of that gold's relations whose both")
    lines.append("  endpoints the agent found — recall cannot exceed it. 'κ|pair' = label")
    lines.append("  agreement once both linked the same pair.")
    for ref in names + [consensus_name]:
        lines.append("")
        lines.append(f"  gold: {ref}")
        lines.append(f"    {'run':<32}{'n':>4}{'ent.bnd':>9}{'ent.str':>9}"
                     f"{'pairs':>8}{'triplets':>10}{'ceil':>8}{'κ|pair':>9}")
        for r in sorted(rows, key=lambda r: -(r["res"][ref]["triplets"][args.primary_mode]["f1"] or 0)):
            d = r["res"][ref]
            lines.append(
                f"    {r['run']:<32}{r['n']:>4}"
                f"{_n(d['entities']['boundary']['f1']):>9}"
                f"{_n(d['entities']['strict']['f1']):>9}"
                f"{_n(d['pairs'][args.primary_mode]['f1']):>8}"
                f"{_n(d['triplets'][args.primary_mode]['f1']):>10}"
                f"{_n(d['argument_ceiling'][args.primary_mode]['recall_ceiling']):>8}"
                f"{_n(d['labeling_given_pair']['cohen_kappa']):>9}")

    # human ceiling, computed here so it cannot drift from the agent rows
    iaa = RI.compute(RI.load_doc(GOLD["Mark"]), RI.load_doc(GOLD["Davnah"]),
                     "Mark", "Davnah", view=args.label_view,
                     primary_mode=args.primary_mode)
    lines.append("")
    lines.append("  HUMAN CEILING — Mark vs Davnah, same sentences, symmetric F1")
    lines.append(f"    {'Mark vs Davnah':<32}{iaa['coverage']['doubly_annotated']:>4}"
                 f"{0.607:>9.3f}{0.493:>9.3f}"
                 f"{iaa['detection_f1'][args.primary_mode]['f1']:>8.3f}"
                 f"{iaa['labeled_f1'][args.primary_mode]['f1']:>10.3f}"
                 f"{iaa['argument_ceiling'][args.primary_mode]['ceiling_f1']:>8.3f}"
                 f"{iaa['labeling_given_detection']['cohen_kappa']:>9.3f}")
    lines.append("    (entity F1s from scripts/human_iaa_report.py on the same two files)")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    (args.out_dir / f"MATRIX.{args.label_view}.txt").write_text(text + "\n", encoding="utf8")
    print(f"  written to {args.out_dir}/MATRIX.{args.label_view}.txt")


if __name__ == "__main__":
    main()
