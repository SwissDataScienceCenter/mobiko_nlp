#!/usr/bin/env python3
"""
Layer 1 for RELATIONS — the multi-agent pipeline scored against human gold on
entities, relation pairs and full triplets, with the agent as the prediction and
each annotator (and their consensus) as gold.

This is the relation counterpart of eval_layer1_output.py, and it borrows the
matching engine of eval_relation_iaa.py so that a number here and a number in
the human IAA report mean the same thing. What differs is the direction: IAA is
symmetric (2·agree / (n_A + n_B)), while here the agent predicts and the human
is gold, so PRECISION and RECALL are reported separately — they are what tells
an over-generating pipeline from a conservative one, and the pipeline
over-generates badly (≈1030 relations against Mark's 873 / Davnah's 906).

Three layers, all with the same offset-based argument matching:

  entities        did the agent mark the span?            (boundary / strict)
  relation pairs  did it link the same two entities?      (label ignored)
  triplets        …and give the pair the same label?

and the diagnostic that makes the triplet number readable:

  argument ceiling   the share of gold relations whose BOTH endpoints exist in
                     the agent's entity layer. A relation whose arguments the
                     agent never found cannot be recovered, so recall above this
                     is unreachable — the same ceiling logic as the human IAA.
  labeling | pair    accuracy and Cohen's kappa on the label, conditioned on a
                     pair both sides linked. The only number that isolates "does
                     it choose the right relation type".

Gold references, mirroring eval_layer1_output:
  * each annotator separately,
  * their CONSENSUS — entities both marked, and relations both linked with the
    same label. Small and high-precision; agreement with it is the strongest
    claim available from two annotators.

Scope. A sentence counts when the gold has something to find there and the run
covers it (two runs stop early, at 135 and 106 of the 155 sentences). A sentence
the agent left empty is recall loss, not an excluded sentence — that is the
difference from the IAA convention, which never scores singly-annotated work.

Hyper-relations. The pipeline emits only flat relations, while the humans nest
heavily (Mark 347, Davnah 139). Scoring those as misses would say more about
the output schema than about the pipeline, so they are EXCLUDED from the gold
pool and reported as a separate count — the honest ceiling on what this output
format can express.

Usage
-----
    python eval_relation_agent.py \
        --agent-jsonl data/auto_annotated_clean_multi_run/multi_pass_deepseek_1 \
        --human Mark=data/aug_runs/Mark_relation_input_relation/Mark/relations_merged.jsonl \
        --human Davnah=data/aug_runs/Davnah_relation_input_relation/Davnah/relations_merged.jsonl \
        [--label-view raw|synonyms|families] [--primary-mode boundary] \
        [--output output/eval_reports/relation_agent/<run>.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_relation_iaa as RI  # noqa: E402

ARG_MODES = RI.ARG_MODES
N_BOOT = 5000
SEED = 42


# ───────────────────────── loading ─────────────────────────

def _norm_text(t: str) -> str:
    return " ".join((t or "").lower().split())


def load_agent(path: Path) -> Dict[str, dict]:
    """
    {normalised sentence: native sentence dict} from a DeliberationRecord JSONL.

    `final_entities` / `final_relations` carry char offsets into the sentence,
    which is all the matching needs; everything else in the record is ignored.
    A record whose sentence repeats an earlier one keeps the first (runs do not
    repeat sentences; this only guards the key).
    """
    out: Dict[str, dict] = {}
    with path.open(encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            text = rec.get("sentence", "")
            key = _norm_text(text)
            if key in out:
                continue
            spans = []
            for e in rec.get("final_entities") or []:
                if isinstance(e.get("start"), int) and isinstance(e.get("end"), int):
                    spans.append({"start_char": e["start"], "end_char": e["end"],
                                  "type": e.get("entity_type", ""),
                                  "text": e.get("text", "")})
            rels = []
            for r in rec.get("final_relations") or []:
                e1, e2 = r.get("e1") or {}, r.get("e2") or {}
                if not (isinstance(e1.get("start"), int) and isinstance(e2.get("start"), int)):
                    continue
                rels.append({
                    "relation": r.get("relation", ""), "level": 0,
                    "e1_kind": "entity", "e2_kind": "entity",
                    "e1": {"start_char": e1["start"], "end_char": e1["end"],
                           "type": e1.get("entity_type", ""), "text": e1.get("text", "")},
                    "e2": {"start_char": e2["start"], "end_char": e2["end"],
                           "type": e2.get("entity_type", ""), "text": e2.get("text", "")},
                })
            out[key] = {"text": text, "spans": spans, "relations": rels}
    return out


def load_gold(path: Path) -> Dict[str, dict]:
    """{normalised sentence: native sentence dict} from a merged relation file."""
    return {_norm_text(s.get("text", "")): s for s in RI.load_doc(path)}


# ───────────────────────── consensus gold ─────────────────────────

def build_consensus(golds: Sequence[Dict[str, dict]], mode: str,
                    view: str) -> Dict[str, dict]:
    """
    The gold both annotators back: entities they both marked (under `mode`) and
    relations they both linked with the same label. Built by the same greedy
    one-to-one matching as everything else, so a consensus item is exactly an
    item the IAA counted as an agreement.
    """
    a, b = golds[0], golds[1]
    out: Dict[str, dict] = {}
    for key in set(a) & set(b):
        sa, sb = a[key], b[key]
        ents_a, ents_b = entity_spans(sa), entity_spans(sb)
        spans = []
        used: set = set()
        for i, ea in enumerate(ents_a):
            for j, eb in enumerate(ents_b):
                if j in used:
                    continue
                if RI.arg_score(ea, eb, mode) is not None:
                    used.add(j)
                    spans.append({"start_char": ea.start, "end_char": ea.end,
                                  "type": ea.type, "text": ea.text})
                    break
        ra, _, _ = RI.sentence_relations(sa, view)
        rb, _, _ = RI.sentence_relations(sb, view)
        rels = []
        for i, _j, _sw in RI.match_relations(ra, rb, mode, labeled=True):
            r = ra[i]
            rels.append({
                "relation": r.raw_label, "level": 0,
                "e1_kind": "entity", "e2_kind": "entity",
                "e1": {"start_char": r.e1.start, "end_char": r.e1.end,
                       "type": r.e1.type, "text": r.e1.text},
                "e2": {"start_char": r.e2.start, "end_char": r.e2.end,
                       "type": r.e2.type, "text": r.e2.text},
            })
        out[key] = {"text": sa.get("text", ""), "spans": spans, "relations": rels}
    return out


# ───────────────────────── statistics ─────────────────────────

def prf(tp: int, n_pred: int, n_gold: int) -> Dict[str, Any]:
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    f = (2 * p * r / (p + r)) if (p and r) else (0.0 if (n_pred or n_gold) else None)
    return {"precision": p, "recall": r, "f1": f,
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def bootstrap_f1_ci(per_sentence: List[Tuple[int, int, int]]
                    ) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for the micro F1, resampling SENTENCES (the unit of
    independence), not items."""
    import random
    if not per_sentence:
        return (None, None)
    rng = random.Random(SEED)
    n = len(per_sentence)
    f1s = []
    for _ in range(N_BOOT):
        tp = npred = ngold = 0
        for _ in range(n):
            t, p, g = per_sentence[rng.randrange(n)]
            tp += t; npred += p; ngold += g
        denom = npred + ngold
        f1s.append(2 * tp / denom if denom else 0.0)
    f1s.sort()
    return (f1s[int(0.025 * N_BOOT)], f1s[int(0.975 * N_BOOT)])


# ───────────────────────── scoring ─────────────────────────

def entity_spans(sent: dict) -> List[RI.Arg]:
    """The entity layer of one sentence, as the span reports count it: a span
    with no offsets cannot match anything, so scoring it would only deflate
    recall (two of Davnah's spans, incl. one with empty text, are like this),
    and a RELATION-typed span is a relation hint, not an entity."""
    return [a for a in RI.entity_args(sent)
            if a.has_offsets and not a.type.startswith("RELATION")]


def _entity_counter(spans: List[RI.Arg], mode: str) -> Counter:
    if mode == "strict":
        return Counter((s.start, s.end, s.type) for s in spans)
    if mode == "boundary":
        return Counter((s.start, s.end) for s in spans)
    if mode == "text":
        return Counter((s.text,) for s in spans)
    raise ValueError(mode)


def score(agent: Dict[str, dict], gold: Dict[str, dict], view: str,
          primary_mode: str,
          gold_pair: Optional[Tuple[Dict[str, dict], Dict[str, dict]]] = None,
          ) -> Dict[str, Any]:
    """
    Every metric for one (run, gold reference) pair.

    `gold_pair` marks the CONSENSUS reference and carries the two annotator
    files it was built from. The entity gold is then the per-mode intersection
    of the two — an entity is consensus under `strict` only if both gave it the
    same type, which is not the same set as the boundary consensus. Taking one
    set for both rows (e.g. boundary-matched spans carrying the first
    annotator's type) would credit the agent for a type the second annotator
    rejected. Layer 1 does the same thing for spans; this keeps the two
    reconcilable.
    """
    # Scope: the gold sentence must have something to find, and the run must
    # cover it. An empty agent sentence stays in — that is recall loss.
    keys = sorted(k for k in gold if k in agent)

    def gold_entity_counter(k: str, mode: str) -> Counter:
        if gold_pair is None:
            return _entity_counter(entity_spans(gold[k]), mode)
        a, b = gold_pair
        ca = _entity_counter(entity_spans(a[k]), mode) if k in a else Counter()
        cb = _entity_counter(entity_spans(b[k]), mode) if k in b else Counter()
        return ca & cb

    scored_ent = [k for k in keys
                  if any(gold_entity_counter(k, m) for m in ("boundary", "strict"))]
    scored_rel = [k for k in keys if RI.sentence_relations(gold[k], view)[0]]

    res: Dict[str, Any] = {
        "scope": {
            "gold_sentences": len(gold),
            "run_sentences": len(agent),
            "covered": len(keys),
            "not_in_run": len(set(gold) - set(agent)),
            "scored_entities": len(scored_ent),
            "scored_relations": len(scored_rel),
            "agent_empty_entities": sum(1 for k in scored_ent if not agent[k]["spans"]),
            "agent_empty_relations": sum(1 for k in scored_rel if not agent[k]["relations"]),
        }
    }

    # ── entities ──
    ent: Dict[str, Any] = {}
    for mode in ("boundary", "strict", "text"):
        tp = npred = ngold = 0
        per_sent = []
        for k in scored_ent:
            pc = _entity_counter(entity_spans(agent[k]), mode)
            gc = gold_entity_counter(k, mode)
            s_tp = sum((pc & gc).values())
            tp += s_tp; npred += sum(pc.values()); ngold += sum(gc.values())
            per_sent.append((s_tp, sum(pc.values()), sum(gc.values())))
        ent[mode] = prf(tp, npred, ngold)
        ent[mode]["f1_ci95"] = list(bootstrap_f1_ci(per_sent))
    res["entities"] = ent

    # ── relations: the gold pool excludes hyper-relations (see module docstring) ──
    pred_rels = {k: RI.sentence_relations(agent[k], view)[0] for k in scored_rel}
    gold_rels: Dict[str, List[RI.Rel]] = {}
    hyper_gold = 0
    for k in scored_rel:
        rels, hyper, _ = RI.sentence_relations(gold[k], view)
        gold_rels[k] = rels
        hyper_gold += hyper
    res["scope"]["gold_hyper_excluded"] = hyper_gold
    res["scope"]["gold_relations"] = sum(len(v) for v in gold_rels.values())
    res["scope"]["agent_relations"] = sum(len(v) for v in pred_rels.values())

    for tag, labeled in (("pairs", False), ("triplets", True)):
        per_mode: Dict[str, Any] = {}
        for mode in ARG_MODES:
            tp = npred = ngold = 0
            per_sent = []
            for k in scored_rel:
                m = RI.match_relations(pred_rels[k], gold_rels[k], mode, labeled=labeled)
                tp += len(m)
                npred += len(pred_rels[k]); ngold += len(gold_rels[k])
                per_sent.append((len(m), len(pred_rels[k]), len(gold_rels[k])))
            per_mode[mode] = prf(tp, npred, ngold)
            per_mode[mode]["f1_ci95"] = list(bootstrap_f1_ci(per_sent))
        res[tag] = per_mode

    # ── argument ceiling: gold relations the agent's entity layer can support ──
    ceiling: Dict[str, Any] = {}
    for mode in ARG_MODES:
        ok = 0
        n = 0
        for k in scored_rel:
            ents = entity_spans(agent[k])
            for r in gold_rels[k]:
                n += 1
                ok += 1 if RI._endpoint_present(r, ents, mode) else 0
        ceiling[mode] = {"gold_relations_supported": ok, "gold_relations": n,
                         "recall_ceiling": (ok / n) if n else None}
    res["argument_ceiling"] = ceiling

    # ── labeling given a co-detected pair ──
    label_pairs: List[Tuple[str, str]] = []
    confusion: Counter = Counter()
    for k in scored_rel:
        for i, j, _sw in RI.match_relations(pred_rels[k], gold_rels[k],
                                            primary_mode, labeled=False):
            label_pairs.append((pred_rels[k][i].label, gold_rels[k][j].label))
            confusion[(pred_rels[k][i].label, gold_rels[k][j].label)] += 1
    n_pairs = len(label_pairs)
    res["labeling_given_pair"] = {
        "co_detected_pairs": n_pairs,
        "accuracy": (sum(1 for a, b in label_pairs if a == b) / n_pairs) if n_pairs else None,
        "cohen_kappa": RI.cohen_kappa(label_pairs),
    }
    res["_confusion"] = confusion

    # ── per-label triplet F1 under the primary mode ──
    tp_by_label: Counter = Counter()
    for k in scored_rel:
        for i, _j, _sw in RI.match_relations(pred_rels[k], gold_rels[k],
                                             primary_mode, labeled=True):
            tp_by_label[pred_rels[k][i].label] += 1
    cp = Counter(r.label for v in pred_rels.values() for r in v)
    cg = Counter(r.label for v in gold_rels.values() for r in v)
    res["per_label"] = {
        lab: prf(tp_by_label[lab], cp[lab], cg[lab])
        for lab in sorted(set(cp) | set(cg))
    }
    return res


# ───────────────────────── reporting ─────────────────────────

def _n(x: Optional[float]) -> str:
    return "  N/A" if x is None else f"{x:.3f}"


def print_report(run_name: str, results: Dict[str, Dict[str, Any]],
                 view: str, primary_mode: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  MULTI-AGENT PIPELINE vs HUMAN GOLD — {run_name}")
    print(f"{'=' * 78}")
    print(f"  label view: {view}   primary argument match: {primary_mode}")

    first = next(iter(results.values()))
    sc = first["scope"]
    print("\n  Scope")
    print(f"    Gold sentences / covered by the run:   {sc['gold_sentences']} / {sc['covered']}"
          + (f"   ({sc['not_in_run']} not in the run)" if sc["not_in_run"] else ""))
    print(f"    Scored for entities / relations:       {sc['scored_entities']}"
          f" / {sc['scored_relations']}")
    print(f"    Sentences the agent left empty:        {sc['agent_empty_entities']}"
          f" entities, {sc['agent_empty_relations']} relations")

    for ref, r in results.items():
        s = r["scope"]
        print(f"\n  ── gold: {ref} "
              f"({s['gold_relations']} relations, agent {s['agent_relations']}; "
              f"{s['gold_hyper_excluded']} gold hyper-relations excluded)")
        print(f"    {'layer':<26}{'P':>8}{'R':>8}{'F1':>8}   {'TP':>5} {'pred':>5} {'gold':>5}")
        rows = [("entities (boundary)", r["entities"]["boundary"]),
                ("entities (strict +type)", r["entities"]["strict"]),
                (f"relation pairs ({primary_mode})", r["pairs"][primary_mode]),
                (f"triplets ({primary_mode})", r["triplets"][primary_mode])]
        for name, d in rows:
            print(f"    {name:<26}{_n(d['precision']):>8}{_n(d['recall']):>8}"
                  f"{_n(d['f1']):>8}   {d['tp']:>5} {d['n_pred']:>5} {d['n_gold']:>5}")
        c = r["argument_ceiling"][primary_mode]
        lg = r["labeling_given_pair"]
        print(f"    recall ceiling from the agent's entity layer: "
              f"{_n(c['recall_ceiling'])}"
              f"   ({c['gold_relations_supported']}/{c['gold_relations']} gold relations"
              f" have both endpoints)")
        print(f"    labeling given a co-detected pair: accuracy {_n(lg['accuracy'])}"
              f"   kappa {_n(lg['cohen_kappa'])}   (n={lg['co_detected_pairs']})")

    print("\n  Argument-match ladder (triplet F1)")
    print(f"    {'gold':<22}" + "".join(f"{m:>10}" for m in ARG_MODES))
    for ref, r in results.items():
        print(f"    {ref:<22}" + "".join(_n(r['triplets'][m]['f1']).rjust(10)
                                         for m in ARG_MODES))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-agent pipeline vs human gold: entities, relation pairs, triplets.")
    ap.add_argument("--agent-jsonl", type=Path, required=True)
    ap.add_argument("--human", action="append", default=[], metavar="NAME=PATH",
                    required=True, help="Repeatable. e.g. --human Mark=.../relations_merged.jsonl")
    ap.add_argument("--label-view", choices=RI.LABEL_VIEWS, default="raw")
    ap.add_argument("--primary-mode", choices=ARG_MODES, default="boundary")
    ap.add_argument("--no-consensus", action="store_true",
                    help="Skip the consensus reference (built from the first two humans)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    agent = load_agent(args.agent_jsonl)
    names, golds = [], []
    for spec in args.human:
        name, _, path = spec.partition("=")
        names.append(name)
        golds.append(load_gold(Path(path)))

    refs: List[Tuple[str, Dict[str, dict]]] = list(zip(names, golds))
    if len(golds) >= 2 and not args.no_consensus:
        refs.append((f"Consensus ({'∩'.join(names[:2])})",
                     build_consensus(golds[:2], args.primary_mode, args.label_view)))

    results = {}
    for name, g in refs:
        pair = (golds[0], golds[1]) if name.startswith("Consensus") else None
        results[name] = score(agent, g, args.label_view, args.primary_mode,
                              gold_pair=pair)
    run_name = args.agent_jsonl.name
    print_report(run_name, results, args.label_view, args.primary_mode)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"run": run_name, "agent_jsonl": str(args.agent_jsonl),
                   "label_view": args.label_view, "primary_mode": args.primary_mode,
                   "references": {k: {kk: vv for kk, vv in v.items()
                                      if not kk.startswith("_")}
                                  for k, v in results.items()}}
        for k, v in results.items():
            payload["references"][k]["label_confusion"] = {
                f"{a} -> {b}": c for (a, b), c in v["_confusion"].most_common()}
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf8")
        print(f"  JSON written to {args.output}")


if __name__ == "__main__":
    main()
