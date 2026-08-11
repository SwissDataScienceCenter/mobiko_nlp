#!/usr/bin/env python
"""P3 sampler — draw K independent annotator passes per sentence.

WHY THIS AND NOT A TEMPERATURE SWEEP
The RQ2 signal (span type-logprob -> human type-disagreement) is positive in
every run with logprob headroom and null in every saturated run. Two readings
survive: headroom is the cause, or model identity is. Qwen can't arbitrate
because it reports top-1 ~=
1.0000 on essentially every span — and it already
does so at annotator temperature 0.7, so it is not a near-greedy-decoding
artefact. Raising temperature further would (a) possibly not move the reported
logprobs at all, depending on whether the server scales them, and (b) degrade the
annotation, changing which spans exist and so breaking comparability with the
human-disagreement pairing.

Sampling variability sidesteps both problems. Run the ANNOTATOR'S INITIAL PASS
K times at the configuration under test and measure how often it changes its
mind. That is a model-behaviour measurement, independent of how the server
reports logprobs, and it leaves the configuration untouched.

It resolves either way: if qwen returns the same label K/K on nearly every span,
qwen is genuinely certain, the signal is unmeasurable there, and the finding is
model-dependent — a clean result, not a failed experiment.

Only the annotator runs. No critic, no adjudicator, no deliberation, so cost is
K annotator calls per sentence rather than K full records.

Usage:
    # MoBiKo (default corpus, unchanged behaviour)
    python scripts/xrun/selfconsistency_sample.py \
        --annotator-model qwen3-35B-vllm \
        --input data/manually_labeled_last \
        --output output/selfconsistency/qwen36_35B_K10.jsonl \
        --k 10

    # CoNLL-2003 MTurk: 4-type schema, its own guideline, no relations/seeds
    python scripts/xrun/selfconsistency_sample.py --corpus conll \
        --annotator-model qwen3-35B-vllm \
        --input data/conll_mturk/sample_n5_w10_400/sentences.txt \
        --output output/selfconsistency/conll_qwen_K10.jsonl \
        --k 10

    # the guideline-free arm on either corpus
    ... --corpus conll --no-guideline
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO, _REPO / "src" / "multi_agent_annotation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.resources_updated.entity_schema import SCHEMA_BIODIV_SHORT, SCHEMA_BIODIV_LIST  # noqa: E402
from src.resources_updated.entity_schema_conll import (  # noqa: E402
    SCHEMA_CONLL_SHORT, SCHEMA_CONLL_LIST)
from src.multi_agent_annotation.demo_ag2 import load_sentences  # noqa: E402
from src.multi_agent_annotation.multi_agent_annotation_ag2 import (  # noqa: E402
    MultiAgentAnnotator,
    _per_entity_type_logprobs,
    last_content_token_logprobs,
)


# ── corpus presets ──────────────────────────────────────────────────────────
# One switch rather than six flags, because every resource here has a MoBiKo
# default that applies SILENTLY when omitted: entity_types_list falls back to the
# 15 biodiversity types, and guideline_path=None resolves to the MoBiKo guideline
# rather than to no guideline. Setting them one at a time means a single forgotten
# flag produces a run that looks fine and is annotating newswire against a
# biodiversity schema. Selecting a named corpus sets all of them together.
CORPORA = {
    "mobiko": {
        "entity_schema": SCHEMA_BIODIV_SHORT,
        "entity_types": SCHEMA_BIODIV_LIST,
        "guideline": None,            # None -> the pipeline's MoBiKo default
        "decision_support": True,
        # entity_schema.py holds the LABEL list; load_schema wants the RELATION
        # schema, which is relation_schema_new.py (7 relations — the vocabulary
        # the scored runs actually use).
        "schema": _REPO / "src/resources_updated/relation_schema_new.py",
        "seeds": _REPO / "src/resources_updated/manual_seeds_filled.py",
    },
    "conll": {
        "entity_schema": SCHEMA_CONLL_SHORT,
        "entity_types": SCHEMA_CONLL_LIST,
        "guideline": _REPO / "src/multi_agent_annotation/CoNLL_label_guidance.md",
        # MoBiKo's Decision_support.csv is biodiversity-specific; there is no
        # CoNLL equivalent, and the annotator treats absence as an empty table.
        "decision_support": False,
        # CoNLL-2003 annotates no relations. None (not a path) so schema_lookup
        # reports nothing valid instead of offering biodiversity relations.
        "schema": None,
        "seeds": None,
    },
}


def _acquire_shard_lock(output: Path):
    """Take an exclusive lock on this shard's output file, or exit.

    Under --resume the output is opened in APPEND mode, so two launches of the
    same shard each read the same set of already-done sentences and then each
    append their own copy — writing every sentence twice. That is exactly how the
    qwen K=10 run came to merge 30 records from 28 sentences, and duplicated
    sentences are not independent observations: they inflate n and bias rho and p
    in the scored report. The scorer now dedupes, but silently producing the
    duplicates and paying for the extra calls is still worth preventing.

    Failing fast here costs one syscall. The returned handle must stay referenced
    for the life of the process: closing it releases the lock.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(
            f"ERROR: another process is already writing {output}\n"
            f"       (lock held on {lock_path})\n"
            f"       Two samplers on one shard file append every sentence twice.\n"
            f"       Wait for the running shard to finish, or pass a different --output.")
    fh.write(f"pid {os.getpid()}\n")
    fh.flush()
    return fh


def _label_variants(usable_samples):
    """{span text: set of labels assigned across samples} — the live diagnostic."""
    out = {}
    for sample in usable_samples:
        for ent in sample:
            key = " ".join((ent.get("text") or "").lower().split())
            if key:
                out.setdefault(key, set()).add(ent.get("entity_type", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator-model", default="qwen3-35B-vllm")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10,
                    help="samples per sentence (default 10)")
    ap.add_argument("--num-sentences", type=int, default=None,
                    help="how many sentences to sample, counting from --start")
    ap.add_argument("--start", type=int, default=0,
                    help="0-based offset into the sentence list. Use with "
                         "--num-sentences to SHARD a run across parallel "
                         "processes, each with its OWN --output, then cat the "
                         "shards together. In-process threading is unsafe here: "
                         "the pipeline's logprob capture is module-global and "
                         "documented as relying on sequential use, so separate "
                         "processes are the safe way to parallelise.")
    ap.add_argument("--annotator-temperature", type=float, default=None,
                    help="default None keeps the pipeline's 0.7 — the point is to "
                         "measure the configuration actually used, not a new one")
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="mobiko",
                    help="selects entity schema, guideline, decision support, "
                         "relation schema and seeds together (see CORPORA). "
                         "Individual flags below override the preset.")
    ap.add_argument("--schema", type=Path, default=None,
                    help="relation schema; defaults to the corpus preset")
    ap.add_argument("--seeds", type=Path, default=None,
                    help="relation seeds; defaults to the corpus preset")
    ap.add_argument("--guideline", type=Path, default=None,
                    help="guideline .md/.docx; defaults to the corpus preset")
    ap.add_argument("--no-guideline", action="store_true",
                    help="run with NO guideline at all. Needed because omitting "
                         "--guideline means 'use the preset', not 'use none'.")
    ap.add_argument("--no-decision-support", action="store_true",
                    help="run with no decision-support table")
    ap.add_argument("--resume", action="store_true",
                    help="skip sentences already in the output file")
    ap.add_argument("--progress-file", type=Path, default=None,
                    help="also append [PROG] lines here, so a long run can be "
                         "watched with `tail -f` without the agent transcript")
    args = ap.parse_args()

    # Held for the whole process — see _acquire_shard_lock. Taken before the model
    # is constructed so a double launch fails instantly instead of after init.
    _shard_lock = _acquire_shard_lock(args.output)  # noqa: F841

    sentences = load_sentences(args.input.resolve())
    _total = len(sentences)
    if args.start:
        sentences = sentences[args.start:]
    if args.num_sentences:
        sentences = sentences[: args.num_sentences]
    if args.start or args.num_sentences:
        print(f"[OK] shard: sentences {args.start}..{args.start + len(sentences) - 1} "
              f"of {_total}", flush=True)

    done = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("{"):
                done.add(json.loads(line).get("sentence", ""))
        print(f"[resume] {len(done)} sentence(s) already sampled")

    # Resources are NOT loaded here on purpose. MultiAgentAnnotator.__init__
    # loads the schema, seeds, decision support and guideline itself and calls
    # _init_tool_state — including the .md-vs-.docx dispatch that the default
    # guideline (a .md file) requires. Duplicating that here only creates a
    # second copy to drift from the pipeline, which is exactly how this script
    # first broke: it called the .docx loader on a .md path.
    preset = CORPORA[args.corpus]
    schema_path = args.schema or preset["schema"]
    seeds_path = args.seeds or preset["seeds"]
    guideline_path = args.guideline or preset["guideline"]
    use_guideline = not args.no_guideline
    use_decision_support = preset["decision_support"] and not args.no_decision_support

    annotator = MultiAgentAnnotator(
        annotator_model=args.annotator_model,
        critic_model=args.annotator_model,          # unused, must be constructible
        adjudicator_model=args.annotator_model,     # unused
        # .resolve() only when a path was actually selected: the annotator treats
        # None as "no relation schema / no seeds", which is what CoNLL needs.
        schema_path=schema_path.resolve() if schema_path else None,
        seeds_path=seeds_path.resolve() if seeds_path else None,
        guideline_path=guideline_path.resolve() if guideline_path else None,
        use_guideline=use_guideline,
        use_decision_support=use_decision_support,
        entity_schema_str=preset["entity_schema"],
        entity_types_list=preset["entity_types"],
        annotator_temperature=args.annotator_temperature,
        input_path=args.input.resolve(),
    )
    meta = dict(annotator.run_meta)
    meta["selfconsistency_k"] = args.k
    todo = [x for x in sentences if x not in done]
    # Progress goes to STDERR with a fixed prefix. autogen writes the full agent
    # transcript to stdout, so anything interleaved there is unfindable; stderr
    # keeps it separable ("2>progress.log", or grep PROG). flush=True because a
    # single call takes minutes and buffered output would look like a hang.
    _pf = args.progress_file.open("a", encoding="utf-8") if args.progress_file else None

    def prog(msg):
        line = f"[PROG] {msg}"
        print(line, file=sys.stderr, flush=True)
        if _pf:
            _pf.write(line + "\n")
            _pf.flush()

    prog(f"annotator={args.annotator_model} t={meta.get('annotator_temperature')} "
         f"K={args.k} sentences={len(todo)} (of {len(sentences)}; "
         f"{len(done)} already done) -> {args.output}")
    # Echo the EFFECTIVE resources, not the requested ones. Every value here has
    # a silent MoBiKo fallback, so a run annotating the wrong corpus looks
    # completely normal unless the config is printed where the log will show it.
    prog(f"corpus={args.corpus} types={meta.get('entity_types')} "
         f"guideline_sections={meta.get('guideline_sections')} "
         f"decision_support={meta.get('decision_support_sections')} "
         f"guideline_search={meta.get('guideline_search_registered')} "
         f"relations={'yes' if schema_path else 'none'}")
    prog(f"expect {len(todo) * args.k} annotator calls total")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and args.output.exists()) else "w"
    t0 = time.perf_counter()
    n_unanimous = n_spans_seen = calls_done = 0
    with args.output.open(mode, encoding="utf-8") as out:
        for i, sentence in enumerate(todo, 1):
            preview = " ".join(sentence.split())[:60]
            prog(f"sentence {i}/{len(todo)} START  \"{preview}...\"")
            s_t0 = time.perf_counter()
            task_msg = f'Annotate this sentence:\n\n"{sentence}"'
            samples, failures = [], 0
            for _k in range(1, args.k + 1):
                k_t0 = time.perf_counter()
                try:
                    content, _rec = annotator._run_agent_turn(
                        annotator.annotator, annotator.annotator_executor, task_msg)
                    parsed = MultiAgentAnnotator._parse_annotator_output(content)
                    if parsed is None:
                        failures += 1
                        samples.append(None)      # keep K aligned; scorer skips None
                    else:
                        # Capture the REPORTED confidence for this same pass, so one
                        # dataset holds both measures on the same spans. The pilot
                        # showed qwen flipping labels on 35% of spans while its
                        # logprobs read ~1.0000; storing both here lets us check
                        # that disconnect directly instead of inferring it across
                        # datasets. Same alignment call the pipeline uses for
                        # annotator_entity_logprobs.
                        lp_by_span = {}
                        _toks = last_content_token_logprobs()
                        if _toks:
                            for lp in _per_entity_type_logprobs(_toks, parsed.entities):
                                alts = lp.get("type_top_alternatives") or []
                                lp_by_span[lp.get("text", "")] = {
                                    "type_mean_logprob": lp.get("type_mean_logprob"),
                                    "type_max_entropy": lp.get("type_max_entropy"),
                                    "top1_prob": alts[0][1] if alts else None,
                                }
                        # entity_type is canonicalised by the pydantic validator
                        samples.append([
                            dict({"text": e.text, "entity_type": e.entity_type},
                                 **lp_by_span.get(e.text, {}))
                            for e in parsed.entities])
                except Exception as exc:          # never lose the whole run to one call
                    failures += 1
                    samples.append(None)
                    prog(f"  sentence {i} sample {_k}: ERROR {type(exc).__name__}: "
                         f"{str(exc)[:120]}")
                calls_done += 1
                n_ent = len(samples[-1]) if samples[-1] is not None else 0
                # ETA from mean call time so far, over all remaining calls
                mean_call = (time.perf_counter() - t0) / max(calls_done, 1)
                remaining = (len(todo) - i) * args.k + (args.k - _k)
                prog(f"  sentence {i}/{len(todo)} sample {_k}/{args.k} done "
                     f"({time.perf_counter() - k_t0:.0f}s, {n_ent} entities) | "
                     f"mean {mean_call:.0f}s/call | ETA {remaining * mean_call / 60:.0f} min")

            out.write(json.dumps({"sentence": sentence, "samples": samples,
                                  "n_parse_failures": failures,
                                  "run_meta": meta}, ensure_ascii=False) + "\n")
            out.flush()

            # Running diagnostic: the whole experiment turns on whether the model
            # ever changes its label between samples. Reporting it live means an
            # all-unanimous model can be spotted after a few sentences and the run
            # killed, instead of after every sentence has been paid for.
            usable = [x for x in samples if x is not None]
            for text, labels in _label_variants(usable).items():
                n_spans_seen += 1
                if len(labels) == 1:
                    n_unanimous += 1
            pct = (100 * n_unanimous / n_spans_seen) if n_spans_seen else 0.0
            prog(f"sentence {i}/{len(todo)} DONE  {len(usable)}/{args.k} usable, "
                 f"{failures} failed, {time.perf_counter() - s_t0:.0f}s | "
                 f"cumulative unanimity {pct:.1f}% of {n_spans_seen} spans")
            if n_spans_seen >= 50 and pct == 100.0:
                prog("NOTE every span so far is unanimous across samples. If this "
                     "holds, the model has no sampling variability and the test "
                     "cannot run for it — that is the answer; consider stopping.")
    print(f"\nWritten to: {args.output}")
    print("Score it with: python scripts/xrun/selfconsistency_score.py "
          f"--samples {args.output}")


if __name__ == "__main__":
    main()
