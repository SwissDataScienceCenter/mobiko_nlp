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
    python scripts/xrun/selfconsistency_sample.py \
        --annotator-model qwen3-35B-vllm \
        --input data/manually_labeled_last \
        --output output/selfconsistency/qwen36_35B_K10.jsonl \
        --k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO, _REPO / "src" / "multi_agent_annotation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.resources_updated.entity_schema import SCHEMA_BIODIV_SHORT, SCHEMA_BIODIV_LIST  # noqa: E402
from src.multi_agent_annotation.demo_ag2 import load_sentences  # noqa: E402
from src.multi_agent_annotation.multi_agent_annotation_ag2 import (  # noqa: E402
    MultiAgentAnnotator,
)


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
    ap.add_argument("--num-sentences", type=int, default=None)
    ap.add_argument("--annotator-temperature", type=float, default=None,
                    help="default None keeps the pipeline's 0.7 — the point is to "
                         "measure the configuration actually used, not a new one")
    # entity_schema.py holds the LABEL list; load_schema wants the RELATION
    # schema, which is relation_schema_new.py (7 relations — the vocabulary the
    # scored runs actually use). demo_ag2 has no defaults for these and requires
    # them explicitly; defaults here keep the pilot a one-liner.
    ap.add_argument("--schema", type=Path,
                    default=_REPO / "src/resources_updated/relation_schema_new.py")
    ap.add_argument("--seeds", type=Path,
                    default=_REPO / "src/resources_updated/manual_seeds_filled.py")
    ap.add_argument("--resume", action="store_true",
                    help="skip sentences already in the output file")
    args = ap.parse_args()

    sentences = load_sentences(args.input.resolve())
    if args.num_sentences:
        sentences = sentences[: args.num_sentences]

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
    annotator = MultiAgentAnnotator(
        annotator_model=args.annotator_model,
        critic_model=args.annotator_model,          # unused, must be constructible
        adjudicator_model=args.annotator_model,     # unused
        schema_path=args.schema.resolve(),
        seeds_path=args.seeds.resolve(),
        entity_schema_str=SCHEMA_BIODIV_SHORT,
        entity_types_list=SCHEMA_BIODIV_LIST,
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
    def prog(msg):
        print(f"[PROG] {msg}", file=sys.stderr, flush=True)

    prog(f"annotator={args.annotator_model} t={meta.get('annotator_temperature')} "
         f"K={args.k} sentences={len(todo)} (of {len(sentences)}; "
         f"{len(done)} already done) -> {args.output}")
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
                        # entity_type is canonicalised by the pydantic validator
                        samples.append([{"text": e.text, "entity_type": e.entity_type}
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
