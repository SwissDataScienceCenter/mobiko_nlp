#!/usr/bin/env bash
#
# Run the multi-agent pipeline over data/second_iteration_data/*.txt and build a
# single relation-annotation input file from the results.
#
# One pipeline run per .txt file. That is not about output granularity -- the
# converter puts every document in one JSON, keyed by doc_id -- it is the only
# way to know which sentence came from which PMC: demo_ag2's load_sentences()
# globs a directory, flattens every line into one list and discards the
# filename, and 8 lines are duplicated across these 12 documents. Running per
# file also keeps each run JSONL a write-once record of that document and lets
# an interrupted corpus resume at file granularity.
#
# Any extra arguments are passed straight through to demo_ag2.py, so the run
# configuration is chosen at call time:
#
#   scripts/run_second_iteration.sh                                  # defaults
#   scripts/run_second_iteration.sh --single-pass                    # annotator only
#   scripts/run_second_iteration.sh --max-rounds 3 --strict-critic
#   scripts/run_second_iteration.sh --annotator-model rcp-kimi-2.7 \
#       --critic-model rcp-kimi-2.7 --adjudicator-model rcp-kimi-2.7
#
# Environment overrides:
#   INPUT_DIR   source .txt files       (default data/second_iteration_data)
#   RUN_DIR     run JSONLs              (default data/auto_annotated/second_iteration)
#   OUT         annotation JSON         (default data/aug_runs/second_iteration_relation_input.json)
#   ONLY        space-separated stems to run, e.g. ONLY="PMC6023484"
#
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.mobiko-venv/bin/python}
INPUT_DIR=${INPUT_DIR:-data/second_iteration_data}
RUN_DIR=${RUN_DIR:-data/auto_annotated/second_iteration}
OUT=${OUT:-data/aug_runs/second_iteration_relation_input.json}
ONLY=${ONLY:-}

# Nothing in annotate_batch creates these: it calls Path(output).write_text("")
# directly, which raises FileNotFoundError only after the annotator has been
# built and the guideline sections embedded.
mkdir -p "$RUN_DIR" "$(dirname "$OUT")"

shopt -s nullglob
files=("$INPUT_DIR"/*.txt)
if [ ${#files[@]} -eq 0 ]; then
    echo "[ERROR] No .txt files in $INPUT_DIR" >&2
    exit 1
fi

for f in "${files[@]}"; do
    pmc=$(basename "$f" .txt)
    if [ -n "$ONLY" ] && ! grep -qw -- "$pmc" <<<"$ONLY"; then
        continue
    fi
    out="$RUN_DIR/$pmc.jsonl"

    echo
    echo "=============================================================="
    echo "  $pmc  ($(grep -c . "$f") sentences)"
    echo "=============================================================="

    # --output is always explicit: without --resume, annotate_batch truncates
    # the target to zero bytes before the first sentence, and demo_ag2's default
    # --output points at an existing run file.
    "$PY" src/multi_agent_annotation/demo_ag2.py \
        --input "$f" \
        --output "$out" \
        --no-relation-schema-in-prompt \
        --resume \
        "$@"

    # A parse failure writes an empty record rather than no record, and resume
    # decides what is done by sentence text, so such a sentence is marked
    # permanently done and never retried. Dropping those records is the only way
    # to give them another pass; genuine empty results (status "constrained")
    # are left alone.
    "$PY" scripts/prune_failed_records.py "$out" || true
done

echo
echo "=============================================================="
echo "  Building $OUT"
echo "=============================================================="
"$PY" scripts/agent_jsonl_to_relation_input.py \
    --in "$RUN_DIR" \
    --source-dir "$INPUT_DIR" \
    --out "$OUT"
