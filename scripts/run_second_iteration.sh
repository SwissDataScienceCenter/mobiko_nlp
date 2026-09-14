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

# One pipeline pass over a source file. Called twice per document when the first
# pass left parse failures behind: the second call re-attempts exactly those,
# because --resume skips every sentence still present in the output.
run_pipeline() {
    local src="$1" dest="$2"
    shift 2
    "$PY" src/multi_agent_annotation/demo_ag2.py \
        --input "$src" \
        --output "$dest" \
        --resume \
        "$@"
}

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
    run_pipeline "$f" "$out" "$@"

    # A parse failure writes an EMPTY record rather than no record, and --resume
    # decides what is done by sentence text, so such a sentence is marked
    # permanently done and is never retried. Deleting the record is the only way
    # to give it another pass -- which is why the prune has to be followed by a
    # second run. Genuine empty results (status "constrained") are a real answer
    # from the model and are left alone by prune_failed_records.py.
    #
    # --dry-run first: the script exits 0 whether or not there is anything to
    # delete, so the decision has to come from its output, and there is no point
    # taking a backup or spending a second pass when nothing failed. Its exit
    # status is checked separately -- a dry run that FAILED says nothing about
    # whether there are failures to prune, and must not fall through into
    # pruning for real.
    if ! dry_out=$("$PY" scripts/prune_failed_records.py "$out" --dry-run 2>&1); then
        echo "  [WARN] could not inspect $pmc for parse failures; skipping retry" >&2
        printf '%s\n' "$dry_out" >&2
    elif grep -q "Nothing to delete." <<<"$dry_out"; then
        echo "  no parse failures to retry."
    else
        # Our own backup, stamped to the second. prune_failed_records.py stamps
        # its own with the DATE only and refuses to overwrite one, so a second
        # prune of the same file on the same day would abort -- which is exactly
        # what a retry pass does.
        backup="$out.pre_prune_$(date +%Y%m%dT%H%M%S).bak"
        cp -p "$out" "$backup"
        echo "  backup: $backup"

        if "$PY" scripts/prune_failed_records.py "$out" --no-backup; then
            echo
            echo "  --- retrying the pruned sentences ---"
            # Same arguments; --resume skips everything still in the file, so
            # this pass touches only the sentences just deleted. One retry, not
            # a loop: anything that fails again keeps its parse-failed record
            # and is reported by the converter instead of being hidden.
            run_pipeline "$f" "$out" "$@"
        else
            # Don't abandon the remaining documents over one prune: their run
            # JSONLs are intact and the backup above still holds this one.
            echo "  [WARN] prune failed for $pmc; keeping the run as it is" >&2
            cp -p "$backup" "$out"
        fi
    fi
done

echo
echo "=============================================================="
echo "  Building $OUT"
echo "=============================================================="
"$PY" scripts/agent_jsonl_to_relation_input.py \
    --in "$RUN_DIR" \
    --source-dir "$INPUT_DIR" \
    --out "$OUT"
