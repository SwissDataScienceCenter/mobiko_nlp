#!/usr/bin/env bash
# Re-run the 8-eval suite for one agent JSONL into one report dir.
# usage: run_evals.sh <agent-jsonl> <outdir>
set -u
ROOT=/home/katinska/mobiko_nlp
PY=$ROOT/.mobiko-venv/bin/python
E=$ROOT/src/multi_agent_annotation/evaluation
MARK=$ROOT/data/aug_runs/combined_M_D_Mark_postprocessed.jsonl
DAV=$ROOT/data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl

AGENT="$1"
OUT="$2"
mkdir -p "$OUT"
cd "$ROOT"

run () {  # run <script> <stem> <extra args...>
  local script="$1"; local stem="$2"; shift 2
  echo "  -> $stem"
  $PY "$E/$script" --agent-jsonl "$AGENT" "$@" \
      --output "$OUT/$stem.json" > "$OUT/$stem.txt" 2> "$OUT/.$stem.err"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "     FAILED rc=$rc"; tail -5 "$OUT/.$stem.err"
  else
    rm -f "$OUT/.$stem.err"
  fi
}

run eval_layer1_output.py        layer1    --human-jsonl "$MARK" "$DAV" --names Mark Davnah
run eval_layer2_correlation.py   layer2    --human-jsonl "$MARK" "$DAV"
run eval_layer34_deliberation.py layer34   --human-jsonl "$MARK" "$DAV"
run eval_deliberation_history.py history   --human-jsonl "$MARK" "$DAV"
run eval_guideline_adherence.py  guideline
run eval_difficulty_model.py     diffmodel --human-jsonl "$MARK" "$DAV"
run eval_difficulty_split.py     diffsplit --mark "$MARK" --davnah "$DAV"
run eval_logprob_uncertainty.py  logprob   --human-jsonl "$MARK" "$DAV"
echo "done: $OUT"
