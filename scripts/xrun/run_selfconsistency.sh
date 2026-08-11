#!/usr/bin/env bash
# Self-consistency run (P3): sample each sentence K times, then score.
#
# Shards across parallel PROCESSES, not threads: the pipeline's logprob capture
# is module-global and documented as relying on sequential use, so threads would
# corrupt each other's state. Each shard gets its own output file — sharing one
# would interleave writes and corrupt it.
#
# Usage
#   scripts/xrun/run_selfconsistency.sh                       # defaults below
#   MODEL=rcp-kimi-2.7 TAG=kimi27 scripts/xrun/run_selfconsistency.sh
#   SENTENCES=56 SHARDS=8 scripts/xrun/run_selfconsistency.sh
#   PLAN=1 scripts/xrun/run_selfconsistency.sh                # print plan, run nothing
#
# Resume after an interrupted run: re-run the same command with RESUME=1.
# Each shard skips sentences already present in its own output file.

set -uo pipefail

# ── config (override by env var) ────────────────────────────────────────────
MODEL="${MODEL:-rcp-kimi-2.7}"
TAG="${TAG:-rcp-kimi-2.7}"
K="${K:-10}"
SENTENCES="${SENTENCES:-28}"      # total sentences across all shards
SHARDS="${SHARDS:-4}"             # parallel processes
INPUT="${INPUT:-data/manually_labeled_last}"
OUTDIR="${OUTDIR:-output/selfconsistency}"
PYTHON="${PYTHON:-python3}"
RESUME="${RESUME:-1}"
PLAN="${PLAN:-0}"

# Repo root from this script's location, so it works from any CWD and on any
# checkout (this is run on a different machine from where it was written).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

PER_SHARD=$(( SENTENCES / SHARDS ))
if (( PER_SHARD * SHARDS != SENTENCES )); then
  echo "ERROR: SENTENCES ($SENTENCES) must divide evenly by SHARDS ($SHARDS)." >&2
  echo "       e.g. SENTENCES=28 SHARDS=4, or SENTENCES=30 SHARDS=5" >&2
  exit 2
fi
if (( K < 5 )); then
  echo "ERROR: K=$K is too small. The scorer needs a span to appear in >=5" >&2
  echo "       samples to score its typing, so K<5 yields zero typing spans." >&2
  exit 2
fi

MERGED="$OUTDIR/${TAG}_K${K}.jsonl"
REPORT="$OUTDIR/${TAG}_K${K}_report.json"

echo "======================================================================"
echo "  SELF-CONSISTENCY RUN"
echo "======================================================================"
echo "  repo        $ROOT"
echo "  python      $PYTHON"
echo "  model       $MODEL"
echo "  input       $INPUT"
echo "  K           $K samples per sentence"
echo "  sentences   $SENTENCES  ($SHARDS shards x $PER_SHARD)"
echo "  calls       $(( SENTENCES * K ))  total annotator calls"
echo "  merged      $MERGED"
echo "  resume      $RESUME"
echo
for (( i=0; i<SHARDS; i++ )); do
  printf "    shard %d: sentences %d..%d -> %s/%s_K%s_s%d.jsonl\n" \
    "$i" "$(( i * PER_SHARD ))" "$(( i * PER_SHARD + PER_SHARD - 1 ))" \
    "$OUTDIR" "$TAG" "$K" "$i"
done
echo
if [[ "$PLAN" == "1" ]]; then
  echo "PLAN=1 — nothing executed."
  exit 0
fi

if [[ ! -e "$INPUT" ]]; then
  echo "ERROR: input not found: $INPUT" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

RESUME_FLAG=()
[[ "$RESUME" == "1" ]] && RESUME_FLAG=(--resume)

# ── launch shards ───────────────────────────────────────────────────────────
pids=(); shard_out=(); shard_log=()
for (( i=0; i<SHARDS; i++ )); do
  out="$OUTDIR/${TAG}_K${K}_s${i}.jsonl"
  log="$OUTDIR/${TAG}_K${K}_s${i}.progress.log"
  full="$OUTDIR/${TAG}_K${K}_s${i}.full.log"
  shard_out+=("$out"); shard_log+=("$log")
  "$PYTHON" scripts/xrun/selfconsistency_sample.py \
      --annotator-model "$MODEL" \
      --input "$INPUT" \
      --output "$out" \
      --k "$K" \
      --start "$(( i * PER_SHARD ))" \
      --num-sentences "$PER_SHARD" \
      --progress-file "$log" \
      "${RESUME_FLAG[@]}" \
      > "$full" 2>&1 &
  pids+=("$!")
  echo "  started shard $i (pid ${pids[-1]})"
done

echo
echo "  Watch progress:   tail -f ${shard_log[0]}"
echo "  All shards:       tail -f $OUTDIR/${TAG}_K${K}_s*.progress.log"
echo "  Full transcript:  $OUTDIR/${TAG}_K${K}_s0.full.log"
echo "  (the agent transcript is thousands of lines; .progress.log is the readable one)"
echo
echo "  waiting for $SHARDS shard(s)..."

failed=0
for (( i=0; i<SHARDS; i++ )); do
  if ! wait "${pids[$i]}"; then
    echo "  !! shard $i FAILED — see $OUTDIR/${TAG}_K${K}_s${i}.full.log" >&2
    failed=$(( failed + 1 ))
  else
    echo "  shard $i finished"
  fi
done

# ── merge ───────────────────────────────────────────────────────────────────
echo
present=()
for f in "${shard_out[@]}"; do
  [[ -s "$f" ]] && present+=("$f")
done
if (( ${#present[@]} == 0 )); then
  echo "ERROR: no shard produced output. Check the .full.log files." >&2
  exit 1
fi
cat "${present[@]}" > "$MERGED"
# "lines", not "sentences": under RESUME=1 a shard file is appended to, so it can
# hold a sentence twice and this count then overstates the sentences. The scorer
# dedupes and reports the true unique count.
echo "  merged ${#present[@]} shard(s) -> $MERGED ($(wc -l < "$MERGED") lines)"
if (( failed > 0 )); then
  echo "  WARNING: $failed shard(s) failed; scoring the PARTIAL merge." >&2
  echo "           Re-run with RESUME=1 to fill the gaps." >&2
fi

# ── score ───────────────────────────────────────────────────────────────────
echo
"$PYTHON" scripts/xrun/selfconsistency_score.py \
    --samples "$MERGED" --output "$REPORT"

echo
echo "======================================================================"
echo "  Report saved: $REPORT"
echo "  Read the DIAGNOSTIC block first (is there variability to correlate?),"
echo "  then MECHANISM (did reported top-1 stay ~1.0 on spans that flipped?)."
echo "======================================================================"
