#!/usr/bin/env bash
# Run the pinned grid, one cell at a time, resumable.
#
# grid_config.py --print emits the commands but running them needs three things
# a pasted one-liner does not give you, and this job is long enough that all
# three bite:
#
#   RESUME      A cell is 400 sentences. Losing one to a VPN drop or a gateway
#               503 at sentence 380 and restarting from zero is most of a day.
#               --resume continues an existing output file.
#   SKIP DONE   So that re-invoking after a crash does not re-run finished cells.
#   PER-CELL LOGS  One combined log for 18 runs is unreadable, and the thing you
#               need after a failure is that cell's traceback.
#
# The exports from --print (MOBIKO_CODE_VERSION, *_TOP_LOGPROBS) are evaluated
# here rather than dropped, because the code version is what every output record
# is stamped with and the top_logprobs are the RQ2 entropy instrument.
#
#   scripts/xrun/run_grid.sh conll              # the blind arm, 18 cells
#   scripts/xrun/run_grid.sh conll blind        # same, explicit
#   scripts/xrun/run_grid.sh mobiko multi solo  # the review baseline
#   DRY_RUN=1 scripts/xrun/run_grid.sh conll    # print what would run
#   ONLY=apertus scripts/xrun/run_grid.sh conll # just the Apertus cells
#   WAIT_MAX=7200 scripts/xrun/run_grid.sh conll  # tolerate a 2h outage
#
# Launch it detached, or a dropped SSH session SIGHUPs the whole job:
#   nohup setsid ./scripts/xrun/run_grid.sh conll blind > logs/grid.log 2>&1 &
set -uo pipefail

ROOT="/home/katinska/mobiko_nlp"
cd "$ROOT" || exit 1

CORPUS="${1:-conll}"
shift || true
ARMS=("${@:-blind}")

# RCP_API_KEY. Without it every cell dies on auth after the model list prints,
# which looks like a model problem rather than a credentials one.
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
if [[ -z "${RCP_API_KEY:-}" ]]; then
  echo "FAIL: RCP_API_KEY unset and not in .env" >&2; exit 1
fi

# Wait for the endpoint to answer, up to WAIT_MAX seconds. This is the whole
# point of the gate: a dropped VPN is the common interruption on this job, and
# the pipeline's own retry is 2 attempts per sentence, which a multi-minute
# outage outlasts. Without the gate, an outage does not pause the run -- every
# sentence fails fast on DNS, the cell "finishes" in a minute having written
# nothing, and the loop marches through the remaining cells doing the same.
# Resume would recover it all on the next launch, but the 80-hour job would be
# over in 20 minutes and you would not find out until morning.
WAIT_MAX="${WAIT_MAX:-3600}"
endpoint_up() {
  curl -s -o /dev/null -m 15 \
    -H "Authorization: Bearer $RCP_API_KEY" \
    https://inference.rcp.epfl.ch/v1/models 2>/dev/null
}
await_endpoint() {
  local waited=0
  endpoint_up && return 0
  echo "    endpoint unreachable -- waiting (up to ${WAIT_MAX}s)" >&2
  while (( waited < WAIT_MAX )); do
    sleep 30; waited=$((waited + 30))
    if endpoint_up; then
      echo "    endpoint back after ${waited}s" >&2; return 0
    fi
  done
  echo "FAIL: endpoint still unreachable after ${WAIT_MAX}s" >&2
  return 1
}

export PYTHONPATH=".:src/multi_agent_annotation${PYTHONPATH:+:$PYTHONPATH}"
PY=".mobiko-venv/bin/python"
PLAN="$(mktemp)"
trap 'rm -f "$PLAN"' EXIT

# Preflight. Refuses to launch on a moved guideline hash, a withdrawn model or a
# precision-suffixed id -- the three faults that produce output which cannot be
# cited and are invisible in the records.
if ! $PY scripts/xrun/grid_config.py --check --corpus "$CORPUS" --arms "${ARMS[@]}"; then
  echo "FAIL: preflight failed; not launching" >&2; exit 1
fi

$PY scripts/xrun/grid_config.py --print --corpus "$CORPUS" --arms "${ARMS[@]}" > "$PLAN" || exit 1
# shellcheck disable=SC1090
eval "$(grep '^export ' "$PLAN")"

LOGDIR="logs/grid_${CORPUS}"
mkdir -p "$LOGDIR" data/auto_annotated

# ONLY=<substring> restricts the cells, so several models can run at once in
# separate shells without racing for the same output file. Sequential within a
# shell is deliberate: the cells share one RCP endpoint.
mapfile -t CELLS < <(grep -v '^export ' "$PLAN" | { [[ -n "${ONLY:-}" ]] && grep -- "$ONLY" || cat; })
if [[ ${#CELLS[@]} -eq 0 ]]; then
  echo "FAIL: ONLY='${ONLY:-}' matched no cells" >&2; exit 1
fi
echo
echo "corpus $CORPUS   arms ${ARMS[*]}   ${#CELLS[@]} cells   logs in $LOGDIR"
echo

n=0; ran=0; skipped=0; failed=0
BAD_MODELS=()
for cmd in "${CELLS[@]}"; do
  n=$((n + 1))
  out="$(sed -n 's/.*--output \([^ ]*\).*/\1/p' <<<"$cmd")"
  name="$(basename "${out%.jsonl}")"
  log="$LOGDIR/$name.log"

  # A model that already failed a cell with the endpoint up will fail its other
  # replicates identically; do not spend hours proving it three times.
  this_model="$(sed -n 's/.*--annotator-model \([^ ]*\).*/\1/p' <<<"$cmd")"
  if [[ " ${BAD_MODELS[*]:-} " == *" $this_model "* ]]; then
    printf '[%2d/%2d] skip  %-52s model already failed\n' \
        "$n" "${#CELLS[@]}" "$name"
    skipped=$((skipped + 1)); continue
  fi

  # "Done" = the output has as many records as the input has sentences. Mere
  # existence is not enough: a cell killed partway leaves a short file that
  # --resume should continue, not one this loop should skip.
  want="$(grep -cve '^[[:space:]]*$' \
      "$(sed -n 's/.*--input \([^ ]*\).*/\1/p' <<<"$cmd")" 2>/dev/null || echo 0)"
  have=0
  [[ -f "$out" ]] && have="$(grep -c '^{' "$out" 2>/dev/null || echo 0)"
  if [[ "$want" -gt 0 && "$have" -ge "$want" ]]; then
    printf '[%2d/%2d] skip  %-52s %s records\n' "$n" "${#CELLS[@]}" "$name" "$have"
    skipped=$((skipped + 1)); continue
  fi

  printf '[%2d/%2d] run   %-52s %s/%s  %s\n' \
      "$n" "${#CELLS[@]}" "$name" "$have" "$want" "$(date +%H:%M)"
  if [[ -n "${DRY_RUN:-}" ]]; then echo "        $cmd --resume"; continue; fi

  await_endpoint || exit 1

  $cmd --resume >>"$log" 2>&1; rc=$?
  after=0
  [[ -f "$out" ]] && after="$(grep -c '^{' "$out" 2>/dev/null || echo 0)"

  if [[ "$after" -le "$have" ]]; then
    # Zero new records: either the network went down, or this model cannot hold
    # the output contract. The two need opposite responses, and re-probing the
    # endpoint separates them. Stopping on both -- which this did at first --
    # means one model that cannot emit the blind-pair JSON halts every cell
    # after it, including the models already known to work. Three of the six
    # have never run this mode.
    failed=$((failed + 1))
    tail -5 "$log" | sed 's/^/        /' >&2
    if endpoint_up; then
      # Endpoint fine, so it is this model. Skip its remaining replicates and
      # carry on with the other models.
      model="$(sed -n 's/.*--annotator-model \([^ ]*\).*/\1/p' <<<"$cmd")"
      BAD_MODELS+=("$model")
      echo "        SKIP MODEL $model: wrote 0 new records, endpoint is up" \
           "(exit $rc) -- see $log" >&2
      continue
    fi
    echo "        STOPPED at $name: 0 new records and the endpoint is down" \
         "(exit $rc)" >&2
    echo "        Nothing is lost; re-run to resume." >&2
    break
  fi

  if [[ "$rc" -eq 0 ]]; then
    ran=$((ran + 1))
  else
    # Progress was made but the cell exited non-zero: partial, and resumable.
    failed=$((failed + 1))
    echo "        PARTIAL $name: $after/$want (exit $rc) -- see $log" >&2
    # Keep going: one bad model should not abandon the other five.
  fi
done

echo
echo "ran $ran   skipped $skipped   failed $failed"
if [[ ${#BAD_MODELS[@]} -gt 0 ]]; then
  echo "models that produced nothing: ${BAD_MODELS[*]}"
  echo "  (endpoint was up, so check that model's log for the parse failure)"
fi
[[ "$failed" -eq 0 ]] || exit 1
