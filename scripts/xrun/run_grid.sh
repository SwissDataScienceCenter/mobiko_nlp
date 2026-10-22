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
for cmd in "${CELLS[@]}"; do
  n=$((n + 1))
  out="$(sed -n 's/.*--output \([^ ]*\).*/\1/p' <<<"$cmd")"
  name="$(basename "${out%.jsonl}")"
  log="$LOGDIR/$name.log"

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

  if $cmd --resume >>"$log" 2>&1; then
    ran=$((ran + 1))
  else
    failed=$((failed + 1))
    echo "        FAILED (exit $?) -- see $log" >&2
    tail -3 "$log" | sed 's/^/        /' >&2
    # Keep going: one unreachable model should not abandon the other five.
  fi
done

echo
echo "ran $ran   skipped $skipped   failed $failed"
[[ "$failed" -eq 0 ]] || exit 1
