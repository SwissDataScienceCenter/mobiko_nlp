#!/usr/bin/env bash
# Launch run_selfconsistency.sh on the RunAI pod from a local PyCharm run config.
#
# This script runs LOCALLY. PyCharm's Shell Script configurations have no remote
# interpreter option (only Python configurations do), so the remote hop lives
# here rather than in the run configuration.
#
# Two transports:
#   TRANSPORT=runai (default) — `runai training exec` into the pod. Needs no ssh,
#       no password, no port-forward: the RunAI CLI is already authenticated.
#   TRANSPORT=ssh             — ssh to $HOST. Requires a live `runai training
#       port-forward` AND a public key in the pod's /myhome/.ssh/authorized_keys.
#       Use this only if you specifically want the ssh path; runai is sturdier.
#
# Usage
#   scripts/xrun/run_selfconsistency_remote.sh                  # foreground; Ctrl-C kills the remote run
#   DETACH=1 scripts/xrun/run_selfconsistency_remote.sh         # survives disconnects, then tails the log
#   PLAN=1   scripts/xrun/run_selfconsistency_remote.sh         # print the remote plan, run nothing
#   SYNC=1   scripts/xrun/run_selfconsistency_remote.sh         # rsync code first (ssh transport only)
#   MODEL=rcp-kimi-2.7 TAG=kimi27 SENTENCES=56 SHARDS=8 scripts/xrun/run_selfconsistency_remote.sh
#
# Every knob of run_selfconsistency.sh (MODEL, TAG, K, SENTENCES, SHARDS, INPUT,
# OUTDIR, RESUME, PLAN) is forwarded verbatim, so the two scripts share one
# interface — this one only adds where it runs.

set -uo pipefail

# ── remote target (override by env var) ─────────────────────────────────────
TRANSPORT="${TRANSPORT:-runai}"           # runai | ssh
POD="${POD:-simplegpu5}"                  # RunAI workload name
PROJECT="${PROJECT:-mobiko-anisia}"
HOST="${HOST:-runai}"                     # ssh transport only (~/.ssh/config: localhost:8888)
REMOTE_ROOT="${REMOTE_ROOT:-/mydata/mobiko/anisia}"    # writable PVC, = PyCharm deployment mapping
REMOTE_PYTHON="${REMOTE_PYTHON:-/myhome/.virtualenvs/mobiko_nlp/bin/python}"
DETACH="${DETACH:-0}"
SYNC="${SYNC:-0}"
TAIL="${TAIL:-1}"                         # after DETACH=1, follow the log

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LOCAL_ROOT" || exit 1

# ── env forwarded into the remote run ───────────────────────────────────────
# PYTHON is NOT in this list: it is set from REMOTE_PYTHON below, since the
# local interpreter path is meaningless on the pod.
FORWARD=(MODEL TAG K SENTENCES SHARDS INPUT OUTDIR RESUME PLAN
         MOBIKO_ENV_FILE ADJUDICATOR_MODEL ANNOTATOR_TOP_LOGPROBS
         RCP_ENABLE_THINKING DEPENDENCY_RELATION_HINTS
         GUIDELINE_SEARCH_BACKEND GUIDELINE_SEARCH_EMBEDDING_MODEL)
env_prefix=""
for v in "${FORWARD[@]}"; do
  [[ -n "${!v:-}" ]] && env_prefix+="$(printf '%s=%q ' "$v" "${!v}")"
done
env_prefix+="$(printf 'PYTHON=%q ' "$REMOTE_PYTHON")"

# Only used to name the detached launcher log; run_selfconsistency.sh remains
# the single source of truth for the actual defaults.
LOG="${OUTDIR:-output/selfconsistency}/${TAG:-qwen36_35B}_K${K:-10}.remote.log"

echo "======================================================================"
echo "  REMOTE SELF-CONSISTENCY LAUNCH"
echo "======================================================================"
echo "  transport   $TRANSPORT$([[ "$TRANSPORT" == "runai" ]] && echo "  ($POD @ $PROJECT)" || echo "  ($HOST)")"
echo "  remote root $REMOTE_ROOT"
echo "  remote py   $REMOTE_PYTHON"
echo "  mode        $([[ "$DETACH" == "1" ]] && echo "detached (log: $LOG)" || echo "foreground")"
echo "  forwarding  ${env_prefix}"
echo

# ── optional code push (ssh transport only) ─────────────────────────────────
# Code only — output/ is deliberately never pushed, it would clobber remote
# results. Under TRANSPORT=runai, sync via PyCharm's auto-upload or kubectl cp.
if [[ "$SYNC" == "1" ]]; then
  if [[ "$TRANSPORT" != "ssh" ]]; then
    echo "  NOTE: SYNC=1 needs TRANSPORT=ssh (rsync has no runai transport); skipping." >&2
  else
    echo "  syncing scripts/ src/ -> $HOST:$REMOTE_ROOT"
    rsync -az --exclude '__pycache__/' --exclude '*.pyc' \
        scripts src "$HOST:$REMOTE_ROOT/" || exit 1
  fi
  echo
fi

# ── build the remote payload ────────────────────────────────────────────────
# Preflight checks live inside the payload so the whole thing is one round trip.
q_root=$(printf %q "$REMOTE_ROOT")
q_py=$(printf %q "$REMOTE_PYTHON")
q_log=$(printf %q "$LOG")

read -r -d '' remote_pre <<EOF
set -uo pipefail
cd $q_root || { echo "ERROR: remote repo not found: $REMOTE_ROOT" >&2; exit 2; }
[ -x $q_py ] || { echo "ERROR: remote python missing or not executable: $REMOTE_PYTHON" >&2; exit 2; }
[ -f scripts/xrun/run_selfconsistency.sh ] || { echo "ERROR: scripts/xrun/run_selfconsistency.sh not on remote — upload it first" >&2; exit 2; }
EOF

if [[ "$DETACH" == "1" ]]; then
  remote_cmd="$remote_pre
mkdir -p \"\$(dirname $q_log)\"
nohup env $env_prefix bash scripts/xrun/run_selfconsistency.sh > $q_log 2>&1 < /dev/null &
echo \"  launched pid \$! on \$(hostname), log: $LOG\""
else
  remote_cmd="$remote_pre
exec env $env_prefix bash scripts/xrun/run_selfconsistency.sh"
fi

# ── dispatch ────────────────────────────────────────────────────────────────
# NOTE: `runai training exec` collapses any nonzero remote exit code to 1, so
# pass/fail is trustworthy but the specific code is not. Read the log for detail.
run_remote() {   # $1 = command string, $2 = "tty" to request a tty
  if [[ "$TRANSPORT" == "runai" ]]; then
    local tty_flags=()
    # Only ask for a tty when stdin actually is one, else the exec errors out
    # in a non-terminal PyCharm run window.
    [[ "${2:-}" == "tty" && -t 0 ]] && tty_flags=(-i -t)
    runai training exec "$POD" -p "$PROJECT" "${tty_flags[@]}" -- bash -c "$1"
  else
    # -tt forces a tty so Ctrl-C reaches the remote shards.
    if [[ "${2:-}" == "tty" ]]; then ssh -tt "$HOST" "$1"; else ssh "$HOST" "$1"; fi
  fi
}

if [[ "$DETACH" == "1" ]]; then
  run_remote "$remote_cmd" || exit 1
  if [[ "$TAIL" == "1" ]]; then
    echo
    echo "  following the log — Ctrl-C stops the tail, NOT the run"
    echo
    run_remote "cd $q_root && tail -n +1 -f $q_log" tty
  else
    echo
    if [[ "$TRANSPORT" == "runai" ]]; then
      echo "  follow with: runai training exec $POD -p $PROJECT -- tail -f $REMOTE_ROOT/$LOG"
    else
      echo "  follow with: ssh $HOST 'tail -f $REMOTE_ROOT/$LOG'"
    fi
  fi
else
  run_remote "$remote_cmd" tty
fi
