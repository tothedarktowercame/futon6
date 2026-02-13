#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_FILE="${1:-$ROOT/data/first-proof/problem4-deepdive-results.jsonl}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT/data/first-proof/deepdive-logs/$RUN_STAMP"
PID_FILE="$LOG_DIR/pids.txt"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$RESULTS_FILE")"

echo "# Problem 4 deep-dive launch" > "$PID_FILE"
echo "# UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$PID_FILE"
echo "# results: $RESULTS_FILE" >> "$PID_FILE"
echo >> "$PID_FILE"

launch_job() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  local session="p4deep-${RUN_STAMP}-${name}"
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf "%q" "$arg")")
  done
  local cmd="${quoted[*]}"
  tmux new-session -d -s "$session" "cd $(printf "%q" "$ROOT") && $cmd >> $(printf "%q" "$log") 2>&1"
  local pid
  pid="$(tmux list-panes -t "$session" -F "#{pane_pid}" | head -n 1)"
  echo "$name $pid $session $log" | tee -a "$PID_FILE"
}

PY="python3"
SCRIPT="$ROOT/scripts/verify-p4-deepdive.py"
BASE_ARGS=(--results "$RESULTS_FILE")

# Priority D first, then A/B/C tracks.
launch_job d1 \
  "$PY" "$SCRIPT" "${BASE_ARGS[@]}" --seed 2026021301 d1 \
  --n-min 3 --n-max 8 --trials 50000

launch_job d2 \
  "$PY" "$SCRIPT" "${BASE_ARGS[@]}" --seed 2026021302 d2 \
  --n-min 3 --n-max 7 --trials 12000 --t-grid "0,0.02,0.05,0.1,0.2,0.5,1,2,5"

launch_job a2 \
  "$PY" "$SCRIPT" "${BASE_ARGS[@]}" --seed 2026021303 a2 \
  --n-values "4,5,6" --trials 300 --samples 280

launch_job c1 \
  "$PY" "$SCRIPT" "${BASE_ARGS[@]}" --seed 2026021304 c1 \
  --interior-points 500 --boundary-points 500 --hessian-dx 0.0002

launch_job b1 \
  "$PY" "$SCRIPT" "${BASE_ARGS[@]}" --seed 2026021305 b1

launch_job b2_interval \
  "$PY" "$ROOT/scripts/verify-p4-n4-interval.py"

launch_job d3 \
  "$PY" "$SCRIPT" "${BASE_ARGS[@]}" --seed 2026021306 d3 \
  --numeric-samples 500000

echo
echo "Launched deep-dive jobs."
echo "PID file: $PID_FILE"
echo "Log dir : $LOG_DIR"
