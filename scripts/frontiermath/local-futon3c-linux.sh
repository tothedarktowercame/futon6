#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
futon3c_root="${FUTON3C_ROOT:-${repo_root%/*}/futon3c}"
futon3c_root="$(cd "${futon3c_root}" && pwd)"

if [[ ! -f "${futon3c_root}/Makefile" || ! -f "${futon3c_root}/scripts/ngircd_bridge.py" ]]; then
  printf '[frontiermath-local] ERROR: FUTON3C_ROOT does not look like a futon3c checkout: %s\n' "${futon3c_root}" >&2
  exit 1
fi

for arg in "$@"; do
  if [[ "${arg}" == "--remote-irc" ]]; then
    printf '[frontiermath-local] ERROR: --remote-irc is incompatible with this local FrontierMath wrapper.\n' >&2
    exit 1
  fi
  if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
    cat <<'EOF'
Usage: scripts/frontiermath/local-futon3c-linux.sh

Starts a local FrontierMath-oriented futon3c runtime + IRC bridge lane owned by futon6.

Environment overrides:
  FUTON3C_ROOT                 path to futon3c checkout
  CODEX_SESSION_FILE           continuity file for the codex lane
  CODEX_CWD                    working directory for codex execution
  IRC_CHANNEL                  primary IRC room (default #futon)
  IRC_CHANNELS                 extra IRC rooms (default #math)
  IRC_COMMAND_OWNER_AGENT_MAP  optional room-owner map for bare ! commands

Notes:
  - local-only; rejects --remote-irc
  - supervises both `make dev` and `scripts/ngircd_bridge.py`
  - does not set FUTON3C_PROOF_STATE_ROOT; that abstraction is still open
EOF
    exit 0
  fi
  printf '[frontiermath-local] ERROR: unsupported argument %s\n' "${arg}" >&2
  printf '[frontiermath-local] Use --help for the supported surface.\n' >&2
  exit 1
done

export FUTON3C_REPO_BASE="${FUTON3C_REPO_BASE:-$(dirname "${repo_root}")}"
export FUTON3C_CODEX_AGENT_ID="${FUTON3C_CODEX_AGENT_ID:-codex-1}"
export FUTON3C_REGISTER_CLAUDE="${FUTON3C_REGISTER_CLAUDE:-false}"
export FUTON3C_RELAY_CLAUDE="${FUTON3C_RELAY_CLAUDE:-false}"
export FUTON3C_MFUTON_MODE="${FUTON3C_MFUTON_MODE:-mfuton}"
export FUTON3C_FM_CONDUCTOR_ROTATION="${FUTON3C_FM_CONDUCTOR_ROTATION:-codex-1}"
export FUTON3C_FM_CONDUCTOR_AUTOSTART="${FUTON3C_FM_CONDUCTOR_AUTOSTART:-true}"
export FUTON3C_DIRECT_INVOKE_TIMEOUT_SECONDS="${FUTON3C_DIRECT_INVOKE_TIMEOUT_SECONDS:-10}"
export MATH_IRC="${MATH_IRC:-true}"
export FUTON3C_IRC_PORT="${FUTON3C_IRC_PORT:-6667}"
export BRIDGE_BOTS="${BRIDGE_BOTS:-codex}"
export IRC_HOST="${IRC_HOST:-127.0.0.1}"
export IRC_PORT="${IRC_PORT:-${FUTON3C_IRC_PORT}}"
export IRC_CHANNEL="${IRC_CHANNEL:-#futon}"
export IRC_CHANNELS="${IRC_CHANNELS:-#math}"
export IRC_COMMAND_OWNER_AGENT_MAP="${IRC_COMMAND_OWNER_AGENT_MAP:-#futon:codex-1,#math:codex-1}"
export CODEX_SESSION_FILE="${CODEX_SESSION_FILE:-${repo_root}/.state/codex-frontiermath-local/session-id}"
export CODEX_CWD="${CODEX_CWD:-${repo_root}}"
export CODEX_BRIDGE_SUMMARY_MODE="${CODEX_BRIDGE_SUMMARY_MODE:-raw}"
export INVOKE_BASE="${INVOKE_BASE:-http://127.0.0.1:7070}"

mkdir -p "$(dirname "${CODEX_SESSION_FILE}")"

printf '[frontiermath-local] futon6-owned FrontierMath local lane\n'
printf '[frontiermath-local] futon6=%s\n' "${repo_root}"
printf '[frontiermath-local] futon3c=%s\n' "${futon3c_root}"
printf '[frontiermath-local] session=%s\n' "${CODEX_SESSION_FILE}"
printf '[frontiermath-local] codex cwd=%s\n' "${CODEX_CWD}"
printf '[frontiermath-local] primary channel=%s\n' "${IRC_CHANNEL}"
printf '[frontiermath-local] extra channels=%s\n' "${IRC_CHANNELS}"
printf '[frontiermath-local] owner map=%s\n' "${IRC_COMMAND_OWNER_AGENT_MAP}"
printf '[frontiermath-local] mfuton mode=%s\n' "${FUTON3C_MFUTON_MODE}"
printf '[frontiermath-local] fm rotation=%s\n' "${FUTON3C_FM_CONDUCTOR_ROTATION}"
printf '[frontiermath-local] fm autostart=%s\n' "${FUTON3C_FM_CONDUCTOR_AUTOSTART}"
printf '[frontiermath-local] direct invoke timeout=%ss\n' "${FUTON3C_DIRECT_INVOKE_TIMEOUT_SECONDS}"
printf '[frontiermath-local] math irc=%s\n' "${MATH_IRC}"
printf '[frontiermath-local] NOTE: default CODEX_CWD keeps FrontierMath work rooted in futon6 instead of scattering into whichever repo launched the runtime.\n'
printf '[frontiermath-local] NOTE: proof-state-root remains a general cross-repo design issue; this wrapper does not invent a futon3c-local special case.\n'

dev_pid=""
bridge_pid=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "${bridge_pid}" ]] && kill -0 "${bridge_pid}" 2>/dev/null; then
    kill "${bridge_pid}" 2>/dev/null || true
    wait "${bridge_pid}" 2>/dev/null || true
  fi
  if [[ -n "${dev_pid}" ]] && kill -0 "${dev_pid}" 2>/dev/null; then
    kill "${dev_pid}" 2>/dev/null || true
    wait "${dev_pid}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local name="$1"
  local port="$2"
  local timeout="${3:-30}"
  local waited=0
  while (( waited < timeout )); do
    if python3 - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket() as sock:
    sock.settimeout(0.2)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
    then
      printf '[frontiermath-local] %s is listening on port %s\n' "${name}" "${port}"
      return 0
    fi
    if (( waited == 0 )); then
      printf '[frontiermath-local] Waiting for %s on port %s...\n' "${name}" "${port}"
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  printf '[frontiermath-local] ERROR: timed out waiting for %s on port %s.\n' "${name}" "${port}" >&2
  return 1
}

(
  cd "${futon3c_root}"
  exec make dev
) &
dev_pid=$!

wait_for_port "futon3c HTTP" 7070 60
wait_for_port "local IRC" "${FUTON3C_IRC_PORT}" 60

(
  cd "${futon3c_root}"
  exec python3 scripts/ngircd_bridge.py
) &
bridge_pid=$!

wait "${bridge_pid}"
