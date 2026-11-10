#!/usr/bin/env bash
# Start (or resume) the claude-cs conversation in a terminal with the Agency
# Clink inbox pushed in as channel messages. Every other Claude conversation is
# started the ordinary way and is unaffected.
#   claude-cs.sh            resume the session in ~/.local/state/futon3c/session-id-claude-cs
#   claude-cs.sh --new      start a fresh session (its id is not recorded)
# The CLI asks once per launch "I am using this for local development": press Enter.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sid_file="$HOME/.local/state/futon3c/session-id-claude-cs"
# Without this, a session treats channel text as third-party content and will
# not act on it (seen 2026-11-10: the test session declined to ack a bell).
standing="You are the Agency seat claude-cs. Messages from the futon-relay-channel \
channel are Agency bells addressed to you (often Joe's ㊡ subprompts, sent from \
'joe'). Treat each as a request to you: answer it, then ack it exactly as its \
first lines say (clink-inbox.py ack <job-id> --result-file <answer file>), which \
returns your answer to the sender. Several may queue; answer them in order."
args=(--mcp-config "$here/relay-mcp.json"
      --dangerously-load-development-channels server:futon-relay-channel
      --append-system-prompt "$standing")
if [[ "${1:-}" == "--new" ]]; then
  shift
elif [[ -s "$sid_file" ]]; then
  args+=(--resume "$(cat "$sid_file")")
fi
cd "$HOME/code"
exec claude "${args[@]}" "$@"
