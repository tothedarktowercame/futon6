#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; futon4a="${FUTON4A_ROOT:-/home/joe/code/futon4a}"
runtime="$(mktemp -d)"; port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
cleanup(){ FUTON4A_F2_RUNTIME_DIR="$runtime" "$futon4a/scripts/stop-f2-demo.sh" >/dev/null 2>&1 || true; rmdir "$runtime" 2>/dev/null || true; }; trap cleanup EXIT
FUTON4A_F2_RUNTIME_DIR="$runtime" FUTON4A_F2_PORT="$port" "$futon4a/scripts/launch-f2-demo.sh"
manifest="$(sed -n 's/.*:manifest\/revision-id "\([^"]*\)".*/\1/p' "$runtime/ready.edn")"
F2_SERVICE_URL="http://127.0.0.1:$port" F2_READY_FILE="$runtime/ready.edn" \
  F2_MANIFEST_REVISION="$manifest" FUTON4A_ROOT="$futon4a" node "$root/scripts/f2-cross-client.js"
