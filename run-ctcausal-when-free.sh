#!/usr/bin/env bash
# Queue: wait for shard-a IATC loop to drain, then run the ct-causal corpus
# (E-mining-qual-loop / strange-loop set) as the replacement stream.
set -u
cd "$HOME/code/futon6"
while pgrep -f "mark3_iatc_loop.*candidates-run-a" >/dev/null; do sleep 300; done
[ -d data/iatc-candidates-ctcausal ] || { echo "no ctcausal candidates"; exit 1; }
OPENAI_BASE_URL=http://127.0.0.1:8090/v1 OPENAI_API_KEY=x FUTON6_LLM_TIMEOUT=7200 \
  exec .venv/bin/python -u scripts/mark3_iatc_loop.py \
    --candidates data/iatc-candidates-ctcausal \
    --out data/iatc-argument-graphs/run-ctcausal \
    --backend openai --model glm-4.5-air
