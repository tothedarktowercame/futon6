# mark5 run playbook — clean GPU pass over math.CT (pre-staged)

The validated one-shot sequence, captured 2026-06-22 from a live box run. Every
command here was exercised that day (S0/S3 mark4-proven; S1 dev-validated 199/200;
S4 live 70B 9/9; S5–S8 on real CT). The **candidates are pre-staged on dev**
(`data/iatc-candidates-ct200`, 199 enriched), so the box run is a **pure GPU pass**
— no S1, no fetch.

Prereqs on dev: `data/iatc-candidates-ct200/*.candidate.json` present (rebuild with
`scripts/emit_marks.py --list holes/math-ct-200.ids.txt` then
`scripts/mark3_extract_candidates.py --papers $(cat holes/math-ct-200.ids.txt) --out data/iatc-candidates-ct200`).

## 0. provision (your shell — paid, ~$3/hr)
Per `futon0/README-linode.md`:
```
export LINODE_CLI_TOKEN='<token>'
linode-cli linodes create --region us-ord --type g2-gpu-rtx4000a4-s \
  --image linode/ubuntu24.04 --stackscript_id 2142757 \
  --label mark5-$(date +%Y%m%d) --root_pass "$(openssl rand -base64 18)" \
  --authorized_keys "$(cat ~/.ssh/id_ed25519.pub)"
# wait ~5 min for driver-install + reboot; grab the IP
```

## 1. deploy code + pre-staged candidates (dev shell)  — IP=<box>
```
B=root@$IP
git archive clean-proof-structure | ssh $B 'mkdir -p futon6 && tar -x -C futon6'
tar -cz data/iatc-candidates-ct200 | ssh $B 'tar -xz -C futon6'   # ~2 MB, 199 candidates
```
(no eprints/marks needed on the box — S1 is already done on dev.)

## 2. deps + serve (on box)
```
ssh $B 'cd futon6 && bash scripts/linode-postsetup-deps.sh'                  # bb
ssh $B 'cd futon6 && python3 -m venv .venv && .venv/bin/pip install -q edn_format numpy "huggingface_hub[cli]"'
# pre-pull model (dodge the inline-download stall), then serve:
ssh $B 'cd futon6 && until .venv/bin/hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 >~/hf.log 2>&1; do sleep 8; done'
ssh $B 'cd futon6 && bash scripts/linode-4gpu-setup.sh'                      # serves :8000 as mark4-70b, ~70s load
```

## 3. S3 — IATC loop (GPU, on box) over the pre-staged candidates
```
ssh $B 'cd futon6 && CANDIDATES=data/iatc-candidates-ct200 OUT=data/iatc-argument-graphs/ct200-run bash scripts/linode-4gpu-run.sh'
# linode-4gpu-run verifies candidates are v2-enriched (no local marks -> no re-extract) then runs mark3_iatc_loop
```

## 4. S4 — box-typing (GPU, on box) — model is mark4-70b, NOT the HF id
```
ssh $B 'cd futon6 && .venv/bin/python scripts/clean_box_typing.py \
  --graphs data/iatc-argument-graphs/ct200-run --out holes/clean-ct200 \
  --endpoint http://localhost:8000/v1/chat/completions --model mark4-70b'
# in-loop vocab gate + per-graph argcheck; cyclic-equivalence proofs logged + set aside
```

## 5. S5–S8 — recognition / comprehension / embed+entropy / export (CPU, on box)
```
ssh $B 'cd futon6 && .venv/bin/python scripts/clean_structure_embed.py --clean-dir holes/clean-ct200 --out data/showcases/clean-ct200-demo'
ssh $B 'cd futon6 && .venv/bin/python scripts/clean_entropy_gate.py --embed data/showcases/clean-ct200-demo/clean-embed.json'   # G-entropy gate
ssh $B 'cd futon6 && .venv/bin/python scripts/clean_graph_export.py --clean-dir holes/clean-ct200 \
  --out data/showcases/clean-ct200-demo/ingest --embed-json data/showcases/clean-ct200-demo/clean-embed.json'
# S6 (clean_comprehension) needs the concept substrate (data/warp/*) — ship it or run S6 back on dev.
```

## 6. pull results back + tear down
```
tar -cz -C ~/futon6 holes/clean-ct200 data/showcases/clean-ct200-demo | ... (scp to dev)   # the neo4j cypher + pgvector sql for Rob
linode-cli linodes delete <id>     # your shell — stop the meter
```

### gotchas (live findings, 2026-06-22)
- served model name is **`mark4-70b`** (not the HF id) — S4 `--model mark4-70b`.
- **pre-pull** the model; vLLM's inline download stalls unauthenticated.
- `VLLM_USE_FLASHINFER_SAMPLER=0` + `--enforce-eager` (setup.sh sets both).
- ids are **safe-form** (`math__NNNN`); the apostrophe/slash fixes are in `iatc_to_clean` + `dp_paper_view` + `build_ct_manifest`.
- S6 needs the substrate; everything else is self-contained on the box.
