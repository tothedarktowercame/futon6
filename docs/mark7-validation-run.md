# Mark7 small validation run (3 papers)

This procedure checks the runner from #52 on real serving hardware at a small scale.
The corpus is `holes/mark7-validation-3.ids.txt`:

| Paper | Proofs identified by S1 | Why it is included |
|---|---|---|
| `1908.02202` | 2 | a plain paper |
| `1205.6522` | 4 | small, with several proofs |
| `0708.1921` | 14 | author macros (`\thm…\eth`, `\prf…\frp`); its paper graph was malformed before the S1 fixes |

The expository cap is 5 regions per paper, fixed in the manifest. A pass qualifies
only this corpus, this cap, and the serving stack used. It says nothing about the
full arXiv math corpus.

**Acceptance** means all of the following:
- preflight and conformance pass;
- S1–S12 are ledgered, and every item-level stage's final invocation has zero rejected and zero errored items;
- replay passes with no warnings;
- the retrieved archive verifies and replays on the receiving machine;
- one interrupted S3 resumes cleanly.

Earlier rejected attempts stay in `stage-attempts.jsonl` and are reported with the
result, never deleted.

A local rehearsal against `scripts/rehearsal_endpoint.py` (synthetic answers, no
model) has already run all of these steps. It exercised interruption and resume,
and found and fixed two runner defects. The rehearsal checks plumbing only; it is
not a validation result.

## 1. Host

**Linode:** `g2-gpu-rtx4000a4-s` (4× RTX4000 Ada), image `linode/ubuntu24.04`,
StackScript `2142757` (NVIDIA driver bootstrap, then reboot). Follow
`futon0/README-linode.md` for creation and the post-reboot `nvidia-smi` check.

**Superpod or another host:** any Linux host with an OpenAI-compatible endpoint
whose grammar enforces JSON schemas. Conformance checks enum, `maxLength`, and
integer `minimum`/`maximum` binding.

## 2. Checkout, substrate, eprints

```bash
mkdir -p ~/code ~/data/eprints && cd ~/code
git clone --branch work/pr51-response https://github.com/tothedarktowercame/futon6.git
git -C futon6 rev-parse HEAD                        # record this
(cd futon6/data && sha256sum -c mark7-ct-substrate.tgz.sha256)
tar -xzf futon6/data/mark7-ct-substrate.tgz -C ~/code   # futon6/data/... and futon3/...
# copy the three eprints into ~/data/eprints: 1908.02202.tar.gz 1205.6522.tar.gz 0708.1921.tar.gz
cd futon6 && scripts/linode-postsetup-deps.sh       # babashka, LaTeXML
```

## 3. Model serving (Linode path)

Download the model with `hf` before serving; vLLM's own downloader has stalled
(see `README-linode.md`):

```bash
python3 -m venv ~/mark4-venv && source ~/mark4-venv/bin/activate
pip install -q -U "huggingface_hub[cli]"
until hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4; do sleep 8; done
scripts/linode-4gpu-setup.sh                         # vLLM 0.23.0, TP=4, served as mark4-70b
pip install -q edn-format sentence-transformers spacy networkx numpy
```

Record the model revision with
`hf download … --dry-run` or the snapshot directory name under
`~/.cache/huggingface/hub/models--hugging-quants--Meta-Llama-3.1-70B-Instruct-AWQ-INT4/snapshots/`.

The largest prompt plus output in this corpus is well under vLLM's default
`MAX_MODEL_LEN=16384`. Larger corpora need a measured value.

## 4. Configuration

```bash
export FUTON6_EPRINTS=~/data/eprints
export FUTON6_PYTHON_CMD="$HOME/mark4-venv/bin/python -u"
export OPENAI_BASE_URL=http://localhost:8000/v1
export MODEL=mark4-70b                               # the *served* name, not the HF id
export FUTON6_MODEL_REVISION=<snapshot hash>
export FUTON6_EXPOSITORY_CAP_PER_PAPER=5
RUN=mark7-val3-<host>-01
```

## 5. Run

```bash
$HOME/mark4-venv/bin/python scripts/preflight.py --ids holes/mark7-validation-3.ids.txt
$HOME/mark4-venv/bin/python scripts/conformance.py --json /tmp/$RUN-conformance.json
$HOME/mark4-venv/bin/python scripts/linode_stepper.py --run --profile linode \
  --run-id $RUN --corpus-id mark7-validation-3 --ids holes/mark7-validation-3.ids.txt \
  --run-dir ~/runs/$RUN --from S1 --reuse S0 STAGE --no-halt 2>&1 | tee ~/runs/$RUN.log
```

**Interruption test.** While S3 is running (some `*.response.json` files exist under
`~/runs/$RUN/artifacts/graphs/.attempts/$RUN/S3-a001/`), kill the runner:
`pkill -9 -f linode_stepper.py; pkill -9 -f mark3_iatc_loop.py`. Then resume:

```bash
$HOME/mark4-venv/bin/python scripts/linode_stepper.py --run --profile linode \
  --run-id $RUN --corpus-id mark7-validation-3 --run-dir ~/runs/$RUN \
  --from S3 --reuse S0 STAGE --no-halt 2>&1 | tee -a ~/runs/$RUN.log
```

`stage-attempts.jsonl` should show `S3-a001 interrupted`, followed by `S3-a002`
carrying the proofs accepted before the kill.

If a stage fails, the runner halts with its item counts and reasons in
`stage-attempts.jsonl` and `accounting/<stage>/<invocation>/`. Record them, then
either re-invoke that stage with `--from`, which retries only non-accepted items,
or stop and report. Never mark a failed stage done.

## 6. Replay, retrieve, verify

```bash
$HOME/mark4-venv/bin/python scripts/replay_e2e.py --run-dir ~/runs/$RUN
$HOME/mark4-venv/bin/python scripts/retrieve_run.py pack --run-dir ~/runs/$RUN --output ~/runs/$RUN.tgz
# copy ~/runs/$RUN.tgz to durable storage; on the receiving machine, with a futon6 checkout:
python3 scripts/retrieve_run.py verify $RUN.tgz --extract-to <durable>/$RUN
```

Tear the host down only after the receiving machine has verified the archive.

## 7. Report

Post these on #52:
- the commit, host, GPU and serving stack, and model revision;
- the conformance JSON;
- for each stage, the per-invocation counts from `stage-attempts.jsonl`;
- every rejection reason, grouped by rule;
- the replay output and the verification result.
