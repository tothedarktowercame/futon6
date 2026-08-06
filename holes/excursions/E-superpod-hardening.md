# E-superpod-hardening — findings from running the mark7 harness off-Superpod

**Opened:** 2026-08-05 (Fable session, Joe driving).
**Method:** the Zone CPU probe (256 GB box, GLM-4.5-Air Q4 via llama.cpp's
OpenAI endpoint, top-100 citation-ranked math.CT, run-id `mark7z`) doubles as
a **staging environment for the superpod runner itself**: every defect it
surfaces is one Rob's 20 h window doesn't have to. This note tracks those
findings to closure. Companion: `holes/mark7-superpod-run-playbook.md` §7
(the run-facing summary); the probe's own artifacts are under
`data/runs/mark7z` and `data/iatc-argument-graphs/run` on Zone.

Status legend: **OPEN** (needs a fix before the Superpod window) ·
**FIXED-UNCOMMITTED** (edited in both working trees, needs commit) ·
**DOCUMENTED** (no code change needed; contract now written down) ·
**ASSET** (not a defect — something the probe made available).

## Findings

### H1 — S2's stepper command is a stub; corpus-fresh substrate build is not wired in. **TIER-1 FIXED · tier-2 OPEN**

**Tier-1 fix (2026-08-05):** new `scripts/warp_substrate_check.py` makes
substrate-corpus match a *measured gate* (verifies the five substrate files,
reports run-ids-covered fraction, fails <95%), and the stepper's S2 cmd is now
`warp_substrate_check --ids {IDS} && coverage_inline --concepts
data/warp/concept-usage.json --field paper_concepts` (the documented
committed-artifact mode). Verified locally: top-100 → 100% match, full 4,616 →
97.7% (substrate holds 9,738 papers). Caveat carried in the stepper note:
committed concept-usage is df≥10-filtered, so the coverage curve reads flat —
the raw-stream instrument (S1 dumping per-paper raw concepts) and full WARP
spine portability (de-hardcoding `warp_run.py`'s dev-box paths) remain
**tier-2 OPEN** and are what "corpus-fresh" needs for any beyond-math.CT run.

`linode_stepper.py` S2 runs `{PY} scripts/coverage_inline.py` bare;
`coverage_inline.py` requires `--concepts <json>` and dies with
`TypeError: expected str, bytes or os.PathLike object, not NoneType`
(reproduced on Zone 2026-08-05; would reproduce identically on the
Superpod). The DAG contract's whole point after mark5 was "S2 MUST be
corpus-fresh," but the stepper has no command that *builds* the WARP spine
(`data/warp/{concept-index,def-snippets,defined-index,concept-usage}.json`)
for the run corpus — and `warp_run.py`, the plausible builder, carries
hardcoded dev-box paths (`/home/joe/code/storage/futon6/data/...` for
eprints and anatomy), so it cannot run on another host as-is.

**Fix needed:** wire a real S2 command: substrate build parameterized by
`FUTON6_EPRINTS`/run ids (de-hardcode `warp_run.py` paths), then
`coverage_inline.py --concepts <the fresh output>`. The mark7z probe
proceeded with `--reuse S2` on the *shipped* substrate — defensible only
because top-100 ⊂ the 4,616-paper corpus that substrate was mined from;
a whole-domain Superpod run has no such out.

### H2 — resume/ledger choreography is undocumented and costs restarts. **FIXED**

**Fix (2026-08-05):** boot steps now print the exact resume incantation
(`--from <next> --reuse <boot stages>`) with an explanation that boot steps
never ledger-record; and both the playbook §3 and the Rob handoff "Run it"
blocks carry the working command verbatim (with `-u`, per H6).

The playbook's turnkey command starts at S0. S0 and STAGE are boot steps:
each prints its note and **exits**; neither writes a ledger entry. So the
documented path is: run → S0 pause → rerun `--from STAGE` → pause → rerun
`--from S1` → **S1 BLOCKED** ("upstream S0 has no passing ledger entry") →
discover `--reuse`. Four invocations, one undocumented flag, mid-window.
The working incantation (verified on Zone) is:

```
linode_stepper.py --run --profile superpod --from S1 --reuse S0 STAGE \
  --ids ... --run-dir ... --corpus-id ... --run-id ... --no-halt
```

**Fix needed:** either have boot steps ledger-record when their note is
printed, or put the `--from S1 --reuse S0 STAGE` form in the playbook's
"Run it" block verbatim.

### H3 — hardcoded LLM timeouts assume GPU throughput. **FIXED** (committed with this excursion's tier-1 batch, 2026-08-05)

`mark3_iatc_loop.py` (300 s), `mark3_expository_loop.py` (300 s),
`clean_box_typing.py` (120 s) all had literal `urlopen(..., timeout=N)`.
A CPU endpoint at ~4 tok/s blows the 300 s ceiling on any real graph
(observed: smoke step 3 died in `readline` with `TimeoutError`). All three
now read `FUTON6_LLM_TIMEOUT` (defaults unchanged, so GPU runs are
unaffected). Edited in the laptop tree and synced to Zone; **needs a
commit**. Batch congestion on the Superpod could plausibly hit the same
ceilings — set the env var there too.

### H4 — window arithmetic is ~2.5× off on proofs-per-paper. **RE-DERIVED in playbook §4**

Playbook §4 budgets ~27k proofs ≈ 6/paper across 4,510 papers. Measured on
the top-100 most-cited: **1,525 all-proofs candidates from 91/100 papers**
(~15.3 extractable proofs per contributing paper). If the citation head is
representative of anything but itself this is a tail-vs-head question, but
the 20 h completion claim should be re-derived before booking; the sweep
design tolerates non-completion, the *expectations* shouldn't.
(Also: 9/100 papers yielded zero candidates — worth a one-line census of
why, since the same rate at 4,616 ≈ 400 papers contributing nothing.)

### H5 — the "4-GPU" wrapper is endpoint-agnostic; its env contract is the
interface. **DOCUMENTED**

`linode-4gpu-run.sh` needs only `PORT/MODEL/REPO/VENV/PYTHON` (+ optional
`IDS_LIST/ALL_PROOFS/RUN_EVAL`): it waits on `/v1/models` and runs the loop
sequentially. It drove llama.cpp on CPU unmodified. The name oversells the
GPU coupling; the env contract above is the real interface and is now
written down here.

### H6 — the stepper's own log is block-buffered; a live run looks stalled. **FIXED (docs)**
`-u` is now in both run commands (playbook §3, handoff).

`linode_stepper.py` run via `nohup ... > log` shows an **empty or stale
log while healthy** (Python buffers; stage children run `-u` but the
stepper's own prints don't flush). On Zone this manufactured a
false "stuck at S0" diagnosis and two unnecessary restarts — the
wall-clock-is-not-stuckness failure mode, self-inflicted. **Fix:** launch
with `python -u` in the playbook command (or `PYTHONUNBUFFERED=1`); trust
the phase ledger and process table over the log tail meanwhile.

### H7 — remote ops footgun: `pkill -f`/`pgrep -f` self-match over ssh. **DOCUMENTED**

Twice during the probe, a `pkill -f linode_stepper` (or a pgrep check)
embedded in an ssh command matched the remote shell carrying that very
string — killing the wrapper before the relaunch, or reporting a dead run
as alive. Discipline: `pkill -x` on the binary name, or split the pattern
(`P=linode; pgrep -f "${P}_stepper"`), and verify via ledger/process table,
never via the incantation that contains the pattern.

### H9 — S7's stepper command hardcoded `--model mark4-70b`. **FIXED**

Any deployment whose served model isn't literally named `mark4-70b` gets
silent 404s (or, worse, a different model that happens to answer). Now
`--model ${MODEL:-mark4-70b}` — same env var the S3 wrapper honors.
Found by static read during the e2e-sample planning (2026-08-06); the
stage had never been executed off the original box.

### H10 — S4's model and region-cap are unwired. **PARTIALLY FIXED**

Two defects: (a) the stepper's S4 cmd never passed `--model`, so the
expository loop fell back to its default `meta-llama/Llama-3.1-8B-Instruct`
— a silent 404 on any server not serving that exact name. FIXED:
`--model ${MODEL:-...}`. (b) The playbook caps S4 at ~30 regions/paper
(one paper had 466), but `mark3_extract_expository_candidates.py` has no
cap flag and the stepper can't express one — the cmd runs ALL regions.
OPEN: the e2e run applies a deterministic out-of-band trim (first 30 per
paper by filename) between extract and loop; the durable fix is a
`--max-regions-per-paper` flag threaded through both scripts.

### H11 — expository loop had no per-candidate endpoint-error containment. **FIXED**

One oversized region (context-exceeded → HTTP 500) killed the whole S4
batch (observed on Zone 2026-08-06; the IATC loop already contains these
per-candidate). Now: try/except around the call, `last_error` recorded,
and 400/413/500 stop retrying that candidate (same prompt cannot shrink).
Sizing note for concurrent legs: llama.cpp's unified KV pool is shared
across slots, so `-c` must cover the SUM of concurrent prompts — S4's
region prompts + S7's graph prompts overflowed 32k; Zone now runs
`-c 65536`. On the Superpod, vLLM sizing differs but the same additive
logic applies to `max-model-len` × concurrency.

### H8 — model-sensitivity comparison now available. **ASSET**

The mark7z artifacts (GLM-4.5-Air, gates enforced) give a same-corpus
comparison set against any Superpod run (70B-class). First data point: the
smoke proof (`math__0608040` p0) generated a graph that passed argcheck +
substance on the first attempt. Gate pass-rates per model, per paper tier,
fall out of the two runs' ledgers for free.

## Close-out criteria

Tier-1 close-out (H1t1/H2/H3/H4/H6) landed 2026-08-05 in one commit; the
corrected S2 was exercised for real on Zone (see Log). **Remaining before
full close:** H1 tier 2 — the raw-concept-stream instrument and WARP spine
portability (de-hardcode `warp_run.py` + per-script paths) — required for
any beyond-math.CT domain run; plus an eventual clean dry pass on the
Superpod profile itself. H5/H7/H8 are documentation and carry no code.

## Measurements (Zone, GLM-4.5-Air Q4, 32-core CPU)

- Single-stream decode: **4.1 t/s**. Two concurrent streams: 3.70 + 2.81 =
  **6.5 t/s aggregate = 1.58×** — the MoE batching penalty is real (expert
  reads don't amortize across sequences the way dense weights do) but
  two-way sharding still buys ~1.6×.
- **Mixed prefill/decode interference:** while one slot prompt-processes a
  ~5k-token candidate, the other slot's decode drops to ~1 t/s. With more
  streams this worsens; 2-way is likely the sweet spot on this box.
- `mark3_iatc_loop.py:311` resumes (`final.exists()` skip), so sharding is
  adoptable mid-run at zero cost — stop, split candidates into disjoint
  dirs, relaunch N loops at the same `--out`.

## Log

- 2026-08-05 — opened with H1–H8 from the Zone probe, same day as the
  probe itself. mark7z S3 loop running (1,525 candidates) at time of
  writing. Two-stream throughput measured live (see Measurements).
