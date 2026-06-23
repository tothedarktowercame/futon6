# mark7 — handoff for Rob (full math.CT, one 20-hour Superpod window)

One run, full primary math.CT (4,616 papers), every lesson from our mark5/mark6 work baked
in, instrumented as an **accretion sweep** so the window yields rising "improve-as-we-run"
curves even if it doesn't finish. Deep design + learning goals: `mark7-superpod-run-playbook.md`.
This doc is just **what you do**.

## What we hand you (3 things)

1. **The repo** — `futon6` at `master` (latest). The pipeline + the `linode_stepper.py`
   runner (it has a `superpod` profile).
2. **`data/mark7-substrate.tgz`** (15 MB, **committed in the repo** — your clone already has it;
   nothing to transfer out-of-band). The concept substrate + futon3 pattern library, symlinks
   dereferenced. Extract at `~/code/`: `tar -xzf futon6/data/mark7-substrate.tgz -C ~/code/`
   (lands as `futon6/data/...` and `futon3/...`).
3. **The manifests** (in the repo): `holes/math-ct-full.ids.txt` (citation-ranked, the
   default) and `holes/math-ct-chrono.ids.txt` (chronological alternative).

## What you fill in (your cluster — 3 things)

1. **`export FUTON6_EPRINTS=<your arXiv-math eprint dir>`** — the pipeline reads eprints from
   here instead of fetching (you already have all of arXiv math). **It must resolve our ids**:
   for id `math__0608040` it looks for `$FUTON6_EPRINTS/math__0608040.tar.gz` (also tries
   `.gz`/`.tex`). Our ids are safe-form (`/`→`__`): `math__0608040`, `2311.05789`. **If your
   store is named differently** (e.g. `math/0608040/…`), tell us the layout and we'll add a
   one-line path adapter — don't burn window time on it.
2. **Serve your model** OpenAI-compatible across the 8 GPUs (vLLM does this), then
   `export OPENAI_BASE_URL=http://localhost:<port>/v1 OPENAI_API_KEY=x` and pass `--model <id>`.
3. **Extract the substrate**: `tar -xzf mark7-substrate.tgz -C ~/code/`.

## SMOKE TEST FIRST (5 min — do NOT skip before a 20h window)

Confirms eprints resolve + the model serves, on ONE paper:

```bash
cd ~/code/futon6
FUTON6_EPRINTS=$YOUR_EPRINTS .venv/bin/python scripts/emit_marks.py --list <(echo math__0608040)   # eprint resolves?
OPENAI_BASE_URL=$URL OPENAI_API_KEY=x .venv/bin/python scripts/mark3_extract_candidates.py --papers math__0608040 --all-proofs --out /tmp/sc
OPENAI_BASE_URL=$URL .venv/bin/python scripts/mark3_iatc_loop.py --candidates /tmp/sc --out /tmp/sg --backend openai --model $MODEL  # model serves?
```

If both produce output, you're configured. If `emit_marks` errors with "no eprint for …",
the `FUTON6_EPRINTS` naming doesn't match — ping us.

## Run it

```bash
cd ~/code/futon6
# (substrate extracted; model served; FUTON6_EPRINTS exported)
FUTON6_EPRINTS=$YOUR_EPRINTS OPENAI_BASE_URL=$URL OPENAI_API_KEY=x RUN_ID=mark7 CORPUS=math-ct-full \
  .venv/bin/python scripts/linode_stepper.py --run --profile superpod \
    --run-dir data/runs/mark7 --corpus-id math-ct-full --run-id mark7
```

- Set the run's id-list to `holes/math-ct-full.ids.txt` (citation-ranked) — process most-cited
  first so the backbone is covered early and a short window still gets the important papers.
- It **halts at each stage gate** for a glance; the completeness ledger refuses any stage whose
  upstream didn't run for this corpus. Stage order: `S1 anatomy · S2 concepts · S3 IATC ·
  S4 expository · S5 comprehension · S6 paper-graph · S7 CLean-embed · S8 export · S9 APM ·
  S10 lexicon+reground · S11 structural+whole-paper · S12 accretion-sweep`.
- ~28k all-proofs; full completion in 20h is plausible with 8-GPU batching, but **the run need
  not finish** — S12 checkpoints the curves as it goes.

## Send back (BEFORE you release the alloc — we lost mark6's CLeans this way)

`rsync` these to us:
- `data/iatc-argument-graphs/mark7` (IATC graphs) · `holes/clean-mark7` (CLeans EDN)
- `data/iatc-paper-graphs/mark7` (object B) · `data/showcases/clean-mark7-demo` (the structure
  embedding — your CLean index) · `data/expository-scope-graphs/mark7`
- `data/runs/mark7` (metrics, ledger, harvested lexicons, accretion curves)

That's it. The one thing that needs your eyes before committing the window is the **smoke
test** — everything else is turnkey.
