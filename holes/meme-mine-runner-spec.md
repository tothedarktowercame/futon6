# MEME-MINE runner spec (the GPU mining runner for M-operational-vocabulary)

**Date:** 2026-06-25 · **Owner:** Joe + claude-1 · **Status:** SPEC (v0 validated on an 11-ask sample, CPU; scale on a Linode GPU).
**For:** [[M-operational-vocabulary]] — mine human→agent turns into `(have, want)` memes that ground the WM's move-priors. Validated by `futon6/scripts/mission_mine_memes.py` + `futon6/data/meme-mine/`.

**Runner BUILT (2026-06-25):** `futon6/scripts/meme_mine_runner.py` — all three layers, stub-validated end-to-end (turn-read + auto/system exclusion + thread-window + Layer-1 `香` pre-tag + evidence-check + Layer-3 dedup-report + output); the `openai` Layer-2 path mirrors the proven `sfc_symbol_grounding.py` vLLM call, GPU-ready.

### Run
**One command (mirrors `linode-4gpu-run.sh`):** `scripts/linode-meme-mine.sh` — waits for vLLM, runs Layer 2 (`--backend openai`), then the non-fatal CPU consume tail (bridge → floor/cert → concept-tag) + owner-review notes. Env knobs: `PORT MODEL REPO VENV PYTHON TURNS_DIR LIMIT`.

**Do NOT rsync futon6 to the box — it's ~31 GB of corpora+venv this run never touches.** Only the GPU *inference* belongs on the box; everything else is local CPU over ~17 MB.
```bash
# RECOMMENDED — tunnel, sync NOTHING. Run on dev; the box only serves vLLM (it pulls weights from HF):
ssh -L 8000:localhost:8000 <box> &
OPENAI_BASE_URL=http://localhost:8000/v1 scripts/linode-meme-mine.sh        # full joint mine
# smoke (no GPU): futon6/.venv/bin/python scripts/meme_mine_joint.py --backend stub --limit 8
# ON-BOX (only if you must): copy ONLY these scripts + data/{diffsub-scopes,diffsub-moves-mined,
#   capability-graph} + ../futon3a/resources/notions/minilm_{pattern,mission}_embeddings.json
#   + ~/.claude/projects/ (~17 MB). The .py hardcode /home/joe/code paths → need those paths or a fix.
```
Output: `data/meme-mine/{joint-memes,resolved-memes}.openai.json · diffsub-moves-meme.edn · action-cert.json · concept-index.json` + a self-describing report.

### Smoke-test gate — run BEFORE the full paid pass (added 2026-06-25 after the first run shipped ~50% bad)
Stub mode does NOT exercise the prompt; only a real-backend smoke does. Do `--limit 12 --backend openai`
first and eyeball these cheap ratios; scale to the full corpus only when they're in band:
- **op operational-share** — ops should be MOVE-CLASS verbs (build/dispatch/deploy/wire/find/refine…), not
  discourse acts. Expect ≳70% operational + the rest `op="none"`; **discourse verbs (elaborate/contrast/
  compare/request) are a fail** — that was F3 (≈50% discourse on the v1 run).
- **endpoint tier split** — expect roughly contextual-majority with named anchors + some unsupported (the
  sample was contextual 59% / named 23% / unsupported 18%); **named ≫ contextual is a fail** (F4: over-grounding).
- **new_patterns firing rate** — expect a SMALL minority of asks to propose one (load-bearing op, no candidate
  fits); **≈1-per-ask is a fail** (F2: the v1 run fired on 95%).
Provenance is already enforced upstream: `read_asks` gates on `transcript_provenance.is_operator` (E-patch fix).

## The pipeline — three layers (deterministic brackets, LLM core)

The unification is **`SFC2b`-then-`SFC-NORM`**: conversational endpoints must be *grounded to a referent* (LLM) **before** the cut-and-dried exact-merge (CPU) applies. SFC-NORM is CPU only because proof concepts arrive near-canonical; meme endpoints arrive as free text, so the make-it-canonical step is an LLM turn.

1. **Layer 1 — CPU `香` pre-tag (salience).** Exact/alias-match endpoint phrases against the registry of known ids (agent-ids, R-numbers, mission stems, capabilities, patterns, file paths, technologies). Cheap; this is also the existing turn↔pattern tagging. Catches the *named* endpoints (≈23%).
2. **Layer 2 — LLM extract + resolve + cite (GPU; ONE pass).** Read the ask **in its thread window** (see finding C) and emit the meme `(have, want, op, maturity)` **and** resolve each endpoint to a referent, **citing verbatim evidence or marking `:unsupported`** (the SFC2b discipline → avoids `間` false-salience). Folded into extraction because resolving "it" / "the CT prior" needs the same context the extraction already loaded — one read, not two.
3. **Layer 3 — CPU dedup (`SFC-NORM`).** Once endpoints carry canonical ids, exact-merge on **`(have.ref, want.ref, op)`** onto the arrow store (endpoint-identity, unify-not-mint).

## Resolved-meme schema (locked; exemplar = `data/meme-mine/resolved-memes.json`)

```
{ id, ask, provenance:{project, session},
  meme:{ have:{text, ref|null, tier:named|contextual|unsupported, evidence},
         want:{text, ref|null, tier, evidence},
         op, maturity:open|correlated|constructed, salience_terms[] } }
```
`ref` is a canonical id (`agent/…`, `hole/…`, `artifact/…`, `concept/…`, `task/…`, `feature/…`, mission stem, `scope/capability/…`) or `null` when `:unsupported`.

## Sample validation (11 asks, CPU + claude-1-as-LLM)

- **Op vocabulary (empirical):** dispatch · relate · find · build · deploy · assign · preregister · reconstruct · reuse · investigate — ~10 classes from 11 asks vs 3 hand-coded.
- **Endpoint resolution tiers (22 endpoints):** contextual **59%** · named **23%** · unsupported **18%** → resolution belongs in Layer 2 (LLM), CPU pre-pass handles the named quarter.
- **Dedup:** 8/11 fully resolved → 8 unique `(have,want,op)` keys, 0 collisions at N=11 (expected); named anchors `{R2d×2, claude-4, codex-1, tech}` are the unification points that collide at scale.

## Findings the runner must honour

- **A — dedup key includes `op`.** Self-edge memes (have.ref == want.ref, e.g. `build CT-prior` vs `dispatch R2d`) would false-merge on `(have,want)` alone; the op disambiguates.
- **B — pure-op memes exist.** Some asks are operations with no entity endpoints (e.g. *"hot reload and commit"* → both `:unsupported`). Keep them — they enrich the op-vocabulary — but flag op-only; they don't attach to a mission's prior.
- **C — resolve over a thread window, not the isolated turn.** *"Please redispatch"* (the work) has its antecedent in a prior turn. The runner feeds the LLM the conversation window, not the single turn.
- **D — `:unsupported` wants are SORRY-MINE's territory.** A want that is a *search target* (*"the commit that broke it"*) is legitimately `:unsupported`-until-found — an open arrow whose `want` is a query. These are the duals SORRY-MINE harvests from agent→human turns.
- **E — named anchors are the grounding bridge.** A meme whose `ref` is a mission stem / `hole/…` / `scope/capability/…` is what *replaces* a structurally-borrowed weak move-prior with the actually-asked-for operation (the M-operational-vocabulary frontier payoff). Layer 1 + Layer 3 hinge on these.

## Inputs / outputs

- **In:** `~/.claude/projects/*/*.jsonl` (turns, with thread order); the registry for Layer 1 (`futon3a` pattern/agent ids, star-map capabilities, mission stems, R-numbers).
- **Out:** `data/meme-mine/resolved-memes.json` → Layer-3 unify → arrow store (`futon3a` memes, `(have,want)` with `:advances-cap` *declared* per the M-wm-policies seam, never inferred). The named-anchor memes then feed `mission_mine_moves.py` as real provenance, replacing the borrowed structural priors.

## GPU vs CPU split (the answer to "build unification into the GPU runner?")

- **GPU:** Layer 2 (extract + resolve + cite) — the served model over all ~6.4k+ human→agent turns. The expensive, semantic, 59%-contextual core.
- **CPU:** Layer 1 (salience pre-tag) and Layer 3 (exact-merge dedup) — bracket the GPU pass; cheap and deterministic.
