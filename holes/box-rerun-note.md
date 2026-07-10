# Box re-run note — meme-mine + goals-and-holes (smoke-gated)

**For:** the agent managing the GPU box (vLLM `mark4-70b` is up on `:8000`).
**Why:** a review found both mining prompts over-firing — forward `op` was ~50% discourse-not-operation
+ `new_patterns` on 95% of asks (F2/F3/F4); backward `correction` was ~56% (over-classified). **Both INSTRs
are now fixed** (`meme_mine_joint.py`, `c_mine_joint.py`), and the operator-provenance leak is closed
(both readers gate on `transcript_provenance.is_operator`). Re-run both — but **smoke first** (a full pass
shipped half-bad last time; catch it at n=12 for cents).

## Recommended path — TUNNEL, sync NOTHING
Run the scripts **on dev**; the box only serves the model. Don't rsync futon6 (~31 GB this run never touches).
```bash
ssh -L 8000:localhost:8000 <box> &      # tunnel the vLLM port; transcripts + repo stay on dev
```

## Step 1 — SMOKE each pass (LIMIT=12), check the go/no-go bands BEFORE the full run
```bash
cd ~/code/futon6
OPENAI_BASE_URL=http://localhost:8000/v1 LIMIT=12 scripts/linode-meme-mine.sh            # forward (memes)
OPENAI_BASE_URL=http://localhost:8000/v1 LIMIT=12 scripts/linode-goals-and-holes-mine.sh  # backward (C-entries)
```
**GO/NO-GO bands** (each runner also prints its own review notes):
- **Forward** `data/meme-mine/joint-memes.openai.json` — op operational-share ≳70% (discourse verbs
  elaborate/contrast/compare ⇒ FAIL); endpoint tiers contextual-majority (named ≫ contextual ⇒ FAIL);
  `new_patterns` a small minority of asks (~1-per-ask ⇒ FAIL).
- **Backward** `data/c-vector/c-entries.openai.json` — **reach ≫ correction** (a flood of `correction`
  ⇒ FAIL — v1 was 56%); spot-check ≥5 `reply_span`s are REAL overrides, not continuation/agreement/recap;
  grounding healthy; provenance leak = 0.

If a band fails, **stop** — the INSTR needs another pass; ping claude-1. Do **not** scale a failing smoke.

## Step 2 — full run (only once both smokes pass)
Drop `LIMIT` (default 0 = all). Same two commands. Outputs are gitignored under `data/*` (verbatim spans
stay on dev). The forward consume tail (moves/floor/cert) runs automatically; fold the backward C-entries
into the belly locally with `bb scripts/c_vector.bb`.

**Note:** the prior partial backward run is preserved at `data/c-vector/c-entries.openai.pre-instr-fix.json`
(the 400-record sample we learned from) — don't overwrite it; the re-run writes `c-entries.openai.json`.
Full background: `holes/meme-mine-runner-spec.md` (smoke-gate section) + `holes/E-patch-agent-evidence-leaks.md`.
