# Box handoff #2 — re-smoke with tightened INSTR + a GOLDEN few-shot set

**For:** the agent managing the box (vLLM `mark4-70b` up on `:8000`). Supersedes the step-1 smoke in
`box-rerun-note.md`. Two levers now compound: (a) **INSTR tightened again** by claude-1 — forward
`new_patterns` (rate-anchored "<1 in 5, default []") and backward `correction` (default reach/[]; agreement
≠ correction); (b) **your golden few-shot set** prepended to the 70B prompt.

## ⚠️ The one thing to get right: golden = HAND-VERIFIED, not raw smoke
The smoke outputs were ~half wrong (forward `new_patterns` 75%, backward `correction` 7:1 with agreement
misreads). **Do NOT paste raw smoke as golden — it would teach the model the error.** Build golden by hand:
keep the verified-CORRECT examples, and include a few **labeled NEGATIVES** (the wrong ones, marked with the
right answer). A handful each (≈3–5 positive + 2–3 negative per pass) is plenty for few-shot.

**Forward (meme) golden — from the good smoke memes:**
- ✅ `"M-typed-holes mission" → "manifest of concepts to formalise in Lean"`, op=`create`, new_patterns=`[]`
- ✅ `"T2" → "T3"`, op=`update`, new_patterns=`[]`
- ✅ `"M-first-flights" → "semi-formalisation"`, op=`investigate`, new_patterns=`[]`
- ❌ NEGATIVE: a routine build/fix/create ask that the smoke gave a `new_patterns` entry → show the SAME ask
  with `new_patterns: []` (the lesson: routine asks are not new patterns).

**Backward (C-entry) golden — A READY-BUILT balanced set exists: `data/c-vector/golden-backward.json`**
(9 exemplars: 3 reach+ / 3 correction+ with explicit pivots / 2 agreement→reach / 1 recap→[]; each an
`input → ideal` pair + a `lesson`; its ideal-outputs PASS `check_goals_holes_gates.py`). Use it directly or
extend. The shape to aim for:
- ✅ correction: agent proposed X; reply *"before we do that, can we scope a Codex handoff…"* →
  `flavour=correction, preferred.value="scope the Codex handoff first"` (a real pivot, both sides nameable).
- ✅ reach: agent orienting toward M-typed-holes → `flavour=reach`, reply_span=null.
- ❌ NEGATIVE: *"Yes let's commit the TypedHole — as for T3 you can bell it"* → `flavour=reach` (or []),
  NOT correction (agreement-plus-detail is agreement).
- ❌ NEGATIVE: *"So we had an M-typed-bells mission, which was…"* → `[]` (recap is not a correction).

Inject the golden pairs as few-shot exemplars ahead of the real input (leading user/assistant turns, or
appended to the system prompt) — input → ideal-JSON for the positives, input → corrected-JSON for the negatives.

### Validate the golden set against the gates BEFORE feeding it to the 70B (author ≠ reviewer)
The gates are now runnable checkers — ONE per direction. Dump your golden *ideal-outputs* in the artifact
shape (a list of joint-meme records, or a list of C-entry records) and run the matching validator:
```bash
futon6/.venv/bin/python scripts/check_meme_mine_gates.py   <your-golden-memes.json>     # FORWARD (memes)
futon6/.venv/bin/python scripts/check_goals_holes_gates.py <your-golden-centries.json>  # BACKWARD (C-entries)
```
The backward validator adds two C-entry-specific hard gates beyond the basic bands: **I1-evidence** (every
C-entry must cite its verbatim span — reach→assistant_span, correction→reply_span; an empty span is a
fabricated preference) and **correction-target** (each correction names a redirected target). It separates
the two failure modes — classification (reach vs correction) vs value-extraction — so you see which is off.
It prints PASS/FAIL per band and exits nonzero on any hard-gate fail. **A golden that FAILS is mis-curated**
(it would teach the model the error) — fix it before priming. (Validated: it correctly FAILs the raw smoke —
forward new_patterns 75%, backward correction 88%.) **Caveat:** the `correction-precision` check is a
conservative proxy — it flags any correction whose `reply_span` lacks an explicit pivot marker
("not that / instead / rather / no,"). A genuine but implicitly-phrased pivot ("before we do that, can we…")
will flag — so prefer golden corrections with an EXPLICIT pivot, or accept the flag knowingly. Same checker
runs on the smoke and full outputs.

## Re-smoke, then scale (tunnel; sync nothing)
```bash
cd ~/code/futon6
OPENAI_BASE_URL=http://localhost:8000/v1 LIMIT=12 scripts/linode-meme-mine.sh
OPENAI_BASE_URL=http://localhost:8000/v1 LIMIT=12 scripts/linode-goals-and-holes-mine.sh
```
**GO/NO-GO bands** (same as before): forward — op operational (✅ already passing), `new_patterns` a clear
minority (≪75%), tiers reasonable; backward — **reach ≥ correction** with corrections being genuine pivots
(no agreement/recap), grounding healthy, leak 0. **Pass both ⇒ drop `LIMIT` for the full run.** Fail ⇒ ping
claude-1; do not scale.

## Fallback (if backward `correction` STILL over-fires after golden + INSTR)
The fix is then structural, not prompt: hard-gate correction on the CPU `CORRECTION_CUE` in
`c_mine_joint.read_pairs` (only allow a `correction` when a contrastive cue actually fired in the reply).
claude-1 wires it — don't tune the prompt a third time.

## Don't clobber
`data/c-vector/c-entries.openai.pre-instr-fix.json` (the 400-record learning sample) and the `*.pre-instr-fix.*`
backups must survive. Outputs are gitignored (`data/*`); verbatim spans stay on dev.
