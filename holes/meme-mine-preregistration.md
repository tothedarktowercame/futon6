# MEME-MINE box run — preregistration

**Date:** 2026-06-25 · **Owner:** Joe + claude-1 · **Mission:** [[M-operational-vocabulary]]
**The run:** `scripts/linode-meme-mine.sh` on a 4-GPU Linode (`g2-gpu-rtx4000a4-s`, vLLM `mark4-70b`),
mining this box's human→agent turns into `(have, want)` memes. Spec: `holes/meme-mine-runner-spec.md`.
Registered **before** commissioning so the predictions can't be rationalised after the fact.

## 1. What we plan to learn (falsifiable predictions)

Anchored on the CPU validation (11-ask hand-sample + the 198-mission scan).

- **P1 — extraction quality holds at scale.** The openai endpoint-resolution tiers land near the hand-sample
  (contextual 40–70%, named 15–35%, unsupported <30%). **Falsified** if unsupported >40% (the 70B can't pin
  referents → prompt/model too weak) **or** named >50% (over-confident — likely 間 false-grounding slipping
  past the verbatim evidence-check).
- **P2 — the op-vocabulary is a bounded operational *language*.** Distinct op-classes over the full corpus are
  more than the 3 hand-coded but **bounded** — predict ~15–40 classes covering ≥90% of memes after merging
  singletons. **Falsified** if effectively unbounded (hundreds of one-off ops = noise, not a vocabulary).
- **P3 — coverage is real, not a small-N artefact.** The bridge produces meme-grounded moves for a substantial
  share of the 82 ask-covered missions — predict **≥40 of 198** rollout missions upgrade
  structure-borrowed → meme-grounded in `action-cert.json`. **Falsified** if <10 (asks don't resolve to
  missions at scale → the random-sample 0 was the rule, not just small-N).
- **P4 — provenance beats the guess (qualitative).** For missions with both a structure-borrowed and a
  meme-grounded move, the meme-grounded op is a recognisable *real* operation (spot-check ≥10) — better
  provenance than the borrowed phase-advance.
- **P5 — joint retrieval lifts grounding (the trick).** Shown retrieved candidate missions+patterns, the
  model grounds a turn to ≥1 real mission far more often than the turns-only design (which got ~0). Predict
  **≥40% of turns** ground to ≥1 candidate mission. **Falsified** if <15% (retrieval recall or model grounding
  is too weak → the candidates aren't reaching/helping the model).
- **P6 — the model characterizes with existing patterns AND grows the vocabulary (R17).** A majority of
  grounded turns carry ≥1 `pattern_app` (existing patterns cover the operational reality); and the model
  proposes **new patterns** only for the genuinely-novel minority, each with IF/HOWEVER/THEN/BECAUSE +
  verbatim evidence. **Test (qualitative, spot-check ≥10):** new-pattern proposals are plausible, non-redundant
  heuristics, not hallucinations — the niche-construction signal. (No hard threshold; this is exploratory.)

## 2. Data touched (and the privacy posture)

- **Input:** `~/.claude/projects/*/*.jsonl` on **this box only** (~6.4k human→agent asks + a 4-turn thread
  window each). These are **private conversation transcripts.**
- **JOINT mining (Joe's correction, 2026-06-25 — supersedes the turns-only design).** We HAVE all three
  objects (turns = inference steps · missions = preprints · patterns = heuristics), so the CPU is the
  *retriever* and the GPU does the joint reasoning:
  - *CPU retriever* (concept-tag / co-embedding) selects, per turn, the top-K **candidate missions + patterns**.
  - *GPU joint reasoner* (`meme_mine_joint.py`) is shown the **turn + thread window + retrieved candidate
    mission summaries + pattern titles**, and grounds endpoints to real ids, characterizes which patterns the
    turn instantiates, composes cascades, and **proposes new patterns** (R17). So the model **does** see
    missions + patterns — deliberately. (Mission summaries + pattern titles are work artifacts, far less
    sensitive than the private transcripts.)
- **What actually leaves dev, by mode:**
  - *Tunnel mode (recommended):* runner + tail run on **dev**; the **per-turn prompts** (ask + thread +
    retrieved candidate mission-summaries + pattern-titles) cross to our box's model. The full transcripts and
    the full mission/pattern corpus **stay on dev**; only the top-K retrieved descriptions per turn cross.
  - *On-box mode:* rsync transcripts + futon6 (incl. the mission scopes, pattern embeddings, caps) to our own
    ephemeral box.
- **Output:** `data/meme-mine/{resolved-memes.openai.json, diffsub-moves-meme.edn, action-cert.json,
  concept-index.json}`. These **embed verbatim ask spans** (the evidence citations) → **as sensitive as the
  transcripts.** Keep `.gitignore`d; do not leave them on the box; **decommission the box immediately after.**
- **Not touched:** mesh peers' transcripts (London/laptop); the live substrate-2 / `:7071` (sim-only, zero
  writes); the live War Machine (no arming, no acting).

## 3. Method (one line each)

**CPU retrieve** candidate missions+patterns per turn (concept-tag / co-embedding) → **GPU joint reason**
(`meme_mine_joint.py`, vLLM 70B): ground endpoints to real ids · characterize pattern-applications · compose
cascades · propose new patterns — each grounded to a verbatim span or `:unsupported` → **CPU consume tail**
(bridge meme-grounded moves → floor/cert → concept-tag index). All sim-only; the model is shown mission
summaries + pattern titles (work artifacts), never the transcripts beyond the turn+thread it is mining.

## 4. Pre-specified analysis (computed from the outputs, no new choices)

Endpoint-tier distribution · distinct op-class count + coverage curve · # missions upgraded to meme-grounded
in `action-cert.json` · median ΔG of meme-grounded vs structure-borrowed moves · 10 spot-checks of
ask→meme→move faithfulness. (The runner already self-reports tiers/op-vocab/dedup; the tail reports the floor.)

## 5. Cost

Provision + StackScript reboot + model pre-pull + serve ≈ 15–20 min; mine ~6.4k asks at ~1–2 s/ask
(batched: less) ≈ **1–3 GPU-hours**. **Verify current `g2-gpu-rtx4000a4-s` pricing before commissioning**
(not asserting a $ figure from memory). Decommission immediately after (privacy + cost).

## 6. Decision rule (set in advance)

- **P1∧P2∧P3 hold** → wire the meme-grounded move-set into the live WM act-gate (R16 criterion (2) closed with
  real provenance) + targeted re-mines for the highest-value missions.
- **P1 fails** (noisy extraction) → tune the Layer-2 prompt (`meme_mine_runner.py INSTR`), re-run a small
  `LIMIT=` box batch before the full mine.
- **P3 fails** (coverage) → the `turn→mission` link is the bottleneck; lean on the `concept-index` routing /
  harden `M-autoclock-in` rather than re-running the box.

## 7. What we will NOT conclude

The mine produces **sim-only provenance, not authorization** — it does **not** say the WM should act
(R16 loop-closure / operator arming stays separate and gated). Coverage on **this box's** transcripts is not
mesh-wide. A meme-grounded move is a better *prior*, not a verified outcome (that's R14 γ, post-loop).
