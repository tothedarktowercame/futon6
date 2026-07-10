# Post-mortem — backward (goals-and-holes) full run, golden-primed, 2026-06-25

**Outcome:** STOPPED before completion by Joe (tired, late). Run was not viable in a reasonable
window. Forward run was fine; this is backward only. Forensics captured to
`scratchpad/forensics/` (vllm-serve.log, dev-run-state.txt, req-per-min.txt).

## What ran
- `c_mine_joint.py`, golden-primed (9 backward exemplars prepended), tightened INSTR, operator-filtered.
- `LIMIT=0` → **3,682 pairs** (the full operator-filtered backward corpus).
- Tunnel mode (vLLM on Linode 99755702, transcripts on dev). Keepalive tunnel.

## The two problems (compounding)

1. **Corpus size vs. throughput.** 3,682 pairs is 2.6× the forward run's 1,398. Even at a healthy
   rate this is a ~10 h run on this hardware (quantized 70B, ~27 tok/s steady gen). NB Joe had earlier
   chosen a **1,500-cap** for backward (swap-plan); the later golden handoff said "drop LIMIT for the
   full run." Those two instructions conflicted and it launched uncapped — should have been flagged at
   launch. **Lesson: reconcile a standing cap-choice against a later "full run" instruction before launching.**

2. **Intermittent slowness — leading hypothesis: long-lived tunnel degradation (NOT priming, NOT proven).**
   Throughput was not uniform — req/min oscillated 6–11 (normal) with dips to 1/min. Live snapshot during
   a dip: vLLM engine `Running: 0 reqs / Waiting: 0 / KV 0%` (idle) while the dev client sat blocked in
   `poll()` waiting on a response. Server side was clean (1 error in the whole log; gen throughput
   0–51 tok/s, the 0s = idle windows). Effective ~4–5/min avg → ETA ~12–15 h.

   **What the evidence rules out and points to** (corrected from the first overconfident "tunnel
   lost-requests" framing):
   - **Priming-per-request cost is RULED OUT.** The golden+priming A/B at n≈12 was NOT slow. Priming
     overhead (the 9-exemplar prepend) is constant per call, so if it were the cause it would have shown
     proportionally at n≈12. It didn't. → not priming.
   - **The slowness scales with run DURATION / tunnel age, not per-request work.** n≈12 burst = fine;
     forward (1,398, same tunnel, ran *earlier* on a younger connection) = fine at ~12/min; backward
     (3,682, same tunnel, inherited the connection after ~2 h of forward) = crawled. Plus an earlier
     **confirmed** tunnel half-open *death*. The variable tracking the slowness is the age/length of the
     long-lived SSH forward — keepalive (`ServerAliveInterval=15`) mitigated it (no second full death) but
     did not eliminate intermittent drops.
   - **Still a hypothesis, not proven.** Root cause is unconfirmed because the client was never
     instrumented with per-request latencies — the one measurement that would settle it. **Do this next
     time** (see recommendations).

   The dominant, certain factor remains the corpus SIZE (3,682 pairs = overnight run regardless). The
   tunnel behaviour was a secondary contributor.

## Durable loss
None of consequence. The run never reached the 200-entry checkpoint, so no partial C-entries were
written (in-memory only, lost on stop). `data/c-vector/c-entries.openai.json` still holds the prior
12-pair smoke (8 records) — NOT this run. The prior 400-record learning sample is preserved at
`c-entries.openai.pre-instr-fix.json`.

## Recommendations for the re-run (claude-1 / next session)
- **Cap it** — `LIMIT=1500` (Joe's original choice) or smaller; 3,682 uncapped is an overnight haul.
- **Instrument per-request latency** in `c_mine_joint.call_openai` (wall-time per call + a running
  min/median/p95 printed every N pairs). This is the measurement that would have settled the diagnosis
  above — if latencies climb over the run, it's the tunnel; if they're flat, look elsewhere. Cheap; do it.
- **Kill the tunnel dependency for a long haul** — either run on-box (accept transcripts-on-box, or a
  scripts+turns-only rsync), or add a client-side **retry-on-timeout** in `c_mine_joint.call_openai`
  (currently one 180 s attempt then skip — a lost request = a dropped pair AND 180 s wasted). A shorter
  timeout + 2–3 retries would both speed recovery and stop silently dropping pairs.
- **Lower the per-call cost** — the 9-exemplar golden prepend is large on every call; vLLM prefix-caches
  it (~50% hit) but consider trimming to ~5 exemplars (forward used 5 and gated fine).
- Forward run is the validated win (new_patterns 6.6%, contextual tiers 38%); its artifacts are final
  on Dionysus + snapshotted to `data/meme-mine/*.v2primed.*`.

### Infra for the re-run (ranked by leverage)
1. **Batch concurrent requests (free, biggest win).** Both runners send requests ONE AT A TIME, so the
   GPU sat mostly idle — the observed ~27 tok/s was a single request's worth, not the card's capacity.
   vLLM does continuous batching: 8–16 concurrent in-flight requests (asyncio / thread pool in
   `call_openai`) multiplies aggregate throughput on the SAME hardware. A ~15 h sequential run could
   plausibly be 2–3 h. Costs nothing.
2. **Use a closer region — London `gb-lon`** (GPU-capable, same price as `us-ord`). ~5 ms vs ~90–100 ms
   to Chicago. NB this helps long-lived-tunnel RELIABILITY (shorter path), NOT raw speed — a 30–70 s
   generation dwarfs network RTT. Other near-UK GPU regions: Frankfurt, Amsterdam, Paris, Stockholm.
3. **Cap at LIMIT=1500** (Joe's original choice).
4. **Hardware ceiling on Linode:** their GPU lineup is RTX 4000 Ada (`g2`, $2.96/hr for 4×, what we used
   — their *newer* card) and RTX 6000 (`g1`, Turing 2018, $6/hr — older, not faster for this). **No
   A100/H100 on Linode** — genuinely faster silicon means another provider (Lambda/RunPod/CoreWeave) at
   the cost of re-doing the StackScript/setup. Only worth it if batching+London isn't enough.

## Open findings already logged for claude-1 (from the forward gate-check)
- `check_meme_mine_gates.py` crashes on a list-valued meme `ref` (`TypeError: unhashable type: 'list'`;
  4 memes) — needs coerce/skip.
- Forward op-vocabulary drift — 8 discourse ops + ~110 off-spec verbs outside the move-class allowlist.
- Runner should normalize `ref` to a string.
