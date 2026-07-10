# Excursion: E-patch-agent-evidence-leaks — the Evidence Landscape commingles operator, agent, and harness turns

**Date:** 2026-06-25
**Status:** DERIVE-1 DONE (the shared classifier is built and both miners route through it; the
forward leak is measured). Ingestion-time tagging (DERIVE-2) and the mesh-source stamp (DERIVE-3)
remain deferred.

> **F1 (2026-06-25) — shared classifier shipped + forward leak measured.**
> - **`scripts/transcript_provenance.py`** — the ONE test `classify(record) → operator|agent|harness|unknown`
>   + `is_operator(record)`, seeded by the validated `c_mine_joint` promptSource logic and this memory's rules.
> - **Both miners routed through it:** `c_mine_joint.read_pairs` (replaced its inlined copy) and the
>   previously-unaudited `meme_mine_runner.read_asks` (was AUTO_CALLERS + a partial bell-regex; **missed
>   `promptSource` sdk/system bells entirely**). Both compile + stub-run clean.
> - **Forward leak quantified** (`scripts/audit_transcript_provenance.py`): of 1480 OLD read_asks ask-candidates,
>   **106 (7.2%) were non-operator** (24 agent + 82 harness) — now dropped; 1374 operator asks kept. Corpus-wide
>   the user turns split operator 4012 / agent 1703 / harness 885 / unknown 0.
> - **Data cleaned too** (`scripts/postfilter_meme_mine_leak.py`): re-derived each artifact record's provenance
>   authoritatively (corpus index by ask-id → full-record `classify`; 1329/1330 via metadata, 1 via body fallback),
>   **dropped 78/1330 non-operator** (19 agent + 59 harness), backed up originals to `*.pre-f1`, regenerated the
>   consume tail (moves + cert). Verified the one ambiguous drop (`Caller: joe-repl`) is correct — those 3 are
>   `sdk` path-probes ("reply only: ack-repl"), not operator preferences; real Joe is `Caller: joe` (2630, kept).
> - **F2 is NOT this leak.** On the cleaned set `new_patterns` still fire on **95%** of asks (recall 27%→29% only) —
>   so the R17 over-proposing is INSTR over-firing on genuine operator asks, a SEPARATE fix (meme_mine_joint INSTR),
>   not provenance contamination. The original IDENTIFY-era note (4137→3565) below predates the shared extraction.
**Repo:** futon6 mining code (`scripts/c_mine_joint.py`, `scripts/meme_mine_runner.py`) reading
the shared corpus `~/.claude/projects/*/*.jsonl`. The source-level fix reaches into ingestion /
the Agency mesh (futon3c).
**Spawned from:** the goals-and-holes backward run. Joe: inter-agent content should not be landing
in the Evidence Landscape — and "I assume we have not yet patched the real source of the leak."
Correct. We patched one reader.

## HEAD (one line)

**We filtered the symptom in one miner; the source is unpatched.** The transcript Evidence
Landscape commingles three kinds of `user` turn — **operator** (Joe), **inter-agent** (bells/
whistles delivered by Agency), and **harness** (task-notifications, system-reminders, autorunner
control prompts) — with no canonical provenance partition. So every consumer must independently
re-derive "is this Joe?", and they already disagree. The fix isn't a better per-miner filter; it's
**one provenance classifier the whole landscape shares**.

## What just happened (the evidence)

`c_mine_joint.py` mines the belly (operator preferences) from `(assistant-turn, human-reply)` pairs.
Its `read_pairs` operator-only test keyed only on message *content* (`Caller:` wrapper), so a turn
with no wrapper got `caller=None` and was treated as Joe. But `caller=None` covers **both** direct-CLI
Joe **and** raw agent bells like `claude-1 → claude-2: SCOPE CHANGE …`. Result: **572 of 4137 pairs
(~14%) were inter-agent or harness turns** — agent whistles, `<task-notification>` injects,
`Reply with exactly: OK2` autorunner prompts — landing in the belly as if they were Joe's preferences.

Patched 2026-06-25 by filtering on the harness **`promptSource`** metadata (`typed`=Joe; `sdk`/`system`=
programmatic bell / harness, unless `Caller: joe`; legacy turns → body-authorship fallback). Verified:
4137 → 3565 clean pairs, 0 agent/harness replies remaining, Joe's turns (incl. third-person
orchestration like "codex-3 is idle, whistle it") spared. See `reference_transcript_operator_provenance`.

**That fix is a band-aid on one consumer.** The leak is upstream of it.

## The real source

The corpus has no provenance partition. Into each agent's session transcript, Agency delivers other
agents' bells **as ordinary `user` turns**, and the harness injects its own control/notification turns
the same way. The only disambiguators are the harness `promptSource` field and the surface `Caller:`
wrapper — and those are:
- **non-uniform** — legacy turns predate `promptSource` (value `<none>`), so any classifier needs a
  body-heuristic fallback that is inherently fuzzier;
- **re-derived per consumer, divergently** — `c_mine_joint.read_pairs` (now `promptSource`-aware) and
  `meme_mine_runner.read_asks` (keys on `AUTO_CALLERS` + an ASK-cue regex, **unaudited for this leak**)
  implement *different* operator tests. A bell phrased as a request ("claude-1 → claude-2: please wire X")
  can pass the forward filter. Every future reader (substrate-2 ingest, concept-tag, the next miner)
  starts from raw commingled turns and re-rolls the test — or forgets to.

So "patch the leak" per miner is N filters that drift. The landscape itself is undifferentiated.

## DERIVE — three places to actually patch (cheapest → most source-level)

1. **Shared classifier (one function).** Extract the operator test into a single
   `transcript_provenance.classify(record) → :operator | :agent | :harness | :unknown`, and route
   every miner through it. The `c_mine_joint` `promptSource` logic is the validated seed. Cheapest;
   doesn't touch source data; immediately de-duplicates the divergence and lets us audit/raise one test.
2. **Ingestion-time tagging.** A pass that walks the transcripts once and writes the provenance tag per
   turn (sidecar `.edn`, or a field in the durable store). Consumers *filter on the tag*, never re-derive.
   Partitions the landscape once, correctly; survives consumers that forget to filter (they read tagged
   data). This is the natural "Evidence Landscape is clean by construction" move.
3. **At the mesh source (Agency, futon3c).** When a bell/whistle is delivered into a session, stamp the
   injected turn with explicit, unambiguous provenance at *write* time (richer than today's `promptSource`).
   Closest to the true source — no turn is ever born ambiguous — but cross-repo and touches the live mesh.

These compose: (1) is the unit of truth (2) and (3) both reuse; (2) makes it durable; (3) removes the
legacy/heuristic fallback for all *future* turns.

## ARGUE

**IF** the futon stack mines the operator's belly/methods from the transcript Evidence Landscape,
**HOWEVER** that landscape commingles operator, inter-agent, and harness turns with only fuzzy,
per-consumer-re-derived provenance,
**THEN** lift the operator/agent/harness test into one shared classifier (DERIVE-1) and tag the
landscape at ingestion (DERIVE-2), rather than adding another per-miner filter,
**BECAUSE** a leak that lives in the *shared source* reappears in every consumer that forgets the
filter or implements it differently — and two of ours already disagree (`read_pairs` vs `read_asks`).

## VERIFY (what the c_mine_joint fix already establishes)

`promptSource`-based classification is empirically sound on this corpus: it removed exactly the
agent/harness turns (spot-checked) and spared Joe's, including the hard third-person-orchestration case
that a naive agent-id filter would wrongly drop (**mention ≠ authorship**). So DERIVE-1's seed is proven;
the open work is (a) audit `read_asks` against the same classifier and quantify the forward-pass leak,
(b) extract the shared function, (c) decide whether to also tag at ingestion.

## Scope

- **In:** one shared transcript-provenance classifier; audit + fix of `read_asks`; (optionally) an
  ingestion-time tagging pass over `~/.claude/projects`.
- **Out:** changing how Agency delivers bells (DERIVE-3) — that's a futon3c mesh change, note it and defer.
- **Owner:** single agent, end-to-end.
- **Exit:** every futon6 miner reads operator turns through the one classifier; the forward-pass leak is
  measured and closed; `reference_transcript_operator_provenance` points at the shared function, not at
  one miner's inlined copy.

## Related

- `reference_transcript_operator_provenance` — the `promptSource`/`Caller`/authorship rules (the seed).
- `feedback_meme_mine_joint_null_crash` — the other durability fix from the same run.
- [[E-crossed-bells]] (futon3c) — same mesh-commingling root, seen from the agents' side: a delivered
  bell carries no agent-visible thread. Here it carries no *consumer-visible* provenance. One source.
