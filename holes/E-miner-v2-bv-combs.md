# E-miner-v2-bv-combs — type the wiring diagrams with BV connectives (combs over :composes)

**Excursion (bounded, single-owner). Spun out 2026-06-14. Owner: a Claude
agent (conceptually rich — CT/BV typing). Bell claude-1 back with results + shas.**

## Goal
Type the mined wiring diagrams with **BV connectives**, building **combs over
the `:composes` skeleton**. This is an *application* of the CT mining that
deliberately puts **external exotype eustress** on the mining (Joe): a harder
downstream consumer (BV-typed combs) pressures the mining to emit well-typed,
genuinely-composable structure — and wherever the `:composes` skeleton is too
thin to type, that gap is the eustress signal that feeds back to improve the
mining.

## Scope / steps
1. Locate the mined `:composes` skeleton (grep `:composes` — `futon6/holes/golden-graphs/`,
   `futon5` CT DSL / wiring diagrams). Understand its current shape.
2. Define the BV connective typing (BV = the deep-inference connectives;
   combs = parametrised/comb-shaped morphisms over the composition skeleton).
3. Attempt to type the skeleton; **record every place the skeleton is
   insufficient to type** — that residue is the load-bearing output (it's the
   eustress: a typed-consumer's demands that the mining must rise to meet).
4. Feed the gap-list back as concrete mining-improvement items (cross-ref the
   CT-mining work: dp_paper_view scopes / the :composes detector).

## Acceptance
A BV-typing pass over the `:composes` skeleton; combs built where typeable; a
**gap-list** of where the mining's output is insufficient (the eustress
deliverable) with concrete feedback to the mining. Commit artifacts + this note.

## Constraints
Never restart the futon3c JVM. Co-Authored-By: Claude Fable 5
<noreply@anthropic.com>. Bell claude-1 back with {what typed, the gap-list/
eustress signal} + shas.

---

## RESULTS (claude-2, 2026-06-14)

**Owner:** claude-2 (end-to-end). **Typer:** `scripts/bv_comb_typer.py`
(stdlib-only, deterministic). **Artifact:** `data/bv-comb-typing.edn`
(EDN, bracket-balanced).

### Where the skeleton lives
`:composes` is NOT in the proof-anatomy golden graphs (those use
`:decomposes-into`/`:built-from`/`:parses`…) and NOT in futon5. It is the
WIRING layer emitted by `mission_triple_miner.py` into
`data/mission-triples/*.edn`: `:application` nodes (checkpoints = witnessed
applications) joined by `:composes` `:from`/`:to` edges "in authored checkpoint
order." Only **4 of 81** mission-triples carry any wiring: M-first-flights (23
ckpts), M-futon1a-rebuild (5), M-operational-readiness (5), M-pilot-appearance
(4) — 37 checkpoints, 33 `:composes` edges.

### BV typing pass — what typed
- **seq** types every chain: each mission's `:composes` chain ⟶
  `{:bv/seq [ckpt-0 … ckpt-n]}` (the non-commutative "before/after" connective —
  the natural reading of `:composes`). All 4 typed at the skeleton level.
- **copar/par exercised: ZERO.** All 4 skeletons are **pure linear chains**
  (0 forks, 0 joins). BV's whole point — seq *interleaved* with par/copar via
  the medial rule — is structurally unreachable on this skeleton.
- **Interface (composability) check** via the only available boundary proxy —
  the checkpoint's "Test state: N tests" — checking cod(a) ≤ dom(b): only
  **4/33** edges are `:typed-monotone` (the M-operational-readiness
  686≤691≤695≤700 run); **29/33** are `:gap-no-boundary-type`; 0 mismatches.
- **Combs built:** the endpoints comb `⟨first ; - ; last⟩` per mission (the
  context with the interior abstracted to a hole). All **4 are
  `:skeleton-only`** — NONE is `:interface-verified`, because every chain's
  endpoint lacks a boundary type (even the best-typed mission drops its test
  state at the final checkpoint). So combs exist as skeletons but cannot be
  *verified* composable.

### Gap-list — the load-bearing eustress signal (5 gaps; full text in the EDN)
1. **`:gap-no-typed-interface`** — 29/33 edges un-interface-typeable; nodes have
   prose `:witness` but no dom/cod object. → miner: emit
   `:interface {:in … :out …}` per `:application`.
2. **`:gap-boundary-type-prose-buried`** — only 8/37 checkpoints expose any
   boundary proxy ("Test state: N tests"); partial & inconsistent. → miner:
   structured `:test-count`/`:gates-met`/`:capability-set` so the wire type is
   total.
3. **`:gap-no-par-copar`** — all chains linear; BV collapses to seq;
   `:jointly-with` is spec-only, never emitted. → miner: detect
   dataflow-independent checkpoints, emit par/copar (or `:jointly-with`), not a
   forced chain.
4. **`:gap-authored-order-not-dataflow`** — `:composes` is "authored order," a
   narrative wire, not a typed morphism boundary. → miner: separate
   temporal-order from dataflow-dependency; only the latter is a true
   `:composes`.
5. **`:gap-no-unit`** — no identity/unit checkpoint, so combs can't be
   normalised. → miner: emit an explicit unit/no-op at phase starts.

**Eustress verdict:** the typed consumer (BV combs) shows the mining's wiring is
structurally impoverished in two load-bearing ways — (a) **no typed interfaces**
(only seq skeleton + a prose-buried partial proxy types at all), and (b) **no
concurrency** (pure authored-order chains, so par/copar/`:jointly-with` and the
deep-inference content are unreachable). Fixing (1)+(2) unlocks verified combs;
fixing (3)+(4) unlocks the BV connectives beyond seq. Reproduce:
`python3 scripts/bv_comb_typer.py`.
