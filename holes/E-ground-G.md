# Excursion: E-ground-G

*Ground the rollout's value `G` in a signal external to the prior's own metric — then re-run T1.
Until we have reason to think a grounded `G` makes multi-step search pay rent, we do not proceed
with the rest of the apparatus.*

**Type:** E-prefix excursion (bounded scope-out, single-agent-owned end-to-end).
**Owner:** claude-3.
**Scopes out from:** `M-differentiable-substrate` (the producer/value-prior); couples to the car
`M-peradam-grounding` (CH2) and the rollout `futon2.aif.rollout`.
**Spawned:** 2026-06-10, by Joe, from the **A2/T1 result** in `C-falsifiable-missions` §5.
**Status:** CHARTERED (IDENTIFY).

## Why (the forcing result)

A2/T1 measured the multi-step rollout paying **0 rent** (0/24 roots; futon2 `scripts/t1_rent.clj`,
`d9f9020`) — and *not* because the paths are forced (9/24 roots branch; the wide beam explored up to
20 policies). The wide beam **never diverged from the greedy-`:prior` first move** because the value
`G(π)=Σγ^t g(s_t)` is built from `g = :delta-g/:score` — **the same metric the `:prior` is derived
from.** Prior and value are aligned *by construction*: **you cannot search past your own metric.**
This is the closed-loop diagnosis made empirical. T4 (does the prior carry real info) would only
re-confirm the same root cause, so the checklist march is **halted** here (Joe, 2026-06-10).

## The hypothesis to test (falsifiable)

If `g(s_t)` is re-grounded in a signal **external to the prior's metric**, the prior's first-move
choice may no longer be value-optimal, and **multi-step search may begin to pay rent.** Re-running
T1 with a grounded `G` is the falsifiable test:
- **T1-rent > 0 with grounded `G`** ⟹ grounding is the unlock; the rollout/search apparatus is
  vindicated-conditional-on-grounding → **proceed** (and the car's value-channel becomes load-bearing).
- **T1-rent ≈ 0 even with grounded `G`** ⟹ a deeper problem (the search itself, or the chosen
  grounding signal, is inadequate) — a real, recorded negative → **do not proceed**; re-scope.

## Candidate grounding signals (external to the prior's metric)

Ranked by availability-now vs gold-standard:
1. **Realized substrate-2 discharge/closure** — did the move's hole *actually* close in the real
   meme-arrow store (futon3a / 7071), vs the prior's *estimate* that it would. **Available now** on
   historical data; the prior's estimate and the realized outcome can diverge → the divergence is
   exactly what lets search pay rent. *Leading candidate for the v0 test.*
2. **PURs** (pattern-use-records) — did the pattern application succeed (the M-pattern-posteriors
   signal; the watcher already parses these). Available now; pattern-grain not move-grain (a
   credit-assignment step needed).
3. **Real peradams via CH2** — the 3-witness grounded reward bound to `:move/id` (the car). The
   gold standard (un-game-able), but **sparse** and **post-arm** (slow, Track B). The v1 grounding
   once the car emits real peradams.

## Scope

- **IN:** re-define `g(s_t)` (futon2 `rollout/move-cost`) to draw on a grounded signal (candidate 1
  first); a grounding-data adaptor (read realized closure from substrate-2 for the moves' `:want`s);
  re-run `t1_rent.clj` with grounded `G`; report rent honestly. A complementary T4 read if useful.
- **OUT:** building the full grounded training loop; R2; wiring the car's live peradams (that's the
  car). This excursion is a **measurement** — does a grounded value make search pay rent *at all*.
- **The halt it enforces:** `C-falsifiable-missions` §5 A2/T4+ and further rollout investment stay
  **paused** until E-ground-G reports rent>0 (proceed) or rent≈0 (re-scope).

## Success / done

A grounded-`G` T1 number, reported either way, with a recorded proceed/re-scope decision — and, if
proceed, the grounding signal named as the value the rollout should consume (the bridge to the car's
CH2). This is itself an observable-style discharge (measurement happened + decision recorded).

## Open question for Joe (the load-bearing design choice)

**Which grounding signal for the v0 test** — realized substrate-2 closure (candidate 1, available
now, my recommendation) — or do you want to wait for real peradams (candidate 3, the gold standard
but gated on the slow car)? My lean: **test with candidate 1 now** (it's external-to-the-prior and
available), because it answers "does grounding help *at all*" cheaply, before we invest in the slow
peradam path. If candidate 1 shows rent, the car's peradams are the *better* grounding of the same
shape; if candidate 1 shows none, we've learned something important before the car.
