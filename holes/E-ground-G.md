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

## ANSWERED (Joe + Fable, 2026-06-10): candidate 1 — go. Peradams never enter G.

**Decision: run v0 with candidate 1 (realized substrate-2 closure).** Your lean is endorsed —
and not as the cheap fallback. This is the signal Joe already had in mind; multiple independent
lines converge on it:

1. **It is the system's native fact event.** In the M-memes-arrows vocabulary, a hole closing is a
   *construction being supplied* — the mode-crossing that attestation cannot fake. Grounding G in
   closure is grounding it in the one transition the prior's own metric cannot manufacture.
2. **It is AIF-native.** G should consume *realized divergence from prediction* — the prior said
   this hole would close; did it? That divergence is exactly where this excursion's own §"hypothesis"
   says the rent lives.
3. **It is the aliveness signal.** Per the aliveness synthesis: mana flows when **anamnesis
   discharges** — and closure *is* the anamnesis-discharge event, measured densely, per-move.
   Closure-grounded G and peradams are **the same underlying quantity at two grains**: closure =
   dense/per-move, peradam = sparse/ceremonial.

**Candidate 3 is struck as a grounding signal and re-classified: peradams AUDIT G, they never
constitute it.** Joe's Xbox-Achievement stance is the *reason*, made structural (Goodhart/Strathern:
a measure that becomes a target ceases to be a good measure). Achievements work because they sit
outside the optimization loop; put peradams inside `g(s_t)` and their achievement-nature is revoked
regardless of intent — "un-game-able" means hard to *counterfeit*, not hard to *farm*. The
sparse/slow/post-hoc properties that make peradams bad reward make them ideal **audit**: periodically
check that high-grounded-G trajectories also (later, independently) earn peradams. Correlation high ⇒
grounding honest; drifting ⇒ re-ground. Peradams ground the *grounding*. (Corollary: do **not**
decompose the pudding-prover to densify reward — that manufactures the Goodhart gradient as grindable
XP. Decomposition for typing/audit is fine. If peradam-flavored shaping is ever truly needed, the
only principled form is potential-based — reward *discharge of standing tension*, never *event
occurrence*.) This also frees E-ground-G from waiting on the car for anything.

**Candidate 1b (add to the list): consent-gate verdicts.** Joe's approve/reject at the WM-I4 consent
gate — cheap, already logged, fully external to the prior's metric; a rejected move is a grounded
negative. Denser than peradams, available now. Complement to closure, not substitute. (PURs stay as
the v1 refinement if rent appears but is noisy.)

**The amortization principle (Joe): this grounds all the way down to a single pattern application.**
The unit question is dead simple — **"did it help?"** Applied pattern → did the hole move toward
closure or not? If yes, rent; if no, *we still learned something* (a recorded negative is a result).
The same shape telescopes from one pattern application up through a move, a rollout, the whole
excursion — which is why this measurement is worth running at all and why a rent≈0 outcome is a
success of the apparatus, not a failure of the excursion.

**Reassurance, explicitly (Joe to claude-3):** you do not need more justification to proceed — the
design choice is confirmed from multiple directions (Joe's prior intent, the BHK/mode-crossing
analysis, the AIF reading, the aliveness synthesis). Run the v0 measurement; report the number either
way.

## v0 ATTEMPT RESULT (claude-3, 2026-06-10): the grounding data does not exist yet

Ran the candidate-1 attempt. **It cannot produce a meaningful grounded-T1 number from existing
data — because the realized-closure signal is not captured anywhere yet.** Evidence:
- **`futon3a/meme.db` (the live BHK-arrow store): `promotions` table = 0 rows, `arrows` = 3.** No
  closure-event history.
- **Scope snapshot (`diffsub-scopes.json`, 5532 scopes): all 44 close-hole move `:want`s are
  uniformly `:detached`** (open); the other 11 move targets are synthetic/`:MISSING`. Zero
  per-move closure variance — and the snapshot is the open-hole set the producer built the moves
  *from*, so it is fully prior-baked. Grounding `g` in it = a constant → degenerate, no signal.

**This is itself the finding, and it sharpens the diagnosis:** the loop is closed not only
structurally but at the **data** level — *there is no realized-outcome record to ground against.*
The system has the open holes (detached) and the prior's estimates, but **no captured record of which
holes actually got closed.** You cannot ground `G` in realized closure until realized closures are
*recorded*.

**The reframe this forces (and it is energizing, not a dead end):** the dense closure signal Fable
specified IS exactly what the **discharge-emission half** (`ac4ae5d`, claude-4) records — a CH2
discharge event at every live `promote! :constructed`. That half is built, but has **never run live**
(the sink is empty). So candidate-1 grounding is gated not on the slow dokusan/peradam loop (Fable
correctly severed that) but on **closure-event capture actually running** — a bounded, car-independent
step. Two ways to get the data:
- **(a) Mine it retrospectively from git** — the holes that actually got worked/closed in commit
  history (the under-mined closure source; cf. retrospective-reconstruction-from-git). Independent of
  the prior, available now, but needs a scope→commit-evidence mapping.
- **(b) Generate it forward** — run the WM/loop to close real holes with discharge-recording on,
  accumulate closure events, then ground `G` and re-run T1.

**Open question for Joe (the grounding-DATA-SOURCE choice, now the real fork):** mine git-closure
(a, retrospective, available now) or generate-forward via the discharge-emission (b)? The original
candidate list assumed the closure data existed; it doesn't — so the excursion's true v0 is
*acquiring the realized-closure signal*, not consuming it.

## Per-closure evidence shape (Joe, 2026-06-10): the M-memes-arrows three-state maturation

Each closure carries **steppable evidence** in the canonical one-arrow-three-states shape
(`futon3a/holes/labs/M-memes-arrows/reference-case-one-arrow-three-stages.edn`): a closure is an
arrow-type keyed by its `(have, want)` endpoints, stepping
`:correlated` (a hunch — co-occurrence, no method) → `:open` (a typed gap = the **sorry**, method
absent) → `:constructed` (the runnable **method = BHK arrow = wiring diagram**). The closure IS the
`:open → :constructed` step; the evidence is that record, with `:token-identity-proof` (same
endpoints — matured, not re-minted) and `:provenance` (commit / cg-chain). The recorder's `:evidence`
field carries this map verbatim (no recorder change needed).

```clojure
;; the :evidence each append-closure! entry carries
{:arrow {:have "<precursor / what we had>" :want "<what the closure constructs>"
         :endpoint-key [:<have-key> :<want-key>]}
 :states
 [{:state :correlated :plain "a hunch"            :evidence "<co-occurrence / rationale>" :construction nil}
  {:state :open        :plain "a gap with a known shape"
   :goal "<the committed goal>" :type-fixed-by "<the contract that fixes the type>"
   :method-absent "<what's missing>" :construction nil}
  {:state :constructed :plain "the machine / the proof" :addressed-at "<date>"
   :construction {:method "<the runnable artifact>" :prerequisite "<deps>"
                  :commit "<sha>" :artifact "<path>"} :same-token? true}]
 :token-identity-proof "same (have,want) endpoints; the arrow matured, it was not re-minted"
 :provenance "<where the construction lives + the commit/cg-chain>"}
```

### Worked example — `kit-outbox` (currently at `:open`; closing fills `:constructed`)
```clojure
{:arrow {:have "daily-scan + interest-network + eoi-new (three working pieces, no pipeline)"
         :want "a staged outbox: scan -> interest-network match -> eoi-new draft -> staged (ready to send)"
         :endpoint-key [:scan+interest+eoi-pieces :staged-outbox]}
 :states
 [{:state :correlated :plain "a hunch"
   :evidence "the three pieces exist + co-occur in cold-outreach work; the registry note 'mostly WIRING existing pieces'"
   :construction nil}
  {:state :open :plain "a gap with a known shape"   ;; <-- kit-outbox sits HERE now (:held island)
   :goal "wire the four steps into one staged-outbox pipeline; clears T2.2"
   :type-fixed-by "T2.2 :pudding-requires — an engine-authored draft in the outbox for a :cold-scan-lead"
   :method-absent "the wiring (the pieces are unconnected; it is an unclaimed island, off-map)"
   :construction nil}
  {:state :constructed :plain "the machine" :addressed-at "PENDING"
   :construction {:method "PENDING — the staged-outbox pipeline" :prerequisite "scan / interest-network / eoi-new"
                  :commit "PENDING" :artifact "PENDING"} :same-token? false}]
 :token-identity-proof "same endpoints across states; will mature in place, not be re-minted"
 :provenance "futon7 registry :kit-outbox; closure lands when the pipeline is wired + committed"}
```
The steppability is the point: anyone can read the record top-to-bottom and see exactly where the
hole *is* (here, `:open`) and what `:constructed` will require — closure is filling the last node.

## Per-closure Lab Notes (Joe, 2026-06-10): make the evidence *visible*

For the **early** closures, ONE consolidated Lab Note `holes/early-closures.md` holds each closure
inline (Markdown + embedded **Mermaid**), so the evidence is legible + **critique-able** by Joe and
Fable — not just machine-parseable EDN. Each Lab Note renders:
- the **three-state maturation** (`:correlated → :open → :constructed`) as a Mermaid state diagram;
- the **wiring diagram** (the `:constructed` state) as a Mermaid flowchart — *see* the construction;
- the **cascade** (`:correlated`) where one exists (thin cascades are themselves a data point);
- provenance, the finding, and an explicit **critique surface** (what Joe/Fable might challenge).
Cascade = real `construct_cascade` Library patterns (NOT a prose hunch); sorry = a real substrate-2
subset (NOT a noted gap) — anchored in patterns + substrate-2 (Joe). First closure: §Closure 01 (q5).

## THE LANDING (Joe, 2026-06-10): the grounding signal is the cascade-fold closure loop — peradam-free

The five early closures resolved E-ground-G's central question. The grounded signal external to the
prior's metric is **the closure-fold loop itself**, and it carries (at least) **three learnable
signals, none of which implicate a peradam**:
1. **pattern-utility** — did a selected pattern's rule actually fold the hole? → update its posterior
   (this is M-pattern-posteriors' signal, now grounded in *fold-usage*, not self-grading).
2. **pattern-missing** — did the fold need a pattern absent from the cascade? → a recorded library
   gap (author it); lowers cascade-coverage confidence for that ψ-shape.
3. **edge-correctness** — were the phylogeny edges the *right* interconnections for this problem? →
   **upvote** the edges that connected the used patterns; **seed** a new co-application edge when two
   patterns are used together but had no phylogeny edge. The phylogeny **learns from closures**.

This is dense (per-pattern, per-edge, per-closure), real (fold-success is a fact, not an estimate),
external to the prior's metric, and **needs no car / no peradams** (peradams remain the sparse audit
— Goodhart-safe, outside the loop). *We were hunting a grounding signal; it was the cascade-fold loop.*

## THE FIX — phylogeny-grounded `construct_cascade` (spec for the build)

**Defect (proven across Closures 01 + 03):** `construct_cascade` ranks by MiniLM cosine only and
ignores the 2,538-edge phylogeny — it over-selects non-combiners (01) and misses foldable holes (03);
its `C` tracks neither meaningfulness nor foldability.

**Build 1 — the core fix (this is the dispatch):**
- (a) **Emit the phylogeny as data.** Refactor `futon6/scripts/pattern_phylogeny.py` to ALSO write
  `futon6/data/pattern-phylogeny-edges.json` = `{patterns:[...], descent:[[x,y]...], co_app:[[a,b,w]...]}`
  (the `cites`/`co` it already computes), keyed by pattern stem. (Today it only writes HTML.)
- (b) **Phylogeny-ground the selection.** In `cascade_construct.py`, change the coherence-greedy step
  to grow along the phylogeny: marginal score `m'(p) = rel(p|ψ) · (α + connectivity(p, chosen))`
  where `connectivity` rewards a descent or co-application edge from `p` to the already-chosen set
  (the cascade grows as a connected semi-lattice, not scattered cosine-neighbours). Patterns with
  **no phylogeny node** are down-weighted and surfaced separately as `:coverage-candidates` (not
  silently mixed in — they are the "is a pattern missing?" signal).
- (c) **Output the structure.** `construct_cascade` returns, in addition to the ranked list, the
  `:semi-lattice {:descent [...] :co-app [...]}` among the chosen patterns + the dropped
  `:non-phylogeny` set. So every cascade is a graph, not a chain.
- **Acceptance:** re-run the kit-outbox + inv-tripwire queries; kit-outbox's on-topic patterns should
  come out *connected* (semi-lattice), and inv-tripwire's near-zero-structure should be *visible in
  the output* (flagged low-connectivity), not hidden behind a cosine ranking.

**Build 2 — the learning hooks (follow-on, designed not dispatched):** after a recorded closure, take
the fold's used-pattern set + the edges among them → (1) bump pattern posteriors (used=useful);
(2) seed/upvote co-app edges among co-used patterns; (3) log missing-pattern gaps. Persist; the next
cascade is better. This is the peradam-free ML loop above, made to run.

## BUILD 2 — the learning hooks (the peradam-free ML loop, designed in full)

**Input:** a recorded closure (closure-ledger entry + its Lab-Note fold) =
`{hole, cascade (proposed patterns), used (the subset whose rules folded it), fold-success?}`.
**A script `futon6/scripts/cascade_learn.py` reads new closures and emits three updates; the next
`construct_cascade` reads the updated artifacts. No peradams anywhere.**

### Update 1 — pattern-utility → posteriors  (grounds M-pattern-posteriors)
For each pattern `p` in the cascade of a *successful* fold:
- `p` **used** in the fold → Beta success bump (`α += 1`): used = useful, observed not self-graded.
- `p` **proposed but unused** → weak negative (`β += κ`, small κ): offered, not load-bearing here.
Persist to `pattern_posteriors.json` (the file already exists; now it is **grounded in fold-usage**,
not self-grading — this is precisely the grounded update path M-pattern-posteriors had escrowed).

### Update 2 — edge-correctness → the phylogeny LEARNS  (Joe's "upvote and seed")
For each pair `(a,b)` **both used** in a successful fold:
- they **have** a phylogeny edge (descent or co-app) → **upvote** it (`w += 1`) — the interconnection
  was the right one for this problem.
- they have **no** edge → **seed** a new co-application edge `(a,b, w=1, :origin closure)` — they
  combined in a real closure; that *is* a new co-application observation.
Persist to `futon6/data/pattern-phylogeny-learned.json` (an overlay ON TOP of the computed
`pattern-phylogeny-edges.json` from Build 1). `construct_cascade` reads computed ∪ learned, so each
closure makes the next cascade's semi-lattice better-fitted to the kind of problem being solved.

### Update 3 — pattern-missing → coverage gaps  (library growth signal)
If a fold needed a pattern **not in the cascade** (named at fold time), append to
`futon6/data/cascade-coverage-gaps.edn` `{:hole :psi :missing "<description>" :at}`. This is the
"author a new pattern" backlog and lowers cascade-coverage confidence for that ψ-shape.

### The loop (grounded, peradam-free)
```
closure (real fold) ──► cascade_learn.py ──► {posteriors, learned-edges, gap-log}
        ▲                                                   │
        └───────────  better construct_cascade  ◄───────────┘
```
Every closure improves the next cascade. The signal is "did the pattern/edge help fold a real hole" —
dense, real, external to the prior, **un-gameable without doing real work** (you cannot fake a fold).
Peradams never enter; they remain the sparse audit that periodically checks this loop hasn't drifted.

**Campaign implication:** this releases the **M-pattern-posteriors grounded-path escrow** —
its "peradam-attributed outcome moves a posterior" requirement is satisfied by the *closure-attributed*
outcome instead (a denser, car-independent grounding of the same shape). To record at STANDARD-VERIFY.
