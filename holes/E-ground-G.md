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

## BUILD 1 — reviewed PASS (claude-3, real gate) + the kit-outbox finding

Commits: futon6 `f14436a` (phylogeny edges export), futon3a `999f3e4` (phylogeny-grounded cascade).
**What I checked (auditable):** read the diff (the marginal is now
`rel·(alpha + connectivity(p,chosen))`, alpha=0.3; phylogeny loaded from the new JSON; non-phylogeny
hits surfaced separately; low-connectivity flagged); **re-ran the demos myself** (kit-outbox: 27→4
patterns, a connected semi-lattice, 14 non-phylogeny surfaced; inv-tripwire: now 9 patterns / 15
edges, on-topic *tension* patterns + LOW-CONNECTIVITY flag); verified return keys preserved
(`cascade/C/H/T` + new `semi-lattice/non-phylogeny/coverage-candidates`); `py_compile` clean. **No
bugs found; the implementation matches the spec.**

**THE FINDING (a live proof that Build 2 is necessary):** the new kit-outbox cascade **drops the two
patterns that actually folded it in Closure 02** (`mission-anchored-scan`, `mission-unlocks-eoi`).
*Why:* they have **zero phylogeny edges to the seed's cluster** — the static phylogeny (co-occurrence
in existing mission docs) doesn't yet know the outbox pattern-language combines. The greedy faithfully
followed the seed's connected component and missed theirs. **This is exactly the gap Build 2 closes:**
Closure 02's fold *used those two together*, so the learning hook would **seed a co-application edge**
between them (and to the outbox cluster), and the *next* phylogeny-grounded cascade would grow into
them. The fix, by grounding honestly, reveals what the phylogeny doesn't know — and the closures are
what teach it. The system discovers its own blind spot by trying to use its knowledge.

**Closure 03 finding, updated:** inv-tripwire is NOT a hard "cascade-miss" under the fix — the
phylogeny-grounded cascade finds an on-topic *tension* cluster (`structural-tension-as-observation`,
`social/tension-before-code`) with real structure (15 edges). The embedding-only "miss" was partly the
embedding's fault; phylogeny-grounding does better. (The low-connectivity flag still fires honestly.)

**Property to track (not a bug):** the phylogeny-greedy is single-cluster-from-seed — it can miss
relevant patterns in other components until the learning loop (Build 2) seeds the cross-cluster edges.
This is the honest state of an incomplete phylogeny; Build 2 resolves it organically.

## THE CURRICULUM ARM (Joe, 2026-06-10): the loop looks for NEW things to learn — the epistemic arm of AIF

The closure-learning loop (Build 2) is only the **exploitation / pragmatic** arm — it learns what's
learnable and builds the capability model's **depth**. Joe's addition: the loop should also **look for
new things to learn** — the **exploration / epistemic** arm. Together they are the two terms of the
Expected Free Energy: **EFE = pragmatic value (close what you can) + epistemic value (seek what would
teach you the most)**. The grounded learning loop is, fully, an **active-inference capability loop**.

What plays each role (already-existing pieces, now named):
- **The Pudding Prover registry = a forward-model of the CV.** Its `:held` theses are capabilities
  *predicated on learning them* — "if you knew it now it wouldn't be listed." The held set IS the
  curriculum / goal-structure; `:satisfied` = achieved CV. [[M-pudding-peradams]] / the registry.
- **M-capability-star-map / M-futonzero-capability = the navigable capability model** (Khan-prior /
  stereolithograph-posterior, missions-as-λ). Its **frontier** (reachable-but-not-yet-had) *is* the
  curriculum's next steps.
- **Two movements:** **build** the model (depth — exploitation, Build 2) and **enlarge** it (breadth —
  propose new capabilities into the CV, the curriculum arm).

**The curriculum signal = expected information gain.** Prefer holes/closures that would *teach* a new
pattern, a new phylogeny edge, or a new capability (expand the model) over those that merely exploit
known patterns. Build 2's **pattern-missing** update is the *seed* of this; the structured version
reads the star-map frontier + the gap-log and proposes *what to learn next*.

**Safety (inherited, load-bearing):** enlargement inherits the star-map's **I4 exogeneity** — "water
doesn't flow uphill." The loop can *notice* it lacks a capability but **cannot unilaterally chase
one** (especially a dangerous one): there is no downhill path to a goal the operator didn't
pre-register. Enlargement of the CV therefore goes **through the operator** (the consent-gate /
dokusan) — the loop *proposes* curriculum, the operator *ratifies* what enters it. [[feedback_operator_not_sovereign]].

**Still peradam-free.** The curriculum is the registry/star-map structure; the drive is
information-gain (model-internal); peradams remain the sparse audit. No imported reward anywhere.

### Build sequence (the whole grounded AIF loop)
- **Build 1** ✓ — phylogeny-grounded `construct_cascade` (reviewed PASS).
- **Build 2** — the closure-learning hooks (posteriors + phylogeny-learns + gap-log) = the
  **exploitation** arm; builds the model's depth.
- **Build 3** — the **curriculum arm**: read the capability-star-map frontier + the gap-log → propose
  new things to learn (highest expected information gain), gated through the operator. = the
  **exploration** arm; enlarges the model's breadth.
- The three together = E-ground-G's real shape: a grounded, peradam-free **active-inference capability
  loop** — exploit (close) + explore (enlarge), navigated by the star-map, predicated on the CV.

## BUILD 3 — the curriculum arm = COUPLE the closure-loop to the star-map's EFE scheduler (designed)

**Key realisation:** the exploration/curriculum arm is *already built* — it is **M-capability-star-map**
(`futon0/holes/missions/M-capability-star-map.graph.edn` + `.../web/.../capability-star-map.graph.json`).
It is a navigable capability graph with an **EFE-over-graph scheduler**
(`aif/expected-free-energy-scorecard`: `G = risk + ambiguity + INFO + cost`) that picks the next
*small, ready* action, and it carries the **I4-exogeneity safety** verbatim ("rolls downhill toward
goals you wrote down in advance; can notice it lacks an ability but can't decide to chase one"). So
Build 3 is **not** new machinery — it is the **coupling** of E-ground-G's grounded closure-loop
(Build 2) to that existing scheduler. The two halves were built separately; Build 3 unifies them into
one active-inference loop.

**The coupling (three wires):**
1. **Grounded model → the EFE `info` term.** Feed Build 2's grounded posteriors + learned phylogeny
   into the scheduler's epistemic term: a hole foldable with **known high-posterior patterns +
   existing edges** = *low* info-gain (exploit, we already know it); a hole needing a **coverage-gap
   pattern** or a **cross-cluster edge the phylogeny lacks** = *high* info-gain (explore, it would
   teach). The grounded model is exactly what tells the scheduler *known vs novel*.
2. **EFE picks the next hole.** `EFE-min = pragmatic (can we close it?) + epistemic (would it teach?)`
   over the bounded candidate set (star-map frontier ∩ open substrate-2 holes).
3. **Closure feeds back.** The fold of the EFE-picked hole → `cascade_learn.py` (Build 2) → updates
   posteriors + learned edges + gap-log → the scheduler re-scores. **The loop closes.**

**Safety (inherited, not added):** the scheduler's downhill-only / pre-registered-goal property holds;
enlarging the CV (adding a new `:held` thesis / capability) is **operator-gated** (consent / dokusan).
The loop proposes curriculum; Joe ratifies what enters it. [[feedback_operator_not_sovereign]].

**Build 3 deliverable (after Build 2 lands + is reviewed):** a thin coupling — feed
`pattern_posteriors.grounded.json` + `pattern-phylogeny-learned.json` + `cascade-coverage-gaps.edn`
into the star-map EFE scorecard's info term; emit a ranked **curriculum proposal** (top-K holes by
EFE = pragmatic+epistemic) to the operator pane for ratification. The closure of a ratified pick
re-enters Build 2. = E-ground-G's grounded learning ⨝ M-capability-star-map's scheduler, one loop.

## BUILD 2 — reviewed PASS (claude-3, real gate) + the loop demonstrated end-to-end

Commits: futon6 `bb86d76` (cascade_learn.py + the 3 updates), futon3a `21ace82` (cascade reads
computed ∪ learned) + review-fix (Beta(1,1) prior). **What I checked (auditable):** read the diff
(posteriors bump used-patterns; co-used pairs upvote-or-seed the phylogeny overlay; gaps logged);
**re-ran `cascade_learn.py` myself — idempotent (sha256 stable)**; verified the overlay
(`mission-anchored-scan↔mission-unlocks-eoi` upvoted 1→2; `model-recompute-schedule↔prototype-maturity-lifecycle`
**seeded** w=1); **ran `learned-demo` myself** and confirmed the downstream effect. **Found + fixed
one issue** (the Beta(1,1) prior). py_compile clean; construct_cascade keys preserved.

**THE LOOP, DEMONSTRATED END-TO-END (the falsifiable test passing):** Closure 05 (kit-cadence) used
`model-recompute-schedule` + `prototype-maturity-lifecycle` together → `cascade_learn` **seeded** the
co-app edge → the *next* kit-cadence cascade (computed ∪ learned) **adds `model-recompute-schedule`**
(size 6→7) where computed-only omitted it. **A real closure taught the phylogeny an edge that improved
the next cascade.** Grounded, dense, real, peradam-free — exactly the claim. Minor note (non-blocking):
the EDN parse in cascade_learn.py is regex-based — fine for the controlled input, swap to a real EDN
reader if it grows.

## THE CYBORG ⨝ THE SHARED STORE (Joe → E-mission-head, 2026-06-10)

`E-mission-head.md` §2.5 writes its own cascade→sorry→wiring diagram **into the same
`futon3a/meme.db`** (scope-tag `diagram/E-mission-head`), under **the same E-ground-G standard**
(realized closure, no laundering). Verified live: `meme.db` went **3 → 10 arrows** (was empty when
E-ground-G's v0 first looked); the E-mission-head diagram = **4 `:constructed` + 3 `:open`**. *"We
(Joe) are the Cyborg version of that same learning loop."* This unifies several threads:

1. **The store is the shared substrate, written by BOTH agents and the Cyborg.** Agent
   pattern-fold closures (`closure-folds.edn` / the closure-ledger) and Cyborg mission-diagram
   closures (`meme.db` arrows) are the *same kind of thing* — `:correlated → :open → :constructed`
   maturations — in the same store. The grounded learning loop is a **human-agent shared loop**,
   not an agent-only one. (Cyborg / coupled-thinking, made literal: Joe writes to the store the
   agents learn from, and learns from the store the agents write to.)
2. **The original candidate-1 grounding signal is now being populated — by the Cyborg.** E-ground-G's
   v0 found `meme.db` empty (no realized-closure data); E-mission-head's `promote!`-ing of its 3
   `:open` sorries → `:constructed` are exactly the realized substrate-2 closure events candidate-1
   wanted. The Cyborg is the populator the loop was waiting for.
3. **E-mission-head is the GOAL half; E-ground-G is the LEARNING half — they meet at the store.**
   The mission HEAD-as-AIF-object (priors/preferences/observations/policies; the 10 `observe.clj`
   channels) is the **satisfaction-conditions** grounded-G discharges against — i.e. the CV /
   curriculum structure the curriculum arm (Build 3) explores toward. E-mission-head §HEAD says it
   explicitly: "a HEAD mapped to AIF terminal vocabulary is what grounded-G ultimately discharges
   against."
4. **The curriculum surface (the Build-3 question I held for Joe) is answered:** the mission-mode
   **lifeform-viewer** (E-mission-head's open sorry `readout → mission-mode-lifeform-lane`) — the
   panel that shows a mission's *health/strength*. Curriculum proposals ("what to learn next", by
   EFE) surface there, in Joe's Cyborg interface, where his ratification already lives.

**Integration (a real step, not done):** point E-ground-G's closure-learning at `meme.db` (the
unified store) so Cyborg closures feed it too — needs a grain bridge (meme.db arrows are entity-keyed
mission-diagram arrows; the pattern-learning loop is pattern-stem-keyed). Recognise the shared
standard + store now; build the bridge when the curriculum arm (Build 3) lands.
