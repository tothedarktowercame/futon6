# APM-Lean Ground Control — handover notes

**Written 2026-08-09 ~18:40 BST by claude-3, for claude-7 taking over.**
Operator: Joe (surface `emacs-repl`). This is the M-diagramprover / APM Lean
formalization campaign.

---

## 0. The one thing that will waste your first hour

**The campaign lives on ZONE, not on Dionysus.** `ssh zone-joe`
(104.243.39.24). I lost a pass to this: I ran the review tooling in
`/home/joe/code/futon3c/holes/labs/M-diagramprover/apm-driver` on Dionysus,
got "No such file", and briefly thought work had been destroyed.

| | Dionysus (local) | Zone (`ssh zone-joe`) |
|---|---|---|
| `apm-driver/` | statement bank only, 373 manifest rows | **the real campaign, 2559 rows** |
| bridge lane | absent | `bridge_lane.py`, `bridge_packets.py`, `bridge_review.py` |
| Lean corpus | absent | `/home/joe/code/apm-lean/problems/<pid>/lean/Main.lean` |
| Mathlib | absent | `/home/joe/code/apm-lean/.lake/packages/mathlib/Mathlib` |
| Agency :7070 | your parks + your session live here | a **separate** Agency, 48 agents |

Both hosts run an Agency on :7070 **with overlapping agent names**. An agent
name is meaningless without the host. `zai-1` exists on both and is a different
agent on each. `claude-3` (and presumably `claude-7`) is **not** on Zone's
roster, so Zone cannot bell you and your parks cannot ride Zone job-ids.

Also: two scripts that earlier park payloads told me to run — `night_sweep.py`
and `dump_bridges.py` — **do not exist on either host and never did**. Payloads
had been carrying invented tool names forward for several cycles. If a payload
names a tool, check it exists before trusting the instruction.

## 1. The regime

Codex formalizes an APM problem into Lean → Claude reviews the statement for
fidelity → the statement is **frozen by hash** → provers work against the frozen
statement. Frozen contracts are `frozen_hash` (main theorem) and
`frozen_declarations` (declaration set). **Neither covers `def` bodies** — that
is an open gap Joe knows about and has not yet ruled on.

When a prover cannot close a problem it reports one of:

- **Tier A** — names a specific missing Mathlib lemma ("prove-or-find").
- **Tier B** — adds a *bridge*: a statable intermediate lemma that would unblock.
- **defective statement** — allowed, and must carry brief evidence.

Joe's standing model (his words, 2026-08-09): the closer should read the
literature and model the proof; if they then report a missing Mathlib item,
**trust them and go extend Mathlib** rather than re-running the same loop. Tier
B items effectively *become* new ConstructionTargets.

## 2. Your job: review. It is a real gate, not a rubber stamp

### Tier B — the bridge test

Two questions, both required:

1. **Is it TRUE?**
2. **Is it something OTHER than the goal repackaged, AND does it supply
   non-trivial mathematics?**

Note it is **not** "is it strictly weaker" — that test was wrong twice and would
have rejected `t02A05`, whose bridge was *stronger* (a closed form).

**Seven rejection kinds, every one of which produces a TRUE statement** — which
is exactly why truth alone cannot be the test:

| kind | shape | example |
|---|---|---|
| verbatim | a conjunct handed back unchanged | `t02A06` bridge_2 |
| unfolded | a definition expanded in place | `m94J04` |
| conjunction/iff split | goal's own conjuncts, nothing added | `m02J03`, `b97J03` |
| equivalent reformulation | same content, different clothes | `m96J03`, `m00A02` |
| dependency ordering | assumes conjunct 1, concludes conjunct 2 | `a01A12` |
| restriction | goal with a strengthened hypothesis | `t94J04` (`2 ≤ k` for `0 < k`) |
| **trivial / easy-half** (new, 1 instance) | genuinely different, but no mathematics | `m97A01` |

The seventh is flagged as a **first instance, not an established pattern** —
promote it only if it recurs.

**ALWAYS READ THE DEFINITIONS THE BRIDGE MENTIONS.** This is the single
highest-yield habit and it has caught near-misses repeatedly:

- `m94J04` looked like a concrete computation until I read the defs — `u` is `f`
  times basis 0 and `curlValue` is an explicit indicator, so the bridge was the
  goal with inner products expanded, **and the file already proved that
  expansion** as `apm_m94J04_inner_u_curlTest`.
- `m96J03` wore **two disguises at once** (reformulation *and* unfolding);
  either alone was catchable, together they read as a different lemma.
- `m00A02` collapsed onto the goal via `apm_m00A02_expansion_coefficients_unique`
  — proved, no sorry, **forty lines above in the same file**.

Recurring tell: *the file itself already contains the lemma that collapses the
bridge onto the goal.* Grep the file before ruling.

Contrast — bridges that were **accepted**: `t01J06` (supplied a *mechanism*: a
closed extension forces the sphere integral to vanish), `m03J04` (constructed an
explicit isomorphism), and the model case `m96A05`, whose bridge_1 discharges
precisely the `hscheme` hypothesis its own file's `apm_m96a05_unique_of_det`
openly demands.

### Tier A — grep Mathlib AND THIS REPO before believing the gap

> **THE SCREEN IS TWO-SIDED. I got this wrong (claude-7, 2026-08-09), Joe caught
> it.** The heading below used to say only "grep Mathlib", so that is all I did
> — for nine problems. It is half the check. The other half:
>
> ```bash
> ssh zone-joe; cd /home/joe/code/apm-lean
> grep -in "<concept>" LEMMA-INDEX.md        # 2098 proved lemmas, 182 importable
> ls ConstructionTargets/                     # 17 importable modules
> ```
>
> `LEMMA-INDEX.md`'s own header says **"GREP THIS BEFORE RE-DERIVING
> ANYTHING"** and names `import ConstructionTargets.Rouche` as its example. The
> closer packet already tells closers to consult both; the *review* protocol
> did not, so the reviewer was screening on strictly less than the author.
>
> How it bit: I "corrected" `a97J08` for claiming disk Rouché exists, having
> found nothing in Mathlib. It exists in **this repo** —
> `ConstructionTargets.Rouche.zeroCountInClosedBall_add_eq` — and `a01J05` and
> `aunk04` already import it. The report was accurate; my correction was not.
> Withdrawn in the manifest.
>
> **The tell I should have caught:** every other misdirection this session had a
> closer too *pessimistic* about existing material. That was the only one where
> I claimed a closer was too *optimistic*. When a finding inverts the
> established direction of error, check your setup before publishing it.
>
> Re-audited all five surviving accepts against the index — `b97J04`, `m02J01`,
> `m95J01`, `m96J04`, `b93J01` all clear. Rejects are unaffected (a Mathlib
> declaration is still there regardless of the repo index). Damage was confined
> to that one paragraph.

**2 of 5 Tier A items I checked named a "Mathlib gap" that was not one.** Both
reports were honest and internally coherent; they simply had not found the
declaration. That is a ~40% misdirection rate on a lane whose premise is "trust
them and extend Mathlib," so **the grep is mandatory** before routing anything
to a Mathlib lane. It costs one command.

- `m97A03` — named `intervalIntegral.continuousWithinAt_of_dominated_interval`
  as missing; it **exists** at `MeasureTheory/Integral/DominatedConvergence.lean:278`.
  Real work was local instantiation. Reclassified, not rejected.
- `m02A06` — claimed no Mathlib declaration packages the sup-norm contraction;
  `ContractingWith.exists_fixedPoint'` (`Topology/MetricSpace/Contracting.lean:151`)
  is exactly the right shape. It *also* stated its "missing lemma" in the
  problem's own vocabulary (`apm_m02a06_greenTransform`), i.e. the goal in
  disguise — Tier A can fail the repackaging test too.
- Genuinely absent, correctly diagnosed: `b93A01` (two-prime permutation bound —
  I verified it is true and sharp), `b98A04` (Jordan–Hölder factor occurrence),
  `m95A04` (Mathlib has **no** weak-derivative API at all).

## 3. Running the loop

```bash
ssh zone-joe
cd ~/code/futon3c/holes/labs/M-diagramprover/apm-driver
python3 bridge_review.py                 # queue + AWAITING count; --seen <ids> marks done
python3 /tmp/dump_batch.py <pid> [<pid>...]   # FULL job text + the Lean source
```

`dump_batch.py` is mine, written this session; it exists on Zone at `/tmp` and
is reproduced in this repo's history if it vanishes. It dumps the full result
**plus** `apm-lean/problems/<pid>/lean/*.lean`, because reviewing without the
definitions is how you get fooled.

Record verdicts with a script modelled on `/tmp/review_day7.py` … `day9.py`
(append to `sc.MANIFEST` with status `bridges-reviewed` / `bridges-rejected` /
`tier-a-reviewed` / `tier-a-rejected`, then `bridge_review.py --seen <ids>`).

**Keep `check=True` on that final `subprocess.run`.** The `day2`–`day6` scripts
omitted it, so a failed `--seen` marking would have been invisible. I checked —
those did land — but do not re-introduce the hole.

**Batch ~6 at most.** Bulk review measurably degrades: 51-at-once gave a 22%
defect rate versus 14% in sixes, and *every* miss was in a local `def`, not the
statement itself.

## 4. State as of handover

- **Bridge queue: 91 awaiting** (169 jobs, 78 reviewed). Bridge lane fully
  drained of dispatches; everything left is review.
  → **claude-7, 2026-08-09 ~19:30 BST: now 85 awaiting** (84 reviewed).
  Reviewed 6: `b93J01` (sound), `b97J04` (accepted on 1 of 3),
  `m97A04` **rejected**, `m95J01` (accepted on 1 of 2), `m02J01` (accepted),
  `m93J03` **rejected**. Detail in §4a below.
- This session reviewed 8: `m96A05` sound; `m00A02`, `m97A01` rejected;
  `m95A04`, `b98A04`, `b93A01` Tier A accepted; `m97A03` reclassified;
  `m02A06` Tier A rejected.
- **`codex-3` is mid-flight** on `POST /api/alpha/park/complete` — job
  `invoke-1786296498904-287-2e4cd3ca`, I am parked on it as `park-065bd6f2`.
  **This needs your review when it lands** (see §6).
- Also live: `park-9adc9314` on sentinel dep `apm-bridge-review` (clock-only).
  Reconcile the two parks on your first wake.
- 4 repaired statement defects await review: `m99J04`, `m99J06`, `t03J04`, `t94A06`.
- Confirmed defect clusters: manifold conventions (4 — `t01J04`, `t94A06`,
  `t03J04`, `t01A08`; **narrower than feared**, not systemic across all 30),
  totalised junk values (`m96J02`, `m93J05`), Mayer–Vietoris/cellular-to-singular.
- Screen counts are **upper bounds, not defect counts**: `deriv`-with-`ContDiffOn`
  39, `.toReal` 12, `sSup`/`sInf` 6. `a99J07`'s `sSup` was checked and is fine.

**Untested hypothesis, do not build on it:** every SOUND manifold verdict
justified itself by *concreteness* (embedded subset of ℝⁿ, or a specific
quotient, so second countability and Hausdorff are inherited), while defects
quantify over an *arbitrary* manifold. That predicts a mechanical screen. **I
tried to verify it with a grep and the grep was wrong** — it labelled `t01A08`
concrete when that file quantifies over arbitrary `E`, `H`, `X`. Any screen
needs a real structural check of the binders.

## 4a. claude-7's first batch — the Mathlib grep is now the top screen

Three Tier A items, every named gap grepped before ruling. **The screen fired on
2 of 3.**

- **`b93J01` — SOUND.** Outcome A, proved locally. I re-derived the counting
  argument rather than trusting it: `x² = 1` has exactly 2 solutions in `Q₈`
  (`1` and the central involution; `±i, ±j, ±k` all square to `−1`) and exactly
  6 in `D₄` (identity, the half-turn `r²`, and all four reflections), and a
  `MulEquiv` carries that subtype bijectively, so `2 = 6` is the contradiction.
  All four *remaining* gaps it names are genuinely absent — Mathlib has only
  `alternatingGroup.isSimpleGroup_five` (`Alternating.lean:385`, and its own
  module header says general `n ≥ 5` is future work), nothing for "A₄ has no
  subgroup of order 6", nothing for order-`pq` cyclicity under `p ∤ q−1`.
- **`b97J04` — ACCEPTED ON 1 OF 3.** `charpoly S = charpoly (S+N)` for
  commuting semisimple/nilpotent is stated in pure library vocabulary and is
  genuinely missing (Mathlib has `IsNilpotent.charpoly_eq_X_pow_finrank` and
  `charpoly_nilpotent_tfae` in `Eigenspace/Zero.lean`, but no
  commuting-perturbation lemma at all) → route it. The other two are the frozen
  goal's own conjuncts in the problem's `APM*` vocabulary. One of them isn't
  even an absence: the reverse-inclusion route needs
  `IsFinitelySemisimple.maxGenEigenspace_eq_eigenspace`
  (`Eigenspace/Semisimple.lean:69`) and
  `isNilpotent_restrict_maxGenEigenspace_sub_algebraMap`
  (`Eigenspace/Basic.lean:618`), **both of which exist** — it is local assembly.
- **`m97A04` — REJECTED.** Both named lemmas are computations about the
  problem's own `apm_m97a04_matrix`. Decisively, the report says its second
  lemma will "provide Gershgorin strict row dominance" — but
  **`Matrix.det_ne_zero_of_sum_row_lt_diag` already exists** at
  `LinearAlgebra/Matrix/Gershgorin.lean:62`, and its hypothesis is *literally*
  the `∑ j ∈ Finset.univ.erase k, ‖A k j‖ < ‖A k k‖` shape being hand-built.
  This is instantiation, not prove-or-find.

Second batch (`m95J01`, `m02J01`, `m93J03`) — the screen fired on 2 of 3 again.

- **`m95J01` — accepted on 1 of 2.** The triangular right inverse for
  `p ↦ p'' − 2p'` on `degreeLE n` is genuine and clean (the `apm_` prefix is
  only in the *name*; the statement is pure `Polynomial` vocabulary). I checked
  it is true by dimension count rather than assuming. But its second lemma,
  "adapted Gram–Schmidt", **is already in Mathlib almost verbatim**:
  `gramSchmidt_mem_span` (`GramSchmidtOrtho.lean:136`) states exactly the
  claimed missing conclusion, with `span_gramSchmidt_Iic` (:148) and
  `gramSchmidtOrthonormalBasis` (:32) completing it.
- **`m02J01` — accepted.** Genuine absence, confirmed by grep: Mathlib has a
  real test-function API and differentiation as a bundled operator
  (`fderivCLM`), but **no primitive/antiderivative anywhere under
  `Analysis/Distribution`**. And it supplies a *mechanism*, not the goal — the
  accepted `t01J06`/`m96A05` class.
- **`m93J03` — rejected.** It says Mathlib gives only *ordinary* Fréchet
  derivatives under the integral and nothing strict. Half right — there is no
  strict integral variant — but the strict property doesn't need one, because
  the **upgrade lemma exists**:
  `hasStrictFDerivAt_of_hasFDerivAt_of_continuousAt` (`MeanValue.lean:803`),
  with `ContDiffAt.hasStrictFDerivAt` (`ContDiff/RCLike.lean:62`) as the C¹
  route. Ordinary derivative + continuity ⇒ strict. Route already complete.

**Running Tier A misdirection: 6 of 14 named gaps checked were not gaps**
(`m97A03`, `m02A06`, `m97A04`, `m93J03`, `m95J01`'s second lemma, and
`b97J04`'s second). The rate is stable across two reviewers and twelve
problems, so treat §2's grep as the load-bearing screen on this lane, not a
spot-check. Two sub-patterns worth carrying forward:

1. **Read the prose, not just the ```lean block.** In `b97J04`, `m97A04` and
   `m93J03` the report's own prose named the correct route while its formal
   "missing lemma" block named the goal restated. The block is where the
   repackaging hides.
2. **The near-miss shape is "the engine exists, the instantiation doesn't."**
   Four of the six were not wrong about the mathematics — they had simply not
   found `ContractingWith.exists_fixedPoint'`, `det_ne_zero_of_sum_row_lt_diag`,
   `gramSchmidt_mem_span`, `hasStrictFDerivAt_of_hasFDerivAt_of_continuousAt`.
   These are all *composition* points, which suggests the closers search for
   the theorem they want rather than for the last step of a chain.

## 4d. THE TRAPPED-LEMMA AUDIT — measured, and it is structural

Joe flagged this as major concern after §4c. I measured it rather than
estimating (`/tmp/c7_lemma_audit.py` on Zone, parses `LEMMA-INDEX.md`).

```
TOTAL proved lemmas         2098
  importable (LIB:)          182   across  17 modules
  trapped in problem files  1916   across 362 problems     <- 91%

Promotability (vocabulary test on the SIGNATURE):
  mentions an apm_ definition -> problem-specific   773
  mentions NO apm_ definition -> candidate general 1143   (59% of trapped)
     minus short/bare-named (may use unprefixed local defs)  572
     CONSERVATIVELY PROMOTABLE                               571

Redundancy:
  identical normalised signature in 2+ places   54 groups, 125 lemmas
  same name (pid stripped) in 2+ places         93 names,  219 lemmas
```

**571 conservatively promotable against 182 currently importable — the
reusable library is sized at roughly a quarter of what is already proved.**

**The sharpest single finding.** The worst redundancy group is twelve
lemniscate-component lemmas appearing **three times each**: once in
`ConstructionTargets/LemniscateComponents.lean` and again, independently, in
`a00J04` and `a01A08`. I checked the imports: **neither problem imports the
module** — both pull raw Mathlib. So the promotion already happened and the
problem files still bypass it. That subset is fixable with an import rewrite
and no new mathematics.

**Why this bears on the review lane.** A closer that cannot `import` a lemma
must read the source and re-derive it, or give up and report a missing Mathlib
lemma. That is very likely a contributor to the 6-of-14 Tier A misdirection
rate in §4a — two of the three refinements in §4c came from *trapped* general
lemmas (`apm_m98a05_hasStrictFDerivAt_of_contDiff` is not problem-specific in
any respect; it just lives in m98A05). The trapped library and the false-gap
rate look like the same problem seen from two ends.

### The scribe has NOT been run on the recent campaign

`scribe.md` (the SCRIBE PASS template: draft + promotion, memory entries with
retrievable tags, the required **hunger audit** of memory queries that returned
empty, approvals to claude-10) is **unmodified since 2026-08-04 09:22**, while
the campaign ran on the 8th and 9th. `ams-scribe-1` appears 52 times in
`bridge-pilot-jobs.jsonl` — but as a **bridge-lane SEAT**, i.e. it was
repurposed as a generic hole-closing worker, not running scribe passes.

Joe's inference is correct and now has a mechanism: a memory pointing at a
lemma that cannot be imported is a "go read this file and re-derive it"
pointer, not a reuse pointer. With 91% trapped, that is what most math-lane
memories would be. **Fix the importability first, or the scribe writes an index
of things nobody can use.**

### Recommended order (needs Joe's ruling, not yet authorised)

1. **Import-rewrite the already-promoted duplicates** — cheapest, no new
   proofs, and it validates the whole idea. Start with `a00J04`/`a01A08` →
   `ConstructionTargets.LemniscateComponents`.
2. **Promotion pass on the 571.** The §4b vocabulary test is exactly the filter
   that selects them mechanically; group by subject into new
   `ConstructionTargets/` modules.
3. **Dedupe the 54 statement-collision groups** as part of (2).
4. **Only then** re-run the scribe, so its memories carry importable pointers.

## 4m. LOOP COMPLETE — queue drained 91 → 0 (2026-08-09)

The autonomous chain ran **26 iterations** and terminated on its own guard
(`STOP: queue empty (AWAITING REVIEW = 0). Not arming.`). `bridge_review.py`
reports **169 jobs / 169 reviewed / 0 awaiting**; no parks outstanding.

**claude-7 verdicts: 94 records across 90 distinct problems.**

| status | n |
|---|---|
| `bridges-rejected` | 33 |
| `bridges-reviewed` | 27 |
| `tier-a-reviewed` | 19 |
| `tier-a-rejected` | 15 |

≈ 49% accepted, 51% rejected.

### The three recurring causes

1. **Repackaging** — the largest by far. Sub-kinds, all observed repeatedly:
   *verbatim* (conjunct handed back), *curried* (∀ turned into binders —
   the same proposition), *unfolded* (definition delta-expanded: `t96J06`,
   `m98J02`, `t94J08`, `t97A04`), *conjunction split*, and *restriction*
   (`t01J05`, `b94J01`). Extreme case: `a96J08`, where the "bridge" is the
   **entire frozen theorem**, binders and all.
2. **Engine exists** — the named gap is real mathematics but Mathlib already
   has it, or has the last step: `ContractingWith.exists_fixedPoint'`,
   `det_ne_zero_of_sum_row_lt_diag`, `gramSchmidt_mem_span`,
   `hasStrictFDerivAt_of_hasFDerivAt_of_continuousAt`,
   `cauchy_map_of_uniformCauchySeqOn_fderiv`, `Integrable.tendsto_setIntegral_nhds_zero`,
   `IsEisensteinAt.irreducible`, `IsCyclic Gal(L/K)`, Jensen/value-distribution.
3. **Blocked on algebraic topology — 18 problems**, one root cause.

### The single most useful finding: the topology backlog is small

Eighteen problems (`a97J08`, `t96J05`, `t94A07`, `t01A02`, `t96J06`, `t00J01`,
`t02A04`, `t01J05`, `t92J05`, `t97A04`, `t97A02`, `t97A03`, `t00A01`, `t03J02`,
`t97J02`, `t02A03`, `t94J08`, `t01A01`) block on the same missing package.
Mathlib's gap runs deeper than expected: **no fundamental group is computed
anywhere — not the torus, not even the circle**; `SingularHomology` is a bare
functor; no Lefschetz, no Poincaré–Hopf, no mod-2 intersection theory, no
regular value theorem, no winding number.

But the *asks* converge on a handful of items, and **`t91A05` names the
cheapest**: it needs only `Nontrivial (FundamentalGroup S¹ base)` — not
`≅ ℤ`. One explicit loop and a proof it is not null-homotopic. That single item
feeds no-retraction, Brouwer, and (with π₁(RP²)) `t01A01`. **If one piece of the
topology backlog is done first, it is this.**

### Cross-problem consolidations found (one lemma → two problems each)

- cyclic-number criterion `gcd(n, φ(n)) = 1 → IsCyclic` — `b93J01` + `b01A02`
- sup-norm contraction on a ball — `m02A06` + `m94A05`
- global Picard–Lindelöf by continuation — `m93J06` + `m00A05`
- Gaussian heat kernel — `a96A04` + `m98J05` (**the only cross-class pair**)
- regular value theorem — `t96J05` + `t97J05`

### Promotion candidates (general, trapped in problem files)

`apm_m98a05_hasStrictFDerivAt_of_contDiff`,
`t95J05_hausdorffIntegral_eq_zero_of_isometry_odd`,
`bpm_1_8_1_concaveOn_of_pointwise_tendsto` (**not even in `LEMMA-INDEX.md`**).

### Repairs applied during the loop

- **`bridge_lane.py` race fixed** — `a95J03` was dispatched twice, 0.8 s apart,
  to the same seat. Worklist was clean (167 rows, 167 unique), so a
  read-then-write race was the only cause; added a non-blocking `fcntl` lock.
- **Packet policy** — seven rejection shapes, two-sided search receipt,
  no-duplication policy naming the 94% same-class figure.
- **`a97J08` verdict corrected** (see §2) after Joe caught the one-sided screen.

### Corpus defects recorded, not applied (frozen-file rule)

`native_decide` in 10 files — `b00J01, b01A02, b94A01, b94J03, b96J02, b96J03,
b97A01, b98A01, b99A02, t03J03`. **Nine of ten are class `b`**; `t03J03` is the
lone exception. `b96J04` demonstrated kernel `decide` suffices as a replacement.

### Behaviour worth knowing before the restart

The failure is **not** capability. In nearly every rejection the closer's
*prose* names the correct missing mathematics while its Lean block restates the
goal — `a96J08` diagnosed the absent residue theorem precisely, then delivered
the theorem itself as its "bridge". Two closers did the right thing unprompted:
`a97J06` flagged its bridge as **shared with `a03J05`**, and `t96A08` recorded a
cross-problem search of `t00J04` **with a documented discard and reason**. The
behaviour exists in the population; it is just rare. That is a better prior for
the packet changes working than if none had been seen.

## 4l. AUTONOMOUS REVIEW LOOP — armed 2026-08-09 (Joe: no reprompt, no approval)

`/tmp/c7_chain.py <n>` on Dionysus. Each iteration parks `claude-7` on a fresh
dep `apm-review-chain-<n>` carrying the *next* iteration's payload, then
`POST /park/complete`s it — which resumes claude-7 as a new turn. Verified end
to end: park → `released-count 1` → invoke job for `claude-7` with
`caller=parked-resume` queued.

- surface **`headless`** — NOT `emacs-*`; buffer surfaces route to a ready-inbox
  that needs a poller (see the `/parked` note in §5). Headless enqueues on the
  agent's own drainer lane, which is how bells already reach claude-7.
- `deadline-ms` absolute epoch-ms, +1h, as the backstop if a tick is lost.
- Session id is baked in; **re-check it after any JVM restart** — parks do not
  survive one (§5) and the route activation in §7 doesn't either.

**The payload is a self-contained protocol, not a reminder** — context may be
summarised away across 20+ iterations, so each wake carries the full checklist:
batch of 3; read the Lean source and the definitions; **two-sided** search
(Mathlib *and* `LEMMA-INDEX.md`/`ConstructionTargets`) with the five
engine-exists examples named; the vocabulary test; duplication recorded as fact
with the same-class hint; record with `check=True`; re-arm last.

**Bounded by construction — these are correctness stops, not approval gates:**
refuses to arm at `AWAITING REVIEW = 0`, at iteration cap 30, and after two
consecutive iterations with no verdicts. Also stops if one pid fails twice.

### Scope judgement on "build the repairs as you go"

Joe asked for repairs in-cycle. I split them, and this narrows his instruction,
so it is stated explicitly rather than assumed:

- **Applied in-loop:** recording scripts, corrections to my own earlier verdicts
  (appended to the manifest, never silently rewritten), ground-control doc
  updates, restating a badly-phrased target inside the verdict note.
- **Recorded, NOT applied in-loop:** anything editing a statement-frozen problem
  file — including the `native_decide` fixes. Reason in §4e: the freeze hashes
  **text**, so an edit can pass the gate while silently changing the
  proposition. An unattended loop is the worst possible context in which to take
  that risk, because nobody is watching the diff. The loop records the defect
  and the fix it would apply.

If Joe wants corpus repairs inside the loop too, the safe way is a separate
attended pass with the freeze-integrity baseline (Phase 0) diffed before and
after — not an unsupervised edit.

## 4k. NO-DUPLICATION POLICY — deployed, and the review protocol now records dups

Per Joe (2026-08-09): *"eliminate duplication going forward and note this as a
policy change for all associated agents… still give an honest account of what's
happened so far… note any duplications but not as stop-the-line errors, just as
facts."*

**Deployed** into `bridge_packets.py`'s `COMMON` block, so it reaches Tier A and
Tier B and both the pilot and the unattended lane. It tells closers to grep the
**statement shape** rather than a guessed name, to **look hardest at solved
problems in their own prelim class** (naming the measured 94%), to reuse by
`import` when importable and by cited attribution when not, and to flag anything
they prove that looks general so it can be promoted. Two honest caveats are
built in: **the index can be stale**, so absence is weak evidence; and **finding
a duplicate is not an error and not a reason to stop.**

**Review protocol addition:** verdicts now record duplication as a plain fact.

### First duplication-as-fact — and it is a reuse *win*

`b01A02` names two missing lemmas: groups of order **85** and of order **255**
are cyclic. Both true (85 = 5·17, φ = 64, gcd = 1; 255 = 3·5·17, φ = 128,
gcd = 1 — both cyclic numbers), both in clean library vocabulary, and confirmed
absent from Mathlib and from this repo.

`b93J01`, reviewed earlier the same session, names as *its* residual gap the
cyclicity of groups of order `p·q` when `p ∤ q−1`. **That is the same theorem**:
for `p < q`, `gcd(pq, (p−1)(q−1)) = 1` exactly when `p ∤ q−1`.

So rather than routing three order-specific lemmas, route **one**:
`Nat.Coprime n (Nat.totient n) → Nat.card G = n → IsCyclic G`. One promoted
lemma closes gaps in two problems — **both class `b`**, exactly as the 94%
same-class finding predicts. This is the policy paying off on its first
application, and it is why "more to choose from" is the right frame: the
duplicate pair is what *revealed* the general theorem.

### Systemic defect found in passing: `native_decide`

`b01A02`'s verbatim axiom output carries
`…_native.native_decide.ax_1_1` / `ax_1_3`. The campaign's own HARD RULES ban
`native_decide` and say it downgrades an outcome to **defective**. The report
disclosed it honestly and correctly noted the hop did not introduce it — but the
artifact is contaminated, and the use is gratuitous (discharging
`Nat.card S = 17` from `Nat.card G = 255`, arithmetic `norm_num` should close).

**`native_decide` appears in 10 problem files corpus-wide.** That is its own
sweep item — flagged, not blocking, per the ruling.

## 4j. OPERATOR RULING (Joe, 2026-08-09) — file as learning, clean up post-APM

Joe's call, and it supersedes the §4i action proposal:

- **Do NOT restructure the lane mid-stream.** Some classes are largely absorbed,
  so 4-way class stratification is not straightforwardly available.
- **File §4i as a learning for next time**, and accept working a bit more slowly
  to avoid the pitfall.
- **Post-hoc cleanup AFTER the whole APM is done**, not mid-stream.
- **The main point is that agents should learn to REUSE lemmas going forward.**
- **"If they have duplicates then they have more to choose from."**

That last line is the right reframe and it aligns exactly with the additive-only
policy in §4e: **do not delete duplicates — cross-reference them.** A duplicate
is extra surface area for reuse, not purely waste. The index should show all
variants (Fable's `variants.jsonl`), making the choice richer rather than
pruning it. Deletion was never safe anyway; now it is not even desirable.

### Two facts for the record, one of which qualifies the premise

**Remaining sorries by class** (counted 2026-08-09):

| class | problems | still carrying `sorry` | remaining |
|---|---|---|---|
| a | 151 | 44 | 29% |
| m | 102 | **79** | 77% |
| t | 146 | **85** | 58% |
| b | 76 | 40 | 53% |

Class `a` is indeed largely absorbed. But `m`, `t` and `b` all still hold 40+
open problems each, so **4-way stratification still has headroom on these
numbers** — every class could supply a concurrent slot for some time yet.
Flagging once, as data; the scheduling call is Joe's and it stands. (Note the
bridge *review* queue and the *sorry* count are different populations — the
bridge lane's own worklist is exhausted, which may be the binding constraint.)

**THE INDEX HAS NO AUTOMATION AT ALL.** `lemma_index.py` has no caller: nothing
in the driver invokes it, there is no cron entry, no shell hook. It is a manual
script someone runs. That fully explains the 9 lifetime regenerations and the
staleness that caused 7 of the 15 post-index duplicates.

**This is the cheapest possible lever on "agents should reuse going forward"**,
and it needs no restructuring of anything: regenerate the index whenever a close
lands (or on a short timer). The generator is a lexical scan — cheap. Without
it, every instruction to "grep the index" is aimed at a snapshot that can be a
day old, and reuse cannot improve no matter what the packet says.

## 4i. Duplication is CLASS-LOCAL — 94% (Joe's hypothesis, confirmed)

Joe: *"we were hoisted by our own concurrency petard — what we should have done
is concurrent proofs from the different prelim classes (a, m, t, b)."* Tested:

```
duplicate groups within a single prelim class : 51   (stratification prevents)
duplicate groups spanning 2+ classes          :  3   (it cannot)
                              class-local share: 94%
```

The entire cross-class residue is **one** cluster: `a96A04` / `m98J05` sharing
three heat-kernel lemmas (`Integrable (heatKernel t)`, `∫ heatKernel = 1`,
`0 ≤ heatKernel`). So class is an excellent proxy for subject adjacency — not
perfect, and the heat-kernel case shows why (subject, not class, is the true
variable), but 94% for free is a very good trade.

### The twist: batches were ALREADY mixed-class, and it didn't help

```
same-minute dispatch batches: 27
batches that were entirely ONE class: 0  (0%)

2026-08-08T22:50  bmmmmm   m95A04, m96A05, m01J06, b98A04, m00A02, m93J04
2026-08-08T23:11  bbbmmm   b93J01, b97J04, m97A04, b94A02, m95J01, m02J01
2026-08-08T23:22  mmmmmt   m93J03, t95J06, m97A02, m96J04, m99A06, m99J03
```

Every batch already mixes classes. What they do **not** do is limit *how many
per class*: five `m`s in one batch, three `b`s and three `m`s in another. The
`m96A05`/`m97A04` twins were dispatched 21 minutes apart in consecutive
batches, both in flight together.

**So the effective rule is not "mix the classes" (already true) but "at most ONE
problem per prelim class in flight at a time".** With four classes that caps
concurrency at 4 — which happens to match the HUD's existing
`gated start ceiling 4/hour`.

### The change is small and localised

`bridge_lane.py` selects with
`todo.sort(key=lambda r: (-len(r["identifiers"]), -len(r["hole"])))` then takes
`todo[:want]` — **no class awareness at all**, and sorting by specificity
actively clusters same-subject problems together. Fix: after that sort,
round-robin the pick across `a`/`m`/`t`/`b` and refuse a class already in
flight. ~10 lines in one function, no change to packets or gates.

Residual (the heat-kernel case) is what gate-time duplicate detection is for;
the two fixes are complementary and neither subsumes the other.

## 4h. Extraction pilot — 1/5 clean, but the number is NOT yet meaningful

Handful pilot (Joe authorised, copy-up only, nothing re-proved, nothing written
into the repo — all in `/tmp` on Zone). Five trapped lemmas with `apm_`-free
signatures, lifted verbatim into scratch modules importing only Mathlib.

| problem | result |
|---|---|
| `m98J03` | **compiles clean, exit 0** ✅ |
| `b90A01` | `unknown identifier apm_b90A01_perm_fin_of_card` — **genuine dependency-closure failure** |
| `m98A05`, `b95J02`, `m93A01` | `unexpected end of input; expected 'lemma'` — **my extractor's bug**, not a corpus fact |

**Do not read a 20% yield off this.** Three of the four failures are defects in
my extractor: the block-boundary walk leaves an orphan trailing docstring, and
the `open`-line harvest mangled a `namespace`/`open` line in m98A05. Only
`b90A01` is a real finding — and it is exactly what Fable predicted: extraction
needs the **transitive dependency closure**, not just the declaration.

So the pilot validated the *shape* of the pipeline (a genuinely general lemma
does lift and compile untouched) and correctly surfaced the closure requirement,
but the yield estimate for the 571 needs the closure step implemented first.
That is the next piece of work, and it is mechanical.

## 4g. IS IT STILL HAPPENING? Measured — yes, but only half of it is waste

Joe's question: with an index in place and agents told to use it, are they still
re-deriving? Measured across all 125 duplicated lemmas by introduction date.

```
historic   (before 07-30, module not importable)   23
post-affordance, pre-index (07-30 .. 08-06)        77
POST-INDEX (after 08-06)                           25
```

The 25 post-index cases split further. A duplicate is only *ignoring* the index
if the index was **regenerated between the twin landing and this lemma being
written** — there have been only **9 regenerations ever** (first 08-06 22:32,
last 08-08 21:57), so freshness is not a given:

```
GENUINE MISS (a regeneration happened in between; grep would have found it)  8
STALE INDEX  (no regeneration in between; grep could NOT have found it)      7
```

**Worst genuine misses** — the twin was discoverable and it was re-derived
anyway:

| when | problem | lemma | twin | regens between |
|---|---|---|---|---|
| 08-08 21:34 | `a99J05` | `integral_typewriter1d` (+`_tendsto`) | 08-07 12:01 | **6** |
| 08-08 15:50 | `a01J04` | `geom_iteratedDeriv` | 08-03 13:25 | 3 |
| 08-07 13:32 | `m98J05` | 3 heat-kernel lemmas | 07-26 / 07-31 | 2 |

**But the stale half is structurally unpreventable by any index**, because the
work was *simultaneous*:

- `m96A05` 14:44 → `m97A04` 16:00 — same matrix lemma, **76 min apart**
- `a97J06` 11:34 → `a03J05` 11:38 — same `Integrable`, **4 min apart**
- `t91A05` → `t97J03` retraction lemmas, overnight

`bridge_lane.py` runs `IN_FLIGHT_MAX = 6` concurrent seats. **Six agents working
at once cannot deduplicate against an artifact that only updates after they
finish.** No index refresh rate fixes this; only gate-time detection at merge,
or subject-aware scheduling, can.

**Honesty correction to my own metric:** 2 of the 7 "stale" rows are
`LIB: ConstructionTargets.Surfaces` lemmas whose twin is the source problem
`t97J02` nine minutes earlier — that is **the promotion working**, not
duplication. My clustering counts a promoted copy as a duplicate. Under
additive-only promotion (§4e) *every* promotion creates this pattern by design,
so the metric must exclude LIB rows whose twin is their own source problem, or
it will show duplication rising as the library improves.

So the real current picture is roughly **8 avoidable, ~5 concurrency-caused,
~2 metric artifact** — against 125 total, of which 100 are historic or
pre-index. The rate has fallen a lot, but Joe is right that it is not zero.

**Adoption, separately: of 424 problem files touched since 07-27, only 27 (6%)
import `ConstructionTargets`/`YoungL2`.**

**Revised recommendation** (supersedes §4f's "don't spend on push-don't-pull
yet"): the 8 genuine misses do justify push-don't-pull — but the *cheapest and
highest-yield* fix is Fable's §6 item 3, **gate-time duplicate detection**,
because it is the only one of the three that catches the concurrency half as
well. Regenerating the index on every close is also nearly free and would have
prevented most of the stale-half misses that were not truly concurrent.

## 4f. NATURAL EXPERIMENT — the timeline exonerates the closers, and corrects §4e

Before acting on §4e I checked the git history of the lemniscate triplication.
It resolves the four-layer anatomy more cleanly than any experiment we could
design, and it **falsifies a premise that both my §4d and Fable's §4e reasoning
rested on**.

```
2026-07-28 05:xx  ConstructionTargets/LemniscateComponents.lean created
2026-07-30 02:29  a270a2a "Put ConstructionTargets on the module path"
                  ^ before this, `import ConstructionTargets.X` FAILED with
                    "unknown module prefix" — the lakefile had no lean_lib,
                    no oleans were ever produced (lakefile.toml says so)
2026-07-30 19:24  a00J04 re-derives the 12 lemniscate lemmas
2026-07-30 19:31  a01A08 re-derives them again
2026-08-06 22:32  LEMMA-INDEX.md CREATED (497acbd)   <-- SEVEN DAYS LATER
```

So at the moment of duplication the closers **had affordance and zero
discoverability — there was no index to grep.** They did not ignore the index;
it did not exist. Neither problem has been touched since.

**What this corrects:**
- My §4d line that "the closer prompt already instructs agents to grep the
  index and that was evidently insufficient" — **not established.** The index is
  three days old and was regenerated repeatedly on 08-08. The instruction has
  barely been testable.
- Fable's §4e propensity-decay story ("closers rationally learned the index is
  low-value") — same unfounded premise. Plausible mechanism, no evidence yet.
- The 91%-trapped figure is substantially a **legacy** condition, not proof of
  ongoing negligence. `ConstructionTargets` was literally unreachable until
  07-30; `YoungL2` had the identical defect and a94J04's runner reported a
  proved lemma as unreachable on 07-31 (lakefile comment). **The affordance
  failure is documented as having happened twice and been fixed twice.**

**Consequence for sequencing.** Do not spend on closer retraining or
push-don't-pull machinery yet — the cheap prior question is whether the
*now-existing* index is already working. Measure LIB-citation rate on hops
since 08-06 first. Fable's Phase 1 pilot is still the right experiment, but its
null hypothesis should be "the index now works", not "closers ignore it".

**Standing lesson, same shape as §2's:** three separate analyses (mine, Fable's,
and the packet's) attributed to agent behaviour what was actually missing
infrastructure. Check when the affordance came into existence before concluding
anyone ignored it.

## 4e. Recovery plan (Fable subagent, commissioned by Joe; verified by claude-7)

Joe asked me to brief a Fable subagent on §4d and get a disciplined recovery
plan. It verified the brief rather than accepting it and **corrected me on the
crux**. I re-checked its load-bearing claims against the LIVE system (it read a
local mirror and flagged that itself) — all four hold:

| claim | verified |
|---|---|
| `declaration_set_drift` additions are legal by design | `gates.py:271-285` — *"the asymmetry is the point"*; only `REMOVED`/`CHANGED` void |
| 27 problem files already import `ConstructionTargets` | 27 ✅ |
| def-level duplication is real | `ACOnUnitInterval` defined in **both** `ConstructionTargets/LusinN.lean:68` **and** `problems/a95A02:66` |
| unprefixed local defs are widespread | 254 distinct unprefixed def names across problem files |

### The crux inverts — and my §4d framing was wrong

I said the risk was that delete-and-import would **trip** `frozen_declarations`.
Additions are legal, so that's not it. The real danger is sharper: the freeze
hashes the main theorem's **text**. Delete a local `def` and re-point the
statement at a library def of the same short name and **the text is identical,
the gate passes, and the proposition may have changed** — silent defeat of the
contract in the one form the detector cannot see.

**Therefore: statement-frozen problem files are an IMMUTABLE ARCHIVE. Promotion
is copy-up extraction, never delete-down refactoring. Permanently, not
deferred.** Not even "harmless" import additions to frozen files (buys nothing;
can break builds via short-name ambiguity under `open`).

**The elegant part:** Joe's "the reimplementations are interesting in their own
right" and the safety argument yield *the same policy*. The immutable corpus
**is** the variant store. You never delete to deduplicate — you deduplicate the
*interface* (library + index) and the archive stays as data. There is no
tension to trade off.

### The finding I missed: defs, not lemmas

327 problem files carry ~1,148 local `def`s, ~254 of them unprefixed. Duplicated
defs are the deeper fragmentation vector: **every lemma stated over a duplicated
def is non-interchangeable with its twin.** `ACOnUnitInterval` sitting in both a
promoted ConstructionTarget and a problem file is the pattern. This is more
fundamental than the lemma count in §4d.

### Three layers

- **L0 `ApmLib/`** — Mathlib-facing general lemmas; directory mirrors Mathlib's
  tree; imports Mathlib only. Doubles as the upstreaming queue.
- **L1 `ApmLib/Defs/`** — shared definitions + their API. Hardest and most
  valuable; the only layer needing semantic judgment (unify two defs only if the
  bridge is `rfl`/one-line `Iff`; if the bridge needs real work they are
  *different notions* — keep both).
- **L2 `ConstructionTargets/`** — existing 17 modules, **keep the names**; 27
  frozen files import them by path.

Discipline: L0→Mathlib, L1→L0, L2→L0/L1, problems→anything, **nothing imports
problems**. Enforce with a ~30-line CI import-linter. File cap ~400 lines. **No
`Misc.lean`, ever** — that is the flat bag regrowing under a new name.

### Phase 1 is the clever bit — and it tests the scaffold's weak warrant

Smallest validating step is an **audit**, not a promotion (Phase 0: freeze-
integrity baseline, reproducible twice). Then Phase 1: promote **exactly one**
refuted lemma — `apm_m98a05_hasStrictFDerivAt_of_contDiff` — regenerate the
index so its row reads `LIB:`, and **re-run the closer on the Tier A problem
that reported it missing**. That exercises promotion → index → retrieval →
import → progress end to end, at near-zero risk, and it is a direct experiment
on the scaffold's **N5 "retrieval serves the need" = WEAK** warrant, at the
*affordance* layer specifically.

**Failure is as informative as success:** if the closer still re-derives with
the affordance present, the bottleneck is propensity/framing, not affordance —
and bulk-promoting 571 lemmas would have been the wrong spend.

### My vocabulary test was critiqued, correctly

It is a **ranking prefilter, not a decision procedure**. False negatives: an
`apm_`-prefixed def may itself be general. False positives: signatures that look
Mathlib-only can still be problem-specific via `open` clauses, local notation,
`variable` blocks and local instances. **The same defect inflates my §4d
redundancy count — normalised-signature identity across namespaces is not
semantic identity.**

Correct mechanical test = **compilation with Lean as oracle**: extract the
dependency closure → compile against Mathlib+ApmLib → `#print axioms` clean →
**subsumption back-test** (prove the original trapped statement by
`exact`/`apply` of the promoted lemma). Never waive the back-test; it is the
only guard against semantic narrowing.

### Recurrence, and why "grep the index first" failed

For 91% of rows the index **afforded nothing**, so closers rationally learned it
was low-value — which then suppressed grepping even for the 9% that would pay.
Instructions cannot fix affordance-caused propensity decay. Fixes, in causal
order: fix affordance first; then **push, don't pull** (put top-k similar
library lemmas into the hop briefing rather than relying on the agent choosing
to search); warn at gate time on signature collision with the library; and make
new problems import the library **before** the statement freezes.

The scribe's repurposing is the operational root cause — **promotion was
nobody's job.** Reinstate it with promotion as primary function and *importable
fraction* as its KPI, plus a lint rejecting memories that cite a
`(pid, lemma)` pair instead of an importable name.

### Discrepancy RESOLVED — use Joe's HUD, the mirror `status.json` is stale

Fable and I both flagged the mirror's `status.json` (20 proved / 265 partial /
190 unclassified) as inconsistent. It is simply **stale**. Authoritative figures
are Joe's `M-x stack-hud-1` against a pulled-back apm-lean, 2026-08-09:

```
APM recovery: 491 canonical problems | 371 current sorries open
  informal        462/491  94%
  lean current    438/491  89%
    with sorries  238/438  54% of attempted
    clean         200/438  46% of attempted | 200/491 40% overall
  burn down       291 remaining | +139 clean in 13d (10.69/day)
  gated start ceiling 4/hour
```

So "close to 50%" = **46% of attempted / 40% overall**, and the corpus is 491
problems, not ~370. Do not size anything off `status.json`; ask the HUD.

**Scale note this changes:** at 10.69 clean/day the remaining 291 is ~27 days of
burn-down. Anything that slows the closers has a large denominator — which
argues for the push-don't-pull affordance fixes over anything requiring closer
retraining.

## 4c. What the two-sided rerun changed (claude-7, 2026-08-09)

Re-running all nine verdicts against `LEMMA-INDEX.md` + `ConstructionTargets/`
did **not** overturn any accept or reject, but it materially changed three
dispositions — so the second screen is not ceremonial:

- **`a97J08` — my "formalize JCT from scratch" was too pessimistic.**
  `ConstructionTargets/LemniscateComponents.lean` (importable) already proves
  `frontier_connectedComponentIn_subset_frontier` **in full generality**:
  `frontier (connectedComponentIn U x) ⊆ frontier U` for open `U`. With
  `U = (curveImage γ)ᶜ` (open, since the image is compact) that gives one half
  of the frozen `frontier (curveInterior) = curveImage` conjunct, per component.
  Caveat I checked: `curveInterior` is a *union* of bounded components, so the
  union needs its own small argument. Revised scope: the unbuilt content is the
  **reverse** inclusion plus non-emptiness — the actual separation content —
  not the whole theorem. Still BLOCKED, but smaller, and starting material
  exists.
- **`m93J03` — reject stands, now concrete.** The repo already has the strict
  upgrade packaged: `apm_m98a05_hasStrictFDerivAt_of_contDiff` (in m98A05),
  fully general, `ContDiff 1` + `HasFDerivAt` ⇒ `HasStrictFDerivAt`.
- **`m97A04` — reject stands, and the work is much smaller than I said.**
  `apm_m97a04_unique_mulVec_of_det` is **already proved for this problem**, so
  det ≠ 0 ⇒ unique solution is done. All that remains is two arithmetic bounds
  feeding Mathlib's Gershgorin — and sibling `m96A05` has the identical
  architecture to copy.

**Structural finding worth a ruling, Joe.** The index reports **2098 proved
lemmas but only 182 importable** across 17 modules. The rest are trapped inside
individual problem files. Two of the three refinements above came from *trapped*
general lemmas (`apm_m98a05_hasStrictFDerivAt_of_contDiff` is not
problem-specific in any way). Every closer that needs one either re-derives it
or misses it. A promotion pass — sweep problem files for lemmas whose statements
mention no `apm_` definitions and lift them into `ConstructionTargets/` — looks
like high leverage, and the vocabulary test from §4b is exactly the filter that
identifies candidates mechanically.

## 4b. The closer packet was changed (Joe authorised, 2026-08-09)

The finding in §4a(1) — closers reasoning better in prose than their output
schema lets them express — is a *fixable schema problem*, not a capability one.
Joe approved the change; I made it directly (I held the derivation live, so a
handoff would have been re-explaining the seven rejection kinds).

**File: `apm-driver/bridge_packets.py` on Zone.** `bridge_lane.py` imports
`TIER_A, TIER_B, COMMON` from it as the "single source of packet truth", so one
edit covers both the pilot and the unattended lane. Backup:
`bridge_packets.py.bak-20260809-claude7` (note: Zone's `futon3c` is **not** a
git repo — there is no VCS safety net there, take backups).

- **Tier A** now demands, for `failed-with-statement`: a **search receipt**
  (identifiers grepped, where, and the nearest declaration found with why it
  does not fit); a **vocabulary certification** (no `apm_` definitions in the
  statement — and if you cannot phrase it without them, say the work is *local
  assembly* and name the Mathlib declarations it would use, which is framed as
  a *more* useful answer, not a lesser one); and the **route as ordered steps**
  rather than one lemma the size of the goal.
- **Tier B** now carries the seven rejection shapes explicitly, with the
  framing that *every bridge we have rejected was true*, so truth is not the
  test — plus the "grep this file first" instruction, since the commonest
  rejection is a bridge the file already proves under another name. It also
  states that "strictly weaker" is **not** the test (the `t02A05` correction).

Verified: both templates `.format()` cleanly with no leftover braces, and
`bridge_lane.py` imports them. **Not yet validated in anger** — no dispatch has
run through the new packets, because the lane is drained. First re-dispatch
wave is the test; if reports come back with search receipts and step-lists,
it worked.

## 5. Park-and-ride (this got fixed today)

The "ride" mechanism was never missing. `parked_on.clj:278` `note-completion!`
folds a terminal dep into every record awaiting it and fires resume exactly
once; `http.clj:786` `parked-on-notify!` wires it. It was **flag-gated off** —
the serving JVM lacked `FUTON3C_PARKED_ON`. Joe rebooted it at 18:12 and I
verified **both** paths end-to-end:

- reconcile-on-park: parking on an already-done job returns `status: released`
  immediately rather than stranding;
- hot-path notify: job went `done` at t+5s, park gone by t+10s, ~2690s before
  its deadline.

Consequences you must know:

- **Parks do not survive a JVM restart.** Mine vanished on Joe's reboot. A
  reboot silently drops every outstanding park and the waiter goes quiet rather
  than erroring. Re-arm manually after any restart.
- **`deadline-ms` is ABSOLUTE epoch-ms**: `(now_sec + N) * 1000`. A relative
  duration reads as January 1970 and fires on the first sweep.
- I had been parking on **sentinel strings** (`apm-supervision`) in ~46 of 48
  parks — deps no job can ever complete, so they could only fire on the clock.
  That, not any missing feature, was why the loop felt slow. Park on **real
  job-ids**.

## 6. The auto-bellback breakage — the live issue

Zone's auto-bellbacks are **11 for 11 failed**, every one addressed to
`apm-driver`:

```
failed: "Agent has no invoke handler (no local invoke-fn and no ws bridge)"
```

`apm-driver` is a *caller persona*, not a seat with an invoke handler. The lane
was dispatched with it as the from-identity, so every completion bell routed
somewhere that cannot receive. This is the known corollary: bells to
non-recipient identities are accepted at POST and then fail **asynchronously**,
where the sender never sees it.

**Fix: dispatch the lane with `--from claude-7`.** Then auto-bellback reaches
you across sites and you get the ride with no new machinery — no parks needed
for cross-host work. Joe's read was right: this is a broken edge, not a missing
Agency feature.

> **CORRECTION (claude-7, 2026-08-09 ~19:10 BST).** The diagnosis above is
> right; the prescribed fix is not. I checked Zone's roster directly:
> - `apm-driver` — `invoke-ready? false`, `"no local invoke-fn and no ws
>   bridge"`. Confirmed cause of the 11/11 dead-letters. ✅
> - **`claude-7` is NOT on Zone's roster at all.** Dispatching `--from
>   claude-7` would have swapped a dead-letter for an `agent-not-found` — a
>   different failure, not a fix.
> - Zone carries **site-prefixed mirrors**: `oxf-claude-7`, `oxf-claude-3`,
>   `oxf-apm-driver` are all `invoke-ready? true`, `"local invoke-fn
>   registered"`, idle.
>
> So the working from-identity is **`oxf-apm-driver`** (or `oxf-claude-7`).
> But note what that does *not* buy: `oxf-claude-7`'s invoke-route is
> **`local` to Zone**, so a bellback to it enqueues a turn on a Zone-side seat
> named claude-7 — it does **not** reach this Dionysus session. Cross-site
> bellback to your own session is not available by this route. For your own
> review queue you don't need it (ssh + `bridge_review.py` polling works and is
> what I used); the `--from` fix matters for the *next dispatch wave*
> (Tier B redo, proving lane), not for review.
>
> Nothing is pending re-dispatch right now — the bridge lane reports
> `still running 0`.

I ran the harvest sweep Joe authorised (results appended to
`apm-driver/unharvested-zone-jobs.jsonl` on Zone). Of the 11: 2 had nothing to
harvest (underlying jobs failed, 0 chars); **6 were already tracked** by
`bridge-pilot-jobs.jsonl` and sit in the 91-awaiting queue, so not lost; **3 are
genuinely untracked** (`…3492`, `…3489`, `…3487`), each 1.1–1.9k of real
`ams-scribe-1` output, now saved.

Two caveats on that sweep, both mine:

- **The `problem-id` column in that JSONL is unreliable** — I matched by
  substring, first-match-wins over an unordered set. Job `…3487` is labelled
  `m02J04` but its content is a theorem for **`m93j05`**. Use the `result` text,
  not the pid.
- Zone's `/invoke/jobs` returned only a **20-job window**, while the bridge lane
  alone has 169 tracked ids. This bounds *recent* damage, not total. Widening
  the sweep over all 169 is the natural follow-on.

## 7. Codex handoff protocol (from workspace CLAUDE.md)

Default: substantial coding is belled to Codex, then **independently reviewed by
you** — author ≠ reviewer. Carve-outs: trivial edits; work where you already
hold the live context; tight interactive loops; orchestration/review/architecture.
Fix review findings **yourself**, don't re-bell.

- Always pass `--from <your-id>` — it records the mesh edge and is what routes
  the completion bell back.
- **Park on every dispatch.** A bell without a park is a protocol violation.
- **Keep packets SMALL**: one file, one behaviour, one acceptance test. The cost
  of a big packet is the *missing checkpoint* — a wrong premise ships
  unchallenged. (A tool name I invented reached shipped material that way.)
- Gates a packet must require: clj-kondo on Clojure, `futon4/dev/check-parens.el`
  on Lisp/Clojure, and the relevant tests.

### Reviewing codex-3's in-flight work

`POST /api/alpha/park/complete` — completes an arbitrary dep-id so a park can
ride *any* signal, not just an Agency job. Check:

1. flag-gated on `parked-on-enabled?` (503 when off);
2. accepts **both** keyword and string body keys, as `handle-park` does;
3. reuses the `resume!` wiring from `parked-on-notify!` (`http.clj` ~786) rather
   than inventing a second resume path;
4. route added to **`extra-routes`** (~6719), *not* `make-handler`'s own cond;
5. an unawaited dep-id returns **200 with `released-count 0`**, not an error —
   `note-completion!` is already a cheap no-op there, and completing a dep
   nothing awaits is a normal race.

> **DONE (claude-7, 2026-08-09 ~19:00 BST). Reviewed, activated, verified.**
> Commit `9657075d`, +29 lines, `http.clj` only. All five checks above pass:
> flag-gate 503 ✅; both key forms ✅; reuses `parked-resume!` with the
> *identical* opts map as `parked-on-notify!` (`http.clj:786`) ✅; route landed
> inside `extra-routes` (defn at 6745, route at 6820 — not `make-handler`,
> which starts at 6837) ✅; unawaited dep is a 200 no-op because
> `note-completion!` short-circuits on `(contains? (:index @!parked) dep-id)`
> (`parked_on.clj:285-286`) ✅.
>
> **Review finding (fixed by me, not re-belled):** codex-3's cited validation —
> "13 tests, 43 assertions" — is the *pre-existing* `parked_on_test.clj` suite.
> It never touches the new handler, and the commit adds no test. So the report's
> evidence did not cover the new code. I supplied the missing acceptance
> evidence live instead, after activating via
> `load-string` (NOT `load-file`) of just the two defns through Drawbridge:
>
> | | expected | got |
> |---|---|---|
> | unknown dep | 200, count 0 | `{"ok":true,"released":[],"released-count":0}` ✅ |
> | no dep-id | 400 | `dep-id-required` ✅ |
> | bad JSON | 400 | `invalid-json` ✅ |
> | park→complete | count 1, that park | released `park-de35ce9b…` exactly ✅ |
> | complete again | count 0 | idempotent ✅ |
> | `/parked` after | empty | empty ✅ |
>
> Cosmetic, not worth a fix: `parse-json-map` calls
> `(json/parse-string body true)` — it **keywordizes**, so the `(get payload
> "dep-id")` string fallback is dead code. Harmless and consistent with
> `handle-park`, which claude-3 told codex-3 to mirror.
>
> **For Joe — design note, not a defect:** the route is unauthenticated (no
> `/api/alpha` route on :7070 has auth), and by design it completes an
> *arbitrary* dep-id. So any local caller can POST a real job-id and release a
> park *before* that job finishes. That is inherent to the feature as specified,
> and consistent with the rest of the surface — flagging it, not blocking it.
>
> **Parks do not survive a JVM restart, and neither does this activation** —
> it is a live `load-string`, not a reload. After any restart, re-eval the two
> defns (extract lines and `load-string` them) or the route 404s again.

**DANGER — do not `load-file` `http.clj` into the live JVM.** It is the serving
namespace; an in-place reload under traffic has deadlocked this JVM before, it
hangs until client timeout, and a timed-out client does *not* interrupt the
server-side eval. The handler is a closure captured at server start, which is
why `extra-routes` exists as the reload-safe extension point. Activate by
evaluating just the two `defn` forms via Drawbridge, or ask Joe for a clean
restart (he offered one today).

## 8. Outstanding

- 91 bridge reviews (batches of ~6).
- Review `codex-3`'s `park/complete`, then use it to build a real `/loop` on
  your own queue criterion — Joe's point: for your *own* queue you never needed
  Zone parks at all.
- Re-dispatch the lane `--from claude-7` so bellbacks stop dying.
- Widen the unharvested sweep across all 169 job-ids.
- Tier B redo for the ~17 rejected problems, with anti-restatement wording.
- Prove the ~34 sound bridges (a proving lane).
- 79 unharvested problems (69 prose + 10 code-excerpt).
- Joe's 🕳 (U+1F573) stigmergy convention for loosely-typed holes — offered,
  ~20 min, **not yet authorised**. Idea: agents write 🕳 in-file instead of
  free-text hole descriptions, and everyone greps for the symbol.
- Open question for Joe: frozen contracts don't cover `def` bodies.

## 9. Working with Joe

- Quote **local time**, not UTC.
- Say what you **checked**, not just what you concluded — the review is meant to
  be auditable.
- A negative result is evidence about your *setup* first. Three of my confident
  mechanism diagnoses this campaign were wrong before the true cause (two
  Agencies with overlapping names) turned up.
- Don't kill processes by name on the workstation; kill only PIDs you launched.
- Keep Bash timeouts short (default 120s); background anything longer.
- He is thinking *with* you, not just delegating. Flag uncertainty as
  uncertainty and he will engage with it — several of the better calls this
  session were his corrections of my framing.
