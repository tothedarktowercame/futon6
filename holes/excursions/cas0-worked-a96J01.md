# CAS-0 worked example #2 — apm-a96J01 (uniformly-convergent series, divergent sup-norms)

*Second empirical data point for CAS-0. Chosen as a **construction/existence** proof —
a different shape from #1 (a93J05, reduce-to-known-theorem) — to find where the 36
`math-informal` patterns run out. It did: this proof yielded the **first new pattern**.
CPU/semi-manual. Source: `futon3c/data/apm-informal-proofs/apm-a96J01.md`. claude-1, 2026-06-17.*

## The proof, decomposed

**Q:** ∃ nonneg continuous `(fₙ)` on `[0,1]` with `Σfₙ` uniformly convergent but
`Σ‖fₙ‖_∞ = ∞`? **A: yes**, by construction:

1. Partition `[0,1]` into consecutive intervals `Iₙ`, `|Iₙ| = 1/(n(n+1))` (sum to 1 by
   telescoping `1/n − 1/(n+1)`).
2. `fₙ` := a continuous tent on `Iₙ`, peak `1/n`, support **exactly** `Iₙ`, zero elsewhere.
3. `‖fₙ‖_∞ = 1/n` ⇒ `Σ‖fₙ‖_∞ = Σ1/n = ∞` (harmonic).
4. **Disjoint supports** ⇒ at each `x`, at most one `fₙ(x) ≠ 0` ⇒ tail
   `sup_x Σ_{n>N} fₙ(x) ≤ sup_{n>N}‖fₙ‖_∞ = 1/(N+1) → 0` ⇒ uniform convergence.

## Step → pattern (the 36 cover 4/5; **one genuinely new**)

| step | pattern | status |
|---|---|---|
| (whole) prove ∃ by exhibiting | **construct-an-explicit-witness** | existing ✓ |
| 1 build the partition `Iₙ` | **construct-auxiliary-object** | existing ✓ |
| 1+2 **make the supports disjoint** so the bumps don't interfere | **separate-into-independent-pieces** | **NEW — written** |
| 3 `Σ1/n` diverges | **reduce-to-known-result** (harmonic series, `:cites`) | existing ✓ |
| 4 tail bound `≤ 1/(N+1)` | **estimate-by-bounding** | existing ✓ |

**The gap, and why it's real.** The load-bearing idea of this proof is *disjointness*: the
bumps are engineered not to overlap, which makes (a) the sup-norm sum exactly the harmonic
series and (b) uniform convergence trivial (at most one term nonzero per point, so the tail
sup is just the sup of the pieces). I checked the three nearest existing patterns against
this move and none captures it:
- `exploit-symmetry` — group actions / invariance / WLOG. Not this.
- `quotient-by-irrelevance` — *collapses* objects that agree. Opposite direction.
- `local-to-global` — *patches overlapping* pieces (open covers, partitions of unity, gluing).
  This proof is its **dual**: design the overlaps *away* so no patching is needed.

So I wrote **`math-informal/separate-into-independent-pieces`** (`[✂️/分]`, registered in
`resources/sigils/patterns-index.tsv`): *"engineer the pieces to have disjoint/independent
support so cross-terms vanish and a global aggregate property collapses to a per-piece one."*
Its `HOWEVER`: independence must be *earned* — verify supports really are disjoint and that
disjointness delivers the aggregate property. Recurs far beyond analysis (orthogonal vectors,
independent random variables, disjoint cycles, almost-disjoint families).

## Wiring + sorry (same mechanism as #1, now with the new pattern in the chain)

**Wiring:**
```
construct-aux(partition Iₙ, |Iₙ|=1/(n(n+1)))  ──▷  Σ|Iₙ|=1 (telescoping)
            │
separate-into-independent-pieces(supports Iₙ disjoint)
            ├──▷  ‖fₙ‖=1/n ; reduce-to-known(harmonic)  ──▷  Σ‖fₙ‖_∞ = ∞   ┐
            └──▷  ≤1 nonzero per x ; estimate-by-bounding(tail ≤ 1/(N+1))  ─┴─▷  Σfₙ unif. conv.
                                                                                   ▽
                                            construct-an-explicit-witness  ──▷  ∃ such (fₙ) ∎
```

**Sorry** (= matched patterns' undischarged `HOWEVER`s — mechanism reconfirmed):
| sorry | from pattern | obligation |
|---|---|---|
| S1 `Σ 1/(n(n+1)) = 1` | construct-auxiliary-object | the telescoping sum (asserted, ✓ easy) |
| S2 supports `Iₙ` pairwise disjoint ⇒ ≤1 term nonzero per `x` | **separate-into-independent-pieces**'s HOWEVER ("disjointness must be earned + must deliver the property") | the consecutive `Iₙ` are disjoint by construction; "exactly `Iₙ`" support |
| S3 a continuous tent of given peak/support exists | construct-an-explicit-witness | the explicit tent function (asserted) |

## What this checkpoint adds (vs #1)

- **The mechanism generalizes across proof shapes.** #1 (reduce-to-known) and #2
  (construction) are different shapes, yet *both* induce wiring (patterns' conclusions) +
  sorry (patterns' HOWEVERs) the same way. The mechanism is shape-independent — strong
  evidence it's the right spine for CAS-SEL.
- **The pool genuinely grows from worked proofs (Q4).** #1 added 0 patterns, #2 added 1.
  This is exactly the "pattern-seeding loop" the prior-art sketch predicted (repeated
  held-sorry shapes → new library patterns). The seeding is **demand-driven**, not a bulk
  mining pass — confirming the empirical/by-not-fiat call: we wrote the *one* pattern this
  proof needed, not a speculative taxonomy.
- **`construct-*` family co-fires.** Both `construct-an-explicit-witness` (the existence
  shape) and `construct-auxiliary-object` (the partition) fire — like `reduce-to-known`'s
  `:cites` slot in #1, suggesting `select` matches *families* with parameters, not atomic labels.

## Next
- Proof #3: a **different** shape again — an **induction** or a **proof-by-contradiction**
  (`argue-by-contradiction` / `induction-and-well-ordering` exist; test whether they
  cover a real APM induction, or whether the induction *schema* needs a finer hole).
- After #3–#4: the recurring step→pattern→(wiring,sorry) mechanism + the 1–2 new patterns
  are the concrete input to **CAS-SEL-1** (the spec spike runs *on* these worked examples).
