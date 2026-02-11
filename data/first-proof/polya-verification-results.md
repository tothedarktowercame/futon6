# Polya Verification Results

## Problem 8: Maslov Index (verify-p8-maslov.py)

### Results

| Configuration | Maslov index distribution |
|---|---|
| Random triples (3-face) | {-2: 18%, 0: 62%, 2: 20%} |
| Paired quadruples (4-face, ideal) | {-2: 11%, 0: 72%, 2: 17%} |
| Unpaired quadruples (4 random) | {-4: 1%, -2: 28%, 0: 41%, 2: 29%, 4: 1%} |

### What this tells us

**Partially confirms, partially challenges the solution.**

1. The specific hand-constructed quadruple (L1, L2, L3, L4 as two
   transverse pairs) DOES give Maslov index 0. ✓

2. But random paired quadruples (perturbations of two sheets) give
   nonzero Maslov index 28% of the time. This means the pairing
   alone is insufficient — the vertex needs additional geometric
   constraints (local flatness / topological submanifold condition)
   to force Maslov index 0.

3. The 3-face case is NOT always obstructed: 62% of random triples
   have Maslov index 0. So "3 faces → nonzero Maslov → obstructed"
   is too strong. The correct statement is: 3-face vertices CAN have
   nonzero Maslov (and generically do for certain configurations),
   while 4-face vertices with the right pairing structure always
   have zero Maslov.

### Correction needed for Problem 8

The solution should be more careful about WHY the 4-face condition
forces Maslov index 0. It's not just "opposite faces pair into sheets"
— it's that the topological submanifold condition at the vertex
constrains the Lagrangian planes to have specific relative positions
in Lambda(2) = U(2)/O(2). This extra constraint is what forces the
cancellation.

**Confidence change: stays at Medium.** The mechanism is right
(paired sheets → Lagrangian surgery) but the Maslov argument needs
tightening.

---

## Problem 6: Epsilon-Light Subsets (verify-p6-epsilon-light.py)

### Results

| Graph | Minimum c_eff across epsilon values |
|---|---|
| K_6, K_8, K_10 | 0.667 (at small n rounding) |
| C_8, C_10, C_12 | 0.667 (at eps=0.75) |
| P_8, P_10, P_12 | 0.667 (at eps=0.75) |
| Star_8, Star_10 | 1.167+ (always large) |
| Barbell_4, Barbell_5 | 0.800 |

**Overall minimum: c_eff = 0.667 at (C_8, eps=0.75)**

### What this tells us

**Strongly confirms the solution.**

1. c >= 1/2 holds across ALL tested graphs and epsilon values. ✓

2. The minimum c_eff = 0.667 > 0.5, so there's headroom.

3. Key insight from cycle/path graphs: maximum epsilon-light subset
   is ALWAYS n/2 (an independent set), regardless of epsilon! Taking
   every other vertex gives L_S = 0 (no internal edges), which is
   trivially epsilon-light for any epsilon > 0. So c_eff >= 1/(2*eps)
   for these graphs, which diverges as eps -> 0.

4. K_n is the TIGHT case: max |S| = floor(eps * n), giving c = 1.
   This was already in the solution and is confirmed numerically.

5. Barbell graphs (two cliques joined by a bridge) give c_eff = 0.8,
   which is interesting — the bridge doesn't hurt much because you
   can take vertices from both cliques.

**Confidence change: Medium → Medium-high.** The numerical evidence
is clean and the independent-set lower bound (c >= 1/2 for any graph
with maximum degree d, take an independent set of size >= n/(d+1))
provides a simple proof that c > 0.

### New insight: independent set argument

For ANY graph G on n vertices with max degree d:
- An independent set I satisfies L_I = 0 (no internal edges), so I
  is epsilon-light for ALL epsilon > 0
- By greedy algorithm, |I| >= n / (d+1)
- But this gives c = 1/((d+1)*epsilon), not a universal c

For the universal bound: use the random independent set. Include each
vertex independently with probability p, then remove vertices with
neighbors in S. Expected surviving set size ~ n*p*(1-p)^d >= n*p*e^{-pd}
for small p. Setting p = 1/d gives expected size ~ n/(d*e). But d can
be up to n-1...

Actually, the solution's probabilistic argument (sample with p = epsilon,
use L_S <= epsilon * L in expectation) is more sophisticated. The
numerical verification just confirms it works.
