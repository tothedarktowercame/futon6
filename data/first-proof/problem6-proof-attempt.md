# Problem 6: Proof Attempt — Leverage-Based Vertex Selection

Date: 2026-02-12
Author: Claude (building on Codex method wiring library)

## Key Structural Insight (New)

**The star domination approach from the existing solution is fundamentally too
lossy.** It replaces edge indicators Z_u Z_v with (Z_u + Z_v)/2, converting a
quadratic dependence on sampling probability (expectation p²) into a linear
one (expectation p). This destroys the headroom needed for concentration.

Concrete example: For the star graph, star domination gives ||A_v|| = 1/2 for
ALL vertices, making matrix concentration impossible — yet the actual problem
is trivially solved (take all leaves, L_{G[S]} = 0).

**The correct approach works directly with L_{G[S]} = Σ_e Z_u Z_v X_e and
exploits a necessary condition from leverage scores.**

## The Leverage Threshold Lemma

**Lemma (Necessary condition).** Let M = L^{+/2} L_{G[S]} L^{+/2} and
X_e = w_e L^{+/2} b_e b_e^T L^{+/2} (rank-1 PSD, with ||X_e|| = τ_e).

If edge e has both endpoints in S, then M ≥ X_e, so ||M|| ≥ τ_e.

Therefore: **if τ_e > ε and both endpoints of e are in S, then L_{G[S]} ≤ εL
is violated.**

*Proof.* M = Σ_{f⊆S} X_f ≥ X_e (since all terms are PSD and edge e is one of
them). So ||M|| ≥ ||X_e|| = τ_e > ε. □

**Consequence:** S must be an independent set in G_H = (V, H) where
H = {e ∈ E : τ_e > ε} is the subgraph of "ε-heavy" edges.

## The Turán Bound on Heavy Edges

**Lemma.** |H| ≤ (n − k)/ε where k is the number of connected components.

*Proof.* Each heavy edge has τ_e > ε. Since Σ_e τ_e = n − k (standard
effective resistance identity), the count follows by averaging. □

**Lemma (Independent set in G_H).** α(G_H) ≥ εn/(2 + ε).

*Proof.* By Turán's theorem: for a graph on n vertices with m edges,
α ≥ n²/(2m + n). Applied to G_H with m ≤ (n−k)/ε ≤ n/ε:

    α(G_H) ≥ n² / (2n/ε + n) = n / (2/ε + 1) = εn / (2 + ε).

For ε ≤ 1: α(G_H) ≥ εn/3. □

So there exists an independent set I in the heavy subgraph with |I| ≥ εn/3.
All edges internal to I are ε-light (τ_f ≤ ε).

## Case Analysis

### Case 1: All edges within I are absent (I is independent in G itself)

Then L_{G[I]} = 0 ≤ εL trivially. Take S = I, |S| ≥ εn/3.

This covers: trees, low-degree graphs (max degree < 2/ε), bipartite graphs
where I is one side, etc.

### Case 2: There exist light edges within I

All internal edges f have ||X_f|| = τ_f ≤ ε. We need:

    ||Σ_{f⊆I} X_f|| ≤ ε.

Let α_I = ||Σ_{f: both endpoints in I} X_f|| ≤ 1 (since Σ_{all e} X_e = I).

**Sub-case 2a: α_I ≤ ε.** Then I itself is ε-light. Take S = I. Done with
c_0 = 1/3.

**Sub-case 2b: α_I > ε.** The independent set I has too much internal spectral
mass. Need to thin I further.

## The Thinning Problem (Where the Gap Remains)

In sub-case 2b, we need S ⊆ I with |S| ≥ c_0 ε n and ||Σ_{f⊆S} X_f|| ≤ ε.

**Random subsampling at rate q:** Include each vertex of I with probability q.

Note: since operator norm is convex, Jensen gives E[||M_S||] ≥ ||E[M_S]|| =
q² α_I, which is a LOWER bound, not useful for our upper bound goal.

Instead, use Markov on the trace: E[tr(M_S)] = q² T_I where T_I = Σ_{f⊆I} τ_f.
By Markov: Pr[tr(M_S) > ε] ≤ q² T_I / ε.
Since ||M_S|| ≤ tr(M_S): Pr[||M_S|| > ε] ≤ q² T_I / ε.
For fixed I and q, if q² T_I ≤ ε/4 then Pr[||M_S|| ≤ ε] ≥ 3/4.

To also enforce size, use Chernoff:

    E[|S|] = q|I|,   Pr[|S| < q|I|/2] ≤ exp(-q|I|/8).

So with positive probability both events hold provided q² T_I <= ε/4 and q|I|
is at least a moderate constant.

What this yields UNIFORMLY from only T_I ≤ n and |I| ≥ εn/3:
- worst-case T_I = n forces q = O(√(ε/n)),
- then q|I| = O(ε^{3/2}√n), i.e., sublinear in n.

Hence the trace/Markov route, by itself and without extra structure on T_I,
cannot reach a universal linear-size guarantee c_0*εn.

The trace/Markov bound is too crude. A tighter approach: for random q-sampling,
the expected number of internal edges is ≈ q² m_I, each with τ_f ≤ ε. So
E[tr(M_S)] = q² T_I. And tr(M_S)/rank(M_S) ≤ ||M_S|| ≤ tr(M_S). For K_n,
||M_S|| = |S|/n ≪ tr(M_S) = |S|²/n — the gap is a factor of |S| from
eigenvalue spreading. The subsampling argument cannot exploit this spreading.

**Operational bound (heuristic, not rigorous):** If we ASSUME ||M_S|| behaves
like E[M_S] (i.e., spreading holds), then ||M_S|| ≈ q² α_I, and setting
q = √(ε/α_I) ≤ √ε gives |S| ≈ √ε · |I| ≈ ε^{3/2}n, i.e., c_0 = O(√ε).
This heuristic bound is consistent with the trace/Markov bound but NOT
rigorously derived — the Jensen inequality goes the wrong way for an upper
bound on E[||M||].

**In either case, c_0 is not universal.** The subsampling approach is
inherently limited because:
- The rigorous trace/Markov minimax scaling is sublinear (O(ε^{3/2}√n))
- The heuristic spreading bound gives c_0 = O(√ε)
- Both reflect the quadratic-vs-linear scaling mismatch

## Why the Complete Graph Escapes This Bound

For K_n with S of size s: L_{G[S]} = (s/n) L (exactly, by symmetry). So the
condition becomes s/n ≤ ε, giving s = εn, c_0 = 1.

The key: K_n has the special property that L_{G[S]} ∝ L for ANY S. This
proportionality makes the operator norm equal to the "average," with no
spectral concentration penalty.

General graphs lack this proportionality. The operator norm can exceed the
average eigenvalue by a factor related to the spectral structure.

## Strategies for Closing the Gap

### Strategy A: MSS Interlacing Families (D5 in Codex library)

The Marcus-Spielman-Srivastava theorem on interlacing families could give
existence of a good assignment with better bounds than random subsampling.

**Heuristic extrapolation (NOT a theorem):** If an MSS-type bound existed for
quadratic chaos — i.e., for independent Bernoulli ε_v and PSD atoms X_f, an
existence result of the form:

    λ_max(Σ ε_u ε_v X_f) ≤ (√μ + √R)²

where μ = ||E[Σ ε_u ε_v X_f]|| and R = max ||X_f|| — then in our setting
(R = ε, μ = q²α_I): bound = (√(q²α_I) + √ε)².
For this ≤ ε: q√α_I + √ε ≤ √ε, requiring q = 0. Not useful directly.

**Caveat:** Classical MSS/interlacing families apply to LINEAR sums Σ ε_i A_i,
not quadratic products Σ ε_u ε_v X_f. No published result directly gives the
above bound for the quadratic case. The extrapolation is used here only to
illustrate why even an optimistic MSS-type bound would not immediately help.

**The issue:** MSS gives bounds in terms of (√μ + √R)², which is always ≥ R.
Since R = max τ_f can be up to ε (for light edges), the bound is ≥ ε. No room.

MSS might help if applied differently — e.g., to a REFINED decomposition that
splits each X_f into smaller atoms. This is the D5+D7 hybrid suggested by
Codex.

### Strategy B: Direct Spectral Argument for Dense Light Subgraphs

When α_I > ε (many light edges within I), the graph G restricted to I is
"spectrally dense." For such graphs, one might show that L_{G[I]} is
approximately proportional to L (restricted to I), analogous to K_n.

If L_{G[I]} ≈ α_I · L (in some approximate sense), then taking S ⊆ I of size
εn/α_I gives L_{G[S]} ≈ (εn/α_I)² · α_I/n² · L... this gets complicated.

### Strategy C: Iterative Peeling

Remove vertices from I one at a time, always removing the vertex that
contributes most to the top eigenvalue of M = Σ_{f⊆S} X_f. Track the spectral
decrease using a potential function.

The challenge: each removal decreases the top eigenvalue by at most ||A_v^I||
(the vertex's spectral contribution), and we need to decrease from α_I to ε.
This requires removing Θ(α_I/max ||A_v^I||) vertices, which could be Θ(n).

### Strategy D: Solve on Spectral Coordinates Directly

Work in the eigenbasis of L. The condition L_{G[S]} ≤ εL becomes:

    u_k^T L_{G[S]} u_k ≤ ε λ_k for each eigenvector u_k of L

plus cross-term conditions. Each diagonal condition is a SCALAR inequality
that can be analyzed with scalar concentration. The union bound over d ≈ n
eigenvectors introduces a factor of n, but for "generic" eigenvectors the
per-direction probability is O(1/n), so the union bound is O(1).

The bottleneck: "spiky" eigenvectors (those concentrated on few vertices) that
align with specific edges.

## What Is Proved vs. What Remains Open

### Proved unconditionally:

1. **Leverage threshold necessary condition:** Any ε-light set must be
   independent in the heavy subgraph G_H = {e : τ_e > ε}.

2. **Turán lower bound:** There exists I independent in G_H with |I| ≥ εn/3.

3. **Case 1 resolution:** When I is also independent in G (low-degree graphs,
   trees, etc.), c_0 = 1/3 works with L_{G[I]} = 0.

4. **Case 2a resolution:** When the light edges within I have bounded spectral
   contribution (||Σ X_f^I|| ≤ ε), c_0 = 1/3 works with S = I.

5. **Star domination critique:** The existing writeup's approach via
   A_v = (1/2)Σ_{u~v} X_{uv} converts the quadratic p² dependence to linear
   p, severely degrading the concentration headroom. This is demonstrated
   (not formally proved impossible) by the star graph example, where star
   domination gives ||A_v|| = 1/2 for all vertices, making standard matrix
   concentration unable to achieve ε-lightness despite the problem being
   trivially solvable. Whether a cleverer use of star domination could
   still yield universal c_0 is not ruled out, but appears unlikely given
   the structural mismatch in scaling.

### Remains open:

6. **Case 2b:** When ||Σ X_f^I|| > ε (light edges have large total spectral
   contribution within I). Current best: c_0 = O(√ε), not universal.

7. **Closing the gap** requires either:
   - A non-concentration argument (structural/algebraic)
   - A refined application of MSS/KS to the vertex-indicator chaos
   - An argument that Case 2b actually implies additional structure
     that can be exploited

## Relationship to Codex Method Library

The leverage threshold lemma and Turán argument are new and not present in
any of D1–D10. They provide a STRUCTURAL reduction that the Codex library's
methods can then be applied to:

- **D5 (interlacing families):** Most relevant for Case 2b — proving existence
  of a good vertex subset among light-edge vertices. Needs adaptation to
  handle the quadratic chaos Σ Z_u Z_v X_f.

- **D7 (matrix Chernoff for strongly Rayleigh):** Could help if the vertex
  indicators for the independent set I can be modeled as a strongly Rayleigh
  distribution (possible via spanning tree / DPP sampling on G[I]).

- **D4 (restricted invertibility):** The barrier-method structure resembles
  our iterative peeling (Strategy C). The quantitative subset size from D4
  might adapt to give universal c_0.

- **D9 (Schur complement):** The chain L_{G[S]} ≤ L[S,S] and Schur(L,S) ≤ L[S,S]
  (Schur complement subtracts a PSD term from L[S,S]) might give a cleaner
  spectral bound that avoids the concentration issues.

## Strategy 3 Analysis: Decoupling + Conditional MSS (Formalized)

### Setup

Start from I' (pruned independent set in G_H):
- |I'| ≥ εn/6, max leverage degree ℓ_v^{I'} ≤ 12/ε within I'
- (Pruning: remove vertices with ℓ_v > 12/ε; by Markov on average ℓ ≈ 2,
  this removes ≤ εn/6 vertices from I, leaving |I'| ≥ εn/6)

### Bipartition + Conditioning

1. Randomly partition I' into A, B (each vertex → A or B with prob 1/2)
2. Subsample S_A ⊆ A at rate p_A. For each v ∈ B, define:
       C_v = Σ_{u ∈ S_A, u~v} X_{uv}
3. M_{cross} = Σ_{v∈B} Z_v^{(B)} C_v — LINEAR in independent Z^{(B)} variables

### MSS Application

R = max_v ||C_v|| ≤ p_A · ℓ_max ≤ p_A · 12/ε.
Setting p_A = ε²/12: R ≤ ε. ✓

μ = ||Σ_{v∈B} p_B C_v|| ≤ p_B · p_A · ||Σ_{f in I'} X_f|| ≤ p_B · p_A ≤ p_B ε²/12.
With p_B = 1: μ ≤ ε²/12.

MSS bound: ||M_{cross}|| ≤ (√μ + √R)² = (ε/√12 + √ε)² = ε(1/√12 + 1)² ≈ 1.8ε.
Cross-edge spectral norm: O(ε). ✓

### Why This Isn't Enough

**M = M_{AA} + M_{cross} + M_{BB}.** Only M_{cross} is controlled.

M_{AA} and M_{BB} have the SAME quadratic structure as the original problem
on smaller vertex sets (|A|, |B| ≈ |I'|/2). Recursive application reduces
the spectral norm by factor ~1/4 per level but also reduces vertex count
by factor ~1/2. After k levels:

    spectral norm ≈ (1/4)^k · α_I
    vertex count ≈ (1/2)^k · |I'|

For spectral norm ≤ ε: k ≥ log_4(α_I/ε) ≈ log_4(1/ε)
Vertex count: (1/2)^k |I'| ≈ √ε · |I'|

Result: c_0 = O(√ε). The recursion reproduces the subsampling bound exactly.

### Diagnosis

The quadratic-vs-linear scaling mismatch is fundamental:
- Spectral contribution ∝ (sampling rate)² [quadratic]
- Set size ∝ (sampling rate)¹ [linear]

Every technique that reduces to "subsample and control" hits this wall.
A proof with universal c_0 must exploit structure BEYOND concentration —
either algebraic (spectral spreading is automatic for our X_f) or
combinatorial (Case 2b can't arise given the leverage threshold).

### Comparative Table of Approaches

| Technique                    | c_0 bound  | Bottleneck                          |
|------------------------------|------------|-------------------------------------|
| Trace / Markov               | sublinear O(ε^{3/2}√n) (worst-case T_I=n) | ||M|| ≤ tr(M), loses dim factor n |
| Star domination + Bernstein  | O(ε/log n) | Converts p² → p, destroys headroom |
| Decoupling + MSS (1-shot)    | O(ε²)     | p_A = ε²/12 to shrink atoms; no recursion |
| Decoupling + MSS (recursive) | O(√ε)     | 4^k spectral vs 2^k vertex scaling  |
| Greedy / chromatic number    | O(ε)      | IS in d≈1/ε graph has size ε·|I|   |
| Rank spreading heuristic     | O(ε²)     | tr/rank is lower bound, not upper   |

Note: "1-shot" takes S = S_A only (avoids within-group terms but pays p_A = ε²).
"Recursive" applies bipartition repeatedly, recovering the √ε subsampling bound.

### What Would Close It

1. **Quadratic MSS theorem**: interlacing families for degree-2 polynomial
   chaos. No existing result; MSS machinery might adapt.

2. **Structural impossibility for Case 2b**: show that I independent in G_H
   with all τ_f ≤ ε automatically implies ||Σ X_f^I|| ≤ ε. Would make
   leverage threshold + Turán a complete proof.

3. **Spectral graph theory bypass**: bound ||L^{+/2} L_{G[S]} L^{+/2}||
   directly from graph-theoretic properties of S (expansion, conductance)
   without matrix concentration.

## Case 2b Blueprint: Greedy Barrier + Core Regularization

This section records what can be proved now, and the exact next lemmas needed.

### Step 0 (proved): bounded-leverage core extraction

Let I be the Turan independent set in G_H, m = |I|, and

    T_I := sum_{f subseteq I} tau_f,      ell_v := sum_{u in I, u~v} tau_{uv}.

Then

    (1/m) sum_{v in I} ell_v = 2 T_I / m.

By Markov, at least m/2 vertices satisfy

    ell_v <= 4 T_I / m.

So there exists I0 subseteq I with |I0| >= m/2 and

    ell_v <= 4 T_I / m   for all v in I0.

Using m >= epsilon*n/3 and T_I <= n gives the unconditional coarse bound
ell_v <= 12/epsilon on I0.

This is a deterministic regularization step; no probability used.

### Step 1 (proved): trace-only greedy has a sublinear ceiling

Suppose a greedy process builds S_t subseteq I0 one vertex at a time and, at
step t, picks v minimizing the trace increment

    Delta_t(v) := tr(C_v(S_t)),   C_v(S_t) := sum_{u in S_t, u~v} X_{uv}.

Let D := max_{v in I0} ell_v. Then for R_t = I0 \\ S_t (|R_t| = m0 - t):

    average_{v in R_t} Delta_t(v)
    = (1/|R_t|) sum_{u in S_t, v in R_t} tau_{uv}
    <= (1/|R_t|) sum_{u in S_t} ell_u
    <= t D / (m0 - t),

so the chosen vertex satisfies Delta_t <= t D/(m0-t). Summing:

    tr(M_{S_t}) <= sum_{j=0}^{t-1} j D/(m0-j)
                <= D * t^2/(m0-t).

Hence this trace-only certificate class (q-subsampling + Markov/trace, or
greedy rules whose control is only through tr(M) with the above averaging
bound) cannot yield linear-size c0*epsilon*n in worst case from current
hypotheses; it gives sublinear scaling.

This isolates what must be improved: certify lambda_max directly, not via trace.

### Step 2 (open target): lambda_max-aware barrier greedy

Define (for fixed barrier u = epsilon and current M_t < u I):

    B_t := (uI - M_t)^(-1),
    Y_t(v) := B_t^(1/2) C_v(S_t) B_t^(1/2),
    score(v) := ||Y_t(v)||,
    drift(v) := tr(B_t C_v(S_t) B_t).

Woodbury/Neumann gives the one-step bound

    Phi_{t+1} - Phi_t <= drift(v)/(1 - score(v)),
    Phi_t := tr(B_t),

whenever score(v) < 1.

So a viable route is:
1. prove existence of v with score(v) <= theta < 1 and small drift(v),
2. add such v repeatedly for Omega(m0) steps,
3. maintain M_t <= epsilon I throughout.

The missing theorem-level ingredient is an operator-valued averaging lemma on
{C_v(S_t)} over remaining vertices that is stronger than trace averaging.

### Minimal lemma that would close Case 2b

It is enough to prove the following for some universal constants
gamma, theta, K > 0:

For every t <= gamma m0 and every S_t subseteq I0 with |S_t| = t and
M_t <= epsilon I, there exists v in I0 \\ S_t such that

    score(v) <= theta   and   drift(v) <= K/(m0 - t).

Then the barrier potential telescopes and yields a set S of size
|S| >= gamma m0 >= (gamma/6) * epsilon * n with M_S <= epsilon I.

That gives universal c0 = gamma/6.

### Empirical signal (quick probe; not part of proof)

A small ad hoc numerical test (n <= 40, synthetic families, 89 Case-2b
instances) using the greedy rule

    choose v minimizing lambda_max(M_t + C_v(S_t))

produced |S|/(epsilon*n) ratios:
- min 0.778
- median ~1.0 to 1.28 (family-dependent)

No small-instance counterexample appeared. This supports focusing on
lambda_max-aware (not trace-only) greedy.

## Case 2b Formalization: Sublemmas and Reduction

This section turns the "one missing lemma" statement into explicit components.

### Setup

Fix Case 2b data:
- I independent in G_H, with |I| >= epsilon*n/3
- all internal edges satisfy tau_f <= epsilon
- alpha_I := ||sum_{f subseteq I} X_f|| > epsilon.

Let I0 subseteq I be the regularized core from Step 0, with
m0 := |I0| >= |I|/2 and leverage-degree bound

    ell_v := sum_{u in I0, u~v} tau_{uv} <= D,   D := 4 T_I/|I| <= 12/epsilon.

Greedy state at time t:
- S_t subseteq I0, |S_t| = t
- R_t := I0 \\ S_t, r_t := |R_t| = m0 - t
- M_t := sum_{f subseteq S_t} X_f
- for v in R_t:
      C_t(v) := sum_{u in S_t, u~v} X_{uv}
  so that M_{t+1} = M_t + C_t(v_t) if v_t is chosen.

Barrier objects (u = epsilon):

    B_t := (u I - M_t)^(-1),
    score_t(v) := || B_t^(1/2) C_t(v) B_t^(1/2) ||,
    drift_t(v) := tr(B_t C_t(v) B_t).

### Sublemma L1 (proved): averaging bound for drift

For every t < m0,

    (1/r_t) sum_{v in R_t} drift_t(v)
    <= (t D / r_t) * tr(B_t^2).

Proof sketch:
1. Define Q_t := sum_{v in R_t} C_t(v)
             = sum_{u in S_t, v in R_t, u~v} X_{uv}.
2. By PSD ordering:
       Q_t <= sum_{u in S_t} sum_{v in I0, u~v} X_{uv} =: sum_{u in S_t} A_u^core.
3. Each A_u^core is PSD with
       ||A_u^core|| <= tr(A_u^core) = ell_u <= D,
   hence A_u^core <= D I and Q_t <= t D I.
4. Therefore
       sum_{v in R_t} drift_t(v)
     = tr(B_t Q_t B_t)
     <= t D tr(B_t^2),
   then divide by r_t.

So there exists v in R_t with

    drift_t(v) <= (t D / r_t) * tr(B_t^2).

### Sublemma L2 (open): good-step score control

There exist universal constants gamma in (0,1), theta in (0,1) such that:
for every t <= gamma * epsilon * n with M_t <= epsilon I, there is some
v in R_t with

    score_t(v) <= theta.

This is the first genuinely missing operator-valued statement.

### Sublemma L3 (open): good-step drift control

There is a universal K > 0 such that under the same hypotheses as L2, one can
choose v in R_t (possibly among those from L2) with

    drift_t(v) <= K / r_t.

Combined with L2, this gives a controlled potential increment per step.

### Reduction proposition (proved): L2 + L3 imply linear-size Case 2b closure

Assume L2 and L3 hold with constants gamma, theta, K.
Define gamma0 := min(gamma, 1/6), and set T := floor(gamma0 * epsilon * n).
Since m0 >= epsilon*n/6, we have T <= gamma0*epsilon*n <= m0, so T steps are
admissible.

Choose v_t at each step t = 0,...,T-1 satisfying both bounds.

If score_t(v_t) < 1, Neumann-series expansion gives

    (I - Y)^(-1) <= I + Y/(1-||Y||)     for Y := B_t^(1/2) C_t(v_t) B_t^(1/2),

hence

    Phi_{t+1} - Phi_t <= drift_t(v_t)/(1-score_t(v_t))
                       <= K / ((1-theta) r_t),
    Phi_t := tr(B_t).

Summing for t = 0,...,T-1:

    Phi_T <= Phi_0 + (K/(1-theta)) * sum_{j=m0-T+1}^{m0} (1/j)
         <= Phi_0 + (K/(1-theta)) * log(m0/(m0-T)).

All updates satisfy score_t(v_t) < 1, so M_t stays strictly below epsilon I.
Thus M_T <= epsilon I and S_T is epsilon-light, with

    |S_T| = T = floor(gamma0 * epsilon * n).

Up to integer rounding, this gives a universal linear bound

    |S_T| >= c0 * epsilon * n,   c0 = gamma0/2,

for all epsilon*n >= 2 (the finite epsilon*n < 2 cases are absorbed by
rounding/convention constants).

### Sharper reduction (proved): L2* alone already gives existence

For pure existence (not potential control), drift bounds are unnecessary.
Assume L2* with constants c_step > 0 and theta < 1:

    for every t <= c_step * epsilon * n with M_t <= epsilon I,
    there exists v in R_t with score_t(v) <= theta.

Then for such v:

    B_t^(1/2) C_t(v) B_t^(1/2) <= theta I
    => C_t(v) <= theta (epsilon I - M_t),

so

    M_{t+1} = M_t + C_t(v)
            <= (1-theta) M_t + theta epsilon I
            <= epsilon I.

Hence the barrier is preserved inductively for T = floor(c_step * epsilon * n)
steps (clamped by |I0| if needed). Therefore an epsilon-light set of size
|S| = Omega(epsilon n) exists.

So the theorem-level bridge can be reframed as proving L2* only; L3 is useful
for quantitative potential tracking but not logically necessary for existence.

### What remains to prove (now explicit)

Core open statement: L2* (or L2 in a form implying L2*). L3 is optional.

### Candidate L2 (discrepancy-flavored) and current evidence

The natural quantitative target is:

    Conjecture L2*.
    There exist universal constants c_step > 0 and theta < 1 such that
    for every Case-2b state with M_t <= epsilon I and t <= c_step * epsilon * n,
    there exists v in R_t with score_t(v) <= theta.

This is weaker than "for all t <= gamma m0", and correctly matches the theorem
goal (we only need Omega(epsilon n) vertices, not a constant fraction of m0).

Calibration note: this epsilon*n horizon is not cosmetic. Numerically and
structurally, requiring good-score steps up to a constant fraction of m0 is too
strong on dense examples, while t = Theta(epsilon n) matches the actual target
size and is consistent with all tested instances.

Attempted direct averaging bound (proved but too weak):

For Y_v := B_t^(1/2) C_t(v) B_t^(1/2) (PSD),

    min_{v in R_t} ||Y_v||
    <= (1/r_t) sum_{v in R_t} ||Y_v||
    <= (1/r_t) sum_{v in R_t} tr(Y_v)
    = (1/r_t) tr(B_t Q_t)
    <= (t D / r_t) tr(B_t),

using Q_t <= t D I from L1 proof. This does not yield a universal theta<1
because tr(B_t) can grow with n and barrier proximity.

So any successful proof of L2* must use anisotropy/cancellation beyond scalar
trace averaging.

One possible route is a matrix discrepancy/paving statement on the family
{B_t^(1/2) C_t(v) B_t^(1/2)}_{v in R_t}, giving a guaranteed low-norm member
without relying only on trace averages.

Empirical check (ad hoc, not proof):
- Graph families: Erdos-Renyi, complete, dumbbell
- n in {24, 32, 40, 48, 64}
- epsilon in {0.12, 0.15, 0.2, 0.25, 0.3}
- Case-2b filter: alpha_I > epsilon
- Trajectory: greedy choose v minimizing score_t(v)
- Horizon: t <= floor(0.5 * epsilon * n)

Observed on 313 baseline Case-2b trajectories (greedy min-score updates):
- max_t min_v score_t(v): median 0.347, 90th pct 0.521, 99th pct 0.667, max 0.667
- all baseline trajectories had max_t min_v score_t(v) <= 0.95

Additional stress test (2040 randomized trajectories, choosing uniformly among
the 5 best-score candidates each step):
- max_t min_v score_t(v): median 0.417, 90th pct 0.556, 99th pct 0.667, max 0.667
- all randomized trajectories had max_t min_v score_t(v) <= 0.95

Exhaustive small-state check (n <= 14, 13 Case-2b instances, all subsets S with
|S| < floor(0.5*epsilon*n)):
- worst observed min_v score_t(v): 0.476

Caveat: this is trajectory-based evidence (along one greedy path), not a
worst-case-over-all-S_t guarantee.

### Obstruction note (why coarse bounds are insufficient)

Using only D and ||B_t|| gives

    score_t(v) <= ||B_t|| * ||C_t(v)|| <= ||B_t|| * D,

which is useless near the barrier because D <= 12/epsilon is large and
||B_t|| can grow as M_t approaches epsilon I.
So L2 must use anisotropic structure of {X_{uv}} beyond trace/leverage totals.

## MSS/KS Mapping Attempt (for L2*)

We now map L2* against standard interlacing/paving templates to isolate
exactly what new theorem is needed.

### Grouped-atom formulation at step t

Define barrier-normalized edge atoms and fixed vertex groups:

    A_{u,v}^{(t)} := B_t^(1/2) X_{uv} B_t^(1/2)   (PSD),
    C_t(v) = sum_{u in S_t, u~v} X_{uv},
    Y_t(v) := B_t^(1/2) C_t(v) B_t^(1/2) = sum_{u in S_t, u~v} A_{u,v}^{(t)}.

L2* asks for a universal theta<1 such that for all t<=c_step*epsilon*n:

    exists v in R_t with ||Y_t(v)|| <= theta.

### Template comparison

1) MSS interlacing (rank-1 random sum): controls one realization of
   sum_i xi_i A_i from expectation + atom bounds.
   Gap: L2* is not a random-sum existence question; it is a fixed-group
   minimum-over-v question on grouped sums Y_t(v).

2) Kadison-Singer/Weaver paving: partition a family of atoms into blocks
   with controlled block norm.
   Gap: block assignment is optimized by the theorem, but in L2* blocks are
   pre-determined by vertex v (no repartition freedom).

3) Matrix Chernoff/Freedman: concentration for random subset sums.
   Gap: gives high-probability bounds for sampled sums, but L2* needs
   deterministic existence of one low-norm group at every step.

### Proved but insufficient averaging bounds

- From L1 and trace averaging:

      min_v ||Y_t(v)|| <= (tD/r_t) tr(B_t),

  not enough for universal theta<1 because tr(B_t) may scale with dimension
  and barrier proximity.

- Strong sufficient condition tested and falsified numerically in general:

      sum_{v in R_t} Y_t(v) <= rho I with rho<1.

  In sampled Case-2b trajectories, ||sum_v Y_t(v)|| is often >>1 (dense cases),
  while min_v ||Y_t(v)|| remains <1. So L2* cannot rely on this strong budget
  condition.

### Candidate missing theorem (grouped paving lemma)

Conjecture GPL (step-wise grouped paving):
There exist universal c_step>0 and theta<1 such that for every Case-2b state
at time t<=c_step*epsilon*n, the fixed family {Y_t(v)}_{v in R_t} satisfies

    min_{v in R_t} ||Y_t(v)|| <= theta.

with assumptions only from the problem setup (tau_f<=epsilon on I,
sum_e X_e=I, and leverage-degree regularization on I0).

This is precisely the theorem-level bridge needed for an unconditional proof.

## Notation Reference

- L: graph Laplacian, L = Σ_e w_e b_e b_e^T
- L_{G[S]}: induced subgraph Laplacian (zero-padded to R^n)
- X_e = w_e L^{+/2} b_e b_e^T L^{+/2}: normalized edge PSD matrix
- τ_e = ||X_e|| = w_e · R_eff(u,v): leverage score of edge e
- A_v = (1/2) Σ_{u~v} X_{uv}: vertex star contribution (star domination)
- ℓ_v = tr(A_v): vertex leverage degree
- G_H = (V, {e : τ_e > ε}): heavy edge subgraph
- I: independent set in G_H
- M = L^{+/2} L_{G[S]} L^{+/2}: normalized induced Laplacian
