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
For this < 1 (so a good S exists): need q² T_I < ε, i.e., q < √(ε/T_I).

With T_I ≤ n: q < √(ε/n). Expected size: E[|S|] = q|I| ≥ √(ε/n) · εn/3
= ε^{3/2}√n / 3. For |S| ≥ c_0 εn: c_0 ≤ √(ε/n)/3 → 0 as n → ∞.

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
- The trace bound gives c_0 → 0 as n → ∞
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
| Trace / Markov               | → 0 (n)   | ||M|| ≤ tr(M), loses dim factor n   |
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
