# Problem 6 Cycle 5b Codex Handoff: Literature Search for dbar0 < 1

Date: 2026-02-13
Agent: Claude -> Codex
Type: Targeted literature search + proof attempt

## The Sub-Lemma

We need to prove:

> For any connected graph G on n vertices, any vertex subset S of size t
> within the light independent set I_0 (all edges have leverage tau_e < eps),
> at any step of the barrier greedy:
>
> **dbar0 = tr(F) / (r * eps) < 1**
>
> where tr(F) = sum of cross-edge leverages (edges between S and R = I_0 \ S),
> r = |R|, and eps in (0,1).

Equivalently: **the average cross-edge leverage per remaining vertex is
strictly less than eps.**

Equivalently: **L_R > n - 1 - tau - r*eps**, where L_R is the total leverage
of R-internal edges and tau = tr(M) is the total leverage of S-internal edges.

## Why This Matters

This is the last sub-lemma needed to close GPL-H (the main gap in Problem 6).
The proof chain is:

1. dbar0 < 1 (THIS SUB-LEMMA)
2. alpha * dbar0 <= 1/(3-eps) (PROVED, operator bound)
3. dbar = dbar0 + (alpha*dbar0) * x/(1-x) (PROVED, Neumann decomposition)
4. With dbar0 < 3/4 and items 2-3: dbar < 0.95 < 1 (CLOSES GPL-H)

Even dbar0 < 1 alone, combined with the strict inequality from alpha < 1
(PROVED) and the continuity of the assembly, would close it.

## What We Know

- Empirically: max dbar0 = 0.755 across 678 steps on 15 graph families
  (Codex C5 data). Margin to 1: 24.5%.
- K_n: dbar0 = 2t/(n*eps) = 2/3 at horizon (exact).
- The bound tr(F) <= n-1 (Foster) is too loose by factor ~n because it
  ignores R-internal leverage L_R.
- The sub-lemma is equivalent to: "most leverage stays within the larger
  component R of the vertex partition."

## Search Tasks

### Task 1: Effective Resistance Distribution in Vertex Partitions

Search for results about how effective resistance / leverage distributes
across a vertex partition {S, R} with |S| << |R|.

Keywords to search (MathOverflow, ArXiv, spectral graph theory texts):
- "effective resistance" + "vertex partition" + "leverage score distribution"
- "Foster's theorem" + "subgraph" + "leverage sum"
- "electrical network" + "cut" + "resistance budget"
- "spectral sparsification" + "leverage" + "subset"
- Spielman-Teng, Spielman-Srivastava leverage sampling bounds
- Marcus-Spielman-Srivastava leverage score properties

Specific questions:
1. Is there a known bound on sum_{e in cut(S,R)} tau_e in terms of |S|, |R|?
2. Is there a "localization" result: leverage of edges within a large set R
   accounts for at least (|R|/n)^2 fraction of total leverage?
3. Do the Spielman-Srivastava sparsification papers bound the cross-cut
   leverage? Their sampling probabilities p_e ~ tau_e suggest awareness of
   this distribution.

### Task 2: Schur Complement and Cross-Edge Leverages

The Schur complement L/S (eliminating S-vertices from L) relates to the
effective resistance structure of the R-subgraph. Specifically:

- L_R^{Schur} = L[R,R] - L[R,S] L[S,S]^{-1} L[S,R]
- Effective resistances in L/S are >= those in L (Rayleigh monotonicity)
- The cross-edge leverages relate to L[R,S] entries

Search for:
- "Schur complement" + "Laplacian" + "effective resistance"
- "graph Schur complement" + "leverage scores"
- Chandra-Raghavan-Ruzzo-Smolensky-Tiwari (CRRST) commute time results
- Connections between L/S and the leverage distribution across the cut

### Task 3: Electrical Flow Interpretation

tr(F) = sum of cross-edge leverages = sum of effective resistances weighted
by edge weights, for edges crossing {S, R}.

In electrical network terms: tr(F) is the total "importance" of the cut edges
(how much current they'd carry if randomly probed).

Search for:
- "electrical flow" + "cut" + "importance"
- "random spanning tree" + "edge cut probability"
  (leverage = probability edge is in uniform random spanning tree)
- Kirchhoff's theorem connections to cut-edge probabilities
- Lyons-Peres "Probability on Trees and Networks" — random spanning tree chapter

Key insight: tau_e = Pr[e in random spanning tree]. So:
tr(F) = E[number of cut edges in random spanning tree].
dbar0 < 1 iff E[cut edges] < r * eps.

This is a clean probabilistic statement: the expected number of tree-edges
crossing {S, R} is less than |R| * eps.

### Task 4: Matroid / Spanning Tree Bounds

Since tau_e = Pr[e in UST] (uniform spanning tree), we need:

E[|{tree edges crossing (S,R)}|] < |R| * eps.

The number of tree edges crossing (S,R) in any spanning tree is >= |S|
(since S must be connected to R). So E[cut tree edges] >= |S|.

For dbar0 < 1: need |S| < |R| * eps, i.e., t < (n-t)*eps, i.e.,
t < n*eps/(1+eps). At horizon t = eps*n/3:
eps*n/3 < n*eps/(1+eps) iff 1/3 < 1/(1+eps) iff 1+eps < 3 iff eps < 2.
Always true! So the LOWER bound on cut-tree-edges (|S|) is always < |R|*eps.

But we need the UPPER bound: E[cut tree edges] < |R|*eps. The lower bound
being small is necessary but not sufficient.

Search for:
- "spanning tree" + "cut edges" + "expected number"
- "uniform spanning tree" + "edge boundary" + "expectation"
- Kirchhoff matrix tree theorem + edge cut statistics
- Lyons "Determinantal probability" results on UST edge statistics

### Task 5: Direct Proof Attempt via Rayleigh Quotient

Try to prove dbar0 < 1 directly:

dbar0 = (1/(r*eps)) sum_{(u,v) cross} tau_{uv}
      = (1/(r*eps)) sum_{(u,v) cross} (e_u - e_v)^T L^+ (e_u - e_v)

This is a sum of quadratic forms. Can it be bounded using the spectral
properties of L?

Write: sum_cross (e_u-e_v)^T L^+ (e_u-e_v) = tr(L^+ * sum_cross (e_u-e_v)(e_u-e_v)^T)
= tr(L^+ * L_cut)

where L_cut is the Laplacian of the bipartite subgraph (cross-edges only).

So: dbar0 = tr(L^+ L_cut) / (r*eps).

Since L_cut <= L (the cross-edges are a subset of all edges):
tr(L^+ L_cut) <= tr(L^+ L) = tr(Pi) = n-1. (Too loose.)

But L_cut is specifically the CUT part of L. The Laplacian decomposes as:
L = L_int + L_cut + L_R (internal + cut + R-internal).

Can we bound tr(L^+ L_cut) using the structure of the cut?

Relevant: tr(L^+ L_cut) = sum_i lambda_i^+ * (L_cut's i-th eigenvalue in
L's eigenbasis). If L_cut is "spectrally small" relative to L...

Try to show: tr(L^+ L_cut) / (r*eps) < 1 using:
- L_cut has rank <= min(|S|, |R|) (bipartite structure)
- ||L_cut|| <= max vertex degree in cut
- Spectral properties of bipartite Laplacians

## Priority

Task 3 (UST interpretation) >= Task 4 (matroid bounds) >> Task 1 (general search)
>= Task 2 (Schur complement) >= Task 5 (direct proof).

The UST interpretation is cleanest: dbar0 < 1 iff E[cut tree edges] < r*eps.
This is a concrete probabilistic statement that may have a known answer.

## Output

- `data/first-proof/problem6-codex-cycle5b-results.json`
- `data/first-proof/problem6-codex-cycle5b-verification.md`

Include: any relevant theorems found, proof sketches, counterexample attempts,
and assessment of whether dbar0 < 1 can be proved from known results.

## Context Files

- Solution: `data/first-proof/problem6-solution.md`
- Blocking results: `data/first-proof/problem6-blocking-results.md`
- C5 results: `data/first-proof/problem6-codex-cycle5-results.json`
- C5 verification: `data/first-proof/problem6-codex-cycle5-verification.md`
- C5 handoff: `data/first-proof/problem6-codex-cycle5-handoff.md`
