# Problem 6 Cycle 4 Codex Handoff

Date: 2026-02-13
Agent: Claude → Codex
Type: Library research + symbolic/numerical probes

## Context

We're proving that every graph G has an epsilon-light vertex subset S with
|S| >= c*eps*n (c universal). The proof is complete for K_n. For general G,
the gap has been narrowed to a single trace inequality:

> **alpha = tr(P_M F) / tr(F) < 1/2**

where:
- M = sum of X_e for edges internal to selected set S (PSD, rank <= t-1)
- F = sum of X_e for cross-edges (one endpoint in S, other in R = I_0 \ S)
- X_e = L^{+/2} w_e b_e b_e^T L^{+/2} (normalized edge matrix)
- P_M = projection onto col(M)
- Pi = I - (1/n)J (projection onto range of L)
- Key constraint: F + M <= Pi (Loewner order)

This is equivalent to: cross-edge leverage mass outside col(M) exceeds mass
inside col(M). Verified numerically with 8-251% margin on all tested graphs.
K_n is the tightest case at alpha = (t-1)/(2t) < 1/2.

For K_n: operator bound rho_1 <= (t-1)/(2t) is proved exactly.
For general G: open.

## Task 1: Literature Search — Is the epsilon-light subset problem known?

Search MathOverflow, ArXiv, and standard references for any of these:

**The problem itself (different names it might have):**
- "epsilon-light subgraph" or "epsilon-light vertex subset"
- "spectral vertex sparsification" (as opposed to edge sparsification)
- "induced subgraph Laplacian approximation" L_S <= eps*L
- "vertex subset with bounded spectral norm"
- "light induced subgraph" in spectral graph theory

**Closely related known results:**
- Batson-Spielman-Srivastava (2012) barrier method — does their potential
  function analysis extend to vertex selection? Their barrier is for EDGE
  weights, but ours selects VERTICES.
- Lee-Sun (2017) constructive Kadison-Singer — any vertex-selection variant?
- Spielman-Srivastava (2011) graph sparsification by effective resistances —
  any vertex-subset analogue?
- Marcus-Spielman-Srivastava (2015) interlacing families — does the partition
  existence theorem directly apply to vertex subsets?
- Anari-Gharan strongly Rayleigh / DPP framework — vertex sampling via DPP
  with leverage score kernel?

**The specific algebraic condition:**
- For PSD matrices M, F with M + F <= Pi (projection), is
  tr(MF) <= (1/2)*||M||*tr(F) a known inequality? Under what conditions?
- "Alignment of PSD matrices with subspace" or "trace of product bounded by
  half spectral norm times trace"
- Anderson's paving conjecture (proved by MSS 2013) — does it imply anything
  about our alpha < 1/2?

**Output for Task 1:** For each lead found, state:
- Paper/post reference (title, authors, year, ArXiv ID if available)
- Which aspect of our problem it addresses
- Whether it resolves the gap, narrows it, or just provides technique

## Task 2: Symbolic — Prove alpha < 1/2 via effective resistance

The cross-edge vectors decompose as z_{uv} = L^{+/2}(e_u - e_v) = z_u - z_v
where z_w = L^{+/2} e_w.

For alpha < 1/2, we need: sum_{cross (u,v)} ||P_M z_{uv}||^2 < (1/2) sum ||z_{uv}||^2.

**Attempt to prove using effective resistance structure:**

1. For v in R (not yet selected), bound ||P_M z_v||^2 in terms of the
   effective resistance R_eff(v, S) between v and the selected set S.
   Key identity: z_v = L^{+/2} e_v, and col(M) = span of S-internal edge
   vectors. The projection ||P_M z_v||^2 measures how much of v's
   "electrical potential" is captured by the S-internal circuit.

2. For u in S (already selected), ||P_M z_u||^2 is larger (u is IN S),
   but bounded by ||z_u||^2 = (L^+)_{uu} (the diagonal of L^+).

3. The cross terms <P_M z_u, P_M z_v> involve the effective resistance
   between u and v restricted to col(M).

**Try to show:** sum_v ||P_M z_v||^2 is small relative to sum ||z_v||^2,
using the fact that R is an independent set in the heavy graph (all cross-edges
are light: tau_{uv} <= eps).

**Output for Task 2:** Either a proof sketch or a precise identification of
where the argument breaks, with the tightest bound achievable.

## Task 3: Symbolic — BSS potential function adaptation

The BSS (2012) barrier method uses the potential Phi = tr(U) + tr(L) where
U = (upper - A)^{-1} and L = (A - lower)^{-1} for a matrix A in a spectral
interval [lower, upper].

**Question:** Can the BSS potential function argument be adapted to our
vertex-selection setting to directly prove dbar < 1, bypassing the rho
analysis entirely?

Key differences from standard BSS:
- We select vertices (which contribute MULTIPLE edges), not single edges
- The contribution C_t(v) = sum_{u in S, u~v} X_{uv} is not rank-1 in general
- The barrier is one-sided: M_t < eps*I (no lower barrier needed)

BSS Theorem 3.1 gives: if there exists v with
  tr(U C_v U) / tr(U) + tr(L C_v L) / tr(L) < 1
then the barrier is maintained. In our setting, with only an upper barrier:
  tr(B C_v B) / tr(B) < 1  where B = (eps*I - M)^{-1}

This is exactly dbar < 1 (our condition). The BSS analysis proceeds by
showing the potential drop is controlled. Does their counting argument
(relating potential drop to the number of items) give dbar < 1 for vertex
selection?

**Output for Task 3:** Either an adaptation of the BSS argument that proves
dbar < 1 for vertex selection, or a precise identification of where the
adaptation fails.

## Task 4: Numerical — Interlacing probe

For the barrier greedy on the standard test suite (K_40, K_80, ER_60,
Barbell_40, Star_40, Grid_8x5; eps in {0.2, 0.3, 0.5}):

At each step t, compute:
1. The characteristic polynomials p_v(x) = det(xI - Y_t(v)) for each v in R_t
2. The average polynomial Q(x) = (1/r_t) sum_v p_v(x)
3. Check: do the {p_v} form an interlacing family? Specifically, for any
   partition of R_t into groups, do the group-averaged polynomials interlace
   with Q? (Test with random partitions into 2 groups, 10 trials per step.)
4. Check: is Q real-rooted? (Already verified in Cycle 1 — confirm.)
5. Compute: largest root of Q vs dbar. Is largest_root < dbar always?
   How much tighter is the root bound?
6. If Q is real-rooted and has all nonneg roots, compute ratio:
   max_root / (sum_roots / degree). This measures how much tighter the
   largest root is compared to the trace bound (dbar).

**Output for Task 4:** JSON with interlacing check results, root data,
and assessment of whether the interlacing approach gives a strictly tighter
bound than the trace/dbar approach.

## Task 5: Numerical — Effective resistance correlation

For the same test suite, at each step t:

1. For each v in R_t, compute:
   - eff_res_v_S = effective resistance from v to S_t (as a group)
     (Hint: R_eff(v, S) = (L^+)_{vv} - 2*(L^+)_{v,S}*1_S/|S| + ...)
     Actually, compute as: insert a "super-node" collapsing S, measure R_eff.
     Or more precisely: R_eff(v, S) = min_{w in S} R_eff(v, w) is one option,
     but the electrical version is: connect all S vertices, measure R to v.
     Use: R_eff(v, S) = e_v^T L_S^+ e_v where L_S is L with S-vertices
     identified. Or simpler: ||P_{col(M)^perp} z_v||^2 / ||z_v||^2 =
     fraction of v's leverage outside col(M).

   - alpha_v = ||P_M z_v||^2 / ||z_v||^2 (per-vertex alignment with col(M))

   - per-edge alpha for each cross-edge (u,v):
     alpha_{uv} = ||P_M z_{uv}||^2 / tau_{uv}

2. Report: max alpha_v, max alpha_{uv}, and their correlation with
   effective resistance and leverage scores.

3. Key question: is max_{v in R} alpha_v < 1/2?
   (This would prove alpha < 1/2 via the strongest possible per-vertex bound.)
   If not, what is the max, and which graphs violate it?

**Output for Task 5:** JSON with per-vertex and per-edge alpha data,
correlation with effective resistance, and whether the per-vertex bound
alpha_v < 1/2 holds.

## Priority

Task 1 (literature) >> Task 3 (BSS adaptation) > Task 2 (symbolic) >=
Task 4 (interlacing) >= Task 5 (eff res).

If the literature search finds a known result, report immediately — it may
make the other tasks unnecessary.

## File locations

- Prior results: `data/first-proof/problem6-codex-cycle3-results.json`
- Cycle 4 findings: `data/first-proof/problem6-cycle4-findings.md`
- Alpha data: `data/first-proof/alpha-rho-analysis.json`
- Greedy script: `scripts/verify-p6-cycle3-codex.py`
- Alpha script: `scripts/compute-alpha-rho.py`
- Output: `data/first-proof/problem6-codex-cycle4-results.json`
- Verification: `data/first-proof/problem6-codex-cycle4-verification.md`

## Graph suite

Same as Cycle 3: K_40, K_80, ER_60(p=0.5), Barbell_40(n=80), Star_40,
Grid_8x5. Epsilons: {0.2, 0.3, 0.5}. Greedy on Turan I_0, horizon
T = floor(eps*m_0/3) with max(1,...) guard.
