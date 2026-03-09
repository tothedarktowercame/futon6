# FM-001: R(B_{n-1}, B_n) = 4n-1 — Formal Proof

## Problem Statement

Let B_k denote the book graph K_2 + K̄_k (k triangles sharing a common edge).
Prove that R(B_{n-1}, B_n) ≥ 4n-1 for all n ≥ 2 such that q = 2n-1 is a
prime power with q ≡ 1 (mod 4).

Combined with the Rousseau-Sheehan upper bound R(B_{n-1}, B_n) ≤ 4n-1
(Theorem 1 of [RS78]), this gives **R(B_{n-1}, B_n) = 4n-1** exactly.

## Definitions

**Book graph**: B_k = K_2 + K̄_k. Equivalently, B_k is the graph with vertices
{u, v, w_1, ..., w_k} where u,v are the "spine" and each w_i is adjacent to
both u and v. A graph G contains B_k iff there exists an edge (u,v) with
|Γ(u,v)| ≥ k, where Γ(u,v) = {w : w adj u and w adj v}.

**2-block-circulant graph**: Given an abelian group G and subsets D₁₁, D₁₂,
D₂₂ ⊆ G, define Γ_G(D₁₁, D₁₂, D₂₂) with vertex set V₁ ⊔ V₂ (V₁ = V₂ = G)
where x,y are adjacent iff:
- x,y ∈ V₁ and y-x ∈ D₁₁
- x,y ∈ V₂ and y-x ∈ D₂₂
- x ∈ V₁, y ∈ V₂ and y-x ∈ D₁₂

**Intersection counts**: For X, Y ⊆ G and d ∈ G:
- Δ(X, Y, d) = |{(x,y) ∈ X×Y : x-y = d}|
- Σ(X, Y, d) = |{(x,y) ∈ X×Y : x+y = d}|

## Construction

Let q = 2n-1 be a prime power with q ≡ 1 (mod 4). Let Q and N denote the
sets of nonzero quadratic residues and non-residues in F_q, respectively.
Note |Q| = |N| = (q-1)/2 = n-1.

**Define**: G = Γ_{F_q}(Q, Q, N) — a 2-block-circulant graph on 2q = 4n-2
vertices with:
- D₁₁ = Q (QR adjacency within block 1)
- D₁₂ = Q (QR adjacency between blocks)
- D₂₂ = N (NR adjacency within block 2)

## Key Lemma (QR/NR Intersection Counts)

For q ≡ 1 (mod 4) prime power, Q and N in F_q:

| Δ(X,Y,d)        | d ∈ Q       | d ∈ N       | d = 0 |
|------------------|-------------|-------------|-------|
| Δ(Q, Q, d)      | (q-5)/4     | (q-1)/4     | —     |
| Δ(N, N, d)      | (q-1)/4     | (q-5)/4     | —     |
| Δ(Q, N, d)      | (q-1)/4     | (q-1)/4     | 0     |

*Proof*: Standard character sum computation over F_q using Euler's criterion.
See Wesley [W25] Lemma 10 or any reference on Paley graphs.

## Complements

Since q ≡ 1 (mod 4), we have -1 ∈ Q. Therefore Q = -Q and N = -N.

- D̄₁₁ = F_q\{0}\Q = N
- D̄₂₂ = F_q\{0}\N = Q
- D̄₁₂ = F_q\Q = N ∪ {0}

## Proof of B_{n-1}-Freeness (Graph G)

We verify |Γ(u,v)| < n-1 for all edges (u,v) in G.

**Case 1**: u,v ∈ V₁, v-u ∈ D₁₁ = Q.
|Γ(u,v)| = Δ(D₁₁, D₁₁, v-u) + Δ(D₁₂, D₁₂, v-u)
         = Δ(Q, Q, d) + Δ(Q, Q, d)  where d ∈ Q
         = 2·((q-5)/4)
         = (q-5)/2 = n-3.
Since n-3 < n-1: ✓

**Case 2**: u,v ∈ V₂, v-u ∈ D₂₂ = N.
|Γ(u,v)| = Δ(D₂₂, D₂₂, v-u) + Δ(D₁₂, D₁₂, v-u)
         = Δ(N, N, d) + Δ(Q, Q, d)  where d ∈ N
         = (q-5)/4 + (q-1)/4
         = (q-3)/2 = n-2.
Since n-2 < n-1: ✓

**Case 3**: u ∈ V₁, v ∈ V₂, v-u ∈ D₁₂ = Q.
|Γ(u,v)| = Σ(D₁₁, D₁₂, v-u) + Δ(D₁₂, D₂₂, v-u)
         = Δ(Q, -Q, d) + Δ(Q, N, d)  (since -1 ∈ Q, so -Q = Q)
         = Δ(Q, Q, d) + Δ(Q, N, d)   where d ∈ Q
         = (q-5)/4 + (q-1)/4
         = (q-3)/2 = n-2.
Since n-2 < n-1: ✓

**Maximum over all cases**: max(n-3, n-2, n-2) = n-2 < n-1.
Therefore G is B_{n-1}-free. □

## Proof of B_n-Freeness (Complement Ḡ)

We verify |Γ̄(u,v)| < n for all edges (u,v) in Ḡ (complement of G).

**Case 4**: u,v ∈ V₁, v-u ∈ D̄₁₁ = N.
|Γ̄(u,v)| = Δ(D̄₁₁, D̄₁₁, d) + Δ(D̄₁₂, D̄₁₂, d)
          = Δ(N, N, d) + Δ(N∪{0}, N∪{0}, d)  where d ∈ N
          = ((q-5)/4) + ((q-5)/4 + 2)
          = (q-5)/2 + 2 = n-1.
Since n-1 < n: ✓

Note: Δ(N∪{0}, N∪{0}, d) = Δ(N,N,d) + [d ∈ N] + [-d ∈ N]
     = (q-5)/4 + 1 + 1 = (q-5)/4 + 2 (since d ∈ N and -d ∈ N because -1 ∈ Q).

**Case 5**: u,v ∈ V₂, v-u ∈ D̄₂₂ = Q.
|Γ̄(u,v)| = Δ(D̄₂₂, D̄₂₂, d) + Δ(D̄₁₂, D̄₁₂, d)
          = Δ(Q, Q, d) + Δ(N∪{0}, N∪{0}, d)  where d ∈ Q
          = (q-5)/4 + ((q-1)/4 + 0)
          = (q-3)/2 = n-2.
Since n-2 < n: ✓

Note: Δ(N∪{0}, N∪{0}, d) for d ∈ Q: Δ(N,N,d) + [d ∈ N∪{0}] + [-d ∈ N∪{0}]
     = (q-1)/4 + 0 + 0 = (q-1)/4 (since d ∈ Q, not in N∪{0}).

**Case 6**: u ∈ V₁, v ∈ V₂, v-u ∈ D̄₁₂ = N∪{0}.

Sub-case 6a: d = 0.
|Γ̄(u,v)| = Σ(D̄₁₁, D̄₁₂, 0) + Δ(D̄₁₂, D̄₂₂, 0)
          = Δ(N, -(N∪{0}), 0) + Δ(N∪{0}, Q, 0)
          = Δ(N, N∪{0}, 0) + Δ(N∪{0}, Q, 0)
          = (0 + 1) + (0 + 0)   [Δ(N,N,0)=0 but 0∈N∪{0} adds if 0∈N: no.
           Actually: Δ(N, N∪{0}, 0) = |{(x,y): x∈N, y∈N∪{0}, x-y=0}| = |N∩(N∪{0})| = |N| = (q-1)/2]
          Wait — Δ(X,Y,0) = |X ∩ Y|.
          = |N ∩ (N∪{0})| + |(N∪{0}) ∩ Q| = |N| + |∅| = (q-1)/2 + 0 = n-1.
Since n-1 < n: ✓

Sub-case 6b: d ∈ N, d ≠ 0.
|Γ̄(u,v)| = Σ(D̄₁₁, D̄₁₂, d) + Δ(D̄₁₂, D̄₂₂, d)
          = Δ(N, N∪{0}, d) + Δ(N∪{0}, Q, d)  where d ∈ N
          = (Δ(N,N,d) + [d∈N]) + (Δ(N,Q,d) + [d∈Q])
          = ((q-5)/4 + 1) + ((q-1)/4 + 0)
          = (q-1)/2 = n-1.
Since n-1 < n: ✓

**Maximum over all complement cases**: max(n-1, n-2, n-1, n-1) = n-1 < n.
Therefore Ḡ is B_n-free. □

## Main Result

**Theorem**: If q = 2n-1 is a prime power with q ≡ 1 (mod 4), then
R(B_{n-1}, B_n) = 4n-1.

*Proof*: The graph Γ_{F_q}(Q, Q, N) has 2q = 4n-2 vertices and is both
B_{n-1}-free and has B_n-free complement. Therefore R(B_{n-1}, B_n) ≥ 4n-1.
The matching upper bound R(B_{n-1}, B_n) ≤ 4n-1 is due to Rousseau and
Sheehan [RS78]. □

## Computational Verification

For n = 25: q = 49 = 7², F_q = GF(49) = F_7[x]/(x²+1).

Computed all |Γ(u,v)| for all 6 edge cases:
- Case 1 (V₁-V₁, d∈Q): max CN = 22 = n-3 ✓
- Case 2 (V₂-V₂, d∈N): max CN = 23 = n-2 ✓
- Case 3 (V₁-V₂, d∈Q): max CN = 23 = n-2 ✓
- Case 4 (comp V₁-V₁, d∈N): max CN = 24 = n-1 ✓
- Case 5 (comp V₂-V₂, d∈Q): max CN = 23 = n-2 ✓
- Case 6 (comp V₁-V₂, d∈N∪{0}): max CN = 24 = n-1 ✓

All values match the theoretical predictions exactly.

## Applicability

The condition "q = 2n-1 is prime power ≡ 1 (mod 4)" is satisfied by
infinitely many n, including:
n ∈ {3, 5, 7, 9, 13, 15, 19, 21, 25, 27, 31, 37, 41, 45, 49, ...}

For n ≤ 20, Wesley [W25] verified R(B_{n-1}, B_n) = 4n-1 computationally
via SAT/IP solvers, covering cases where q is not a prime power.

The conjecture R(B_{n-1}, B_n) = 4n-1 for ALL n ≥ 2 remains open.

## References

- [RS78] C.C. Rousseau and J. Sheehan, "On Ramsey numbers for books,"
  *Journal of Graph Theory* 2 (1978), 77–87.
- [W25] W.J. Wesley, "Lower Bounds for Book Ramsey Numbers,"
  arXiv:2410.03625v2, September 2025.

## Mission Record

- **Problem**: FM-001 (FrontierMath pilot)
- **Mode**: MAP (completed FALSIFY → CONSTRUCT → VERIFY → MAP)
- **Key conjecture**: H-C2-wesley-2block (CONFIRMED)
- **Ledger version**: 53+
- **Commits**: futon3c 47b6508, futon6 f2209eb
- **Dead ends catalogued**: Paley extension (impossible), Cayley Z_98 (failed),
  SRG(98,k,≤23,μ) (impossible)
