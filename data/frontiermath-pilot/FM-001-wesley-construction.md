# FM-001: Wesley 2-Block-Circulant Construction (SOLVED)

## Source

Wesley, W.J. "Lower Bounds for Book Ramsey Numbers" (arXiv:2410.03625v2,
September 2025). Theorem 2.

## The Key Insight We Were Missing

The construction is NOT Paley(q) + one vertex. It's a **2-block-circulant
graph** — two copies of F_q glued together with structured cross-connections.

**Theorem 2 (Wesley):** If q = 2n-1 is a prime power with q ≡ 1 (mod 4),
then R(B_{n-1}, B_n) = 4n-1.

Combined with Rousseau-Sheehan's upper bound R(B_{n-1}, B_n) ≤ 4n-1
(Theorem 1), this gives **exact equality**.

## Construction

Graph Γ_{F_q}(Q, Q, N) on **2q = 4n-2** vertices:

- Vertex set: V₁ ⊔ V₂, each a copy of F_q
- D₁₁ = Q (QR in F_q): edges within V₁
- D₁₂ = Q: cross-edges between V₁ and V₂
- D₂₂ = N (NR in F_q): edges within V₂

Adjacency rules:
- (u,v) ∈ V₁×V₁: adjacent iff v-u ∈ Q
- (u,v) ∈ V₂×V₂: adjacent iff v-u ∈ N
- (u,v) ∈ V₁×V₂: adjacent iff v-u ∈ Q

## Why It Works

Using Lemma 10 (QR/NR intersection counts in F_q):

**G-edges (max CN):**
- V₁-V₁ (d ∈ Q): CN = 2·Δ(Q,Q,d) = (q-5)/2 = n-3
- V₂-V₂ (d ∈ N): CN = Δ(N,N,d) + Δ(Q,Q,d) = (q-5)/4 + (q-1)/4 = n-2
- V₁-V₂ (d ∈ Q): CN = Σ(Q,Q,d) + Δ(Q,N,d) = n-2

Max G-CN = n-2 < n-1 → **B_{n-1}-free ✓**

**Complement edges (max CN):**
- V₁-V₁ comp (d ∈ N): CN = Δ(N,N,d) + Δ(N∪{0},N∪{0},d) = n-1
- V₂-V₂ comp (d ∈ Q): CN = n-2
- V₁-V₂ comp (d ∈ N∪{0}): CN ≤ n-1

Max complement-CN = n-1 < n → **B_n-free ✓**

## Computational Verification (n=25)

q = 49 = 7², F_q = GF(49) = F_7[x]/(x²+1)

```
Γ_{GF(49)}(Q, Q, N) on 98 vertices:
  V1-V1 (d∈Q): max CN = 22  (= n-3 ✓)
  V2-V2 (d∈N): max CN = 23  (= n-2 ✓)
  V1-V2 (d∈Q): max CN = 23  (= n-2 ✓)
  Comp V1-V1 (d∈N): max CN = 24  (= n-1 ✓)
  Comp V2-V2 (d∈Q): max CN = 23  (= n-2 ✓)
  Comp V1-V2 (d∈N∪0): max CN = 24  (= n-1 ✓)

  B_24-free: max CN = 23 < 24 ✓
  Complement B_25-free: max CN = 24 < 25 ✓
  R(B_24, B_25) = 99 = 4·25-1
```

## Why Our Earlier Approaches Failed

1. **1-block extension** (Paley + ∞): Creates an asymmetric vertex.
   Independence number bound (α ≤ 6) conflicts with complement CN
   requirement (|S| ≥ 24). **Provably impossible** for any S.

2. **Cayley on Z_98**: Single cyclic group lacks the structure needed.
   No SRG(98,k,≤23,μ) exists. SA search stuck at CN=26.

3. **The fix**: Two copies of F_q with **asymmetric connection rules**
   (QR within V₁, NR within V₂, QR cross). The asymmetry between
   blocks is what creates room: V₁ has lower G-CN (n-3 vs n-2) but
   higher complement-CN (n-1 vs n-2), and vice versa. The two blocks
   trade off G-freeness against complement-freeness.

## Applicability

Valid whenever q = 2n-1 is a prime power ≡ 1 (mod 4).
First 10 valid n: 3, 5, 7, 9, 13, 15, 19, 21, 25, 27.
28 valid values in [2, 100]. Density ~28%.

The paper conjectures R(B_{n-1}, B_n) = 4n-1 for ALL n, and verifies
computationally for n ≤ 20 (via SAT/IP for n where q isn't prime power).

## For FM-001

This **solves** the R(B_{n-1}, B_n) ≥ 4n-1 problem for the infinite
family where 2n-1 is prime power ≡ 1 (mod 4). The proof is:

1. Rousseau-Sheehan upper bound: R(B_{n-1}, B_n) ≤ 4n-1 (Goodman's formula)
2. Wesley lower bound: Γ_{F_q}(Q,Q,N) on 2q vertices is (B_{n-1}, B_n)-free
3. Therefore R(B_{n-1}, B_n) = 4n-1

For ALL n: still conjectural, but verified for n ≤ 20 computationally.
