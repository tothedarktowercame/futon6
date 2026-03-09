# FM-001 C1-structure: Closing the Paley Off-by-One Gap

## The Problem

We need a graph G on **4n-2** vertices that is B_{n-1}-free with B_n-free
complement. Paley(q) for prime q ≡ 1 (mod 4) gives **q** vertices.
The nearest suitable prime to 4n-2 is typically q = 4n-3, leaving a
one-vertex gap.

## Key Structural Result: B_k-freeness of Paley(q)

**Theorem** (from strongly regular graph parameters):
Paley(q) is strongly regular with λ = (q-5)/4 — every edge has exactly
(q-5)/4 common neighbors. Since B_k requires ≥ k common neighbors on
some edge:

> **Paley(q) is B_k-free iff (q-5)/4 < k, i.e., q < 4k + 5.**

### Verification for our parameters

For q = 4n-3 (prime ≡ 1 mod 4):

- **λ = (4n-3-5)/4 = (4n-8)/4 = n-2**
- B_{n-1}-freeness: need λ < n-1. We have n-2 < n-1. **✓**
- Complement B_n-freeness: Paley is self-complementary, same λ. Need λ < n.
  We have n-2 < n. **✓**

So **Paley(4n-3) is a valid witness on 4n-3 vertices**. The structure is
perfect — it's just one vertex short.

### Concrete instances

| n  | 4n-2 | q=4n-3 | λ=n-2 | B_{n-1}-free? | Complement B_n-free? |
|----|------|--------|-------|---------------|---------------------|
| 25 | 98   | 97 ✓   | 23    | 23 < 24 ✓     | 23 < 25 ✓           |
| 50 | 198  | 197 ✓  | 48    | 48 < 49 ✓     | 48 < 50 ✓           |

(97 and 197 are both prime and ≡ 1 mod 4.)

## Three Approaches to Close the Gap

### Approach 1: Extended Paley (Paley + ∞)

Add a vertex ∞ to Paley(q), adjacent to all quadratic residues in GF(q).

- **Degree of ∞**: (q-1)/2 (same as all other vertices)
- **Common neighbors of (∞, r)** where r ∈ QR:
  |{s ∈ QR : s-r ∈ QR}| ≈ (q-5)/4 = n-2 by pseudorandomness
- **Risk**: exact count varies per r. If any edge involving ∞ has ≥ n-1
  common neighbors, B_{n-1}-freeness breaks

**Status**: NEEDS COMPUTATIONAL VERIFICATION for q=97.
Test: compute max_{r ∈ QR} |{s ∈ QR : s-r ∈ QR}| for q=97.
If max < 24, the extended Paley(97) + ∞ is our T1 witness.

### Approach 2: Cayley graph on Z_{4n-2}

Skip the extension entirely. Define G as a Cayley graph on Z_{4n-2} with
connection set C where |C| = 2n-1 (half the non-identity elements).

Choose C to satisfy:
- G is B_{n-1}-free: for every edge, common neighborhood size < n-1
- G̅ is B_n-free: same condition in complement

**Advantage**: exactly the right number of vertices
**Challenge**: no obvious algebraic criterion for C that guarantees
B_k-freeness. May need computational search for C.

### Approach 3: Paley(q) + prescribed neighborhood

Instead of making ∞ adjacent to all QRs, optimize the adjacency set S
of ∞ to minimize max common-neighbor count. This is a constrained
optimization: find S ⊆ V(Paley(97)) with |S| ≈ 48 such that:
- max_{v ∈ S} |S ∩ N(v)| < 24  (B_{24}-freeness for G)
- max_{v ∉ S} |(V\S) ∩ (V\N(v))| < 25  (B_{25}-freeness for complement)

## Recommended Next Step

**Dispatch to Codex**: Compute Paley(97) explicitly, check if the
standard extension (∞ adjacent to QRs) yields B_{24}-freeness on
98 vertices. This is a bounded computation:

1. Compute QR set for GF(97)
2. Build Paley(97) adjacency
3. Add vertex 98 adjacent to QRs
4. For every edge involving vertex 98: count common neighbors
5. If max < 24: output adjacency string → T1 SOLVED

If that fails, try Approach 3 (optimize S) or Approach 2 (Cayley search).
