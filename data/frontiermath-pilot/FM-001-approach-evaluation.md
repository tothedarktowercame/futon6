# FM-001: Post-Refutation Approach Evaluation

## Context

Extended Paley (approach 1) was refuted by the **twin obstruction** (commit
`e88f30e`). Vertex 0 and ∞ have identical neighborhoods (both = QR mod 97),
so complement edge (0,∞) has 48 common neighbors — violating B_25-freeness.

This document evaluates the remaining approaches.

## Approach 3: Optimized Extension (Paley(97) + ∞ with non-QR adjacency)

### Why small perturbations fail

The twin obstruction is structural, not accidental. The core problem:

- Vertex 0 in Paley(97) has neighbors = QR (the 48 quadratic residues)
- If ∞ is adjacent to S where S ≈ QR, then 0 and ∞ are near-twins
- In complement, (0,∞) common neighbors = {v : v ∉ N(0), v ∉ S} = NR \ S
- With S = QR: complement CN = 48 (all NR). Swapping one QR↔NR: CN = 47

**Quantitative analysis**: To get complement CN(0,∞) < 25, we need
|NR \ S| < 25, i.e., S must contain at least 24 of the 48 NR elements.
But then |S| ≥ 24 (from NR) + remaining QR elements. For the G-edge
constraint on ∞'s edges, each s ∈ S needs |S ∩ N(s)| < 24.

More precisely: if ∞ is adjacent to 0 (making it a G-edge), then
CN_G(∞, 0) = |S ∩ QR| (since N_Paley(0) = QR). If S contains k QRs,
then CN_G(∞, 0) = k. We need k < 24.

But if S contains < 24 QRs, then S must contain ≥ |S| - 23 NRs to
maintain degree. For |S| ≈ 48: at least 25 NRs in S. Then for
complement edge (0, ∞): CN_complement(0, ∞) = |NR \ S| = 48 - (NR ∩ S).
If NR ∩ S ≥ 25, then |NR \ S| ≤ 23 < 25 ✓

**So there IS a window**: |S ∩ QR| ≤ 23 AND |S ∩ NR| ≥ 24.
This means S is NR-heavy: majority non-residues, minority residues.

### The catch: every vertex is "vertex 0"

The twin problem isn't unique to vertex 0. For EVERY vertex v in Paley(97):
- N_Paley(v) = {w : w-v ∈ QR} (48 vertices)
- If ∞ adjacent to S, then complement edge (v, ∞) has CN = |(V\S) ∩ (V\N(v))|
- Equivalently: |{w ∉ S : w-v ∈ NR}| = |NR_v \ S| where NR_v = {v+r : r ∈ NR}

Since v ranges over all of Z_97, the "NR cosets" NR_v = v + NR cover all of
Z_97 in a structured way. For ∞ not adjacent to v (v ∉ S), we need
|NR_v \ S| < 25 for ALL such v.

For v ∈ S (∞ adjacent to v), we need |S ∩ N(v)| < 24, i.e.,
|S ∩ (v + QR)| < 24.

**This is a set-packing/covering problem over Z_97** — find S such that:
1. ∀v ∈ S: |S ∩ (v + QR)| < 24
2. ∀v ∉ S: |(v + NR) \ S| < 25

### Feasibility of approach 3

This is computationally tractable for q=97:
- Search space: choose |S| from ~44 to ~50 (7 sizes)
- For each size: try structured sets (cosets, arithmetic progressions,
  union of small cosets) and random sampling
- Verification is O(97²) per candidate — fast

**Risk**: The constraints may be unsatisfiable. The strongly regular
structure of Paley creates uniform pressure across all vertices, and the
two constraints pull in opposite directions (S needs to be QR-light near
each member, but NR-heavy near each non-member).

**Verdict**: MEDIUM feasibility. Worth a bounded computational search
(< 1 hour), but have a fallback ready.

## Approach 2: Cayley Graph on Z_98

### Why this avoids the problem entirely

A Cayley graph on Z_98 with connection set C (|C| = 48, C = -C, 0 ∉ C)
gives a 48-regular graph on exactly 98 vertices. No extension needed.

The graph G has edge (u,v) iff (v-u) mod 98 ∈ C. Common neighbors of
edge (u,v) where d = (v-u) mod 98 ∈ C:

  CN(u,v) = |{w : (w-u) mod 98 ∈ C AND (w-v) mod 98 ∈ C}|
           = |C ∩ (C + d)|   (where C+d = {c+d mod 98 : c ∈ C})

For B_{24}-freeness: need |C ∩ (C+d)| < 24 for all d ∈ C.
For complement B_{25}-freeness: need |(C̄) ∩ (C̄+d)| < 25 for all d ∉ C ∪ {0},
where C̄ = Z_98 \ (C ∪ {0}) (the 49 non-connection elements excluding 0).

### Structural advantage

98 = 2 × 49 = 2 × 7². Rich subgroup structure:
- Z_98 has subgroups of order 1, 2, 7, 14, 49, 98
- Can build C from cosets of subgroups
- The quotient Z_98/⟨49⟩ ≅ Z_49 and Z_98/⟨14⟩ ≅ Z_14 give natural decompositions

### Candidate connection sets

**Strategy A: Quadratic-residue-inspired**
Use QR structure from Z_97 mapped into Z_98:
- Take QR mod 97 (48 elements in {1,...,96})
- Embed into Z_98 (add element 97 if needed, or drop one element)
- Check B_k-freeness conditions

**Strategy B: Difference-set approach**
Find a (98, 48, λ)-difference set in Z_98 with λ < 24. By Fisher's
inequality, λ = 48×47/97 ≈ 23.3, which is borderline but may work
since 98 is not prime and exact difference sets with λ = 23 may exist.

**Strategy C: Greedy/hill-climbing**
Start with a random symmetric C, compute max |C ∩ (C+d)| over d ∈ C
and max |C̄ ∩ (C̄+d)| over d ∉ C. Swap elements between C and C̄ to
reduce the maximum. Local search with restarts.

**Strategy D: QR mod 7 × Z_2 structure**
Since 98 = 2 × 49, identify Z_98 with Z_2 × Z_49. Use quadratic residues
in Z_49 (which has QR structure since 49 = 7²) to define C. The 24 QRs
in Z_49 give a natural starting point: C = {(0,r) : r ∈ QR_49} ∪ {(1,r) : r ∈ S'}
for some S' ⊆ Z_49 with |S'| = 24.

### Computational recipe for Codex

```python
def cayley_check(n_vertices, C):
    """Check B_k-freeness of Cayley(Z_n, C) and its complement.
    C: set of connection elements (symmetric: c in C iff -c mod n in C).
    Returns max common neighbors for G-edges and complement-edges.
    """
    n = n_vertices
    C_set = set(C)
    C_bar = set(range(1, n)) - C_set  # non-connections (exclude 0)

    # G-edges: d in C
    max_cn_G = 0
    for d in C_set:
        # |C ∩ (C + d)|
        shifted = {(c + d) % n for c in C_set}
        cn = len(C_set & shifted)
        max_cn_G = max(max_cn_G, cn)

    # Complement edges: d in C_bar
    max_cn_comp = 0
    for d in C_bar:
        # |C_bar ∩ (C_bar + d)|
        shifted = {(c + d) % n for c in C_bar}
        cn = len(C_bar & shifted)
        max_cn_comp = max(max_cn_comp, cn)

    return max_cn_G, max_cn_comp

def search_cayley_98(n=25, attempts=10000):
    """Search for connection set C in Z_98 giving valid witness."""
    import random
    v = 4*n - 2  # 98
    target_size = v // 2 - 1  # 48 (half of non-zero elements)

    best_score = float('inf')
    best_C = None

    for _ in range(attempts):
        # Generate random symmetric C with |C| = 48
        # Choose 24 elements from {1,...,48}, add their negatives
        half = random.sample(range(1, 49), 24)
        C = set()
        for h in half:
            C.add(h)
            C.add(v - h)  # -h mod 98

        max_G, max_comp = cayley_check(v, C)
        score = max(max_G - (n-2), 0) + max(max_comp - (n-1), 0)

        if score < best_score:
            best_score = score
            best_C = C
            if score == 0:
                print(f"FOUND WITNESS: max_cn_G={max_G} < {n-1}, max_cn_comp={max_comp} < {n}")
                return sorted(C)

    print(f"Best score: {best_score}, max_G={max_G}, max_comp={max_comp}")
    return sorted(best_C) if best_C else None

# Also try structured approaches
def try_qr_inspired(n=25):
    """Try QR-based connection set on Z_98."""
    v = 4*n - 2  # 98

    # QR mod 97, embedded in Z_98
    qr97 = {(x*x) % 97 for x in range(1, 97)}  # 48 elements in {1..96}
    # All fit in Z_98 since max = 96 < 98
    max_G, max_comp = cayley_check(v, qr97)
    print(f"QR(97) in Z_98: max_cn_G={max_G}, max_cn_comp={max_comp}")
    if max_G < n-1 and max_comp < n:
        print("VALID WITNESS!")
        return sorted(qr97)

    # QR mod 49 (= QR mod 7, lifted)
    qr49 = {(x*x) % 49 for x in range(1, 49)}  # QRs in Z_49
    # Lift to Z_98: take elements and their +49 shifts
    C = set()
    for r in qr49:
        if r != 0:
            C.add(r)
            C.add(r + 49)
    # Ensure symmetric and right size
    C_sym = set()
    for c in C:
        C_sym.add(c)
        C_sym.add((98 - c) % 98)
    C_sym.discard(0)
    # Trim or pad to 48
    # ...this needs refinement based on actual |C_sym|

    return None
```

### Feasibility assessment

| Factor | Approach 3 (optimized S) | Approach 2 (Cayley Z_98) |
|--------|--------------------------|--------------------------|
| Search space | S ⊆ {0,...,96}, |S|≈48 | C ⊆ {1,...,97}, |C|=48, symmetric |
| Effective DOF | ~C(97,48) ≈ 10²⁸ | ~C(48,24) ≈ 10¹³ (symmetry halves) |
| Structure available | Paley regularity helps | Subgroup structure of Z_98 |
| QR-based seed | Start from QR, perturb | Embed QR(97) into Z_98 |
| Verification cost | O(97²) per candidate | O(98²) per candidate |
| Extension artifacts | Yes (∞ is asymmetric) | No (all vertices equivalent) |
| Self-complementary? | No (∞ breaks it) | Possible if C is chosen well |

## Recommendation

**Primary path: Approach 2 (Cayley on Z_98)**.

Reasons:
1. No extension vertex — all vertices are equivalent by Z_98 symmetry
2. Smaller effective search space due to symmetry constraint (C = -C)
3. Rich algebraic structure (Z_98 = Z_2 × Z_49) provides structured seeds
4. The QR(97) embedding into Z_98 is a natural first candidate — it
   preserves the Paley structure while adding the 98th vertex organically
5. If random search fails, hill-climbing on the max-CN score is well-defined

**Secondary path: Approach 3 (optimized S for extension)**.

Worth a bounded search (< 1 hour) in parallel. The quantitative analysis
shows a feasibility window exists, but the uniform pressure from Paley's
regularity may make it unsatisfiable.

**Codex dispatch priority**:
1. First: `try_qr_inspired(25)` — check if QR(97) embedded in Z_98 works
2. Second: `search_cayley_98(25, 100000)` — random search with scoring
3. Third: Hill-climbing on best candidate from step 2
4. Parallel: bounded approach-3 search (try_extension with optimized S)

## For n=50 (T2) and general n (T3)

Same methodology scales:
- n=50: Cayley on Z_198 = Z_2 × Z_9 × Z_11, connection set from QR(197)
- General: Cayley on Z_{4n-2}, seed from QR(4n-3) if prime, else nearest prime

## Status

- H-C1-extended-paley: **REFUTED** (twin obstruction)
- H-C1-cayley-alternative: **UNTESTED** — primary path, ready for Codex
- New conjecture needed: H-C2-cayley-qr-embed (QR(97) in Z_98 as witness)
