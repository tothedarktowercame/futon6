# FM-001: Post-Refutation Approach Evaluation

## Context

Extended Paley (approach 1) was refuted by the **twin obstruction** (commit
`e88f30e`). Vertex 0 and ∞ have identical neighborhoods (both = QR mod 97),
so complement edge (0,∞) has 48 common neighbors — violating B_25-freeness.

This document evaluates the remaining approaches.

## Approach 3: Paley(97) + ∞ Extension — PROVED IMPOSSIBLE

### The twin obstruction was a symptom; the disease is deeper

The S=QR twin obstruction (commit `e88f30e`) showed ONE bad adjacency
set. But the following argument proves NO adjacency set S works:

**Constraint 1 (G B_24-free on original edges):**
For any Paley edge (u,v) with v-u ∈ QR: CN_G(u,v) = 23 + (1 if u,v ∈ S).
For B_24-free: need CN < 24, so NOT both u,v ∈ S.
⟹ **S must be independent in Paley(97)**.
Independence number α(Paley(97)) ≤ √97 ≈ 9 (Hoffman bound).
Actual search: α ≤ 6.

**Constraint 2 (complement B_25-free on ∞-edges):**
For v ∉ S: CN_complement(∞,v) = |(v+NR) \ S| = 48 - |S ∩ (v+NR)|.
For B_25-free: need 48 - |S ∩ (v+NR)| < 25, i.e., |S ∩ (v+NR)| ≥ 24.
But |S| ≤ 6, so |S ∩ (v+NR)| ≤ 6 < 24. **IMPOSSIBLE.**

**Conclusion:** Constraints 1 and 2 are jointly unsatisfiable.
No extension of Paley(97) by one vertex can be both B_24-free and
complement B_25-free, for ANY choice of adjacency set S.

This generalizes: for Paley(4n-3) with n ≥ 3, α ≤ √(4n-3) ≈ 2√n,
but complement constraint requires |S| ≥ 2n-1. Since 2√n ≪ 2n-1,
the Paley extension approach fails for ALL n.

**Verdict**: IMPOSSIBLE. Not "hard to find" — mathematically impossible.

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

## Approach 2: Cayley on Z_98 — Computational Results

### QR(97) direct embedding: FAILED

QR(97) (48 elements in {1,...,96}) embedded into Z_98:
- **Not symmetric** mod 98 (QR(97) is symmetric mod 97, not 98)
- max G-CN = 27, max complement-CN = 28
- Fails badly on both constraints

### Symmetric search: FAILED

Simulated annealing with 20 restarts × 50K steps on symmetric connection
sets (|C|=48, C = -C mod 98):
- Best: max G-CN = 26, max complement-CN = 26 (score 5)
- Target: G-CN ≤ 23, complement-CN ≤ 24 (score 0)
- Gap of 3 on G-edges, 2 on complement — not close

Tried |C| ∈ {44,...,52}: no size achieves score < 5.

### General graph search: FAILED

SA on unrestricted 98-vertex graphs (no Cayley symmetry):
- After 10K steps: max G-CN = 33, max complement-CN = 34 (score 20)
- Converging slowly, still far from target

### SRG feasibility: NO SOLUTIONS

For srg(98, k, λ, μ) with λ ≤ 23 and complement-λ ≤ 24:
From k(k-1) = kλ + (97-k)μ and 97 prime: requires 97|k or 97|μ.
No integer solutions exist with the required bounds. **No strongly
regular graph on 98 vertices has the right parameters.**

## Strategic Reassessment

All explicit construction approaches have failed:

| Approach | Status | Reason |
|----------|--------|--------|
| Extended Paley (S=QR) | REFUTED | Twin obstruction |
| Extended Paley (any S) | IMPOSSIBLE | α ≤ 6 vs complement needs |S| ≥ 24 |
| Cayley Z_98 (QR embed) | FAILED | Not symmetric mod 98, CN=27 |
| Cayley Z_98 (search) | FAILED | Best CN=26, target ≤23 |
| General 98-vertex SA | FAILED | Converging too slowly |
| SRG(98,k,≤23,μ) | IMPOSSIBLE | 97 prime, no integer solutions |

**This suggests the proof of R(B_{n-1}, B_n) ≥ 4n-1 may not use
explicit construction on 4n-2 vertices.** Alternative proof strategies:

1. **Non-constructive**: Probabilistic method (Lovász Local Lemma),
   algebraic dimension arguments, or entropy methods
2. **Inductive/recursive**: Build from smaller cases
3. **Ramsey goodness**: If B_{n-1} is "Ramsey good" wrt B_n, exact
   formulas may apply (Nikiforov-Rousseau 2005)
4. **Different graph family**: Perhaps non-vertex-transitive, non-regular,
   or using a completely different combinatorial structure

**Recommendation**: Pivot from CONSTRUCT to re-examine FALSIFY evidence
and F2-literature. The problem may require a proof-theoretic approach
(Ramsey goodness, Szemerédi regularity, or similar) rather than
explicit witness construction.

## Status

- H-C1-extended-paley: **REFUTED** (twin obstruction + impossibility proof)
- H-C1-cayley-alternative: **TESTED, FAILED** (Cayley Z_98 search unsuccessful)
- H-C2-cayley-qr-embed: **REFUTED** (not symmetric mod 98, CN=27)
- Approach 3 (any Paley extension): **PROVED IMPOSSIBLE**
- **STRATEGIC PIVOT NEEDED**: construction approach may be wrong entirely
