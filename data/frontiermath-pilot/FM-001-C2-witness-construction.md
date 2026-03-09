# FM-001 C2-strategy: Witness Construction Plan

## Feasibility Ranking

| Rank | Approach | Feasibility | Risk |
|------|----------|-------------|------|
| 1 | Extended Paley (∞ → optimized S) | HIGH — bounded search | Tightness at boundary |
| 2 | Cayley on Z_{4n-2} | MEDIUM — larger search space | No algebraic guarantee |
| 3 | Random/hybrid | LOW — exponential space | Intractable at n=50 |

**Verdict: Extended Paley is fastest**, but the construction is TIGHT —
the expected common-neighbor count is ~n-1, right at the B_{n-1}-freeness
boundary. Must verify computationally.

## The Core Construction: Extended Paley(q) + ∞

### Why Paley(97) works on 97 vertices

Paley(q) is strongly regular with λ = (q-5)/4 common neighbors per edge.
For q=97: λ = 23. Since 23 < 24, Paley(97) is B_24-free. Self-complementary,
so complement is also B_24-free (and thus B_25-free since 23 < 25).

### The extension to 98 vertices

Add vertex ∞ (index 97) with adjacency set S ⊆ {0,...,96}. Constraints:

**For B_24-freeness of G (edges involving ∞):**
For every s ∈ S: |S ∩ N_Paley(s)| < 24

**For B_25-freeness of complement (non-edges involving ∞):**
For every t ∉ S: |{u ∉ S : u ∉ N_Paley(t), u ≠ t}| < 25
Equivalently: |(V\S\{t}) ∩ (V\N_Paley(t))| < 25

### Tightness Analysis

For |S| = 48 (standard: S = QR mod 97):
- Expected |S ∩ N(v)| ≈ 48 × 48/97 ≈ 23.75
- Need max < 24. **BORDERLINE** — Weil bound allows deviation up to √97 ≈ 10

For |S| = 47:
- Expected |S ∩ N(v)| ≈ 47 × 48/97 ≈ 23.2 — **more room**
- Complement: ∞ has degree 50 in complement
- Expected complement intersection ≈ 50 × 48/97 ≈ 24.7 < 25 — **also OK**

**Recommendation: try |S| = 47 first (one QR removed), then |S| = 48.**

## Computational Recipe for Codex

### Step 1: Build Paley(97)
```python
def quadratic_residues(q):
    """Return set of QRs mod q (nonzero squares)."""
    return {(x * x) % q for x in range(1, q)}

def paley_adj(q):
    """Build Paley(q) adjacency: edge iff difference is QR."""
    qr = quadratic_residues(q)
    adj = [[0]*q for _ in range(q)]
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in qr:
                adj[i][j] = adj[j][i] = 1
    return adj
```

### Step 2: Try extended Paley with S = QR
```python
def try_extension(q, S, n):
    """Check if Paley(q) + ∞ adjacent to S is B_{n-1}-free
    and complement B_n-free."""
    qr = quadratic_residues(q)
    # Check B_{n-1}-freeness: edges involving ∞
    max_cn_G = 0
    for s in S:
        # Common neighbors of (∞, s): elements of S that are Paley-adjacent to s
        cn = sum(1 for t in S if t != s and (t - s) % q in qr)
        max_cn_G = max(max_cn_G, cn)

    # Check B_n-freeness in complement: non-edges involving ∞
    V = set(range(q))
    notS = V - S
    max_cn_Gc = 0
    for t in notS:
        # Common neighbors of (∞, t) in complement:
        # vertices NOT in S that are NOT Paley-adjacent to t
        cn = sum(1 for u in notS if u != t and (u - t) % q not in qr)
        max_cn_Gc = max(max_cn_Gc, cn)

    return {
        'max_cn_G': max_cn_G,       # need < n-1 (< 24 for n=25)
        'max_cn_Gc': max_cn_Gc,     # need < n (< 25 for n=25)
        'B_n1_free': max_cn_G < n - 1,
        'B_n_free_complement': max_cn_Gc < n,
        'WITNESS_VALID': max_cn_G < n - 1 and max_cn_Gc < n,
    }
```

### Step 3: Search over S if standard extension fails
```python
def search_extension(q, n, attempts=1000):
    """Try removing/adding vertices from QR set to find valid S."""
    import random
    qr = list(quadratic_residues(q))

    # Try S = QR first
    result = try_extension(q, set(qr), n)
    if result['WITNESS_VALID']:
        return set(qr), result

    # Try S = QR minus each element
    for r in qr:
        S = set(qr) - {r}
        result = try_extension(q, S, n)
        if result['WITNESS_VALID']:
            return S, result

    # Try S = QR minus one, plus one NR
    nr = list(set(range(q)) - set(qr) - {0})
    for _ in range(attempts):
        remove = random.choice(qr)
        add = random.choice(nr)
        S = (set(qr) - {remove}) | {add}
        result = try_extension(q, S, n)
        if result['WITNESS_VALID']:
            return S, result

    return None, {'WITNESS_VALID': False}
```

### Step 4: Output adjacency string
```python
def adjacency_string(adj, q_plus_1):
    """Column-major binary adjacency string for the (q+1)-vertex graph."""
    bits = []
    for j in range(q_plus_1):
        for i in range(q_plus_1):
            if i < j:  # lower triangle, column-major
                bits.append(str(adj[i][j]))
    return ''.join(bits)
```

## For n=50 (T2)

Same approach with q=197. Paley(197): λ = (197-5)/4 = 48.
Need B_49-free (48 < 49 ✓) and complement B_50-free (48 < 50 ✓).
Extension to 198: same tightness analysis applies.

## For general n (T3)

Algorithm:
1. Find largest prime q ≤ 4n-2 with q ≡ 1 (mod 4)
2. Build Paley(q)
3. If q = 4n-2: done (unlikely since 4n-2 is even)
4. If q = 4n-3: extend by 1 vertex, search for valid S
5. If q < 4n-3: extend by multiple vertices (harder, may need Cayley fallback)

Runtime: O(n²) for Paley construction, O(n²) for verification. Well under 10 minutes for n ≤ 100.

## Status

- **Next dispatch**: Codex runs Step 1-3 for q=97, n=25
- **Conjecture**: H-C1-extended-paley (untested)
- **If it works**: output adjacency string → T1 SOLVED
- **If it fails**: escalate to Cayley search (H-C1-cayley-alternative)
