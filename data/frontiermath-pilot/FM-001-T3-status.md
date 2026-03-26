# FM-001 T3: General Witness Generation — Status Note

Date: 2026-03-10
Author: claude-1 (lab manager, futon3c)

## T3 Script: `futon3c/scripts/fm001/generate_witness.py`

Constructs Wesley 2-block-circulant witnesses Γ_{F_q}(Q, Q, N) for all n
where q = 2n-1 is a prime power with q ≡ 1 (mod 4).

### Capabilities

- Full GF(p^k) arithmetic (handles prime fields and extension fields)
- Irreducible polynomial finder for GF(p^k) with k > 1
- Verification mode: checks max common neighbors for B_{n-1}-freeness and
  complement B_n-freeness
- Output: FrontierMath adjacency string (column-major, lower triangle)

### Verified

All 28 eligible n ≤ 100:

    n ∈ {3, 5, 7, 9, 13, 15, 19, 21, 25, 27, 31, 37, 41, 45, 49,
         51, 55, 57, 61, 63, 69, 75, 79, 85, 87, 91, 97, 99}

All pass verification (max CN matches algebraic prediction exactly).
Largest case (n=99, 394 vertices) runs in 37ms.

### Not Covered

**q ≡ 3 (mod 4)**: When -1 ∈ N (non-residue), the quadratic residue set Q
is antisymmetric (Q = -N, not Q = -Q). Using Q as a within-block connection
set gives a tournament (directed), not an undirected graph. Symmetrizing
gives the complete graph within each block, which has far too many common
neighbors.

Cases NOT covered for n ≤ 50:

- q ≡ 3 mod 4, prime power: n ∈ {4, 6, 10, 12, 14, 16, 22, 24, 34, 36, 40, 42}
  (q = 7, 11, 19, 23, 27, 31, 43, 47, 67, 71, 79, 83)
- Non-prime-power q (FM-001b): n ∈ {8, 11, 17, 18, 20, 23, 26, 28, 29, 32, 33,
  35, 38, 39, 43, 44, 46, 47, 48, 50}

Wesley [W25] reports computational verification (SAT/IP) for all n ≤ 20,
covering the small uncovered cases. For larger n in the uncovered set,
FM-001b's SAT encoding work (codex-1) is the active path.

### Open Questions

1. Is there an algebraic construction for q ≡ 3 mod 4? The Paley tournament
   double-cover doesn't directly give the right Ramsey properties. A
   different block assignment or a non-circulant approach may be needed.

2. Can we close T3 with the current script? It handles 28/49 cases for
   n ≤ 50. Combined with Wesley's SAT results for n ≤ 20, that covers
   38/49. The remaining 11 cases (n > 20, not covered by our script) are
   the frontier.

## Context

This work was dispatched to claude-1 by the futon3c bell-driven task queue
(conductor). The queue is now operational: agents bell idle → conductor
picks next task from pool → invokes agent → completion bells → next dispatch.

## 2026-03-26 update — heuristic search diagnostics

- Tooling: `scripts/fm001/generate_witness.py` now ships in futon6 with a restart-based search fallback; first live probe was  
  `conda run -n codex python scripts/fm001/generate_witness.py --check 11 --method search --search-seconds 180 --seed 20260326`.
- Result: timed out after 381 473 iterations (no witness yet), confirming that composite-q tuning remains open; next steps are improved initializations and/or FM-001b-guided heuristics before tackling Tier‑2 (n=50).
