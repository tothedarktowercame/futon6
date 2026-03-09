# FM-001 F2-literature: Book Graph Ramsey Constructions

## Obligation
F2-literature: existing constructions or impossibility results for R(B_{n-1}, B_n).

## Key Results

### 1. Rousseau-Sheehan (1978) — The Upper Bound
- **Result**: R(B_{n-1}, B_n) <= 4n - 1
- **Method**: Counting argument on common neighborhoods of the shared edge in B_k
- **Implication**: Our task is to show this is TIGHT by constructing witnesses on 4n-2 vertices

### 2. Nikiforov-Rousseau (2005) — Ramsey Goodness
- **Result**: Book graphs B_n are "Ramsey good" — R(B_n, K_m) = (n-1)(m-1) + 1 for m sufficiently large
- **Relevance**: Establishes that books behave predictably in Ramsey theory. Does NOT directly give the off-diagonal R(B_{n-1}, B_n), but confirms the family is well-behaved
- **Technique**: Turán-type bounds on independence numbers

### 3. Nikiforov (2011) — Extended Ramsey Goodness
- **Result**: Broader graph families have predictable Ramsey behavior
- **Key insight**: For book graphs, the critical invariants are max-triangle-free subgraph size and complement structure
- **Relevance**: Suggests structural characterization of B_k-freeness (feeds into C1-structure)

### 4. Small Cases (confirmed by SAT, n=2,3,4)
- R(B_1, B_2) = 7  (= 4(2) - 1)
- R(B_2, B_3) = 11 (= 4(3) - 1)
- R(B_3, B_4) = 15 (= 4(4) - 1)
- All match 4n-1, strongly suggesting the bound is tight for all n

### 5. Paley Graph Constructions — The Standard Tool
- Paley(q) for prime power q = 1 (mod 4): self-complementary, strongly regular (q, (q-1)/2, (q-5)/4, (q-1)/4)
- **Off-by-one problem**: We need 4n-2 vertices. Paley gives q vertices for prime power q.
  - n=25: need 98, Paley(97) gives 97 (one short)
  - n=50: need 198, Paley(197) gives 197 (one short)
- **Possible fixes**: Paley + isolated vertex, or use Cayley graph on Z_{4n-2} instead

## Open Questions (conjectures to test)

### H-F2-goodness
Does the Ramsey goodness framework directly yield the off-diagonal construction? Need to check if the goodness proof is constructive or just existential.

### H-F2-offbyone
The Paley graph is always one vertex short. Is there a standard remedy?
- (a) Find prime power q = 4n-2 = 1 (mod 4)? For n=25: 98 = 2 x 49. Not prime power. For n=50: 198 = 2 x 99. Not prime power.
- (b) Paley(q) + one vertex with prescribed adjacency? Self-complementarity breaks.
- (c) Cayley graph on Z_{4n-2} with carefully chosen connection set? Need B_k-freeness analysis.
- (d) Different algebraic construction altogether (e.g., using finite fields of characteristic 2)?

## Implications for C1-structure
The literature points to two structural questions:
1. What is the maximum number of triangles sharing a common edge in Paley(q)? (This determines B_k-freeness)
2. How does adding/removing one vertex affect the book subgraph structure?

## Status
- Ledger conjectures: H-F2-goodness (untested), H-F2-offbyone (untested)
- Active heuristics: Paley analogy, Rousseau-Sheehan context, Nikiforov extension, small case pattern
- Next move: Test H-F2-offbyone computationally — check B_k-freeness of Paley(97) for k=24
