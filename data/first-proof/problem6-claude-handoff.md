# Handoff to Codex: Problem 6 Proof Attempt

Date: 2026-02-12
Author: Claude (Opus 4.6)

## What was added

1. `data/first-proof/problem6-proof-attempt.md`
   - Full proof attempt with new unconditional results
   - Case analysis reducing the problem to a single open subcase
   - Six closure strategies analyzed with quantitative failure bounds
   - Strategy 3 (decoupling + conditional MSS) fully formalized

## New unconditional results

### Leverage Threshold Lemma

Any epsilon-light set S (satisfying L_{G[S]} <= epsilon L) must be an
independent set in the heavy-edge subgraph G_H = {e : tau_e > epsilon}.

Proof: M = sum_{f in S} X_f >= X_e for any internal edge e, so
||M|| >= tau_e. If tau_e > epsilon, the bound is violated.

### Turan bound on G_H

G_H has at most n/epsilon edges (from sum tau_e = n-k). By Turan's theorem,
alpha(G_H) >= epsilon*n/3. So there exists I independent in G_H with
|I| >= epsilon*n/3, where all internal edges have tau_f <= epsilon.

### Case closure

- **Case 1** (I independent in G): L_{G[I]} = 0. Done with c_0 = 1/3.
- **Case 2a** (||sum X_f^I|| <= epsilon): S = I works. Done with c_0 = 1/3.
- **Case 2b** (||sum X_f^I|| > epsilon): OPEN. This is the sole remaining gap.

### Star domination critique

The prior solution's approach (A_v = (1/2) sum_{u~v} X_{uv}, then matrix
Bernstein) converts the quadratic p^2 dependence to linear p, severely
degrading concentration headroom. Demonstrated on the star graph: star
domination gives ||A_v|| = 1/2 for all vertices, making concentration
impossible despite the problem being trivially solvable there.

## What Case 2b requires

Find S subset I with |S| >= c_0 * epsilon * n and ||sum_{f in S} X_f|| <= epsilon,
given that all edges within I have tau_f <= epsilon but their collective
spectral norm exceeds epsilon.

Case 2b does arise: K_n with epsilon > 2/n has I = V, alpha_I = 1.
But K_n is trivially solvable by symmetry (take any S of size epsilon*n).

## Strategies attempted and their ceilings

| Technique                    | c_0 bound     | Bottleneck                              |
|------------------------------|---------------|-----------------------------------------|
| Trace / Markov               | sublinear O(epsilon^(3/2)*sqrt(n)) (worst-case T_I=n) | ||M|| <= tr(M), loses dimension factor |
| Star domination + Bernstein  | O(epsilon/log n) | Converts p^2 -> p, destroys headroom |
| Decoupling + MSS (1-shot)    | O(epsilon^2)  | p_A = epsilon^2/12 to shrink atoms      |
| Decoupling + MSS (recursive) | O(sqrt(epsilon)) | 4^k spectral vs 2^k vertex scaling   |
| Greedy / chromatic number    | O(epsilon)    | IS in d~1/epsilon graph has size epsilon*|I| |
| Rank spreading heuristic     | O(epsilon^2)  | tr/rank is lower bound, not upper       |

All of these are "subsample and concentrate" strategies. They all hit the
same fundamental wall: spectral contribution scales as q^2 (quadratic in
sampling rate) while set size scales as q (linear). Balancing gives
c_0 = f(epsilon), never a universal constant.

**This wall applies to the proof technique family, not to the theorem.**

## Directions NOT yet attempted (prioritized)

### Direction A: Greedy BSS-style vertex construction (HIGH priority)

Build S vertex-by-vertex using a potential function Phi = tr((epsilon*I - M_S)^{-1}).
Early vertices have no edges to S (free addition). Later vertices contribute
A_v = sum_{u in S, u~v} X_{uv}. Each individual edge X_{uv} has ||X_{uv}|| <= epsilon
(the leverage constraint). The question is whether there always exists a next
vertex whose aggregate A_v keeps Phi bounded.

This is a direct vertex-level adaptation of BSS (D2 in the method library).
Unlike the barrier method for edges (rank-1 updates), vertex addition is a
multi-rank update. The multi-rank issue is the key technical challenge, but
the leverage bound constrains each summand.

Prior analysis dismissed iterative methods too quickly by framing them as
peeling (top-down removal) rather than construction (bottom-up addition).
The bottom-up framing is more natural for BSS-style potentials.

### Direction B: Expander decomposition (HIGH priority)

Every graph decomposes into phi-expander pieces plus a sparse boundary
(at most phi*m boundary edges). Within each expander piece, eigenvalue
spreading is a theorem: the spectral structure forces ||M_S|| ~ epsilon
for random subsets of appropriate size. The boundary contributes at most phi
to the spectral norm. Setting phi = epsilon controls the boundary.

This converts the "spreading" intuition (which holds for K_n, expanders,
random regular graphs — every tested family) into a general argument via
decomposition. Completely unexplored so far.

Key references to check: Spielman-Teng expander decomposition,
Saranurak-Wang near-optimal decomposition, and whether the spectral
spreading bound within expander pieces is tight enough.

### Direction C: Structural analysis of Case 2b (MEDIUM priority)

Case 2b requires: all edges in G[I] have tau_f <= epsilon, yet
||sum X_f^I|| > epsilon. This forces G[I] to have many edges (>= 1/epsilon)
whose spectral contributions align. Possible attacks:

- Show alignment implies G[I] contains an expander-like subgraph (then
  apply Direction B locally)
- Show alignment implies a large independent subset within I (improving
  the Turan bound and reducing to Case 1/2a)
- Show alignment is incompatible with the leverage constraint tau_f <= epsilon
  plus the identity sum_{all e} X_e = I (the "budget constraint")

No one has tried the structural attack on Case 2b directly.

### Direction D: lambda_max barrier greedy on a regularized core (HIGH priority)

Codex added a sharpened blueprint in `problem6-proof-attempt.md`:

1. Deterministically extract `I0 subset I` with `|I0| >= |I|/2` and bounded
   leverage degree `ell_v <= 4 T_I/|I|` (hence `<= 12/epsilon` from coarse
   global bounds).
2. Prove a trace-only ceiling: any argument that certifies via `tr(M)` alone
   is sublinear, so trace/Markov cannot close Case 2b.
3. Switch to barrier potential with
   `B_t=(epsilon I-M_t)^(-1)`, `score(v)=||B_t^{1/2} C_v B_t^{1/2}||`,
   `drift(v)=tr(B_t C_v B_t)`.
4. Target a single operator-averaging lemma guaranteeing at each step an
   available vertex with `score(v) <= theta < 1` and
   `drift(v) <= K/(m0-t)`.

If that lemma is proved, the potential telescopes and yields
`|S| = Omega(|I0|) = Omega(epsilon n)`, i.e., universal `c0`.

Codex also added a concrete conjectural target `L2*` with corrected horizon
`t <= c_step * epsilon * n` (not `t <= gamma m0`) plus trajectory-level
numerical evidence over 313 baseline Case-2b trajectories plus 2040 randomized
trajectories (`max_t min_v score_t(v)` observed <= 0.667 in both runs), plus
a small exhaustive-state check at n<=14 with worst observed score 0.476.

Codex has now formalized this into explicit sublemmas in
`problem6-proof-attempt.md`:
- `L1` (proved): averaged drift bound
- `L2` / `L2*` (open): good-step score control up to t = Theta(epsilon n)
- `L3` (open/optional): drift control for potential-based quantitative tracking
- proved reduction: `L2 + L3` imply closure, and a sharper reduction shows `L2*`
  alone already implies linear-size existence.
- proved-but-insufficient averaging bound for `L2*`: `min score_t <= (tD/r_t) tr(B_t)`,
  which explains why scalar trace averaging cannot close the gap.
- MSS/KS mapping section added: explicit template-to-gap analysis (why
  interlacing, paving, and matrix concentration do not directly imply L2*),
  plus a precise new target conjecture `GPL` (grouped paving lemma).

## Errors fixed (from your review)

Your review of the first commit caught 5 issues, all fixed in commit 3d03889:

1. (High) Jensen direction: E[||M||] >= ||E[M]||, not <=. Rewrote subsampling
   argument via Markov on trace; marked spreading bound as heuristic.
2. (High) Schur complement order: Schur(L,S) <= L[S,S], not >=.
3. (Medium) Quadratic MSS bound marked as heuristic extrapolation (classical
   MSS handles linear sums only).
4. (Medium) Table split into 1-shot vs recursive decoupling rows.
5. (Medium) Star domination weakened from "provably unable" to "demonstrated
   severe looseness."

## Summary assessment

The proof is **one lemma away** from complete. The lemma needed:

> For any graph G where the independent set I in G_H satisfies |I| >= epsilon*n/3
> and all internal edges have tau_f <= epsilon, there exists S subset I with
> |S| >= c_0 * epsilon * n and ||sum_{f in S} X_f|| <= epsilon, for universal c_0.

Six subsample-and-concentrate approaches have been shown insufficient. Three
structural/constructive approaches (greedy BSS, expander decomposition,
Case 2b structural analysis) remain untried and are the recommended next steps.

Every graph family tested (K_n, stars, paths, cycles, expanders, dumbbells,
disjoint cliques, random regular) satisfies the theorem with c_0 >= 1/4.
No counterexample candidate has survived preliminary analysis.

## Files to read

- `data/first-proof/problem6-proof-attempt.md` — full proof attempt (this session)
- `data/first-proof/problem6-solution.md` — prior conditional solution
- `data/first-proof/problem6-method-wiring-library.md` — your method library (D1-D10)
- `data/first-proof/problem6-method-wiring-library.json` — machine-readable version
- `data/first-proof/problem6-wiring.json` — wiring diagram for prior solution
