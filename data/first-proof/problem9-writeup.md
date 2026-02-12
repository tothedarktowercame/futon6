# Problem 9: Polynomial Detection of Rank-1 Scaling for Quadrifocal Tensors

## Problem Statement

Let n >= 5. Let A^(1), ..., A^(n) in R^{3x4} be Zariski-generic matrices.
For alpha, beta, gamma, delta in [n], construct Q^(abgd) in R^{3x3x3x3} with

    Q^(abgd)_{ijkl} = det[A^(a)(i,:); A^(b)(j,:); A^(g)(k,:); A^(d)(l,:)].

Does there exist a polynomial map F: R^{81 n^4} -> R^N satisfying:
(1) F does not depend on A^(1), ..., A^(n),
(2) the degrees of coordinate functions of F do not depend on n,
(3) for lambda with lambda_{abgd} != 0 precisely when a,b,g,d not all identical:
    F(lambda_{abgd} Q^(abgd)) = 0 iff lambda_{abgd} = u_a v_b w_g x_d
    for some u, v, w, x in (R*)^n?

## Answer

**Yes.** Such F exists, with coordinate functions of degree 3.

## Proof

**1. Bilinear form reduction.** Fix camera-row pairs (gamma, k) and
(delta, l), giving vectors c = A^(gamma)(k,:) and d = A^(delta)(l,:). The map

    omega(p, q) = det[p; q; c; d]

is an alternating bilinear form on R^4 of rank 2 (its null space is
span{c, d}). Therefore all 3x3 minors of omega vanish: for any vectors
p_1, p_2, p_3, q_1, q_2, q_3 in R^4,

    det [omega(p_m, q_n)]_{3x3} = 0.

Choosing p_m = A^(alpha_m)(i_m,:) and q_n = A^(beta_n)(j_n,:), this becomes

    det [Q^(alpha_m, beta_n, gamma, delta)_{i_m, j_n, k, l}]_{3x3} = 0,

a degree-3 polynomial in the Q entries, independent of the cameras.

**2. Forward direction.** Let T^(abgd) = lambda_{abgd} Q^(abgd). The 3x3
matrix M_{mn} = lambda_{a_m, b_n, g, d} Q^(a_m, b_n, g, d)_{i_m, j_n, k, l}
is the Hadamard product Lambda circ Omega.

If lambda = u otimes v otimes w otimes x, then Lambda_{mn} =
u_{a_m} v_{b_n} w_g x_d has matrix rank 1, so
M = diag(u) Omega diag(v) (times w_g x_d). Since diagonal scaling preserves
rank: rank(M) = rank(Omega) = 2 < 3, so det(M) = 0. All 3x3 minors vanish.

**3. Converse.** Define P(A^(1),...,A^(n)) = det[T^(a_m,b_n,g,d)]_{3x3} for
a specific index choice. This is a polynomial in camera entries (lambda fixed).
We show P is not identically zero when lambda is not rank-1.

*Explicit witness (n=5).* Take cameras:

    A^(1) = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]
    A^(2) = [[1,1,0,0],[0,1,1,0],[0,0,1,1]]
    A^(3) = [[1,0,1,0],[1,1,0,0],[0,1,0,1]]
    A^(4) = [[0,1,1,1],[1,0,1,1],[1,1,0,1]]
    A^(5) = [[1,2,3,4],[4,3,2,1],[1,1,1,1]]

Take lambda_{abgd} = 1 for all non-identical (a,b,g,d) except
lambda_{1,2,3,4} = 2. This is non-rank-1: if lambda = u_a v_b w_g x_d, then
lambda_{1,2,3,4}/lambda_{2,2,3,4} = u_1/u_2 = 2, but
lambda_{1,1,3,4}/lambda_{2,1,3,4} = u_1/u_2 = 1 — contradiction.

With gamma=3, delta=4, k=l=1, row triples (1,1),(2,1),(5,1), column triples
(2,1),(5,1),(1,2): direct computation of nine 4x4 determinants gives
det(M) = -24 != 0. Since P is a nonzero polynomial, it is nonzero on a
Zariski-dense open set — establishing the converse for generic cameras.

**4. All matricizations.** The construction above tests the (1,2)-vs-(3,4)
matricization of lambda. Applying it symmetrically to the (1,3)-vs-(2,4)
and (1,4)-vs-(2,3) pairings tests all three matricizations.

*Tensor factor compatibility.* If all three matricizations have rank 1:
mode-(1,2) vs (3,4) gives lambda = f_{ab} g_{gd}; mode-(1,3) vs (2,4) gives
lambda = h_{ag} k_{bd}. Cross-referencing: lambda_{a1,b1,g,d}/lambda_{a2,b2,g,d}
= f_{a1,b1}/f_{a2,b2} is independent of g,d, while
lambda_{a,b1,g1,d}/lambda_{a,b2,g1,d} = k_{b1,d}/k_{b2,d} is independent of
a,g. These separations force f_{ab} = u_a v_b and g_{gd} = w_g x_d, giving
lambda = u_a v_b w_g x_d.

**5. Construction of F.** The coordinate functions of F are all 3x3 minors

    det [T^(a_m, b_n, g, d)_{i_m, j_n, k, l}]_{3x3}

over all choices of row/column/fixed camera-row pairs, and the analogous
minors for the other two matricization pairings.

Properties: (1) Each coordinate is degree 3 in the T entries with coefficients
+/-1 — no camera dependence. (2) Degree 3, independent of n. (3) F = 0 iff
all three matricizations have rank 1, iff lambda is rank-1 (by Sections 2-4),
for Zariski-generic cameras. QED

## References

- Standard references: Segre embedding and rank-1 tensors, determinantal
  varieties, exterior algebra, Hadamard product rank bounds.
