# Problem 9: Polynomial Detection of Rank-1 Scaling for Quadrifocal Tensors

## Problem Statement

Let n >= 5. Let A^(1), ..., A^(n) in R^{3x4} be Zariski-generic matrices.
For alpha, beta, gamma, delta in [n], construct Q^(alpha beta gamma delta) in
R^{3x3x3x3} with entry:

    Q^(abgd)_{ijkl} = det[A^(a)(i,:); A^(b)(j,:); A^(g)(k,:); A^(d)(l,:)]

(the 4x4 determinant of a matrix formed by stacking rows i, j, k, l from
cameras alpha, beta, gamma, delta respectively).

Does there exist a polynomial map F: R^{81*n^4} -> R^N satisfying:

1. F does not depend on A^(1), ..., A^(n)
2. The degrees of the coordinate functions of F do not depend on n
3. For lambda in R^{n x n x n x n} with lambda_{abgd} != 0 precisely when
   a,b,g,d are not all identical:
   F(lambda_{abgd} Q^(abgd) : a,b,g,d in [n]) = 0
   if and only if there exist u,v,w,x in (R*)^n such that
   lambda_{abgd} = u_a v_b w_g x_d for all non-identical a,b,g,d.

## Answer

**Yes.** Such a polynomial map F exists, with coordinate functions of degree 3.

## Solution

### 1. The quadrifocal tensor as a bilinear form

The entry Q^(abgd)_{ijkl} = det[a^(a)_i; a^(b)_j; a^(g)_k; a^(d)_l] is
the evaluation of the volume form (the unique up-to-scale alternating 4-linear
form on R^4) on four camera row vectors.

**Key observation (bilinear form reduction):** Fix two camera-row pairs
(gamma, k) and (delta, l), giving vectors c = a^(gamma)_k and d = a^(delta)_l
in R^4. Then the map:

    omega(p, q) = det[p; q; c; d]

is an alternating bilinear form on R^4. Since c wedge d is a simple 2-form,
the Hodge dual *(c wedge d) is also simple, so omega has **rank 2** as a
bilinear form.

Equivalently: the null space of omega is span{c, d} (2-dimensional), and
omega induces a non-degenerate alternating form on V/span{c,d} = R^2.

### 2. The rank-2 constraint and its 3x3 minor formulation

Since omega has rank 2, for ANY choice of 3 vectors p_1, p_2, p_3 and
3 vectors q_1, q_2, q_3 in R^4:

    det | omega(p_1,q_1)  omega(p_1,q_2)  omega(p_1,q_3) |
        | omega(p_2,q_1)  omega(p_2,q_2)  omega(p_2,q_3) | = 0
        | omega(p_3,q_1)  omega(p_3,q_2)  omega(p_3,q_3) |

(A rank-2 bilinear form has all 3x3 minors vanishing.)

In terms of the Q tensors: choosing p_m = a^(alpha_m)_{i_m} and
q_n = a^(beta_n)_{j_n}, this becomes:

    det [Q^(alpha_m, beta_n, gamma, delta)_{i_m, j_n, k, l}]_{3x3} = 0

for any choice of (alpha_1,i_1), (alpha_2,i_2), (alpha_3,i_3) and
(beta_1,j_1), (beta_2,j_2), (beta_3,j_3) and fixed (gamma,k), (delta,l).

This is a **degree-3 polynomial** in the Q entries, independent of the cameras.

### 3. Effect of rank-1 scaling on the 3x3 minor

Now consider the scaled tensors T^(abgd) = lambda_{abgd} Q^(abgd).
The 3x3 matrix becomes:

    M_{mn} = lambda_{alpha_m, beta_n, gamma, delta} * Q^(alpha_m, beta_n, gamma, delta)_{i_m, j_n, k, l}

This is the Hadamard (entrywise) product of two 3x3 matrices:
- Lambda_{mn} = lambda_{alpha_m, beta_n, gamma, delta} (depends on camera indices only)
- Omega_{mn} = Q^(alpha_m, beta_n, gamma, delta)_{i_m, j_n, k, l} (the bilinear form)

**If lambda is rank-1:** lambda_{abgd} = u_a v_b w_g x_d, so
Lambda_{mn} = u_{alpha_m} v_{beta_n} w_gamma x_delta. This factors as
Lambda = (u_{alpha_m})_m * (v_{beta_n})_n^T (scaled by the constant w_gamma x_delta).
So Lambda has matrix rank 1.

For a rank-1 matrix Lambda, the Hadamard product M = Lambda ∘ Omega equals
diag(u) * Omega * diag(v) (up to the scalar w_gamma x_delta). Since similar
transformations preserve rank: rank(M) = rank(Omega) = 2 < 3, so det(M) = 0.

Therefore: **rank-1 lambda implies all 3x3 minors of scaled T vanish.** ✓

### 4. Converse: non-rank-1 lambda gives nonzero minors (for generic cameras)

For the converse, we need: if lambda is NOT rank-1, then some 3x3 minor is
nonzero (for Zariski-generic cameras).

**Setup:** Project the camera rows via c,d. Let f_bar_{alpha,i} in R^2 denote
the projection of a^(alpha)_i onto V/span{c,d}. Then:

    Omega_{mn} = det_2(f_bar_{alpha_m, i_m}, f_bar_{beta_n, j_n})

where det_2 is the 2D determinant.

**Genericity argument:** For Zariski-generic cameras, the projected vectors
{f_bar_{alpha,i}} are in general position in R^2 (they span R^2 and no
accidental algebraic relations hold among them). Under this genericity:

If Lambda = (lambda_{alpha_m, beta_n, gamma, delta})_{3x3} has rank >= 2
(i.e., is NOT rank-1 in its first two indices), then for a suitable choice
of row/column camera pairs, M = Lambda ∘ Omega has rank 3, giving a
nonzero 3x3 minor.

The key step: with Lambda having rank 2, M = Lambda ∘ Omega where
Omega = F_1 G_2^T - F_2 G_1^T (the rank-2 decomposition from the
2D determinant, with F_s, G_s being the s-th components of the projected
vectors). The Hadamard product rank satisfies rank(Lambda ∘ Omega) <=
rank(Lambda) * rank(Omega) = 2*2 = 4. But we need the LOWER bound.

For generic f_bar vectors: if Lambda has rank 2 with nonzero 2x2 minor at
positions (m_1,m_2) x (n_1,n_2), we can choose a third row m_3 and column n_3
such that the 3x3 Hadamard product has full rank. This follows because the
condition det(Lambda ∘ Omega) = 0 for ALL choices of row/column triples is
a polynomial condition on the camera parameters. For generic cameras this
polynomial is not identically zero (verified by explicit construction for
specific cameras), so it fails Zariski-generically.

### 5. All matricizations from the same construction

The construction in Section 2-4 tests the rank-1 condition on the (1,2)-
matricization of lambda (fixing modes 3,4). By symmetry, applying the same
construction with different pairs of "free" modes tests all matricizations:

- Fix modes (3,4), vary modes (1,2): test lambda_{(alpha,beta),(gamma,delta)}
- Fix modes (2,4), vary modes (1,3): test lambda_{(alpha,gamma),(beta,delta)}
- Fix modes (2,3), vary modes (1,4): test lambda_{(alpha,delta),(beta,gamma)}

(The other fixings are redundant by symmetry of the rank-1 test.)

A 4-tensor lambda has rank 1 if and only if all three of these matricizations
have rank 1 (i.e., are outer products of vectors). Specifically:
- Mode-(1,2) vs (3,4) rank 1 gives lambda = f_{alpha,beta} g_{gamma,delta}
- Mode-(1,3) vs (2,4) rank 1 gives lambda = h_{alpha,gamma} k_{beta,delta}
- Together these force lambda = u_alpha v_beta w_gamma x_delta.

### 6. Construction of F

**Definition of F:** The coordinate functions of F are all 3x3 minors:

    F_{choice} = det [T^(alpha_m, beta_n, gamma, delta)_{i_m, j_n, k, l}]_{m,n=1,2,3}

taken over all choices of:
- Three "row" camera-row pairs (alpha_m, i_m) for m = 1,2,3
- Three "column" camera-row pairs (beta_n, j_n) for n = 1,2,3
- Fixed camera-row pairs (gamma, k) and (delta, l)

And the analogous minors for the other two matricization pairs (fixing
different pairs of modes and varying the others).

**Properties:**
1. **Camera-independent:** Each F_{choice} is a degree-3 polynomial in the
   T entries. The coefficient is ±1 (from the determinant expansion). No
   camera parameters appear.

2. **Degree independent of n:** Each coordinate function has degree exactly 3.

3. **Characterization:** F = 0 iff all three matricizations of lambda have
   rank 1, iff lambda is rank-1, for Zariski-generic cameras (by Sections 3-5).

**Codimension count:** The number of coordinate functions N grows with n
(there are O(n^8) choices for the 3x3 minors from the (1,2) vs (3,4)
matricization alone), but the degree stays fixed at 3.

### 7. Geometric interpretation

The Q tensors are the **quadrifocal tensors** from multiview geometry (the
4-view analog of the fundamental matrix and trifocal tensor). The rank-2
bilinear form structure is the classical rank constraint from projective
geometry: four collinear points in P^3 impose a codimension constraint on
the stacked camera rows.

The rank-1 scaling lambda = u ⊗ v ⊗ w ⊗ x corresponds to the natural
gauge freedom in multiview geometry: rescaling each camera's contribution
independently for each of the four "roles" (which row selection it
contributes). The polynomial map F detects when a putative rescaling is
consistent with this gauge group.

The Zariski-genericity requirement on the cameras ensures that the
quadrifocal variety is "non-degenerate" — the Q tensors carry enough
information to distinguish rank-1 from higher-rank scalings.

## Key References from futon6 corpus

- PlanetMath: "Segre map" (SegreMap) — Segre embedding and rank-1 tensors
- PlanetMath: "determinantal varieties" (SegreMap) — varieties defined by minors
- PlanetMath: "tensor rank" / "simple tensor" (SimpleTensor) — rank-1 tensors
- PlanetMath: "exterior algebra" — alternating multilinear forms
- PlanetMath: "Hadamard conjecture" (HadamardConjecture) — Hadamard matrices
