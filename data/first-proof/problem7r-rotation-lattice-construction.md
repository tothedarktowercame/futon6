# Problem 7: Rotation Lattice Construction (Approach IV — E2-rot)

Date: 2026-02-12

## Purpose

Construct an explicit arithmetic uniform lattice in `Isom(H^{2k+1})` containing
an order-2 rotational involution (codimension-2 fixed set). This resolves the
lattice-existence question that was the key bottleneck for Approach IV (rotation
route) in `problem7-solution.md`.

## Construction

### Step 1: Quadratic form over a totally real number field

Let `k = Q(sqrt(2))` with ring of integers `O_k = Z[sqrt(2)]`. Define the
quadratic form in `n+1` variables (where `n = 2k+1` is odd, `n >= 7`):

```
f(x_0, x_1, ..., x_n) = (1 - sqrt(2)) x_0^2 + x_1^2 + x_2^2 + ... + x_n^2
```

**Signature analysis.** The field `k = Q(sqrt(2))` has two real embeddings:
`sigma_1: sqrt(2) -> +sqrt(2)` and `sigma_2: sqrt(2) -> -sqrt(2)`.

- Under `sigma_1`: the coefficient of `x_0^2` is `1 - sqrt(2) < 0`, so
  `f^{sigma_1}` has signature `(n, 1)`.
- Under `sigma_2`: the coefficient of `x_0^2` is `1 + sqrt(2) > 0`, so
  `f^{sigma_2}` has signature `(n+1, 0)` (positive definite).

Thus `G = SO(f)` is an algebraic group over `k` such that:
- `G(k_{sigma_1}) = SO(n, 1)` (the real hyperbolic isometry group),
- `G(k_{sigma_2})` is compact (definite form).

### Step 2: Arithmetic lattice

The group `Gamma_0 = SO(f, O_k)` (integer points of `G`) is an arithmetic
subgroup of `SO(n, 1)` via the embedding `sigma_1`.

**Cocompactness.** By the Borel-Harish-Chandra theorem, `Gamma_0` is a lattice
in `G(k_{sigma_1}) = SO(n, 1)`. By the Godement compactness criterion
(equivalently, Borel-Harish-Chandra §12), an arithmetic lattice derived from
an algebraic group over a number field is cocompact if and only if the group
is anisotropic over `k`. Since `f^{sigma_2}` is positive definite, `G` is
anisotropic over `k` (there are no nontrivial `k`-rational isotropic vectors
for `f`, because such a vector would also be isotropic for the definite form
`f^{sigma_2}`). Hence `Gamma_0` is a **uniform** (cocompact) lattice.

### Step 3: Order-2 rotation

Define the involution:

```
sigma = diag(1, -1, -1, 1, 1, ..., 1)
```

acting on `(x_0, x_1, ..., x_n)` by negating coordinates `x_1` and `x_2`.

**Claim: `sigma in SO(f, O_k)`.**

- *Preserves `f`:* The form `f` splits as `f = f_0(x_0) + f_{12}(x_1, x_2) + f_{rest}(x_3, ..., x_n)` where `f_0 = (1-sqrt(2))x_0^2`, `f_{12} = x_1^2 + x_2^2`, and `f_{rest} = x_3^2 + ... + x_n^2`. Since `sigma` fixes `x_0, x_3, ..., x_n` and negates `x_1, x_2`, and since `f_{12}(−x_1, −x_2) = f_{12}(x_1, x_2)`, we have `f(sigma(x)) = f(x)`.
- *Determinant 1:* `det(sigma) = 1 cdot (-1)^2 cdot 1^{n-2} = +1`. So `sigma in SO(f)`, not just `O(f)`.
- *Integer entries:* `sigma` is a diagonal matrix with entries in `{+1, -1} subset O_k`.

Hence `sigma in SO(f, O_k) = Gamma_0`.

**Order and fixed set.** `sigma^2 = Id`, so `sigma` has order 2. The fixed
subspace of `sigma` on `R^{n+1}` is `{x : x_1 = x_2 = 0}`, which is
`(n-1)`-dimensional (coordinates `x_0, x_3, ..., x_n`). The restriction
of `f` to this subspace is `(1-sqrt(2))x_0^2 + x_3^2 + ... + x_n^2`, which
has signature `(n-2, 1)` under `sigma_1`. So the fixed set of `sigma` acting
on `H^n` is a totally geodesic copy of `H^{n-2}` — **codimension 2**.

### Step 4: Congruence subgroup and extension

Let `I` be a proper ideal of `O_k` with `I` coprime to `2` (e.g., `I = (3)`).
Define the principal congruence subgroup:

```
pi = Gamma_0(I) = ker(SO(f, O_k) -> SO(f, O_k / I))
```

**Properties of `pi`:**

1. *Torsion-free:* By Minkowski's lemma (or Selberg's lemma with explicit
   level), for `I` coprime to `2`, the kernel of reduction mod `I` contains
   no nontrivial elements of finite order. (Any torsion element would have
   eigenvalues that are roots of unity, but reduction mod a prime `p > 2`
   separates these from 1.) In particular, `sigma notin pi` since
   `sigma` reduces to a nontrivial element mod `I` (it has eigenvalue `-1`,
   and `-1 notequiv 1 mod I` when `I` is coprime to 2).

2. *Normal in `Gamma_0`:* As a congruence kernel, `pi` is normal in `Gamma_0`.

3. *Finite index:* `[Gamma_0 : pi] < infty` since `O_k / I` is finite.

4. *`M = H^n / pi` is a closed hyperbolic manifold:* `pi` is torsion-free
   and cocompact (finite-index subgroup of cocompact `Gamma_0`), so the
   quotient is a closed manifold with contractible universal cover `H^n`.

**The extension.** Since `sigma in Gamma_0` and `sigma notin pi`, and
`sigma^2 = Id in pi` (in fact `sigma^2 = Id` in `Gamma_0`), the element
`sigma` projects to a nontrivial element of order 2 in `Gamma_0 / pi`.
Set `Gamma = <pi, sigma>`. Then:

```
1 -> pi -> Gamma -> Z/2 -> 1
```

where `Gamma` is a subgroup of `Gamma_0` (hence a cocompact lattice in
`SO(n, 1)`) containing the order-2 element `sigma`.

### Step 5: Fixed-set Euler characteristic

The involution `sigma` acts on `M = H^n / pi` (well-defined since `pi` is
normal in `Gamma` and `sigma` normalizes `pi`). The fixed set `Fix(sigma)` on
`M` is the image of `H^{n-2} cap pi`-orbits — a (possibly disconnected)
closed, totally geodesic submanifold of dimension `n - 2`.

For `n = 2k + 1` odd:
- Each fixed component has dimension `n - 2 = 2k - 1` (odd).
- Every closed odd-dimensional manifold has Euler characteristic zero
  (by Poincare duality).

So `chi(C) = 0` for every connected component `C` of the fixed set.

### Step 6: Fowler criterion and FH(Q)

Apply Fowler's Main Theorem (arXiv:1204.4667):

- The group `G = Z/2 = <sigma>` acts on the finite CW complex `Bpi = M`.
- The only nontrivial subgroup of `Z/2` is itself.
- Every connected component of the fixed set `M^{Z/2}` has `chi = 0` (Step 5).
- Therefore the orbifold extension group `Gamma = pi_1((EG x M) / G)` lies
  in `FH(Q)`.

That is: there exists a finite CW complex `Y` with `pi_1(Y) = Gamma` and
`H_*(Y_tilde; Q) = 0` for `* > 0`.

## Summary

| Claim | Status | Reference |
|---|---|---|
| `Gamma_0 = SO(f, O_k)` is a uniform lattice in `SO(n,1)` | Verified | Borel-Harish-Chandra; Godement criterion |
| `sigma in SO(f, O_k)` has order 2 | Verified | Direct computation (Step 3) |
| Fixed set of `sigma` on `H^n` is `H^{n-2}` (codim 2) | Verified | Form restriction (Step 3) |
| `pi = Gamma_0(I)` is torsion-free, `sigma notin pi` | Verified | Minkowski's lemma; `I` coprime to 2 |
| `chi(C) = 0` for fixed components when `n` odd | Verified | Poincare duality for odd-dim closed manifolds |
| `Gamma in FH(Q)` | Verified | Fowler Main Theorem (1204.4667) |

**Lattice existence for Approach IV: DISCHARGED.**

The rotation route (Approach IV) now has E2 fully resolved. The remaining
open problem is obligation S (manifold upgrade), which can proceed via either:
- Wall surgery in odd dimension (favorable L-theory parity), or
- Equivariant surgery on `(M, sigma)` using Costenoble-Waner (codim-2 gap
  satisfied).

## References

- A. Borel, Harish-Chandra, *Arithmetic Subgroups of Algebraic Groups*,
  Annals of Mathematics 75 (1962), 485-535.
- J. Millson, M. S. Raghunathan, *Geometric Construction of Cohomology for
  Arithmetic Groups I*, Proc. Indian Acad. Sci. 90 (1981), 103-123.
- J. Fowler, *Finiteness Properties of Rational Poincare Duality Groups*,
  arXiv:1204.4667.
- A. Douba, F. Vargas Pallete, *On Reflections of Congruence Hyperbolic
  Manifolds*, arXiv:2506.23994.
