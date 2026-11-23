# a96J06 reader-oriented telling

## Formal audit

**[B0]** Audit `apm_a96j06`: for each (r>0), produce (\zeta) inside the
radius-(r) ball with (r<\lVert p(\zeta)\rVert).

**[B1]** After `intro r hr`, `by_contra hcontra` and `push_neg at hcontra`
give

\[
\forall z,\quad \lVert z\rVert<r\Longrightarrow \lVert p(z)\rVert\le r.
\]

**[B2]** Schwarz rigidity uses the unit ball, while `hcontra` is a radius-(r)
bound.  Remove this mismatch with

\[
g(z)=r^{-1}p(rz).
\]

`hgzero : g 0 = 0` and `apm_a96J06_rescaled_hasDerivAt_zero` yield the
normalizations, including `hgderiv : deriv g 0 = 1`.

**[B3]** `hgdiff` gives differentiability on the open unit ball.  For (w)
there, (rw) lies in the radius-(r) ball; `hcontra`, divided by positive (r),
therefore gives (\lVert g(w)\rVert\le1).  Rewriting with `hgzero` centres this
closed target ball at (g(0)), yielding `hgmap`: this is the first use of
`hgzero`, and the dependency recorded by the mapping proof.

**[B4]** `hzero_mem`, `dslope_same`, and `hgderiv` give

\[
\lVert\operatorname{dslope}(g,0,0)\rVert=1/1.
\]

`Complex.affine_of_mapsTo_ball_of_norm_dslope_eq_div` consumes these facts,
`hgdiff`, `hgmap`, and strict convexity, returning `haffine`:

\[
g(w)=g(0)+(w-0)\operatorname{dslope}(g,0,0).
\]

`hgzero` first centred the target ball in `hgmap`; the lemma does not consume
it.  Lean uses it here again with `hgderiv` to simplify the affine formula to
`hgeq : g(w)=w`.  Thus the slope
is (1), not a rotation from the textbook.

**[B5]** `heval` applies `hgeq` at (r^{-1}\zeta) and clears (r), giving equal
evaluations of `apm_a96J06_poly N c` and (X).  In `hset`, the injective
sequence (r/(n+2)) supplies infinitely many agreement points.  Over the domain
(\mathbb C), `Polynomial.eq_of_infinite_eval_eq` consumes that infinite set
and yields `hpoly`.  Infinitude is essential: over (\mathbb F_2), the distinct
polynomials (X^2-X) and (0) agree everywhere.

`congrArg (fun p => p.coeff N)` sends `hpoly` to coefficient equality
`hcoeff`.  Then `apm_a96J06_coeff_top N hN c` gives the left coefficient
(c_N), and `hXN` gives the coefficient of (X) as zero since (hN : 2≤N).
Hence (c_N=0).

**[B6]** `hcN (hcoeff.trans hXN)` contradicts `hcN : c N ≠ 0`, discharging
`hcontra` for arbitrary positive (r), restoring the outer quantifier and the
existential witness.
