# a96J06 candidate realizations

Status: generated candidates for reader evaluation; not accepted exposition

The bracketed Blueprint node IDs are evaluation markup.  They are not intended
for the final paper.

## Expert reference

**[B1]** Fix (r>0) and suppose, for a contradiction, that
(lVert p(z)Vertle r) throughout the radius-(r) disk.  **[B2]** Rescale
by

\[
g(w)=r^{-1}p(rw).
\]

Then (g) maps the unit disk to the closed unit disk, fixes (0), and has
(g'(0)=1).  **[B4]** The equality case of Schwarz's lemma therefore gives
(g(w)=w) on the unit disk.  **[B5]** Hence (p(z)=z) on the radius-(r)
disk; since the two polynomials agree on infinitely many points, they are
identical.  Comparing their degree-(N) coefficients gives (c_N=0), contrary
to hypothesis.  **[B6]** Thus every radius-(r) disk contains a point
(zeta) with (lVert p(zeta)Vert>r).

## Teaching

**[B0]** The obstacle is that the radius (r) varies, whereas Schwarz's lemma
is stated on the unit disk.  The useful move is therefore to remove the scale
before doing any coefficient estimates.

**[B1]** Fix (r>0).  Suppose, contrary to the desired conclusion, that
(lVert p(z)Vertle r) whenever (lVert zVert<r).  **[B2]** A point
(w) in the unit disk corresponds to (rw) in the radius-(r) disk, so set

\[
g(w)=\frac{p(rw)}{r}.
\]

This normalization divides the output bound by the same (r): it turns our
assumption into (lVert g(w)Vertle1) on the unit disk.  It also preserves
the two decisive normalizations, (g(0)=0) and (g'(0)=1).

**[B3]** The function (g) is holomorphic, maps the unit disk into the closed
unit disk, and fixes its centre.  Schwarz's lemma first gives the usual bound
(lVert g(w)VertlelVert wVert).  **[B4]** More is true here: the
derivative already has the maximal possible norm at the origin.  The equality
case says that (g) must be a rotation.  Since (g'(0)=1), that rotation is
the identity, so (g(w)=w).

**[B5]** Undoing the scaling shows that (p(z)=z) throughout the original
disk.  Two polynomials that agree on infinitely many points are identical;
therefore all coefficients of degree at least two vanish.  In particular
(c_N=0), contradicting the assumption that the top coefficient is nonzero.
**[B6]** Our disk-boundedness supposition was false.  Since (r>0) was
arbitrary, every such disk contains a (zeta) for which
(lVert p(zeta)Vert>r).

## Formal audit

**[B0]** Audit target: for each (r>0), construct a (zeta) with
(lVertzetaVert<r) and (r<lVert p(zeta)Vert).  **[B1]** After
introducing (r>0), negating the existential conclusion gives

\[
orall z,quad lVert zVert<r\Longrightarrow
lVert p(z)Vertle r.
\]

**[B2]** Define (g(z)=r^{-1}p(rz)).  The separately proved rescaling lemma
gives (g'(0)=1), and direct simplification gives (g(0)=0).

**[B3]** Polynomial differentiability establishes differentiability of (g)
on the unit ball.  If (lVert wVert<1), then
(lVert rwVert<r); the negated conclusion therefore yields
(lVert g(w)Vertle1).  Thus (g) maps the open unit ball into the closed
unit ball centred at (g(0)).

**[B4]** `Complex.affine_of_mapsTo_ball_of_norm_dslope_eq_div`, applied with
the derivative identity at (0), yields the affine equality case.  With
(g(0)=0) and (g'(0)=1), this specializes to (g(w)=w) throughout the unit
ball.

**[B5]** For every (zeta) in the radius-(r) ball, apply the preceding
identity to (r^{-1}zeta) and clear the nonzero factor (r).  This gives

\[
(\operatorname{poly} p)(\zeta)=X(\zeta).
\]

The sequence (r/(n+2)) supplies an infinite set of such evaluation points,
so polynomial extensionality gives
(operatorname{poly} p=X).  Taking the coefficient of degree (N), the
left side is (c_N), while the coefficient of (X) is zero because
(2le N).

**[B6]** Hence (c_N=0), contradicting the input hypothesis (c_N\ne0).
This discharges the contradiction for the arbitrary positive radius (r) and
proves the required existential point.

## Reader tasks for the first comparison

Do not ask only which version is “clearest.”  For each version record whether
a reader can:

1. recover the six-step route after one reading;
2. identify scaling as the strategic hinge;
3. say which assumption the final contradiction discharges;
4. distinguish Schwarz's inequality from its equality case;
5. locate the bridge that exists in Lean but is compressed in the original
   informal proof: equality on infinitely many points implies polynomial
   identity;
6. state one detail that could be removed without losing the intended profile.

These task answers can be compared with the mandatory nodes in
`alignment.json`; they are evidence about a presentation, not a proxy for
formal correctness.
