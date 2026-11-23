# a96J06 reader-oriented telling

## Teaching

**[B0]** Let

\[
p(z)=z+\sum_{n=2}^{N}c_nz^n,
\qquad c_N\ne0.
\]

The claim is that the nonzero higher term prevents (p) from mapping every
point of any centered disk back within that disk's radius.

**[B1]** Fix (r>0), and suppose for a contradiction that
(\lVert p(z)\rVert\le r) whenever (\lVert z\rVert<r).  We shall derive
(c_N=0) for this fixed radius, contradicting the displayed hypothesis.

**[B2]** The available rigidity theorem is stated for the unit disk, while our
assumption concerns a disk of radius (r).  This mismatch explains the change
of scale

\[
g(w)=\frac{p(rw)}r.
\]

Indeed, (\lVert w\rVert<1) implies (\lVert rw\rVert<r), so the supposition
gives (\lVert p(rw)\rVert\le r), and division by (r) gives
(\lVert g(w)\rVert\le1).  Thus scaling converts precisely the radius-(r)
bound into the unit-disk bound needed for Schwarz rigidity.  It also gives
(g(0)=0) and preserves the normalized derivative (g'(0)=1).

**[B3]** Being a polynomial in (w), (g) is holomorphic on the open unit ball.
Since (g(0)=0), the bound of B2 says that (g) maps that ball into the closed
ball of radius (1) centered at (g(0)).

**[B4]** The point (0) lies in that ball; there the divided slope is the
derivative and has norm (1).  Together with B3, these supply the equality-case
lemma's hypotheses.  The lemma returns the affine formula

\[
g(w)=g(0)+(w-0)\,\operatorname{dslope}(g,0,0)
\]

throughout the unit ball.  Since
(\operatorname{dslope}(g,0,0)=g'(0)=1), this formula has slope (1), not the
undetermined rotation factor of the textbook statement.  The derivative identity is what makes the
divided-slope norm attain the radius ratio required by the lemma.  It does not assume
(g(0)=0); that fact centered the target ball in B3 and is used here to remove
the constant term.  Consequently
(g(w)=w) on the unit ball.

**[B5]** Undoing the scale gives (p(z)=z) throughout the radius-(r) ball.
That ball supplies infinitely many evaluation points, and over (\mathbb C),
agreement on an infinite set makes the two polynomials equal.
The infinitude matters: over (\mathbb F_2), the nonzero polynomial
(X^2-X) vanishes everywhere.  Applying the
coefficient function at degree (N) to the polynomial equality gives equal
degree-(N) coefficients.  The left one is (c_N), while that of (X) is zero
because (N\ge2); hence (c_N=0).

**[B6]** This contradicts (c_N\ne0), so the disk-boundedness supposition was
false.  Since (r>0) was arbitrary, every radius-(r) disk contains some
(\zeta) with (\lVert\zeta\rVert<r) and
(\lVert p(\zeta)\rVert>r).  This restores the outer quantifier over
positive radii.
