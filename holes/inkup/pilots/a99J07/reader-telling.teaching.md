# a99J07 reader-oriented telling

## Teaching

**[B0]** Let \(p\) be a complex polynomial of degree \(N\ge1\), and let
\(M=\sup_{\lvert z\rvert\le1}\lvert p(z)\rvert\).  We prove
\[
\lvert p(w)\rvert\le M\max\{1,\lvert w\rvert\}^{N}
\]
for every \(w\in\mathbb C\).

**[B1]** The closed unit disk is compact and \(p\) is continuous, so the set
of values \(\lvert p(z)\rvert\) on it is bounded above.  Consequently its
supremum \(M\) exists and \(\lvert p(z)\rvert\le M\) whenever
\(\lvert z\rvert\le1\).

**[B2]** We split according as \(\lvert w\rvert\le1\) or
\(\lvert w\rvert>1\).  The definition of \(M\) handles the first case, but
it gives no direct information about \(p(w)\) in the second.

**[B3]** If \(\lvert w\rvert\le1\), then \(\lvert p(w)\rvert\le M\), and
\(\max\{1,\lvert w\rvert\}^{N}=1\), giving the required inequality.

**[B4]** For the exterior case, let \(q\) be the polynomial obtained by
reversing the coefficients of \(p\) up to degree \(N\).  For \(z\ne0\) it
satisfies
\[
q(z)=z^{N}p(1/z).
\]
This is the useful object because it moves the exterior point \(w\) to
\(1/w\) inside the disk while retaining exactly the factor
\(\lvert w\rvert^{N}\) needed in the conclusion.

**[B5]** If \(\lvert v\rvert=1\), then also \(\lvert1/v\rvert=1\), and the
reflection identity gives \(\lvert q(v)\rvert=\lvert p(1/v)\rvert\le M\).

**[B6]** The polynomial boundary lemma applies because \(q\) is a polynomial
and \(u=1/w\) lies in the closed unit disk.  It supplies a point \(v\) on
the unit circle such that
\[
\lvert q(u)\rvert\le\lvert q(v)\rvert.
\]
The circle estimate just proved then yields \(\lvert q(1/w)\rvert\le M\).
Thus the method needs an interior evaluation point and a polynomial; its
boundary witness is what carries the circle bound into the disk.

**[B7]** Since \(\lvert w\rvert>1\), the point \(1/w\) is in the disk, and
the reflection identity at \(1/w\) gives
\[
p(w)=q(1/w)w^{N}.
\]
Hence \(\lvert p(w)\rvert\le M\lvert w\rvert^{N}\), which is the required
bound because \(\max\{1,\lvert w\rvert\}=\lvert w\rvert\).

**[B8]** The two cases cover every complex \(w\), so the stated inequality
holds everywhere.
