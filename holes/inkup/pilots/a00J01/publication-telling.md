# a00J01 publication telling

## Publication

**[B1]** Let \(\mu\) be Lebesgue measure restricted to \([0,1]\), and write
\[
L=\lVert f\rVert_\infty
=\inf\{M\ge0:\lvert f\rvert\le M\text{ almost everywhere}\}.
\]

**[B2]** Since \(\mu([0,1])=1\), exponent monotonicity gives
\(\lVert f\rVert_p\le L\).

**[B3]** For every real \(t>0\), \(t^{1/n}\to1\) as \(n\to\infty\).

**[B4]** Fix \(\alpha<L\), choose \(\alpha<\beta<L\), take a measurable
representative \(g=f\) almost everywhere, and, to obtain one lower region
while the exponent varies, set \(s=\{x:\beta<\lVert g(x)\rVert\}\).
If \(\mu(s)=0\), then \(\lVert g\rVert_\infty\le\beta\), whence
\(L\le\beta\), a contradiction; moreover, \(\mu(s)<\infty\) since
\(\mu([0,1])=1\).

**[B5]** Since \(0<\mu(s)<\infty\) and \(\alpha<\beta\), the preceding limit gives a positive natural number \(n\)
with \(\alpha\le\beta\mu(s)^{1/n}\).  The level-set estimate, using measurable
\(s\) and \(\beta\le\lVert g(x)\rVert\) on \(s\), yields
\(\alpha\le\lVert g\rVert_n\).

**[B6]** For \(p\ge n\), exponent monotonicity on the probability space gives
\(\lVert g\rVert_n\le\lVert g\rVert_p\); combining this with the preceding bound and
almost-everywhere equality gives
\(\alpha\le\lVert f\rVert_p\).

**[B7]** Combining this eventual lower bound for every \(\alpha<L\) with the upper bound
yields \(\lVert f\rVert_p\to L=\lVert f\rVert_\infty\).
