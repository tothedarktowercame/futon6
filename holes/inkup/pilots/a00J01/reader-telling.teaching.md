# a00J01 reader-oriented telling

## Teaching

**[B0]** On \([0,1]\), let \(f\) be essentially bounded and almost everywhere
strongly measurable.  We shall prove that
\[
\lVert f\rVert_p\longrightarrow\lVert f\rVert_\infty
\qquad\text{as }p\to\infty.
\]

**[B1]** The essential-supremum norm is
\[
\lVert f\rVert_\infty
=\inf\{M\ge0:\lvert f\rvert\le M\text{ almost everywhere}\}.
\]

**[B2]** Restricted Lebesgue measure on \([0,1]\) has total mass \(1\), so
exponent monotonicity with \(p\le\infty\) gives
\(\lVert f\rVert_p\le\lVert f\rVert_\infty\) for every exponent \(p\).
Thus the comparison consumes unit total measure, not a hypothesis
\(p\ge1\); this says nothing about whether the resulting quantity is a norm.

**[B3]** For every real \(t>0\),
\[
t^{1/n}\longrightarrow1
\qquad\text{as }n\to\infty,
\]
because \(1/n\to0\) and the real power \(t^x\) is continuous at \(x=0\).

**[B4]** Fix \(\alpha<\lVert f\rVert_\infty\), choose
\(\alpha<\beta<\lVert f\rVert_\infty\), and let \(g\) be the measurable
representative equal to \(f\) almost everywhere.  The fixed set
\[
s=\{x:\beta<\lVert g(x)\rVert\}
\]
is measurable, has finite \(\mu\)-measure since \(\mu(\text{univ})=1\), and has positive
measure: otherwise \(g\) would be bounded by \(\beta\) almost everywhere,
forcing \(\lVert f\rVert_\infty\le\beta\), a contradiction.  This set is used
because it supplies one lower estimate while the exponent varies.

**[B5]** Since \(0<\mu(s)<\infty\), B3 gives a positive natural number \(n\)
for which
\[
\alpha\le\beta\,\mu(s)^{1/n}.
\]
On \(s\) we have \(\beta\le\lVert g(x)\rVert\), so the level-set estimate
gives \(\alpha\le\lVert g\rVert_n\).

**[B6]** The measure on \([0,1]\) is a probability measure, hence
\(\lVert g\rVert_n\le\lVert g\rVert_p\) whenever \(p\ge n\); combining this
with B5 and the almost-everywhere equality of \(f\) and \(g\) gives
\(\alpha\le\lVert f\rVert_p\) for every \(p\ge n\).

**[B7]** We have therefore shown that the norms are eventually above every
\(\alpha<\lVert f\rVert_\infty\), while B2 bounds them above by
\(\lVert f\rVert_\infty\); the order characterization of convergence yields
\(\lVert f\rVert_p\to\lVert f\rVert_\infty\).
