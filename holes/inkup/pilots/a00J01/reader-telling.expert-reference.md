# a00J01 reader-oriented telling

## Expert reference

**[B1]** Write
\[
\lVert f\rVert_\infty
=\inf\{M\ge0:\lvert f\rvert\le M\text{ almost everywhere}\}.
\]

**[B2]** Since restricted Lebesgue measure on \([0,1]\) is a probability
measure, \(\lVert f\rVert_p\le\lVert f\rVert_\infty\) for \(p\ge1\).

**[B4]** Fix \(\alpha<\lVert f\rVert_\infty\), choose
\(\alpha<\beta<\lVert f\rVert_\infty\), take a measurable representative
\(g=f\) almost everywhere, and set \(s=\{x:\beta<\lVert g(x)\rVert\}\).
This fixed set has positive measure, since otherwise
\(\lVert g\rVert_\infty\le\beta\) and hence
\(\lVert f\rVert_\infty\le\beta\), and it has finite measure because the
ambient measure is a probability measure; thus it supplies a uniform lower
estimate while the exponent varies.

**[B5]** The standard limit \(\mu(s)^{1/n}\to1\) gives a positive natural
number \(n\) such that
\[
\alpha\le\beta\,\mu(s)^{1/n}\le\lVert g\rVert_n.
\]
Monotonicity of \(L^p\) quantities on a probability space and
\(g=f\) almost everywhere then give
\(\alpha\le\lVert f\rVert_p\) for every \(p\ge n\).

**[B7]** Thus the quantities are eventually above every
\(\alpha<\lVert f\rVert_\infty\) and are bounded above by
\(\lVert f\rVert_\infty\), so
\(\lVert f\rVert_p\to\lVert f\rVert_\infty\).
