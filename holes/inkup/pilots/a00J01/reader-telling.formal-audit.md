# a00J01 reader-oriented telling

## Formal audit

**[B0]** For \(f\) satisfying \(\texttt{hf : AEStronglyMeasurable f μ}\) and
finite \(\lVert f\rVert_\infty\), \(\texttt{eLpNorm_tendsto_top}\) proves
\(\lVert f\rVert_p\to\lVert f\rVert_\infty\) as \(p\to\infty\), where
\(\mu=\operatorname{volume}|_{[0,1]}\).

**[B1]** The top-exponent quantity used below is the essential supremum,
\[
\lVert f\rVert_\infty
=\inf\{M\ge0:\lvert f\rvert\le M\text{ almost everywhere}\}.
\]

**[B2]** The main theorem installs \(\texttt{IsProbabilityMeasure μ}\) from
\(\mu(\text{univ})=1\); for arbitrary \(p\), the upper branch of
\(\texttt{tendsto_order}\) applies
\(\texttt{eLpNorm_le_eLpNorm_of_exponent_le}\) to \(p\le\infty\), obtaining
\(\lVert f\rVert_p\le\lVert f\rVert_\infty\) without using \(1\le p\).
Thus the comparison consumes unit total measure, and makes no assertion that
the quantity is a norm below \(p=1\).

**[B3]** \(\texttt{tendsto_rpow_inv_nat_atTop}\) proves
\(t^{1/n}\to1\) for real \(t>0\): \(\texttt{hlim}\) gives \(1/n\to0\),
while \(\texttt{hcont}\) is continuity of \(t^x\) at \(0\).

**[B4]** In \(\texttt{lower_bound}\), let
\(g=\texttt{hf.mk}\ f\); \(\texttt{hfg}\) states \(f=g\) almost everywhere
and \(\texttt{hg}\) states strong measurability of \(g\).
Choose \(\alpha<\beta<\lVert f\rVert_\infty\), let
\(s=\{x:\beta<\lVert g(x)\rVert\}\), and obtain measurability as
\(\texttt{hs}\).  For \(\texttt{hμs}\), a hypothetical \(\mu(s)=0\) gives
the almost-everywhere bound \(\texttt{hae}\), hence
\(\texttt{hgb : ‖g‖∞ ≤ β}\) and by \(\texttt{eLpNorm_congr_ae}\) the bound
\(\texttt{hfb : ‖f‖∞ ≤ β}\), contradicting
\(\beta<\lVert f\rVert_\infty\).  The finite-measure check
\(\texttt{hμs_top}\) follows from \(\mu(s)\le\mu(\text{univ})=1\).
This fixed positive-measure set supplies the exponent-independent lower region
used by \(\texttt{hlevel}\) and \(\texttt{hn_lower}\).

**[B5]** From \(\texttt{hμs}\) and \(\texttt{hμs_top}\),
\(\texttt{hμs_real}\) gives \(0<(\mu s).\text{toReal}\); finiteness
\(\texttt{hβ_top}\) and \(\alpha<\beta\) give
\(\texttt{hβ_real : α.toReal < β.toReal}\).
The limit \(\texttt{hlim}\) and \(\texttt{hevent}\) select a positive natural
number \(n\).  The \(\text{toReal}\) comparison is transported back to
\(\text{ENNReal}\) as \(\texttt{hα_prod}\);
\(\texttt{hlevel}\) records \(\beta\le\lVert g(x)\rVert\) on \(s\), and
\(\texttt{le_eLpNorm_of_bddBelow}\) yields
\(\texttt{hn_lower : α ≤ eLpNorm g n μ}\).

**[B6]** With the probability-measure instance, for every \(p\ge n\),
\(\texttt{eLpNorm_le_eLpNorm_of_exponent_le}\) gives
\(\lVert g\rVert_n\le\lVert g\rVert_p\); transitivity with
\(\texttt{hn_lower}\) and \(\texttt{eLpNorm_congr_ae hfg}\) gives
\(\alpha\le\lVert f\rVert_p\).

**[B7]** The lower branch of \(\texttt{tendsto_order}\) first chooses
\(\alpha<\beta<\lVert f\rVert_\infty\), applies
\(\texttt{lower_bound hf β hβ}\), and composes its eventual non-strict bound
with \(\alpha<\beta\); the upper branch uses B2.  These two branches establish
\(\lVert f\rVert_p\to\lVert f\rVert_\infty\).
