# a97A02 reader-oriented telling

## Teaching

**[B0]** Fix \( f \in L^{p}([0,1]) \) with \( 1 \le p < \infty \) and \( h > 0 \),
and set
\[ f_h(t) = \frac{1}{2h} \int_{t-h}^{t+h} f . \]
We prove that \( f_h \) is continuous on \( [0,1] \) and that
\( \lvert f_h(t)\rvert \le (2h)^{-1/p} \lVert f\rVert_{p} \) there.
The single hypothesis \( f \in L^{p} \) must deliver both conclusions, and it
gives you nothing pointwise: no values of \( f \), no differentiability, only
integral estimates.

**[B1]** First settle what the integral means. The function \( f \) is defined
only on \( [0,1] \), yet when \( t \) lies within \( h \) of an endpoint the
window \( [t-h,t+h] \) sticks out of the interval. The reading is the zero
extension: integrate against Lebesgue measure restricted to \( [0,1] \), so
the part of the window outside contributes nothing, which is exactly the
integral of the extended function, and the average is the window integral
divided by \( 2h \). Near an endpoint you are averaging over a shorter
effective window, still dividing by \( 2h \).

**[B2]** Next, \( f \) must be integrable for those integrals to exist. On the
finite measure space \( [0,1] \), \( L^{p} \) sits inside \( L^{1} \) once
\( p \ge 1 \), so the hypothesis \( f \in L^{p} \) gives \( f \in L^{1} \).

**[B3]** Clipping can only shrink. The restricted measure of the window
\( [t-h,t+h] \) is at most the Lebesgue measure \( 2h \) of the unclipped
interval, and this holds for every \( t \), near the endpoints or not.

**[B4]** Now the continuity, and a choice of object. The tempting route is to
estimate \( \lvert f_h(t)-f_h(s)\rvert \) through the symmetric difference of
the two windows. For \( p > 1 \) that works, with care: the difference has
measure \( 2\min(\lvert s-t\rvert, 2h) \), not the \( 2\lvert s-t\rvert \) one
first writes down, since once \( \lvert s-t\rvert \) exceeds \( 2h \) the
windows are disjoint. At \( p = 1 \) the Hölder estimate gives no decay, and
one falls back on absolute continuity of the integral. That fallback alone
proves continuity for every \( p \ge 1 \), so we use it from the start, through
the primitive
\[ F(b) = \int_{(-\infty,b]} f , \]
one function of one endpoint, with no moving window. Since \( f \) is
integrable, as in B2, \( F \) is continuous: for \( b < c \),
\[ F(c) - F(b) = \int_{(b,c]} f , \]
and by absolute continuity of the integral this is small whenever the
interval \( (b,c] \) has small measure. Hence \( t \mapsto F(t\pm h) \) is
continuous on \( [0,1] \).

**[B5]** The interval identity
\[ \int_{[t-h,t+h]} f = F(t+h) - F(t-h) \]
turns the moving numerator into a difference of two continuous functions of
\( t \); the single point \( t-h \) has measure zero, so it does not matter
that the right side omits it. Dividing by the constant \( 2h \) preserves
continuity.

**[B6]** For the bound, apply Hölder on the window \( E = [t-h,t+h] \):
\[ \int_{E} \lvert f\rvert \le \lVert f\rVert_{p}\, \mu(E)^{1-1/p}
   \le \lVert f\rVert_{p} (2h)^{1-1/p}, \]
using the measure bound from B3 in the second step. The constant that will
survive is not arbitrary: \( (2h)^{-1/p} \) is the \( L^{q} \) norm of the
normalised window indicator \( (2h)^{-1}\mathbf{1}_{[t-h,t+h]} \), a constant
\( (2h)^{-1} \) on a set of measure \( 2h \), giving
\( (2h)^{-1}(2h)^{1/q} = (2h)^{-1/p} \), with \( 1/q = 0 \) when \( p = 1 \).

**[B7]** Finally \( \lvert\int_{E} f\rvert \le \int_{E}\lvert f\rvert \), and
dividing by \( 2h \) combines the exponents,
\( (1-1/p)-1 = -1/p \), leaving
\( \lvert f_h(t)\rvert \le (2h)^{-1/p}\lVert f\rVert_{p} \). Notice which
hypotheses the chain consumed: measurability, the window bound, and
\( p \ge 1 \). The restriction \( t \in [0,1] \) never entered, so the same
estimate holds for every real \( t \).

**[B8]** The two conclusions were proved independently: continuity from the
primitive route, the bound from Hölder on the window. They meet only in the
final conjunction, which holds for every \( h > 0 \) and every
\( 1 \le p < \infty \).
