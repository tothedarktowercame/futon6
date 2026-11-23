# a99J07 reader-oriented telling

## Formal audit

**[B0]** For \(hN:1\le N\) and \(hdegree:p.natDegree=N\), the theorem
apm_a99j07 proves
\[
\lVert p(w)\rVert\le a99J07SupNormUnitDisk(p)
\max\{1,\lVert w\rVert\}^{N}
\]
for every \(w\).

**[B1]** The lemma apm_a99J07_supNorm_bddAbove uses compactness of the
closed ball and continuity of \(z\mapsto\lVert p(z)\rVert\) to make the
defining value set bounded above.  Thus apm_a99J07_eval_le_supNorm applies
le_csSup and yields \(\lVert p(z)\rVert\le a99J07SupNormUnitDisk(p)\) from
\(\lVert z\rVert\le1\).

**[B2]** The proof splits on \(hw:\lVert w\rVert\le1\); its other branch
has \(hwlt:1<\lVert w\rVert\).

**[B3]** In the first branch, apm_a99J07_eval_le_supNorm p hw gives the
value bound, while max_eq_left hw reduces the right-hand factor to \(1\).

**[B4]** In the exterior branch, \(hwlt\) implies \(hw0:w\ne0\), hence an
Invertible w instance and
\(hu:\lVert1/w\rVert\le1\).  The auxiliary object is the polynomial
p.reflect N; Polynomial.eval₂_reflect_mul_pow, using hdegree.le, gives
\[
(p.reflect\ N)(1/w)\,w^{N}=p(w).
\]
This reflection is introduced because it moves the exterior evaluation to
the disk while retaining the factor \(w^{N}\) needed on return.

**[B5]** For \(hv:\lVert v\rVert=1\),
apm_a99J07_reflect_boundary p N hdegree.le hv uses the reflection identity
and \(\lVert1/v\rVert=1\) to give
\(\lVert(p.reflect\ N)(v)\rVert\le a99J07SupNormUnitDisk(p)\).

**[B6]** The lemma apm_a99J07_poly_le_boundary applies because
p.reflect N is a polynomial and \(hu\) places \(1/w\) in the closed disk.
It returns \(v\) with \(hv:\lVert v\rVert=1\) and
\[
huv:\lVert(p.reflect\ N)(1/w)\rVert
\le\lVert(p.reflect\ N)(v)\rVert.
\]
Composing huv with the circle estimate hvb gives hqu, the required
bound at \(1/w\).  The lemma obtains its boundary witness by compactness;
if a maximizer is interior,
Complex.norm_eqOn_closure_of_isPreconnected_of_isMaxOn makes the norm
constant through the closed disk.

**[B7]** The identity heq and hqu give
\[
\lVert p(w)\rVert\le a99J07SupNormUnitDisk(p)\lVert w\rVert^{N};
\]
max_eq_right (le_of_lt hwlt) puts this in the theorem's stated form.

**[B8]** The branches \(hw\) and \(\neg hw\) exhaust all \(w\), so the
universal claim follows.
