# b95J01 reader-oriented telling

## Formal audit

**[B0]** The theorem apm_b95j01, for a group \(G\), a subgroup \(H\) and the
instance \(\mathrm{H.FiniteIndex}\), concludes the conjunction of part (a)
for every \(g\) — a \(k:\mathbb{N}\) with \(0<k\), \(g^{k}\in H\),
least among positive exponents, dividing every \(m:\mathbb{Z}\) with
\(g^{m}\in H\), and dividing \(\mathrm{H.index}\) under
\(\mathrm{H.Normal}\) — and the counterexample conjunct about
b95j01CounterexampleSubgroup and b95j01CounterexampleElement.

**[B1]** The set is \(T\), defined literally as the comap of
\((\mathrm{zpowersHom}\ G)\ g\) restricted to \(H\), cast to an
\(\mathrm{AddSubgroup}\ \mathbb{Z}\); the step \(\mathrm{hT}\) proves
\(m\in T\leftrightarrow g^{m}\in H\) by rfl. So the packaging is the comap
of the powers homomorphism, and subgroup closure is inherited, not checked.

**[B2]** The positive exponent comes from
\(\mathrm{H.exists\_pow\_mem\_index\_ne\_zero}\) applied to
\(\mathrm{Subgroup.FiniteIndex.index\_ne\_zero}\) and \(g\), giving
\(n>0\) with \(g^{n}\in H\); the step \(\mathrm{hnT}\) transports it into
\(T\) by zpow_natCast. This lemma is the single consumption of the
FiniteIndex instance — without the instance no such lemma applies — and it
is the Lean form of the pigeonhole argument on the cosets
\(H,gH,\dots,g^{n}H\).

**[B3]** The generator comes from \(\mathrm{Int.subgroup\_cyclic}\ T\),
giving \(T=a\mathbb{Z}\); \(\mathrm{ha0}\) derives \(a\ne0\), since
\(a=0\) would force \(T=\bot\) and then \(n=0\), contradicting \(n>0\)
by omega.

**[B4]** The criterion \(\mathrm{hmem}\) rewrites membership in \(T\) to
\((a.\mathrm{natAbs}:\mathbb{Z})\mid m\) through
\(\mathrm{AddSubgroup.zmultiples\_eq\_closure}\),
\(\mathrm{Int.zmultiples\_natAbs}\) and
\(\mathrm{Int.mem\_zmultiples\_iff}\).

**[B5]** The witness is \(a.\mathrm{natAbs}\), with positivity from
\(\mathrm{Int.natAbs\_pos}\), membership by \(\mathrm{hmem}\) after
zpow_natCast, leastness from \(\mathrm{Nat.le\_of\_dvd}\), divisibility by
\(\mathrm{hmem}\) directly.

**[B6]** Under the \(\mathrm{Normal}\) instance installed from the
hypothesis, the membership \(g^{\mathrm{H.index}}\in H\) is
\(\mathrm{H.pow\_index\_mem}\ g\). This lemma is the Lean form of the
quotient-group and Lagroute: in \(G/H\), of order \([G:H]\), the coset
\(gH\) has order the least positive exponent, which is \(k\), so \(k\)
divides \([G:H]\).

**[B7]** The step applies \(\mathrm{hmem}\) at
\(m=(\mathrm{H.index}:\mathbb{Z})\) after rewriting with zpow_natCast and
\(\mathrm{H.pow\_index\_mem}\), then exact_mod_cast, giving
\(k\mid\mathrm{H.index}\).

**[B8]** The counterexample subgroup is
b95j01CounterexampleSubgroup, the zpowers of
\(\mathrm{Equiv.swap}\ 0\ 1\) — the transposition \((1\,2)\) in the Lean
0-based numbering. Non-normality is proved by \(\mathrm{hnormal.conj\_mem}\):
conjugating the generator by b95j01CounterexampleElement
(\(\mathrm{Equiv.swap}\ 0\ 2\), i.e. \((1\,3)\)) lands outside the subgroup,
using \(\mathrm{orderOf\_eq\_prime}\) for order \(2\) and
\(\mathrm{mem\_zpowers\_iff\_mem\_range\_orderOf}\). The index \(3\) comes
from \(\mathrm{card\_mul\_index}\) with
\(\mathrm{Nat.card\_zpowers}\) and \(|\mathrm{Perm}(\mathrm{Fin}\ 3)|=6\).

**[B9]** The element is outside the subgroup by
\(\mathrm{mem\_zpowers\_iff\_mem\_range\_orderOf}\) with
\(\mathrm{orderOf\_eq\_prime}\) giving order \(2\) — so the least positive
exponent is \(2\) — and its square is \(1\) by decide, hence in the
subgroup. Since the index is \(3\), the final \(\neg(2\mid3)\) is by omega,
so \(k=2\) does not divide the index and part (b)'s conclusion fails. A
3-cycle would not have refuted anything: its least positive exponent is
\(1\) or \(3\), both dividing the index \(3\).
