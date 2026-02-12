# Problem 4 (n>=4) Library Findings from Local MO/MSE Dumps

Date: 2026-02-12

Scope:
- `se-data/mathoverflow.net/Posts.xml` (snapshot 2024-04-06)
- `se-data/math.stackexchange.com/Posts.xml` (snapshot 2024-04-07)

Method:
- Targeted phrase search against question rows (`PostTypeId="1"`), then manual triage.
- Priority terms from `problem4-ngeq4-research-brief.md` Q4.7-Q4.12.

## Executive summary

- Best concrete hit for finite-free algebra structure remains MO `287724` (+ answer `287799`).
- Best hit for additive/compression asymptotic heuristics is MO `454139` (+ answers `454386`, `454391`, `454586`).
- Strong HCIZ/Weingarten references exist in MO (`114267`, `228718`, `256066`, `288059`, `248315`, `419941`).
- No direct local hit found for the core Q4.8 statement
  `1/Phi*(mu boxplus nu) >= 1/Phi*(mu) + 1/Phi*(nu)`,
  and no local derivative-induction theorem hit for Q4.12.

---

## Q4.7: finite-n analog of free entropy

1. What do we actually know about logarithmic energy?
- URL/ID: https://mathoverflow.net/questions/70724
- Date/score: 2011-07-19, score 14
- Why relevant: explicitly links logarithmic energy to free probability/free entropy (Voiculescu framing).
- Relation to Q4.7: conceptual support; no finite-`n` finite-free entropy functional is given.

Assessment:
- Useful for motivation and language, not a direct proof route.

---

## Q4.8: free Fisher information and Phi_n

Targeted local searches for:
- `free Fisher information`
- `free Fisher`
- `free information inequality`
- `free Cramer-Rao`
- `Stam inequality` (with free-probability context)

Result:
- No direct MO/MSE thread located that states or analyzes the specific free-probability inequality needed for Q4.8.
- Returned Cramer-Rao/Stam matches were classical statistics/probability and not finite-free/free-convolution specific.

Assessment:
- Negative finding is strong: local corpus does not currently support Q4.8 directly.

---

## Q4.9: MSS bilinear weights and superadditivity

1. A question about finite free convolution
- URL/ID: https://mathoverflow.net/questions/287724
- Date/score: 2017-12-04, score 2
- Why relevant: exact finite-free convolution setup with expected characteristic polynomial identities.

2. Answer (bilinearity + induction)
- URL/ID: https://mathoverflow.net/a/287799
- Date/score: 2017-12-05, score 5
- Why relevant: explicit use of bilinearity/induction structure; aligns with MSS-weight algebra route.

3. k-th largest root in common interlacing polynomials
- URL/ID: https://mathoverflow.net/questions/162056
- Date/score: 2014-04-01, score 1
- Why relevant: interlacing-families root-control perspective tied to MSS-era machinery.
- Relation to Q4.9: structural background only; no direct sum-of-squares/surplus identity.

Assessment:
- Strong support for algebraic route, but no local post gives a ready-made nonnegative decomposition of the Q4 surplus.

---

## Q4.10: Weingarten / HCIZ approach

1. Integration over the orthogonal group
- URL/ID: https://mathoverflow.net/questions/114267
- Date/score: 2012-11-23, score 11
- Why relevant: explicit orthogonal-group moment integrals; discusses Weingarten limitations/combinatorics.

2. Weingarten function for unitary group
- URL/ID: https://mathoverflow.net/questions/228718
- Date/score: 2016-01-18, score 7
- Why relevant: direct formula and interpretation for unitary Weingarten calculus.

3. Formula for U(N) integration wanted
- URL/ID: https://mathoverflow.net/questions/256066
- Date/score: 2016-11-30, score 11
- Why relevant: explicitly asks for Haar-unitary integral formulas beyond standard symmetric-group character expansions.

4. Haar Measure Integral
- URL/ID: https://mathoverflow.net/questions/288059
- Date/score: 2017-12-09, score 7
- Why relevant: exponential integrals over Haar measure; touches re-summation via Weingarten.

5. Littlewood-Richardson rule and the Harish-Chandra-Itzykson-Zuber integral
- URL/ID: https://mathoverflow.net/questions/248315
- Date/score: 2016-08-26, score 14
- Why relevant: HCIZ-side representation/combinatorics bridge.

6. Harish-Chandra-Itzykson-Zuber integral with two terms
- URL/ID: https://mathoverflow.net/questions/419941
- Date/score: 2022-04-08, score 2
- Why relevant: directly asks about multi-term HCIZ-type generalization/asymptotics.

7. Itzykson-Zuber integral over orthogonal groups (MSE)
- URL/ID: https://math.stackexchange.com/questions/845327
- Date/score: 2014-06-24, score 4
- Why relevant: orthogonal-group IZ integral references and techniques.

Assessment:
- Q4.10 has the strongest local library support after Q4.9.

---

## Q4.11: tight cases / scale separation

1. Free probability: A unitary group heuristic for the relationship between additive free convolution and free compression
- URL/ID: https://mathoverflow.net/questions/454139
- Date/score: 2023-09-07, score 7
- Why relevant: directly studies additive convolution vs compression scaling.

2. Key answers to 454139:
- https://mathoverflow.net/a/454386 (cumulant-scaling dimensional argument)
- https://mathoverflow.net/a/454391 (block-unitary decomposition heuristic)
- https://mathoverflow.net/a/454586 (entry-cumulant cycle scaling explanation)

Relation to Q4.11:
- Provides asymptotic/scaling heuristics highly adjacent to scale-separated extremal behavior.
- Not a direct theorem about the specific `Phi_n` surplus in the extreme-scale regime.

Assessment:
- Best available local analog for scale-separation reasoning.

---

## Q4.12: induction on degree via differentiation

Targeted local searches for:
- `finite free derivative`
- `finite free differentiation`
- `polynomial convolution derivative`
- `Rolle + interlacing + free convolution`
- `boxplus_n`-style terms

Result:
- No direct MO/MSE hit found that states a differentiation/degree-reduction identity for finite free convolution in the needed form.
- Most derivative/interlacing hits were unrelated analysis/calculus threads.

Assessment:
- Negative finding: local forum corpus does not currently provide a clean Q4.12 induction theorem pointer.

---

## Practical use for Claude

Most actionable local references:
1. MO 287724 + 287799 (finite-free bilinearity/induction mechanics).
2. MO 454139 + 454386/454391/454586 (scaling/compression heuristics).
3. MO 114267, 228718, 248315, 419941 (Weingarten/HCIZ technical toolbox).

Hard gaps that remain from local corpus:
1. Q4.8 (free Fisher reciprocal superadditivity in free convolution).
2. Q4.12 (explicit differentiation-driven induction identity for finite free convolution).
