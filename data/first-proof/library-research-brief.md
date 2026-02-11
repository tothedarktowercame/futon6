# Library Research Brief: First Proof Problems 3 & 5

**Purpose:** Targeted search over math.stackexchange and mathoverflow corpus
to stress-test our low-confidence solutions. Formulated as the questions we
would ask if posting to these sites.

---

## Problem 5: O-Slice Connectivity via Geometric Fixed Points

### What we claimed

The O-slice filtration for an incomplete transfer system O is built from
O-slice cells G_+ wedge_H S^{n rho_H^O} (H in T_O), and O-slice connectivity
of a connective G-spectrum X is characterized by:

    Phi^H X is (n * d_H^O - 1)-connected for all H in T_O

where d_H^O = dim_R(rho_H^O) / |H|.

### Why we're uncertain

The characterization may be tautological — it might just be restating the
definition of O-slice connectivity in different notation. The real question
the problem authors are asking might be deeper: perhaps there is a *surprising*
characterization, analogous to how the HHR slice theorem was non-obvious
when first proved.

### Research Questions (search these)

**Q5.1: Does the incomplete slice filtration already exist in the literature?**

Search terms:
- "incomplete transfer system" "slice filtration"
- "N-infinity operad" "slice"
- "incomplete" "slice cells"
- Blumberg + Hill + slice (Blumberg is a problem author AND a key figure in
  equivariant homotopy theory — this is not a coincidence)

What we're looking for: Has anyone defined the O-slice filtration? If so,
what is their characterization theorem? Our answer might be wrong if the
actual definition uses different slice cells.

**Q5.2: What is the "right" notion of O-slice cell?**

Search terms:
- "slice cell" "regular representation" incomplete
- "slice cell" "N-infinity" OR "N_infty"
- Hill Hopkins Ravenel "slice" "representation sphere"

The crux: in the complete case, slice cells are G_+ wedge_H S^{n rho_H}
where rho_H is the regular representation. For incomplete transfer systems,
what replaces rho_H? We guessed "rho_H^O = the O-regular representation"
but this might be wrong. Alternatives:
- Still use full rho_H but restrict which H appear
- Use a representation determined by the N-infinity operad structure
- Use the "real regular representation of the transfer system" (whatever
  that means)

**Q5.3: Is the geometric fixed point characterization non-trivial?**

Search terms:
- "geometric fixed points" "slice connectivity" characterization
- "slice" "connective" "if and only if" "geometric fixed points"
- Phi^H "slice" theorem

In the HHR case, the theorem Phi^H X is (n|G/H|-1)-connected iff X is
slice >= n is genuinely useful because one side is about equivariant
homotopy and the other is about ordinary homotopy of fixed points. Is
there an analogous non-trivial reduction for the O-slice case? Or does
the incomplete transfer system make the fixed-point side just as
complicated as the equivariant side?

**Q5.4: What role does Blumberg play?**

Search terms:
- Blumberg "transfer system" OR "N-infinity"
- Blumberg Hill "incomplete"
- Blumberg "equivariant commutative ring spectra"

Blumberg is both a First Proof problem author and a leading figure in
this area. His recent work with Hill on N-infinity operads and transfer
systems likely contains the answer or at least the right framework.
Finding MO answers by Blumberg about transfer systems would be extremely
high value.

---

## Problem 3: ASEP Markov Chain with Interpolation Polynomial Stationary Distribution

### What we claimed

A multispecies exclusion process with Hecke algebra transition rates
(T_i generators, rates t/(1+t) and 1/(1+t)) has stationary distribution
given by F*_mu(x; q=1, t) / P*_lambda(x; q=1, t).

### Why we're uncertain

1. We never verified the exchange relations for interpolation ASEP polynomials
   at q=1 actually match the Hecke algebra rates
2. The "nontriviality" condition ("transition probabilities are not described
   using the F*_mu polynomials") is subtle — our rates are rational functions
   of x_i and t, but they might secretly encode F*_mu values
3. The connection between interpolation ASEP polynomials and the standard
   multispecies ASEP literature (Cantini-de Gier-Wheeler, Corteel-Mandelshtam-
   Williams) needs verification

### Research Questions (search these)

**Q3.1: What is the relationship between interpolation ASEP polynomials and the multispecies ASEP?**

Search terms:
- "interpolation ASEP polynomial"
- "multispecies ASEP" "stationary distribution" polynomial
- "multispecies exclusion process" "Macdonald"
- Corteel Mandelshtam Williams ASEP

What we're looking for: explicit statements about which exclusion process
has which polynomial family as its stationary distribution. Especially at
q=1. The Corteel-Mandelshtam-Williams paper introduced the interpolation
ASEP polynomials — do they already give a Markov chain?

**Q3.2: Exchange relations for interpolation ASEP polynomials**

Search terms:
- "interpolation" "exchange relation" ASEP
- "nonsymmetric Macdonald" "exchange" "Hecke"
- "interpolation polynomial" T_i OR "Hecke algebra"
- "Knop-Sahi" interpolation

The exchange relations T_i F*_mu = c F*_mu + d F*_{s_i mu} are the core
mechanism. At q=1 these should simplify dramatically. We need to know:
- What are c and d explicitly at q=1?
- Do they factor as rational functions of (mu_i, mu_{i+1}, t, x_i, x_{i+1})?
- Is the detailed balance for these coefficients obvious or does it require work?

**Q3.3: What does "nontrivial" mean precisely?**

Search terms:
- "nontrivial Markov chain" "stationary distribution" "polynomial"
- site:arxiv.org 2602.05192 (to find discussions of the paper)
- "First Proof" benchmark math

The problem says the chain should be "nontrivial" meaning transition
probabilities aren't described using the F*_mu polynomials. This is a
subtle condition. A chain that swaps adjacent entries with rate
F*_{s_i mu} / F*_mu would be "trivial" (it's just the Metropolis
algorithm applied to the target distribution). Our Hecke algebra rates
don't explicitly involve F*_mu, but the Hecke algebra IS the algebra
that DEFINES these polynomials. Is that circular?

**Q3.4: Hall-Littlewood specialization of the multispecies ASEP**

Search terms:
- "Hall-Littlewood" "exclusion process"
- "multispecies ASEP" q=1 OR "Hall-Littlewood"
- "nonsymmetric Hall-Littlewood" stationary

At q=1, interpolation Macdonald polynomials become interpolation
Hall-Littlewood polynomials. The ASEP should simplify to a more classical
object. Is the resulting Markov chain already known in the literature?

---

## Search Strategy Notes for Codex

1. **MathOverflow first**: research-level questions about these topics are
   more likely on MO than math.SE. Prioritize MO results.

2. **Author search**: Both problems involve authors of the First Proof paper.
   - Problem 5: Blumberg (equivariant homotopy theory)
   - Problem 3: Williams (algebraic combinatorics, Macdonald polynomials)
   Search for MO posts/answers by these authors.

3. **Date range**: These are current research topics. Posts from 2020-2026
   are most relevant. Pre-2020 posts establish the classical theory.

4. **What counts as a hit**: We don't need someone to have solved our exact
   problem. What helps most is:
   - A definition of O-slice cells that confirms or contradicts ours
   - An exchange relation formula at q=1 we can verify
   - A known Markov chain for Hall-Littlewood polynomials
   - Any post by Blumberg or Williams about these topics

5. **Return format**: For each hit, return:
   - Post title, URL/ID, author, date
   - Key claim or definition (1-2 sentences)
   - How it relates to our question (confirm / contradict / extend)
