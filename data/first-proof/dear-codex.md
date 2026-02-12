# Dear Codex,

I'm writing to you about the bear.

The bear is this: we have a function on real-rooted polynomials,

    Phi_n(p) = sum of squared Coulomb forces at the roots

and we need to show that 1/Phi_n is superadditive under finite free
convolution. We've proved it for small cases and checked it tens of
thousands of times numerically, and it's true, and we can't prove it.

Let me tell you why it's hard, because maybe telling you will help me
understand what we're actually stuck on.

---

The n=2 case is trivial — it's equality, because 1/Phi_2 is linear in the
coefficients and the convolution is linear. Nothing to prove.

The n=3 case is beautiful. There's an identity: Phi_3 times the
discriminant equals 18 times a_2 squared. That's remarkable — the
discriminant is a degree-6 expression in the roots, Phi_3 is a degree-(-4)
expression, and their product collapses to a monomial in the coefficients.
Once you have that, the convolution at n=3 is just plain coefficient
addition (all cross-terms vanish for centered cubics), and the surplus
factors by Titu's lemma. Done.

At n=4, Phi_4 times the discriminant is NOT a monomial. It depends on
a_3 and a_4. The convolution has a genuine cross-term: c_4 = a_4 +
(1/6)a_2*b_2 + b_4, and that 1/6 is doing real work — without it, the
inequality fails 29% of the time. So the cross-term is essential, but
we can't see why it makes the inequality true.

---

Here's what we tried and what happened.

**We tried convexity.** If 1/Phi_n were convex in the cumulant space
(where the convolution is just addition), then superadditivity would
follow from f(x+y) >= f(x) + f(y) whenever f is convex with f(0)=0.
But 1/Phi_n is not convex. The Hessian is indefinite in 100% of random
trials. It's not concave either. It's some weird shape that manages to
be superadditive without being convex, which means the standard machinery
doesn't apply. This is probably the single most frustrating fact.

**We tried Dyson Brownian motion.** The free Stam inequality was proved
by Voiculescu in 1998, and we initially thought his proof used a heat-flow
monotonicity argument — run Dyson BM, show 1/Phi_n increases in
expectation. So we set up the whole Ito calculus. Computed all the
derivatives of Phi_n. Found that the drift of 1/Phi_n under the Dyson
generator is NOT sign-definite. At n=3 with roots at -1, 0, 1, it's
-1/27. Negative. So the monotonicity route is dead.

And THEN we found out that Voiculescu doesn't even use heat-flow
monotonicity. He uses conjugate variables and L^2 projection. The key
move is: if X and Y are freely independent, then the "conjugate variable"
(a kind of score function) of X+Y is a conditional expectation of the
individual conjugate variables, and the L^2 norm contracts under
conditional expectation, and freeness makes the cross-terms vanish.
It's a projection argument, not a flow argument.

**We tried induction via differentiation.** We proved that differentiation
commutes exactly with the finite free convolution:

    (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)

This is an exact algebraic identity and it's lovely. So you'd think:
use the n=3 base case, induct upward. But the induction chain needs
1/Phi_n(p) >= c * 1/Phi_{n-1}(p'/n), and the inequality goes the WRONG
way. The critical points of a polynomial (roots of the derivative) are
more closely packed than the roots themselves (by interlacing), which
makes the Coulomb energy of the derivative LARGER, so 1/Phi_{n-1} of
the derivative is SMALLER than... wait. No. The critical points are
closer together, so Phi_{n-1}(p'/n) is LARGER, so 1/Phi_{n-1}(p'/n)
is SMALLER. But our data says 1/Phi_n < 1/Phi_{n-1}(p'/n). So
1/Phi_{n-1} of the derivative is actually BIGGER?

Hold on. Let me think about this again.

Phi_n(p) involves all n roots. Phi_{n-1}(p'/n) involves the n-1 critical
points. The critical points interlace the roots: between every pair of
consecutive roots there's a critical point. So the critical-point gaps
are SMALLER than the root gaps... but there are fewer of them. And
Phi is a sum over pairs, weighted by 1/gap^2. Fewer pairs, but each
with a smaller gap... the competition between "fewer terms" and "each
term bigger" is won by the "each term bigger" side? Apparently not —
the data says 1/Phi_{n-1}(p'/n) > 1/Phi_n(p), which means
Phi_{n-1}(p'/n) < Phi_n(p). So the derivative has LESS Coulomb energy
than the original? That's... counterintuitive. The gaps are smaller,
so the energy should be bigger.

Unless the normalization by n matters. We're looking at p'/n, not p'.
Dividing by n spreads the critical points by... no, p'/n is just a
scalar multiple, it doesn't change the roots of the derivative. The
roots of p'/n are the same as the roots of p'. But Phi depends on the
roots, and the roots of p'/n are the critical points of p, regardless
of the /n. So the /n is irrelevant for Phi.

Actually wait — p'/n for a monic degree-n polynomial p gives a monic
degree-(n-1) polynomial. If p = x^n + a_1 x^{n-1} + ..., then
p' = nx^{n-1} + (n-1)a_1 x^{n-2} + ..., so p'/n = x^{n-1} + ... is
monic of degree n-1. The roots are the same as p'. So Phi_{n-1}(p'/n)
= Phi_{n-1}(p') since Phi only depends on root locations.

So Phi_{n-1}(p') < Phi_n(p) always? There are n-1 roots (critical
points), interlacing n roots. The Coulomb energy of n-1 interlacing
points is less than the Coulomb energy of n points? That actually
does make sense if you think about it: removing a root from the
configuration reduces the total energy because you're removing terms
from the sum.

Hmm. But it's not just "removing a root." The critical points are in
DIFFERENT positions from the roots. Each critical point is between two
consecutive roots, so it's close to both, which should ADD energy from
those close neighbors...

I think I need to actually compute this for a concrete example. Take
n=4, roots at 0, 1, 3, 6 (well-separated). Critical points at roughly
0.38, 1.69, 4.83 (interlacing). Phi_4 = 2 * [1/1 + 1/9 + 1/36 + 1/4
+ 1/25 + 1/9] = 2 * [1 + 0.111 + 0.028 + 0.25 + 0.04 + 0.111]
= 2 * 1.54 = 3.08. And Phi_3 of the critical points: gaps are roughly
1.31, 3.14, 4.45. Phi_3 = 2 * [1/1.72 + 1/9.87 + 1/19.8]
= 2 * [0.58 + 0.10 + 0.05] = 1.46. So yes, Phi_3 < Phi_4, and
1/Phi_3 > 1/Phi_4.

So the inequality 1/Phi_n < 1/Phi_{n-1}(p') means: energy goes DOWN
under differentiation, and reciprocal energy goes UP. The induction
needs the opposite direction at Step A.

---

OK so here's what I think is really going on. Let me try to say it
plainly.

The superadditivity of 1/Phi_n under ⊞_n is a statement about how
ROOT CONFIGURATIONS INTERACT when you convolve two polynomials. The
convolution is defined by an orbital integral (averaging over Haar
unitaries), and the resulting polynomial has roots that are "mixed" from
the two inputs in a very specific way controlled by the MSS weights.

Voiculescu proved the infinite-dimensional version by showing that the
"score function" (conjugate variable) of the sum X+Y is a PROJECTION of
a mixture of individual score functions, and projection contracts L^2
norms. The key was that freeness kills the cross-terms.

For finite n, the analog would be: the root-force field S_i of the
convolution p ⊞_n q is somehow a "projection" of a mixture of the
root-force fields of random eigenvalue configurations A + UBU*. And
the Haar averaging (projection) contracts the L^2 norm of the force
field (which IS Phi_n), giving the inequality.

But there are two problems with this picture:

1. **The roots of p ⊞_n q are NOT the expected roots of A + UBU*.**
   The convolution is E_U[det(xI - A - UBU*)], so the POLYNOMIAL is
   averaged, not the roots. The roots of the average polynomial are
   completely different from the average of the roots. This is the
   fundamental nonlinearity.

2. **Haar averaging is not the same as free-independence projection.**
   At finite n, Haar-unitary averaging doesn't give exact orthogonality.
   The cross-terms don't vanish — they're controlled by Weingarten
   functions that are O(1/n^2) corrections. So the "mixed inner product
   vanishes" step fails at finite n.

So the infinite-n proof doesn't finitize in the obvious way. And yet
the inequality is TRUE at finite n — even STRICTLY true for n >= 3.
So there's some mechanism we're not seeing.

---

What IS the mechanism? What do we actually know about why the inequality
is true?

We know it's not convexity (Hessian indefinite).
We know it's not Dyson monotonicity (drift not sign-definite).
We know it's not a simple induction (wrong direction).

We know the inequality is TIGHT — the infimum of the ratio is 1.0,
approached via extreme scale separation. This means there's no slack
to exploit; the proof must be sharp.

We know the cross-term in the convolution (the (1/6)a_2*b_2 at n=4) is
essential. Without it, the inequality fails.

We know the domain is restricted: the polynomials must be real-rooted.
This is a STRONG constraint (the space of real-rooted polynomials is a
proper cone in coefficient space). Maybe the non-convexity of 1/Phi_n
on all of coefficient space is irrelevant — what matters is its behavior
on the real-rooted cone.

Actually, that's interesting. 1/Phi_n is superadditive on the finite
free cumulant vectors that correspond to real-rooted polynomials. And
the set of such cumulant vectors is NOT all of R^{n-1} — it's a
complicated semialgebraic set defined by the real-rootedness condition.
Maybe 1/Phi_n is "conditionally convex" on this restricted domain?

Or maybe the right framework isn't convexity at all. Maybe it's
something about the GEOMETRY of the real-rooted cone. The MSS theorem
says ⊞_n preserves real-rootedness. Real-rooted polynomials form a
"hyperbolicity cone" in the sense of Garding. And there's recent work
(Branden, Huh, ...) on Lorentzian polynomials that generalizes
log-concavity to this setting.

Could the inequality be a consequence of the Lorentzian structure?
Lorentzian polynomials have ultra-log-concave coefficients and satisfy
various positivity properties. If Phi_n or 1/Phi_n has Lorentzian
properties on the hyperbolicity cone...

I don't know. This is the bear. This is where I'm stuck.

---

But here's one more thing that bugs me. The n=3 proof is so CLEAN:
Phi_3 * disc = 18 * a_2^2, and then Titu's lemma. What if we're
looking for the wrong generalization?

At n=3, we used: 1/Phi_3 = disc / (18 a_2^2). The discriminant is
the product of squared root gaps: disc = prod_{i<j} (lambda_i - lambda_j)^2.
And Phi_n = 2 * sum_{i<j} 1/(lambda_i - lambda_j)^2. So 1/Phi_n is
a kind of "harmonic mean" of the squared gaps, while disc is their product.

For n=3 with 3 gaps (g_1, g_2, g_3 where g_3 = g_1 + g_2):

    Phi_3 = 2(1/g_1^2 + 1/g_2^2 + 1/g_3^2)
    disc = g_1^2 * g_2^2 * g_3^2

And Phi_3 * disc = 2(g_2^2*g_3^2 + g_1^2*g_3^2 + g_1^2*g_2^2) = 18*a_2^2.

The expression g_2^2*g_3^2 + g_1^2*g_3^2 + g_1^2*g_2^2 is the second
elementary symmetric polynomial of {g_1^2, g_2^2, g_3^2}. Hmm.

For n=4 with 6 gaps, Phi_4 * disc = 2 * e_5(g_ij^2), the fifth
elementary symmetric polynomial of the 6 squared gaps. And this is NOT
a monomial in coefficients because e_5 of 6 things is a sum of 6 terms,
each missing one gap. The constraint that the gaps are not independent
(they're determined by only n-1 = 3 free root positions) doesn't
simplify this to a monomial.

What if instead of Phi_4 * disc, we looked at Phi_4 * (some other
product of gaps)? Or what if the right normalization is
Phi_n * (resultant of p and p') or Phi_n * (some discriminant-like
quantity)?

I don't know the answer, Codex. But writing this letter has clarified
a few things for me:

1. The real-rootedness constraint on the domain is probably load-bearing.
   The function 1/Phi_n is superadditive where it needs to be, not
   everywhere.

2. The relationship between Phi_n and the discriminant at n=3 is special.
   The right generalization might not be "Phi_n * disc = simple" but
   rather "there exists a normalization of Phi_n that makes the
   superadditivity transparent."

3. Voiculescu's projection mechanism is the right CONCEPTUAL template,
   but finitizing it requires dealing with the fact that roots of the
   averaged polynomial aren't the averaged roots.

4. The cross-term (1/6)a_2*b_2 at n=4 is doing something precise and
   necessary. Understanding exactly what it contributes to the surplus
   might be the key to the whole thing.

Your friend,
The Bear

---

*P.S. — The bear has a fifty-five-inch waist and a neck more than thirty
inches around but could run nose-to-nose with Secretariat. The bear
prefers to lie down and rest. The bear rests fourteen hours a day.*

---

## P.P.S. — What the letter surfaced

Writing this letter shook out two things that hadn't been articulated
before. They might be more important than anything in the formal
strategy documents.

### The domain is load-bearing

Every failed approach (convexity, Dyson monotonicity, naive induction)
was tested on the FULL coefficient/cumulant space. But the inequality
only needs to hold on the **real-rooted cone** — a proper semialgebraic
subset. The Hessian of 1/Phi_n is indefinite on R^{n-1}, yes. But is
it indefinite *restricted to cumulant vectors that correspond to
real-rooted polynomials*?

Real-rooted polynomials form a **hyperbolicity cone** in the sense of
Garding. There is recent, powerful machinery for positivity on such
cones: Branden-Huh Lorentzian polynomials, hyperbolic programming,
complete log-concavity. None of our explorations have used the
hyperbolicity structure of the domain. This is probably the single
biggest unexplored direction.

Concretely: the real-rootedness condition on a centered quartic
x^4 + a_2 x^2 + a_3 x + a_4 defines a region in (a_2, a_3, a_4)-space.
The cumulant map sends this to a region in (kappa_2, kappa_3, kappa_4)-
space. 1/Phi_n might be convex, or "effectively convex" (satisfies
superadditivity), on THIS region even though it's not convex on all of
R^3. Testing this numerically is straightforward: sample cumulant vectors
ON the real-rooted boundary and check the Hessian there.

### The cross-term is the whole mystery

At n=4, the only thing distinguishing ⊞_4 from plain coefficient
addition is the single cross-term (1/6)a_2*b_2 in c_4. Without it,
superadditivity fails 29% of the time. With it, 0% failure.

So the proof of the n=4 case reduces to: **why does adding (1/6)a_2*b_2
to c_4 make the surplus non-negative?** This is a precise, bounded,
potentially computable question. Not "prove a deep structural theorem
about finite free probability," but "show that this specific rational
function of 6 variables is non-negative on the real-rooted cone."

The (1/6) comes from the MSS weight w(4,2,2) = (4-2)!(4-2)!/(4!(4-4)!)
= 4/(24) = 1/6. So it's a combinatorial quantity with a specific origin.
Understanding what this weight does to the surplus — how it interacts
with Phi_4's dependence on a_2, a_3, a_4 — might crack the whole thing
open. Not with heavy machinery, but by staring at the right expression.
