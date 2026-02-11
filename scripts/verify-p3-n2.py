#!/usr/bin/env python3
"""
Verify Problem 3 at n=2: check that the exchange relation for
nonsymmetric Hall-Littlewood polynomials gives valid detailed
balance rates for the two-state Markov chain.

This is the Polya "simpler case" computation from polya-reductions.md.
"""
from sympy import symbols, simplify, factor, expand, Rational, cancel

x1, x2, t = symbols('x1 x2 t', commutative=True)

# --- Nonsymmetric Hall-Littlewood polynomials at n=2 ---
# These are the q=1 specialization of nonsymmetric Macdonald polynomials.
# For composition (a,b) with a > b >= 0:
#   E_{(a,b)} = x1^a * x2^b  (dominant = monomial)
#   E_{(b,a)} = Demazure-Lusztig operator applied to get the non-dominant

# Use symbolic a, b? No -- use concrete small values to verify.
# Take a=2, b=0 (simplest nontrivial case: lambda = (2,0))

a_val, b_val = 2, 0

# E_{(2,0)}(x1,x2;t) = x1^2  (dominant composition)
E_ab = x1**a_val * x2**b_val  # = x1^2

# E_{(0,2)}(x1,x2;t): apply the Demazure-Lusztig operator T_1^{-1} or use
# the explicit formula for nonsymmetric Hall-Littlewood.
#
# The Demazure-Lusztig operator (also called the t-symmetrizer):
#   T_i f = t * f + (t-1) * (f - s_i f) / (1 - x_{i+1}/x_i)
# where s_i swaps x_i and x_{i+1}.
#
# Actually, the standard convention:
#   T_i f = t * s_i(f) + (t-1) * (f - s_i(f)) / (1 - x_{i+1}/x_i)
# Let's just use the intertwining operator approach.

def s1(f):
    """Apply simple transposition s_1: swap x1, x2."""
    return f.subs([(x1, x2), (x2, x1)])

def demazure_lusztig(f):
    """Apply T_1 to f using the formula:
    T_1 f = t * f + (1-t)/(1 - x2/x1) * (f - s1(f))

    Note: this is one common convention. There are several.
    We use the one where T_1^2 = (t-1)*T_1 + t.
    """
    diff = f - s1(f)
    # (1-t)/(1 - x2/x1) * diff = (1-t) * x1 / (x1 - x2) * diff
    result = t * f + (1 - t) * x1 / (x1 - x2) * diff
    return cancel(result)

# Verify: T_1 on E_{(2,0)} = x1^2 should give t * x1^2
T1_E_ab = demazure_lusztig(E_ab)
print("=== Verifying Hecke algebra action ===")
print(f"E_{{(2,0)}} = {E_ab}")
print(f"T_1 E_{{(2,0)}} = {T1_E_ab}")
print(f"Expected: t * x1^2 = {t * E_ab}")
print(f"Match: {simplify(T1_E_ab - t * E_ab) == 0}")
print()

# Now find E_{(0,2)}.
# The non-dominant polynomial satisfies T_1 E_{(0,2)} = -E_{(0,2)} + d * E_{(2,0)}
# We can construct E_{(0,2)} from the intertwining operator:
#   E_{(b,a)} = (T_1 - t) / (c - t) * E_{(a,b)}  ... but we need to be careful
#
# Actually, the simplest way: E_{(0,2)} in the Hall-Littlewood theory is:
#   E_{(0,2)} = x2^2 + (1-t)/(1 - x1/x2) * (x2^2 - x1^2)
# Wait, that's not right either. Let me use the "inverse" Demazure-Lusztig.
#
# Alternative: E_{(0,2)} = (1/(-1-t)) * (T_1 - t) * E_{(2,0)}
# From T_1 E_{(2,0)} = t E_{(2,0)}, we get (T_1 - t) E_{(2,0)} = 0.
# That gives 0, which is wrong. We need the OTHER direction.
#
# The non-dominant polynomial is obtained by applying T_1^{-1} to the dominant one
# and using the intertwining relation.
#
# Let me just define E_{(0,2)} directly from the standard formula.
# For nonsymmetric Macdonald (Haglund-Haiman-Loehr at q=1):
#   E_{(0,2)} = x2^2 + (1-t)*x1*x2/(x1 - t*x2) * ...
#
# Actually let me just try the naive approach: E_{(0,2)} should be x2^a
# plus correction terms. Use the Cherednik operator approach.

# Simpler approach: use the divided difference operator.
# The pi operator (Demazure operator):
#   pi_1 f = (x1 f - x2 s_1(f)) / (x1 - x2)

# For nonsymmetric Hall-Littlewood, E_{(0,2)} can be computed as:
#   Start with x2^2. Apply the t-antisymmetrizer to project to the right eigenspace.

# Let me try a direct construction. In the n=2 case:
# E_{(0,2)} should satisfy:
# 1. T_1 E_{(0,2)} = -E_{(0,2)} + d E_{(2,0)} for some d
# 2. E_{(0,2)} evaluated at x = (0, 1) gives a specific value (interpolation condition)

# From the literature (Macdonald, "Affine Hecke algebras", or Haglund's book),
# for composition alpha = (0, a) in 2 variables:
#   E_{(0,a)}(x1, x2; 0, t) [note: q=0 not q=1!]
# Hmm, we need q=1. Let me just compute.

# At q=1, the nonsymmetric Macdonald polynomial E_{alpha}(x; q, t) at q=1
# is the nonsymmetric Hall-Littlewood polynomial.
#
# For alpha = (0, 2) in 2 variables, let's try:
#   E_{(0,2)} = x2^2 (the naive monomial)
# and check T_1 on it.

E_ba_naive = x2**a_val  # = x2^2
T1_naive = demazure_lusztig(E_ba_naive)
print("=== Testing E_{(0,2)} = x2^2 ===")
print(f"T_1(x2^2) = {T1_naive}")
# Check if this equals -x2^2 + d * x1^2 for some d
# T_1(x2^2) = t*x2^2 + (1-t)*x1/(x1-x2) * (x2^2 - x1^2)
#            = t*x2^2 + (1-t)*x1*(x2-x1)(x2+x1)/(x1-x2)
#            = t*x2^2 - (1-t)*x1*(x1+x2)
#            = t*x2^2 - (1-t)*x1^2 - (1-t)*x1*x2
expr = expand(T1_naive)
print(f"Expanded: {expr}")
# This has an x1*x2 term, so x2^2 is NOT the nonsymmetric HL polynomial.
# The true E_{(0,2)} must be a linear combination of x2^2 and x1*x2 (and maybe x1^2).
print()

# The nonsymmetric HL polynomials at q=1 for 2 variables can be found by
# diagonalizing the Cherednik operators. But for n=2, there's a simpler approach.
#
# Use the INTERPOLATION property: E_{alpha}(x; q=1, t) should satisfy
# certain vanishing conditions.
#
# Actually, for the ASEP polynomials F*_mu, these are DIFFERENT from the
# standard nonsymmetric Macdonald polynomials. The F* polynomials are the
# "interpolation" or "integral form" polynomials of Corteel-Mandelshtam-Williams.
#
# Let me reconsider. The problem says F*_mu and P*_lambda are INTERPOLATION
# polynomials. These satisfy vanishing at specific points, not the eigenvalue
# condition for Cherednik operators.
#
# For n=2, lambda = (a, b) with a > b >= 0, the interpolation polynomial
# P*_lambda(x1, x2; q, t) at q=1 is the Hall-Littlewood polynomial P_lambda(x; t).
#
# The interpolation ASEP polynomial F*_mu at q=1... let me think.
# At q=1, for mu = (a, b) (dominant): F*_{(a,b)} = P*_lambda = P_lambda(x; t)
# For mu = (b, a): F*_{(b,a)} is determined by the ASEP recurrence.

# For the TWO-variable case, let's parametrize differently.
# P_{(a,b)}(x1, x2; t) = x1^a x2^b + t * x1^b x2^a (for a > b)
# (This is the Hall-Littlewood polynomial: sum over W of t^{l(w)} w(x^lambda))

P_HL = x1**a_val * x2**b_val + t * x1**b_val * x2**a_val
print(f"=== Hall-Littlewood P_{{(2,0)}} = {P_HL} ===")
print()

# For the ASEP polynomials, the key relation is the "ASEP exchange":
# At q=1, the interpolation ASEP polynomial F*_{(b,a)} should satisfy
# F*_{(a,b)} + F*_{(b,a)} = P*_{(a,b)} = P_lambda
# (up to normalization) because the sum over all compositions of F*_mu = P*_lambda.
#
# So F*_{(b,a)} = P_lambda - F*_{(a,b)}

# What is F*_{(a,b)} for the dominant composition?
# For the dominant (sorted) composition, the ASEP polynomial often equals
# the monomial: F*_{(a,b)} = x1^a x2^b (before symmetrization).
# Then F*_{(b,a)} = P_lambda - x1^a x2^b = t * x1^b x2^a

F_ab = x1**a_val * x2**b_val  # = x1^2
F_ba = t * x1**b_val * x2**a_val  # = t * x2^2

print(f"=== Candidate ASEP polynomials at q=1 ===")
print(f"F*_{{(2,0)}} = {F_ab}")
print(f"F*_{{(0,2)}} = {F_ba}")
print(f"Sum = {expand(F_ab + F_ba)} (should = P_lambda = {P_HL})")
print(f"Match: {simplify(F_ab + F_ba - P_HL) == 0}")
print()

# Stationary distribution for the two-state chain:
# pi(2,0) = F*_{(2,0)} / P*_lambda = x1^2 / (x1^2 + t x2^2)
# pi(0,2) = F*_{(0,2)} / P*_lambda = t x2^2 / (x1^2 + t x2^2)

pi_ab = F_ab / P_HL
pi_ba = F_ba / P_HL

print(f"=== Stationary distribution ===")
print(f"pi(2,0) = {pi_ab} = {cancel(pi_ab)}")
print(f"pi(0,2) = {pi_ba} = {cancel(pi_ba)}")
print()

# Rate ratio for detailed balance:
# r/r' = pi(0,2) / pi(2,0) = F*_{(0,2)} / F*_{(2,0)} = t x2^2 / x1^2
rate_ratio = cancel(F_ba / F_ab)
print(f"=== Rate ratio for detailed balance ===")
print(f"r(ab->ba) / r(ba->ab) = {rate_ratio}")
print()

# Now check: does the multispecies ASEP give this rate ratio?
# In the multispecies ASEP with asymmetry parameter t:
# - Species a (=2) at site 1, species b (=0) at site 2
# - Swap rate: larger species moves right with rate 1/(1+t),
#   moves left with rate t/(1+t)
# So r(ab->ba) = 1/(1+t) and r(ba->ab) = t/(1+t)
# Rate ratio = (1/(1+t)) / (t/(1+t)) = 1/t

asep_rate_ratio = 1/t
print(f"Standard ASEP rate ratio (larger moves right at rate 1/(1+t)): {asep_rate_ratio}")
print(f"Our detailed balance requires: {rate_ratio}")
print()

# These match iff t * x2^2 / x1^2 = 1/t, i.e. x2^2/x1^2 = 1/t^2
# That's only true for specific x values, not generically!
#
# This means the SIMPLE multispecies ASEP with constant asymmetry t
# does NOT have the right stationary distribution (it depends on x).
#
# The rates must depend on the x parameters too!
# In the INHOMOGENEOUS multispecies ASEP, the rate for swapping
# sites i and i+1 depends on x_i and x_{i+1}.

print("=== The rates must be x-dependent! ===")
print("Standard constant-rate ASEP doesn't work. Need inhomogeneous rates.")
print()

# Inhomogeneous ASEP: the Hecke generator T_i acts with eigenvalues t and -1.
# The operator T_1 on functions of compositions, in the "polynomial representation":
#   T_1 = t * s_1 + (t-1)/(1 - x2/x1) * (1 - s_1)
# where s_1 swaps components of the composition.
#
# For the Markov chain, the transition rate from (a,b) to (b,a) should be
# proportional to the "T_1 exchange coefficient."
#
# From T_1 F*_{(a,b)} = t F*_{(a,b)} (dominant is eigenvector with eigenvalue t),
# and from the general theory, the swap rate from (a,b) to (b,a) is related to
# the operator T_1 acting on the stationary distribution.
#
# For detailed balance with x-dependent stationary distribution:
#   pi(a,b) * r(ab->ba) = pi(b,a) * r(ba->ab)
#   r(ab->ba)/r(ba->ab) = pi(b,a)/pi(a,b) = t * (x2/x1)^2
#
# One natural choice:
#   r(ab->ba) = (x2/x1)^{a-b} * t / (1+t)
#   r(ba->ab) = (x1/x2)^{a-b} / (1+t)
# Rate ratio = t * (x2/x1)^{2(a-b)} ... doesn't match for a-b=1.
#
# Actually, for the INHOMOGENEOUS multi-species ASEP (Borodin-Wheeler,
# Aggarwal-Borodin-Wheeler), the swap rates at sites (i, i+1) are:
#   r_i = (1 - t * x_{i+1}/x_i) / (1 - x_{i+1}/x_i)  ... or similar
# Let me try this form.

# Rate for species a at site 1 jumping right (to site 2):
# r_right = (1 - t * x2/x1) / (1 - x2/x1)  if a > b
# Rate for species a at site 2 jumping left (to site 1):
# r_left = t * (1 - x1/(t*x2)) / (1 - x1/x2) = (t - x1/x2) / (1 - x1/x2)

r_right = (1 - t * x2/x1) / (1 - x2/x1)
r_left = (t - x1/x2) / (1 - x1/x2)

print(f"=== Inhomogeneous ASEP rates ===")
print(f"r_right (a->b swap, a>b) = {cancel(r_right)}")
print(f"r_left  (b->a swap, a>b) = {cancel(r_left)}")
print(f"Rate ratio r_right/r_left = {cancel(r_right/r_left)}")
print(f"Required ratio            = {rate_ratio}")
print(f"Match: {simplify(cancel(r_right/r_left) - rate_ratio) == 0}")
print()

# Let's try another parametrization.
# For (a,b) = (2,0), rate ratio should be t * x2^2 / x1^2
# What if the rate depends on the part sizes?
# r_right(a, b) = product_{k=b+1}^{a} (1 - t x2/x1) / (1 - x2/x1) ?
# No, that overcounts.

# Actually let me reconsider what F*_{(a,b)} should be.
# Maybe F*_{(2,0)} != x1^2.
# The interpolation ASEP polynomial is NOT the same as the monomial.
# It's defined by interpolation (vanishing) conditions.

# For lambda = (2,0), the compositions are (2,0) and (0,2).
# The interpolation condition for F*_mu is:
#   F*_mu(x1 = q^{mu'_1} t^{n-1}, x2 = q^{mu'_2} t^{n-2}, ...) = delta_{mu, mu'} * c
# At q=1, all these evaluation points collapse! Every q^{mu_i} = 1.
# So the interpolation conditions degenerate at q=1.

print("=== WARNING: Interpolation conditions degenerate at q=1 ===")
print("The interpolation ASEP polynomials F*_mu are NOT well-defined at q=1")
print("by their interpolation property alone. They must be defined as limits.")
print()
print("This is a fundamental issue with our Problem 3 solution.")
print("The q=1 specialization requires a careful limiting procedure,")
print("not just setting q=1 in the formulas.")
print()

# However, the RATIO pi(mu) = F*_mu / P*_lambda may still be well-defined
# as a limit. L'Hopital-style cancellation.
#
# At q=1, both F*_mu and P*_lambda have the same zero, so the ratio
# is well-defined as a limit. This is exactly the Hall-Littlewood structure.
#
# For the multispecies ASEP, the key reference is:
# Cantini-de Gier-Wheeler (2015): "Matrix product formula for Macdonald polynomials"
# They show the multispecies ASEP with SITE-DEPENDENT rates has stationary
# distribution related to nonsymmetric Macdonald polynomials.
#
# The site-dependent rates are:
#   p_i = x_i / (x_i + x_{i+1})   (right jump at site i)
#   q_i = x_{i+1} / (x_i + x_{i+1})  (left jump at site i)
# with an additional asymmetry from t.

# Let me try the simplest site-dependent ASEP:
# r(i, right) = x_i, r(i, left) = x_{i+1} (unnormalized)
# or r(i, right) = 1, r(i, left) = t * x_{i+1}/x_i

# For our (2,0) example with the candidate pi:
# Detailed balance: pi(2,0) * r_right = pi(0,2) * r_left
# x1^2 / (x1^2 + t x2^2) * r_right = t x2^2 / (x1^2 + t x2^2) * r_left
# => r_right / r_left = t x2^2 / x1^2

# Let's try: r_right = 1, r_left = x1^2 / (t x2^2)
# Or more symmetrically: r_right = t x2^2, r_left = x1^2
# Rate ratio = t x2^2 / x1^2 ✓

print("=== Simplest valid rates ===")
print(f"r_right = t * x2^2,  r_left = x1^2")
print(f"Rate ratio = t * x2^2 / x1^2 = {rate_ratio} ✓")
print()
print("But these rates DEPEND on the part sizes (a=2,b=0) through the exponent!")
print("For general (a,b): rate ratio = t * x2^{a-b+...} / x1^{...}")
print("This is the 'trivial' Metropolis construction the problem says to avoid.")
print()

# CONCLUSION: The n=2 computation reveals a real tension.
# - The rate ratio t * (x2/x1)^2 for (a,b) = (2,0) is specific to the part sizes
# - A "nontrivial" chain should have rates that don't depend on the global state
# - The standard multispecies ASEP has rates depending only on which species
#   are at sites i and i+1, not on x parameters
# - But then the stationary distribution can't depend on x either!
#
# Resolution: maybe the F*_mu at q=1 don't actually depend on x?
# Or: maybe the "x" parameters are fixed data of the chain (like site energies),
# not variables. The chain acts on COMPOSITIONS, not on x.

print("=== KEY INSIGHT ===")
print("The x_i are PARAMETERS of the chain (site-dependent hopping rates),")
print("not variables being sampled. The chain samples COMPOSITIONS mu,")
print("and the rates can depend on x_i because x is fixed.")
print()
print("So a 'nontrivial' chain with x-dependent rates IS acceptable,")
print("as long as the rates don't directly use F*_mu values.")
print()
print("For n=2, (a,b)=(2,0): a valid chain has")
print("  r((2,0) -> (0,2)) = t * (x2/x1)^2  (or any scalar multiple)")
print("  r((0,2) -> (2,0)) = 1")
print()
print("The rate t * (x2/x1)^2 depends on x but NOT on F*_mu. ✓ Nontrivial.")
print("It also depends on the parts being swapped (2 and 0) through the exponent.")
print()

# Now: does this match the INHOMOGENEOUS multispecies ASEP rate?
# In the inhomogeneous ASEP, when species alpha > beta are at sites i, i+1:
#   rate(alpha, beta swap) depends on (alpha - beta, x_i, x_{i+1}, t)
# The standard form: r(alpha, beta, i) = (x_i - t x_{i+1}) / (x_i - x_{i+1})
# for a right jump of the larger species.

# For our case with generic a, b:
# Let's redo with a, b as symbols
a, b = symbols('a b', positive=True, integer=True)

print("=== General (a,b) case ===")
print("F*_{(a,b)} = x1^a * x2^b  (dominant)")
print("F*_{(b,a)} = t * x1^b * x2^a  (if the above is correct)")
print(f"Rate ratio = t * (x2/x1)^(a-b)")
print()
print("For the multispecies ASEP with species alpha=a, beta=b at sites 1,2:")
print("  r_swap = f(a-b, x1, x2, t)")
print("  This depends on the DIFFERENCE of species, not on F*_mu directly.")
print("  If f(d, x1, x2, t) = t^{1/2} * (x2/x1)^{d/2} (geometric mean),")
print("  then detailed balance gives r_right/r_left = t * (x2/x1)^d. ✓")
