#!/usr/bin/env python3
"""FM-001b: IP encoding for 2-block-circulant Ramsey witness on Z_q.

Finds D11, D12 ⊆ Z_q such that Γ_{Z_q}(D11, D12, D̄11) is
(B_{n-1}, B_n)-free, proving R(B_{n-1}, B_n) ≥ 4n-1.

Uses Wesley's complement ansatz: D22 = Z_q\{0}\D11.
Symmetry: D11 = -D11, D12 symmetric under negation mod q.

Usage:
    python3 FM-001b-ip-encoding.py [n]
    Default n=50 (q=99). Finds witness for R(B_49, B_50) ≥ 199.

Requires: PuLP (pip install pulp) or scipy. Falls back to brute CBC.
"""

import sys
from itertools import product


def solve_block_circulant(n, solver_name=None):
    """Find 2-block-circulant (B_{n-1}, B_n)-free graph on Z_{2n-1}."""
    try:
        import pulp
    except ImportError:
        print("ERROR: pip install pulp")
        sys.exit(1)

    q = 2 * n - 1
    r = n - 1  # B_{n-1}-free: CN < r for G-edges
    s = n      # B_n-free: CN < s for complement edges

    print(f"n={n}, q={q}, vertices=2q={2*q}")
    print(f"Need: G-edge CN < {r}, complement CN < {s}")

    prob = pulp.LpProblem(f"FM001b_q{q}", pulp.LpMinimize)

    # Binary variables
    # x[i] = 1 iff i ∈ D11, for i = 1..q-1 (x[0] = 0 always)
    # z[i] = 1 iff i ∈ D12, for i = 0..q-1
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(1, q)}
    z = {i: pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(q)}

    # Symmetry: x[i] = x[q-i], z[i] = z[q-i]
    for i in range(1, (q + 1) // 2):
        j = q - i
        if j != i:
            prob += x[j] == x[i], f"sym_x_{i}"
            prob += z[j] == z[i], f"sym_z_{i}"

    # Complement ansatz: D22[i] = 1 - x[i] for i != 0
    # D̄11 = D22, D̄12 = {i : z[i] = 0}

    # Helper: Δ(X, Y, d) for indicator variables
    # Δ(X, Y, d) = Σ_i X[i+d] * Y[i]  (indices mod q)
    # We linearize products using standard IP techniques:
    #   p_ij <= x_i, p_ij <= y_j, p_ij >= x_i + y_j - 1

    def add_product_var(prob, name, a, b):
        """Create p = a * b for binary a, b."""
        p = pulp.LpVariable(name, cat="Binary")
        prob += p <= a, f"{name}_ub1"
        prob += p <= b, f"{name}_ub2"
        prob += p >= a + b - 1, f"{name}_lb"
        return p

    def get_x(i):
        """Get x[i mod q], with x[0] = 0."""
        i = i % q
        return x[i] if i != 0 else 0

    def get_xbar(i):
        """Get D̄11[i] = 1 - x[i] for i != 0, undefined for i = 0."""
        i = i % q
        if i == 0:
            return 0  # 0 ∉ D̄11 (D̄11 = D22 ⊆ Z_q\{0})
        return 1 - x[i]

    def get_z(i):
        return z[i % q]

    def get_zbar(i):
        return 1 - z[i % q]

    # For each difference d, compute CN as sum of products
    # This creates O(q^2) product variables — feasible for q=99

    print(f"Building constraints... ({q*q*4} product variables approx)")

    # Constraint (1): Δ(D11,D11,d) + Δ(D12,D12,d) < r for d ∈ D11
    # = Σ_i x[(i+d)%q] * x[i] + Σ_i z[(i+d)%q] * z[i] < r
    # But this is only active when d ∈ D11, i.e., x[d] = 1.
    # We encode: sum <= r - 1 + M*(1 - x[d]) for big-M

    M = q  # big-M value

    for d in range(1, q):
        # Δ(D11, D11, d)
        xx_prods = []
        for i in range(1, q):
            j = (i + d) % q
            if j == 0:
                continue  # x[0] = 0
            p = add_product_var(prob, f"xx_{d}_{i}", get_x(j), get_x(i))
            xx_prods.append(p)

        # Δ(D12, D12, d)
        zz_prods = []
        for i in range(q):
            j = (i + d) % q
            p = add_product_var(prob, f"zz_{d}_{i}", get_z(j), get_z(i))
            zz_prods.append(p)

        cn_sum = pulp.lpSum(xx_prods) + pulp.lpSum(zz_prods)

        # Constraint (1): active when x[d]=1 (d ∈ D11)
        prob += cn_sum <= r - 1 + M * (1 - get_x(d)), f"C1_d{d}"

        # Constraint (2): active when x[d]=0, d!=0 (d ∈ D̄11 = D22)
        # Δ(D̄11, D̄11, d) + Δ(D12, D12, d) < r
        xxbar_prods = []
        for i in range(1, q):
            j = (i + d) % q
            if j == 0:
                continue
            p = add_product_var(prob, f"xxb_{d}_{i}", get_xbar(j), get_xbar(i))
            xxbar_prods.append(p)

        cn_bar_sum = pulp.lpSum(xxbar_prods) + pulp.lpSum(zz_prods)

        # Active when d ∈ D22 = D̄11, i.e., x[d] = 0 (and d != 0)
        prob += cn_bar_sum <= r - 1 + M * get_x(d), f"C2_d{d}"

    # Constraint (3): Σ(D11, D12, d) + Δ(D12, D22, d) < r for d ∈ D12
    # Σ(D11, D12, d) = Σ_i x[(d-i)%q] * z[i]   (x+y = d means x = d-y)
    # Δ(D12, D22, d) = Σ_i z[(i+d)%q] * (1-x[i]) for i != 0
    for d in range(q):
        sig_prods = []
        for i in range(q):
            j = (d - i) % q
            if j == 0:
                continue  # x[0] = 0
            p = add_product_var(prob, f"xz_{d}_{i}", get_x(j), get_z(i))
            sig_prods.append(p)

        zxb_prods = []
        for i in range(1, q):
            j = (i + d) % q
            p = add_product_var(prob, f"zxb_{d}_{i}", get_z(j), get_xbar(i))
            zxb_prods.append(p)

        cn3 = pulp.lpSum(sig_prods) + pulp.lpSum(zxb_prods)
        prob += cn3 <= r - 1 + M * (1 - get_z(d)), f"C3_d{d}"

    # Constraint (6): Σ(D̄11, D̄12, d) + Δ(D̄12, D11, d) < s for d ∈ D̄12
    # Active when z[d] = 0
    for d in range(q):
        sigbar_prods = []
        for i in range(q):
            j = (d - i) % q
            if j == 0:
                continue
            p = add_product_var(prob, f"xbzb_{d}_{i}", get_xbar(j), get_zbar(i))
            sigbar_prods.append(p)

        zbx_prods = []
        for i in range(1, q):
            j = (i + d) % q
            p = add_product_var(prob, f"zbx_{d}_{i}", get_zbar(j), get_x(i))
            zbx_prods.append(p)

        cn6 = pulp.lpSum(sigbar_prods) + pulp.lpSum(zbx_prods)
        prob += cn6 <= s - 1 + M * get_z(d), f"C6_d{d}"

    # Objective: minimize 0 (feasibility problem)
    prob += 0

    print(f"Variables: {len(prob.variables())}")
    print(f"Constraints: {len(prob.constraints)}")
    print("Solving...")

    if solver_name:
        solver = pulp.getSolver(solver_name, msg=True, timeLimit=3600)
    else:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=3600)

    prob.solve(solver)

    print(f"Status: {pulp.LpStatus[prob.status]}")

    if prob.status == 1:  # Optimal
        D11 = sorted([i for i in range(1, q) if pulp.value(x[i]) > 0.5])
        D12 = sorted([i for i in range(q) if pulp.value(z[i]) > 0.5])
        D22 = sorted([i for i in range(1, q) if i not in D11])
        print(f"\nFOUND WITNESS!")
        print(f"|D11| = {len(D11)}, |D12| = {len(D12)}, |D22| = {len(D22)}")
        print(f"D11 = {D11}")
        print(f"D12 = {D12}")
        return D11, D12, D22
    else:
        print("No solution found within time limit.")
        return None


def verify_witness(q, D11, D12, D22, n):
    """Verify the 2-block-circulant witness satisfies all constraints."""
    D11_set = set(D11)
    D12_set = set(D12)
    D22_set = set(D22)
    D11_bar = D22_set  # complement ansatz
    D12_bar = set(range(q)) - D12_set

    def delta(X, Y, d):
        return sum(1 for y in Y if (y + d) % q in X)

    def sigma(X, Y, d):
        return sum(1 for y in Y if (d - y) % q in X)

    max_G = 0
    max_comp = 0

    for d in range(1, q):
        # G-edge cases
        if d in D11_set:
            cn = delta(D11_set, D11_set, d) + delta(D12_set, D12_set, d)
            max_G = max(max_G, cn)
        if d in D22_set:
            cn = delta(D22_set, D22_set, d) + delta(D12_set, D12_set, d)
            max_G = max(max_G, cn)

        # Complement cases
        if d in D11_bar:
            cn = delta(D11_bar, D11_bar, d) + delta(D12_bar, D12_bar, d)
            max_comp = max(max_comp, cn)
        if d in set(range(1,q)) - D22_set:
            cn = delta(set(range(1,q))-D22_set, set(range(1,q))-D22_set, d) + \
                 delta(D12_bar, D12_bar, d)
            max_comp = max(max_comp, cn)

    for d in D12_set:
        cn = sigma(D11_set, D12_set, d) + delta(D12_set, D22_set, d)
        max_G = max(max_G, cn)

    for d in D12_bar:
        cn = sigma(D11_bar, D12_bar, d) + delta(D12_bar, D11_set, d)
        max_comp = max(max_comp, cn)

    print(f"\nVerification: max G-CN = {max_G} (need < {n-1}), "
          f"max comp-CN = {max_comp} (need < {n})")
    ok = max_G < n - 1 and max_comp < n
    print(f"{'VALID' if ok else 'INVALID'}")
    return ok


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"FM-001b: Searching for R(B_{{{n-1}}}, B_{{{n}}}) >= {4*n-1} witness")
    print(f"2-block-circulant on Z_{{{2*n-1}}} with complement ansatz\n")

    result = solve_block_circulant(n)
    if result:
        D11, D12, D22 = result
        verify_witness(2*n-1, D11, D12, D22, n)
