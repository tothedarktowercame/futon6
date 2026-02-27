"""
Problem 10 Convergence Gap Analysis (P10-G2 + SR-4)

Tests the spectral-equivalence claim in Section 5 of the P10 solution:
  (1-δ)P ≤ A_τ ≤ (1+δ)P  with δ bounded away from 1

This script:
  1. Constructs small RKHS-constrained tensor CP problems
  2. Measures the actual spectral-equivalence δ for uniform sampling
  3. Tests adversarial sampling patterns (SR-4: counterexample-first)
  4. Runs PCG and verifies convergence matches the claimed rate
  5. Outputs structured JSON results

Key question: Does δ blow up for non-uniform/adversarial sampling,
or for high-coherence factor matrices?
"""

import json
import sys
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator, cg

np.random.seed(42)


def make_kernel(n, tau=0.1):
    """Generate a PD kernel K_τ = XX^T + τI."""
    X = np.random.randn(n, max(3, n // 2))
    K = X @ X.T + tau * np.eye(n)
    return K


def make_factors(dims, r):
    """Generate random factor matrices for a d-way tensor."""
    return [np.random.randn(ni, r) for ni in dims]


def khatri_rao(factors):
    """Compute the Khatri-Rao (column-wise Kronecker) product."""
    result = factors[0]
    for A in factors[1:]:
        n1, r = result.shape
        n2 = A.shape[0]
        # Column-wise Kronecker: each column is kron(result[:,j], A[:,j])
        out = np.zeros((n1 * n2, r))
        for j in range(r):
            out[:, j] = np.kron(result[:, j], A[:, j])
        result = out
    return result


def hadamard_gram(factors):
    """Compute Z^T Z via Hadamard product of Gram matrices."""
    result = factors[0].T @ factors[0]
    for A in factors[1:]:
        result = result * (A.T @ A)  # Hadamard product
    return result


def sample_entries(n, M, q, mode='uniform'):
    """Sample q observed entries from an n×M tensor unfolding.

    mode:
      'uniform' — uniform random without replacement
      'adversarial_row' — concentrate observations in few rows
      'adversarial_leverage' — sample proportional to leverage scores
    """
    N = n * M
    if mode == 'uniform':
        flat_idx = np.random.choice(N, size=min(q, N), replace=False)
        rows = flat_idx // M
        cols = flat_idx % M
    elif mode == 'adversarial_row':
        # Concentrate 80% of observations in first row
        n_concentrated = int(0.8 * q)
        n_rest = q - n_concentrated
        cols_concentrated = np.random.choice(M, size=min(n_concentrated, M), replace=False)
        rows_concentrated = np.zeros(len(cols_concentrated), dtype=int)
        if n_rest > 0:
            flat_rest = np.random.choice(N, size=n_rest, replace=True)
            rows_rest = flat_rest // M
            cols_rest = flat_rest % M
            rows = np.concatenate([rows_concentrated, rows_rest])
            cols = np.concatenate([cols_concentrated, cols_rest])
        else:
            rows = rows_concentrated
            cols = cols_concentrated
    elif mode == 'adversarial_diagonal':
        # Only observe diagonal-like entries (i, i mod M)
        indices = np.arange(min(q, min(n, M)))
        rows = indices % n
        cols = indices % M
        if len(rows) < q:
            extra = q - len(rows)
            rows = np.concatenate([rows, np.random.randint(0, n, extra)])
            cols = np.concatenate([cols, np.random.randint(0, M, extra)])
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    return rows.astype(int), cols.astype(int)


def build_system(K_tau, Z, obs_rows, obs_cols, lam, n, r):
    """Build the explicit system matrix A and preconditioner P.

    A = (Z⊗K_τ)^T D (Z⊗K_τ) + λ(I_r⊗K_τ)
    P = H⊗K_τ  where H = c·Z^TZ + λI_r
    """
    nr = n * r
    M = Z.shape[0]
    N = n * M
    q = len(obs_rows)
    c = q / N

    # Build Φ = Z⊗K_τ (N×nr) — we only need it for the explicit A
    # But for small problems this is fine
    Phi = np.kron(Z, K_tau)  # (M*n) × (r*n) = N × nr

    # Build D = diagonal mask
    D = np.zeros(N)
    for i, j in zip(obs_rows, obs_cols):
        D[i * M + j] = 1.0  # position in vectorized n×M

    # Wait, the indexing convention: if we vectorize an n×M matrix column-major,
    # entry (i,j) is at position j*n + i. But for Kronecker product Z⊗K,
    # (Z⊗K)vec(V) = vec(KVZ^T), so vec is column-major over the n×r matrix V,
    # but the output is in the N-dimensional space indexed as (i,j) pairs of
    # the n×M unfolding.
    #
    # For the Kronecker Φ = Z⊗K_τ, row (j*n + i) of Φ corresponds to
    # entry (i,j) of the n×M unfolding.

    D_diag = np.zeros(N)
    for row, col in zip(obs_rows, obs_cols):
        idx = col * n + row  # column-major indexing for n×M
        D_diag[idx] = 1.0

    D_mat = np.diag(D_diag)

    # A = Φ^T D Φ + λ(I_r ⊗ K_τ)
    A = Phi.T @ D_mat @ Phi + lam * np.kron(np.eye(r), K_tau)

    # P = H ⊗ K_τ where H = c·Z^TZ + λI_r
    ZtZ = Z.T @ Z
    H = c * ZtZ + lam * np.eye(r)
    P = np.kron(H, K_tau)

    return A, P, c


def spectral_equivalence(A, P):
    """Compute the spectral-equivalence parameter δ.

    Returns δ such that (1-δ)P ≤ A ≤ (1+δ)P.
    δ = max(|eigenvalue(P^{-1/2} A P^{-1/2}) - 1|).
    Also returns κ(P^{-1}A).
    """
    # Compute P^{-1/2} A P^{-1/2}
    L_P = la.cholesky(P, lower=True)
    L_P_inv = la.solve_triangular(L_P, np.eye(L_P.shape[0]), lower=True)
    B = L_P_inv @ A @ L_P_inv.T

    # Eigenvalues of B
    eigvals = la.eigvalsh(B)

    delta = np.max(np.abs(eigvals - 1.0))
    kappa = np.max(eigvals) / np.min(eigvals)

    return {
        'delta': float(delta),
        'kappa': float(kappa),
        'eig_min': float(np.min(eigvals)),
        'eig_max': float(np.max(eigvals)),
        'all_positive': bool(np.all(eigvals > 0)),
    }


def run_pcg(A, P, b, tol=1e-10, maxiter=200):
    """Run PCG and return convergence info."""
    n = A.shape[0]

    A_op = LinearOperator((n, n), matvec=lambda x: A @ x)

    # Preconditioner: P^{-1}
    L_P = la.cholesky(P, lower=True)

    def precond(x):
        y = la.solve_triangular(L_P, x, lower=True)
        return la.solve_triangular(L_P.T, y, lower=False)

    M_op = LinearOperator((n, n), matvec=precond)

    residuals = []
    def callback(xk):
        r = b - A @ xk
        residuals.append(float(la.norm(r) / la.norm(b)))

    x, info = cg(A_op, b, M=M_op, rtol=tol, maxiter=maxiter, callback=callback)

    return {
        'converged': info == 0,
        'iterations': len(residuals),
        'final_residual': float(residuals[-1]) if residuals else None,
        'info': int(info),
    }


def verify_implicit_matvec(K_tau, Z, obs_rows, obs_cols, lam, n, r):
    """Verify the implicit matvec matches the explicit one."""
    M = Z.shape[0]
    N = n * M
    q = len(obs_rows)

    # Random input
    V = np.random.randn(n, r)
    v = V.flatten(order='F')  # column-major vec(V)

    # Explicit matvec via full system
    Phi = np.kron(Z, K_tau)
    D_diag = np.zeros(N)
    for row, col in zip(obs_rows, obs_cols):
        D_diag[col * n + row] = 1.0
    D_mat = np.diag(D_diag)
    A = Phi.T @ D_mat @ Phi + lam * np.kron(np.eye(r), K_tau)
    explicit = A @ v

    # Implicit matvec (Algorithm from Section 2)
    # Step 1: U = K_τ V
    U = K_tau @ V

    # Step 2: Forward at observed entries
    u_vals = np.zeros(q)
    for idx in range(q):
        i, j = obs_rows[idx], obs_cols[idx]
        u_vals[idx] = U[i, :] @ Z[j, :]

    # Step 3: Sparse W' and adjoint
    Wprime = np.zeros((n, M))
    for idx in range(q):
        i, j = obs_rows[idx], obs_cols[idx]
        Wprime[i, j] += u_vals[idx]

    # Step 4: K_τ(W'Z) + λK_τV
    Y = K_tau @ (Wprime @ Z) + lam * (K_tau @ V)
    implicit = Y.flatten(order='F')

    error = la.norm(explicit - implicit) / la.norm(explicit)
    return float(error)


def verify_kronecker_inverse(K_tau, H, n, r):
    """Verify (H⊗K_τ)^{-1} = H^{-1}⊗K_τ^{-1}."""
    P = np.kron(H, K_tau)
    P_inv_explicit = la.inv(P)
    P_inv_kron = np.kron(la.inv(H), la.inv(K_tau))
    error = la.norm(P_inv_explicit - P_inv_kron) / la.norm(P_inv_explicit)
    return float(error)


def verify_hadamard_gram(factors, Z):
    """Verify Z^T Z = ⊙(A_i^T A_i)."""
    ZtZ_direct = Z.T @ Z
    ZtZ_hadamard = hadamard_gram(factors)
    error = la.norm(ZtZ_direct - ZtZ_hadamard) / la.norm(ZtZ_direct)
    return float(error)


def run_experiment(n, r, dims_other, q_frac, sampling_mode, lam=1.0, tau=0.1):
    """Run a full experiment for given parameters."""
    K_tau = make_kernel(n, tau)

    # Factor matrices: mode-k has n rows, others have dims_other[i] rows
    factors_other = make_factors(dims_other, r)
    Z = khatri_rao(factors_other)
    M = Z.shape[0]
    N = n * M

    q = max(r + 1, int(q_frac * N))
    q = min(q, N)

    obs_rows, obs_cols = sample_entries(n, M, q, mode=sampling_mode)
    q_actual = len(obs_rows)

    # Build system
    A, P, c = build_system(K_tau, Z, obs_rows, obs_cols, lam, n, r)

    # Check SPD
    A_eigvals = la.eigvalsh(A)
    a_spd = bool(np.all(A_eigvals > 0))

    P_eigvals = la.eigvalsh(P)
    p_spd = bool(np.all(P_eigvals > 0))

    # Spectral equivalence
    if a_spd and p_spd:
        spec = spectral_equivalence(A, P)
    else:
        spec = {'delta': float('inf'), 'kappa': float('inf'),
                'eig_min': float(np.min(A_eigvals)),
                'eig_max': float(np.max(A_eigvals)),
                'all_positive': False}

    # PCG convergence
    b = np.random.randn(n * r)
    if a_spd and p_spd:
        pcg = run_pcg(A, P, b)
    else:
        pcg = {'converged': False, 'iterations': 0, 'final_residual': None, 'info': -1}

    # Verify identities
    matvec_err = verify_implicit_matvec(K_tau, Z, obs_rows, obs_cols, lam, n, r)
    kron_inv_err = verify_kronecker_inverse(
        K_tau, c * Z.T @ Z + lam * np.eye(r), n, r)
    hadamard_err = verify_hadamard_gram(factors_other, Z)

    return {
        'params': {
            'n': n, 'r': r, 'M': M, 'N': N, 'q': q_actual,
            'q_frac': q_actual / N, 'lambda': lam, 'tau': tau,
            'sampling': sampling_mode, 'dims_other': dims_other,
        },
        'spd': {'A_spd': a_spd, 'P_spd': p_spd},
        'spectral_equivalence': spec,
        'pcg': pcg,
        'identity_errors': {
            'implicit_matvec': matvec_err,
            'kronecker_inverse': kron_inv_err,
            'hadamard_gram': hadamard_err,
        },
    }


def main():
    results = {
        'problem': 'P10',
        'target_gap': 'P10-G2 (convergence claim strength)',
        'sr4_test': 'counterexample-first testing for convergence assumptions',
        'experiments': [],
    }

    # ================================================================
    # Test 1: Uniform sampling at various q/N ratios
    # ================================================================
    print("=== Test 1: Uniform sampling, varying q/N ===")
    for q_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        exp = run_experiment(n=6, r=2, dims_other=[4, 5], q_frac=q_frac,
                             sampling_mode='uniform')
        exp['test_group'] = 'uniform_varying_q'
        results['experiments'].append(exp)
        delta = exp['spectral_equivalence']['delta']
        kappa = exp['spectral_equivalence']['kappa']
        converged = exp['pcg']['converged']
        iters = exp['pcg']['iterations']
        print(f"  q/N={q_frac:.1f}: δ={delta:.4f}, κ={kappa:.2f}, "
              f"PCG {'converged' if converged else 'FAILED'} in {iters} iters")

    # ================================================================
    # Test 2: Adversarial row-concentrated sampling (SR-4 falsification)
    # ================================================================
    print("\n=== Test 2: Adversarial row-concentrated sampling (SR-4) ===")
    for q_frac in [0.3, 0.5, 0.7]:
        exp = run_experiment(n=6, r=2, dims_other=[4, 5], q_frac=q_frac,
                             sampling_mode='adversarial_row')
        exp['test_group'] = 'adversarial_row'
        results['experiments'].append(exp)
        delta = exp['spectral_equivalence']['delta']
        kappa = exp['spectral_equivalence']['kappa']
        converged = exp['pcg']['converged']
        iters = exp['pcg']['iterations']
        print(f"  q/N={q_frac:.1f}: δ={delta:.4f}, κ={kappa:.2f}, "
              f"PCG {'converged' if converged else 'FAILED'} in {iters} iters")

    # ================================================================
    # Test 3: Adversarial diagonal sampling (SR-4 falsification)
    # ================================================================
    print("\n=== Test 3: Adversarial diagonal sampling (SR-4) ===")
    for q_frac in [0.3, 0.5, 0.7]:
        exp = run_experiment(n=6, r=2, dims_other=[4, 5], q_frac=q_frac,
                             sampling_mode='adversarial_diagonal')
        exp['test_group'] = 'adversarial_diagonal'
        results['experiments'].append(exp)
        delta = exp['spectral_equivalence']['delta']
        kappa = exp['spectral_equivalence']['kappa']
        converged = exp['pcg']['converged']
        iters = exp['pcg']['iterations']
        print(f"  q/N={q_frac:.1f}: δ={delta:.4f}, κ={kappa:.2f}, "
              f"PCG {'converged' if converged else 'FAILED'} in {iters} iters")

    # ================================================================
    # Test 4: Scaling test — does δ grow with problem size?
    # ================================================================
    print("\n=== Test 4: Scaling (n growing, q/N=0.5, uniform) ===")
    for n in [4, 6, 8, 10]:
        exp = run_experiment(n=n, r=2, dims_other=[4, 5], q_frac=0.5,
                             sampling_mode='uniform')
        exp['test_group'] = 'scaling_n'
        results['experiments'].append(exp)
        delta = exp['spectral_equivalence']['delta']
        kappa = exp['spectral_equivalence']['kappa']
        print(f"  n={n}: δ={delta:.4f}, κ={kappa:.2f}")

    # ================================================================
    # Test 5: λ sensitivity — does small λ break spectral equivalence?
    # ================================================================
    print("\n=== Test 5: λ sensitivity (uniform, q/N=0.5) ===")
    for lam in [10.0, 1.0, 0.1, 0.01, 0.001]:
        exp = run_experiment(n=6, r=2, dims_other=[4, 5], q_frac=0.5,
                             sampling_mode='uniform', lam=lam)
        exp['test_group'] = 'lambda_sensitivity'
        results['experiments'].append(exp)
        delta = exp['spectral_equivalence']['delta']
        kappa = exp['spectral_equivalence']['kappa']
        converged = exp['pcg']['converged']
        iters = exp['pcg']['iterations']
        print(f"  λ={lam:.3f}: δ={delta:.4f}, κ={kappa:.2f}, "
              f"PCG {'converged' if converged else 'FAILED'} in {iters} iters")

    # ================================================================
    # Test 6: High-coherence factors (SR-4: stress test)
    # ================================================================
    print("\n=== Test 6: High-coherence factors (SR-4 stress test) ===")
    # Standard random factors (low coherence)
    exp_low = run_experiment(n=6, r=2, dims_other=[4, 5], q_frac=0.5,
                             sampling_mode='uniform')
    exp_low['test_group'] = 'coherence_low'
    results['experiments'].append(exp_low)
    delta_low = exp_low['spectral_equivalence']['delta']

    # High-coherence: make factor matrices nearly rank-1
    np.random.seed(123)
    K_tau = make_kernel(6, 0.1)
    v1 = np.random.randn(4, 1)
    v2 = np.random.randn(5, 1)
    A1 = v1 + 0.01 * np.random.randn(4, 2)  # nearly rank-1
    A2 = v2 + 0.01 * np.random.randn(5, 2)
    Z_hc = khatri_rao([A1, A2])
    M = Z_hc.shape[0]
    N = 6 * M
    q = N // 2
    obs_r, obs_c = sample_entries(6, M, q, 'uniform')
    A_sys, P_sys, c = build_system(K_tau, Z_hc, obs_r, obs_c, 1.0, 6, 2)
    spec_hc = spectral_equivalence(A_sys, P_sys)
    delta_high = spec_hc['delta']
    results['experiments'].append({
        'test_group': 'coherence_high',
        'params': {'n': 6, 'r': 2, 'coherence': 'high (near rank-1 factors)'},
        'spectral_equivalence': spec_hc,
    })
    print(f"  Low coherence:  δ={delta_low:.4f}")
    print(f"  High coherence: δ={delta_high:.4f}")
    np.random.seed(42)  # restore seed

    # ================================================================
    # Test 7: Identity verifications (mathematical correctness)
    # ================================================================
    print("\n=== Test 7: Identity verifications ===")
    exp = results['experiments'][0]  # use first experiment
    ie = exp['identity_errors']
    all_pass = all(v < 1e-10 for v in ie.values())
    for name, err in ie.items():
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  {name}: error={err:.2e} [{status}]")

    # ================================================================
    # Summary
    # ================================================================
    uniform_deltas = [e['spectral_equivalence']['delta']
                      for e in results['experiments']
                      if e.get('test_group') == 'uniform_varying_q']
    adversarial_deltas = [e['spectral_equivalence']['delta']
                          for e in results['experiments']
                          if e.get('test_group', '').startswith('adversarial')]

    summary = {
        'identities_verified': all_pass,
        'uniform_delta_range': [min(uniform_deltas), max(uniform_deltas)],
        'adversarial_delta_range': ([min(adversarial_deltas), max(adversarial_deltas)]
                                    if adversarial_deltas else None),
        'adversarial_worse': (max(adversarial_deltas) > max(uniform_deltas)
                              if adversarial_deltas else None),
        'high_coherence_worse': delta_high > delta_low,
        'small_lambda_blowup': any(
            e['spectral_equivalence']['kappa'] > 100
            for e in results['experiments']
            if e.get('test_group') == 'lambda_sensitivity'),
        'pcg_always_converges_uniform': all(
            e.get('pcg', {}).get('converged', False)
            for e in results['experiments']
            if e.get('test_group') == 'uniform_varying_q'),
        'gap_assessment': None,
    }

    # Assess the gap
    if summary['adversarial_worse'] or summary['high_coherence_worse'] or summary['small_lambda_blowup']:
        conditions = []
        if summary['adversarial_worse']:
            conditions.append("adversarial sampling degrades δ")
        if summary['high_coherence_worse']:
            conditions.append("high-coherence factors degrade δ")
        if summary['small_lambda_blowup']:
            conditions.append("small λ causes κ blowup")
        summary['gap_assessment'] = (
            f"P10-G2 GAP CONFIRMED: spectral equivalence IS sampling-dependent. "
            f"Conditions: {'; '.join(conditions)}. "
            f"The convergence claim requires explicit sampling/coherence assumptions. "
            f"Uniform δ range: {summary['uniform_delta_range']}, "
            f"adversarial δ range: {summary['adversarial_delta_range']}."
        )
    else:
        summary['gap_assessment'] = (
            "P10-G2 gap not reproduced at these dimensions — "
            "δ appears stable across sampling patterns."
        )

    results['summary'] = summary

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Identities verified: {summary['identities_verified']}")
    print(f"  Uniform δ range: {summary['uniform_delta_range']}")
    print(f"  Adversarial δ range: {summary['adversarial_delta_range']}")
    print(f"  Adversarial worse than uniform: {summary['adversarial_worse']}")
    print(f"  High coherence worse: {summary['high_coherence_worse']}")
    print(f"  Small λ blowup: {summary['small_lambda_blowup']}")
    print(f"  PCG always converges (uniform): {summary['pcg_always_converges_uniform']}")
    print(f"\n  GAP ASSESSMENT: {summary['gap_assessment']}")

    # Write JSON results
    outfile = 'data/first-proof/problem10-convergence-gap-results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results written to: {outfile}")

    return results


if __name__ == '__main__':
    main()
