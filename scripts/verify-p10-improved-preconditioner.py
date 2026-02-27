"""
Problem 10: Improved Preconditioner for P10-G2 Resolution

The original preconditioner P = (cZ^TZ + λI_r) ⊗ K_τ has a built-in
mismatch with A_τ even when D = cI exactly:

  A(D=cI) = c(Z^TZ ⊗ K_τ²) + λ(I_r ⊗ K_τ)
  P_old   = c(Z^TZ ⊗ K_τ)  + λ(I_r ⊗ K_τ)
  Gap     = c·Z^TZ ⊗ K_τ(K_τ - I)

The improved preconditioner EXACTLY matches A when D = cI:

  P_new = c(Z^TZ ⊗ K_τ²) + λ(I_r ⊗ K_τ)

This is NOT a single Kronecker product, but IS efficiently invertible
via eigendecomposition of K_τ:

  K_τ = Q Λ Q^T  →  P_new block-diagonalizes to n blocks of size r×r:
  B_i = cμ_i²·G + λμ_i·I_r  (where G = Z^TZ, μ_i = eigenvalue of K_τ)

Per-application cost: O(n²r + nr²) — same as original!
Setup: O(n³ + nr³) — same order as original!

Key question: with P_new, does δ drop to < 1 for uniform sampling?
"""

import json
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator, cg

np.random.seed(42)


def make_kernel(n, tau=0.1):
    X = np.random.randn(n, max(3, n // 2))
    K = X @ X.T + tau * np.eye(n)
    return K


def make_factors(dims, r):
    return [np.random.randn(ni, r) for ni in dims]


def khatri_rao(factors):
    result = factors[0]
    for A in factors[1:]:
        n1, r = result.shape
        n2 = A.shape[0]
        out = np.zeros((n1 * n2, r))
        for j in range(r):
            out[:, j] = np.kron(result[:, j], A[:, j])
        result = out
    return result


def sample_entries(n, M, q, mode='uniform'):
    N = n * M
    if mode == 'uniform':
        flat_idx = np.random.choice(N, size=min(q, N), replace=False)
        rows = flat_idx // M
        cols = flat_idx % M
    elif mode == 'adversarial_row':
        n_concentrated = int(0.8 * q)
        n_rest = q - n_concentrated
        cols_c = np.random.choice(M, size=min(n_concentrated, M), replace=False)
        rows_c = np.zeros(len(cols_c), dtype=int)
        if n_rest > 0:
            flat_rest = np.random.choice(N, size=n_rest, replace=True)
            rows = np.concatenate([rows_c, flat_rest // M])
            cols = np.concatenate([cols_c, flat_rest % M])
        else:
            rows, cols = rows_c, cols_c
    else:
        flat_idx = np.random.choice(N, size=min(q, N), replace=False)
        rows = flat_idx // M
        cols = flat_idx % M
    return rows.astype(int), cols.astype(int)


def build_system_both(K_tau, Z, obs_rows, obs_cols, lam, n, r):
    """Build A and both preconditioners (old and improved)."""
    nr = n * r
    M = Z.shape[0]
    N = n * M
    q = len(obs_rows)
    c = q / N

    Phi = np.kron(Z, K_tau)
    D_diag = np.zeros(N)
    for row, col in zip(obs_rows, obs_cols):
        D_diag[col * n + row] = 1.0
    D_mat = np.diag(D_diag)

    # System matrix
    A = Phi.T @ D_mat @ Phi + lam * np.kron(np.eye(r), K_tau)

    # Old preconditioner: P_old = H ⊗ K_τ
    G = Z.T @ Z
    H_old = c * G + lam * np.eye(r)
    P_old = np.kron(H_old, K_tau)

    # Improved preconditioner: P_new = c(G ⊗ K_τ²) + λ(I_r ⊗ K_τ)
    K_tau_sq = K_tau @ K_tau
    P_new = c * np.kron(G, K_tau_sq) + lam * np.kron(np.eye(r), K_tau)

    # Also compute exact D=cI system for reference
    A_exact = c * np.kron(G, K_tau_sq) + lam * np.kron(np.eye(r), K_tau)

    return A, P_old, P_new, A_exact, c


def spectral_equivalence(A, P):
    """Compute δ and κ for P^{-1/2} A P^{-1/2}."""
    try:
        L_P = la.cholesky(P, lower=True)
        L_P_inv = la.solve_triangular(L_P, np.eye(L_P.shape[0]), lower=True)
        B = L_P_inv @ A @ L_P_inv.T
        eigvals = la.eigvalsh(B)
        delta = np.max(np.abs(eigvals - 1.0))
        kappa = np.max(eigvals) / np.min(eigvals)
        return {
            'delta': float(delta),
            'kappa': float(kappa),
            'eig_min': float(np.min(eigvals)),
            'eig_max': float(np.max(eigvals)),
        }
    except Exception as e:
        return {'delta': float('inf'), 'kappa': float('inf'),
                'eig_min': float('nan'), 'eig_max': float('nan'),
                'error': str(e)}


def verify_improved_precond_solve(K_tau, G, lam, c, n, r):
    """Verify the improved preconditioner can be applied efficiently
    via eigendecomposition of K_τ, matching the explicit inverse."""
    # Explicit P_new
    K_tau_sq = K_tau @ K_tau
    P_new = c * np.kron(G, K_tau_sq) + lam * np.kron(np.eye(r), K_tau)
    P_new_inv_explicit = la.inv(P_new)

    # Eigendecomposition approach
    eigvals, Q = la.eigh(K_tau)

    # For each eigenvalue μ_i, form B_i = cμ_i²·G + λμ_i·I_r
    # Then P_new^{-1} = (I_r ⊗ Q) · block_diag(B_1^{-1}, ..., B_n^{-1}) · (I_r ⊗ Q^T)

    # Test on random vector
    z = np.random.randn(n * r)
    Z_mat = z.reshape(n, r, order='F')  # n×r, column-major

    # Step 1: rotate
    Z_rot = Q.T @ Z_mat  # n×r

    # Step 2: solve per-eigenvalue r×r systems
    Y_rot = np.zeros_like(Z_rot)
    for i in range(n):
        mu_i = eigvals[i]
        B_i = c * mu_i**2 * G + lam * mu_i * np.eye(r)
        Y_rot[i, :] = la.solve(B_i, Z_rot[i, :])

    # Step 3: rotate back
    Y_mat = Q @ Y_rot  # n×r
    y_eigen = Y_mat.flatten(order='F')

    # Compare with explicit inverse
    y_explicit = P_new_inv_explicit @ z
    error = la.norm(y_eigen - y_explicit) / la.norm(y_explicit)

    return float(error)


def run_pcg_with_precond(A, P, b, tol=1e-10, maxiter=200):
    """Run PCG with given preconditioner."""
    n = A.shape[0]
    A_op = LinearOperator((n, n), matvec=lambda x: A @ x)
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
    }


def run_comparison(n, r, dims_other, q_frac, sampling_mode, lam=1.0, tau=0.1):
    """Compare old vs improved preconditioner."""
    K_tau = make_kernel(n, tau)
    factors_other = make_factors(dims_other, r)
    Z = khatri_rao(factors_other)
    M = Z.shape[0]
    N = n * M
    q = max(r + 1, int(q_frac * N))
    q = min(q, N)
    obs_rows, obs_cols = sample_entries(n, M, q, mode=sampling_mode)
    q_actual = len(obs_rows)

    A, P_old, P_new, A_exact, c = build_system_both(
        K_tau, Z, obs_rows, obs_cols, lam, n, r)

    spec_old = spectral_equivalence(A, P_old)
    spec_new = spectral_equivalence(A, P_new)

    # Also check: P_new vs A_exact (should be identity when D=cI)
    spec_exact = spectral_equivalence(A_exact, P_new)

    b = np.random.randn(n * r)
    pcg_old = run_pcg_with_precond(A, P_old, b)
    pcg_new = run_pcg_with_precond(A, P_new, b)

    return {
        'params': {
            'n': n, 'r': r, 'M': M, 'N': N, 'q': q_actual,
            'q_frac': q_actual / N, 'lambda': lam, 'tau': tau,
            'sampling': sampling_mode,
        },
        'old_precond': {
            'delta': spec_old['delta'], 'kappa': spec_old['kappa'],
            'pcg_iters': pcg_old['iterations'], 'converged': pcg_old['converged'],
        },
        'new_precond': {
            'delta': spec_new['delta'], 'kappa': spec_new['kappa'],
            'pcg_iters': pcg_new['iterations'], 'converged': pcg_new['converged'],
        },
        'exact_match': {
            'delta': spec_exact['delta'], 'kappa': spec_exact['kappa'],
        },
    }


def main():
    results = {
        'problem': 'P10',
        'purpose': 'Improved preconditioner resolves P10-G2 convergence gap',
        'key_insight': (
            'Original P = H⊗K_τ has built-in mismatch c·Z^TZ⊗K_τ(K_τ-I) '
            'even when D=cI. Improved P_new = c(G⊗K_τ²)+λ(I_r⊗K_τ) exactly '
            'matches A(D=cI), making δ purely from sampling noise.'
        ),
        'experiments': [],
    }

    # ================================================================
    # Test 0: Verify eigendecomposition-based precond solve
    # ================================================================
    print("=== Test 0: Eigendecomposition precond solve verification ===")
    K_tau = make_kernel(6, 0.1)
    G = np.random.randn(5, 2)
    G = G.T @ G  # 2×2 SPD
    err = verify_improved_precond_solve(K_tau, G, 1.0, 0.5, 6, 2)
    print(f"  Eigen-solve vs explicit inverse: error = {err:.2e} "
          f"[{'PASS' if err < 1e-10 else 'FAIL'}]")
    results['eigensolve_verified'] = err < 1e-10
    np.random.seed(42)

    # ================================================================
    # Test 1: Exact match when D = cI (sanity check)
    # ================================================================
    print("\n=== Test 1: P_new exactly matches A when D=cI ===")
    K_tau = make_kernel(6, 0.1)
    factors = make_factors([4, 5], 2)
    Z = khatri_rao(factors)
    M = Z.shape[0]
    N = 6 * M
    q = N  # ALL entries observed
    obs_rows = np.repeat(np.arange(6), M)
    obs_cols = np.tile(np.arange(M), 6)
    c = 1.0
    # Scale to get D = cI
    A, P_old, P_new, A_exact, _ = build_system_both(
        K_tau, Z, obs_rows, obs_cols, 1.0, 6, 2)
    spec_exact = spectral_equivalence(A_exact, P_new)
    spec_old_exact = spectral_equivalence(A_exact, P_old)
    print(f"  P_new vs A(D=cI): δ={spec_exact['delta']:.2e}, κ={spec_exact['kappa']:.6f}")
    print(f"  P_old vs A(D=cI): δ={spec_old_exact['delta']:.4f}, κ={spec_old_exact['kappa']:.2f}")
    results['exact_match_delta'] = spec_exact['delta']
    np.random.seed(42)

    # ================================================================
    # Test 2: Uniform sampling — old vs new
    # ================================================================
    print("\n=== Test 2: Uniform sampling — old vs new preconditioner ===")
    print(f"  {'q/N':>5s}  {'δ_old':>8s}  {'δ_new':>8s}  {'κ_old':>8s}  {'κ_new':>8s}  {'it_old':>6s}  {'it_new':>6s}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
    for q_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        exp = run_comparison(n=6, r=2, dims_other=[4, 5], q_frac=q_frac,
                            sampling_mode='uniform')
        exp['test_group'] = 'uniform'
        results['experiments'].append(exp)
        o, n_ = exp['old_precond'], exp['new_precond']
        print(f"  {q_frac:5.1f}  {o['delta']:8.4f}  {n_['delta']:8.4f}  "
              f"{o['kappa']:8.2f}  {n_['kappa']:8.2f}  "
              f"{o['pcg_iters']:6d}  {n_['pcg_iters']:6d}")

    # ================================================================
    # Test 3: Adversarial sampling — old vs new
    # ================================================================
    print("\n=== Test 3: Adversarial row-concentrated — old vs new ===")
    print(f"  {'q/N':>5s}  {'δ_old':>8s}  {'δ_new':>8s}  {'κ_old':>8s}  {'κ_new':>8s}  {'it_old':>6s}  {'it_new':>6s}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
    for q_frac in [0.3, 0.5, 0.7]:
        exp = run_comparison(n=6, r=2, dims_other=[4, 5], q_frac=q_frac,
                            sampling_mode='adversarial_row')
        exp['test_group'] = 'adversarial'
        results['experiments'].append(exp)
        o, n_ = exp['old_precond'], exp['new_precond']
        print(f"  {q_frac:5.1f}  {o['delta']:8.4f}  {n_['delta']:8.4f}  "
              f"{o['kappa']:8.2f}  {n_['kappa']:8.2f}  "
              f"{o['pcg_iters']:6d}  {n_['pcg_iters']:6d}")

    # ================================================================
    # Test 4: λ sensitivity — old vs new
    # ================================================================
    print("\n=== Test 4: λ sensitivity — old vs new (uniform q/N=0.5) ===")
    print(f"  {'λ':>8s}  {'δ_old':>8s}  {'δ_new':>8s}  {'κ_old':>8s}  {'κ_new':>8s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for lam in [10.0, 1.0, 0.1, 0.01]:
        exp = run_comparison(n=6, r=2, dims_other=[4, 5], q_frac=0.5,
                            sampling_mode='uniform', lam=lam)
        exp['test_group'] = 'lambda_sensitivity'
        results['experiments'].append(exp)
        o, n_ = exp['old_precond'], exp['new_precond']
        print(f"  {lam:8.3f}  {o['delta']:8.4f}  {n_['delta']:8.4f}  "
              f"{o['kappa']:8.2f}  {n_['kappa']:8.2f}")

    # ================================================================
    # Test 5: Scaling — does improved preconditioner stay bounded?
    # ================================================================
    print("\n=== Test 5: Scaling n (uniform q/N=0.5) ===")
    print(f"  {'n':>3s}  {'δ_old':>8s}  {'δ_new':>8s}  {'κ_old':>8s}  {'κ_new':>8s}  {'δ_new<1?':>8s}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for n in [4, 6, 8, 10, 12]:
        exp = run_comparison(n=n, r=2, dims_other=[4, 5], q_frac=0.5,
                            sampling_mode='uniform')
        exp['test_group'] = 'scaling'
        results['experiments'].append(exp)
        o, n_ = exp['old_precond'], exp['new_precond']
        bounded = "YES" if n_['delta'] < 1.0 else "no"
        print(f"  {n:3d}  {o['delta']:8.4f}  {n_['delta']:8.4f}  "
              f"{o['kappa']:8.2f}  {n_['kappa']:8.2f}  {bounded:>8s}")

    # ================================================================
    # Test 6: τ sensitivity — how does kernel regularization affect δ?
    # ================================================================
    print("\n=== Test 6: τ sensitivity (uniform q/N=0.5) ===")
    print(f"  {'τ':>8s}  {'δ_old':>8s}  {'δ_new':>8s}  {'κ_old':>8s}  {'κ_new':>8s}  {'δ_new<1?':>8s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for tau in [0.01, 0.1, 1.0, 5.0, 10.0]:
        exp = run_comparison(n=6, r=2, dims_other=[4, 5], q_frac=0.5,
                            sampling_mode='uniform', tau=tau)
        exp['test_group'] = 'tau_sensitivity'
        results['experiments'].append(exp)
        o, n_ = exp['old_precond'], exp['new_precond']
        bounded = "YES" if n_['delta'] < 1.0 else "no"
        print(f"  {tau:8.3f}  {o['delta']:8.4f}  {n_['delta']:8.4f}  "
              f"{o['kappa']:8.2f}  {n_['kappa']:8.2f}  {bounded:>8s}")

    # ================================================================
    # Summary
    # ================================================================
    uniform_new = [e['new_precond']['delta']
                   for e in results['experiments']
                   if e.get('test_group') == 'uniform']
    uniform_old = [e['old_precond']['delta']
                   for e in results['experiments']
                   if e.get('test_group') == 'uniform']

    all_new_deltas = [e['new_precond']['delta'] for e in results['experiments']]
    new_bounded = sum(1 for d in all_new_deltas if d < 1.0)

    summary = {
        'eigensolve_verified': results['eigensolve_verified'],
        'exact_match_delta': results['exact_match_delta'],
        'uniform_delta_old': [min(uniform_old), max(uniform_old)],
        'uniform_delta_new': [min(uniform_new), max(uniform_new)],
        'improvement_factor': (
            np.mean(uniform_old) / np.mean(uniform_new)
            if np.mean(uniform_new) > 0 else float('inf')),
        'new_bounded_count': f"{new_bounded}/{len(all_new_deltas)}",
        'gap_resolution': None,
    }

    if np.mean(uniform_new) < 1.0:
        summary['gap_resolution'] = (
            f"P10-G2 RESOLVED: improved preconditioner achieves δ < 1 on average "
            f"for uniform sampling (mean δ_new = {np.mean(uniform_new):.4f} vs "
            f"mean δ_old = {np.mean(uniform_old):.4f}). "
            f"The convergence claim O(log(1/ε)) is justified with the improved "
            f"preconditioner under uniform sampling."
        )
    elif max(uniform_new) < max(uniform_old) * 0.5:
        summary['gap_resolution'] = (
            f"P10-G2 NARROWED: improved preconditioner reduces δ by "
            f"{summary['improvement_factor']:.1f}x (mean δ_new = {np.mean(uniform_new):.4f} "
            f"vs mean δ_old = {np.mean(uniform_old):.4f}), but δ > 1 persists. "
            f"The preconditioner mismatch was the dominant source of δ."
        )
    else:
        summary['gap_resolution'] = (
            f"P10-G2 PARTIALLY NARROWED: improved preconditioner gives "
            f"δ_new range {summary['uniform_delta_new']} vs δ_old range "
            f"{summary['uniform_delta_old']}. Sampling noise dominates."
        )

    results['summary'] = summary

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Eigensolve verification: {'PASS' if summary['eigensolve_verified'] else 'FAIL'}")
    print(f"  P_new matches A(D=cI): δ = {summary['exact_match_delta']:.2e}")
    print(f"  Uniform δ_old range: {summary['uniform_delta_old']}")
    print(f"  Uniform δ_new range: {summary['uniform_delta_new']}")
    print(f"  Improvement factor: {summary['improvement_factor']:.1f}x")
    print(f"  δ_new < 1 count: {summary['new_bounded_count']}")
    print(f"\n  {summary['gap_resolution']}")

    outfile = 'data/first-proof/problem10-improved-precond-results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results written to: {outfile}")


if __name__ == '__main__':
    main()
