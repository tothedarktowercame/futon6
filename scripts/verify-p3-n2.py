#!/usr/bin/env python3
"""n=2 sanity check for Problem 3.

For lambda=(a,0), the inhomogeneous t-PushTASEP has two states:
    (a,0)  <->  (0,a)
with transition rates:
    (a,0)->(0,a): 1/x1
    (0,a)->(a,0): 1/x2

This script computes the stationary distribution and verifies detailed balance.
"""

from sympy import symbols, simplify


def main() -> None:
    x1, x2 = symbols("x1 x2", positive=True)

    # Two-state chain rates
    r12 = 1 / x1  # (a,0) -> (0,a)
    r21 = 1 / x2  # (0,a) -> (a,0)

    # Stationary distribution for two-state CTMC
    pi_10 = simplify(r21 / (r12 + r21))  # state (a,0)
    pi_01 = simplify(r12 / (r12 + r21))  # state (0,a)

    print("Two-state t-PushTASEP (lambda=(a,0))")
    print(f"r((a,0)->(0,a)) = {r12}")
    print(f"r((0,a)->(a,0)) = {r21}")
    print()
    print(f"pi(a,0) = {pi_10}")
    print(f"pi(0,a) = {pi_01}")
    print(f"ratio pi(0,a)/pi(a,0) = {simplify(pi_01 / pi_10)}")
    print()

    # Detailed balance check
    db = simplify(pi_10 * r12 - pi_01 * r21)
    print(f"detailed-balance residual: {db}")
    print("OK" if db == 0 else "FAILED")


if __name__ == "__main__":
    main()
