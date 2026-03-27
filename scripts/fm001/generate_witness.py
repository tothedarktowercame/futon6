#!/usr/bin/env python3
"""FM-001 witness generator with Wesley + heuristic search modes."""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_power_decompose(q: int) -> Tuple[int, int] | None:
    if q < 2:
        return None
    if is_prime(q):
        return q, 1
    for p in range(2, int(math.isqrt(q)) + 1):
        if not is_prime(p) or q % p:
            continue
        k = 0
        value = q
        while value % p == 0:
            value //= p
            k += 1
        if value == 1:
            return p, k
    return None


class FiniteField:
    def __init__(self, p: int, k: int):
        self.p = p
        self.k = k
        self.q = p**k
        if k == 1:
            self.elements = [tuple([i]) for i in range(p)]
            self.zero = self.elements[0]
            self.one = self.elements[1]
            self._irr_poly: Tuple[int, ...] | None = None
        else:
            self._irr_poly = self._find_irreducible_poly()
            self.elements = [
                tuple(coeffs) for coeffs in itertools.product(range(p), repeat=k)
            ]
            self.zero = tuple(0 for _ in range(k))
            self.one = (1,) + tuple(0 for _ in range(k - 1))

    def _find_irreducible_poly(self) -> Tuple[int, ...]:
        for coeffs in itertools.product(range(self.p), repeat=self.k):
            cand = tuple(coeffs + (1,))
            if self._is_irreducible(cand):
                return cand
        raise ValueError("no irreducible polynomial found")

    def _poly_mod(self, dividend: List[int], divisor: Sequence[int]) -> List[int]:
        divisor = list(divisor)
        acc = list(dividend)
        while len(acc) >= len(divisor):
            if acc[-1]:
                factor = acc[-1] * pow(divisor[-1], self.p - 2, self.p) % self.p
                for i in range(len(divisor)):
                    pos = len(acc) - len(divisor) + i
                    acc[pos] = (acc[pos] - factor * divisor[i]) % self.p
            acc.pop()
        while acc and acc[-1] == 0:
            acc.pop()
        return acc or [0]

    def _poly_mul(
        self, a: Sequence[int], b: Sequence[int], mod_poly: Sequence[int]
    ) -> List[int]:
        result = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                result[i + j] = (result[i + j] + ai * bj) % self.p
        return self._poly_mod(result, mod_poly)

    def _poly_pow(
        self, base: Sequence[int], exp: int, mod_poly: Sequence[int]
    ) -> List[int]:
        result = [1]
        base = self._poly_mod(list(base), mod_poly)
        while exp:
            if exp & 1:
                result = self._poly_mul(result, base, mod_poly)
            base = self._poly_mul(base, base, mod_poly)
            exp >>= 1
        return result

    def _poly_gcd(self, a: Sequence[int], b: Sequence[int]) -> List[int]:
        a = list(a)
        b = list(b)
        while b and not all(c == 0 for c in b):
            a, b = b, self._poly_mod(a, b)
        if not a:
            return [0]
        inv = pow(a[-1], self.p - 2, self.p)
        return [(c * inv) % self.p for c in a]

    def _is_irreducible(self, poly: Sequence[int]) -> bool:
        k = len(poly) - 1
        for candidate in range(self.p):
            value = 0
            for idx, coeff in enumerate(poly):
                value = (value + coeff * pow(candidate, idx, self.p)) % self.p
            if value == 0:
                return False
        if k == 1:
            return True
        x_poly = [0, 1]
        for i in range(1, k):
            x_pi = self._poly_pow(x_poly, self.p**i, poly)
            diff = x_pi[:]
            while len(diff) < 2:
                diff.append(0)
            diff[1] = (diff[1] - 1) % self.p
            if len(self._poly_gcd(diff, poly)) > 1:
                return False
        x_pk = self._poly_pow(x_poly, self.p**k, poly)
        diff = x_pk[:]
        while len(diff) < 2:
            diff.append(0)
        diff[1] = (diff[1] - 1) % self.p
        return all(c == 0 for c in self._poly_mod(diff, poly))

    def add(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple((ai + bi) % self.p for ai, bi in zip(a, b))

    def neg(self, a: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple((-ai) % self.p for ai in a)

    def sub(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.add(a, self.neg(b))

    def mul(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.k == 1:
            return ((a[0] * b[0]) % self.p,)
        size = 2 * self.k - 1
        acc = [0] * size
        for i in range(self.k):
            for j in range(self.k):
                acc[i + j] = (acc[i + j] + a[i] * b[j]) % self.p
        assert self._irr_poly is not None
        for idx in range(size - 1, self.k - 1, -1):
            coeff = acc[idx]
            if coeff == 0:
                continue
            for j in range(self.k + 1):
                acc[idx - self.k + j] = (
                    acc[idx - self.k + j] - coeff * self._irr_poly[j]
                ) % self.p
        return tuple(acc[: self.k])

    def pow(self, base: Tuple[int, ...], exp: int) -> Tuple[int, ...]:
        result = self.one
        while exp:
            if exp & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            exp >>= 1
        return result

    def is_zero(self, a: Tuple[int, ...]) -> bool:
        return all(c == 0 for c in a)

    def is_quadratic_residue(self, a: Tuple[int, ...]) -> bool:
        if self.is_zero(a):
            return False
        exponent = (self.q - 1) // 2
        return self.pow(a, exponent) == self.one

    def quadratic_classes(self) -> Tuple[set[Tuple[int, ...]], set[Tuple[int, ...]]]:
        qr: set[Tuple[int, ...]] = set()
        nr: set[Tuple[int, ...]] = set()
        for elem in self.elements:
            if self.is_zero(elem):
                continue
            (qr if self.is_quadratic_residue(elem) else nr).add(elem)
        return qr, nr


def _bitsets_to_string(adj_bitsets: Sequence[int]) -> str:
    bits: List[str] = []
    vertices = len(adj_bitsets)
    for col in range(vertices):
        for row in range(col + 1, vertices):
            bits.append("1" if (adj_bitsets[row] >> col) & 1 else "0")
    return "".join(bits)


def _verify_bitsets(adj_bitsets: Sequence[int], n: int) -> Dict[str, int | bool]:
    vertices = len(adj_bitsets)
    mask = (1 << vertices) - 1
    comp = [((~row) & mask) & ~(1 << idx) for idx, row in enumerate(adj_bitsets)]
    max_cn_g = 0
    max_cn_comp = 0
    for i in range(vertices):
        ai = adj_bitsets[i]
        for j in range(i + 1, vertices):
            aj = adj_bitsets[j]
            cn = (ai & aj).bit_count()
            if (ai >> j) & 1:
                if cn > max_cn_g:
                    max_cn_g = cn
            else:
                cn_comp = (comp[i] & comp[j]).bit_count()
                if cn_comp > max_cn_comp:
                    max_cn_comp = cn_comp
    return {
        "max_cn_g": max_cn_g,
        "max_cn_comp": max_cn_comp,
        "B_{n-1}_free": max_cn_g < n - 1,
        "comp_B_n_free": max_cn_comp < n,
        "valid": max_cn_g < n - 1 and max_cn_comp < n,
    }


def _generate_wesley_bitsets(n: int) -> Tuple[List[int], Dict[str, int]] | Tuple[None, str]:
    q = 2 * n - 1
    if q % 4 != 1:
        return None, f"q={q} ≡ {q % 4} (mod 4); Wesley requires q ≡ 1 (mod 4)."
    prime_power = prime_power_decompose(q)
    if prime_power is None:
        return None, f"q={q} is not a prime power."
    p, k = prime_power
    field = FiniteField(p, k)
    qr_set, nr_set = field.quadratic_classes()
    assert field.neg(field.one) in qr_set
    vertices = 2 * q
    adj_bitsets = [0] * vertices

    def add_edge(u: int, v: int) -> None:
        adj_bitsets[u] |= 1 << v
        adj_bitsets[v] |= 1 << u

    for i in range(vertices):
        for j in range(i + 1, vertices):
            block_i = 0 if i < q else 1
            block_j = 0 if j < q else 1
            elem_i = field.elements[i % q]
            elem_j = field.elements[j % q]
            diff = field.sub(elem_j, elem_i)
            if field.is_zero(diff):
                continue
            if block_i == block_j == 0 and diff in qr_set:
                add_edge(i, j)
            elif block_i == block_j == 1 and diff in nr_set:
                add_edge(i, j)
            elif block_i != block_j and diff in qr_set:
                add_edge(i, j)

    stats: Dict[str, int | float] = {
        "n": n,
        "method": "wesley",
        "q": q,
        "p": p,
        "k": k,
        "vertices": vertices,
        "edges": sum(row.bit_count() for row in adj_bitsets) // 2,
        "total_time": 0.0,
    }
    return adj_bitsets, stats


def _initial_random_bitsets(vertices: int, rng: random.Random) -> List[int]:
    bitsets = [0] * vertices
    for i in range(vertices):
        for j in range(i + 1, vertices):
            if rng.random() < 0.5:
                bitsets[i] |= 1 << j
                bitsets[j] |= 1 << i
    return bitsets


def _collect_violations(
    adj_bitsets: Sequence[int], n: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    vertices = len(adj_bitsets)
    mask = (1 << vertices) - 1
    comp = [((~row) & mask) & ~(1 << idx) for idx, row in enumerate(adj_bitsets)]
    worst_edges: List[Tuple[int, int]] = []
    worst_non_edges: List[Tuple[int, int]] = []
    worst_edge_cn = n - 2
    worst_non_cn = n - 1
    for i in range(vertices):
        ai = adj_bitsets[i]
        for j in range(i + 1, vertices):
            aj = adj_bitsets[j]
            cn = (ai & aj).bit_count()
            is_edge = (ai >> j) & 1
            if is_edge:
                if cn >= n - 1:
                    if cn > worst_edge_cn:
                        worst_edge_cn = cn
                        worst_edges = [(i, j)]
                    elif cn == worst_edge_cn and len(worst_edges) < 64:
                        worst_edges.append((i, j))
            else:
                cn_comp = (comp[i] & comp[j]).bit_count()
                if cn_comp >= n:
                    if cn_comp > worst_non_cn:
                        worst_non_cn = cn_comp
                        worst_non_edges = [(i, j)]
                    elif cn_comp == worst_non_cn and len(worst_non_edges) < 64:
                        worst_non_edges.append((i, j))
    return worst_edges, worst_non_edges


def _search_bitsets(
    n: int, seconds: float, seed: int | None, restart_factor: float
) -> Tuple[List[int], Dict[str, int | float]] | Tuple[None, str]:
    vertices = 4 * n - 2
    deadline = time.time() + max(seconds, 0.0)
    rng = random.Random(seed)
    restart_factor = max(1.0, restart_factor)
    max_iters_per_restart = max(200, int(vertices * restart_factor))
    total_iterations = 0

    while time.time() < deadline:
        adj_bitsets = _initial_random_bitsets(vertices, rng)
        for _ in range(max_iters_per_restart):
            edges_v, non_edges_v = _collect_violations(adj_bitsets, n)
            if not edges_v and not non_edges_v:
                stats: Dict[str, int | float] = {
                    "n": n,
                    "method": "search",
                    "vertices": vertices,
                    "edges": sum(row.bit_count() for row in adj_bitsets) // 2,
                    "iterations": total_iterations,
                }
                return adj_bitsets, stats
            if time.time() > deadline:
                break
            total_iterations += 1
            if edges_v and (not non_edges_v or rng.random() < 0.5):
                i, j = rng.choice(edges_v)
                adj_bitsets[i] &= ~(1 << j)
                adj_bitsets[j] &= ~(1 << i)
            else:
                i, j = rng.choice(non_edges_v)
                adj_bitsets[i] |= 1 << j
                adj_bitsets[j] |= 1 << i

    return None, f"search timed out after {total_iterations} iterations"


def eligible_n_values(max_n: int = 100) -> List[int]:
    values: List[int] = []
    for n in range(2, max_n + 1):
        q = 2 * n - 1
        if q % 4 != 1:
            continue
        if prime_power_decompose(q):
            values.append(n)
    return values


def generate_witness(
    n: int,
    *,
    verify: bool = False,
    method: str = "auto",
    search_seconds: float = 60.0,
    seed: int | None = None,
    restart_factor: float = 50.0,
) -> Tuple[str, Dict[str, int | float | bool]] | Tuple[None, str]:
    methods = {"auto", "wesley", "search"}
    if method not in methods:
        raise ValueError(f"method must be one of {methods}")
    attempts: List[Tuple[List[int], Dict[str, int | float]] | Tuple[None, str]] = []

    def maybe_wesley():
        result = _generate_wesley_bitsets(n)
        attempts.append(result)
        return result

    def maybe_search():
        result = _search_bitsets(n, search_seconds, seed, restart_factor)
        attempts.append(result)
        return result

    if method == "wesley":
        final_result = maybe_wesley()
    elif method == "search":
        final_result = maybe_search()
    else:
        wesley_result = maybe_wesley()
        if wesley_result[0] is not None:
            final_result = wesley_result
        else:
            final_result = maybe_search()

    adj_bitsets, stats = final_result  # type: ignore[misc]
    if adj_bitsets is None:
        reasons = "; ".join(str(reason) for _, reason in attempts if reason)
        return None, reasons or "construction failed"

    bits = _bitsets_to_string(adj_bitsets)
    if verify:
        stats.update(_verify_bitsets(adj_bitsets, n))
    return bits, stats


def _run_all(max_n: int, args: argparse.Namespace) -> None:
    values = eligible_n_values(max_n)
    print(f"Eligible n ≤ {max_n} (Wesley): {values}")
    for n in range(2, max_n + 1):
        bits, stats = generate_witness(
            n,
            verify=False,
            method=args.method,
            search_seconds=args.search_seconds,
            seed=args.seed,
            restart_factor=args.restart_factor,
        )
        if bits is None:
            print(f"n={n:3d} FAILED: {stats}")
            continue
        print(
            f"n={n:3d} method={stats['method']} V={stats['vertices']:3d} "
            f"edges={stats['edges']:6d}"
        )


def _run_single(n: int, args: argparse.Namespace, verify: bool) -> None:
    bits, stats = generate_witness(
        n,
        verify=verify,
        method=args.method,
        search_seconds=args.search_seconds,
        seed=args.seed,
        restart_factor=args.restart_factor,
    )
    if bits is None:
        print(f"n={n}: {stats}")
        sys.exit(1)
    print(
        f"n={n} method={stats['method']} V={stats['vertices']} edges={stats['edges']}"
    )
    if verify:
        print(
            f"  max_CN(G)={stats['max_cn_g']} < {n-1}; "
            f"max_CN(comp)={stats['max_cn_comp']} < {n}"
        )
        print(f"  valid={stats['valid']}")
    else:
        print(bits)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", type=int, metavar="N", help="generate all n ≤ N")
    group.add_argument("--check", type=int, metavar="N", help="generate + verify n")
    group.add_argument("--n", type=int, metavar="N", help="generate n (no verify)")
    parser.add_argument(
        "--method",
        choices=("auto", "wesley", "search"),
        default="auto",
        help="construction strategy",
    )
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=60.0,
        help="time budget for heuristic search",
    )
    parser.add_argument("--seed", type=int, help="RNG seed for search mode")
    parser.add_argument(
        "--restart-factor",
        type=float,
        default=50.0,
        help="multiplier for restart threshold (default vertices*50)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="verify constraints for --n runs"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.all is not None:
        _run_all(args.all, args)
    elif args.check is not None:
        _run_single(args.check, args, verify=True)
    else:
        assert args.n is not None
        _run_single(args.n, args, verify=args.verify)


if __name__ == "__main__":
    main()
