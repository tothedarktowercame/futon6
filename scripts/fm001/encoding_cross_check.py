#!/usr/bin/env python3
"""
FM-001: cross-check the book-Ramsey SAT encoding against an independent one.

Why this exists
---------------
The harness treats its two verdicts very differently.

A SAT verdict is never believed on its own: `budgeted_solve.py` decodes the
model and re-runs `verify_assignment` before it will write the word `sat`, and
an unchecked colouring is recorded as `sat-unverified`. That check is sound
because it tests the COMBINATORIAL property directly and never consults the
CNF — a witness that passes it refutes R(B_{n-1}, B_n) <= 4n-2 no matter what
the encoder did.

An UNSAT verdict has no such route, and it is the stronger claim: it says the
bound HOLDS for that n. Nothing between the solver and that theorem was tested.
Its whole weight rests on two properties of `ramsey_book_sat.build_instance`
that had no test at all:

  soundness      every assignment the CNF admits is a valid colouring;
  completeness   every valid colouring is admitted, up to the symmetry break.

Completeness is the load-bearing one. If the encoder over-constrains — a
cardinality bound off by one, or a symmetry breaker that is not actually
satisfiability-preserving — the CNF goes UNSAT while colourings exist, and the
solver reports a theorem that is not there. That failure is silent: an UNSAT
answer looks the same either way.

What is checked
---------------
An INDEPENDENT encoder (`naive_clauses`) states the forbidden configurations
directly: for every pair (u, v), no (n-1)-subset of common neighbours may be
all-red with uv red, and no n-subset may be all-blue with uv blue. It has no
helper variables, no cardinality network and no symmetry breaking — it is a
transcription of `verify_assignment`, so it agrees with the property by
construction and disagrees with the production encoding whenever that encoding
is wrong. Three checks:

  1. naive-vs-property     the independent encoder accepts exactly the
                           colourings `verify_assignment` accepts. If this
                           fails the cross-check itself is void, so it is
                           reported first and separately.
  2. production-vs-naive   the production CNF admits a colouring exactly when
                           the independent encoder accepts it AND vertex-0
                           monotonicity holds.
  3. symmetry-break        a valid colouring produced by the INDEPENDENT
                           encoder alone is relabelled canonically and must
                           then be admitted by the production CNF. This is the
                           satisfiability-preservation obligation of
                           `add_vertex_zero_monotone_edges`, discharged by
                           example rather than assumed.

What is NOT checked
-------------------
This validates the encoding at small n by sampling; it is not a proof for all
n. And the independent encoder is independent of the production ENCODING, not
of the property: if `verify_assignment` is the wrong formalisation of the book
condition, both agree and both are wrong. That question is upstream of this
file.

Usage
-----
    .venv/bin/python scripts/fm001/encoding_cross_check.py --n 3 --samples 400
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_harness():
    """Load ramsey_book_sat.py so the encoder under test is the SAME code that
    produced the archived CNFs."""
    spec = importlib.util.spec_from_file_location(
        "ramsey_book_sat", HERE / "ramsey_book_sat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def naive_clauses(harness, n: int, edges, vertex_count: int) -> list[list[int]]:
    """The independent encoding: forbid each violating configuration outright.

    `verify_assignment` rejects a colouring when some pair (u, v) has
    red_support >= n-1 (with uv red) or blue_support >= n (with uv blue). Here
    that is said as clauses over the edge variables only — one per offending
    subset. Exponential in n and useless at scale, which is exactly why the
    production encoder uses helpers and cardinality networks; that is also why
    the production encoder needs checking against something simple enough to be
    read.
    """
    clauses: list[list[int]] = []
    for u, v in combinations(range(vertex_count), 2):
        uv = edges[(u, v)]
        others = [w for w in range(vertex_count) if w != u and w != v]
        for subset in combinations(others, n - 1):
            # not (uv red and all of subset are common red neighbours)
            clause = [-uv]
            for w in subset:
                clause.append(-edges[harness.ordered_pair(u, w)])
                clause.append(-edges[harness.ordered_pair(v, w)])
            clauses.append(clause)
        for subset in combinations(others, n):
            # not (uv blue and all of subset are common blue neighbours)
            clause = [uv]
            for w in subset:
                clause.append(edges[harness.ordered_pair(u, w)])
                clause.append(edges[harness.ordered_pair(v, w)])
            clauses.append(clause)
    return clauses


def naive_satisfied(clauses, edges, colouring) -> bool:
    """Evaluate the independent encoding on a colouring. No solver involved:
    every variable it mentions is an edge variable, so it is a direct read."""
    value = {var: colouring[edge] for edge, var in edges.items()}
    for clause in clauses:
        for lit in clause:
            truth = value[abs(lit)]
            if (lit > 0 and truth) or (lit < 0 and not truth):
                break
        else:
            return False
    return True


def monotone_zero(harness, colouring, vertex_count: int) -> bool:
    """The symmetry breaker's own condition: the edges out of vertex 0 are red
    first, then blue."""
    return all(
        (not colouring[harness.ordered_pair(0, j)])
        or colouring[harness.ordered_pair(0, j - 1)]
        for j in range(2, vertex_count))


def canonical_relabelling(harness, colouring, vertex_count: int):
    """Relabel vertices 1.. so the red edges at vertex 0 come first.

    Vertex relabelling is a graph isomorphism, so it preserves the book
    condition exactly; this is the permutation that witnesses that
    `add_vertex_zero_monotone_edges` excludes no colouring, only duplicates.
    """
    red = [j for j in range(1, vertex_count)
           if colouring[harness.ordered_pair(0, j)]]
    blue = [j for j in range(1, vertex_count)
            if not colouring[harness.ordered_pair(0, j)]]
    perm = [0] + red + blue
    return {harness.ordered_pair(i, j):
            colouring[harness.ordered_pair(perm[i], perm[j])]
            for i, j in combinations(range(vertex_count), 2)}


def admits(solver, edges, colouring) -> bool:
    """Does this CNF admit the colouring? The edge variables are fixed as
    assumptions and the solver decides the auxiliary variables, which is the
    only honest way to ask: helper and cardinality variables are not free."""
    assumptions = [edges[edge] if is_red else -edges[edge]
                   for edge, is_red in colouring.items()]
    return bool(solver.solve(assumptions=assumptions))


def random_colouring(edges, rng: random.Random, red_probability: float):
    return {edge: rng.random() < red_probability for edge in edges}


def perturbations(colouring, rng: random.Random, flips: int):
    """A colouring with `flips` edges recoloured.

    Uniform random colourings are never valid at these sizes — every one of
    them violates the book condition somewhere — so sampling only those
    exercises the REJECTION direction and never asks whether the production CNF
    admits what it should. Perturbing a known-valid colouring by a few edges
    puts samples on both sides of the boundary.
    """
    perturbed = dict(colouring)
    for edge in rng.sample(sorted(perturbed), flips):
        perturbed[edge] = not perturbed[edge]
    return perturbed


def cross_check(n: int, samples: int = 400, seed: int = 0,
                solver_name: str = "glucose4") -> dict:
    """Run all three checks and report. `ok` is False if ANY disagreement was
    found; a disagreement is a reason to distrust an UNSAT verdict at this n."""
    from pysat.solvers import Solver

    harness = load_harness()
    vertex_count = 4 * n - 2
    cnf, edges, _pool = harness.build_instance(n)
    independent = naive_clauses(harness, n, edges, vertex_count)

    # Check 3 first: it also supplies the valid colouring the sampler needs.
    with Solver(name=solver_name, bootstrap_with=independent) as free:
        independent_sat = bool(free.solve())
        witness = (harness.decode_model(free.get_model(), edges)
                   if independent_sat else None)

    rng = random.Random(seed)
    property_disagreements = []
    encoding_disagreements = []
    accepted_samples = 0

    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as production:
        symmetry_break = None
        if witness is not None:
            relabelled = canonical_relabelling(harness, witness, vertex_count)
            symmetry_break = {
                "witness_valid": harness.verify_assignment(
                    n, vertex_count, witness),
                "witness_monotone": monotone_zero(
                    harness, witness, vertex_count),
                "relabelled_valid": harness.verify_assignment(
                    n, vertex_count, relabelled),
                "relabelled_monotone": monotone_zero(
                    harness, relabelled, vertex_count),
                "relabelled_admitted": admits(production, edges, relabelled),
            }

        for index in range(samples):
            if witness is not None and index % 2 == 0:
                colouring = perturbations(
                    relabelled if index % 4 == 0 else witness,
                    rng, rng.choice([0, 1, 2, 3]))
            else:
                colouring = random_colouring(
                    edges, rng, rng.choice([0.3, 0.4, 0.5, 0.6, 0.7]))
            by_property = harness.verify_assignment(n, vertex_count, colouring)
            by_independent = naive_satisfied(independent, edges, colouring)
            if by_property != by_independent:
                property_disagreements.append(
                    {"sample": index, "verify_assignment": by_property,
                     "independent_encoder": by_independent})
            if by_property:
                accepted_samples += 1
            expected = by_independent and monotone_zero(
                harness, colouring, vertex_count)
            if admits(production, edges, colouring) != expected:
                encoding_disagreements.append(
                    {"sample": index, "production_admits": not expected,
                     "independent_accepts": by_independent})

    symmetry_break_ok = (
        symmetry_break is None
        or (symmetry_break["witness_valid"]
            and symmetry_break["relabelled_valid"]
            and symmetry_break["relabelled_monotone"]
            and symmetry_break["relabelled_admitted"]))

    return {
        "n": n,
        "vertex_count": vertex_count,
        "samples": samples,
        "seed": seed,
        "production_clauses": len(cnf.clauses),
        "independent_clauses": len(independent),
        "samples_accepted_by_property": accepted_samples,
        "property_disagreements": property_disagreements,
        "encoding_disagreements": encoding_disagreements,
        "independent_encoder_sat": independent_sat,
        "symmetry_break": symmetry_break,
        "ok": (not property_disagreements
               and not encoding_disagreements
               and symmetry_break_ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3,
                    help="book parameter n (builds K_{4n-2}); the independent "
                         "encoder is exponential in n, so keep this small")
    ap.add_argument("--samples", type=int, default=400,
                    help="random colourings to compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default="glucose4", help="pysat solver name")
    ap.add_argument("--json-out", type=Path,
                    help="write the report here as well as to stdout")
    args = ap.parse_args()

    if args.n < 3:
        raise SystemExit("n must be >= 3 for FM-001.")

    report = cross_check(args.n, args.samples, args.seed, args.solver)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    # Non-zero on any disagreement: a mismatch here means an UNSAT verdict from
    # the production encoding is not evidence for the bound at this n.
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
