"""Regression tests for scripts/fm001/encoding_cross_check.py.

FM-001 believes a SAT verdict only after re-checking the colouring, but has
always believed an UNSAT verdict outright — and UNSAT is the claim that the
bound HOLDS. Everything between the solver and that claim is the encoder in
`ramsey_book_sat.build_instance`, which had no test.

The failure that matters is silent: an encoder that over-constrains (a
cardinality bound off by one, or a symmetry breaker that is not
satisfiability-preserving) returns UNSAT while valid colourings exist, and
UNSAT looks the same whether or not it means anything.

These tests pin the cross-check that would catch it, and — the part that is
easy to get wrong — pin that the cross-check is not vacuous: it must actually
sample colourings the encoder should ACCEPT, and it must go red when handed a
broken encoder.
"""

from __future__ import annotations

import importlib.util
import random
from itertools import combinations
from pathlib import Path

import pytest

pytest.importorskip("pysat", reason="python-sat is a dev extra")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fm001" / "encoding_cross_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("encoding_cross_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cc = _load()
harness = cc.load_harness()

N = 3
VERTEX_COUNT = 4 * N - 2


@pytest.fixture(scope="module")
def instance():
    cnf, edges, _pool = harness.build_instance(N)
    independent = cc.naive_clauses(harness, N, edges, VERTEX_COUNT)
    return cnf, edges, independent


@pytest.fixture(scope="module")
def independent_witness(instance):
    """A valid colouring the production encoder had no hand in producing."""
    from pysat.solvers import Solver

    _cnf, edges, independent = instance
    with Solver(name="glucose4", bootstrap_with=independent) as solver:
        assert solver.solve(), "the independent encoding must be satisfiable at n=3"
        return harness.decode_model(solver.get_model(), edges)


def _samples(edges, witness, count, seed):
    rng = random.Random(seed)
    out = []
    for index in range(count):
        if index % 2 == 0:
            out.append(cc.perturbations(witness, rng, rng.choice([0, 1, 2, 3])))
        else:
            out.append(cc.random_colouring(edges, rng, rng.choice([0.4, 0.5, 0.6])))
    return out


def test_independent_encoder_agrees_with_the_property(instance, independent_witness):
    """If this fails the cross-check is void: the independent encoder would no
    longer be a statement of the same condition."""
    _cnf, edges, independent = instance
    for colouring in _samples(edges, independent_witness, 60, seed=1):
        assert (cc.naive_satisfied(independent, edges, colouring)
                == harness.verify_assignment(N, VERTEX_COUNT, colouring))


def test_production_cnf_admits_exactly_the_valid_monotone_colourings(
        instance, independent_witness):
    from pysat.solvers import Solver

    cnf, edges, independent = instance
    with Solver(name="glucose4", bootstrap_with=cnf.clauses) as production:
        for colouring in _samples(edges, independent_witness, 60, seed=2):
            expected = (cc.naive_satisfied(independent, edges, colouring)
                        and cc.monotone_zero(harness, colouring, VERTEX_COUNT))
            assert cc.admits(production, edges, colouring) is expected


def test_symmetry_break_excludes_no_valid_colouring(instance, independent_witness):
    """The satisfiability-preservation obligation of
    add_vertex_zero_monotone_edges, discharged by example.

    The un-relabelled witness is asserted REJECTED as well, so the test cannot
    pass vacuously by the breaker doing nothing."""
    from pysat.solvers import Solver

    cnf, edges, _independent = instance
    assert harness.verify_assignment(N, VERTEX_COUNT, independent_witness)
    assert not cc.monotone_zero(harness, independent_witness, VERTEX_COUNT), (
        "the independent witness happens to satisfy the breaker; this test "
        "would then prove nothing about relabelling")

    relabelled = cc.canonical_relabelling(harness, independent_witness, VERTEX_COUNT)
    assert harness.verify_assignment(N, VERTEX_COUNT, relabelled), (
        "relabelling vertices is an isomorphism and must preserve validity")
    assert cc.monotone_zero(harness, relabelled, VERTEX_COUNT)

    with Solver(name="glucose4", bootstrap_with=cnf.clauses) as production:
        assert not cc.admits(production, edges, independent_witness)
        assert cc.admits(production, edges, relabelled)


def test_canonical_relabelling_is_a_permutation_of_the_same_multiset(
        independent_witness):
    relabelled = cc.canonical_relabelling(harness, independent_witness, VERTEX_COUNT)
    assert sorted(relabelled) == sorted(independent_witness)
    assert (sum(relabelled.values()) == sum(independent_witness.values())), (
        "a relabelling recolours nothing; the red-edge count is invariant")


def test_cross_check_passes_on_the_shipped_encoder():
    report = cc.cross_check(N, samples=60, seed=0)
    assert report["ok"] is True
    assert report["property_disagreements"] == []
    assert report["encoding_disagreements"] == []
    assert report["symmetry_break"]["relabelled_admitted"] is True


def test_cross_check_sampling_is_not_vacuous():
    """Uniform random colourings at this size are never valid, so a sampler
    that only draws them exercises the rejection direction and never asks
    whether the encoder ADMITS what it must. That was true of the first
    version of this check."""
    report = cc.cross_check(N, samples=60, seed=0)
    assert report["samples_accepted_by_property"] > 0


class _PatchedHarness:
    """The real harness with a deliberately broken build_instance."""

    def __init__(self, real, transform):
        self._real = real
        self._transform = transform

    def __getattr__(self, name):
        return getattr(self._real, name)

    def build_instance(self, n, verbose=False):
        cnf, edges, pool = self._real.build_instance(n, verbose)
        self._transform(cnf, edges)
        return cnf, edges, pool


def test_cross_check_detects_an_over_constrained_encoder(
        monkeypatch, independent_witness):
    """The failure this file exists for: an encoder that excludes a colouring
    it should admit. UNSAT from such an encoder is not evidence for the bound."""
    relabelled = cc.canonical_relabelling(harness, independent_witness, VERTEX_COUNT)

    def block_the_witness(cnf, edges):
        cnf.append([-edges[edge] if is_red else edges[edge]
                    for edge, is_red in relabelled.items()])

    monkeypatch.setattr(
        cc, "load_harness",
        lambda: _PatchedHarness(harness, block_the_witness))
    report = cc.cross_check(N, samples=20, seed=0)
    assert report["ok"] is False
    assert report["symmetry_break"]["relabelled_admitted"] is False

    # With no samples at all, the symmetry-break check is the ONLY thing that
    # can catch this. Random sampling may or may not draw the excluded
    # colouring; the deterministic check must not depend on it.
    blind = cc.cross_check(N, samples=0, seed=0)
    assert blind["encoding_disagreements"] == []
    assert blind["ok"] is False


def test_cross_check_detects_an_encoder_that_constrains_nothing(monkeypatch):
    """The opposite failure: a CNF that admits invalid colourings. SAT from it
    would not be a refutation either."""

    def drop_every_clause(cnf, edges):
        cnf.clauses.clear()

    monkeypatch.setattr(
        cc, "load_harness",
        lambda: _PatchedHarness(harness, drop_every_clause))
    report = cc.cross_check(N, samples=40, seed=0)
    assert report["ok"] is False
    assert report["encoding_disagreements"], (
        "an unconstrained CNF admits colourings the property rejects")


def test_naive_encoder_forbids_exactly_the_book_configurations(instance):
    """Read the independent encoder against the definition once, by hand:
    at n=3 a red B_2 is a red edge with 2 common red neighbours, and a blue
    B_3 is a blue edge with 3 common blue neighbours."""
    _cnf, edges, independent = instance
    others = VERTEX_COUNT - 2
    pairs = len(list(combinations(range(VERTEX_COUNT), 2)))
    expected = pairs * (
        len(list(combinations(range(others), N - 1)))
        + len(list(combinations(range(others), N))))
    assert len(independent) == expected
