import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mark4_apm_random_scope_disagreement as rnd


def test_sample_records_is_seeded_and_sorted():
    records = [{"id": f"p{i:02d}"} for i in range(10)]
    a = rnd.sample_records(records, seed=7, n=4)
    b = rnd.sample_records(records, seed=7, n=4)
    c = rnd.sample_records(records, seed=8, n=4)
    assert a == b
    assert a != c
    assert [r["id"] for r in a] == sorted(r["id"] for r in a)


def test_disagreement_summary_counts_scope_not_keyword():
    keyword_sets = {"p1": {"k1", "shared"}, "p2": {"k2"}}
    scope_sets = {"p1": {"s1", "shared"}, "p2": set()}
    got = rnd.disagreement_summary(keyword_sets, scope_sets, ["p1", "p2"])
    assert got["per_proof"]["p1"]["intersection"] == 1
    assert got["per_proof"]["p1"]["scope_not_keyword"] == 1
    assert got["per_proof"]["p1"]["keyword_not_scope"] == 1
    assert got["per_proof"]["p1"]["jaccard"] == 1 / 3
    assert got["proofs_with_scope_not_keyword"] == 1


def test_per_eprint_multichar_coverage_uses_type_and_multichar_symbol():
    proof = [
        {"hx/type": "bind/let", "hx/ends": [{"role": "symbol", "latex": "alpha"}]},
        {"hx/type": "bind/let", "hx/ends": [{"role": "symbol", "latex": "x"}]},
        {"hx/type": "quant/universal", "hx/ends": [{"role": "symbol", "latex": "beta"}]},
    ]
    eprint = [
        {"hx/type": "bind/let", "hx/ends": [{"role": "symbol", "latex": "alpha"}]},
        {"hx/type": "bind/let", "hx/ends": [{"role": "symbol", "latex": "x"}]},
        {"hx/type": "quant/existential", "hx/ends": [{"role": "symbol", "latex": "beta"}]},
    ]
    assert rnd.proof_eprint_multichar_coverage(proof, eprint) == 1 / 3
