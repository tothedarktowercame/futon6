import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import apm_proof_audit as apm


def synthetic_index():
    return {"terms": {"kirillov model": [{"term": "Kirillov model", "resolution-kind": "nnexus", "target": "nnexus:kirillov model", "domains": ["Wikipedia"]}]},
            "candidate-terms": []}


def test_unicode_math_and_bold_env_scopes(tmp_path):
    root = tmp_path / "problems" / "a00T01"
    root.mkdir(parents=True)
    p = root / "informal-solution.md"
    p.write_text(
        "**Definition.** Let f ∈ L^1 and ‖f‖_∞ ≤ 1.\n\n"
        "**Claim.** Fix α < ∞. Then ∫ f → 0. The Kirillov model appears.\n",
        encoding="utf-8",
    )
    r = apm.audit_apm(p, synthetic_index())
    assert r["expr-count"] >= 3
    assert r["scope-count"] >= 4
    assert {"large-operator", "arrow", "relation"} & set(r["expr-types"])
    assert any(s["hx/type"] == "env/definition" for s in r["scopes"])
    assert any(s["hx/type"] == "bind/let" for s in r["scopes"])
    assert any(e["term"] == "Kirillov model" for e in r["external-concepts"])


def test_lean_status_counts_sorries(tmp_path):
    lean = tmp_path / "lean-proofs"
    (lean / "a" ).mkdir(parents=True)
    (lean / "a" / "Main.lean").write_text("theorem x : True := by\n  trivial\n", encoding="utf-8")
    (lean / "b" ).mkdir(parents=True)
    (lean / "b" / "Main.lean").write_text("theorem x : True := by\n  sorry\n", encoding="utf-8")
    assert apm.lean_status("a", lean)["status"] == "sorry-free"
    b = apm.lean_status("b", lean)
    assert b["status"] == "sorry-carrying"
    assert b["sorry-count"] == 1
    assert apm.lean_status("c", lean)["status"] == "no-lean"
