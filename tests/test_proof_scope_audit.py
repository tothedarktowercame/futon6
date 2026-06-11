import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from nlab_skolem_audit import classify_expr
from proof_scope_audit import audit_writeup, expression_records


def test_ascii_expression_classifier_extensions():
    assert classify_expr("Phi_n(p) >= 0") == "relation"
    assert classify_expr("sum_i lambda_i") == "large-operator"
    assert classify_expr("lambda_i") == "greek"
    assert classify_expr("phi |-> phi + psi") == "arrow"


def test_expression_records_include_indented_display_math():
    text = "Proof\n\n    Phi_n(p) = sum_i lambda_i\n    phi |-> phi + psi\n"
    exprs = expression_records(text)
    vals = {e["expr"] for e in exprs}
    assert "Phi_n(p) = sum_i lambda_i" in vals
    assert "phi |-> phi + psi" in vals
    assert {e["type"] for e in exprs} >= {"large-operator", "arrow"}


def test_writeup_register_prose_binders(tmp_path):
    p = tmp_path / "problemX-writeup.md"
    p.write_text(
        "# Synthetic proof\n\n"
        "For a monic polynomial p(x) of degree n, set a = b.\n"
        "Let Phi_n(p) be the root separation energy where lambda_i = roots.\n\n"
        "    Phi_n(p) = sum_i lambda_i\n\n"
        "WLOG set a_1 = 0. Then Phi_n(p) >= 0.\n",
        encoding="utf-8",
    )
    r = audit_writeup(p)
    assert r["expr-count"] > 0
    assert r["scope-count"] >= 4
    assert "p" in r["bound-symbols"]
    assert "a" in r["bound-symbols"]
    assert any(s["hx/type"] == "assume/wlog" for s in r["scopes"])
