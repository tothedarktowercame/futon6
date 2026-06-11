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


def test_problem9_miss_classes_are_scoped(tmp_path):
    p = tmp_path / "problem9-writeup.md"
    p.write_text(
        "Let A^(1), ..., A^(n) in R^{3x4} be Zariski-generic matrices.\n"
        "For alpha, beta, gamma, delta in [n], construct Q^(abgd).\n"
        "Fix camera-row pairs (gamma, k) and (delta, l).\n"
        "Take lambda_{abgd} = 1 for all non-identical tuples.\n"
        "Suppose X has rank 1.\n"
        "Define P(A^(1),...,A^(n)) = det[T^(a_m,b_n,g,d)]_{3x3}.\n\n"
        "    Q^(abgd)_{ijkl} = det[A^(a)(i,:); A^(b)(j,:)]\n",
        encoding="utf-8",
    )
    r = audit_writeup(p)
    labels = {label for s in r["scopes"] for label in s["hx/labels"]}
    assert r["scope-count"] >= 6
    assert {"let-decorated-list-binding", "for-list-binding", "fix-prose-binding",
            "take-relation-binding", "suppose-prose", "define-arglist-binding"} <= labels
    assert {"A", "alpha", "beta", "gamma", "delta", "lambda", "P"} <= set(r["bound-symbols"])
