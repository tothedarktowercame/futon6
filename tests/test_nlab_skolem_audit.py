"""Skolem audit over nlab wiring: scope grading, expression typing, and the
two suspect classes (floating entities / vacuous environments)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from nlab_skolem_audit import audit_page, classify_expr


def test_expression_typer_priority():
    assert classify_expr(r"\forall x. P(x)") == "quantifier"
    assert classify_expr(r"f \colon X \to Y") == "arrow"
    assert classify_expr(r"\sum_{i} a_i") == "large-operator"
    assert classify_expr(r"X = Y") == "relation"
    assert classify_expr(r"\alpha") == "greek"
    assert classify_expr("42") == "number"
    assert classify_expr("C") == "variable"


# Layout: paragraph 1 = an environment (strict); paragraph 2 has a binder
# (weak); paragraph 3 is plain prose (floating); paragraph 4 = an empty
# environment (vacuous).
TEXT = (
    "###### Definition\nLet $C$ be a [[category]].\n\n"      # env span [0,46)
    "Let $D$ be small. Then $D$ sits in [[Cat]].\n\n"        # binder para
    "The idea of $\\alpha$ relates to [[adjunction]].\n\n"   # floating para
    "###### Remark\nNothing bound here at all.\n"            # vacuous env
)

ENV1 = {"hx/type": "env/definition", "hx/content": {"position": 0, "length": 46}}
ENV2 = {"hx/type": "env/remark",
        "hx/content": {"position": TEXT.index("###### Remark"),
                       "length": len(TEXT) - TEXT.index("###### Remark")}}
BINDER = {"hx/type": "bind/typed",
          "hx/content": {"position": TEXT.index("Let $D$")}}

PAGE = {
    "page_id": "nlab-1",
    "page_name": "test",
    "environments": [ENV1, ENV2],
    "discourse": [BINDER],
}


def test_scope_grades_and_vacuous_envs():
    r = audit_page(PAGE, TEXT)
    assert r["expr-grades"] == {"strict": 1, "weak": 2, "floating": 1}
    assert r["link-grades"] == {"strict": 1, "weak": 1, "floating": 1}
    assert [v["type"] for v in r["vacuous-envs"]] == ["env/remark"]
    assert r["expr-types"]["greek"] == 1


def test_directive_links_excluded():
    text = "[[!include foo - contents]]\n\n[[bar]]\n"
    r = audit_page({"page_id": "nlab-2", "page_name": "t",
                    "environments": [], "discourse": []}, text)
    assert r["links"] == 1


# Mini-mission reading: the Idea section is HEAD; its entities are
# head-register (not floating), and its links owe DISCHARGE by a body
# section. [[monad]] is re-linked under Definition; "kleisli category" is
# discharged by plain-text mention (the nLab link-once convention);
# [[operad]] never returns at all.
MINI = (
    "## Idea\n\nA [[monad]] is like an [[operad]], cf. [[Kleisli category]],"
    " but $T$ simpler.\n\n"
    "## Definition\n\nFor a [[monad]] $T$ on $C$: the Kleisli category"
    " arises here, prose without binders.\n\n"
    "## References\n\n* [[Mac Lane]]\n"
)


def test_mini_mission_head_register_and_discharge():
    r = audit_page({"page_id": "nlab-3", "page_name": "t",
                    "environments": [], "discourse": []}, MINI)
    # Idea entities re-graded head-register; Definition prose (no binder,
    # no env) stays floating; References links are neither head nor body.
    assert r["link-grades"]["head-register"] == 3
    assert r["expr-grades"]["head-register"] == 1
    assert r["head-links"] == 3
    assert r["head-discharged"] == 2
    assert r["head-undischarged"] == ["operad"]

def test_unicode_math_classification_extensions():
    assert classify_expr("f → g") == "arrow"
    assert classify_expr("∫ f ≤ 1") == "large-operator"
    assert classify_expr("α") == "greek"
    assert classify_expr("x ∈ E") == "relation"
