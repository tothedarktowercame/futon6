import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import proof_tex_audit


def test_harvest_and_strip_macros_keeps_clean_latex_and_gold_types():
    clean, anns = proof_tex_audit.harvest_and_strip_macros(
        r"\Phi^{\mNumber{4}}_{\mNumber{3}} \mRelation{\in} A"
    )

    assert clean == r"\Phi^{4}_{3} \in A"
    assert [a["type"] for a in anns] == ["number", "number", "relation"]
    assert [a["text"] for a in anns] == ["4", "3", r"\in"]


def test_math_spans_tokenize_inline_and_display_register():
    text = r"Prose \(x \mRelation{=} y\) and \[ \mLargeOperator{\sum}_{i=1}^n a_i \]."
    spans = proof_tex_audit.math_spans(text)

    assert len(spans) == 2
    assert text[spans[0][0]:spans[0][1]] == r"x \mRelation{=} y"
    assert spans[0][2] == "inline"
    assert r"\mLargeOperator" in text[spans[1][0]:spans[1][1]]
    assert spans[1][2] == "display"


def test_gold_diff_reports_agreement_and_disagreement(tmp_path, monkeypatch):
    tex = tmp_path / "problem1-solution-full.tex"
    tex.write_text(
        r"""\section{Problem Statement}
Let \(x\) be a number.
\subsection{Solution}
We have \(x \mRelation{=} y\) and also \(\mNumber{4} + x\)
and \(z \mArrow{=} w\).
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(proof_tex_audit.proof_scope_audit, "detect_scopes", lambda *_: [])

    result = proof_tex_audit.audit_tex(tex)

    # Token-grain gold diff (review fix): each \m* annotation is compared
    # against classify_expr of the annotated TOKEN, not the expression's
    # dominant type. "=" as mRelation agrees; "4" as mNumber agrees; "=" as
    # mArrow (deliberately mis-annotated) disagrees.
    assert result["gold-annotated-count"] == 3
    assert result["gold-agree-count"] == 2
    assert round(result["gold-agreement-rate"], 1) == 66.7
    (d,) = result["gold-disagreements"]
    assert d["token"] == "="
    assert d["gold-type"] == "arrow"
    assert d["classified-type"] == "relation"

