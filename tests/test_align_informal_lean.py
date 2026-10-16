"""The informal-Lean alignment prototype: declarations, docstrings, step matching."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import align_informal_lean as A  # noqa: E402

LEAN = '''import Mathlib

/-- An unrelated earlier docstring about circles. -/
lemma t98A01_circleFive_isCoveringMap : True := trivial

def t98A01IsRegularNfoldCover (n : ℕ) : Prop := True

/-- Euler multiplicativity for five sheets forces both genera to be one. -/
lemma t98A01_orientable_genus_arithmetic (g h : ℕ) (hgh : g = h)
    (hχ : eulerCharacteristic true g = 5 * eulerCharacteristic true h) :
    g = 1 ∧ h = 1 := by
  exact foo t98A01_circleFive_isCoveringMap

theorem apm_t98a01 : True := trivial
'''


def graph():
    def node(i, kind, text, gloss):
        return {"id": i, "kind": kind, "text": text, "gloss": gloss,
                "terms": A.tokens(text) + A.tokens(gloss)}
    nodes = [node("n1", "object", "a regular 5-sheeted covering", "A regular 5-sheeted covering."),
             node("n2", "claim", r"$\chi(\Sigma)=5\chi(\Sigma')$", "Euler characteristic multiplies under a cover."),
             node("n3", "claim", r"$g=h$", "Homeomorphic orientable surfaces have equal genus."),
             node("n4", "claim", r"$g=1$", "If the Euler characteristic is 0, then the genus is 1."),
             node("n5", "claim", r"$z\mapsto z^5$", "The map z to z^5 covers the circle.")]
    edges = [{"id": "e1", "premise": ["n2", "n3"], "conclusion": "n4", "missing": True,
              "warrant": "If the Euler characteristic is 0, then the genus is 1"},
             {"id": "e2", "premise": ["n1"], "conclusion": "n5", "missing": False,
              "warrant": "the circle map is a covering"}]
    return {"graph": "t98A01__p0.edn", "nodes": nodes, "edges": edges}


def test_declarations_take_their_own_docstring_and_uses(tmp_path):
    f = tmp_path / "Main.lean"
    f.write_text(LEAN)
    decls = {d["name"]: d for d in A.lean_decls(f, "t98A01")}
    assert decls["t98A01_orientable_genus_arithmetic"]["doc"].startswith("Euler multiplicativity")
    assert decls["t98A01IsRegularNfoldCover"]["doc"] == ""          # not the circle docstring above it
    assert decls["t98A01_orientable_genus_arithmetic"]["uses"] == ["t98A01_circleFive_isCoveringMap"]


def test_a_lemma_is_matched_to_the_step_it_proves(tmp_path):
    f = tmp_path / "Main.lean"
    f.write_text(LEAN)
    decls = A.lean_decls(f, "t98A01")
    g = graph()
    model = A.Tfidf([d["terms"] for d in decls] + [n["terms"] for n in g["nodes"]])
    res = A.align_paper(g, decls, model, threshold=0.1)
    hole = res["holes"][0]
    assert hole["lemma"] == "t98A01_orientable_genus_arithmetic" and hole["matched"]
    step = res["non_hole_steps"][0]
    assert step["lemma"] == "t98A01_circleFive_isCoveringMap"
    # each lemma answers for one step only
    lemmas = [r["lemma"] for r in res["holes"] + res["non_hole_steps"] if r["lemma"]]
    assert len(lemmas) == len(set(lemmas))


def test_tokens_meet_across_latex_unicode_and_lean_names():
    assert "chi" in A.tokens(r"$\chi(\Sigma)$") and "chi" in A.tokens("χ(Σ)")
    assert {"euler", "chi"} <= set(A.tokens("eulerCharacteristic"))
    assert "cover" in A.tokens("isCoveringMap") and "cover" in A.tokens("coverings")
