import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dp_paper_view as dpv


def test_text_style_proof_region_is_detected():
    text = r"""\begin{document}
Theorem. Every compact object is finite.

\emph{Proof.} Since every cover has a finite subcover, the claim follows. \qed
\end{document}
"""
    marks = dpv.detect_text_proofs(text)
    assert len(marks) == 1
    assert marks[0]["kind"] == "env/proof"
    assert "\\qed" in text[marks[0]["start"]:marks[0]["end"]]


def test_capitalized_proof_environment_canonicalizes_to_proof():
    text = r"\begin{Proof} Hence the desired equality follows. \end{Proof}"
    marks = dpv.detect_tex_environments(text, 0)
    assert len(marks) == 1
    assert marks[0]["kind"] == "env/proof"
