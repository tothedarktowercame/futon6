"""S1 must read a paper the way TeX does: in the master's order, in its encoding.

Both defects made S1 report "no proof identified" for papers full of proofs.
math/0608040 (Higher Topos Theory) joins 29 chapter files before its master, so
all 887 proofs sat in front of the first \\begin{document} and were discarded as
preamble. math/0310337, a Latin-1 French thesis, lost every accented letter to
UTF-8 decoding, so its "\\dem" proof heading and its Lemme/Théorème titles were
unrecognisable.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import anatomy_v0_sweep as sweep
import dp_paper_view as dpv


def tex(name, text):
    return {"file": name, "text": text}


def test_master_first_then_inputs_in_the_order_named_then_the_rest():
    files = [tex("chap2.tex", r"\begin{proof}B\end{proof}"),
             tex("appendix.tex", r"\begin{proof}Z\end{proof}"),
             tex("chap1.tex", r"\begin{proof}A\end{proof}"),
             tex("draft.tex", "unused"),
             tex("book.tex", "\\documentclass{book}\n\\begin{document}\n"
                             "\\include{chap1}\n\\input{chap2.tex}\n% \\input{draft}\n"
                             "\\include{appendix}\n\\end{document}\n")]
    assert [f["file"] for f in dpv.reading_order(files)] == \
        ["book.tex", "chap1.tex", "chap2.tex", "appendix.tex", "draft.tex"]


def test_files_input_in_the_preamble_precede_the_master():
    # Their macros must be preamble to the macro learners, not body.
    files = [tex("These.tex", "\\documentclass{book}\n\\input{Preambule.tex}\n"
                              "\\begin{document}\n\\input{Ch1.tex}\n\\end{document}\n"),
             tex("Ch1.tex", "\\dem Clair. \\findem"),
             tex("Preambule.tex", r"\newcommand{\dem}{\emph{Démonstration :}}")]
    assert [f["file"] for f in dpv.reading_order(files)] == ["Preambule.tex", "These.tex", "Ch1.tex"]


def test_nested_inputs_are_visited_depth_first_and_once():
    files = [tex("main.tex", "\\begin{document}\\input{part}\\input{b}\\end{document}"),
             tex("part.tex", "\\input{a}\\input{b}"), tex("a.tex", "A"), tex("b.tex", "B")]
    assert [f["file"] for f in dpv.reading_order(files)] == ["main.tex", "part.tex", "a.tex", "b.tex"]


def test_a_single_file_paper_is_unchanged():
    files = [tex("main.tex", "\\begin{document}x\\end{document}")]
    assert dpv.reading_order(files) == files


def test_eight_bit_sources_decode_by_their_declared_encoding():
    assert sweep.safe_decode("Théorème".encode("utf-8")) == "Théorème"
    # Undeclared: cp1252, which keeps the accent that UTF-8-with-ignore deleted.
    assert sweep.safe_decode("Démonstration".encode("latin-1")) == "Démonstration"
    cyrillic = "\\usepackage[cp1251]{inputenc}\nДоказательство".encode("cp1251")
    assert sweep.safe_decode(cyrillic).endswith("Доказательство")
    # A chapter declares nothing; the paper-level declaration applies.
    assert sweep.safe_decode("Доказательство".encode("cp1251"), "cp1251") == "Доказательство"


def test_a_commented_out_declaration_is_not_the_encoding():
    raw = b"% \\usepackage[applemac]{inputenc}\n\\usepackage[latin1]{inputenc}\n"
    assert sweep.declared_encoding(raw) == "latin-1"


def test_french_theorem_titles_and_environments_canonicalise():
    pre = ("\\newtheorem{lemme}{Lemme}\n\\newtheorem{theoreme}{Théorème}\n"
           "\\newtheorem{thm}{Th\\'eor\\`eme}\n\\newtheorem{corollaire}{Corollaire}\n")
    learned = dpv.learn_environment_names(pre + "\\begin{document}\\end{document}")
    assert learned == {"lemme": "lemma", "theoreme": "theorem", "thm": "theorem",
                       "corollaire": "corollary"}
    assert dpv._canon("Théorème", {}) == "theorem"
