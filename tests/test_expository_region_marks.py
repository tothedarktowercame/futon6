"""Region carving given S1 marks: author-macro environments are formal blocks."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import expository_region_extract as expo

TEXT = "\n".join([
    r"\begin{document}",                                            # 1
    r"\section{Basics of $S^{\perp_{\infty}}$}",                    # 2
    r"We recall the following definition from the literature here.",  # 3
    "",                                                             # 4
    r"\df{A thing is \textit{nice} if it is nice.}",                # 5
    "",                                                             # 6
    r"Niceness was first studied by several authors long ago.",     # 7
    "",                                                             # 8
    r"\prop{Every thing is nice.}",                                 # 9
    r"\prf Write $x$ for the thing and consider it closely. $$x=x$$", # 10
    r"Then applying the rule twice gives what we want here.",       # 11
    r"$$y=y$$ \eprf",                                               # 12
    "",                                                             # 13
    r"\eeg",                                                        # 14
    "",                                                             # 15
    r"In the next section we give some examples of nice things.",   # 16
    r"\subsection{Examples}",                                       # 17
    r"Here are some examples of the notion we have just defined.",  # 18
    r"\df{A \textit{nicer} thing is nicer than a nice thing.}",     # 19
    r"Nicer things are rarer than nice ones, as one might expect.", # 20
    r"\prop{Every nicer thing is nice.}",                           # 21
    r"\end{document}",                                              # 22
])


def env(kind, first, last):
    starts = [0]
    for line in TEXT.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    return {"kind": f"env/{kind}", "start": starts[first - 1], "end": starts[last] - 1}


MARKS = [env("definition", 5, 5), env("proposition", 9, 9), env("proof", 10, 12),
         env("definition", 19, 19), env("proposition", 21, 21)]


def regions(marks):
    return [(r["type"], r["line_start"], r["line_end"]) for r in expo.extract_regions("p", TEXT, marks)["regions"]]


def test_section_titles_may_hold_nested_braces():
    titles = [s.title for s in expo.parse_sections(TEXT.split("\n"), 1, 22)]
    assert titles == [r"Basics of $S^{\perp_{\infty}}$", "Examples"]


def test_macro_environments_from_s1_marks_bound_the_prose_around_them():
    got = regions(MARKS)
    assert ("section-lead", 3, 3) in got                  # before the first formal block
    assert ("inflight", 7, 7) in got                      # between the definition and the proposition
    assert ("section-tail", 16, 16) in got                # after the last one, before the subsection
    assert ("section-lead", 18, 18) in got
    assert ("inflight", 20, 20) in got
    assert all(lo != 14 for _, lo, _ in got)              # \eeg alone is not prose
    assert len(got) == len(set(got))                      # the subsection does not repeat its parent's gaps


def test_prose_between_displays_in_a_proof_is_typed_in_proof():
    got = regions(MARKS)
    assert ("in-proof", 11, 11) in got
    assert not any(t in ("inflight", "section-lead", "section-tail") and lo <= 11 <= hi for t, lo, hi in got)


def test_without_marks_the_carving_is_unchanged_in_kind():
    got = regions(None)
    assert {t for t, _, _ in got} <= {"leaf-section", "inflight"}
