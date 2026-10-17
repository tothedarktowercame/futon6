"""S1 finds displays and environments where the typeset paper has them, not in line breaks or comments."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import anatomy_v0_sweep as sweep  # noqa: E402
import dp_paper_view as dpv  # noqa: E402
from dp_capabilities.math_envelope import DISPLAY_RE  # noqa: E402


class DisplayMathTests(unittest.TestCase):
    def displays(self, text):
        return [m.group(3) for m in DISPLAY_RE.finditer(text) if m.group(3) is not None]

    def test_a_line_break_with_spacing_is_not_a_display(self):
        # 1402.0031: "\\[.07in]" in an array opened a 45,301-character "display"
        self.assertEqual(self.displays(r"a & b \\[.07in] c & d \\[2pt] e. Then \[ x \] ."), [" x "])

    def test_real_displays_still_found(self):
        self.assertEqual(self.displays(r"so \[ a = b \] and \[c\]"), [" a = b ", "c"])
        self.assertEqual(self.displays("x\\\\\\[y\\] z"), ["y"])            # a line break, then a real \[
        self.assertEqual(self.displays("x\\\\\\\\[3pt] y\\]"), [])          # two line breaks, then [3pt]
        self.assertEqual(self.displays(r"\[ a \\ b \]"), [r" a \\ b "])        # a line break inside a display

    def test_environment_displays_unchanged(self):
        m = DISPLAY_RE.search(r"\begin{align*} a &= b \\[2pt] c &= d \end{align*}")
        self.assertEqual(m.group(1), "align")


class CommentTests(unittest.TestCase):
    def test_mask_keeps_every_offset_and_blanks_only_comments(self):
        text = "a % note\r\nb \\% kept % gone\n%whole line\n\nc"
        masked = sweep.mask_comments(text)
        self.assertEqual(len(masked), len(text))
        self.assertEqual(masked, "a       \r\nb \\% kept       \n           \n\nc")

    def test_commented_out_environments_are_not_structure(self):
        text = ("\\begin{lemma} A. \\end{lemma}\n"
                "%\\begin{proof} %old argument. %\\end{proof}\n"
                "\\begin{proof} Real. \\end{proof}\n")
        found = dpv.detect_tex_environments(sweep.mask_comments(text), 0)
        kinds = sorted((m["kind"], text[m["start"]:m["start"] + 13]) for m in found)
        self.assertEqual(kinds, [("env/lemma", "\\begin{lemma}"), ("env/proof", "\\begin{proof}")])
        proof = next(m for m in found if m["kind"] == "env/proof")
        self.assertTrue(text[proof["start"]:proof["end"]].startswith("\\begin{proof} Real."))


if __name__ == "__main__":
    unittest.main()
