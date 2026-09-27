# ---------------------------------------------------------------------------
# KNOWN FAILING WITHOUT THE INPUTS BELOW (recorded 2026-09-23)
#
# Documented rather than repaired. Each cause is stated so a reader can tell a
# missing input or upstream drift from a defect in the code under test.
#
# 1 failure - babashka:
#   `bb` (babashka) is not on PATH. The gate chain shells out to it and dies
#   with FileNotFoundError before any assertion runs. babashka is a required
#   tool for these tests, not an optional one - see scripts/preflight.py, which
#   refuses without it. Put `bb` on PATH and these pass.
# ---------------------------------------------------------------------------
"""The scope audit names the ways an accepted scope can carry nothing."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import expository_scope_audit as audit

DEFS = ["the structure/theory connected to", "the example instantiating the structure"]
LINE = "in fact the bicategory formed by applying Street's two-sided Grothendieck construction"


def test_each_check_names_its_own_failure():
    assert audit.flags("Street's two-sided Grothendieck construction", LINE, DEFS) == []
    assert "echo" in audit.flags("the structure/theory of functors", "functors", DEFS)
    assert audit.flags("the monoidal centre of a monoidal category", LINE, DEFS) == ["unanchored"]
    assert audit.flags("a bicategory", LINE, DEFS) == ["bare-noun"]


def test_run_shape_counts_bare_parents_and_holds(tmp_path):
    expo, cands = tmp_path / "expo", tmp_path / "cands"
    expo.mkdir(), cands.mkdir()
    (cands / "p.r1.candidate.json").write_text(json.dumps(
        {"window-lines": [10, 11], "source-window": LINE + "\nsecond line"}))
    (expo / "p_r1_L10-11.edn").write_text(
        '{:paper/id "p" :passage/id "p:r1:L10-11" :scopes ['
        '{:id :s1 :kind :connection :source {:lines [10 10]} '
        ':slot-fill {:other-structure "Street\'s two-sided Grothendieck construction"}} '
        '{:id :s2 :kind :connection/example-source :source {:lines [11 11]} '
        ':slot-fill {:example "a diagram"}} '
        '{:id :s3 :kind :generalisation :source {:lines [11 11]} :held-reason "not stated"}]}')
    result = audit.audit(expo, cands)
    s = result["summary"]
    assert (s["scopes"], s["filled"], s["held"], s["unflagged"]) == (3, 2, 1, 1)
    assert s["by-check"]["bare-noun"] == 1 and s["bare-parent-share"] == round(1 / 3, 3)
    by = {r["scope"]: r for r in result["scopes"]}
    assert by["s1"]["bare-parent"] and not by["s2"]["bare-parent"]
    assert by["s2"]["flags"] == ["unanchored", "bare-noun"]      # "diagram" is not on line 11


def test_quoted_share_counts_only_scopes_that_cite_units(tmp_path):
    """Whether a fill is the passage's words is measured here, not gated upstream.

    A scope citing no unit is left out of the denominator rather than counted as
    paraphrase: a v1 graph carries no units at all, and would otherwise read as
    0% quoted when the truth is that nothing was asked of it.
    """
    expo, cands = tmp_path / "expo", tmp_path / "cands"
    expo.mkdir(), cands.mkdir()
    (cands / "p.r1.candidate.json").write_text(json.dumps(
        {"window-lines": [10, 11], "source-window": LINE + "\nsecond line"}))
    (expo / "p_r1_L10-11.edn").write_text(
        '{:paper/id "p" :passage/id "p:r1:L10-11" :scopes ['
        # quoted: cites a unit and carries the extent of its own words
        '{:id :s1 :kind :connection :source {:lines [10 10] :units ["L10-aaaa"]} '
        ':fill-span [3 20] :slot-fill {:other-structure "Street\'s two-sided Grothendieck construction"}} '
        # the model's words: cites a unit, no span
        '{:id :s2 :kind :connection :source {:lines [10 10] :units ["L10-aaaa"]} '
        ':slot-fill {:other-structure "a diagram"}} '
        # cites nothing, so it is not evidence either way
        '{:id :s3 :kind :connection :source {:lines [11 11]} '
        ':slot-fill {:other-structure "a diagram"}}]}')
    s = audit.audit(expo, cands)["summary"]
    assert (s["citing-units"], s["quoted"], s["quoted-share"]) == (2, 1, 0.5)


def test_quoted_share_is_unknown_not_zero_when_nothing_cites_units(tmp_path):
    expo, cands = tmp_path / "expo", tmp_path / "cands"
    expo.mkdir(), cands.mkdir()
    (cands / "p.r1.candidate.json").write_text(json.dumps(
        {"window-lines": [10, 11], "source-window": LINE + "\nsecond line"}))
    (expo / "p_r1_L10-11.edn").write_text(
        '{:paper/id "p" :passage/id "p:r1:L10-11" :scopes ['
        '{:id :s1 :kind :connection :source {:lines [10 10]} '
        ':slot-fill {:other-structure "a diagram"}}]}')
    s = audit.audit(expo, cands)["summary"]
    assert (s["citing-units"], s["quoted"], s["quoted-share"]) == (0, 0, None)
