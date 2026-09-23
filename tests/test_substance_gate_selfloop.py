# ---------------------------------------------------------------------------
# KNOWN FAILING WITHOUT THE INPUTS BELOW (recorded 2026-09-23)
#
# Documented rather than repaired. Each cause is stated so a reader can tell a
# missing input or upstream drift from a defect in the code under test.
#
# 1 failure - genuine, NOT a missing input:
#   test_nonselfloop_not_flagged asserts that a graph whose conclusion :B is NOT
#   among its premises [:A] is not flagged as a self-loop. It IS being flagged.
#   Pure Python (substance_gate.iatc_features / check_iatc_item), with no external
#   tool and no data fixture involved, so this is a real over-flagging bug in the
#   self-loop check and worth fixing on its own terms.
# ---------------------------------------------------------------------------
"""rung-1 regression: the substance self-loop check must read ALL premise tokens,
not just the first — so :premise [:A :B] :conclusion :B (conclusion is the 2nd
premise) is caught. See substance_gate.py self-loop block."""
from __future__ import annotations
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sg", ROOT / "scripts" / "substance_gate.py")
sg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sg)


def _graph(premise: str) -> str:
    nodes = ('{:id :A, :kind :object, :text "a", :source {:lines [1 1]}} '
             '{:id :B, :kind :claim, :text "b", :source {:lines [1 1]}}')
    edge = (f'{{:id :e1, :kind :infer, :relation :therefore, :premise {premise}, '
            f':warrant {{:kind :missing-warrant}}, :conclusion :B, :source {{:lines [1 1]}}}}')
    return (f'{{:paper/id "t", :passage/id "p", :source {{:lines [1 1]}}, '
            f':nodes [{nodes}], :edges [{edge}], :holes []}}')


def _is_selfloop_flagged(text: str) -> bool:
    feats = sg.iatc_features(text)
    fails = sg.check_iatc_item(pathlib.Path("t.edn"), text, feats)
    return any("self-loop" in f for f in fails)


def test_multipremise_selfloop_caught():
    # conclusion :B is the SECOND premise token — the previous first-token-only
    # check missed exactly this class (0712.0724's :e-functor-pitchfork).
    assert _is_selfloop_flagged(_graph("[:A :B]"))


def test_singlepremise_selfloop_still_caught():
    assert _is_selfloop_flagged(_graph(":B"))


def test_nonselfloop_not_flagged():
    # :B is NOT among the premises -> not a self-loop.
    assert not _is_selfloop_flagged(_graph("[:A]"))
