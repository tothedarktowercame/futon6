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
import importlib.util, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sfcg", ROOT / "scripts" / "sfc_symbol_grounding.py")
m = importlib.util.module_from_spec(spec); sys.modules["sfcg"] = m; spec.loader.exec_module(m)

CTX = (ROOT / "holes/excursions/fixtures/sfc-grounding-lclosure.txt").read_text()
FORMULA = r'\overline{M}=\{x\in X\mid \forall f,g:X\to Y\,.\,(f|_M=g|_M\,\Rw\,f\cdot x\cong g\cdot x)\}'

def test_stub_grounds_with_evidence_and_marks_undefined():
    r = m.ground(FORMULA, CTX, "stub", "stub")
    assert r["summary"]["symbols"] == 7
    g = {x["symbol"]: x for x in r["groundings"]}
    assert g["·"]["status"] == "undefined-in-context"   # unicode dot absent from raw TeX
    assert g["X"]["status"] == "grounded" and g["X"]["evidence"] in CTX  # evidence verbatim
    assert r["summary"]["unsupported"] == 0

def test_evidence_check_rejects_unsupported():
    bogus = [{"symbol": "Z", "binding": "b", "evidence": "NOT IN THE CONTEXT", "status": "grounded"}]
    out = m.check(bogus, CTX)
    assert out[0]["status"] == "unsupported"   # defeasible: unverifiable evidence rejected
