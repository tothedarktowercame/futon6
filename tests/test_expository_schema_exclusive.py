"""S4's decoding schema itself says a scope has exactly one of fill and held_reason.

The rule was in the prompt and in problems(), not in the grammar, and the model
broke it in 29% of apm-501-xi-r1's regions; refusing them took S4 under its floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import expository_json  # noqa: E402

UNITS = [{"id": "L1-aaaa"}, {"id": "L2-bbbb"}]


def doc(fill, held):
    return {"scopes": [{"kind": "connection", "units": ["L1-aaaa"], "fill": fill, "held_reason": held}]}


def admits(branch, scope):
    """The string-length constraints of one anyOf branch (all this rule needs)."""
    for key in ("fill", "held_reason"):
        spec, value = branch["properties"][key], scope[key]
        if not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", 10**9):
            return False
    return True


@pytest.mark.parametrize("units", [UNITS, ()])
def test_schema_admits_exactly_one_of_fill_and_held(units):
    items = expository_json.schema(1, 2, {"connection"}, units)["properties"]["scopes"]["items"]
    def ok(fill, held):
        scope = doc(fill, held)["scopes"][0]
        return any(admits(b, scope) for b in items["anyOf"])
    assert ok("Cauchy-Schwarz", "")
    assert ok("", "the passage does not say")
    assert not ok("PDE", "No")
    assert not ok("", "")
    for branch in items["anyOf"]:
        assert set(branch["required"]) >= {"kind", "fill", "held_reason"}
        assert branch["properties"]["kind"]["enum"] == ["connection"]
