"""Shared test configuration — skip data-dependent tests when fixtures are missing."""

import os
import sys
import pytest

# Ensure src/ is on the path so `import futon6` works without pip install -e .
_SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if os.path.abspath(_SRC) not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

_CATTHEORY_EDN = os.path.expanduser("~/code/planetmath/category-theory.edn")
_SKIP_MSG = f"PlanetMath fixture not found: {_CATTHEORY_EDN}"

_DATA_DEPENDENT_CLASSES = {
    "TestLoadEdn", "TestEntitiesToGraph", "TestBuildGraph",
    "TestBuildGraphWithTex", "TestMergeTexBodies",
    "TestEnrichReal", "TestEnrichWithTex",
}


def pytest_collection_modifyitems(config, items):
    if os.path.exists(_CATTHEORY_EDN):
        return
    skip = pytest.mark.skip(reason=_SKIP_MSG)
    for item in items:
        cls = item.cls
        if cls and cls.__name__ in _DATA_DEPENDENT_CLASSES:
            item.add_marker(skip)
