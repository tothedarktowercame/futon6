"""Refusal tests for experiment-stage and replication obligations."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "clean_to_lean.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("clean_to_lean", SCRIPT)
clean_to_lean = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean_to_lean)


def registered_slice5():
    registration = clean_to_lean.load_clean(
        ROOT / "holes" / "clean" / "slice5-confirmation.clean.edn"
    )
    design = clean_to_lean.plain_map(registration["experiment-design"])
    design["seeds"] = clean_to_lean.plain_map(design["seeds"])
    registration["experiment-design"] = design
    return registration


def test_pilot_with_predecessor_refuses():
    registration = copy.deepcopy(registered_slice5())
    replication = registration["experiment-design"]["seeds"]
    replication["stage"] = "pilot"
    with pytest.raises(
        ValueError, match="pilot replication plan must have :predecessor nil"
    ):
        clean_to_lean.validate_experiment(registration)


def test_measured_variation_without_floor_refuses():
    registration = copy.deepcopy(registered_slice5())
    replication = registration["experiment-design"]["seeds"]
    replication.update({
        "stage": "pilot",
        "predecessor": None,
        "seedable?": False,
        "variation": "measured",
    })
    replication.pop("floor-endpoint", None)
    with pytest.raises(
        ValueError, match="measured variation requires a named :floor-endpoint"
    ):
        clean_to_lean.validate_experiment(registration)


def test_nonseedable_pilot_uses_problem_identifiers():
    registration = clean_to_lean.load_clean(
        ROOT / "tests" / "fixtures" / "nonseedable-pilot.clean.edn"
    )
    rendered = clean_to_lean.emit_experiment(registration)
    assert "inductive ProblemId where" in rendered
    assert "ReplicationPlan.pilot" in rendered
    assert "VariationPlan.measured identityFloorEndpoint" in rendered


def test_nonnavigable_treatment_still_refuses():
    registration = clean_to_lean.load_clean(
        ROOT / "tests" / "fixtures" / "slice5-nonnavigable-treatment.clean.edn"
    )
    with pytest.raises(ValueError, match="non-navigable treatment refused"):
        clean_to_lean.validate_experiment(registration)
