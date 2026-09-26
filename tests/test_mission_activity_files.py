from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mission_activity.py"
sys.path.insert(0, str(SCRIPT.parent))  # for futon6_config
SPEC = importlib.util.spec_from_file_location("mission_activity", SCRIPT)
assert SPEC and SPEC.loader
mission_activity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mission_activity
SPEC.loader.exec_module(mission_activity)


NS_INDEX = {
    "alpha.core": ("futon1", "src/alpha/core.clj"),
    "beta.util": ("futon2", "src/beta/util.clj"),
}


class MissionCodeFilesTest(unittest.TestCase):
    def test_two_vars_in_same_file_give_one_entry(self) -> None:
        files, unres = mission_activity.mission_code_files(
            ["alpha.core/foo", "alpha.core/bar"], NS_INDEX
        )
        self.assertEqual(files, [["futon1", "src/alpha/core.clj"]])
        self.assertEqual(unres, 0)

    def test_unresolvable_var_absent_and_counted(self) -> None:
        files, unres = mission_activity.mission_code_files(
            ["alpha.core/foo", "missing.ns/baz", "no/slash/ok/extra"], NS_INDEX
        )
        self.assertEqual(files, [["futon1", "src/alpha/core.clj"]])
        self.assertEqual(unres, 2)

    def test_list_is_sorted(self) -> None:
        files, unres = mission_activity.mission_code_files(
            ["beta.util/x", "alpha.core/y"], NS_INDEX
        )
        self.assertEqual(files, [
            ["futon1", "src/alpha/core.clj"],
            ["futon2", "src/beta/util.clj"],
        ])
        self.assertEqual(unres, 0)


if __name__ == "__main__":
    unittest.main()
