"""Focused tests for FrontierMath proof-frame bridge helpers."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


def _load_bridge_module(root: Path):
    path = root / "scripts" / "frontiermath" / "advance-proof-cycle-from-frame-receipt.py"
    spec = importlib.util.spec_from_file_location("advance_proof_cycle_from_frame_receipt", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadAdminTokenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parent.parent
        cls.mod = _load_bridge_module(cls.root)

    def test_prefers_explicit_env(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            futon3c_root = Path(tmp_dir) / "futon3c"
            futon3c_root.mkdir()
            (futon3c_root / ".admintoken").write_text("from-file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"FUTON3C_ADMIN_TOKEN": "from-env"}, clear=False):
                self.assertEqual(self.mod.read_admin_token(futon3c_root), "from-env")

    def test_uses_file_when_env_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            futon3c_root = Path(tmp_dir) / "futon3c"
            futon3c_root.mkdir()
            (futon3c_root / ".admintoken").write_text("from-file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("FUTON3C_ADMIN_TOKEN", None)
                os.environ.pop("ADMIN_TOKEN", None)
                self.assertEqual(self.mod.read_admin_token(futon3c_root), "from-file")

    def test_falls_back_to_change_me(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("FUTON3C_ADMIN_TOKEN", None)
                os.environ.pop("ADMIN_TOKEN", None)
                self.assertEqual(
                    self.mod.read_admin_token(Path(tmp_dir) / "missing-futon3c-root"),
                    "change-me",
                )


if __name__ == "__main__":
    unittest.main()
