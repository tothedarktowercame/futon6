"""Focused tests for FrontierMath proof-frame workspace helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrameWorkspaceInitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parent.parent
        cls.init_mod = _load_module(
            cls.root / "scripts" / "frontiermath" / "init-proof-frame-workspace.py",
            "init_proof_frame_workspace",
        )
        cls.emit_mod = _load_module(
            cls.root / "scripts" / "frontiermath" / "emit-proof-frame-receipt.py",
            "emit_proof_frame_receipt",
        )
        cls.promote_mod = _load_module(
            cls.root / "scripts" / "frontiermath" / "promote-proof-frame-lean.py",
            "promote_proof_frame_lean",
        )

    def test_build_metadata_and_templates(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            apm_root = Path(tmp_dir) / "apm-lean"
            metadata = self.init_mod.build_metadata("a02J04", "runner-1", apm_lean_root=apm_root)
            self.assertEqual(metadata["proof/problem-id"], "a02J04")
            self.assertIn("ApmCanaries.Frames.A02J04", metadata["frame/module-root"])
            self.assertTrue(metadata["artifacts"]["lean-main"].endswith("Main.lean"))
            self.assertTrue(metadata["artifacts"]["formal-alignment"].endswith("formal-alignment.edn"))
            template = self.init_mod.formal_alignment_template("a02J04")
            self.assertIn(":sanity-check", template)
            self.assertIn(":avoids-assuming-conclusion? false", template)

    def test_receipt_can_embed_workspace_map(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace.json"
            workspace.write_text(
                json.dumps(
                    {
                        "frame/workspace-root": "/tmp/ws",
                        "frame/module-root": "ApmCanaries.Frames.A02J04.FRunner",
                        "frame/lean-root": "/tmp/apm-lean/ApmCanaries/Frames/A02J04/FRunner",
                        "frame/shared-extension-root": "/tmp/apm-lean/ApmCanaries/Local",
                        "artifacts": {
                            "proof-plan": "/tmp/ws/proof-plan.edn",
                            "formal-alignment": "/tmp/ws/formal-alignment.edn",
                            "changelog": "/tmp/ws/changelog.edn",
                            "execute-notes": "/tmp/ws/execute.md",
                            "lean-main": "/tmp/apm-lean/ApmCanaries/Frames/A02J04/FRunner/Main.lean",
                            "lean-scratch": "/tmp/apm-lean/ApmCanaries/Frames/A02J04/FRunner/Scratch.lean",
                            "workspace-metadata": str(workspace),
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = self.emit_mod.load_workspace_metadata(str(workspace))
            self.assertEqual(loaded["frame/module-root"], "ApmCanaries.Frames.A02J04.FRunner")

    def test_promotion_destination_must_live_under_local_namespace(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            apm_root = Path(tmp_dir) / "apm-lean"
            with self.assertRaises(SystemExit):
                self.promote_mod.module_to_path(apm_root, "ApmCanaries.Frames.Bad.Main")


if __name__ == "__main__":
    unittest.main()
