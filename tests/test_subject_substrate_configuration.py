"""What subject configuration adds to the configurable WARP spine (tests/test_warp_spine_configuration.py).

The defaults, the one-override move of the vocabulary, the child environment and the S2 gate's
substrate are tested there. Here: a subject name stays a plain name (it becomes part of file
names), and the run manifest's substrate identity hashes the configured subject's files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import futon6_config as config  # noqa: E402

SUBJECT_VARIABLES = ("FUTON6_SUBJECT", "FUTON6_WARP_DIR", "FUTON6_SUBJECT_DATA")


def without_subject_settings() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in SUBJECT_VARIABLES}


def write_substrate(base: Path, subject: str, ids: list[str]) -> dict[str, str]:
    """A minimal subject bundle, unpacked, and the settings that point at it."""
    warp = base / "futon6/data/warp"
    warp.mkdir(parents=True)
    for name in ("concept-index.json", "def-snippets.json", "defined-index.json"):
        (warp / name).write_text("{}")
    (warp / "concept-usage.json").write_text(
        json.dumps({"paper_concepts": {paper: ["zeta function"] for paper in ids}}))
    data = base / "futon6/data"
    (data / f"concept-encyclopedia-{subject}.json").write_text("{}")
    return {"FUTON6_SUBJECT": subject, "FUTON6_WARP_DIR": str(warp), "FUTON6_SUBJECT_DATA": str(data)}


class SubjectSubstrateConfigurationTests(unittest.TestCase):
    def test_subject_must_be_a_plain_name(self):
        with patch.dict(os.environ, {**without_subject_settings(), "FUTON6_SUBJECT": "../ct"}, clear=True):
            with self.assertRaises(ValueError):
                config.subject()

    def test_manifest_substrate_identity_names_the_subjects_encyclopedia(self):
        import run_manifest

        with tempfile.TemporaryDirectory() as d:
            settings = write_substrate(Path(d), "rh-grh", ["2102.13459"])
            authority = Path(d) / "authority.json"
            authority.write_text("{}")
            futon3 = Path(d) / "futon3"
            (futon3 / "resources/sigils").mkdir(parents=True)
            (futon3 / "resources/sigils/patterns-index.tsv").write_text("")
            for family in ("math-informal", "math-informal-CT"):
                (futon3 / "library" / family).mkdir(parents=True)
                (futon3 / "library" / family / "p.flexiarg").write_text("")
            with patch.dict(os.environ, {**without_subject_settings(), **settings,
                                         "FUTON6_BACKGROUND_CORPUS_INDEX": str(authority),
                                         "FUTON3_ROOT": str(futon3)}, clear=True):
                identity = run_manifest.substrate_identity()
            self.assertIn("concept-encyclopedia-rh-grh.json", identity)
            self.assertIn("warp/concept-usage.json", identity)
            self.assertNotIn("concept-encyclopedia-ct.json", identity)


if __name__ == "__main__":
    unittest.main()
