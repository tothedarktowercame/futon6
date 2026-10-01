import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mark3_extract_candidates as extract


def write_snapshot(root: Path, papers=("p1",), payload=b'{"paper-id":"p1"}'):
    candidate = root / "p1__p0.candidate.json"
    candidate.write_bytes(payload)
    manifest = {
        "schema": extract.SNAPSHOT_SCHEMA,
        "requested-papers": list(papers),
        "all-proofs": True,
        "files": [{"path": candidate.name,
                   "sha256": hashlib.sha256(payload).hexdigest()}],
        "papers": [{"paper-id": "p1", "proof-id": "p1__p0"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class CandidateSnapshot(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_candidates_are_reused_only_when_identity_and_bytes_match(self):
        expected = write_snapshot(self.root)
        self.assertEqual(extract.frozen_snapshot(self.root, ["p1"], all_proofs=True), expected)
        (self.root / "p1__p0.candidate.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            extract.frozen_snapshot(self.root, ["p1"], all_proofs=True)

    def test_frozen_candidates_refuse_a_different_corpus_or_extra_file(self):
        write_snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "run corpus"):
            extract.frozen_snapshot(self.root, ["p2"], all_proofs=True)
        (self.root / "p1__p1.candidate.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "file set"):
            extract.frozen_snapshot(self.root, ["p1"], all_proofs=True)

    def test_cli_resume_does_not_call_extraction(self):
        write_snapshot(self.root)
        ids = self.root / "ids.txt"
        ids.write_text("p1\n")
        with patch.object(sys, "argv", ["mark3_extract_candidates.py", "--list", str(ids),
                                        "--all-proofs", "--reuse-frozen", "--out", str(self.root)]), \
                patch.object(extract, "record_reused_snapshot") as accounted, \
                patch.object(extract, "extract_all", side_effect=AssertionError("must not extract")):
            self.assertEqual(extract.main(), 0)
        accounted.assert_called_once()


if __name__ == "__main__":
    unittest.main()
