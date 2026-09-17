"""Authority availability is a precondition, including off the dev host."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from concept_authority import ConceptAuthority
from build_mark7_substrate import build, MANIFEST, digest
import preflight


def valid_index():
    return {"schema-version": 2, "terms": {
        term: [{"term": term, "target": f"test:{term}", "resolution-kind": "page"}]
        for term in ("hom", "end", "colimit")
    }}


class AuthorityTests(unittest.TestCase):
    def test_missing_malformed_empty_and_unsuitable_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            with self.assertRaises(FileNotFoundError):
                ConceptAuthority(path)
            for data in ("not-json", "[]", json.dumps({"terms": {}}),
                         json.dumps({"schema-version": 2, "terms": {}}),
                         json.dumps({"schema-version": 2, "terms": {"hom": [{}]}}),
                         json.dumps({"schema-version": 2, "terms": {
                             "hom": valid_index()["terms"]["hom"]}})):
                path.write_text(data)
                with self.subTest(data=data), self.assertRaises(ValueError):
                    ConceptAuthority(path)

    def test_configuration_and_required_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps(valid_index()))
            with patch.dict(os.environ, {"FUTON6_BACKGROUND_CORPUS_INDEX": str(path)}):
                authority = ConceptAuthority()
                self.assertEqual(authority.resolve(r"\Hom")["target"], "test:hom")
                self.assertEqual(authority.resolve(r"\End")["target"], "test:end")
                self.assertEqual(authority.resolve(r"\colim")["target"], "test:colimit")
                self.assertIsNone(authority.resolve("nonexistent concept"))

    def test_preflight_refuses_missing_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"FUTON6_BACKGROUND_CORPUS_INDEX": directory + "/missing"}), \
                    patch.object(preflight, "R", []):
                preflight.check_concept_authority()
                self.assertEqual(preflight.R[0][0:2], ("substrate:concept-authority", False))

    def test_invalid_source_does_not_replace_existing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle.tgz"
            output.write_bytes(b"previous bundle")
            index = root / "invalid.json"
            index.write_text('{"schema-version": 2, "terms": {}}')
            with self.assertRaises(ValueError):
                build(root / "base.tgz", index, output)
            self.assertEqual(output.read_bytes(), b"previous bundle")

    def test_bundle_refuses_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.json"
            index.write_text(json.dumps(valid_index()))
            base = root / "base.tgz"
            with tarfile.open(base, "w:gz") as archive:
                link = tarfile.TarInfo("futon6/data/external")
                link.type = tarfile.SYMTYPE
                link.linkname = "/unshipped/data"
                archive.addfile(link)
            with self.assertRaisesRegex(ValueError, "dereference at source"):
                build(base, index, root / "output.tgz")

    def test_bundle_determinism_inventory_and_relocated_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "source.json"
            index.write_text(json.dumps(valid_index()))
            base = root / "base.tgz"
            with tarfile.open(base, "w:gz") as archive:
                item = tarfile.TarInfo("futon3/pattern.tsv")
                item.size = 7
                archive.addfile(item, io.BytesIO(b"pattern"))
            first, second = root / "first.tgz", root / "second.tgz"
            build(base, index, first)
            build(base, index, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            extracted = root / "extracted"
            with tarfile.open(first) as archive:
                manifest = json.load(archive.extractfile(MANIFEST))
                for name, record in manifest["files"].items():
                    data = archive.extractfile(name).read()
                    self.assertEqual(digest(data), record["sha256"])
                    self.assertEqual(len(data), record["bytes"])
                archive.extractall(extracted, filter="data")
            checkout = extracted / "renamed checkout"
            (extracted / "futon6").rename(checkout)
            (checkout / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/concept_authority.py", checkout / "scripts/concept_authority.py")
            shutil.copyfile(ROOT / "scripts/futon6_config.py", checkout / "scripts/futon6_config.py")
            env = dict(os.environ)
            env.pop("FUTON6_BACKGROUND_CORPUS_INDEX", None)
            result = subprocess.run([sys.executable, str(checkout / "scripts/concept_authority.py"),
                                     r"\Hom", r"\End", r"\colim"],
                                    env=env, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("unresolved", result.stdout)


if __name__ == "__main__":
    unittest.main()
