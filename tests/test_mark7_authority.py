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


class NamesAsPapersWriteThem(unittest.TestCase):
    """The authority stores "fubini theorem"; mathematics writes "Fubini's theorem",
    "Hahn-Banach" with an en dash, and "Arzela-Ascoli" with its accent. 91 of the 845
    APM informal proofs say "X's theorem" at least once (Joe, 2026-09-22)."""

    @classmethod
    def setUpClass(cls):
        cls.ca = ConceptAuthority()

    def resolved(self, term):
        hit = self.ca.resolve(term)
        return hit.get("target") if hit else None

    def test_a_result_named_after_a_person_resolves_in_the_possessive(self):
        for possessive in ("Fubini's theorem", "Urysohn's lemma", "Sard's theorem",
                           "Liouville's theorem", "Fatou's lemma", "Young's inequality"):
            self.assertIsNotNone(self.resolved(possessive), possessive)

    def test_the_possessive_falls_back_to_the_stored_form_only_when_it_has_to(self):
        # nLab titles its own entry "Urysohn's lemma", so the possessive is a stored
        # name there and must keep resolving to it; "Fubini's theorem" is stored only
        # in the plain form, and falls back to that.
        self.assertEqual(self.resolved("Fubini's theorem"), self.resolved("Fubini theorem"))
        self.assertEqual(self.ca.resolve("Urysohn's lemma")["matched-on"], "urysohn's lemma")

    def test_an_en_dash_and_an_accent_are_the_same_name(self):
        self.assertEqual(self.resolved("Hahn–Banach theorem"), self.resolved("Hahn-Banach theorem"))
        self.assertEqual(self.resolved("Arzelà-Ascoli theorem"), self.resolved("Arzela-Ascoli theorem"))
        self.assertIsNotNone(self.resolved("Hahn–Banach theorem"))

    def test_folding_is_a_fallback_and_never_displaces_a_stored_name(self):
        # Terms that are stored WITH a dash or an accent must still resolve to
        # themselves, not to some folded neighbour.
        for term in ("Cauchy–Schwarz inequality", "Poincaré duality", "adjoint functor"):
            self.assertIsNotNone(self.resolved(term), term)

    def test_fold_leaves_an_ordinary_name_alone(self):
        import concept_authority
        self.assertEqual(concept_authority.fold("hahn-banach theorem"), "hahn-banach theorem")
        self.assertEqual(concept_authority.fold("arzelà-ascoli"), "arzela-ascoli")
