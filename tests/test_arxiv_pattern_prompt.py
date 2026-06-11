"""Tests for the arxiv-aware Stage 3 prompt builder.

Validates:
- Taxonomy loads from futon3/library/ flexiargs.
- Prompt builder emits a coherent string with all 5 families and leaves.
- Response parser accepts well-formed responses, rejects malformed ones,
  and repairs common local-LLM JSON/taxonomy drift without dropping records.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import futon6.arxiv_pattern_prompt as arxiv_pattern_prompt
from futon6.arxiv_pattern_prompt import (
    FAMILY_PARENTS,
    _default_futon3_library,
    build_arxiv_pattern_prompt,
    load_paper_shape_taxonomy,
    parse_arxiv_pattern_response,
)


_LIBRARY_CANDIDATES = []
if env_library := os.environ.get("FUTON3_LIBRARY"):
    _LIBRARY_CANDIDATES.append(Path(env_library))
if env_root := os.environ.get("FUTON3_ROOT"):
    _LIBRARY_CANDIDATES.append(Path(env_root) / "library")
_LIBRARY_CANDIDATES.extend(
    [
        Path.home() / "code" / "futon3" / "library",
        Path(__file__).resolve().parents[2] / "futon3" / "library",
    ]
)
_LIBRARY_ROOT = next(
    (candidate for candidate in _LIBRARY_CANDIDATES if candidate.exists()),
    _LIBRARY_CANDIDATES[0],
)
_HAS_LIBRARY = _LIBRARY_ROOT.exists()


@unittest.skipUnless(_HAS_LIBRARY, "futon3 library not present")
class TestTaxonomyLoad(unittest.TestCase):
    def test_loads_five_families(self):
        tax = load_paper_shape_taxonomy(_LIBRARY_ROOT)
        self.assertEqual(set(tax.families.keys()), set(FAMILY_PARENTS))

    def test_each_family_has_title(self):
        tax = load_paper_shape_taxonomy(_LIBRARY_ROOT)
        for fid, fam in tax.families.items():
            self.assertTrue(fam.title, f"family {fid} has empty title")

    def test_at_least_one_leaf_per_non_meta_family(self):
        tax = load_paper_shape_taxonomy(_LIBRARY_ROOT)
        for fid in FAMILY_PARENTS:
            if fid.endswith("/clarification-meta"):
                continue
            children = tax.all_leaves_for(fid)
            self.assertTrue(
                children,
                f"family {fid} has no leaves; check member-pattern wiring",
            )

    def test_new_leaves_link_to_their_family(self):
        """The five genuinely-new leaves should declare their @family."""
        tax = load_paper_shape_taxonomy(_LIBRARY_ROOT)
        expected = {
            "math-informal/failure-mode-characterization":
                "math-strategy/characterization-result",
            "math-informal/structural-characterization":
                "math-strategy/characterization-result",
            "math-informal/complexity-classification":
                "math-strategy/characterization-result",
            "math-informal/structural-inclusion":
                "math-strategy/structural-relation-result",
            "math-informal/structural-equivalence":
                "math-strategy/structural-relation-result",
        }
        for leaf_id, expected_family in expected.items():
            self.assertIn(leaf_id, tax.leaves, f"missing new leaf {leaf_id}")
            self.assertEqual(tax.leaves[leaf_id].family, expected_family)

    def test_default_library_finds_sibling_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "futon6"
            fake_module = repo_root / "src" / "futon6" / "arxiv_pattern_prompt.py"
            fake_module.parent.mkdir(parents=True, exist_ok=True)
            fake_module.write_text("# fake module path for resolver test\n", encoding="utf-8")

            sibling_library = repo_root / "futon3" / "library"
            sibling_library.mkdir(parents=True, exist_ok=True)
            for parent_id in FAMILY_PARENTS:
                (sibling_library / f"{parent_id}.flexiarg").parent.mkdir(parents=True, exist_ok=True)
                (sibling_library / f"{parent_id}.flexiarg").write_text("title: ok\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(
                    arxiv_pattern_prompt,
                    "DEFAULT_FUTON3_LIBRARY",
                    repo_root / "missing-home-library",
                ):
                    with mock.patch.object(arxiv_pattern_prompt, "__file__", str(fake_module)):
                        self.assertEqual(_default_futon3_library(), sibling_library)


@unittest.skipUnless(_HAS_LIBRARY, "futon3 library not present")
class TestPromptBuilder(unittest.TestCase):
    def test_prompt_includes_all_families(self):
        prompt = build_arxiv_pattern_prompt(
            paper_id="arxiv-2604.20815v1",
            title="Sharp Zarankiewicz dichotomy",
            abstract="We prove that the Zarankiewicz number ...",
        )
        for fid in FAMILY_PARENTS:
            self.assertIn(fid, prompt, f"prompt missing family {fid}")

    def test_prompt_includes_paper_id_and_title(self):
        prompt = build_arxiv_pattern_prompt(
            paper_id="arxiv-2604.20815v1",
            title="Sharp Zarankiewicz dichotomy",
            abstract="We prove ...",
        )
        self.assertIn("arxiv-2604.20815v1", prompt)
        self.assertIn("Sharp Zarankiewicz dichotomy", prompt)

    def test_prompt_includes_excerpts_when_provided(self):
        prompt = build_arxiv_pattern_prompt(
            paper_id="arxiv-2604.20815v1",
            title="Sharp Zarankiewicz dichotomy",
            abstract="We prove ...",
            theorem_excerpts=["Theorem 1: For every t and r..."],
            proof_excerpts=["Proof. We split into two cases."],
        )
        self.assertIn("Theorem excerpts:", prompt)
        self.assertIn("Proof excerpts:", prompt)

    def test_prompt_clips_oversized_abstract(self):
        long_abs = "X" * 5000
        prompt = build_arxiv_pattern_prompt(
            paper_id="arxiv-test",
            title="t",
            abstract=long_abs,
            char_budget_abstract=200,
        )
        self.assertNotIn("X" * 300, prompt)


@unittest.skipUnless(_HAS_LIBRARY, "futon3 library not present")
class TestResponseParser(unittest.TestCase):
    def test_well_formed_leaf_response(self):
        raw = json.dumps({
            "family": "math-strategy/characterization-result",
            "leaf": "math-informal/split-into-cases",
            "family_confidence": 0.92,
            "leaf_confidence": 0.81,
            "rationale": "Sharp dichotomy theorem with explicit bounds.",
        })
        result = parse_arxiv_pattern_response(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["family"], "math-strategy/characterization-result")
        self.assertEqual(result["leaf"], "math-informal/split-into-cases")

    def test_uncertain_leaf_accepted(self):
        raw = json.dumps({
            "family": "math-strategy/structural-relation-result",
            "leaf": "uncertain",
            "family_confidence": 0.74,
            "leaf_confidence": 0.40,
            "rationale": "Clearly relational, leaf direction unclear.",
        })
        result = parse_arxiv_pattern_response(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["leaf"], "uncertain")

    def test_clarification_meta_without_collapsed_is_repaired(self):
        raw = json.dumps({
            "family": "math-strategy/clarification-meta",
            "leaf": "",
            "family_confidence": 0.85,
            "leaf_confidence": 0.0,
            "rationale": "Triple is single-axis.",
        })
        result = parse_arxiv_pattern_response(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["collapsed"]["reason"], "other")
        self.assertIn("clarification-meta-collapsed-synthesized", result["warnings"])

    def test_clarification_meta_with_collapsed_ok(self):
        raw = json.dumps({
            "family": "math-strategy/clarification-meta",
            "leaf": "",
            "family_confidence": 0.9,
            "leaf_confidence": 0.0,
            "rationale": "Single-axis principle clarification.",
            "collapsed": {
                "reason": "single-axis",
                "explanation": "Paper clarifies one principle.",
            },
        })
        result = parse_arxiv_pattern_response(raw)
        self.assertTrue(result["ok"])

    def test_invalid_family_rejected(self):
        raw = json.dumps({
            "family": "math-strategy/imaginary-result",
            "leaf": "anything",
        })
        result = parse_arxiv_pattern_response(raw)
        self.assertFalse(result["ok"])
        self.assertIn("invalid-family", result["error"])

    def test_invalid_leaf_is_normalized_to_uncertain_for_strategic_family(self):
        raw = json.dumps({
            "family": "math-strategy/existence-result",
            "leaf": "math-informal/imaginary-leaf",
            "leaf_confidence": 0.9,
        })
        result = parse_arxiv_pattern_response(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["leaf"], "uncertain")
        self.assertLessEqual(result["leaf_confidence"], 0.5)
        self.assertIn("invalid-leaf-normalized", result["warnings"][0])

    def test_tex_backslashes_in_rationale_are_repaired(self):
        raw = (
            '{'
            '"family": "math-strategy/characterization-result",'
            '"leaf": "math-informal/structural-characterization",'
            '"family_confidence": 0.9,'
            '"leaf_confidence": 0.8,'
            '"rationale": "Classifies $(\\mathbb{T},\\mathsf{V})$-categories.",'
            '"collapsed": null'
            '}'
        )
        result = parse_arxiv_pattern_response(raw)
        self.assertTrue(result["ok"])
        self.assertIn("\\mathbb", result["rationale"])

    def test_truncated_json_response_is_salvaged_when_family_and_leaf_are_complete(self):
        raw = (
            '{'
            '"family": "math-strategy/characterization-result",'
            '"leaf": "math-informal/structural-characterization",'
            '"family_confidence": 0.9,'
            '"leaf_confidence": 0.8,'
            '"rationale": "The paper provides a characterization of n-categories'
        )
        result = parse_arxiv_pattern_response(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["family"], "math-strategy/characterization-result")
        self.assertEqual(result["leaf"], "math-informal/structural-characterization")
        self.assertIn("truncated-json-salvaged", result["warnings"])

    def test_no_json_in_response(self):
        result = parse_arxiv_pattern_response("Sorry, I cannot answer.")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "no-json-object")


if __name__ == "__main__":
    unittest.main()
