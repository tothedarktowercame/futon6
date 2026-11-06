"""E-comprehension-foundation: the verdict gate, typed holes, and the pilot
go/no-go — the compiled cascade's build steps 1, 3, and the Linode go/no-go.

Each test constructs the exact case the check is named for, against the real
scoring functions (no stubs): the gate must refuse a weakness assertion below
high comprehension, the hole harvest must key holes by (type, concept) against
the real concept-index candidate extraction, and the pilot decision must
withhold 'go' when the score does not rise or a weak-proof appears below the
floor."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import clean_comprehension as cc  # noqa: E402
import clean_typed_holes as cth  # noqa: E402
import comprehension_pilot as cp  # noqa: E402


class TestVerdictGate:
    def test_weak_proof_requires_high_comprehension(self):
        # same 5 ungrounded moves, only comprehension differs
        assert cc.verdict(0.2, 0.3, 5, 0, 0.5, 0.8)[0] == "weak-extraction"
        assert cc.verdict(0.95, 0.9, 5, 0, 0.5, 0.8)[0] == "weak-proof"

    def test_gate_refuses_weakness_below_hi_and_emits_study_more(self):
        # the exact bad case the gate is named for: asserting weakness about a
        # proof we did not comprehend
        with pytest.raises(ValueError, match="study-more/ask-for-help"):
            cc.assert_weakness("0706.1286", comp=0.25, hi=0.8)
        assert cc.next_path(0.25, 0.8) == "study-more/ask-for-help"
        assert cc.next_path(None, 0.8) == "study-more/ask-for-help"

    def test_gate_allows_weakness_at_hi(self):
        assert cc.assert_weakness("x", comp=0.9, hi=0.8) == "weakness-assertable"
        assert cc.next_path(0.9, 0.8) == "weakness-assertable"


FIXTURE_GRAPH = """{:paper/id "0706.1286" :passage/id "p"
 :nodes [{:id :cat-like-bicats :kind :object :text "Cat-like bicategories"}
         {:id :equiv-for-calmod-like :kind :claim
          :text "equivalence for CalMod-like bicategories"}]
 :edges [{:id :e1 :kind :infer :relation :implies :premise :cat-like-bicats
          :warrant {:kind :missing-warrant :text "ring-isomorphism-equivalence"}
          :conclusion :equiv-for-calmod-like}]
 :holes [{:kind :missing-warrant :edge :e1
          :wanted :ring-isomorphism-equivalence-for-calmod-like-bicats}]}"""


class Substrate:
    def __init__(self, index):
        self.concept_index = index


class TestTypedHoles:
    def _write(self, tmp_path):
        g = tmp_path / "0706.1286.edn"
        g.write_text(FIXTURE_GRAPH)
        return g

    def test_hole_keyed_by_type_and_grounded_concept(self, tmp_path):
        # the concept the hole WANTS is drawn from the holed edge's conclusion
        # and grounded against the substrate, keyed with the hole type
        sub = Substrate({"calmod like bicategory": {"defined": True}})
        rows = cth.inventory([self._write(tmp_path)], sub)
        key = next(r for r in rows["typed_holes"]
                   if r["concept"] == "calmod like bicategory")
        assert key["type"] == "missing-warrant"
        assert key["occurrences"] == 1

    def test_ungrounded_hole_reported_not_dropped(self, tmp_path):
        sub = Substrate({"unrelated concept": {}})
        rows = cth.inventory([self._write(tmp_path)], sub)
        assert rows["typed_holes"] == []
        assert rows["ungrounded"] == [
            {"pid": "0706.1286", "type": "missing-warrant",
             "wanted": "ring-isomorphism-equivalence-for-calmod-like-bicats"}]

    def test_recurrence_counted_across_papers(self, tmp_path):
        # two papers whose holes want the same grounded concept: the raw
        # :wanted slugs differ, the (type, concept) key recurs
        sub = Substrate({"calmod like bicategory": {}})
        (tmp_path / "0706.1286.edn").write_text(FIXTURE_GRAPH)
        g2 = FIXTURE_GRAPH.replace("0706.1286", "0801.0199").replace(
            ":paper/id", ":paper/id")  # same hole shape, other paper
        (tmp_path / "0801.0199.edn").write_text(g2)
        rows = cth.inventory(sorted(tmp_path.glob("*.edn")), sub)
        assert rows["recurring"], rows
        assert rows["recurring"][0]["papers"] == ["0706.1286", "0801.0199"]

    def test_corpus_dedupe_counts_reprocessed_paper_once(self, tmp_path):
        # the exact case corpus_inventory is named for: the SAME paper
        # (same pid) harvested in run and run-retry is one paper, never a
        # self-recurrence; recurrence comes from a genuinely distinct paper
        sub = Substrate({"calmod like bicategory": {}})
        a = tmp_path / "run"; b = tmp_path / "run-retry"
        a.mkdir(); b.mkdir()
        (a / "0706.1286.edn").write_text(FIXTURE_GRAPH)
        (b / "0706.1286.edn").write_text(FIXTURE_GRAPH)          # reprocess
        (b / "0801.0199.edn").write_text(
            FIXTURE_GRAPH.replace("0706.1286", "0801.0199"))
        pairs = [(a / "0706.1286.edn", "run"),
                 (b / "0706.1286.edn", "run-retry"),
                 (b / "0801.0199.edn", "run-retry")]
        rows = cth.corpus_inventory(pairs, sub)
        assert rows["n_papers"] == 2
        rec = rows["recurring"][0]
        assert rec["papers"] == ["0706.1286", "0801.0199"]
        assert rec["occurrences"] == 2          # not 3: retry not double-counted
        assert rec["sets"] == ["run", "run-retry"]


class TestPilotGoNoGo:
    def _sizes(self, comps, weak_proof_comp=None):
        proofs = [{"pid": "p", "comprehension": c, "verdict": "partial-comprehension"}
                  for c in comps]
        if weak_proof_comp is not None:
            proofs.append({"pid": "wp", "comprehension": weak_proof_comp,
                           "verdict": "weak-proof"})
        return [{"mean_comprehension": m, "report": {"proofs": proofs}}
                for m in comps]

    def test_go_when_rising_and_separated(self):
        probe = cp.separation_probe(0.5, 0.8)
        d = cp.go_no_go(self._sizes([0.4, 0.46, 0.51]), probe, 0.8)
        assert d["go"] and d["rises_with_corpus"] and d["separates_weak_extraction_from_weak_proof"]

    def test_no_go_when_score_falls_with_corpus(self):
        probe = cp.separation_probe(0.5, 0.8)
        d = cp.go_no_go(self._sizes([0.5, 0.4]), probe, 0.8)
        assert not d["go"] and not d["rises_with_corpus"]

    def test_no_go_when_weak_proof_below_hi(self):
        # the recorded run itself must never show weak-proof under the floor
        probe = cp.separation_probe(0.5, 0.8)
        d = cp.go_no_go(self._sizes([0.4, 0.5], weak_proof_comp=0.6), probe, 0.8)
        assert not d["go"] and not d["no_weak_proof_below_hi"]

    def test_recorded_pilot_is_a_go(self):
        # the committed pilot record must satisfy its own decision rule
        rec = json.load(open(ROOT / "holes/excursions/E-comprehension-foundation.pilot.json"))
        assert rec["decision"]["go"] is True
        assert rec["go_no_go"] == "go"
        assert rec["decision"]["rises_with_corpus"]
        assert rec["decision"]["separates_weak_extraction_from_weak_proof"]
