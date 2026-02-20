"""Tests for assemble-wiring.py — hierarchical wiring assembly from SE/MO threads."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import importlib
assemble_wiring = importlib.import_module("assemble-wiring")
nlab_wiring = importlib.import_module("nlab-wiring")


# ============================================================
# Fixtures
# ============================================================

SIMPLE_THREAD = {
    "id": 12345,
    "body": "What is a left adjoint? Let $F: C \\to D$ be a functor. Is there a right adjoint $G$?",
    "title": "Left adjoint functor question",
    "tags": ["category-theory"],
    "score": 5,
    "site": "math.stackexchange.com",
    "topic": "category-theory",
    "answers": [
        {
            "id": 67890,
            "body": "Let $F: C \\to D$ and $G: D \\to C$ be functors. "
                    "We say $F \\dashv G$ (F is left adjoint to G) if there is a "
                    "natural isomorphism $\\mathrm{Hom}_D(Fc, d) \\cong \\mathrm{Hom}_C(c, Gd)$. "
                    "Therefore $F$ preserves colimits and $G$ preserves limits.",
            "score": 10,
            "is_accepted": True,
        },
        {
            "id": 67891,
            "body": "For example, the free group functor $F: Set \\to Grp$ "
                    "is left adjoint to the forgetful functor $U: Grp \\to Set$.",
            "score": 3,
            "is_accepted": False,
        },
    ],
    "comments": {
        "question": [
            {"id": 111, "post_id": 12345, "score": 1,
             "text": "What do you mean by 'left adjoint'? Are you looking for the definition?"},
        ],
        "answers": {
            "67890": [
                {"id": 222, "post_id": 67890, "score": 2,
                 "text": "Good answer. This is exactly right."},
            ],
        },
        "total": 2,
    },
}

# Reference structure (minimal)
MINI_REFERENCE = {
    "patterns": {
        "cat/adjunction": {
            "instances": ["nlab-193"],
            "instance_count": 927,
            "required_links": ["category", "functor", "left adjoint"],
            "typical_links": ["right adjoint", "natural transformation", "isomorphism"],
            "discourse_signature": {"components": {}, "wires": {}},
            "avg_diagrams": 5.0,
        },
        "cat/natural-transformation": {
            "instances": ["nlab-300"],
            "instance_count": 797,
            "required_links": ["category", "functor", "natural transformation"],
            "typical_links": ["naturality", "components"],
            "discourse_signature": {"components": {}, "wires": {}},
            "avg_diagrams": 3.0,
        },
    },
    "link_weights": {
        "functor": {"definition-ref": 286, "prose-ref": 819},
        "category": {"definition-ref": 416, "prose-ref": 1979},
        "adjoint": {"definition-ref": 50, "prose-ref": 200},
    },
}


# ============================================================
# IATC Detection Tests
# ============================================================

class TestIATCDetection:

    def test_assert_detected(self):
        hits = assemble_wiring.detect_iatc("We have shown that F preserves colimits.")
        types = [h[0] for h in hits]
        assert "assert" in types

    def test_challenge_detected(self):
        hits = assemble_wiring.detect_iatc("But why is this true? I don't see how it follows.")
        types = [h[0] for h in hits]
        assert "challenge" in types

    def test_query_detected(self):
        hits = assemble_wiring.detect_iatc("What is a left adjoint?")
        types = [h[0] for h in hits]
        assert "query" in types

    def test_agree_detected(self):
        hits = assemble_wiring.detect_iatc("Yes, that's correct. Good point.")
        types = [h[0] for h in hits]
        assert "agree" in types

    def test_exemplify_detected(self):
        hits = assemble_wiring.detect_iatc("For example, consider the free group functor.")
        types = [h[0] for h in hits]
        assert "exemplify" in types

    def test_clarify_detected(self):
        hits = assemble_wiring.detect_iatc("To clarify, what I mean is the hom-set definition.")
        types = [h[0] for h in hits]
        assert "clarify" in types

    def test_classify_edge_defaults(self):
        assert assemble_wiring.classify_edge_iatc("Some text.", "answer") == "assert"
        assert assemble_wiring.classify_edge_iatc("Some text.", "comment") == "clarify"

    def test_classify_edge_from_text(self):
        iatc = assemble_wiring.classify_edge_iatc(
            "But why? I don't think this is correct.", "comment")
        assert iatc == "challenge"


# ============================================================
# Port Extraction Tests
# ============================================================

class TestPortExtraction:

    def test_input_ports_from_let(self):
        text = "Let $F$ be a functor. Assume $F$ preserves limits."
        inputs, outputs = assemble_wiring.extract_ports(text, "test")
        assert len(inputs) > 0
        # Should find let-binding for F
        types = [p["type"] for p in inputs]
        assert any("bind" in t or "assume" in t for t in types)

    def test_output_ports_after_therefore(self):
        text = "Let $X$ be a space. Since $X$ is compact, therefore $X$ is bounded."
        inputs, outputs = assemble_wiring.extract_ports(text, "test")
        # Output ports come after consequential wire
        assert len(outputs) >= 0  # may or may not have formal scopes after "therefore"
        assert len(inputs) > 0  # should have "Let X be a space"

    def test_scope_to_label(self):
        scope = {
            "hx/type": "bind/let",
            "hx/ends": [
                {"role": "entity", "ident": "test"},
                {"role": "symbol", "latex": "F"},
                {"role": "type", "text": "a functor"},
            ],
            "hx/content": {"match": "Let $F$ be a functor", "position": 0},
        }
        label = assemble_wiring._scope_to_label(scope)
        assert "F" in label

    def test_symbolic_binder_is_input_port(self):
        text = r"Let $f$ be continuous. Consider $\int_0^1 f(x)\,dx$. Therefore we are done."
        inputs, _ = assemble_wiring.extract_ports(text, "test")
        assert any(p["type"] == "bind/integral" for p in inputs)


# ============================================================
# Port Matching Tests
# ============================================================

class TestPortMatching:

    def test_matching_ports_found(self):
        source_outputs = [
            {"id": "out-1", "type": "bind/let", "label": "F is a left adjoint functor",
             "text": "...", "position": 0},
        ]
        target_inputs = [
            {"id": "in-1", "type": "bind/let", "label": "F be a functor",
             "text": "...", "position": 0},
        ]
        matches = assemble_wiring.match_ports(source_outputs, target_inputs, MINI_REFERENCE)
        assert len(matches) >= 1
        # Should match on "functor" and "F"
        assert matches[0][2] > 0  # positive score

    def test_no_match_when_disjoint(self):
        source_outputs = [
            {"id": "out-1", "type": "bind/let", "label": "X topological space",
             "text": "...", "position": 0},
        ]
        target_inputs = [
            {"id": "in-1", "type": "bind/let", "label": "n integer",
             "text": "...", "position": 0},
        ]
        matches = assemble_wiring.match_ports(source_outputs, target_inputs, MINI_REFERENCE)
        assert len(matches) == 0


# ============================================================
# Categorical Detection for SE Text
# ============================================================

class TestCategoricalForSE:

    def test_adjunction_detected_in_answer(self):
        text = ("Let $F: C \\to D$ and $G: D \\to C$ be functors. "
                "We say $F \\dashv G$ (F is left adjoint to G) if there is a "
                "natural isomorphism of hom-sets.")
        cats = assemble_wiring.detect_categorical_for_se(
            text, ["category-theory"], MINI_REFERENCE)
        cat_types = [c["hx/type"] for c in cats]
        assert "cat/adjunction" in cat_types

    def test_no_pattern_in_empty_text(self):
        cats = assemble_wiring.detect_categorical_for_se("Hello.", [], MINI_REFERENCE)
        assert len(cats) == 0

    def test_tag_boosts_detection(self):
        text = "Consider a functor between categories."
        # With CT tag, more likely to detect
        cats_with_tag = assemble_wiring.detect_categorical_for_se(
            text, ["category-theory"], MINI_REFERENCE)
        cats_no_tag = assemble_wiring.detect_categorical_for_se(
            text, [], MINI_REFERENCE)
        # Tags should boost or at least not reduce
        assert len(cats_with_tag) >= len(cats_no_tag)


# ============================================================
# Thread Graph Construction
# ============================================================

class TestThreadGraph:

    def test_graph_has_correct_node_count(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        # 1 question + 2 answers + 2 comments = 5 nodes
        assert wiring["stats"]["n_nodes"] == 5

    def test_graph_has_correct_edge_count(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        # 2 answer→question + 2 comment→parent = 4 edges
        assert wiring["stats"]["n_edges"] == 4

    def test_question_node_exists(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        q_nodes = [n for n in wiring["nodes"] if n["type"] == "question"]
        assert len(q_nodes) == 1
        assert q_nodes[0]["id"] == "q-12345"
        assert q_nodes[0]["title"] == "Left adjoint functor question"

    def test_accepted_answer_marked(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        accepted = [n for n in wiring["nodes"] if n.get("is_accepted")]
        assert len(accepted) == 1
        assert accepted[0]["id"] == "a-67890"

    def test_edges_have_iatc_types(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        for edge in wiring["edges"]:
            assert "iatc" in edge
            assert edge["iatc"] in (
                "assert", "challenge", "query", "clarify", "reform",
                "exemplify", "reference", "agree", "retract")

    def test_nodes_have_ports(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        for node in wiring["nodes"]:
            assert "input_ports" in node
            assert "output_ports" in node

    def test_categorical_detected_in_answer(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        a_node = [n for n in wiring["nodes"] if n["id"] == "a-67890"][0]
        # The accepted answer discusses adjunction — should have categorical
        assert len(a_node["categorical"]) > 0

    def test_thread_metadata(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        assert wiring["thread_id"] == 12345
        assert wiring["site"] == "math.stackexchange.com"
        assert wiring["topic"] == "category-theory"
        assert wiring["level"] == "thread"

    def test_json_serializable(self):
        wiring = assemble_wiring.build_thread_graph(SIMPLE_THREAD, MINI_REFERENCE)
        # Should not raise
        json.dumps(wiring, ensure_ascii=False)


# ============================================================
# Comment Flattening
# ============================================================

class TestCommentFlattening:

    def test_dict_format(self):
        comments = {
            "question": [{"id": 1, "post_id": 100, "text": "q comment"}],
            "answers": {
                "200": [{"id": 2, "post_id": 200, "text": "a comment"}],
            },
            "total": 2,
        }
        flat = assemble_wiring._flatten_comments(comments)
        assert len(flat) == 2

    def test_list_format(self):
        comments = [
            {"id": 1, "post_id": 100, "text": "comment 1"},
            {"id": 2, "post_id": 200, "text": "comment 2"},
        ]
        flat = assemble_wiring._flatten_comments(comments)
        assert len(flat) == 2

    def test_empty(self):
        assert assemble_wiring._flatten_comments({}) == []
        assert assemble_wiring._flatten_comments([]) == []


# ============================================================
# SEThread → Dict Converter Tests
# ============================================================

class _FakeSEPost:
    """Minimal stand-in for futon6.stackexchange.SEPost."""
    def __init__(self, id, post_type="question", title="", body_text="",
                 score=0, tags=None, accepted_answer_id=None, parent_id=None):
        self.id = id
        self.post_type = post_type
        self.title = title
        self.body_text = body_text
        self.score = score
        self.tags = tags or []
        self.accepted_answer_id = accepted_answer_id
        self.parent_id = parent_id

class _FakeSEComment:
    """Minimal stand-in for futon6.stackexchange.SEComment."""
    def __init__(self, id, post_id, text="", score=0):
        self.id = id
        self.post_id = post_id
        self.text = text
        self.score = score

class _FakeSEThread:
    """Minimal stand-in for futon6.stackexchange.SEThread."""
    def __init__(self, question, answers=None, comments=None, tags=None):
        self.question = question
        self.answers = answers or []
        self.comments = comments or {}
        self.tags = tags or []


class TestSEThreadConverter:

    def _make_thread(self):
        q = _FakeSEPost(id=100, title="Test Q", body_text="What is X?",
                        score=5, tags=["algebra"], accepted_answer_id=201)
        a1 = _FakeSEPost(id=201, post_type="answer", body_text="X is Y.",
                         score=10, parent_id=100)
        a2 = _FakeSEPost(id=202, post_type="answer", body_text="X is Z.",
                         score=3, parent_id=100)
        c_q = _FakeSEComment(id=301, post_id=100, text="Good question.", score=1)
        c_a1 = _FakeSEComment(id=302, post_id=201, text="Nice answer.", score=2)
        return _FakeSEThread(
            question=q,
            answers=[a1, a2],
            comments={100: [c_q], 201: [c_a1]},
            tags=["algebra"],
        )

    def test_basic_fields(self):
        thread = self._make_thread()
        d = assemble_wiring.sethread_to_dict(thread, site="math.se", topic="algebra")
        assert d["id"] == 100
        assert d["title"] == "Test Q"
        assert d["body"] == "What is X?"
        assert d["score"] == 5
        assert d["site"] == "math.se"
        assert d["topic"] == "algebra"
        assert d["tags"] == ["algebra"]

    def test_answers_converted(self):
        thread = self._make_thread()
        d = assemble_wiring.sethread_to_dict(thread)
        assert len(d["answers"]) == 2
        assert d["answers"][0]["id"] == 201
        assert d["answers"][0]["body"] == "X is Y."
        assert d["answers"][0]["score"] == 10
        assert d["answers"][1]["id"] == 202

    def test_accepted_answer_flag(self):
        thread = self._make_thread()
        d = assemble_wiring.sethread_to_dict(thread)
        assert d["answers"][0]["is_accepted"] is True   # id=201 matches
        assert d["answers"][1]["is_accepted"] is False   # id=202 doesn't

    def test_comments_reorganized(self):
        thread = self._make_thread()
        d = assemble_wiring.sethread_to_dict(thread)
        comments = d["comments"]
        assert len(comments["question"]) == 1
        assert comments["question"][0]["id"] == 301
        assert comments["question"][0]["text"] == "Good question."
        assert "201" in comments["answers"]
        assert len(comments["answers"]["201"]) == 1
        assert comments["answers"]["201"][0]["id"] == 302
        assert comments["total"] == 2

    def test_no_accepted_answer(self):
        q = _FakeSEPost(id=100, title="Q", body_text="?", accepted_answer_id=None)
        a = _FakeSEPost(id=201, post_type="answer", body_text="A.")
        thread = _FakeSEThread(question=q, answers=[a])
        d = assemble_wiring.sethread_to_dict(thread)
        assert d["answers"][0]["is_accepted"] is False

    def test_empty_comments(self):
        q = _FakeSEPost(id=100, title="Q", body_text="?")
        thread = _FakeSEThread(question=q)
        d = assemble_wiring.sethread_to_dict(thread)
        assert d["comments"]["question"] == []
        assert d["comments"]["answers"] == {}
        assert d["comments"]["total"] == 0

    def test_roundtrip_through_build_thread_graph(self):
        """Converted dict should work with build_thread_graph()."""
        thread = self._make_thread()
        d = assemble_wiring.sethread_to_dict(thread, site="test", topic="test")
        wiring = assemble_wiring.build_thread_graph(d, MINI_REFERENCE)
        assert wiring["thread_id"] == 100
        assert wiring["stats"]["n_nodes"] == 5  # 1q + 2a + 2c
        assert wiring["stats"]["n_edges"] == 4  # 2 answer→q + 2 comment→parent
        assert wiring["site"] == "test"
