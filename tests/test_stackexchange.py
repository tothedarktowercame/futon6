"""Tests for StackExchange data processing pipeline."""

import tempfile
import os
from pathlib import Path

from futon6.stackexchange import (
    strip_html,
    extract_latex,
    parse_se_tags,
    iter_posts,
    iter_comments,
    load_posts,
    build_qa_pairs,
    build_threads_streaming,
    qa_to_entity,
    qa_to_relations,
    tag_entities,
    corpus_stats,
    SEComment,
    SEThread,
)
from futon6.thread_performatives import (
    detect_performatives,
    build_thread_wiring_diagram,
    diagram_to_dict,
    diagram_to_hyperedges,
    diagram_stats,
    process_threads_to_diagrams,
    merge_llm_edges,
    write_thread_wiring_json,
    ThreadNode,
    ThreadEdge,
    ThreadWiringDiagram,
)


# --- Synthetic SE XML for testing ---

SAMPLE_POSTS_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" Title="What is the uncertainty principle?"
       Body="&lt;p&gt;In quantum mechanics, the &lt;strong&gt;uncertainty principle&lt;/strong&gt; states that $$\\Delta x \\Delta p \\geq \\frac{\\hbar}{2}$$ Can someone explain the physical meaning? The commutator $[\\hat{x}, \\hat{p}] = i\\hbar$ seems key.&lt;/p&gt;"
       Score="42" Tags="&lt;quantum-mechanics&gt;&lt;uncertainty-principle&gt;&lt;operators&gt;"
       AnswerCount="3" AcceptedAnswerId="2" CreationDate="2024-01-15T10:30:00.000" />
  <row Id="2" PostTypeId="2" ParentId="1"
       Body="&lt;p&gt;The uncertainty principle follows from the &lt;em&gt;Robertson inequality&lt;/em&gt;: for any two observables $A$ and $B$, $$\\sigma_A \\sigma_B \\geq \\frac{1}{2}|\\langle [A,B] \\rangle|$$ This is proved by applying the Cauchy-Schwarz inequality to the inner product space of quantum states.&lt;/p&gt;"
       Score="58" CreationDate="2024-01-15T11:00:00.000" />
  <row Id="3" PostTypeId="2" ParentId="1"
       Body="&lt;p&gt;It means you cannot simultaneously know position and momentum with arbitrary precision.&lt;/p&gt;"
       Score="5" CreationDate="2024-01-15T12:00:00.000" />
  <row Id="4" PostTypeId="1" Title="Deriving the Boltzmann distribution"
       Body="&lt;p&gt;How do you derive $P(E) = \\frac{1}{Z} e^{-\\beta E}$ from the microcanonical ensemble? I understand we need to maximize entropy $S = -k_B \\sum_i p_i \\ln p_i$ subject to constraints.&lt;/p&gt;"
       Score="15" Tags="&lt;statistical-mechanics&gt;&lt;thermodynamics&gt;&lt;entropy&gt;"
       AnswerCount="1" CreationDate="2024-02-01T08:00:00.000" />
  <row Id="5" PostTypeId="2" ParentId="4"
       Body="&lt;p&gt;Use Lagrange multipliers. Maximise $$S = -k_B \\sum_i p_i \\ln p_i$$ subject to $\\sum_i p_i = 1$ and $\\sum_i p_i E_i = \\langle E \\rangle$. The solution gives $p_i \\propto e^{-\\beta E_i}$ where $\\beta = 1/k_B T$.&lt;/p&gt;"
       Score="20" CreationDate="2024-02-01T09:00:00.000" />
  <row Id="6" PostTypeId="1" Title="Why is the sky blue?"
       Body="&lt;p&gt;Rayleigh scattering explains this. The cross section goes as $\\sigma \\propto \\lambda^{-4}$.&lt;/p&gt;"
       Score="-2" Tags="&lt;optics&gt;&lt;scattering&gt;"
       AnswerCount="0" CreationDate="2024-03-01T10:00:00.000" />
</posts>
"""


def _write_sample_xml():
    """Write sample XML to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    f.write(SAMPLE_POSTS_XML)
    f.close()
    return f.name


class TestHTMLProcessing:
    def test_strip_html(self):
        result = strip_html("<p>Hello <strong>world</strong></p>")
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_strip_html_entities(self):
        result = strip_html("&lt;p&gt;test&lt;/p&gt;")
        assert "<p>test</p>" == result

    def test_extract_latex_display(self):
        html_str = "text $$\\Delta x \\Delta p \\geq \\frac{\\hbar}{2}$$ more"
        frags = extract_latex(html_str)
        assert len(frags) >= 1
        assert "\\Delta x" in frags[0]

    def test_extract_latex_inline(self):
        html_str = "the commutator $[\\hat{x}, \\hat{p}] = i\\hbar$ is key"
        frags = extract_latex(html_str)
        assert len(frags) >= 1
        assert "\\hat{x}" in frags[0]

    def test_extract_latex_both(self):
        html_str = "$$E = mc^2$$ and $p = mv$"
        frags = extract_latex(html_str)
        assert len(frags) == 2

    def test_parse_se_tags(self):
        tags = parse_se_tags("<quantum-mechanics><uncertainty-principle>")
        assert tags == ["quantum-mechanics", "uncertainty-principle"]

    def test_parse_empty_tags(self):
        assert parse_se_tags("") == []
        assert parse_se_tags(None) == []


class TestXMLParsing:
    def test_iter_posts(self):
        path = _write_sample_xml()
        try:
            posts = list(iter_posts(path))
            assert len(posts) == 5  # 6 rows, but score=-2 filtered by min_score=0
        finally:
            os.unlink(path)

    def test_iter_posts_min_score(self):
        path = _write_sample_xml()
        try:
            posts = list(iter_posts(path, min_score=10))
            # Only posts with score >= 10: ids 1(42), 2(58), 4(15), 5(20)
            assert len(posts) == 4
        finally:
            os.unlink(path)

    def test_post_fields(self):
        path = _write_sample_xml()
        try:
            posts = list(iter_posts(path))
            q = next(p for p in posts if p.id == 1)
            assert q.post_type == "question"
            assert q.title == "What is the uncertainty principle?"
            assert q.score == 42
            assert "quantum-mechanics" in q.tags
            assert q.accepted_answer_id == 2
            assert len(q.body_latex) >= 2  # display + inline
        finally:
            os.unlink(path)

    def test_answer_fields(self):
        path = _write_sample_xml()
        try:
            posts = list(iter_posts(path))
            a = next(p for p in posts if p.id == 2)
            assert a.post_type == "answer"
            assert a.parent_id == 1
            assert a.score == 58
            assert "Robertson" in a.body_text or "Cauchy-Schwarz" in a.body_text
        finally:
            os.unlink(path)


class TestQAPairs:
    def test_build_qa_pairs(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            # Q1 has accepted answer (2), Q4 has answer (5), Q6 has no answer
            assert len(pairs) == 2
        finally:
            os.unlink(path)

    def test_accepted_answer_preferred(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            q1_pair = next(p for p in pairs if p.question.id == 1)
            assert q1_pair.answer.id == 2  # accepted answer
        finally:
            os.unlink(path)

    def test_qa_tags(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            q1_pair = next(p for p in pairs if p.question.id == 1)
            assert "quantum-mechanics" in q1_pair.tags
        finally:
            os.unlink(path)


class TestEntityConversion:
    def test_qa_to_entity(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            entity = qa_to_entity(pairs[0])
            assert entity["entity/type"] == "QAPair"
            assert entity["entity/source"] == "physics.stackexchange"
            assert "entity/id" in entity
            assert "title" in entity
            assert "question-latex" in entity
            assert "answer-latex" in entity
        finally:
            os.unlink(path)

    def test_qa_to_relations(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            q1_pair = next(p for p in pairs if p.question.id == 1)
            rels = qa_to_relations(q1_pair)
            assert len(rels) == 3  # 3 tags
            assert all(r["type"] == "tagged-with" for r in rels)
        finally:
            os.unlink(path)

    def test_qa_to_entity_custom_site(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            entity = qa_to_entity(pairs[0], site="mathoverflow.net")
            assert entity["entity/source"] == "mathoverflow.net"
            assert entity["entity/id"].startswith("se-mathoverflow-")
        finally:
            os.unlink(path)

    def test_qa_to_relations_custom_site_and_id(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            q1_pair = next(p for p in pairs if p.question.id == 1)
            rels = qa_to_relations(
                q1_pair,
                site="mathoverflow.net",
                entity_id="se-mathoverflow-1",
            )
            assert len(rels) == 3
            assert all(r["from"] == "se-mathoverflow-1" for r in rels)
        finally:
            os.unlink(path)

    def test_tag_entities(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            tags = tag_entities(pairs)
            tag_names = [t["name"] for t in tags]
            assert "quantum-mechanics" in tag_names
            assert "statistical-mechanics" in tag_names
        finally:
            os.unlink(path)


class TestStats:
    def test_corpus_stats(self):
        path = _write_sample_xml()
        try:
            posts = load_posts(path)
            pairs = build_qa_pairs(posts)
            stats = corpus_stats(pairs)
            assert stats["qa_pairs"] == 2
            assert stats["unique_tags"] == 6
            assert stats["with_latex"] == 2
            assert stats["total_latex_fragments"] > 0
        finally:
            os.unlink(path)


class TestLaTeXExtraction:
    """Test LaTeX extraction on realistic SE math content."""

    def test_mathjax_display(self):
        body = "We know that $$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$"
        frags = extract_latex(body)
        assert any("\\int" in f for f in frags)

    def test_mathjax_multiple(self):
        body = (
            "Given $f(x) = x^2$ and $g(x) = e^x$, "
            "we compute $$\\frac{d}{dx}[f(g(x))] = 2e^{2x}$$"
        )
        frags = extract_latex(body)
        assert len(frags) >= 3

    def test_html_encoded_latex(self):
        body = "&lt;p&gt;Consider $E = mc^2$ and $$F = ma$$&lt;/p&gt;"
        frags = extract_latex(body)
        assert len(frags) == 2


# --- Synthetic Comments XML for testing ---

SAMPLE_COMMENTS_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<comments>
  <row Id="101" PostId="1" Score="5" Text="Can you clarify what you mean by physical meaning? The math is standard but the interpretation varies." CreationDate="2024-01-15T10:45:00.000" UserId="42" />
  <row Id="102" PostId="2" Score="3" Text="+1 This is correct and well-explained. See also Griffiths chapter 3." CreationDate="2024-01-15T11:30:00.000" UserId="43" />
  <row Id="103" PostId="2" Score="0" Text="However, the Robertson inequality is not the most general form." CreationDate="2024-01-15T12:15:00.000" UserId="44" />
  <row Id="104" PostId="4" Score="2" Text="For example, consider the case of an ideal gas." CreationDate="2024-02-01T08:30:00.000" UserId="45" />
  <row Id="105" PostId="5" Score="1" Text="I was wrong about the constraint, thanks for the correction." CreationDate="2024-02-01T09:30:00.000" UserId="46" />
  <row Id="106" PostId="99" Score="1" Text="This comment is on a non-existent post and should be filtered." CreationDate="2024-03-01T10:00:00.000" UserId="47" />
</comments>
"""


def _write_sample_comments_xml():
    """Write sample Comments XML to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    f.write(SAMPLE_COMMENTS_XML)
    f.close()
    return f.name


class TestCommentsParsing:
    def test_iter_comments(self):
        path = _write_sample_comments_xml()
        try:
            comments = list(iter_comments(path))
            assert len(comments) == 6
        finally:
            os.unlink(path)

    def test_comment_fields(self):
        path = _write_sample_comments_xml()
        try:
            comments = list(iter_comments(path))
            c = next(c for c in comments if c.id == 101)
            assert c.post_id == 1
            assert c.score == 5
            assert "clarify" in c.text.lower()
            assert c.user_id == 42
        finally:
            os.unlink(path)


class TestThreadConstruction:
    def test_build_threads_streaming(self):
        posts_path = _write_sample_xml()
        comments_path = _write_sample_comments_xml()
        try:
            threads = build_threads_streaming(
                posts_path, comments_path, min_score=0)
            # Q1 (score=42), Q4 (score=15); Q6 (score=-2) filtered by min_score=0
            # But min_score=0 means score >= 0, Q6 has score=-2 so filtered
            assert len(threads) == 2
        finally:
            os.unlink(posts_path)
            os.unlink(comments_path)

    def test_thread_has_all_answers(self):
        posts_path = _write_sample_xml()
        comments_path = _write_sample_comments_xml()
        try:
            threads = build_threads_streaming(
                posts_path, comments_path, min_score=0)
            t1 = next(t for t in threads if t.question.id == 1)
            # Q1 has two answers: id=2 (score=58) and id=3 (score=5)
            assert len(t1.answers) == 2
            # Sorted by score descending
            assert t1.answers[0].score >= t1.answers[1].score
        finally:
            os.unlink(posts_path)
            os.unlink(comments_path)

    def test_thread_has_comments(self):
        posts_path = _write_sample_xml()
        comments_path = _write_sample_comments_xml()
        try:
            threads = build_threads_streaming(
                posts_path, comments_path, min_score=0)
            t1 = next(t for t in threads if t.question.id == 1)
            # Q1 has comment 101 on post 1, comments 102+103 on post 2
            assert 1 in t1.comments
            assert 2 in t1.comments
            assert len(t1.comments[1]) == 1  # comment 101
            assert len(t1.comments[2]) == 2  # comments 102, 103
        finally:
            os.unlink(posts_path)
            os.unlink(comments_path)

    def test_thread_filters_unrelated_comments(self):
        posts_path = _write_sample_xml()
        comments_path = _write_sample_comments_xml()
        try:
            threads = build_threads_streaming(
                posts_path, comments_path, min_score=0)
            # Comment 106 (on post 99) should not appear in any thread
            all_comment_ids = []
            for t in threads:
                for clist in t.comments.values():
                    all_comment_ids.extend(c.id for c in clist)
            assert 106 not in all_comment_ids
        finally:
            os.unlink(posts_path)
            os.unlink(comments_path)

    def test_thread_limit(self):
        posts_path = _write_sample_xml()
        comments_path = _write_sample_comments_xml()
        try:
            threads = build_threads_streaming(
                posts_path, comments_path, min_score=0,
                thread_limit=1)
            assert len(threads) == 1
        finally:
            os.unlink(posts_path)
            os.unlink(comments_path)

    def test_threads_without_comments(self):
        posts_path = _write_sample_xml()
        try:
            threads = build_threads_streaming(
                posts_path, comments_xml_path=None, min_score=0)
            assert len(threads) == 2
            # No comments loaded
            t1 = next(t for t in threads if t.question.id == 1)
            assert len(t1.comments) == 0
        finally:
            os.unlink(posts_path)

    def test_thread_tags(self):
        posts_path = _write_sample_xml()
        try:
            threads = build_threads_streaming(
                posts_path, min_score=0)
            t1 = next(t for t in threads if t.question.id == 1)
            assert "quantum-mechanics" in t1.tags
        finally:
            os.unlink(posts_path)


class TestPerformativeDetection:
    def test_detect_challenge(self):
        hits = detect_performatives("However, this is not quite right.")
        types = [h[0] for h in hits]
        assert "challenge" in types

    def test_detect_query(self):
        hits = detect_performatives("Can you explain why this works?")
        types = [h[0] for h in hits]
        assert "query" in types

    def test_detect_clarify(self):
        hits = detect_performatives("To be precise, we need continuity.")
        types = [h[0] for h in hits]
        assert "clarify" in types

    def test_detect_exemplify(self):
        hits = detect_performatives("For example, consider the case of n=3.")
        types = [h[0] for h in hits]
        assert "exemplify" in types

    def test_detect_reference(self):
        hits = detect_performatives("See also arXiv:2301.12345.")
        types = [h[0] for h in hits]
        assert "reference" in types

    def test_detect_agree(self):
        hits = detect_performatives("+1 This is correct and helpful.")
        types = [h[0] for h in hits]
        assert "agree" in types

    def test_detect_retract(self):
        hits = detect_performatives("I was wrong about the bound.")
        types = [h[0] for h in hits]
        assert "retract" in types

    def test_detect_reform(self):
        hits = detect_performatives("Equivalently, this reduces to a fixed-point theorem.")
        types = [h[0] for h in hits]
        assert "reform" in types

    def test_no_detection_on_plain(self):
        hits = detect_performatives("The Boltzmann distribution is fundamental.")
        # "assert" is the default structural type, not detected by regex
        types = [h[0] for h in hits]
        assert "challenge" not in types
        assert "retract" not in types

    def test_multiple_performatives(self):
        text = "However, I think this is incorrect. For example, consider n=1."
        hits = detect_performatives(text)
        types = [h[0] for h in hits]
        assert "challenge" in types
        assert "exemplify" in types


class TestWiringDiagram:
    def _build_sample_threads(self):
        posts_path = _write_sample_xml()
        comments_path = _write_sample_comments_xml()
        try:
            threads = build_threads_streaming(
                posts_path, comments_path, min_score=0)
            return threads, posts_path, comments_path
        except Exception:
            os.unlink(posts_path)
            os.unlink(comments_path)
            raise

    def test_wiring_diagram_nodes(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)
            # 1 question + 2 answers + 3 comments = 6 nodes
            assert len(diagram.nodes) == 6
            node_types = [n.node_type for n in diagram.nodes]
            assert node_types.count("question") == 1
            assert node_types.count("answer") == 2
            assert node_types.count("comment") == 3
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_wiring_diagram_structural_edges(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)
            # 2 answer edges + 3 comment edges = 5 edges
            assert len(diagram.edges) == 5
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_wiring_diagram_classical_detection(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)
            # Comment 102 has "+1" (agree) and "See also" (reference)
            # Comment 103 has "However" (challenge)
            classical_edges = [e for e in diagram.edges if e.detection == "classical"]
            assert len(classical_edges) >= 2
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_diagram_to_dict(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)
            d = diagram_to_dict(diagram)
            assert "thread_id" in d
            assert "nodes" in d
            assert "edges" in d
            assert "hyperedges" in d
            assert "stats" in d
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_hyperedge_format(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)
            hxs = diagram_to_hyperedges(diagram)
            assert len(hxs) == len(diagram.edges)
            for hx in hxs:
                assert "hx/id" in hx
                assert "hx/type" in hx
                assert "hx/ends" in hx
                assert "hx/content" in hx
                assert "hx/labels" in hx
                assert hx["hx/type"].startswith("thread/")
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_process_threads_to_diagrams(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            diagrams, agg_stats = process_threads_to_diagrams(threads)
            assert len(diagrams) == 2
            assert agg_stats["threads_processed"] == 2
            assert agg_stats["total_nodes"] > 0
            assert agg_stats["total_edges"] > 0
            assert agg_stats["unique_performatives"] > 0
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_process_threads_to_file(self):
        threads, pp, cp = self._build_sample_threads()
        outfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False)
        outfile.close()
        try:
            diagrams, agg_stats = process_threads_to_diagrams(
                threads, output_path=outfile.name)
            import json
            with open(outfile.name) as f:
                data = json.load(f)
            assert len(data) == 2
            assert "thread_id" in data[0]
            assert "nodes" in data[0]
        finally:
            os.unlink(pp)
            os.unlink(cp)
            os.unlink(outfile.name)

    def test_merge_llm_edges(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)

            # Find an existing edge to override
            edge = diagram.edges[0]
            old_type = edge.edge_type
            old_detection = edge.detection

            llm_edges = [{
                "source": edge.source,
                "target": edge.target,
                "performative": "reform",
                "reasoning": "The LLM thinks this reframes the problem",
            }]
            merge_llm_edges(diagram, llm_edges)

            assert edge.edge_type == "reform"
            assert edge.detection == "llm"
            assert "reframes" in edge.evidence
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_merge_llm_edges_invalid_type_skipped(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)
            edge = diagram.edges[0]
            old_type = edge.edge_type

            llm_edges = [{
                "source": edge.source,
                "target": edge.target,
                "performative": "INVALID_TYPE",
            }]
            merge_llm_edges(diagram, llm_edges)

            # Edge should not have changed
            assert edge.edge_type == old_type
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_merge_llm_edges_unmatched_skipped(self):
        threads, pp, cp = self._build_sample_threads()
        try:
            t1 = next(t for t in threads if t.question.id == 1)
            diagram = build_thread_wiring_diagram(t1)
            edges_before = [(e.edge_type, e.detection) for e in diagram.edges]

            llm_edges = [{
                "source": "nonexistent-node",
                "target": "also-nonexistent",
                "performative": "agree",
            }]
            merge_llm_edges(diagram, llm_edges)

            # No edges should have changed
            edges_after = [(e.edge_type, e.detection) for e in diagram.edges]
            assert edges_before == edges_after
        finally:
            os.unlink(pp)
            os.unlink(cp)

    def test_write_thread_wiring_json(self):
        import json
        threads, pp, cp = self._build_sample_threads()
        outfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False)
        outfile.close()
        try:
            diagrams, _ = process_threads_to_diagrams(threads)
            write_thread_wiring_json(diagrams, outfile.name)

            with open(outfile.name) as f:
                data = json.load(f)
            assert len(data) == 2
            assert "thread_id" in data[0]
            assert "hyperedges" in data[0]
        finally:
            os.unlink(pp)
            os.unlink(cp)
            os.unlink(outfile.name)
