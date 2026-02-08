"""Tests for StackExchange data processing pipeline."""

import tempfile
import os
from pathlib import Path

from futon6.stackexchange import (
    strip_html,
    extract_latex,
    parse_se_tags,
    iter_posts,
    load_posts,
    build_qa_pairs,
    qa_to_entity,
    qa_to_relations,
    tag_entities,
    corpus_stats,
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
