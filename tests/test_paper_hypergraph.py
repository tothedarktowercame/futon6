"""Tests for Stage 5d — paper hypergraph.

Spec: futon6/holes/missions/M-paper-reverse-morphogenesis.md §5d.
"""

from futon6.paper_hypergraph import (
    _parse_llm_edges,
    extract_paper_hypergraph_classical,
    merge_paper_hypergraphs,
    parse_latex_blocks,
)


SAMPLE_LATEX = r"""
\section{Introduction}
We study the Borel completion adjunction.

\section{Main results}

\begin{definition}\label{def:borel-completion}
The \emph{Borel completion} $\widehat{X}$ of a measurable space $X$ is...
\end{definition}

\begin{theorem}\label{thm:main}
For any measurable space $X$, the Borel completion adjunction yields a
functorial lift of structure.
\end{theorem}

\begin{proof}
Apply Lemma~\ref{lem:lift} together with the Cartan-Eilenberg resolution.
By \cite{Bourbaki63}, the Borel completion is unique up to isomorphism.
\end{proof}

\begin{lemma}\label{lem:lift}
Every measurable map lifts to the completion.
\end{lemma}

\begin{proof}
By standard spectral sequence computation.
\end{proof}
"""


def test_parse_latex_blocks_finds_environments():
    blocks, section_spans = parse_latex_blocks(SAMPLE_LATEX)
    envs = [b.env for b in blocks]
    assert envs.count("theorem") == 1
    assert envs.count("lemma") == 1
    assert envs.count("proof") == 2
    assert envs.count("definition") == 1
    # Two top-level sections
    assert len(section_spans) >= 2


def test_parse_latex_blocks_captures_labels():
    blocks, _ = parse_latex_blocks(SAMPLE_LATEX)
    labels = {b.label for b in blocks if b.label}
    assert "thm:main" in labels
    assert "lem:lift" in labels
    assert "def:borel-completion" in labels


def test_classical_emits_typed_nodes():
    hg = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        concepts=["measurable space", "functorial lift"],
        techniques=["Borel completion adjunction",
                    "Cartan-Eilenberg resolution",
                    "spectral sequence computation"],
    )
    node_types = {n["type"] for n in hg["nodes"]}
    assert "claim" in node_types
    assert "proof" in node_types
    assert "definition" in node_types
    assert "technique" in node_types
    assert "concept" in node_types
    assert "citation" in node_types


def test_classical_derivation_edge_links_theorem_to_proof():
    hg = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        techniques=["Borel completion adjunction",
                    "Cartan-Eilenberg resolution"],
    )
    derivations = [e for e in hg["edges"] if e["type"] == "derivation"]
    assert len(derivations) >= 1
    first = derivations[0]
    # Target is the theorem, proof is the proof, and Cartan-Eilenberg
    # (mentioned in the proof body) appears as depends_on.
    assert any(e.startswith("claim:theorem") for e in first["ends"])
    assert any(e.startswith("proof:") for e in first["ends"])
    assert any(e == "technique:cartan-eilenberg-resolution"
               for e in first["ends"])


def test_classical_derivation_edge_captures_ref():
    """The proof of thm:main references lem:lift via \\ref{lem:lift}. That
    reference should appear as a depends_on end."""
    hg = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        techniques=["Cartan-Eilenberg resolution"],
    )
    derivations = [e for e in hg["edges"] if e["type"] == "derivation"]
    thm_deriv = next(
        (e for e in derivations
         if any(x.startswith("claim:theorem") for x in e["ends"])),
        None,
    )
    assert thm_deriv is not None
    # claim:lemma-1 should be a depends_on end
    lemma_end = next((e for e in thm_deriv["ends"]
                      if e.startswith("claim:lemma")), None)
    assert lemma_end is not None
    assert thm_deriv["roles"].get(lemma_end) == "depends_on"


def test_classical_citation_grounding_edge():
    hg = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        techniques=["Cartan-Eilenberg resolution"],
    )
    cites = [e for e in hg["edges"] if e["type"] == "citation-grounding"]
    assert len(cites) == 1
    assert any(end == "citation:bourbaki63" for end in cites[0]["ends"])


def test_classical_definition_use_edge():
    hg = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        concepts=["measurable space"],
    )
    def_uses = [e for e in hg["edges"] if e["type"] == "definition-use"]
    # "measurable space" is defined (mentioned in the definition block)
    # and used later (in thm:main body).
    assert len(def_uses) >= 1


def test_classical_handles_prose_without_latex():
    """Papers without LaTeX blocks should still produce a valid (if thin)
    hypergraph: concept/technique nodes, but no claim/proof nodes."""
    hg = extract_paper_hypergraph_classical(
        "This paper discusses functors and categories in abstract terms.",
        "test-prose",
        concepts=["functor", "category"],
        techniques=[],
    )
    assert hg["meta"]["n_blocks"] == 0
    assert hg["meta"]["has_theorem_blocks"] is False
    types = {n["type"] for n in hg["nodes"]}
    assert "concept" in types
    # No claim or proof nodes
    assert "claim" not in types
    assert "proof" not in types


def test_classical_all_edges_have_classical_provenance():
    hg = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        concepts=["measurable space"],
        techniques=["Cartan-Eilenberg resolution"],
    )
    for e in hg["edges"]:
        assert e["attrs"]["provenance"] == "classical"


def test_llm_edge_parser_accepts_valid_json():
    raw = '''Here are edges:
[{"type": "motivation-link",
  "ends": ["claim:theorem-1", "technique:borel-completion-adjunction"],
  "roles": {"claim:theorem-1": "resolves"},
  "rationale": "intro motivates the theorem"}]
Done.'''
    edges = _parse_llm_edges(raw)
    assert len(edges) == 1
    assert edges[0]["type"] == "motivation-link"
    assert edges[0]["attrs"]["provenance"] == "llm"
    assert "intro motivates" in edges[0]["attrs"]["rationale"]


def test_llm_edge_parser_rejects_malformed():
    assert _parse_llm_edges("no edges here") == []
    assert _parse_llm_edges("[{not valid}]") == []
    assert _parse_llm_edges('{"not": "array"}') == []


def test_llm_edge_parser_drops_edges_missing_required_fields():
    raw = '[{"type": "derivation"}, {"ends": ["a", "b"]}, {"type": "x", "ends": ["a"]}]'
    edges = _parse_llm_edges(raw)
    # Only the third (has both type and ends) survives.
    assert len(edges) == 1
    assert edges[0]["type"] == "x"


def test_merge_new_edges_carry_llm_provenance():
    classical = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        techniques=["Borel completion adjunction"],
    )
    llm_edges = [{
        "type": "motivation-link",
        "ends": ["claim:theorem-1", "technique:borel-completion-adjunction"],
        "roles": {"claim:theorem-1": "resolves"},
        "attrs": {"provenance": "llm", "rationale": "paper motivates this"},
    }]
    merged = merge_paper_hypergraphs(classical, llm_edges)
    ml_edges = [e for e in merged["edges"] if e["type"] == "motivation-link"]
    assert len(ml_edges) == 1
    assert ml_edges[0]["attrs"]["provenance"] == "llm"
    assert merged["meta"]["n_llm_new_edges"] == 1


def test_merge_duplicate_edge_becomes_both():
    classical = extract_paper_hypergraph_classical(
        SAMPLE_LATEX, "test-001",
        techniques=["Cartan-Eilenberg resolution"],
    )
    # Find an existing derivation edge and build an LLM edge with the
    # same type + end set.
    deriv = next(e for e in classical["edges"] if e["type"] == "derivation")
    llm_edges = [{
        "type": "derivation",
        "ends": list(deriv["ends"]),
        "roles": {},
        "attrs": {"provenance": "llm", "rationale": "LLM agrees"},
    }]
    merged = merge_paper_hypergraphs(classical, llm_edges)
    matching = [e for e in merged["edges"]
                if e["type"] == "derivation"
                and set(e["ends"]) == set(deriv["ends"])]
    assert len(matching) == 1  # not duplicated
    assert matching[0]["attrs"]["provenance"] == "both"
    assert matching[0]["attrs"].get("llm_rationale") == "LLM agrees"
    assert merged["meta"]["n_llm_confirmed_edges"] == 1
