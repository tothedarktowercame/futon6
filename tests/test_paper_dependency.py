from futon6.paper_dependency import build_dependency_graph
from futon6.paper_hypergraph import extract_paper_hypergraph_classical


FIXTURE = r"""
\begin{definition}\label{def:base}Base object.\end{definition}
\begin{lemma}\label{lem:one}By \cref{def:base}.\end{lemma}
\begin{equation}\label{eq:one}x=y\end{equation}
\begin{theorem}\label{thm:two}By \ref{eq:one}.\end{theorem}
\begin{proof}Use \ref{lem:one}.\end{proof}
"""


def test_d1_d2_and_unresolved_equation_reference():
    graph = build_dependency_graph(
        extract_paper_hypergraph_classical(FIXTURE, "fixture"), FIXTURE
    )
    assert graph["counts"] == {"D1": 1, "D2": 1}
    d1 = next(edge for edge in graph["edges"] if edge["rule"] == "D1")
    d2 = next(edge for edge in graph["edges"] if edge["rule"] == "D2")
    assert d1["source"] == "claim:lemma-1"
    assert d1["target"] == "claim:theorem-1"
    assert d1["style"] == "solid"
    assert d2["source"] == "definition:def-1"
    assert d2["target"] == "claim:lemma-1"
    assert d2["style"] == "dashed"
    assert graph["unresolved_refs"] == {"eq": 1}


CYCLE_FIXTURE = r"""
\begin{lemma}\label{lem:a}See \ref{lem:b}.\end{lemma}
\begin{lemma}\label{lem:b}See \ref{lem:a}.\end{lemma}
\begin{lemma}\label{lem:c}Tail: by \ref{lem:a}.\end{lemma}
"""


def test_d2_cycle_flags_and_ids_are_stable():
    hg = extract_paper_hypergraph_classical(CYCLE_FIXTURE, "cycle")
    first = build_dependency_graph(hg, CYCLE_FIXTURE)
    second = build_dependency_graph(hg, CYCLE_FIXTURE)
    assert first["counts"] == {"D2": 3}
    assert first["cycle_scc_count"] == 1
    # Only the two mutual edges are in the cycle; the tail edge a -> c is not.
    flagged = {(e["source"], e["target"]) for e in first["edges"] if e["cycle"]}
    assert flagged == {("claim:lemma-1", "claim:lemma-2"), ("claim:lemma-2", "claim:lemma-1")}
    assert not next(e for e in first["edges"] if e["target"] == "claim:lemma-3")["cycle"]
    assert [edge["id"] for edge in first["edges"]] == [edge["id"] for edge in second["edges"]]


D3_FIXTURE = r"""
\begin{theorem}\label{thm:main}Main theorem.\end{theorem}
\begin{lemma}\label{lem:first}First lemma.\end{lemma}
\begin{lemma}\label{lem:second}Second lemma.\end{lemma}
\begin{proof}[Proof of \cref{thm:main}]
Use \ref{lem:first}.
\end{proof}
"""


def test_d3_proof_header_replaces_positional_claim():
    graph = build_dependency_graph(
        extract_paper_hypergraph_classical(D3_FIXTURE, "d3"), D3_FIXTURE
    )
    assert graph["counts"] == {"D3": 1}
    edge = graph["edges"][0]
    assert edge["rule"] == "D3"
    assert edge["source"] == "claim:lemma-1"
    assert edge["target"] == "claim:theorem-1"
    assert edge["attrs"] == {
        "title": r"Proof of \cref{thm:main}",
        "label": "thm:main",
        "displaced": "claim:lemma-2",
    }


def test_d3_unknown_header_keeps_d1_and_counts_unresolved():
    fixture = r"""
    \begin{lemma}\label{lem:first}First lemma.\end{lemma}
    \begin{lemma}\label{lem:prior}Prior lemma.\end{lemma}
    \begin{proof}[Proof of \cref{thm:missing}]Use \ref{lem:first}.\end{proof}
    """
    graph = build_dependency_graph(
        extract_paper_hypergraph_classical(fixture, "unknown"), fixture
    )
    edge = next(edge for edge in graph["edges"] if edge["rule"] == "D1")
    assert edge["target"] == "claim:lemma-2"
    assert graph["unresolved_refs"]["proof-header"] == 1


def test_d3_header_confirming_preceding_claim_keeps_d1():
    fixture = r"""
    \begin{lemma}\label{lem:first}First lemma.\end{lemma}
    \begin{lemma}\label{lem:prior}Prior lemma.\end{lemma}
    \begin{proof}[Proof of \cref{lem:prior}]Use \ref{lem:first}.\end{proof}
    """
    graph = build_dependency_graph(
        extract_paper_hypergraph_classical(fixture, "confirmed"), fixture
    )
    edge = next(edge for edge in graph["edges"] if edge["rule"] == "D1")
    assert edge["attrs"]["header-confirmed"] is True
    assert edge["attrs"]["displaced"] is None
