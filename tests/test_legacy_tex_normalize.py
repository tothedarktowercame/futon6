from futon6.legacy_tex_normalize import (
    normalize,
    parse_newenvironment_aliases,
    parse_newtheorem_aliases,
)
from futon6.paper_hypergraph import extract_paper_hypergraph_classical


def test_parse_newtheorem_aliases_recognizes_canonical_titles():
    text = r"""
    \newtheorem{thm}{Theorem}
    \newtheorem{lem}{Lemma}
    \newtheorem{rmk}{Remark}
    """
    aliases = parse_newtheorem_aliases(text)
    assert aliases == {"thm": "theorem", "lem": "lemma", "rmk": "remark"}


def test_parse_newtheorem_aliases_ignores_unknown_titles():
    text = r"""
    \newtheorem{foo}{Observation}
    \newtheorem{theorem}{Theorem}
    """
    aliases = parse_newtheorem_aliases(text)
    assert aliases == {}


def test_parse_newtheorem_aliases_strips_tex_formatting_and_punctuation():
    text = r"""
    \newtheorem{Th}{\bf Theorem}[section]
    \newtheorem{Prop}[Th]{\bf Proposition}
    \newtheorem{satz}[lemma]{Theorem.}
    """
    aliases = parse_newtheorem_aliases(text)
    assert aliases == {
        "Th": "theorem",
        "Prop": "proposition",
        "satz": "theorem",
    }


def test_parse_newenvironment_aliases_recognizes_declared_wrappers():
    text = r"""
    \newenvironment{Thm}{\paragraph{Theorem:}\em}{\par}
    \newenvironment{Defn}{\paragraph{Definition:}}{\par}
    \newenvironment{prf}[1]{\begin{trivlist}\item[{\bf Proof}#1.]}{\end{trivlist}}
    """
    aliases = parse_newenvironment_aliases(text)
    assert aliases == {
        "Thm": "theorem",
        "Defn": "definition",
        "prf": "proof",
    }


def test_normalize_rewrites_alias_begin_end_tokens():
    text = r"""
    \newtheorem{thm}{Theorem}
    \begin{thm}\label{thm:main}
    Main statement.
    \end{thm}
    """
    result = normalize(text, paper_id="legacy-001")
    assert r"\begin{theorem}" in result.rewritten_text
    assert r"\end{theorem}" in result.rewritten_text
    assert r"\begin{thm}" not in result.rewritten_text
    assert r"\end{thm}" not in result.rewritten_text
    assert result.alias_map == {"thm": "theorem"}
    assert len(result.rewrites) == 2
    assert all(r.kind == "alias-expanded" for r in result.rewrites)
    assert result.block_annotations
    ann = next(iter(result.block_annotations.values()))
    assert ann["block_origin"] == "alias_expanded"
    assert ann["original_env"] == "thm"
    assert ann["canonical_env"] == "theorem"
    assert all(r.rewritten_span_end > r.rewritten_span_start for r in result.rewrites)


def test_normalize_rewrites_newenvironment_wrappers():
    text = r"""
    \newenvironment{Thm}{\paragraph{Theorem:}\em}{\par}
    \begin{Thm}\label{thm:main}
    Main statement.
    \end{Thm}
    """
    result = normalize(text, paper_id="legacy-003")
    assert r"\begin{theorem}" in result.rewritten_text
    assert r"\end{theorem}" in result.rewritten_text
    assert result.alias_map == {"Thm": "theorem"}
    ann = next(iter(result.block_annotations.values()))
    assert ann["original_env"] == "Thm"
    assert ann["canonical_env"] == "theorem"


def test_normalize_rewrites_proof_alias_args_to_optional_title():
    text = r"""
    \newenvironment{prf}[1]{\begin{trivlist}\item[{\bf Proof}#1.]}{\end{trivlist}}
    \begin{prf}{continued}
    Details.
    \end{prf}
    """
    result = normalize(text, paper_id="legacy-004")
    assert r"\begin{proof}[continued]" in result.rewritten_text
    assert r"\end{proof}" in result.rewritten_text
    assert result.alias_map == {"prf": "proof"}
    ann = next(iter(result.block_annotations.values()))
    assert ann["canonical_env"] == "proof"
    assert ann["original_env"] == "prf"


def test_normalize_rewrites_wrapper_macros_to_canonical_envs():
    text = r"""
    \newcommand{\be}[1]{\begin{#1}}
    \newcommand{\ee}[1]{\end{#1}}
    \newtheorem{thm}{Theorem}
    \be{thm}
    Main statement.
    \ee{thm}
    """
    result = normalize(text, paper_id="legacy-005")
    assert r"\begin{theorem}" in result.rewritten_text
    assert r"\end{theorem}" in result.rewritten_text
    assert r"\be{thm}" not in result.rewritten_text
    ann = next(iter(result.block_annotations.values()))
    assert ann["canonical_env"] == "theorem"
    assert ann["original_env"] == "thm"


def test_normalize_rewrites_zero_arg_wrapper_macros():
    text = r"""
    \newcommand{\btheor}{\begin{theorem}}
    \newcommand{\etheor}{\end{theorem}}
    \btheor
    Main statement.
    \etheor
    """
    result = normalize(text, paper_id="legacy-005b")
    assert r"\begin{theorem}" in result.rewritten_text
    assert r"\end{theorem}" in result.rewritten_text
    assert r"\btheor" not in result.rewritten_text
    assert r"\etheor" not in result.rewritten_text


def test_normalize_recovers_undeclared_standard_env_aliases():
    text = r"""
    \begin{thm}
    Main statement.
    \end{thm}
    """
    result = normalize(text, paper_id="legacy-006")
    assert r"\begin{theorem}" in result.rewritten_text
    assert r"\end{theorem}" in result.rewritten_text
    assert result.alias_map == {"thm": "theorem"}
    ann = next(iter(result.block_annotations.values()))
    assert ann["source_cue"] == "envname-heuristic alias thm->theorem"


def test_normalize_rewrites_let_alias_theorem_and_proof_commands():
    text = r"""
    \let\lem\lemma
    \let\eth\endtheorem
    \let\prf\proof
    \let\frp\endproof
    \begin{document}
    \lem
    Main statement.
    \eth
    \prf
    Proof details.
    \frp
    """
    result = normalize(text, paper_id="legacy-006b")
    assert r"\begin{lemma}" in result.rewritten_text
    assert r"\end{lemma}" in result.rewritten_text
    assert r"\begin{proof}" in result.rewritten_text
    assert r"\end{proof}" in result.rewritten_text
    hypergraph = extract_paper_hypergraph_classical(
        result.rewritten_text,
        paper_id="legacy-006b",
        block_annotations=result.block_annotations,
    )
    claim_nodes = [n for n in hypergraph["nodes"] if n["type"] == "claim"]
    proof_nodes = [n for n in hypergraph["nodes"] if n["type"] == "proof"]
    assert len(claim_nodes) == 1
    assert len(proof_nodes) == 1
    assert claim_nodes[0]["attrs"]["block_origin"] == "alias_expanded"
    assert "let alias lem->lemma" in claim_nodes[0]["attrs"]["source_cue"]


def test_normalize_synthesizes_paragraph_head_claim_blocks():
    text = r"""
    \begin{document}
    \paragraph{Lemma.}
    Any three random variables satisfy the triangle inequality.
    \begin{proof}
    Proof details.
    \end{proof}
    """
    result = normalize(text, paper_id="legacy-007")
    assert r"\begin{lemma}" in result.rewritten_text
    assert r"\end{lemma}" in result.rewritten_text
    assert result.block_annotations
    hypergraph = extract_paper_hypergraph_classical(
        result.rewritten_text,
        paper_id="legacy-007",
        block_annotations=result.block_annotations,
    )
    claim_nodes = [n for n in hypergraph["nodes"] if n["type"] == "claim"]
    proof_nodes = [n for n in hypergraph["nodes"] if n["type"] == "proof"]
    assert len(claim_nodes) == 1
    assert len(proof_nodes) == 1
    assert claim_nodes[0]["attrs"]["block_origin"] == "prose_synthesized"
    assert "paragraph head Lemma.->lemma" in claim_nodes[0]["attrs"]["source_cue"]


def test_normalize_leaves_canonical_envs_unchanged():
    text = r"""
    \begin{theorem}
    Canonical already.
    \end{theorem}
    """
    result = normalize(text, paper_id="legacy-002")
    assert result.rewritten_text == text
    assert result.rewrites == []
    assert result.alias_map == {}
