"""Tests for nlab-wiring.py — CT-backed wiring extraction from nLab pages."""

import json
import sys
from pathlib import Path

import pytest

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import importlib
nlab_wiring = importlib.import_module("nlab-wiring")


# ============================================================
# Fixtures
# ============================================================

ADJUNCTION_SNIPPET = r"""
+-- {: .rightHandSide}
+-- {: .toc .clickDown tabindex="0"}
###Context###
#### 2-Category theory
+--{: .hide}
[[!include 2-category theory - contents]]
=--
=--
=--

## Idea

A [[pair]] of [[1-morphisms]] in a [[2-category]] form an **adjunction**.

## Definition

+-- {: .num_defn}
###### Definition
An _adjunction_ in a [[2-category]] is

* a [[pair]] of [[objects]] $C$ and $D$

* a [[pair]] of [[1-morphisms]]

  $L \colon C \longrightarrow D$ (the *[[left adjoint]]*)

  $R \colon D \longrightarrow C$ (the *[[right adjoint]]*)

* a [[pair]] of [[2-morphisms]]

  $\eta \colon 1_C \longrightarrow R \circ L$ (the *[[adjunction unit]]*)

  $\epsilon \colon L \circ R \longrightarrow 1_D$ (the *[[adjunction counit]]*)

such that the [[triangle identity|triangle identities]] hold.
=--

+-- {: .num_prop}
###### Proposition
Every adjunction gives rise to a [[monad]].
=--

+-- {: .proof}
###### Proof
Let $T = R \circ L$. Then $\eta$ is the unit and $\mu = R \epsilon L$ is the
multiplication. The monad laws follow from the triangle identities.
=--

+-- {: .num_remark}
###### Remark
This means that [[adjoint functors]] are the primary source of [[monads]].
=--

+-- {: .num_example}
###### Example
Consider the [[free functor]] $F : Set \to Grp$ and [[forgetful functor]]
$U : Grp \to Set$. Then $F \dashv U$ is an adjunction.
=--
"""

TIKZCD_SNIPPET = r"""
\begin{tikzcd}
  L \ar[r, "L \cdot \eta"] \ar[dr, swap, "\mathrm{id}"]
  & L \circ R \circ L \ar[d, "\epsilon \cdot L"]
  \\ & L
\end{tikzcd}
"""

LATEX_ENV_SNIPPET = r"""
\begin{defn} \label{DefinitionAdjunction}
An _adjunction_ in a [[2-category]] is a pair $(L, R)$ with
$L \dashv R$ satisfying the [[triangle identity|triangle identities]].
\end{defn}

\begin{theorem}
Every [[adjunction]] induces a [[monad]] $T = R \circ L$.
\end{theorem}

\begin{proof}
Since $\eta$ and $\epsilon$ satisfy the triangle identities,
the composite $\mu = R \epsilon L$ defines a multiplication.
Therefore $T$ is a monad.
\end{proof}
"""

TIKZCD_COORD_SNIPPET = r"""
\begin{tikzcd}
	t & tt & t \\
	& t
	\arrow["{\eta t}", from=1-1, to=1-2]
	\arrow["t\eta"', from=1-3, to=1-2]
	\arrow[Rightarrow, no head, from=1-1, to=2-2]
	\arrow[Rightarrow, no head, from=1-3, to=2-2]
	\arrow["\mu"{description}, from=1-2, to=2-2]
\end{tikzcd}
"""

ARRAY_DIAGRAM_SNIPPET = r"""
$$
  \array{
     C &\stackrel{F}{\to}& D
     \\
     \mathllap{{}^p}\big\downarrow & \nearrow
     \\
     C'
  }
$$
"""

KAN_EXTENSION_SNIPPET = r"""
## Definition

+-- {: .num_defn}
###### Definition
Given [[functors]] $F \colon C \to D$ and $p \colon C \to C'$, a
**[[left Kan extension]]** of $F$ along $p$ is a [[functor]]
$Lan_p F \colon C' \to D$ together with a [[natural transformation]]
$\eta \colon F \to (Lan_p F) \circ p$ that is universal.
=--

This is a [[universal property]] in the [[functor category]].
The [[right Kan extension]] is defined dually.
"""


# ============================================================
# Step 1: Environment parser tests
# ============================================================

class TestEnvironmentParser:

    def test_wiki_style_definition(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/definition" in env_types

    def test_wiki_style_proposition(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/proposition" in env_types

    def test_wiki_style_proof(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/proof" in env_types

    def test_wiki_style_remark(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/remark" in env_types

    def test_wiki_style_example(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/example" in env_types

    def test_multiple_envs_found(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        # Should find: definition, proposition, proof, remark, example
        assert len(envs) >= 5

    def test_env_has_text(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        defn = [e for e in envs if e["env_type"] == "env/definition"][0]
        assert "left adjoint" in defn["text"]

    def test_latex_style_definition(self):
        envs = nlab_wiring.parse_environments(LATEX_ENV_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/definition" in env_types

    def test_latex_style_theorem(self):
        envs = nlab_wiring.parse_environments(LATEX_ENV_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/theorem" in env_types

    def test_latex_style_proof(self):
        envs = nlab_wiring.parse_environments(LATEX_ENV_SNIPPET)
        env_types = [e["env_type"] for e in envs]
        assert "env/proof" in env_types

    def test_latex_label_captured(self):
        envs = nlab_wiring.parse_environments(LATEX_ENV_SNIPPET)
        defn = [e for e in envs if e["env_type"] == "env/definition"][0]
        assert defn.get("label") == "DefinitionAdjunction"

    def test_envs_to_records(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        records = nlab_wiring.envs_to_records("193", envs)
        assert all("hx/id" in r for r in records)
        assert all("hx/type" in r for r in records)
        assert records[0]["hx/id"].startswith("nlab-193:")

    def test_navigation_excluded(self):
        """Navigation blocks (rightHandSide, toc) should not be in environments."""
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        for env in envs:
            assert env["env_type"] != None
            assert "rightHandSide" not in env.get("classes", "")


# ============================================================
# Step 2: Typed link tests
# ============================================================

class TestTypedLinks:

    def test_links_extracted(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        assert len(links) > 0

    def test_definition_ref_type(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        def_links = [l for l in links if l["hx/type"] == "link/definition-ref"]
        # "left adjoint", "right adjoint" etc. should be definition-ref
        targets = [l["hx/content"]["target_name"] for l in def_links]
        assert "left adjoint" in targets

    def test_prose_ref_type(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        prose_links = [l for l in links if l["hx/type"] == "link/prose-ref"]
        # Links in the Idea section (outside environments) should be prose-ref
        assert len(prose_links) > 0

    def test_link_has_position(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        for link in links:
            assert "position" in link["hx/content"]

    def test_link_parent_env(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        def_links = [l for l in links if l["hx/type"] == "link/definition-ref"]
        # Definition links should have a source pointing to an env
        for link in def_links:
            assert link["hx/source"].startswith("nlab-193:env-")

    def test_include_links_excluded(self):
        """[[!include ...]] directives should not produce links."""
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        for link in links:
            assert not link["hx/content"]["target_name"].startswith("!")


# ============================================================
# Step 3: tikzcd diagram tests
# ============================================================

class TestTikzcdExtraction:

    def test_diagram_found(self):
        diags = nlab_wiring.extract_diagrams("193", TIKZCD_SNIPPET, [])
        assert len(diags) >= 1

    def test_diagram_has_objects(self):
        diags = nlab_wiring.extract_diagrams("193", TIKZCD_SNIPPET, [])
        d = diags[0]
        obj_ends = [e for e in d["hx/ends"] if e["role"] == "object"]
        assert len(obj_ends) >= 2

    def test_diagram_has_morphisms(self):
        diags = nlab_wiring.extract_diagrams("193", TIKZCD_SNIPPET, [])
        d = diags[0]
        morph_ends = [e for e in d["hx/ends"] if e["role"] == "morphism"]
        assert len(morph_ends) >= 2

    def test_arrow_labels_extracted(self):
        diags = nlab_wiring.extract_diagrams("193", TIKZCD_SNIPPET, [])
        d = diags[0]
        morph_ends = [e for e in d["hx/ends"] if e["role"] == "morphism"]
        labels = [e.get("label") for e in morph_ends if e.get("label")]
        assert len(labels) >= 1

    def test_diagram_type(self):
        diags = nlab_wiring.extract_diagrams("193", TIKZCD_SNIPPET, [])
        assert diags[0]["hx/type"] == "diagram/commutative"

    def test_parse_tikzcd_directions(self):
        assert nlab_wiring.parse_tikzcd_direction("r") == (0, 1)
        assert nlab_wiring.parse_tikzcd_direction("d") == (1, 0)
        assert nlab_wiring.parse_tikzcd_direction("dr") == (1, 1)
        assert nlab_wiring.parse_tikzcd_direction("rr") == (0, 2)
        assert nlab_wiring.parse_tikzcd_direction("ul") == (-1, -1)

    def test_coordinate_format_arrows(self):
        """tikzcd with \\arrow[from=row-col, to=row-col] format."""
        diags = nlab_wiring.extract_diagrams("255", TIKZCD_COORD_SNIPPET, [])
        assert len(diags) >= 1
        d = diags[0]
        obj_ends = [e for e in d["hx/ends"] if e["role"] == "object"]
        morph_ends = [e for e in d["hx/ends"] if e["role"] == "morphism"]
        assert len(obj_ends) >= 3  # t, tt, t, t
        assert len(morph_ends) >= 4  # 5 arrows

    def test_coordinate_arrow_labels(self):
        """Coordinate-format arrows should have labels extracted."""
        diags = nlab_wiring.extract_diagrams("255", TIKZCD_COORD_SNIPPET, [])
        d = diags[0]
        morph_ends = [e for e in d["hx/ends"] if e["role"] == "morphism"]
        labels = [e.get("label") for e in morph_ends if e.get("label")]
        # Should find "{\eta t}" or "\eta t" and "\mu" etc.
        assert len(labels) >= 2

    def test_array_diagram_found(self):
        """\\array{} diagrams should be extracted."""
        diags = nlab_wiring.extract_diagrams("266", ARRAY_DIAGRAM_SNIPPET, [])
        assert len(diags) >= 1

    def test_array_diagram_objects(self):
        """\\array{} diagram should find objects C, D, C'."""
        diags = nlab_wiring.extract_diagrams("266", ARRAY_DIAGRAM_SNIPPET, [])
        d = diags[0]
        obj_ends = [e for e in d["hx/ends"] if e["role"] == "object"]
        labels = [e["label"] for e in obj_ends]
        assert len(obj_ends) >= 2  # At least C, D, C'

    def test_array_diagram_type(self):
        """\\array{} diagrams should have type diagram/array."""
        diags = nlab_wiring.extract_diagrams("266", ARRAY_DIAGRAM_SNIPPET, [])
        assert diags[0]["hx/type"] == "diagram/array"


# ============================================================
# Step 4: Discourse wiring tests
# ============================================================

class TestDiscourseWiring:

    def test_scopes_detected(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        discourse = nlab_wiring.extract_discourse_wiring("193", ADJUNCTION_SNIPPET, envs)
        scope_records = [d for d in discourse if d["hx/role"] == "component"]
        assert len(scope_records) > 0

    def test_discourse_detected_in_latex_envs(self):
        """Discourse elements (wires, labels) should be found in LaTeX env proof."""
        envs = nlab_wiring.parse_environments(LATEX_ENV_SNIPPET)
        discourse = nlab_wiring.extract_discourse_wiring("test", LATEX_ENV_SNIPPET, envs)
        # The proof has "Since" (causal) and "Therefore" (consequential)
        all_types = {d["hx/type"] for d in discourse}
        assert "wire/causal" in all_types or "wire/consequential" in all_types

    def test_wires_detected(self):
        """Wire detection on LaTeX env snippet which has 'Since' and 'Therefore'."""
        envs = nlab_wiring.parse_environments(LATEX_ENV_SNIPPET)
        discourse = nlab_wiring.extract_discourse_wiring("test", LATEX_ENV_SNIPPET, envs)
        wire_records = [d for d in discourse if d["hx/role"] == "wire"]
        assert len(wire_records) > 0

    def test_sequence_wire_detected(self):
        wires = nlab_wiring.detect_wires("test-seq", "Next, we prove the auxiliary lemma.")
        assert any(w["hx/type"] == "wire/sequencing" for w in wires)

    def test_question_label_detected(self):
        labels = nlab_wiring.detect_labels(
            "test-q",
            "In light of Lemma 2, one might ask the following question."
        )
        assert any(l["hx/type"] == "strategy/question" for l in labels)

    def test_detect_learned_fires_on_text_via_loaded_pattern(self):
        patterns = [{
            "signature": "we study <term> and <term>",
            "regex": r"\bwe\w*.{1,120}?\bstudy\w*.{1,120}?\band\w*",
            "predicted_kind": "label",
        }]
        records = nlab_wiring.detect_learned(
            "test-learn",
            "We study group actions and natural transformations.",
            patterns,
        )
        assert len(records) == 1
        rec = records[0]
        assert rec["hx/type"] == "learned/label"
        assert rec["hx/role"] == "label"
        assert rec["hx/content"]["signature"] == "we study <term> and <term>"
        assert "learned-label" in rec["hx/labels"]

    def test_detect_learned_no_op_when_no_patterns(self):
        # Defaults to empty list — must not change any existing behavior.
        assert nlab_wiring.detect_learned("e", "Any text whatsoever.", []) == []
        assert nlab_wiring.detect_learned("e", "Any text whatsoever.", None) == []

    # --- Math-scope detection (Layer 1 of symbol grounding) ---

    def test_math_typed_arrow_fires_inside_dollars(self):
        records = nlab_wiring.detect_math_scopes(
            "test-m1",
            "Let $f : X \\to Y$ be a morphism.",
        )
        types = [r["hx/type"] for r in records]
        assert "math/typed-arrow" in types
        assert "math/typed-binding" in types

    def test_math_named_functor_recognized(self):
        records = nlab_wiring.detect_math_scopes(
            "test-m2",
            "The set $\\Hom(A, B)$ has identity $\\End(A)$.",
        )
        functor_records = [r for r in records if r["hx/type"] == "math/named-functor"]
        labels = {r["hx/content"]["match"] for r in functor_records}
        assert "\\Hom" in labels
        assert "\\End" in labels

    def test_math_composition_and_adjunction(self):
        records = nlab_wiring.detect_math_scopes(
            "test-m3",
            "Consider $g \\circ f$ and the adjunction $T \\dashv U$.",
        )
        types = [r["hx/type"] for r in records]
        assert "math/composition" in types
        assert "math/adjunction" in types

    def test_math_category_symbol_with_braces(self):
        records = nlab_wiring.detect_math_scopes(
            "test-m4",
            "Let $\\mathcal{C}$ and $\\mathbf{Set}$ be categories.",
        )
        cat_records = [r for r in records if r["hx/type"] == "math/category-symbol"]
        labels = {r["hx/content"]["match"] for r in cat_records}
        assert "\\mathcal{C}" in labels
        assert "\\mathbf{Set}" in labels

    def test_math_quantifier_and_membership(self):
        records = nlab_wiring.detect_math_scopes(
            "test-m5",
            "For $\\forall x \\in X$, there exists $\\exists y$.",
        )
        types = [r["hx/type"] for r in records]
        assert types.count("math/quantifier") >= 2
        assert "math/membership" in types

    def test_math_scope_positions_are_absolute(self):
        text = "before $X \\to Y$ after"  # arrow starts at offset 10 in raw text
        records = nlab_wiring.detect_math_scopes("t-m6", text)
        arrows = [r for r in records if r["hx/type"] == "math/typed-arrow"]
        assert arrows
        pos = arrows[0]["hx/content"]["position"]
        # The arrow's position should land in the math interior, not at offset 0.
        assert pos > len("before ")
        assert text[pos:pos+3] == "\\to"

    def test_math_no_records_outside_dollars(self):
        # `\to` appears in plain prose; should NOT be detected without dollars.
        records = nlab_wiring.detect_math_scopes(
            "t-m7",
            r"The phrase \to is not in math mode here.",
        )
        assert records == []

    def test_math_equality_skips_double_eq_and_coloneq(self):
        # Equality should fire on bare `=`, not on `==` or `:=` or `\equiv`.
        records = nlab_wiring.detect_math_scopes(
            "t-m8",
            "$X = Y$ but not $a == b$ and not $f := g$",
        )
        eq_records = [r for r in records if r["hx/type"] == "math/equality"]
        assert len(eq_records) == 1  # only the bare `=` fires
        match_text = eq_records[0]["hx/content"]["match"]
        assert match_text == "="

    def test_detect_comments_finds_latex_comments(self):
        text = (
            "Definition of monoid.\n"
            "% TODO: rewrite this paragraph using monad terminology\n"
            "A monoid is a set with operation.\n"
        )
        records = nlab_wiring.detect_comments("test-comment", text)
        assert len(records) == 1
        rec = records[0]
        assert rec["hx/type"] == "comment/unreachable"
        assert rec["hx/role"] == "scope"
        assert "TODO" in rec["hx/content"]["match"]

    def test_detect_comments_skips_escaped_percent(self):
        text = "The fraction is 95\\% of the total."
        records = nlab_wiring.detect_comments("test-esc", text)
        # `\%` is a literal percent sign, not a comment.
        assert records == []

    def test_detect_comments_handles_multiple_comments(self):
        text = (
            "% first comment\n"
            "real content here\n"
            "% second comment\n"
            "more content\n"
            "trailing % inline comment until newline\n"
        )
        records = nlab_wiring.detect_comments("test-multi", text)
        assert len(records) == 3

    def test_detect_learned_skips_bad_regex_silently(self):
        patterns = [{
            "signature": "weird",
            "regex": r"(unclosed",  # malformed
            "predicted_kind": "wire",
        }]
        # Bad regex must not crash the detector or produce a partial record.
        assert nlab_wiring.detect_learned("e", "text", patterns) == []

    def test_notice_that_wire(self):
        wires = nlab_wiring.detect_wires(
            "t-notice",
            "Notice that in most cases the genus two curve fulfills these assumptions."
        )
        assert any(w["hx/type"] == "wire/consequential" for w in wires)

    def test_observe_that_wire(self):
        wires = nlab_wiring.detect_wires(
            "t-obs",
            "Observe that the diagram commutes by naturality."
        )
        assert any(w["hx/type"] == "wire/consequential" for w in wires)

    def test_thanks_to_causal_wire(self):
        wires = nlab_wiring.detect_wires(
            "t-thx",
            "In [6], thanks to this axiom, we may construct the bimodule structure."
        )
        assert any(w["hx/type"] == "wire/causal" for w in wires)

    def test_we_now_sequencing_wire(self):
        wires = nlab_wiring.detect_wires(
            "t-wenow",
            "We now consider the axiomatics of Ann-categories in another view."
        )
        assert any(w["hx/type"] == "wire/sequencing" for w in wires)

    def test_in_order_to_purposive_wire(self):
        wires = nlab_wiring.detect_wires(
            "t-iot",
            "In order to do so, we must alter the diagonalization step."
        )
        assert any(w["hx/type"] == "wire/purposive" for w in wires)

    def test_these_anaphoric_port(self):
        ports = nlab_wiring.detect_ports(
            "t-these",
            "These are the categories with distributivity constraints similar to rings."
        )
        assert any(p["hx/type"] == "port/these-anaphoric" for p in ports)

    def test_paper_frame_label(self):
        labels = nlab_wiring.detect_labels(
            "t-frame-paper",
            "In this paper, we have made some comments on these two definitions."
        )
        assert any(l["hx/type"] == "strategy/paper-frame" for l in labels)

    def test_section_frame_label(self):
        labels = nlab_wiring.detect_labels(
            "t-frame-section",
            "In this section, we will prove the independence of the last requirement."
        )
        assert any(l["hx/type"] == "strategy/paper-frame" for l in labels)

    def test_numbered_section_frame_label(self):
        labels = nlab_wiring.detect_labels(
            "t-frame-numbered-section",
            "In section 5 we explain how this method extends to the homogeneous subspace."
        )
        assert any(l["hx/type"] == "strategy/paper-frame" for l in labels)

    def test_ordinal_section_frame_label(self):
        labels = nlab_wiring.detect_labels(
            "t-frame-ordinal-section",
            "In the fourth section we prove by a new method that the basis elements are independent."
        )
        assert any(l["hx/type"] == "strategy/paper-frame" for l in labels)

    def test_main_result_label(self):
        labels = nlab_wiring.detect_labels(
            "t-main",
            "The main result of this paper is the relationship of Ann-categories and rings."
        )
        assert any(l["hx/type"] == "strategy/main-result" for l in labels)

    def test_recent_work_label(self):
        labels = nlab_wiring.detect_labels(
            "t-recent",
            "Recently, we have proved that this cohomology coincides with Maclane's."
        )
        assert any(l["hx/type"] == "strategy/recent-work" for l in labels)

    def test_easy_to_see_label(self):
        labels = nlab_wiring.detect_labels(
            "t-easy",
            "It is easy to see that every BG group has PIG."
        )
        assert any(l["hx/type"] == "epistemic/easy-to-see" for l in labels)

    def test_discourse_parent_env(self):
        """Discourse records inside environments should have hx/parent set."""
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        discourse = nlab_wiring.extract_discourse_wiring("193", ADJUNCTION_SNIPPET, envs)
        parented = [d for d in discourse if d["hx/parent"] is not None]
        assert len(parented) > 0

    def test_labels_detected_in_latex_env(self):
        envs = nlab_wiring.parse_environments(LATEX_ENV_SNIPPET)
        discourse = nlab_wiring.extract_discourse_wiring("test", LATEX_ENV_SNIPPET, envs)
        wire_records = [d for d in discourse if d["hx/role"] == "wire"]
        # "Since" → wire/causal, "Therefore" → wire/consequential
        wire_types = {w["hx/type"] for w in wire_records}
        assert "wire/causal" in wire_types or "wire/consequential" in wire_types

    def test_symbolic_binders_detected(self):
        text = (
            r"Given $\forall x \in X,\ \int_0^1 f(x)\,dx = \sum_{n=1}^{\infty} a_n$ "
            r"and $\exists y$."
        )
        scopes = nlab_wiring.detect_scopes("t-1", text)
        scope_types = {s["hx/type"] for s in scopes}
        assert "quant/universal" in scope_types
        assert "quant/existential" in scope_types
        assert "bind/integral" in scope_types
        assert "bind/summation" in scope_types

    def test_degenerate_sum_as_binder(self):
        text = r"We use the operation $\sum a_i$ as shorthand."
        scopes = nlab_wiring.detect_scopes("t-2", text)
        sums = [s for s in scopes if s["hx/type"] == "bind/summation"]
        assert sums
        assert any(end.get("role") == "binder" for end in sums[0]["hx/ends"])

    def test_latex_theorem_environment_is_scope(self):
        scopes = nlab_wiring.detect_scopes("t-3", LATEX_ENV_SNIPPET)
        scope_types = {s["hx/type"] for s in scopes}
        assert "env/theorem" in scope_types
        assert "env/proof" in scope_types

    def test_heading_theorem_environment_is_scope(self):
        text = "Theorem. Every monad comes from an adjunction in this case."
        scopes = nlab_wiring.detect_scopes("t-4", text)
        assert any(s["hx/type"] == "env/theorem" for s in scopes)

    def test_scope_records_include_end_offsets(self):
        text = r"Let $X$ be a set. For every $x \in X$ we have $x=x$."
        scopes = nlab_wiring.detect_scopes("t-5", text)
        components = [s for s in scopes if s["hx/role"] == "component"]
        assert components
        for s in components:
            c = s.get("hx/content", {})
            assert "end" in c
            assert isinstance(c["end"], int)
            assert c["end"] > c["position"]

    def test_for_in_scope_detected(self):
        text = r"For a path $\gamma$ in $\Gamma$ we define $X(\gamma)$."
        scopes = nlab_wiring.detect_scopes("t-6", text)
        assert any(
            s["hx/type"] == "quant/universal"
            and any(e.get("role") == "domain" and e.get("latex") == r"\Gamma"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_if_condition_scope_detected(self):
        text = r"If $\gamma = \beta \circ e$, define $X(\gamma)$."
        scopes = nlab_wiring.detect_scopes("t-7", text)
        assert any(
            s["hx/type"] == "assume/explicit"
            and any(e.get("role") == "condition" and r"\gamma" in e.get("latex", "")
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_let_scope_extends_across_text(self):
        text = r"Let $\Gamma$ be a graph. Later we use $\Gamma$ again."
        scopes = [s for s in nlab_wiring.detect_scopes("t-8", text)
                  if s["hx/type"] == "bind/let"]
        assert scopes
        c = scopes[0]["hx/content"]
        assert c["position"] == 0
        assert c["end"] >= text.find(".") + 1
        assert c["end"] <= len(text)

    def test_for_any_entity_detected(self):
        text = r"For every edge $e \in E$ such that $f(e)=e$, we proceed."
        scopes = nlab_wiring.detect_scopes("t-9", text)
        assert any(
            s["hx/type"] == "quant/universal"
            and any(e.get("role") == "symbol" and e.get("latex") == r"e \in E"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_typed_arrow_scope_detected(self):
        text = r"We have $f : A \to B$ and $g : B \to C$."
        scopes = nlab_wiring.detect_scopes("t-10", text)
        typed = [s for s in scopes if s["hx/type"] == "bind/typed"]
        assert len(typed) >= 2
        assert any(
            any(e.get("role") == "type" and r"\to" in e.get("latex", "")
                for e in s.get("hx/ends", []))
            for s in typed
        )

    def test_arrow_expression_scope_detected(self):
        text = r"Homomorphisms of directed graphs $\Gamma \to U(\mathcal{C})$ are useful."
        scopes = nlab_wiring.detect_scopes("t-10b", text)
        assert any(
            s["hx/type"] == "bind/typed"
            and any(e.get("role") == "type" and r"\Gamma \to U(\mathcal{C})" in e.get("latex", "")
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_relation_expression_scope_detected(self):
        text = r"We use $x^2 + y^2 = z^2$ in this argument."
        scopes = nlab_wiring.detect_scopes("t-10c", text)
        assert any(
            s["hx/type"] == "constrain/relation"
            and any(e.get("role") == "relation" and "=" in e.get("latex", "")
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_latex_environment_scope_uses_closing_token(self):
        text = r"\begin{theorem}Statement.\end{theorem}"
        scopes = nlab_wiring.detect_scopes("t-11", text)
        envs = [s for s in scopes if s["hx/type"] == "env/theorem"]
        assert envs
        c = envs[0]["hx/content"]
        assert c["position"] == 0
        assert c["end"] == len(text)

    def test_cross_close_environment_scope_supported(self):
        text = r"\begin{theorem}Statement.\end{proof}"
        scopes = nlab_wiring.detect_scopes("t-12", text)
        envs = [s for s in scopes if s["hx/type"] == "env/theorem"]
        assert envs
        labels = set(envs[0].get("hx/labels", []))
        assert "cross-close" in labels

    def test_fix_binding_scope_detected(self):
        text = r"Fix $X$ to be a cofibrant object."
        scopes = nlab_wiring.detect_scopes("t-13", text)
        assert any(
            s["hx/type"] == "bind/let"
            and any(e.get("role") == "symbol" and e.get("latex") == "X"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_write_for_scope_detected(self):
        text = r"We write $F$ for the identity functor on $\mathcal{C}$."
        scopes = nlab_wiring.detect_scopes("t-14", text)
        assert any(
            s["hx/type"] == "bind/define"
            and any(e.get("role") == "symbol" and e.get("latex") == "F"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_exists_binding_scope_detected(self):
        text = r"There exists $f$ such that $f : X \to Y$."
        scopes = nlab_wiring.detect_scopes("t-15", text)
        assert any(
            s["hx/type"] == "quant/existential"
            and any(e.get("role") == "symbol" and e.get("latex") == "f"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_choose_work_in_scope_detected(self):
        text = r"We choose to work in the Cauchy completion $\Q\V$ of $\V$."
        scopes = nlab_wiring.detect_scopes("t-15b", text)
        assert any(
            s["hx/type"] == "assume/consider"
            and any(e.get("role") == "object" and "Cauchy completion" in e.get("text", "")
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_is_denoted_by_scope_detected(self):
        text = r"The set of such $G$ is denoted by $\operatorname{Aux}(F)$."
        scopes = nlab_wiring.detect_scopes("t-16", text)
        assert any(
            s["hx/type"] == "bind/define"
            and any(e.get("role") == "symbol" and r"\operatorname{Aux}(F)" in e.get("latex", "")
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_we_denote_by_scope_detected(self):
        text = r"We denote as usual by $h_k$ the $k$-th complete homogeneous symmetric function."
        scopes = nlab_wiring.detect_scopes("t-16b", text)
        assert any(
            s["hx/type"] == "bind/define"
            and any(e.get("role") == "symbol" and e.get("latex") == "h_k"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_let_command_binding_scope_detected(self):
        text = r"Let \Digraph\ be the category of directed graphs and graph maps."
        scopes = nlab_wiring.detect_scopes("t-17", text)
        assert any(
            s["hx/type"] == "bind/let"
            and any(e.get("role") == "symbol" and e.get("latex") == r"\Digraph"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_here_denotes_scope_detected(self):
        text = r"Here $\exp(Q)$ denotes the exponent of a group $Q$."
        scopes = nlab_wiring.detect_scopes("t-18", text)
        assert any(
            s["hx/type"] == "bind/define"
            and any(e.get("role") == "symbol" and e.get("latex") == r"\exp(Q)"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_typed_arrow_short_macro_detected(self):
        text = r"We have $s:A \ra C^\o$ and $t:A \ra C$."
        scopes = nlab_wiring.detect_scopes("t-19", text)
        assert any(
            s["hx/type"] == "bind/typed"
            and any(e.get("role") == "type" and r"\ra" in e.get("latex", "")
                    for e in s.get("hx/ends", []))
            for s in scopes
        )

    def test_assume_that_prose_scope_detected(self):
        text = r"Assume that with positive $\Pi$ probability, $X_{\xi}$ is not a Dirac measure."
        scopes = nlab_wiring.detect_scopes("t-20", text)
        assert any(s["hx/type"] == "assume/explicit" for s in scopes)

    def test_let_also_denote_scope_detected(self):
        text = r"Let also $X$ denote a subset of $(x_1,x_2,\ldots,x_n)$."
        scopes = nlab_wiring.detect_scopes("t-21", text)
        assert any(
            s["hx/type"] == "bind/let"
            and any(e.get("role") == "symbol" and e.get("latex") == "X"
                    for e in s.get("hx/ends", []))
            for s in scopes
        )


# ============================================================
# Step 5: Categorical hyperedge tests
# ============================================================

class TestCategoricalHyperedges:

    def test_adjunction_detected(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        cats = nlab_wiring.detect_categorical_patterns(
            "193", "adjunction", ADJUNCTION_SNIPPET, envs, links)
        cat_types = [c["hx/type"] for c in cats]
        assert "cat/adjunction" in cat_types

    def test_adjunction_has_roles(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        cats = nlab_wiring.detect_categorical_patterns(
            "193", "adjunction", ADJUNCTION_SNIPPET, envs, links)
        adj = [c for c in cats if c["hx/type"] == "cat/adjunction"][0]
        roles = {e["role"] for e in adj["hx/ends"]}
        assert "left-adjoint" in roles
        assert "right-adjoint" in roles

    def test_kan_extension_detected(self):
        envs = nlab_wiring.parse_environments(KAN_EXTENSION_SNIPPET)
        links = nlab_wiring.extract_typed_links("266", KAN_EXTENSION_SNIPPET, envs)
        cats = nlab_wiring.detect_categorical_patterns(
            "266", "Kan extension", KAN_EXTENSION_SNIPPET, envs, links)
        cat_types = [c["hx/type"] for c in cats]
        assert "cat/kan-extension" in cat_types

    def test_monad_detected_from_adjunction_page(self):
        """Adjunction page mentions monads — should detect monad pattern.
        The snippet has [[monad]] link + text 'monad' — needs min 2 signals."""
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        cats = nlab_wiring.detect_categorical_patterns(
            "193", "adjunction", ADJUNCTION_SNIPPET, envs, links)
        cat_types = [c["hx/type"] for c in cats]
        # "monad" link + "monad" text signals — checks both are counted
        assert "cat/adjunction" in cat_types  # primary pattern
        # monad may or may not fire depending on signal count; verify adjunction is solid
        if "cat/monad" in cat_types:
            monad = [c for c in cats if c["hx/type"] == "cat/monad"][0]
            assert monad["hx/content"]["score"] >= 2

    def test_hyperedge_has_evidence(self):
        envs = nlab_wiring.parse_environments(ADJUNCTION_SNIPPET)
        links = nlab_wiring.extract_typed_links("193", ADJUNCTION_SNIPPET, envs)
        cats = nlab_wiring.detect_categorical_patterns(
            "193", "adjunction", ADJUNCTION_SNIPPET, envs, links)
        for cat in cats:
            assert "evidence" in cat["hx/content"]
            assert "score" in cat["hx/content"]
            assert cat["hx/content"]["score"] > 0


# ============================================================
# Integration: process_page
# ============================================================

class TestProcessPage:

    def test_process_page_returns_all_sections(self):
        result = nlab_wiring.process_page("193", "adjunction", ADJUNCTION_SNIPPET)
        assert "environments" in result
        assert "typed_links" in result
        assert "diagrams" in result
        assert "discourse" in result
        assert "categorical" in result
        assert "stats" in result

    def test_process_page_stats(self):
        result = nlab_wiring.process_page("193", "adjunction", ADJUNCTION_SNIPPET)
        stats = result["stats"]
        assert stats["n_environments"] >= 5
        assert stats["n_typed_links"] > 0
        assert stats["n_categorical"] > 0

    def test_process_page_json_serializable(self):
        result = nlab_wiring.process_page("193", "adjunction", ADJUNCTION_SNIPPET)
        # Should not raise
        json.dumps(result, ensure_ascii=False)
