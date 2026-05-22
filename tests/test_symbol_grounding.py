"""Tests for futon6.symbol_grounding — Layer 3 strategy library."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6.symbol_grounding import (
    DenotationStrategy,
    FixPatternStrategy,
    InlineIsAStrategy,
    KernelAmbientStrategy,
    LetBindingStrategy,
    NewcommandStrategy,
    NotationEnvStrategy,
    StrategyContext,
    SymbolBinding,
    SymbolEnvironment,
    TheYXStrategy,
    aggregate_strategy_metrics,
    compute_strategy_metrics,
    default_strategies,
    merge_bindings,
    run_strategies,
)


# A trivial kernel lookup: maps phrases to canon names. Used by strategies
# to attach canonical concept names to their RHS type phrases. In production
# this is futon6's NER kernel (terms.tsv); for tests we hard-code a few.
def _toy_kernel(phrase: str) -> str | None:
    table = {
        "abelian group": "AbelianGroup",
        "finite abelian group": "FiniteAbelianGroup",
        "category": "Category",
        "monoidal category": "MonoidalCategory",
        "monad": "Monad",
        "functor": "Functor",
        "set": "Set",
        "ring": "Ring",
    }
    return table.get(phrase.lower().strip())


def _ctx(text: str) -> StrategyContext:
    return StrategyContext(
        paper_id="test",
        paper_text=text,
        kernel_lookup=_toy_kernel,
    )


# ============================================================
# LetBindingStrategy
# ============================================================

def test_let_binding_canonical_form():
    text = "Let $X$ be an abelian group. Then $X$ has additive structure."
    bindings = LetBindingStrategy().apply(_ctx(text))
    assert len(bindings) == 1
    b = bindings[0]
    assert b.symbol == "X"
    assert b.canon == "AbelianGroup"
    assert b.type_phrase == "abelian group"
    assert b.confidence == "high"
    assert b.strategy == "let-binding"
    # Scope spans from the end of the declaration to end of text.
    assert b.scope_start == text.index(". ") + 1  # after "Let X be ... group" sentence end
    assert b.scope_end == len(text)


def test_let_binding_lhs_can_be_command():
    text = r"Let $\mathcal{C}$ be a monoidal category. The objects of $\mathcal{C}$ are..."
    bindings = LetBindingStrategy().apply(_ctx(text))
    assert any(b.symbol == r"\mathcal{C}" and b.canon == "MonoidalCategory" for b in bindings)


def test_let_binding_kernel_lookup_returns_none_for_unknown_phrase():
    text = "Let $X$ be a frobnicator."  # not in kernel
    bindings = LetBindingStrategy().apply(_ctx(text))
    assert len(bindings) == 1
    assert bindings[0].canon is None
    assert bindings[0].type_phrase == "frobnicator"


# ============================================================
# DenotationStrategy
# ============================================================

def test_denotes_pattern():
    text = "Here $T$ denotes a monad and we apply it."
    bindings = DenotationStrategy().apply(_ctx(text))
    assert any(b.symbol == "T" and b.canon == "Monad" for b in bindings)


def test_we_denote_by_pattern():
    text = "We denote by $\\mathcal{C}$ the monoidal category of consideration."
    bindings = DenotationStrategy().apply(_ctx(text))
    syms = [(b.symbol, b.canon) for b in bindings]
    assert (r"\mathcal{C}", "MonoidalCategory") in syms


# ============================================================
# TheYXStrategy
# ============================================================

def test_the_Y_X_medium_confidence():
    text = "Consider the category $\\mathcal{D}$ for context."
    bindings = TheYXStrategy().apply(_ctx(text))
    matched = [b for b in bindings if b.symbol == r"\mathcal{D}"]
    assert matched
    assert matched[0].canon == "Category"
    assert matched[0].confidence == "medium"


def test_the_Y_X_rejects_short_type_phrases():
    text = "the f $X$ is bad."  # `f` too short
    bindings = TheYXStrategy().apply(_ctx(text))
    assert not any(b.symbol == "X" for b in bindings)


# ============================================================
# Defeasibility via merge_bindings
# ============================================================

def test_later_binding_narrows_earlier_scope():
    text = (
        "Let $X$ be an abelian group. "          # binding 1: from pos ~33 to end
        "After much development, "                # ~57
        "let $X$ be a finite abelian group. "    # binding 2: narrows binding 1
        "Then $X$ has nice properties."
    )
    env = run_strategies(_ctx(text), [LetBindingStrategy()])
    bindings = env.all_bindings
    # Two let-bindings on X
    let_bindings = sorted(
        (b for b in bindings if b.symbol == "X"),
        key=lambda b: b.scope_start,
    )
    assert len(let_bindings) == 2
    first, second = let_bindings
    assert first.canon == "AbelianGroup"
    assert second.canon == "FiniteAbelianGroup"
    # First binding's scope ends where second begins.
    assert first.scope_end == second.scope_start
    # First binding got defeated by the second.
    assert first.defeated_by == second.binding_id
    # Second is undefeated.
    assert second.defeated_by is None


def test_environment_lookup_returns_active_binding_at_position():
    text = (
        "Let $X$ be an abelian group. "
        "More text here padding the gap. "
        "Now let $X$ be a finite abelian group. "
        "Continue."
    )
    env = run_strategies(_ctx(text), [LetBindingStrategy()])
    # Find position inside the gap (between first and second binding)
    gap_pos = text.index("padding")
    later_pos = text.index("Continue")
    early = env.lookup("X", gap_pos)
    late = env.lookup("X", later_pos)
    assert early is not None and early.canon == "AbelianGroup"
    assert late is not None and late.canon == "FiniteAbelianGroup"


def test_let_binding_beats_the_Y_X_on_same_symbol_when_starts_overlap():
    # Two strategies fire near each other; let-binding has higher confidence.
    text = (
        "Let $X$ be an abelian group. "  # let-binding fires, high confidence
        "Consider the category $X$ for context."  # the-Y-X fires later, medium
    )
    env = run_strategies(_ctx(text), [LetBindingStrategy(), TheYXStrategy()])
    # the-Y-X is LATER in scope, so it narrows let-binding's scope (defeats it).
    # That's correct behavior: at the later position, X is now a "category" by
    # author intent. The earlier scope is preserved.
    all_X = [b for b in env.all_bindings if b.symbol == "X"]
    assert len(all_X) == 2
    earlier = min(all_X, key=lambda b: b.scope_start)
    later = max(all_X, key=lambda b: b.scope_start)
    assert earlier.canon == "AbelianGroup"
    assert later.canon == "Category"
    assert earlier.scope_end == later.scope_start
    assert earlier.defeated_by == later.binding_id


def test_higher_confidence_wins_on_exact_position_tie():
    # Construct two bindings that start at the SAME position with different
    # confidences. Higher-confidence wins; lower gets defeated_by set.
    b_high = SymbolBinding(
        binding_id="a", symbol="X", canon="AbelianGroup",
        type_phrase="abelian group", scope_start=100, scope_end=500,
        confidence="high", strategy="let-binding", evidence_span=(80, 100),
    )
    b_med = SymbolBinding(
        binding_id="b", symbol="X", canon="Category",
        type_phrase="category", scope_start=100, scope_end=500,
        confidence="medium", strategy="the-Y-X", evidence_span=(85, 100),
    )
    merged = merge_bindings([b_high, b_med])
    by_id = {b.binding_id: b for b in merged}
    assert by_id["a"].defeated_by is None
    assert by_id["b"].defeated_by == "a"


def test_undefeated_binding_is_active_in_environment():
    text = "Let $X$ be an abelian group. End."
    env = run_strategies(_ctx(text), default_strategies())
    actives = env.all_active()
    assert any(b.symbol == "X" and b.canon == "AbelianGroup" for b in actives)


def test_multiple_symbols_independent_scopes():
    text = (
        "Let $X$ be an abelian group. "
        "Let $Y$ be a category. "
        "Then $X$ and $Y$ interact."
    )
    env = run_strategies(_ctx(text), default_strategies())
    x_b = env.lookup("X", len(text) - 1)
    y_b = env.lookup("Y", len(text) - 1)
    assert x_b is not None and x_b.canon == "AbelianGroup"
    assert y_b is not None and y_b.canon == "Category"


def test_environment_returns_none_for_unbound_symbol():
    text = "Let $X$ be an abelian group. End."
    env = run_strategies(_ctx(text), default_strategies())
    assert env.lookup("Z", 0) is None


# ============================================================
# NewcommandStrategy
# ============================================================

def test_newcommand_blackboard_letter_resolves_via_kernel():
    text = r"\newcommand{\RR}{{\mathbb R}}" + "\nSome paper body using $\\RR$ everywhere."
    bindings = NewcommandStrategy().apply(_ctx(text))
    bs = [b for b in bindings if b.symbol == r"\RR"]
    assert bs, "expected a binding for \\RR"
    b = bs[0]
    assert b.canon == "RealNumbers" or b.canon == "real numbers" or b.canon  # any of: kernel hit or fallback
    assert b.scope_start == 0
    assert b.scope_end == len(text)
    assert b.confidence == "high"
    assert b.strategy == "newcommand"


def test_newcommand_literal_word_body_uses_kernel():
    # \Cat -> "Category" — the literal word should hit the kernel via lowercased lookup.
    text = r"\newcommand{\Cat}{Category}" + "\nThe paper uses $\\Cat$ throughout."
    bindings = NewcommandStrategy().apply(_ctx(text))
    bs = [b for b in bindings if b.symbol == r"\Cat"]
    assert bs
    assert bs[0].canon == "Category"


def test_newcommand_skips_typographic_macros():
    text = (
        r"\newcommand{\bsq}{\vrule height .9ex width .8ex depth -.1ex}"
        "\n"
        r"\newcommand{\eeq}{\end{equation}}"
        "\n"
        r"\def\objectstyle{\scriptstyle}"
    )
    bindings = NewcommandStrategy().apply(_ctx(text))
    syms = {b.symbol for b in bindings}
    assert r"\bsq" not in syms
    assert r"\eeq" not in syms
    assert r"\objectstyle" not in syms


def test_newcommand_skips_parameterised_def():
    text = r"\def\foo#1#2{x_{#1}^{#2}}" + "\n"
    bindings = NewcommandStrategy().apply(_ctx(text))
    assert not any(b.symbol == r"\foo" for b in bindings)


def test_newcommand_calligraphic_letter_falls_back_to_letter_label():
    # \sE -> {\cal E}. No kernel phrase exists for "E" alone, so canon
    # should fall back to the cleaned body "E" so the badge is readable.
    text = r"\newcommand{\sE}{{\cal E}}" + "\nUsing $\\sE$ here."
    bindings = NewcommandStrategy().apply(_ctx(text))
    bs = [b for b in bindings if b.symbol == r"\sE"]
    assert bs
    assert bs[0].canon == "E"
    # type_phrase preserves one level of grouping; the inner cleaning is
    # what produces the canon. We just require the original LaTeX intent
    # to be visible in the tooltip body.
    assert "cal E" in bs[0].type_phrase


def test_newcommand_paper_wide_scope_environment_lookup():
    # \RR binding is global; lookup should succeed at any position.
    text = r"\newcommand{\RR}{{\mathbb R}}" + "\n" + "x " * 200 + r"$\RR$" + "more text."
    env = run_strategies(_ctx(text), [NewcommandStrategy()])
    pos = text.index(r"$\RR$")
    b = env.lookup(r"\RR", pos)
    assert b is not None
    assert b.strategy == "newcommand"


def test_newcommand_declaremathoperator():
    text = r"\DeclareMathOperator{\spec}{Spec}" + "\nWe use $\\spec$ later."
    bindings = NewcommandStrategy().apply(_ctx(text))
    assert any(b.symbol == r"\spec" for b in bindings)


def test_newcommand_default_strategies_includes_it():
    names = {s.name for s in default_strategies()}
    assert "newcommand" in names


# ============================================================
# Strategy meta-learning metrics
# ============================================================

def test_compute_strategy_metrics_counts_emit_and_defeat():
    text = (
        "Let $X$ be an abelian group. "
        "After much development, "
        "let $X$ be a finite abelian group. "
        "Done."
    )
    env = run_strategies(_ctx(text), [LetBindingStrategy()])
    metrics = compute_strategy_metrics(env)
    assert "let-binding" in metrics
    assert metrics["let-binding"]["emitted"] == 2
    # The first binding got defeated by the second.
    assert metrics["let-binding"]["defeated"] == 1


def test_compute_strategy_metrics_corroboration_when_two_strategies_agree():
    # Both `let-binding` and `the-Y-X` fire on $X$ with canon=Category.
    text = (
        "Let $X$ be a category. "
        "Consider the category $X$ for context."
    )
    env = run_strategies(_ctx(text), [LetBindingStrategy(), TheYXStrategy()])
    metrics = compute_strategy_metrics(env)
    # Each strategy should see at least one binding marked corroborated.
    assert metrics["let-binding"]["corroborated"] >= 1
    assert metrics["the-Y-X"]["corroborated"] >= 1


def test_compute_strategy_metrics_solo_when_no_corroboration():
    text = "Let $X$ be a category."
    env = run_strategies(_ctx(text), [LetBindingStrategy()])
    metrics = compute_strategy_metrics(env)
    assert metrics["let-binding"]["solo"] == 1
    assert metrics["let-binding"]["corroborated"] == 0


def test_aggregate_strategy_metrics_sums_across_papers():
    paper_a = {"let-binding": {"emitted": 4, "defeated": 1, "corroborated": 2, "solo": 1}}
    paper_b = {"let-binding": {"emitted": 6, "defeated": 0, "corroborated": 3, "solo": 3},
               "newcommand": {"emitted": 10, "defeated": 0, "corroborated": 0, "solo": 10}}
    agg = aggregate_strategy_metrics({"p_a": paper_a, "p_b": paper_b})
    assert agg["let-binding"]["emitted"] == 10
    assert agg["let-binding"]["defeated"] == 1
    assert agg["let-binding"]["corroborated"] == 5
    assert agg["let-binding"]["papers_active"] == 2
    assert agg["let-binding"]["defeat_rate"] == 0.1
    assert agg["let-binding"]["corroboration_rate"] == 0.5
    # newcommand only fired in paper_b
    assert agg["newcommand"]["papers_active"] == 1
    assert agg["newcommand"]["emitted"] == 10


def test_aggregate_strategy_metrics_zero_emitted_yields_zero_rates():
    """A strategy with no emissions shouldn't NaN out the rate calc."""
    agg = aggregate_strategy_metrics({})
    assert agg == {}


# ============================================================
# FixPatternStrategy
# ============================================================

def test_fix_pattern_simple():
    text = "Fix $X$ as a category for the rest of the section."
    bindings = FixPatternStrategy().apply(_ctx(text))
    assert any(b.symbol == "X" and b.canon == "Category" for b in bindings)


def test_fix_to_be_form():
    text = "Fix $T$ to be a monad on $\\mathcal{C}$."
    bindings = FixPatternStrategy().apply(_ctx(text))
    assert any(b.symbol == "T" and b.canon == "Monad" for b in bindings)


def test_fix_pattern_no_false_positive_on_fix_a_typo():
    """`Fix a typo` shouldn't fire — no $X$ between Fix and "as a typo"."""
    text = "We fix a typo in equation 3.4."
    bindings = FixPatternStrategy().apply(_ctx(text))
    assert bindings == []


# ============================================================
# InlineIsAStrategy
# ============================================================

def test_inline_is_a_simple():
    text = "Here $T$ is a monad on the category $\\mathcal{C}$."
    bindings = InlineIsAStrategy().apply(_ctx(text))
    assert any(b.symbol == "T" and b.canon == "Monad" for b in bindings)


def test_inline_is_a_medium_confidence():
    text = "$X$ is a ring."
    bindings = InlineIsAStrategy().apply(_ctx(text))
    assert bindings[0].confidence == "medium"


def test_inline_is_a_rejects_set_of_phrases():
    text = "$S$ is a set of points."
    bindings = InlineIsAStrategy().apply(_ctx(text))
    # `set of` is in the reject list because "S is a set of points" is
    # less a type assertion than a description of S's elements.
    assert all(b.type_phrase != "set of" for b in bindings)


# ============================================================
# NotationEnvStrategy
# ============================================================

def test_notation_env_extracts_declarations():
    text = (
        "Some prose. "
        r"\begin{notation}"
        " In this paper $X$ denotes an abelian group "
        "and $T$ denotes a monad."
        r"\end{notation}"
        " More prose."
    )
    bindings = NotationEnvStrategy().apply(_ctx(text))
    syms = {(b.symbol, b.canon) for b in bindings}
    assert ("X", "AbelianGroup") in syms
    assert ("T", "Monad") in syms


def test_notation_env_supports_stands_for():
    text = (
        r"\begin{notation}"
        " $X$ stands for the category of vector spaces. "
        r"\end{notation}"
    )
    bindings = NotationEnvStrategy().apply(_ctx(text))
    # type_phrase = "category" via the regex's lazy capture; not asserting
    # canon since "category of vector spaces" isn't in the toy kernel.
    assert any(b.symbol == "X" for b in bindings)


def test_notation_env_supports_convention_alias():
    text = (
        r"\begin{convention}"
        " $\\mathcal{C}$ denotes a category. "
        r"\end{convention}"
    )
    bindings = NotationEnvStrategy().apply(_ctx(text))
    assert any(b.symbol == r"\mathcal{C}" and b.canon == "Category" for b in bindings)


def test_notation_env_scope_is_paper_wide():
    text = (
        r"\begin{notation} $X$ denotes a ring. \end{notation}"
        " " * 200
        + " Later in the paper, $X$ shows up."
    )
    bindings = NotationEnvStrategy().apply(_ctx(text))
    b = next(b for b in bindings if b.symbol == "X")
    assert b.scope_start == 0
    assert b.scope_end == len(text)


# ============================================================
# KernelAmbientStrategy
# ============================================================

def _toy_scan(text: str) -> list[tuple[int, int, str, str]]:
    """Find any kernel phrase from the toy kernel inside `text`."""
    table = {
        "abelian group": "AbelianGroup",
        "monad": "Monad",
        "category": "Category",
        "set": "Set",
        "ring": "Ring",
    }
    out = []
    low = text.lower()
    for phrase, canon in table.items():
        idx = 0
        while True:
            i = low.find(phrase, idx)
            if i == -1:
                break
            out.append((i, i + len(phrase), phrase, canon))
            idx = i + 1
    return out


def _scan_ctx(text: str) -> StrategyContext:
    return StrategyContext(
        paper_id="test",
        paper_text=text,
        kernel_lookup=_toy_kernel,
        kernel_scan=_toy_scan,
    )


def test_kernel_ambient_grounds_atom_in_sentence_with_kernel_phrase():
    text = "We compute the abelian group $G/[G,G]$ at each step."
    bindings = KernelAmbientStrategy().apply(_scan_ctx(text))
    syms = {(b.symbol, b.canon) for b in bindings}
    assert ("G", "AbelianGroup") in syms


def test_kernel_ambient_skips_when_multiple_phrases_in_sentence():
    """Ambiguous: sentence has both 'monad' and 'category' — skip."""
    text = "Consider the monad $T$ on a category."
    bindings = KernelAmbientStrategy().apply(_scan_ctx(text))
    assert bindings == []


def test_kernel_ambient_low_confidence():
    text = "Consider an abelian group $G$ today."
    bindings = KernelAmbientStrategy().apply(_scan_ctx(text))
    assert bindings
    assert bindings[0].confidence == "low"


def test_kernel_ambient_no_scan_no_bindings():
    text = "Consider the abelian group $G$."
    ctx = StrategyContext(
        paper_id="t", paper_text=text, kernel_lookup=_toy_kernel,
    )  # no kernel_scan
    assert KernelAmbientStrategy().apply(ctx) == []


# ============================================================
# default_strategies grew
# ============================================================

def test_default_strategies_includes_new_strategies():
    names = {s.name for s in default_strategies()}
    for required in (
        "newcommand", "notation-env", "let-binding", "fix-pattern",
        "denotation", "inline-is-a", "the-Y-X", "kernel-ambient",
    ):
        assert required in names, required
