"""Tests for futon6.symbol_grounding — Layer 3 strategy library."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6.symbol_grounding import (
    DenotationStrategy,
    LetBindingStrategy,
    StrategyContext,
    SymbolBinding,
    SymbolEnvironment,
    TheYXStrategy,
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
