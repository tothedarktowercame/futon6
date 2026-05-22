"""Shared symbol-grounding orchestration.

Hosts the pieces that both the QC viewer and the superpod-job Stage 5
need: a kernel-phrase lookup constructor, a kernel scanner constructor,
the math-atom walker, and `detect_grounded_symbols` — the per-paper
pipeline that runs all default strategies, walks atoms, and emits
`math/grounded-symbol` scope records.

Lives next to `structure_seed.py` because it composes the same NER
kernel + scope infrastructure those callers already share.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from . import math_ast as _ma
from . import structure_seed as _ss
from . import symbol_grounding as _sg


def make_kernel_phrase_lookup(singles: dict, multi_index: dict) -> Callable[[str], str | None]:
    """Build a phrase→canon lookup for grounding strategies.

    `singles` maps single-word `term_lower → (term_orig, canon)`.
    `multi_index` maps `first_word → list of (term_lower, term_orig, canon)`.
    Returns a function that, given a phrase like "abelian group", returns
    the kernel's canon name, or None if the phrase isn't known.
    """
    def lookup(phrase: str) -> str | None:
        phrase = (phrase or "").lower().strip()
        if not phrase:
            return None
        if phrase in singles:
            return singles[phrase][1]
        first_word = phrase.split()[0] if phrase else ""
        if first_word in multi_index:
            for term_lower, _orig, canon in multi_index[first_word]:
                if term_lower == phrase:
                    return canon
        return None
    return lookup


def make_kernel_scan(
    singles: dict,
    multi_index: dict,
    spot_terms_fn,
) -> Callable[[str], list[tuple[int, int, str, str | None]]]:
    """Build a kernel scanner: text chunk → list of (start, end, phrase, canon).

    Wraps `structure_seed.find_kernel_term_positions` so the kernel-ambient
    strategy sees the same NER hits the overlay markup does.
    """
    def scan(chunk: str):
        return _ss.find_kernel_term_positions(
            chunk, spot_terms_fn, singles, multi_index,
        )
    return scan


def walk_math_atoms(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield `(atom_text, abs_start, abs_end)` for each ground-able atom.

    Atoms are:
      (a) single alphabetic characters inside `chars` nodes within math
          envelopes (so juxtapositions like `XY` become two candidates),
      (b) full macro-token texts like `\\mathcal{C}` (so a Let-binding
          that captured the same literal matches via exact string).
    """
    for env_start, env_end, int_start, int_end, _kind in _ma.find_math_envelopes(text):
        interior = text[int_start:int_end]
        nodes = _ma.parse_math(interior, base_offset=int_start)
        yield from _walk_atoms(nodes)


def _walk_atoms(nodes):
    for node in nodes:
        if node.kind == "chars":
            for i, ch in enumerate(node.text):
                if ch.isalpha():
                    yield (ch, node.start + i, node.start + i + 1)
        elif node.kind == "macro":
            yield (node.text, node.start, node.end)
        for arg in node.args:
            yield from _walk_atoms(arg["nodes"])


def load_learned_vocab(path) -> list[dict]:
    """Load a `learned-newcommand-vocab.json` (the file Stage 5 emits).

    Returns the `common` slot — a list of {symbol, body, canon,
    papers, support} dicts. If the file is missing or malformed,
    returns an empty list so callers can pass the result unconditionally.
    """
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(data.get("common", []))


def detect_grounded_symbols(
    entity_id: str,
    text: str,
    singles: dict,
    multi_index: dict,
    spot_terms_fn,
    *,
    learned_vocab: list[dict] | None = None,
) -> tuple[list[dict], _sg.SymbolEnvironment, dict]:
    """Run all default strategies on `text`; return (records, env, summary).

    Each emitted record is a `math/grounded-symbol` scope located at an
    atom matched by the SymbolEnvironment. `summary` includes per-
    strategy emit / active counts and the meta-learning metrics
    (emit / defeat / corroboration / solo).

    The strategy-emit gate: NewcommandStrategy always passes (its body-
    fallback canon is informative). Prose strategies only pass when a
    kernel canon was found — without one the regex's phrasal capture is
    too noisy.
    """
    kernel_lookup = make_kernel_phrase_lookup(singles, multi_index)
    kernel_scan = make_kernel_scan(singles, multi_index, spot_terms_fn)
    ctx = _sg.StrategyContext(
        paper_id=entity_id,
        paper_text=text,
        kernel_lookup=kernel_lookup,
        kernel_scan=kernel_scan,
    )
    env = _sg.run_strategies(ctx, _sg.default_strategies(learned_vocab=learned_vocab))

    records = []
    rec_idx = 0
    grounded_atom_count = 0
    for atom_text, start, end in walk_math_atoms(text):
        binding = env.lookup(atom_text, start)
        if binding is None:
            continue
        if binding.strategy != "newcommand" and not binding.canon:
            continue
        if not binding.canon and not binding.type_phrase:
            continue
        grounded_atom_count += 1
        canon_or_fallback = binding.canon or binding.type_phrase[:24]
        role = _ma.classify_atom_role(atom_text)
        records.append({
            "hx/id": f"{entity_id}:grounded-{rec_idx:05d}",
            "hx/role": "scope",
            "hx/type": "math/grounded-symbol",
            "hx/parent": None,
            "hx/content": {
                "match": atom_text,
                "position": start,
                "end": end,
                "canon": binding.canon,
                "type_phrase": binding.type_phrase,
                "strategy": binding.strategy,
                "syntax_role": role,
            },
            "hx/labels": [
                "scope", "math", "grounded",
                f"strategy-{binding.strategy}",
                f"canon-{canon_or_fallback}",
            ],
        })
        rec_idx += 1

    strategy_emit_counts: dict[str, int] = {}
    for b in env.all_bindings:
        strategy_emit_counts[b.strategy] = strategy_emit_counts.get(b.strategy, 0) + 1
    strategy_active_counts: dict[str, int] = {}
    for b in env.all_active():
        strategy_active_counts[b.strategy] = strategy_active_counts.get(b.strategy, 0) + 1

    strategy_metrics = _sg.compute_strategy_metrics(env)

    summary = {
        "total_bindings_emitted": len(env.all_bindings),
        "active_bindings": len(env.all_active()),
        "grounded_atom_count": grounded_atom_count,
        "strategy_emit_counts": dict(sorted(strategy_emit_counts.items())),
        "strategy_active_counts": dict(sorted(strategy_active_counts.items())),
        "strategy_metrics": strategy_metrics,
    }
    return records, env, summary
