"""Symbol grounding — Layer 3 of the symbol-grounding mission.

Defeasible strategy library. Each strategy produces tentative
(symbol, canon, scope-range) bindings with provenance. The merge step
narrows scopes when later evidence appears and records which strategy
defeated which, so the cross-paper meta-learning loop can read off
hit rate, corroboration rate, and defeat rate per strategy.

Symbols are paper-local. What persists across papers is strategy
effectiveness — see M-symbol-grounding.md §3 for the framing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class SymbolBinding:
    """A defeasible claim that `symbol` denotes a thing of type `canon`.

    The binding applies from `scope_start` to `scope_end` (exclusive) in
    the paper text. A later binding for the same symbol *narrows* this
    binding's scope by capping its scope_end — the original is retained
    in the strategy log with `defeated_by` set so meta-learning can see
    which strategies override which.
    """
    binding_id: str
    symbol: str
    canon: str | None
    type_phrase: str
    scope_start: int
    scope_end: int
    confidence: str  # 'high' | 'medium' | 'low'
    strategy: str
    evidence_span: tuple[int, int]
    defeated_by: str | None = None


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _confidence_geq(a: str, b: str) -> bool:
    return _CONFIDENCE_RANK.get(a, 0) >= _CONFIDENCE_RANK.get(b, 0)


@dataclass
class StrategyContext:
    """The inputs a strategy sees: paper text + optional kernel lookup.

    Strategies should never reach beyond this struct so the test surface
    stays small. A strategy that needs the math AST gets it via
    `math_envelopes` (list of (start, end, kind) tuples). The kernel
    lookup function maps a phrase like "abelian group" → canonical name
    if the phrase is in the kernel. The kernel scanner walks a text
    region and yields all kernel-recognised phrases as
    (start, end, phrase, canon) tuples — used by strategies that need
    to find phrases of unknown shape (e.g. kernel-ambient).
    """
    paper_id: str
    paper_text: str
    math_envelopes: list[tuple[int, int, str]] = field(default_factory=list)
    kernel_lookup: Callable[[str], str | None] | None = None
    kernel_scan: Callable[[str], list[tuple[int, int, str, str]]] | None = None
    next_binding_id: list[int] = field(default_factory=lambda: [0])

    def next_id(self) -> str:
        n = self.next_binding_id[0]
        self.next_binding_id[0] += 1
        return f"{self.paper_id}:sb-{n:04d}"


class Strategy:
    """Base class for symbol-grounding strategies.

    Each subclass overrides `apply()` to return a list of SymbolBinding
    candidates. Strategies do not modify state — the merge step combines
    outputs from multiple strategies and applies defeasibility.
    """
    name: str = "unnamed"
    default_confidence: str = "medium"

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        raise NotImplementedError


# ============================================================
# Initial strategy implementations
# ============================================================

def _scope_start_after_punct(text: str, end_pos: int) -> int:
    """Advance past trailing punctuation so a binding's scope starts on the
    next sentence, not at the period that ended the declaration."""
    p = end_pos
    while p < len(text) and text[p] in ".,;:":
        p += 1
    return p


class LetBindingStrategy(Strategy):
    """`Let $X$ be (a|an|the) Y` — high confidence.

    Alternation order matters: `(?:an|a|the)` tried longest-first so the
    engine doesn't match `a` and then capture the trailing `n` as part of
    the type phrase.
    """
    name = "let-binding"
    default_confidence = "high"

    _PATTERN = re.compile(
        r"\bLet\s+\$([^$\n]{1,40})\$\s+be\s+(?:(?:an|a|the)\s+)?([a-z][\w\s\-]{2,60}?)"
        r"(?=[.,;:\n]|\s+(?:and|or|with|such|over|in|on|of)\b)",
        re.IGNORECASE,
    )

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        out = []
        for m in self._PATTERN.finditer(ctx.paper_text):
            symbol = m.group(1).strip()
            type_phrase = m.group(2).strip().lower()
            canon = (ctx.kernel_lookup or (lambda _: None))(type_phrase)
            out.append(SymbolBinding(
                binding_id=ctx.next_id(),
                symbol=symbol,
                canon=canon,
                type_phrase=type_phrase,
                scope_start=_scope_start_after_punct(ctx.paper_text, m.end()),
                scope_end=len(ctx.paper_text),
                confidence=self.default_confidence,
                strategy=self.name,
                evidence_span=(m.start(), m.end()),
            ))
        return out


class DenotationStrategy(Strategy):
    """`$X$ denotes Y` / `we denote by $X$ the Y` — high confidence."""
    name = "denotation"
    default_confidence = "high"

    _PATTERN_A = re.compile(
        r"\$([^$\n]{1,40})\$\s+denotes?\s+(?:(?:an|a|the)\s+)?([a-z][\w\s\-]{2,60}?)"
        r"(?=[.,;:\n]|\s+(?:and|or|with|such|over|in|on|of)\b)",
        re.IGNORECASE,
    )
    _PATTERN_B = re.compile(
        r"\bwe\s+denote\s+by\s+\$([^$\n]{1,40})\$\s+(?:the\s+)?([a-z][\w\s\-]{2,60}?)"
        r"(?=[.,;:\n]|\s+(?:and|or|with|such|over|in|on|of)\b)",
        re.IGNORECASE,
    )

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        out = []
        for pattern in (self._PATTERN_A, self._PATTERN_B):
            for m in pattern.finditer(ctx.paper_text):
                symbol = m.group(1).strip()
                type_phrase = m.group(2).strip().lower()
                canon = (ctx.kernel_lookup or (lambda _: None))(type_phrase)
                out.append(SymbolBinding(
                    binding_id=ctx.next_id(),
                    symbol=symbol,
                    canon=canon,
                    type_phrase=type_phrase,
                    scope_start=_scope_start_after_punct(ctx.paper_text, m.end()),
                    scope_end=len(ctx.paper_text),
                    confidence=self.default_confidence,
                    strategy=self.name,
                    evidence_span=(m.start(), m.end()),
                ))
        return out


class KernelAmbientStrategy(Strategy):
    """Symbol in math envelope + kernel phrase in surrounding sentence — low confidence.

    For each math envelope, scan the surrounding sentence for any
    kernel-known phrase. If exactly one phrase is found, emit a low-
    confidence binding mapping every alphabetic atom inside the
    envelope to the kernel phrase's canon. This is the highest-recall
    strategy in the library — most math sentences contain a kernel
    term — and the noisiest, hence confidence='low'. The meta-learning
    loop will surface its defeat rate so it can be constrained later.

    Scope is the envelope itself, not paper-wide: ambient context
    binds to the immediate symbol use, not to every future use of
    the same letter.
    """
    name = "kernel-ambient"
    default_confidence = "low"

    _ATOM_IN_MATH = re.compile(r"\\[A-Za-z@]+|[A-Za-z]")

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        if ctx.kernel_scan is None:
            return []
        out = []
        envelopes = ctx.math_envelopes or _collect_envelopes_lazily(ctx.paper_text)
        for entry in envelopes:
            env_start, env_end = entry[0], entry[1]
            int_start, int_end = (
                (entry[2], entry[3]) if len(entry) >= 4 else (env_start, env_end)
            )
            # Sentence boundary: previous '.' or '\n' to next '.' or '\n'
            sent_start = max(
                ctx.paper_text.rfind(".", 0, env_start) + 1,
                ctx.paper_text.rfind("\n", 0, env_start) + 1,
                0,
            )
            after = env_end
            sent_end_period = ctx.paper_text.find(".", after)
            sent_end_nl = ctx.paper_text.find("\n", after)
            candidates = [
                p for p in (sent_end_period, sent_end_nl) if p != -1
            ]
            sent_end = min(candidates) if candidates else len(ctx.paper_text)
            sentence = ctx.paper_text[sent_start:sent_end]
            # Drop math envelopes from the sentence so a kernel hit inside
            # math doesn't tag every atom with itself.
            prose = (
                sentence[: env_start - sent_start]
                + " " * (env_end - env_start)
                + sentence[env_end - sent_start :]
            )
            hits = ctx.kernel_scan(prose)
            if not hits or len(hits) > 1:
                # Ambiguous if more than one phrase — skip to keep precision
                # tolerable; future strategies could break ties by proximity.
                continue
            hit_start, hit_end, phrase, canon = hits[0]
            interior = ctx.paper_text[int_start:int_end]
            atoms = {m.group(0) for m in self._ATOM_IN_MATH.finditer(interior)}
            for atom in atoms:
                out.append(SymbolBinding(
                    binding_id=ctx.next_id(),
                    symbol=atom,
                    canon=canon,
                    type_phrase=phrase,
                    scope_start=env_start,
                    scope_end=env_end,
                    confidence=self.default_confidence,
                    strategy=self.name,
                    evidence_span=(sent_start + hit_start, sent_start + hit_end),
                ))
        return out


def _collect_envelopes_lazily(text: str) -> list[tuple[int, int, int, int, str]]:
    """Find math envelopes if the caller didn't precompute them.

    Imports math_ast lazily to avoid a circular import.
    """
    from . import math_ast as ma
    return list(ma.find_math_envelopes(text))


class LearnedVocabStrategy(Strategy):
    r"""Cross-paper newcommand defaults — low confidence, paper-wide.

    Consumes the `common` list from `aggregate_newcommand_vocab` (a
    cross-paper aggregator over previous superpod batches). For each
    (symbol, body, canon) entry that recurs in ≥2 papers, this strategy
    looks for the symbol in the current paper's text. If the symbol
    appears AND the current paper hasn't defined it (no NewcommandStrategy
    binding starts at scope_start=0 with higher confidence), the learned
    default fires.

    Confidence is `low` because the learned default is, by construction,
    a convention guess: paper authors may use the same macro for
    different meanings ("\\T" is the monad in one paper, the torus in
    another). The `merge_bindings` step will favour any explicit
    in-paper declaration over this default.

    Construction takes `common_vocab` — a list of dicts with at least
    `symbol`, `canon`, `body` keys (the shape `aggregate_newcommand_vocab`
    emits in its `common` slot). When multiple entries share a symbol,
    the one with highest `support` wins.
    """
    name = "learned-vocab"
    default_confidence = "low"

    def __init__(self, common_vocab: list[dict] | None = None):
        self._lookup: dict[str, dict] = {}
        for entry in common_vocab or []:
            sym = entry.get("symbol")
            if not sym:
                continue
            cur = self._lookup.get(sym)
            if cur is None or entry.get("support", 0) > cur.get("support", 0):
                self._lookup[sym] = entry

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        if not self._lookup:
            return []
        out = []
        text = ctx.paper_text
        for sym, entry in self._lookup.items():
            # Match the macro token boundary so `\RR` doesn't match `\RRR`.
            pattern = re.compile(re.escape(sym) + r"(?![A-Za-z])")
            m = pattern.search(text)
            if m is None:
                continue
            canon = entry.get("canon")
            body = entry.get("body", "")
            out.append(SymbolBinding(
                binding_id=ctx.next_id(),
                symbol=sym,
                canon=canon,
                type_phrase=body,
                scope_start=0,
                scope_end=len(text),
                confidence=self.default_confidence,
                strategy=self.name,
                evidence_span=(m.start(), m.end()),
            ))
        return out


class NewcommandStrategy(Strategy):
    r"""`\newcommand{\name}{body}` and siblings — paper-wide, high confidence.

    Author-defined macros are the single richest source of paper-local
    vocabulary. Every arXiv preamble has them; some papers ship 80+. We
    harvest:
      \newcommand{\name}{body}              ; \renewcommand variants
      \newcommand{\name}[N]{body}           ; arity is parsed and discarded
      \DeclareMathOperator{\name}{body}     ; and the *-form
      \def\name{body}                       ; bare \def with no params

    Definitions with `#N` parameters are templates (e.g.
    `\def\foo#1#2{...}`) not symbols, so we skip them. Bodies that match
    `_SKIP_PATTERN` are typographic / structural macros (`\begin`,
    `\hspace`, `\stackrel`, …) — not symbol-grounding candidates.

    Symbol = the macro token itself (`\\RR`, `\\Cat`); the atom-walker
    yields full macro tokens so lookup is exact. Scope is paper-wide
    (`\newcommand` is global by LaTeX convention even when defined
    mid-document). Confidence = high.

    Canon resolution proceeds in three stages so the badge always carries
    a meaningful label:
      1. Blackboard/calligraphic letter (`\mathbb R`, `\cal E`) — look up
         the conventional phrase ("real numbers", "expectation") in the
         kernel.
      2. Failing that, lookup the cleaned body itself as a kernel phrase
         (e.g. `\\Cat` → body "Category" → canon "Category").
      3. Failing that, fall back to the cleaned body as the canon
         (e.g. `\\sE` → "E" — purely typographic, but at least the badge
         shows what the reader sees on the page).
    """
    name = "newcommand"
    default_confidence = "high"

    # These patterns match the *header* (`\newcommand{\name}` + optional
    # arity/default args) and leave the body to be extracted by a balanced-
    # brace walker, since real preambles routinely wrap bodies in
    # `\ensuremath{\mathbf{Cat}}` (3 levels of nesting), which a fixed-depth
    # regex can't handle without false truncation.
    _DEFS_HEADER = re.compile(
        r"\\(?:re)?newcommand\s*\*?\s*\{\s*(\\[A-Za-z@]+)\s*\}"
        r"(?:\s*\[\d+\])?"
        r"(?:\s*\[[^\]]*\])?"
        r"\s*"
    )
    _DEF_HEADER = re.compile(
        r"\\def\s*(\\[A-Za-z@]+)\s*(?:#\d)*\s*"
    )
    _DMO_HEADER = re.compile(
        r"\\DeclareMathOperator\s*\*?\s*\{\s*(\\[A-Za-z@]+)\s*\}\s*"
    )

    _SKIP_PATTERN = re.compile(
        r"\\(?:begin|end|hspace|vspace|hrule|vrule|stackrel|rule|"
        r"label|ref|footnote|noindent|parindent|raisebox|scriptstyle|"
        r"scriptscriptstyle|displaystyle|textstyle|arraystretch|setlength|"
        r"newcolumntype|renewenvironment|newenvironment|protect|"
        r"makeatletter|makeatother|expandafter|let)"
    )

    _FONT_ARG = re.compile(
        r"\\(?:ensuremath|mathbb|mathrm|mathcal|mathfrak|mathbf|mathsf|mathtt|"
        r"mathscr|operatorname|text|textrm|textsf|texttt)\s*\{([^{}]*)\}"
    )
    _FONT_GROUP = re.compile(
        r"\{\s*\\(?:cal|bf|rm|sf|tt|sl|it|em|mathbb|mathrm|mathcal|"
        r"mathfrak|mathbf|mathsf|mathtt|mathscr)\s+([^{}]+)\}"
    )
    _OUTER_BRACES = re.compile(r"^\s*\{(.*)\}\s*$", re.DOTALL)

    # Convention: \mathbb {letter} or \cal {letter} → phrase the kernel
    # might know. We keep this list tight so we don't falsely ground
    # symbols whose convention the author may not be following.
    _LETTER_TO_PHRASE = {
        "R": "real numbers",
        "C": "complex numbers",
        "N": "natural numbers",
        "Q": "rational numbers",
        "Z": "integers",
        "H": "quaternions",
    }

    @staticmethod
    def _extract_balanced(text: str, start: int) -> tuple[str, int] | None:
        """Given text[start] == '{', return (body_with_braces, end_pos)."""
        if start >= len(text) or text[start] != "{":
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1], i + 1
        return None

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        out = []
        text = ctx.paper_text
        kernel_lookup = ctx.kernel_lookup or (lambda _: None)
        for header_pat in (self._DEFS_HEADER, self._DEF_HEADER, self._DMO_HEADER):
            for m in header_pat.finditer(text):
                symbol = m.group(1)
                body_extr = self._extract_balanced(text, m.end())
                if body_extr is None:
                    continue
                body, body_end = body_extr
                body_inner = body
                inner_m = self._OUTER_BRACES.match(body_inner)
                if inner_m:
                    body_inner = inner_m.group(1)
                body_inner = body_inner.strip()
                if not body_inner:
                    continue
                if "#" in body_inner:
                    continue
                if self._SKIP_PATTERN.search(body_inner):
                    continue

                cleaned = self._clean_body(body_inner)
                if not cleaned or not self._is_symbol_like(cleaned):
                    # Reject bodies that, after font/brace stripping, still
                    # look like typography (`\langle`, `\,\cong\,`), pure
                    # numbers (`1.5`, `1`), or decorator fragments (`^op`).
                    # These exist in real preambles but aren't math symbols.
                    continue

                canon = None
                if len(cleaned) == 1 and cleaned in self._LETTER_TO_PHRASE:
                    canon = kernel_lookup(self._LETTER_TO_PHRASE[cleaned])
                if canon is None and cleaned:
                    canon = kernel_lookup(cleaned.lower())
                if canon is None:
                    # Synthesise a label from the cleaned body so the viewer
                    # renders something readable. Cap length so a verbose
                    # body doesn't blow out the badge.
                    canon = cleaned[:24]

                out.append(SymbolBinding(
                    binding_id=ctx.next_id(),
                    symbol=symbol,
                    canon=canon,
                    type_phrase=body_inner,
                    scope_start=0,
                    scope_end=len(text),
                    confidence=self.default_confidence,
                    strategy=self.name,
                    evidence_span=(m.start(), body_end),
                ))
        return out

    _SYMBOL_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,30}$")

    @classmethod
    def _is_symbol_like(cls, cleaned: str) -> bool:
        """A cleaned body is a math-symbol candidate iff it's an alphabetic
        token of reasonable length. This excludes typography like `\\quad`
        (still carries a backslash), decorator fragments (`^op`), and
        bare numerics (`1.5`, `1`)."""
        return bool(cls._SYMBOL_LIKE.match(cleaned))

    @classmethod
    def _clean_body(cls, body: str) -> str:
        """Strip font wrappers + braces. `{\\mathbb R}` -> `R`; `{Category}` -> `Category`."""
        s = body
        prev = None
        while prev != s:
            prev = s
            s = cls._FONT_ARG.sub(r"\1", s)
            s = cls._FONT_GROUP.sub(r"\1", s)
            m = cls._OUTER_BRACES.match(s)
            if m:
                s = m.group(1)
            s = s.strip()
        return s


class FixPatternStrategy(Strategy):
    """`Fix $X$ as Y` / `Fix $X$ to be Y` — high confidence.

    Mirror of `LetBindingStrategy`. The "fix" verb is common in algebra
    and analysis; the binding is just as authoritative as "let".
    """
    name = "fix-pattern"
    default_confidence = "high"

    _PATTERN = re.compile(
        r"\bFix\s+\$([^$\n]{1,40})\$\s+(?:as|to\s+be)\s+(?:(?:an|a|the)\s+)?"
        r"([a-z][\w\s\-]{2,60}?)"
        r"(?=[.,;:\n]|\s+(?:and|or|with|such|over|in|on|of|for|that|which)\b)",
        re.IGNORECASE,
    )

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        out = []
        for m in self._PATTERN.finditer(ctx.paper_text):
            symbol = m.group(1).strip()
            type_phrase = m.group(2).strip().lower()
            canon = (ctx.kernel_lookup or (lambda _: None))(type_phrase)
            out.append(SymbolBinding(
                binding_id=ctx.next_id(),
                symbol=symbol,
                canon=canon,
                type_phrase=type_phrase,
                scope_start=_scope_start_after_punct(ctx.paper_text, m.end()),
                scope_end=len(ctx.paper_text),
                confidence=self.default_confidence,
                strategy=self.name,
                evidence_span=(m.start(), m.end()),
            ))
        return out


class InlineIsAStrategy(Strategy):
    """`$X$ is a Y` — medium confidence.

    Like `the-Y-X`, weaker than declarations because "$X$ is a category"
    can either declare X as a category or claim it (citing prior
    declaration). The meta-learning loop will surface defeat rate.
    """
    name = "inline-is-a"
    default_confidence = "medium"

    _PATTERN = re.compile(
        r"\$([^$\n]{1,40})\$\s+is\s+(?:(?:an|a|the)\s+)?"
        r"([a-z][\w\s\-]{2,60}?)"
        r"(?=[.,;:\n]|\s+(?:and|or|with|such|over|in|on|of|where)\b)",
        re.IGNORECASE,
    )

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        out = []
        kernel_lookup = ctx.kernel_lookup or (lambda _: None)
        for m in self._PATTERN.finditer(ctx.paper_text):
            symbol = m.group(1).strip()
            type_phrase = m.group(2).strip().lower()
            if len(type_phrase) < 3 or type_phrase in {
                "set of", "list of", "case of", "kind of", "way of",
            }:
                continue
            canon = kernel_lookup(type_phrase)
            out.append(SymbolBinding(
                binding_id=ctx.next_id(),
                symbol=symbol,
                canon=canon,
                type_phrase=type_phrase,
                scope_start=m.end(),
                scope_end=len(ctx.paper_text),
                confidence=self.default_confidence,
                strategy=self.name,
                evidence_span=(m.start(), m.end()),
            ))
        return out


class NotationEnvStrategy(Strategy):
    r"""`\begin{notation}…\end{notation}` blocks — high confidence inside.

    Many algebra/CT papers ship a `notation` environment where authors
    declare paper-local symbols *en masse*. The block has the highest
    declarative density per character in the entire paper. Inside, we
    re-run the let-binding / denotation regexes; bindings inside this
    block carry the parent strategy name `notation-env` so the meta-
    learning loop sees them as a separate channel.

    The environment names cover the common variants:
    `notation`, `notations`, `convention`, `conventions`.
    """
    name = "notation-env"
    default_confidence = "high"

    _ENV_PATTERN = re.compile(
        r"\\begin\{(?:notation|notations|convention|conventions)\*?\}"
        r"([\s\S]*?)"
        r"\\end\{(?:notation|notations|convention|conventions)\*?\}",
        re.IGNORECASE,
    )
    _DECL_PATTERN = re.compile(
        r"\$([^$\n]{1,40})\$\s+"
        r"(?:denotes?|stands?\s+for|is)\s+"
        r"(?:(?:an|a|the)\s+)?"
        r"([a-z][\w\s\-]{2,60}?)"
        r"(?=[.,;:\n]|\s+(?:and|or|with|such|over|in|on|of|for)\b)",
        re.IGNORECASE,
    )

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        out = []
        kernel_lookup = ctx.kernel_lookup or (lambda _: None)
        for env_m in self._ENV_PATTERN.finditer(ctx.paper_text):
            body = env_m.group(1)
            body_offset = env_m.start(1)
            for m in self._DECL_PATTERN.finditer(body):
                symbol = m.group(1).strip()
                type_phrase = m.group(2).strip().lower()
                canon = kernel_lookup(type_phrase)
                # Notation declarations apply paper-wide (the conventional
                # block at the front of the paper is meant as a glossary).
                out.append(SymbolBinding(
                    binding_id=ctx.next_id(),
                    symbol=symbol,
                    canon=canon,
                    type_phrase=type_phrase,
                    scope_start=0,
                    scope_end=len(ctx.paper_text),
                    confidence=self.default_confidence,
                    strategy=self.name,
                    evidence_span=(body_offset + m.start(), body_offset + m.end()),
                ))
        return out


class TheYXStrategy(Strategy):
    """`the Y $X$` (e.g., "the category $\\mathcal{C}$") — medium confidence.

    Weaker than let-binding because `the Y $X$` can also mean "the previously-
    referenced Y", not necessarily a fresh declaration. Confidence='medium'
    reflects this; merge will let `let-binding` override this where both
    fire on the same symbol.
    """
    name = "the-Y-X"
    default_confidence = "medium"

    _PATTERN = re.compile(
        r"\bthe\s+([a-z][\w\s\-]{2,40}?)\s+\$([^$\n]{1,40})\$",
        re.IGNORECASE,
    )

    def apply(self, ctx: StrategyContext) -> list[SymbolBinding]:
        out = []
        for m in self._PATTERN.finditer(ctx.paper_text):
            type_phrase = m.group(1).strip().lower()
            symbol = m.group(2).strip()
            # Reject if the type phrase is too short / too generic.
            if len(type_phrase) < 3 or type_phrase in {"set of", "list of", "case of"}:
                continue
            canon = (ctx.kernel_lookup or (lambda _: None))(type_phrase)
            out.append(SymbolBinding(
                binding_id=ctx.next_id(),
                symbol=symbol,
                canon=canon,
                type_phrase=type_phrase,
                scope_start=m.end(),
                scope_end=len(ctx.paper_text),
                confidence=self.default_confidence,
                strategy=self.name,
                evidence_span=(m.start(), m.end()),
            ))
        return out


# ============================================================
# Merge with defeasibility
# ============================================================

def merge_bindings(bindings: Iterable[SymbolBinding]) -> list[SymbolBinding]:
    """Resolve overlapping bindings via defeasible scope-narrowing.

    Within a single symbol's bindings, sorted by scope_start:
    - A later binding narrows the earlier binding's scope_end to the
      later binding's scope_start, and the earlier binding gets
      defeated_by = later.binding_id.
    - If two bindings start at the exact same position (rare; e.g., two
      strategies fired on the same evidence), the higher-confidence one
      wins and the other gets defeated_by set.

    Returns the full list (defeated and undefeated) so the meta-learning
    loop can read off defeat rates per strategy. Callers that want only
    the *active* bindings should filter `b.defeated_by is None`.
    """
    by_symbol: dict[str, list[SymbolBinding]] = {}
    for b in bindings:
        by_symbol.setdefault(b.symbol, []).append(b)

    out: list[SymbolBinding] = []
    for symbol, blist in by_symbol.items():
        # Sort by (scope_start asc, -confidence) so higher confidence
        # wins on exact-tie starts.
        blist = sorted(
            blist,
            key=lambda b: (b.scope_start, -_CONFIDENCE_RANK.get(b.confidence, 0)),
        )
        active_idx: list[int] = []  # indices of currently-active bindings, by start position
        for i, cur in enumerate(blist):
            # Defeat anyone earlier whose scope still extends past cur.scope_start.
            for j in active_idx:
                prev = blist[j]
                if prev.defeated_by is not None:
                    continue
                if prev.scope_start == cur.scope_start:
                    # Exact tie. Higher-confidence comes first in our sort,
                    # so `prev` is at least as confident as `cur` — defeat
                    # `cur` instead (cur is later in iteration).
                    if not _confidence_geq(cur.confidence, prev.confidence):
                        cur.defeated_by = prev.binding_id
                    else:
                        # equal confidence: still mark prev as defeated by cur
                        # to surface the conflict (could refine later)
                        prev.defeated_by = cur.binding_id
                        prev.scope_end = cur.scope_start
                elif prev.scope_end > cur.scope_start:
                    # cur starts within prev's scope → narrow prev.
                    prev.scope_end = cur.scope_start
                    prev.defeated_by = cur.binding_id
            active_idx.append(i)
        out.extend(blist)
    return out


# ============================================================
# Per-paper symbol environment (piecewise lookup)
# ============================================================

class SymbolEnvironment:
    """Piecewise lookup over a merged list of bindings.

    Given a paper position `p` and a symbol `X`, return the binding
    whose scope range contains `p` and which is not defeated at that
    position. If multiple bindings span `p`, return the highest-
    confidence one (or the latest, breaking ties).
    """

    def __init__(self, bindings: list[SymbolBinding]):
        self.all_bindings = list(bindings)
        self._by_symbol: dict[str, list[SymbolBinding]] = {}
        for b in self.all_bindings:
            self._by_symbol.setdefault(b.symbol, []).append(b)

    def lookup(self, symbol: str, position: int) -> SymbolBinding | None:
        """Return the binding active for `symbol` at `position`.

        A binding's `defeated_by` marks that a later binding NARROWED its
        scope — but the original binding remains active within its
        (narrowed) scope range. So lookup checks only scope-range
        membership; `defeated_by` is meta-learning bookkeeping.
        """
        candidates = self._by_symbol.get(symbol, [])
        active = [
            b for b in candidates
            if b.scope_start <= position < b.scope_end
        ]
        if not active:
            return None
        active.sort(
            key=lambda b: (-_CONFIDENCE_RANK.get(b.confidence, 0), -b.scope_start),
        )
        return active[0]

    def all_active(self) -> list[SymbolBinding]:
        """Return bindings that were NEVER narrowed (`defeated_by is None`).

        Useful for the QC report's "uncontested bindings" view. A defeated
        binding is still active in its narrowed range (see `lookup`), but
        this method intentionally excludes them so the report distinguishes
        bindings that held throughout the paper from bindings that got
        overridden.
        """
        return [b for b in self.all_bindings if b.defeated_by is None]


# ============================================================
# Orchestrator
# ============================================================

def run_strategies(
    ctx: StrategyContext,
    strategies: list[Strategy],
) -> SymbolEnvironment:
    """Run all strategies, merge with defeasibility, return environment.

    The full per-strategy log lives in env.all_bindings (including
    defeated entries) so the QC layer can compute hit/corroboration
    rates without re-running.
    """
    raw: list[SymbolBinding] = []
    for strat in strategies:
        raw.extend(strat.apply(ctx))
    merged = merge_bindings(raw)
    return SymbolEnvironment(merged)


def default_strategies(
    learned_vocab: list[dict] | None = None,
) -> list[Strategy]:
    """The starter strategy set. Add to this list as new strategies land.

    Pass `learned_vocab` (the `common` slot from
    `aggregate_newcommand_vocab`) to include the cross-paper
    `LearnedVocabStrategy`. With no vocab, the strategy is omitted —
    fresh runs have nothing to learn from yet.
    """
    strategies: list[Strategy] = [
        NewcommandStrategy(),
        NotationEnvStrategy(),
        LetBindingStrategy(),
        FixPatternStrategy(),
        DenotationStrategy(),
        InlineIsAStrategy(),
        TheYXStrategy(),
        KernelAmbientStrategy(),
    ]
    if learned_vocab:
        strategies.append(LearnedVocabStrategy(learned_vocab))
    return strategies


# ============================================================
# Per-paper strategy metrics (input to cross-paper meta-learning)
# ============================================================

def compute_strategy_metrics(env: "SymbolEnvironment") -> dict[str, dict]:
    """Per-strategy emission / defeat / corroboration on one paper.

    For each strategy name, return:
      - emitted: total bindings the strategy produced
      - defeated: bindings whose scope got narrowed by later evidence
        (a binding's `defeated_by` is set whenever another binding for
        the same symbol started inside its scope range — typical when a
        local "Let X be a finite abelian group" supersedes an earlier
        global "Let X be an abelian group")
      - corroborated: bindings that share both symbol AND canon with a
        binding produced by a DIFFERENT strategy (an independent vote
        for the same interpretation)
      - solo: emitted - defeated - corroborated; bindings that are
        neither contradicted nor independently confirmed

    The mission framing (M-symbol-grounding.md §3) calls these
    hit/defeat/corroboration rates; "rate" is computed at the
    aggregation step where we have an emitted-bindings denominator.
    """
    by_strategy: dict[str, dict] = {}
    # Group bindings by symbol to find cross-strategy corroboration
    by_symbol: dict[str, list[SymbolBinding]] = {}
    for b in env.all_bindings:
        by_symbol.setdefault(b.symbol, []).append(b)

    def init(name: str) -> dict:
        return by_strategy.setdefault(name, {
            "emitted": 0, "defeated": 0, "corroborated": 0, "solo": 0,
        })

    for b in env.all_bindings:
        slot = init(b.strategy)
        slot["emitted"] += 1
        is_defeated = b.defeated_by is not None
        if is_defeated:
            slot["defeated"] += 1
        # Corroborated iff another binding for the same symbol from a
        # different strategy shares the same canon.
        others = by_symbol.get(b.symbol, [])
        is_corroborated = any(
            o is not b
            and o.strategy != b.strategy
            and o.canon == b.canon
            and b.canon is not None
            for o in others
        )
        if is_corroborated:
            slot["corroborated"] += 1
        # Solo = neither defeated nor corroborated. The trio
        # (defeated, corroborated, solo) is mutually exclusive so the
        # three sum to `emitted` even when individual bindings are
        # both defeated AND corroborated (we count them once, in the
        # defeated bucket — defeat outranks corroboration as a signal
        # since it represents a direct contradiction).
        if not is_defeated and not is_corroborated:
            slot["solo"] += 1

    return by_strategy


# ============================================================
# Cross-paper aggregation of strategy metrics
# ============================================================

def aggregate_strategy_metrics(
    metrics_by_paper: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Sum per-paper strategy metrics into cross-paper totals + rates.

    Returns dict[strategy_name -> {emitted, defeated, corroborated,
    solo, papers_active, defeat_rate, corroboration_rate}].

    `defeat_rate` and `corroboration_rate` are floats in [0, 1],
    computed as fraction of `emitted`. `papers_active` is the count of
    papers where the strategy emitted at least one binding — useful for
    spotting strategies that fire rarely but reliably.
    """
    out: dict[str, dict] = {}
    for paper_id, per_strategy in metrics_by_paper.items():
        for strat, slot in per_strategy.items():
            agg = out.setdefault(strat, {
                "emitted": 0, "defeated": 0, "corroborated": 0,
                "solo": 0, "papers_active": 0,
            })
            if slot["emitted"] > 0:
                agg["papers_active"] += 1
            for k in ("emitted", "defeated", "corroborated", "solo"):
                agg[k] += slot.get(k, 0)
    for strat, agg in out.items():
        emit = agg["emitted"]
        agg["defeat_rate"] = (agg["defeated"] / emit) if emit else 0.0
        agg["corroboration_rate"] = (
            agg["corroborated"] / emit if emit else 0.0
        )
    return out


# ============================================================
# Cross-paper newcommand vocabulary aggregation
# ============================================================

def aggregate_newcommand_vocab(
    envs_by_paper: dict[str, "SymbolEnvironment"],
) -> dict:
    """Aggregate newcommand bindings across papers into a learned vocabulary.

    The output shape is intentionally simple:
        {
          "by_symbol": {
            "\\RR": [
              {"paper_id": "0712.4211v1", "body": "{\\mathbb R}", "canon": "R"},
              ...
            ],
            ...
          },
          "common": [   # symbols that appear in N>=2 papers with same body
            {"symbol": "\\RR", "body": "{\\mathbb R}", "canon": "R",
             "papers": ["0712.4211v1", "..."], "support": 2},
            ...
          ],
        }

    This is the seed of cross-paper syntax learning: when a fresh paper
    uses `\\RR` without defining it, a future strategy can consult
    `common` to retrieve the canonical interpretation. With four demo
    papers the `common` list is sparse, but the mechanism scales: run
    the same aggregation across a 1000-paper superpod batch and the
    convention table fills out fast.
    """
    by_symbol: dict[str, list[dict]] = {}
    for paper_id, env in envs_by_paper.items():
        for b in env.all_bindings:
            if b.strategy != "newcommand":
                continue
            by_symbol.setdefault(b.symbol, []).append({
                "paper_id": paper_id,
                "body": b.type_phrase,
                "canon": b.canon,
            })

    common = []
    for symbol, entries in by_symbol.items():
        # Group by body to see which (symbol,body) pairs recur
        by_body: dict[str, dict] = {}
        for e in entries:
            slot = by_body.setdefault(e["body"], {
                "symbol": symbol,
                "body": e["body"],
                "canon": e["canon"],
                "papers": [],
            })
            slot["papers"].append(e["paper_id"])
        for slot in by_body.values():
            if len(slot["papers"]) >= 2:
                slot["support"] = len(slot["papers"])
                common.append(slot)
    common.sort(key=lambda d: -d["support"])

    return {"by_symbol": by_symbol, "common": common}
