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
    if the phrase is in the kernel.
    """
    paper_id: str
    paper_text: str
    math_envelopes: list[tuple[int, int, str]] = field(default_factory=list)
    kernel_lookup: Callable[[str], str | None] | None = None
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


def default_strategies() -> list[Strategy]:
    """The starter strategy set. Add to this list as new strategies land."""
    return [
        NewcommandStrategy(),
        LetBindingStrategy(),
        DenotationStrategy(),
        TheYXStrategy(),
    ]


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
