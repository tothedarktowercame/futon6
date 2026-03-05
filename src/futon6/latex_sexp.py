"""LaTeX → s-expression parser for mathematical expressions.

Stage 8 of the futon6 pipeline: parse every $...$ LaTeX expression into
a typed s-expression tree. Each expression becomes a surface for wiring.

    >>> parse(r"\\Gamma=(V,E,s,t)")
    '(= Γ (tuple V E s t))'
    >>> parse(r"s(X(e))=X(s(e))")
    '(= (s (X e)) (X (s e)))'
    >>> parse(r"e : v \\to w")
    '(: e (→ v w))'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

# ---------------------------------------------------------------------------
# S-expression AST
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    value: str
    def __str__(self) -> str:
        return self.value

@dataclass
class App:
    """Function application or operator: (op arg1 arg2 ...)"""
    op: str
    args: list[SExp]
    def __str__(self) -> str:
        if not self.args:
            return self.op
        parts = [self.op] + [str(a) for a in self.args]
        return '(' + ' '.join(parts) + ')'

SExp = Union[Atom, App]


# ---------------------------------------------------------------------------
# Greek / symbol mapping
# ---------------------------------------------------------------------------

GREEK = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π',
    'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ',
    'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
    'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ',
    'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
}

OPERATORS = {
    'in': '∈', 'notin': '∉', 'subset': '⊂', 'subseteq': '⊆',
    'supset': '⊃', 'supseteq': '⊇', 'cup': '∪', 'cap': '∩',
    'circ': '∘', 'cdot': '·', 'times': '×', 'otimes': '⊗',
    'oplus': '⊕', 'to': '→', 'rightarrow': '→', 'leftarrow': '←',
    'Rightarrow': '⇒', 'Leftarrow': '⇐', 'leftrightarrow': '↔',
    'mapsto': '↦', 'hookrightarrow': '↪',
    'forall': '∀', 'exists': '∃', 'neg': '¬',
    'land': '∧', 'lor': '∨', 'implies': '⟹',
    'leq': '≤', 'geq': '≥', 'neq': '≠', 'equiv': '≡',
    'approx': '≈', 'sim': '∼', 'cong': '≅', 'simeq': '≃',
    'pm': '±', 'mp': '∓', 'infty': '∞', 'emptyset': '∅',
    'rightrightarrows': '⇉', 'downarrow': '↓', 'uparrow': '↑',
    'bullet': '•',
}

# Binary operators by precedence (low to high)
BINOP_PRECEDENCE = {
    ':=': 1, '=': 2, '≠': 2, '≡': 2, '≅': 2, '≃': 2, '≈': 2,
    ':': 3,
    '∈': 4, '∉': 4, '⊂': 4, '⊆': 4, '⊃': 4, '⊇': 4,
    '→': 5, '←': 5, '↔': 5, '⇒': 5, '⇐': 5, '↦': 5, '↪': 5,
    '+': 6, '-': 6, '±': 6, '∪': 6, '∨': 6,
    '∘': 7, '·': 7, '×': 7, '⊗': 7, '⊕': 7, '∧': 7,
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str   # 'ident', 'number', 'op', 'lparen', 'rparen', 'lbrace',
                # 'rbrace', 'comma', 'sub', 'sup', 'prime', 'command', 'eof'
    value: str

_TOKEN_RE = re.compile(r"""
    (?P<newline>\\\\)               |  # \\ (LaTeX newline / array row sep)
    (?P<spacing_cmd>\\[,;:! ])      |  # \, \; \: \! \  (spacing commands)
    (?P<command>\\[a-zA-Z]+)       |  # \command
    (?P<number>[0-9]+(?:\.[0-9]+)?) |  # numbers
    (?P<defeq>:=)                   |  # := (before : alone)
    (?P<op>[=:<>+\-])               |  # single-char operators
    (?P<lparen>[(])                  |
    (?P<rparen>[)])                  |
    (?P<lbrace>[{])                  |
    (?P<rbrace>[}])                  |
    (?P<lbracket>\[)                 |
    (?P<rbracket>\])                 |
    (?P<comma>,)                     |
    (?P<sub>_)                       |
    (?P<sup>\^)                      |
    (?P<prime>')                     |
    (?P<star>[*])                     |  # postfix star (dual/adjoint)
    (?P<hash>[#])                     |  # postfix hash (sharp)
    (?P<ident>[a-zA-Z])              |  # single letter identifiers
    (?P<ws>\s+)                      |  # whitespace (skip)
    (?P<amp>&)                       |  # array separator (skip)
    (?P<other>.)                        # anything else
""", re.VERBOSE)


def tokenize(latex: str) -> list[Token]:
    tokens = []
    for m in _TOKEN_RE.finditer(latex):
        kind = m.lastgroup
        val = m.group()
        if kind in ('ws', 'amp', 'spacing_cmd'):
            continue
        if kind == 'newline':
            tokens.append(Token('newline', '\\\\'))
            continue
        if kind == 'star':
            tokens.append(Token('star', '*'))
            continue
        if kind == 'hash':
            tokens.append(Token('hash', '#'))
            continue
        if kind == 'command':
            cmd = val[1:]  # strip backslash
            if cmd in GREEK:
                tokens.append(Token('ident', GREEK[cmd]))
            elif cmd in OPERATORS:
                tokens.append(Token('op', OPERATORS[cmd]))
            elif cmd in ('mathcal', 'mathsf', 'mathrm', 'mathbb',
                         'mathbf', 'mathit', 'text', 'textbf', 'textit',
                         'rm', 'bf', 'it', 'sf', 'operatorname'):
                tokens.append(Token('command', cmd))
            elif cmd == 'frac':
                tokens.append(Token('command', 'frac'))
            elif cmd in ('left', 'right', 'big', 'Big', 'bigg', 'Bigg'):
                continue  # sizing — skip
            elif cmd in ('quad', 'qquad', 'hspace', 'hfill',
                         ',', ';', '!', ':', ' '):
                continue  # spacing — skip (includes \, \; \! \: \ )
            elif cmd == 'bar':
                tokens.append(Token('command', 'bar'))
            elif cmd in ('hat', 'tilde', 'vec', 'dot', 'ddot', 'overline', 'underline'):
                tokens.append(Token('command', cmd))
            elif cmd == 'begin' or cmd == 'end':
                tokens.append(Token('command', cmd))
            else:
                # Unknown command — treat as identifier
                tokens.append(Token('ident', cmd))
        elif kind == 'defeq':
            tokens.append(Token('op', ':='))
        elif kind == 'op':
            if val == ':':
                tokens.append(Token('op', ':'))
            elif val == '=':
                tokens.append(Token('op', '='))
            elif val == '<':
                tokens.append(Token('op', '<'))
            elif val == '>':
                tokens.append(Token('op', '>'))
            elif val == '+':
                tokens.append(Token('op', '+'))
            elif val == '-':
                tokens.append(Token('op', '-'))
        elif kind == 'other':
            if val == '\\':
                continue  # stray backslash
            # skip unknown chars
            continue
        else:
            tokens.append(Token(kind, val))
    tokens.append(Token('eof', ''))
    return tokens


# ---------------------------------------------------------------------------
# Parser (recursive descent with precedence climbing)
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, kind: str) -> Token:
        t = self.advance()
        if t.kind != kind:
            raise ParseError(f"expected {kind}, got {t.kind} ({t.value!r})")
        return t

    def at(self, kind: str, value: str | None = None) -> bool:
        t = self.peek()
        if t.kind != kind:
            return False
        if value is not None and t.value != value:
            return False
        return True

    def parse_expr(self, min_prec: int = 0) -> SExp:
        """Precedence climbing for binary operators, with comma as tuple."""
        left = self.parse_unary()

        while True:
            t = self.peek()

            # Comma at top level → tuple
            if t.kind == 'comma' and min_prec == 0:
                items = [left]
                while self.at('comma'):
                    self.advance()
                    items.append(self.parse_expr(1))
                left = App('tuple', items) if len(items) > 1 else items[0]
                break

            if t.kind != 'op' or t.value not in BINOP_PRECEDENCE:
                break
            prec = BINOP_PRECEDENCE[t.value]
            if prec < min_prec:
                break
            op = self.advance().value
            right = self.parse_expr(prec + 1)
            left = App(op, [left, right])

        return left

    def parse_unary(self) -> SExp:
        """Parse a unary expression (atom, possibly with subscript/superscript/application)."""
        t = self.peek()

        # Quantifiers: ∀, ∃
        if t.kind == 'op' and t.value in ('∀', '∃'):
            self.advance()
            body = self.parse_expr(0)
            return App(t.value, [body])

        # Negation
        if t.kind == 'op' and t.value == '¬':
            self.advance()
            body = self.parse_unary()
            return App('¬', [body])

        base = self.parse_atom()

        # Post-modifiers: subscript, superscript, prime, application
        while True:
            t = self.peek()
            if t.kind == 'sub':
                self.advance()
                sub = self.parse_atom()
                base = App('sub', [base, sub])
            elif t.kind == 'sup':
                self.advance()
                sup = self.parse_atom()
                base = App('sup', [base, sup])
            elif t.kind == 'prime':
                self.advance()
                base = App('prime', [base])
            elif t.kind == 'star':
                self.advance()
                base = App('star', [base])
            elif t.kind == 'hash':
                self.advance()
                base = App('sharp', [base])
            elif t.kind == 'lparen':
                # Function application: f(x) or f(x,y,...)
                args = self.parse_paren_args()
                if len(args) == 1 and isinstance(args[0], Atom) and args[0].value == '-':
                    # f(-) → placeholder notation
                    base = App(str(base), [Atom('□')])
                else:
                    base = App(str(base), args)
            else:
                break

        return base

    def parse_atom(self) -> SExp:
        t = self.peek()

        if t.kind == 'ident':
            self.advance()
            return Atom(t.value)

        if t.kind == 'number':
            self.advance()
            return Atom(t.value)

        if t.kind == 'op' and t.value in ('•', '∞', '∅', '-'):
            self.advance()
            return Atom(t.value)

        if t.kind == 'star':
            self.advance()
            return Atom('*')

        if t.kind == 'hash':
            self.advance()
            return Atom('#')

        if t.kind == 'command':
            return self.parse_command()

        if t.kind == 'lparen':
            return self.parse_paren()

        if t.kind == 'lbrace':
            return self.parse_brace_group()

        if t.kind == 'lbracket':
            self.advance()
            inner = self.parse_expr(0)
            if self.at('rbracket'):
                self.advance()
            return inner

        # Fallback: skip and return placeholder
        self.advance()
        return Atom('?')

    def parse_command(self) -> SExp:
        t = self.advance()
        cmd = t.value

        if cmd in ('rm', 'bf', 'it', 'sf'):
            # Old-style font switches: \rm X or \rm{X} — consume next atom
            if self.at('lbrace'):
                content = self.parse_brace_content()
            else:
                parts = []
                while self.peek().kind in ('ident', 'number'):
                    parts.append(self.advance().value)
                content = ''.join(parts) if parts else cmd
            return Atom(content)

        if cmd in ('mathcal', 'mathsf', 'mathrm', 'mathbb', 'mathbf',
                    'mathit', 'text', 'textbf', 'textit', 'operatorname'):
            # \mathcal{C} → the decorated name
            content = self.parse_brace_content()
            # For single letters in mathcal/mathbb, use unicode if possible
            if cmd == 'mathcal' and len(content) == 1:
                MATHCAL = {'A': '𝒜', 'B': 'ℬ', 'C': '𝒞', 'D': '𝒟', 'E': 'ℰ',
                           'F': 'ℱ', 'G': '𝒢', 'H': 'ℋ', 'I': 'ℐ', 'J': '𝒥',
                           'K': '𝒦', 'L': 'ℒ', 'M': 'ℳ', 'N': '𝒩', 'O': '𝒪',
                           'P': '𝒫', 'Q': '𝒬', 'R': 'ℛ', 'S': '𝒮', 'T': '𝒯',
                           'U': '𝒰', 'V': '𝒱', 'W': '𝒲', 'X': '𝒳', 'Y': '𝒴',
                           'Z': '𝒵'}
                return Atom(MATHCAL.get(content, content))
            elif cmd == 'mathbb' and len(content) == 1:
                MATHBB = {'N': 'ℕ', 'Z': 'ℤ', 'Q': 'ℚ', 'R': 'ℝ', 'C': 'ℂ',
                           'P': 'ℙ', 'F': '𝔽'}
                return Atom(MATHBB.get(content, content))
            else:
                return Atom(content)

        if cmd == 'frac':
            # \frac{a}{b} is standard. Shorthand \frac12 means \frac{1}{2}:
            # when next token is a multi-digit number without braces, split
            # first digit as numerator and rest as denominator.
            if (not self.at('lbrace') and self.peek().kind == 'number'
                    and len(self.peek().value) == 2):
                digits = self.advance().value
                return App('/', [Atom(digits[0]), Atom(digits[1])])
            num = self.parse_brace_group()
            den = self.parse_brace_group()
            return App('/', [num, den])

        if cmd in ('bar', 'overline'):
            arg = self.parse_atom()
            return App('bar', [arg])

        if cmd in ('hat', 'tilde', 'vec', 'dot', 'ddot', 'underline'):
            arg = self.parse_atom()
            return App(cmd, [arg])

        if cmd == 'begin':
            return self.parse_environment()

        if cmd == 'end':
            # consume {envname} and return placeholder
            self.parse_brace_content()
            return Atom('')

        return Atom(cmd)

    def parse_brace_content(self) -> str:
        """Parse {content} and return the raw content string."""
        if not self.at('lbrace'):
            # No brace — take the next single token
            t = self.advance()
            return t.value
        self.advance()  # skip {
        depth = 1
        parts = []
        while depth > 0:
            t = self.advance()
            if t.kind == 'lbrace':
                depth += 1
                parts.append('{')
            elif t.kind == 'rbrace':
                depth -= 1
                if depth > 0:
                    parts.append('}')
            elif t.kind == 'eof':
                break
            else:
                parts.append(t.value)
        return ''.join(parts)

    def parse_brace_group(self) -> SExp:
        """Parse {expr} as a grouped expression."""
        if not self.at('lbrace'):
            return self.parse_atom()
        self.advance()  # skip {
        expr = self.parse_expr(0)
        if self.at('rbrace'):
            self.advance()
        return expr

    def parse_paren(self) -> SExp:
        """Parse (expr) or (a, b, c) as grouping or tuple."""
        self.advance()  # skip (
        if self.at('rparen'):
            self.advance()
            return Atom('()')
        first = self.parse_expr(0)
        if self.at('comma'):
            # Tuple
            items = [first]
            while self.at('comma'):
                self.advance()
                items.append(self.parse_expr(0))
            if self.at('rparen'):
                self.advance()
            return App('tuple', items)
        if self.at('rparen'):
            self.advance()
        return first

    def parse_paren_args(self) -> list[SExp]:
        """Parse (arg1, arg2, ...) for function application."""
        self.advance()  # skip (
        if self.at('rparen'):
            self.advance()
            return []
        args = [self.parse_expr(0)]
        while self.at('comma'):
            self.advance()
            args.append(self.parse_expr(0))
        if self.at('rparen'):
            self.advance()
        return args

    def parse_environment(self) -> SExp:
        r"""Parse \begin{env}...\end{env}. For arrays, extract structure."""
        env_name = self.parse_brace_content()
        if env_name == 'array':
            return self.parse_array()
        # Generic: skip until \end{env_name}
        depth = 1
        while depth > 0:
            t = self.advance()
            if t.kind == 'eof':
                break
            if t.kind == 'command' and t.value == 'begin':
                self.parse_brace_content()
                depth += 1
            elif t.kind == 'command' and t.value == 'end':
                self.parse_brace_content()
                depth -= 1
        return Atom(f'[{env_name}]')

    def parse_array(self) -> SExp:
        """Parse array environment content, extract arrows as graph edges."""
        # Skip column spec {c} or {ccc} etc.
        if self.at('lbrace'):
            self.parse_brace_content()
        # Collect all tokens in each row (separated by \\) until \end{array}
        edges = []
        current_row: list[SExp] = []
        while True:
            t = self.peek()
            if t.kind == 'eof':
                break
            if t.kind == 'command' and t.value == 'end':
                self.advance()
                self.parse_brace_content()
                break
            if t.kind == 'newline':
                # Row separator \\
                self.advance()
                if current_row:
                    edges.extend(self._extract_row_arrows(current_row))
                    current_row = []
                continue
            # In array context, collect ops and atoms individually
            if t.kind == 'op':
                self.advance()
                current_row.append(Atom(t.value))
            elif t.kind == 'ident':
                self.advance()
                current_row.append(Atom(t.value))
            elif t.kind == 'command':
                node = self.parse_command()
                if str(node):
                    current_row.append(node)
            else:
                self.advance()  # skip unknown tokens in array
        if current_row:
            edges.extend(self._extract_row_arrows(current_row))
        if edges:
            return App('graph', edges)
        return Atom('[array]')

    def _extract_row_arrows(self, items: list[SExp]) -> list[SExp]:
        """From a row of [•, →, •] extract (→ • •) edges."""
        edges = []
        arrow_syms = {'→', '←', '↓', '↑', '⇉', '↔', '⇒', '⇐'}
        i = 0
        while i < len(items):
            s = str(items[i])
            if s in arrow_syms and i > 0 and i < len(items) - 1:
                edges.append(App(s, [items[i - 1], items[i + 1]]))
                i += 2
            else:
                i += 1
        return edges


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(latex: str) -> str:
    """Parse a LaTeX math expression into an s-expression string.

    Returns the s-exp as a string. For unparseable input, returns the
    LaTeX wrapped in quotes as a fallback.
    """
    latex = latex.strip()
    if not latex:
        return '""'
    try:
        tokens = tokenize(latex)
        parser = Parser(tokens)
        result = parser.parse_expr(0)
        return str(result)
    except (ParseError, IndexError, RecursionError):
        return f'"{latex}"'


def parse_tree(latex: str) -> SExp:
    """Parse a LaTeX math expression into an s-expression AST."""
    tokens = tokenize(latex)
    parser = Parser(tokens)
    return parser.parse_expr(0)


def parse_all(html: str) -> list[dict]:
    """Extract and parse all $...$ and $$...$$ from an HTML string.

    Returns a list of dicts: {latex, display, sexp, position}.
    """
    results = []
    # Display math first (greedy match)
    for m in re.finditer(r'\$\$(.+?)\$\$', html, re.DOTALL):
        tex = m.group(1).strip()
        results.append({
            'latex': tex,
            'display': True,
            'sexp': parse(tex),
            'position': m.start(),
        })
    # Inline math (avoid matching inside $$...$$)
    display_ranges = [(m.start(), m.end())
                      for m in re.finditer(r'\$\$.+?\$\$', html, re.DOTALL)]
    for m in re.finditer(r'\$([^$\n]+?)\$', html):
        if any(s <= m.start() < e for s, e in display_ranges):
            continue
        tex = m.group(1).strip()
        results.append({
            'latex': tex,
            'display': False,
            'sexp': parse(tex),
            'position': m.start(),
        })
    results.sort(key=lambda r: r['position'])
    return results
