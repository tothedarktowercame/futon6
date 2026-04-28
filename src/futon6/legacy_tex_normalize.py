"""Legacy TeX normalization for paper-stage extraction.

This module is intentionally narrow: it rewrites high-confidence,
source-declared theorem and proof aliases into the canonical environment
names already consumed by paper_hypergraph.py.

It does not attempt prose synthesis yet. The goal is to keep the first
normalization pass conservative and provenance-friendly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_NEWTHEOREM_RE = re.compile(
    r"\\newtheorem\*?\{(?P<alias>[^}]+)\}"
    r"(?:\[[^\]]+\])?"
    r"\{(?P<title>[^}]+)\}"
    r"(?:\[[^\]]+\])?"
)

_CANONICAL_TITLE_TO_ENV = {
    "theorem": "theorem",
    "theorems": "theorem",
    "thm": "theorem",
    "th": "theorem",
    "lemma": "lemma",
    "lemmas": "lemma",
    "lem": "lemma",
    "le": "lemma",
    "proposition": "proposition",
    "propositions": "proposition",
    "prop": "proposition",
    "corollary": "corollary",
    "corollaries": "corollary",
    "cor": "corollary",
    "co": "corollary",
    "definition": "definition",
    "definitions": "definition",
    "defn": "definition",
    "def": "definition",
    "df": "definition",
    "conjecture": "conjecture",
    "conjectures": "conjecture",
    "conj": "conjecture",
    "claim": "claim",
    "claims": "claim",
    "fact": "fact",
    "facts": "fact",
    "remark": "remark",
    "remarks": "remark",
    "rmk": "remark",
    "rem": "remark",
    "rk": "remark",
    "notation": "notation",
    "notations": "notation",
    "nota": "notation",
    "notn": "notation",
    "example": "example",
    "examples": "example",
    "ex": "example",
    "exa": "example",
    "exm": "example",
    "eks": "example",
    "assumption": "assumption",
    "assumptions": "assumption",
    "asm": "assumption",
    "proof": "proof",
    "proofs": "proof",
    "bevis": "proof",
    "demo": "proof",
    "dem": "proof",
}

_PARSED_BLOCK_ENVS = {
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "proof",
}

_SECTION_HEAD_RE = re.compile(r"\\(?:section|subsection|subsubsection)\*?\{")
_PARAGRAPH_HEAD_RE = re.compile(r"\\paragraph\{(?P<head>[^}]*)\}")


@dataclass(frozen=True)
class Rewrite:
    span_start: int
    span_end: int
    rewritten_span_start: int
    rewritten_span_end: int
    kind: str
    source_cue: str
    original_text: str
    rewritten_text: str
    metadata: dict[str, object] = field(default_factory=dict)
    block_annotation: dict[str, str] | None = None


@dataclass
class NormalizationResult:
    rewritten_text: str
    rewrites: list[Rewrite] = field(default_factory=list)
    alias_map: dict[str, str] = field(default_factory=dict)
    block_annotations: dict[int, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class _AliasSpec:
    alias: str
    canonical_env: str
    declaration_kind: str


@dataclass(frozen=True)
class _WrapperSpec:
    name: str
    kind: str
    fixed_env: str | None = None


@dataclass(frozen=True)
class _LetAliasSpec:
    alias: str
    target: str
    kind: str
    env: str | None = None
    decl_span: tuple[int, int] = (0, 0)


def _canonical_env_from_title(title: str) -> str | None:
    normalized = str(title or "")
    normalized = re.sub(r"\$[^$]*\$", " ", normalized)
    normalized = re.sub(r"\\[A-Za-z*]+", " ", normalized)
    normalized = re.sub(r"[^A-Za-z]+", " ", normalized).lower()
    tokens = [tok for tok in normalized.split() if tok]
    for tok in tokens:
        env = _CANONICAL_TITLE_TO_ENV.get(tok)
        if env:
            return env
    return None


def parse_newtheorem_aliases(text: str) -> dict[str, str]:
    """Return alias -> canonical environment for recognized \\newtheorem declarations."""
    alias_map: dict[str, str] = {}
    for match in _NEWTHEOREM_RE.finditer(text):
        alias = match.group("alias").strip()
        canonical = _canonical_env_from_title(match.group("title"))
        if not alias or not canonical:
            continue
        if alias == canonical:
            continue
        alias_map[alias] = canonical
    return alias_map


def _skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _extract_balanced(text: str, idx: int, open_ch: str, close_ch: str) -> tuple[str | None, int]:
    if idx >= len(text) or text[idx] != open_ch:
        return None, idx
    depth = 1
    i = idx + 1
    start = i
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return None, idx


def parse_newenvironment_aliases(text: str) -> dict[str, str]:
    """Return alias -> canonical environment for recognized \\newenvironment declarations."""
    alias_map: dict[str, str] = {}
    needle = r"\newenvironment"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        cursor = _skip_ws(text, idx + len(needle))
        alias, cursor = _extract_balanced(text, cursor, "{", "}")
        if alias is None:
            start = idx + len(needle)
            continue
        cursor = _skip_ws(text, cursor)
        if cursor < len(text) and text[cursor] == "[":
            _ignored, cursor = _extract_balanced(text, cursor, "[", "]")
            cursor = _skip_ws(text, cursor)
        begin_body, cursor = _extract_balanced(text, cursor, "{", "}")
        if begin_body is None:
            start = idx + len(needle)
            continue
        canonical = _canonical_env_from_title(begin_body)
        if alias and canonical and alias != canonical:
            alias_map[alias] = canonical
        start = idx + len(needle)
    return alias_map


def _heuristic_env_alias(alias: str) -> str | None:
    if not alias or not re.fullmatch(r"[A-Za-z]+", alias):
        return None
    canonical = _canonical_env_from_title(alias)
    if not canonical or canonical not in _PARSED_BLOCK_ENVS:
        return None
    if alias == canonical:
        return None
    return canonical


def parse_heuristic_env_aliases(text: str) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    declared_aliases = set(parse_newtheorem_aliases(text)) | set(parse_newenvironment_aliases(text))
    for alias in sorted(set(re.findall(r"\\begin\{([^}]+)\}", text))):
        if alias in declared_aliases:
            continue
        canonical = _heuristic_env_alias(alias)
        if canonical:
            alias_map[alias] = canonical
    return alias_map


def _collect_alias_specs(text: str) -> dict[str, _AliasSpec]:
    specs: dict[str, _AliasSpec] = {}
    for alias, canonical in parse_newtheorem_aliases(text).items():
        specs[alias] = _AliasSpec(
            alias=alias,
            canonical_env=canonical,
            declaration_kind="newtheorem",
        )
    for alias, canonical in parse_newenvironment_aliases(text).items():
        specs[alias] = _AliasSpec(
            alias=alias,
            canonical_env=canonical,
            declaration_kind="newenvironment",
        )
    for alias, canonical in parse_heuristic_env_aliases(text).items():
        specs.setdefault(
            alias,
            _AliasSpec(
                alias=alias,
                canonical_env=canonical,
                declaration_kind="envname-heuristic",
            ),
        )
    return specs


def _rewrite_begin_token(canonical_env: str, args: str) -> str:
    args = args or ""
    if args.startswith("["):
        return rf"\begin{{{canonical_env}}}{args}"
    if canonical_env == "proof" and args.startswith("{"):
        title, _ = _extract_balanced(args, 0, "{", "}")
        if title is not None and title.strip():
            return rf"\begin{{proof}}[{title.strip()}]"
    return rf"\begin{{{canonical_env}}}"


def _parse_newcommand_wrappers(text: str) -> dict[str, _WrapperSpec]:
    specs: dict[str, _WrapperSpec] = {}
    needle = r"\newcommand"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        cursor = _skip_ws(text, idx + len(needle))
        name = None
        if cursor < len(text) and text[cursor] == "{":
            raw_name, cursor = _extract_balanced(text, cursor, "{", "}")
            if raw_name:
                name = raw_name.strip().lstrip("\\")
        elif cursor < len(text) and text[cursor] == "\\":
            cursor += 1
            name_start = cursor
            while cursor < len(text) and text[cursor].isalpha():
                cursor += 1
            name = text[name_start:cursor]
        if not name:
            start = idx + len(needle)
            continue
        cursor = _skip_ws(text, cursor)
        argc = 0
        if cursor < len(text) and text[cursor] == "[":
            argc_text, cursor = _extract_balanced(text, cursor, "[", "]")
            try:
                argc = int(argc_text or "0")
            except ValueError:
                argc = 0
            cursor = _skip_ws(text, cursor)
        body, cursor = _extract_balanced(text, cursor, "{", "}")
        if body is None:
            start = idx + len(needle)
            continue
        compact = re.sub(r"\s+", "", body)
        spec = None
        if argc == 1 and compact == r"\begin{#1}":
            spec = _WrapperSpec(name=name, kind="generic_begin")
        elif argc == 2 and compact == r"\begin{#1}[#2]":
            spec = _WrapperSpec(name=name, kind="generic_begin_titled")
        elif argc == 1 and compact == r"\end{#1}":
            spec = _WrapperSpec(name=name, kind="generic_end")
        else:
            begin_noarg = re.fullmatch(r"\\begin\{([^}]+)\}", compact)
            begin_empty = re.fullmatch(r"\\begin\{([^}]+)\}\{\}", compact)
            begin_arg = re.fullmatch(r"\\begin\{([^}]+)\}\{#1\}", compact)
            end_fixed = re.fullmatch(r"\\end\{([^}]+)\}", compact)
            if argc == 0 and begin_noarg:
                spec = _WrapperSpec(
                    name=name,
                    kind="fixed_begin_no_arg",
                    fixed_env=begin_noarg.group(1),
                )
            elif argc == 0 and begin_empty:
                spec = _WrapperSpec(
                    name=name,
                    kind="fixed_begin_empty_arg",
                    fixed_env=begin_empty.group(1),
                )
            elif argc == 1 and begin_arg:
                spec = _WrapperSpec(
                    name=name,
                    kind="fixed_begin_with_arg",
                    fixed_env=begin_arg.group(1),
                )
            elif argc == 0 and end_fixed:
                spec = _WrapperSpec(
                    name=name,
                    kind="fixed_end",
                    fixed_env=end_fixed.group(1),
                )
        if spec:
            specs[name] = spec
        start = idx + len(needle)
    return specs


def _parse_let_aliases(text: str) -> dict[str, _LetAliasSpec]:
    specs: dict[str, _LetAliasSpec] = {}
    for match in re.finditer(r"\\let\\(?P<alias>[A-Za-z]+)\\(?P<target>[A-Za-z]+)", text):
        alias = match.group("alias")
        target = match.group("target")
        kind = None
        env = None
        if target in _PARSED_BLOCK_ENVS:
            kind = "begin"
            env = target
        elif target == "endproof":
            kind = "end"
            env = "proof"
        elif target == "endtheorem":
            kind = "generic_claim_end"
        elif target.startswith("end") and target[3:] in _PARSED_BLOCK_ENVS:
            kind = "end"
            env = target[3:]
        if kind:
            specs[alias] = _LetAliasSpec(
                alias=alias,
                target=target,
                kind=kind,
                env=env,
                decl_span=(match.start(), match.end()),
            )
    return specs


def _resolve_env_name(
    env: str,
    alias_specs: dict[str, _AliasSpec],
) -> tuple[str, dict[str, str] | None]:
    if env in alias_specs:
        spec = alias_specs[env]
        return spec.canonical_env, {
            "block_origin": "alias_expanded",
            "source_cue": f"{spec.declaration_kind} alias {env}->{spec.canonical_env}",
            "original_env": env,
            "canonical_env": spec.canonical_env,
        }
    canonical = _heuristic_env_alias(env)
    if canonical:
        return canonical, {
            "block_origin": "alias_expanded",
            "source_cue": f"envname-heuristic alias {env}->{canonical}",
            "original_env": env,
            "canonical_env": canonical,
        }
    return env, None


def _pop_last_matching(stack: list[str], predicate) -> str | None:
    for idx in range(len(stack) - 1, -1, -1):
        if predicate(stack[idx]):
            return stack.pop(idx)
    return None


def _plan_wrapper_rewrites(
    text: str,
    alias_specs: dict[str, _AliasSpec],
    *,
    paper_id: str,
) -> list[dict[str, str | int | dict[str, str]]]:
    planned: list[dict[str, str | int | dict[str, str]]] = []
    specs = _parse_newcommand_wrappers(text)
    for name, spec in specs.items():
        source_cue = f"paper={paper_id}; newcommand wrapper \\{name}"
        if spec.kind == "generic_begin":
            pattern = re.compile(rf"\\{re.escape(name)}\{{(?P<env>[^{{}}]+)\}}")
            for match in pattern.finditer(text):
                env = match.group("env").strip()
                resolved_env, block_annotation = _resolve_env_name(env, alias_specs)
                planned.append(
                    {
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "kind": "macro-expanded",
                        "source_cue": source_cue,
                        "original_text": match.group(0),
                        "rewritten_text": rf"\begin{{{resolved_env}}}",
                        "metadata": {
                            "wrapper": name,
                            "env": env,
                            "canonical_env": resolved_env,
                            "token": "begin",
                        },
                        "block_annotation": block_annotation,
                    }
                )
        elif spec.kind == "generic_begin_titled":
            pattern = re.compile(
                rf"\\{re.escape(name)}\{{(?P<env>[^{{}}]+)\}}\{{(?P<title>[^{{}}]*)\}}"
            )
            for match in pattern.finditer(text):
                env = match.group("env").strip()
                resolved_env, block_annotation = _resolve_env_name(env, alias_specs)
                planned.append(
                    {
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "kind": "macro-expanded",
                        "source_cue": source_cue,
                        "original_text": match.group(0),
                        "rewritten_text": (
                            rf"\begin{{{resolved_env}}}[{match.group('title').strip()}]"
                        ),
                        "metadata": {
                            "wrapper": name,
                            "env": env,
                            "canonical_env": resolved_env,
                            "token": "begin",
                        },
                        "block_annotation": block_annotation,
                    }
                )
        elif spec.kind == "generic_end":
            pattern = re.compile(rf"\\{re.escape(name)}\{{(?P<env>[^{{}}]+)\}}")
            for match in pattern.finditer(text):
                env = match.group("env").strip()
                resolved_env, _block_annotation = _resolve_env_name(env, alias_specs)
                planned.append(
                    {
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "kind": "macro-expanded",
                        "source_cue": source_cue,
                        "original_text": match.group(0),
                        "rewritten_text": rf"\end{{{resolved_env}}}",
                        "metadata": {
                            "wrapper": name,
                            "env": env,
                            "canonical_env": resolved_env,
                            "token": "end",
                        },
                    }
                )
        elif spec.kind == "fixed_begin_empty_arg" and spec.fixed_env:
            pattern = re.compile(rf"\\{re.escape(name)}(?![A-Za-z])")
            for match in pattern.finditer(text):
                resolved_env, block_annotation = _resolve_env_name(spec.fixed_env, alias_specs)
                planned.append(
                    {
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "kind": "macro-expanded",
                        "source_cue": source_cue,
                        "original_text": match.group(0),
                        "rewritten_text": _rewrite_begin_token(resolved_env, "{}"),
                        "metadata": {
                            "wrapper": name,
                            "env": spec.fixed_env,
                            "canonical_env": resolved_env,
                            "token": "begin",
                        },
                        "block_annotation": block_annotation,
                    }
                )
        elif spec.kind == "fixed_begin_no_arg" and spec.fixed_env:
            pattern = re.compile(rf"\\{re.escape(name)}(?![A-Za-z])")
            for match in pattern.finditer(text):
                resolved_env, block_annotation = _resolve_env_name(spec.fixed_env, alias_specs)
                planned.append(
                    {
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "kind": "macro-expanded",
                        "source_cue": source_cue,
                        "original_text": match.group(0),
                        "rewritten_text": rf"\begin{{{resolved_env}}}",
                        "metadata": {
                            "wrapper": name,
                            "env": spec.fixed_env,
                            "canonical_env": resolved_env,
                            "token": "begin",
                        },
                        "block_annotation": block_annotation,
                    }
                )
        elif spec.kind == "fixed_begin_with_arg" and spec.fixed_env:
            pattern = re.compile(rf"\\{re.escape(name)}\{{(?P<arg>[^{{}}]*)\}}")
            for match in pattern.finditer(text):
                resolved_env, block_annotation = _resolve_env_name(spec.fixed_env, alias_specs)
                planned.append(
                    {
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "kind": "macro-expanded",
                        "source_cue": source_cue,
                        "original_text": match.group(0),
                        "rewritten_text": _rewrite_begin_token(
                            resolved_env,
                            "{" + match.group("arg").strip() + "}",
                        ),
                        "metadata": {
                            "wrapper": name,
                            "env": spec.fixed_env,
                            "canonical_env": resolved_env,
                            "token": "begin",
                        },
                        "block_annotation": block_annotation,
                    }
                )
        elif spec.kind == "fixed_end" and spec.fixed_env:
            pattern = re.compile(rf"\\{re.escape(name)}(?![A-Za-z])")
            for match in pattern.finditer(text):
                resolved_env, _block_annotation = _resolve_env_name(spec.fixed_env, alias_specs)
                planned.append(
                    {
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "kind": "macro-expanded",
                        "source_cue": source_cue,
                        "original_text": match.group(0),
                        "rewritten_text": rf"\end{{{resolved_env}}}",
                        "metadata": {
                            "wrapper": name,
                            "env": spec.fixed_env,
                            "canonical_env": resolved_env,
                            "token": "end",
                        },
                    }
                )
    return planned


def _plan_let_alias_rewrites(
    text: str,
    *,
    paper_id: str,
) -> list[dict[str, str | int | dict[str, str]]]:
    planned: list[dict[str, str | int | dict[str, str]]] = []
    specs = _parse_let_aliases(text)
    if not specs:
        return planned
    body_start = 0
    document_match = re.search(r"\\begin\{document\}", text)
    if document_match:
        body_start = document_match.end()

    events: list[tuple[int, int, _LetAliasSpec]] = []
    for spec in specs.values():
        pattern = re.compile(rf"\\{re.escape(spec.alias)}(?![A-Za-z])")
        for match in pattern.finditer(text, body_start):
            if spec.decl_span[0] <= match.start() < spec.decl_span[1]:
                continue
            events.append((match.start(), match.end(), spec))
    events.sort(key=lambda item: (item[0], item[1]))

    open_stack: list[str] = []
    for span_start, span_end, spec in events:
        source_cue = f"paper={paper_id}; let alias \\{spec.alias}->\\{spec.target}"
        token = text[span_start:span_end]
        if spec.kind == "begin" and spec.env:
            open_stack.append(spec.env)
            planned.append(
                {
                    "span_start": span_start,
                    "span_end": span_end,
                    "kind": "alias-expanded",
                    "source_cue": source_cue,
                    "original_text": token,
                    "rewritten_text": rf"\begin{{{spec.env}}}",
                    "metadata": {
                        "alias": spec.alias,
                        "canonical_env": spec.env,
                        "token": "begin",
                    },
                    "block_annotation": {
                        "block_origin": "alias_expanded",
                        "source_cue": f"let alias {spec.alias}->{spec.target}",
                        "original_env": spec.alias,
                        "canonical_env": spec.env,
                    },
                }
            )
        elif spec.kind == "end" and spec.env:
            closed_env = _pop_last_matching(open_stack, lambda env: env == spec.env) or spec.env
            planned.append(
                {
                    "span_start": span_start,
                    "span_end": span_end,
                    "kind": "alias-expanded",
                    "source_cue": source_cue,
                    "original_text": token,
                    "rewritten_text": rf"\end{{{closed_env}}}",
                    "metadata": {
                        "alias": spec.alias,
                        "canonical_env": closed_env,
                        "token": "end",
                    },
                }
            )
        elif spec.kind == "generic_claim_end":
            closed_env = _pop_last_matching(
                open_stack,
                lambda env: env in _PARSED_BLOCK_ENVS and env != "proof",
            )
            if not closed_env:
                continue
            planned.append(
                {
                    "span_start": span_start,
                    "span_end": span_end,
                    "kind": "alias-expanded",
                    "source_cue": source_cue,
                    "original_text": token,
                    "rewritten_text": rf"\end{{{closed_env}}}",
                    "metadata": {
                        "alias": spec.alias,
                        "canonical_env": closed_env,
                        "token": "end",
                    },
                }
            )
    return planned


def _boundary_positions(text: str, start: int) -> list[int]:
    positions = [m.start() for m in _SECTION_HEAD_RE.finditer(text, start)]
    positions.extend(m.start() for m in _PARAGRAPH_HEAD_RE.finditer(text, start))
    for pat in (
        r"\\begin\{proof\}",
        r"\\begin\{prf\}",
        r"\\bpr(?![A-Za-z])",
        r"\\bprf\{",
    ):
        positions.extend(m.start() for m in re.finditer(pat, text[start:]))
    adjusted = []
    for pos in positions:
        adjusted.append(pos if pos >= start else start + pos)
    return [pos for pos in adjusted if pos > start]


def _plan_paragraph_synthesis(
    text: str,
    *,
    paper_id: str,
) -> list[dict[str, str | int | dict[str, str]]]:
    planned: list[dict[str, str | int | dict[str, str]]] = []
    document_match = re.search(r"\\begin\{document\}", text)
    if not document_match:
        return planned
    body_start = document_match.end()
    heads = [m for m in _PARAGRAPH_HEAD_RE.finditer(text) if m.start() >= body_start]
    for idx, match in enumerate(heads):
        head = match.group("head").strip()
        canonical = _canonical_env_from_title(head)
        if canonical not in _PARSED_BLOCK_ENVS:
            continue
        normalized_tokens = re.sub(r"[^A-Za-z]+", " ", head).lower().split()
        if len(normalized_tokens) != 1:
            continue
        body_start = match.end()
        next_positions = []
        if idx + 1 < len(heads):
            next_positions.append(heads[idx + 1].start())
        next_positions.extend(m.start() for m in _SECTION_HEAD_RE.finditer(text, body_start))
        for pat in (
            r"\\begin\{proof\}",
            r"\\begin\{prf\}",
            r"\\bpr(?![A-Za-z])",
            r"\\bprf\{",
        ):
            proof_match = re.search(pat, text[body_start:])
            if proof_match:
                next_positions.append(body_start + proof_match.start())
        block_end = min(next_positions) if next_positions else len(text)
        if block_end <= body_start:
            continue
        source_cue = f"paper={paper_id}; paragraph head {head}->{canonical}"
        planned.append(
            {
                "span_start": match.start(),
                "span_end": match.end(),
                "kind": "prose-synthesized",
                "source_cue": source_cue,
                "original_text": match.group(0),
                "rewritten_text": rf"\begin{{{canonical}}}",
                "metadata": {
                    "canonical_env": canonical,
                    "token": "begin",
                    "head": head,
                },
                "block_annotation": {
                    "block_origin": "prose_synthesized",
                    "source_cue": f"paragraph head {head}->{canonical}",
                    "original_env": head,
                    "canonical_env": canonical,
                },
            }
        )
        planned.append(
            {
                "span_start": block_end,
                "span_end": block_end,
                "kind": "prose-synthesized",
                "source_cue": source_cue,
                "original_text": "",
                "rewritten_text": rf"\end{{{canonical}}}",
                "metadata": {
                    "canonical_env": canonical,
                    "token": "end",
                    "head": head,
                },
            }
        )
    return planned


def normalize(text: str, *, paper_id: str) -> NormalizationResult:
    """Rewrite source-declared theorem aliases to canonical env names.

    This is intentionally conservative: only aliases declared by
    \\newtheorem / \\newenvironment and mapped to a known canonical
    theorem-like or proof-like environment are rewritten.
    """
    alias_specs = _collect_alias_specs(text)
    alias_map = {alias: spec.canonical_env for alias, spec in alias_specs.items()}
    if not alias_specs:
        alias_specs = {}

    planned_rewrites: list[dict[str, str | int | dict[str, str]]] = []
    planned_rewrites.extend(_plan_wrapper_rewrites(text, alias_specs, paper_id=paper_id))
    planned_rewrites.extend(_plan_let_alias_rewrites(text, paper_id=paper_id))
    for alias, spec in alias_specs.items():
        source_cue = (
            f"paper={paper_id}; {spec.declaration_kind} alias "
            f"{alias}->{spec.canonical_env}"
        )
        begin_re = re.compile(
            rf"\\begin\{{{re.escape(alias)}\}}(?P<args>(?:\[[^\]]*\])?(?:\{{[^{{}}]*\}})*)"
        )
        end_re = re.compile(rf"\\end\{{{re.escape(alias)}\}}")

        for match in begin_re.finditer(text):
            rewritten_begin = _rewrite_begin_token(
                spec.canonical_env,
                match.group("args") or "",
            )
            planned_rewrites.append(
                {
                    "span_start": match.start(),
                    "span_end": match.end(),
                    "kind": "alias-expanded",
                    "source_cue": source_cue,
                    "original_text": match.group(0),
                    "rewritten_text": rewritten_begin,
                    "metadata": {
                        "alias": alias,
                        "canonical_env": spec.canonical_env,
                        "token": "begin",
                    },
                    "block_annotation": {
                        "block_origin": "alias_expanded",
                        "source_cue": f"{spec.declaration_kind} alias {alias}->{spec.canonical_env}",
                        "original_env": alias,
                        "canonical_env": spec.canonical_env,
                    },
                }
            )

        for match in end_re.finditer(text):
            planned_rewrites.append(
                {
                    "span_start": match.start(),
                    "span_end": match.end(),
                    "kind": "alias-expanded",
                    "source_cue": source_cue,
                    "original_text": match.group(0),
                    "rewritten_text": rf"\end{{{spec.canonical_env}}}",
                    "metadata": {
                        "alias": alias,
                        "canonical_env": spec.canonical_env,
                        "token": "end",
                    },
                }
            )
    planned_rewrites.extend(_plan_paragraph_synthesis(text, paper_id=paper_id))

    if not planned_rewrites:
        return NormalizationResult(rewritten_text=text, alias_map=alias_map)

    planned_rewrites.sort(key=lambda r: (int(r["span_start"]), int(r["span_end"])))

    parts: list[str] = []
    cursor = 0
    rewritten_cursor = 0
    rewrites: list[Rewrite] = []
    for plan in planned_rewrites:
        span_start = int(plan["span_start"])
        span_end = int(plan["span_end"])
        if span_start < cursor:
            continue
        prefix = text[cursor:span_start]
        parts.append(prefix)
        rewritten_cursor += len(prefix)
        rewritten_text = str(plan["rewritten_text"])
        rewritten_span_start = rewritten_cursor
        rewritten_span_end = rewritten_span_start + len(rewritten_text)
        parts.append(rewritten_text)
        rewrites.append(
            Rewrite(
                span_start=span_start,
                span_end=span_end,
                rewritten_span_start=rewritten_span_start,
                rewritten_span_end=rewritten_span_end,
                kind=str(plan["kind"]),
                source_cue=str(plan["source_cue"]),
                original_text=str(plan["original_text"]),
                rewritten_text=rewritten_text,
                metadata=dict(plan["metadata"]),
                block_annotation=dict(plan["block_annotation"]) if plan.get("block_annotation") else None,
            )
        )
        rewritten_cursor = rewritten_span_end
        cursor = span_end

    parts.append(text[cursor:])
    rewritten = "".join(parts)

    block_annotations: dict[int, dict[str, str]] = {}
    for rewrite in rewrites:
        annotation = rewrite.block_annotation
        if not annotation:
            continue
        block_annotations[rewrite.rewritten_span_start] = dict(annotation)

    return NormalizationResult(
        rewritten_text=rewritten,
        rewrites=rewrites,
        alias_map=alias_map,
        block_annotations=block_annotations,
    )
