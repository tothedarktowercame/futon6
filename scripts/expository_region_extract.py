#!/usr/bin/env python3
r"""Extract expository regions from raw LaTeX paper text.

This is a heuristic structural extractor, not a full LaTeX parser.

Operationalization:

* The body starts at the first sectioning command after the abstract when an
  abstract is present; otherwise it starts after ``\maketitle`` or at the first
  sectioning command.  The preamble and abstract are ignored.
* A ``leaf-section`` is a sectioning span whose body has no deeper sectioning
  command and no formal block interval.  Formal blocks are selected theorem-like
  environments, definitions, proofs, displayed math, and list environments used
  as primary mathematical structure.
* An ``inflight`` region is a nonempty prose gap between two adjacent formal
  blocks within the same section and the same formal-parent depth.  The prose
  gap is therefore one markup level shallower than the content of both
  neighboring blocks: if both blocks begin while the formal stack has depth d,
  their content is at depth d + 1 and the gap sits at depth d.  This captures
  prose inside a proof after an ``enumerate`` block and before the next display
  or formal block, e.g. 0905.0595 lines 202--208.

* Given the paper's S1 marks (``extract_regions(..., marks)``, as the S4 candidate
  extractor calls it), formal blocks also include S1's ``env/*`` environments, which
  follow the author's ``\newtheorem`` and macros (``\df{...}``, ``\prf ... \eprf``)
  that the fixed environment list cannot see; each section's own blocks stop at its
  first subsection; a section's prose before its first formal block and after its
  last becomes a ``section-lead`` / ``section-tail`` region; prose inside an S1 proof
  is typed ``in-proof``; and a paragraph with fewer than three prose words is dropped.
  On 0705.0102 (every environment an author macro) this takes the carving from 1
  region, 4.8% of body lines, to 71 regions, 31%.

The extractor prefers recall for expository spans over perfect LaTeX fidelity.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import futon6_config as config


import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = config.marks()

SECTION_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}

SECTION_RE = re.compile(
    r"\\(?P<cmd>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
    # Titles may hold braces two deep ($S^{\perp_{\infty}}$, \ref{prop4}).
    r"\*?(?:\[[^\]]*\])?\{(?P<title>(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
)
BEGIN_ENV_RE = re.compile(r"\\begin\{([^{}]+)\}")
END_ENV_RE = re.compile(r"\\end\{([^{}]+)\}")

FORMAL_ENVS = {
    "theorem",
    "theorem*",
    "thm",
    "thm*",
    "theo",
    "theo*",
    "lemma",
    "lemma*",
    "lem",
    "lem*",
    "proposition",
    "proposition*",
    "prop",
    "prop*",
    "propo",
    "propo*",
    "corollary",
    "corollary*",
    "coro",
    "coro*",
    "definition",
    "definition*",
    "defn",
    "defn*",
    "def",
    "def*",
    "remark",
    "remark*",
    "rem",
    "rem*",
    "proof",
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "eqnarray",
    "eqnarray*",
    "enumerate",
    "itemize",
    "description",
}

DISPLAYISH_KINDS = {
    "display",
    "dollar-display",
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "eqnarray",
    "eqnarray*",
}

LEAF_PROSE_TITLES_RE = re.compile(r"\b(introduction|motivation|conclusion|conclusions)\b", re.I)


@dataclass(frozen=True)
class Section:
    line_start: int
    line_end: int
    level: int
    title: str
    has_deeper_section: bool


@dataclass(frozen=True)
class FormalBlock:
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    kind: str
    parent_depth: int


def normalize_env(name: str) -> str:
    return name.strip()


def strip_tex_comments(line: str) -> str:
    out = []
    escaped = False
    for ch in line:
        if ch == "%" and not escaped:
            break
        out.append(ch)
        escaped = ch == "\\" and not escaped
        if ch != "\\":
            escaped = False
    return "".join(out)


def line_offsets(text: str) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        starts.append(pos)
        pos += len(line)
        ends.append(pos)
    if text and not text.endswith(("\n", "\r")):
        ends[-1] = len(text)
    return starts, ends


def load_text(input_value: str) -> tuple[str, str]:
    path = Path(input_value)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        entity_id = str(payload.get("paper") or path.stem)
        if entity_id.endswith("-dp"):
            entity_id = entity_id[:-3]
        return entity_id, str(payload["text"])

    paper_id = input_value
    path = GOLDEN_DIR / f"fable-{paper_id}-dp-emacs.json"
    if not path.exists():
        raise FileNotFoundError(f"cannot find id or path: {input_value}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entity_id = str(payload.get("paper") or paper_id)
    if entity_id.endswith("-dp"):
        entity_id = entity_id[:-3]
    return entity_id, str(payload["text"])


def find_body_range(lines: list[str]) -> tuple[int, int]:
    abstract_end = 0
    in_abstract = False
    for idx, line in enumerate(lines, 1):
        if re.search(r"\\begin\{abstract\}|\\abstract\b", line):
            in_abstract = True
        if in_abstract and re.search(r"\\end\{abstract\}", line):
            abstract_end = idx
            break

    first_section = 0
    for idx, line in enumerate(lines, 1):
        if idx <= abstract_end:
            continue
        if SECTION_RE.search(line):
            first_section = idx
            break

    if first_section:
        start = first_section
    else:
        # No sectioning: the body still never includes the PREAMBLE. Floor the
        # body at \begin{document} (and after any abstract), then prefer the
        # line after \maketitle if present.
        doc_start = 0
        for idx, line in enumerate(lines, 1):
            if r"\begin{document}" in line:
                doc_start = idx
                break
        base = max(abstract_end, doc_start)
        maketitle = 0
        for idx, line in enumerate(lines, 1):
            if idx <= base:
                continue
            if r"\maketitle" in line:
                maketitle = idx
                break
        start = maketitle + 1 if maketitle else max(1, base + 1)

    end = len(lines)
    for idx in range(start, len(lines) + 1):
        if re.search(r"\\end\{document\}", lines[idx - 1]):
            end = idx - 1
            break
    return start, end


def parse_sections(lines: list[str], body_start: int, body_end: int) -> list[Section]:
    found: list[tuple[int, int, str]] = []
    for idx in range(body_start, body_end + 1):
        m = SECTION_RE.search(lines[idx - 1])
        if not m:
            continue
        found.append((idx, SECTION_LEVELS[m.group("cmd")], m.group("title").strip()))

    sections: list[Section] = []
    if not found:
        return [Section(body_start, body_end, SECTION_LEVELS["section"], "Body", False)]
    for pos, (line_start, level, title) in enumerate(found):
        line_end = body_end
        has_deeper = False
        for next_line, next_level, _ in found[pos + 1 :]:
            if next_level > level:
                has_deeper = True
                continue
            if next_level <= level:
                line_end = next_line - 1
                break
        sections.append(Section(line_start, line_end, level, title, has_deeper))
    return sections


def env_events(line: str) -> list[tuple[int, str, str]]:
    events: list[tuple[int, str, str]] = []
    clean = strip_tex_comments(line)
    for m in BEGIN_ENV_RE.finditer(clean):
        events.append((m.start(), "begin", normalize_env(m.group(1))))
    for m in END_ENV_RE.finditer(clean):
        events.append((m.start(), "end", normalize_env(m.group(1))))
    # Treat display delimiters as formal blocks.  They are sorted with env
    # events, which is enough for line/block cuts even if mixed with prose.
    for token in (r"\[", r"\]"):
        start = 0
        while True:
            pos = clean.find(token, start)
            if pos < 0:
                break
            events.append((pos, "display-open" if token == r"\[" else "display-close", token))
            start = pos + len(token)
    start = 0
    while True:
        pos = clean.find("$$", start)
        if pos < 0:
            break
        events.append((pos, "dollar-display", "$$"))
        start = pos + 2
    return sorted(events, key=lambda item: item[0])


def parse_formal_blocks(
    lines: list[str], starts: list[int], ends: list[int], body_start: int, body_end: int
) -> list[FormalBlock]:
    stack: list[dict[str, Any]] = []
    blocks: list[FormalBlock] = []
    display_stack: list[dict[str, Any]] = []

    for idx in range(body_start, body_end + 1):
        line = lines[idx - 1]
        for col, event, name in env_events(line):
            if event == "begin":
                if name in FORMAL_ENVS:
                    stack.append(
                        {
                            "name": name,
                            "line_start": idx,
                            "char_start": starts[idx - 1] + col,
                            "parent_depth": len(stack),
                        }
                    )
            elif event == "end":
                match_at = None
                for pos in range(len(stack) - 1, -1, -1):
                    if stack[pos]["name"] == name:
                        match_at = pos
                        break
                if match_at is not None:
                    opened = stack.pop(match_at)
                    blocks.append(
                        FormalBlock(
                            line_start=opened["line_start"],
                            line_end=idx,
                            char_start=opened["char_start"],
                            char_end=ends[idx - 1],
                            kind=opened["name"],
                            parent_depth=opened["parent_depth"],
                        )
                    )
            elif event == "display-open":
                display_stack.append(
                    {
                        "line_start": idx,
                        "char_start": starts[idx - 1] + col,
                        "parent_depth": len(stack),
                        "kind": "display",
                    }
                )
            elif event == "display-close":
                if display_stack:
                    opened = display_stack.pop()
                    blocks.append(
                        FormalBlock(
                            line_start=opened["line_start"],
                            line_end=idx,
                            char_start=opened["char_start"],
                            char_end=ends[idx - 1],
                            kind=opened["kind"],
                            parent_depth=opened["parent_depth"],
                        )
                    )
            elif event == "dollar-display":
                if display_stack and display_stack[-1]["kind"] == "dollar-display":
                    opened = display_stack.pop()
                    blocks.append(
                        FormalBlock(
                            line_start=opened["line_start"],
                            line_end=idx,
                            char_start=opened["char_start"],
                            char_end=ends[idx - 1],
                            kind="dollar-display",
                            parent_depth=opened["parent_depth"],
                        )
                    )
                else:
                    display_stack.append(
                        {
                            "line_start": idx,
                            "char_start": starts[idx - 1] + col,
                            "parent_depth": len(stack),
                            "kind": "dollar-display",
                        }
                    )

    # Close unbalanced formal environments at body end rather than failing.
    while stack:
        opened = stack.pop()
        blocks.append(
            FormalBlock(
                line_start=opened["line_start"],
                line_end=body_end,
                char_start=opened["char_start"],
                char_end=ends[body_end - 1],
                kind=opened["name"],
                parent_depth=opened["parent_depth"],
            )
        )
    while display_stack:
        opened = display_stack.pop()
        blocks.append(
            FormalBlock(
                line_start=opened["line_start"],
                line_end=body_end,
                char_start=opened["char_start"],
                char_end=ends[body_end - 1],
                kind=opened["kind"],
                parent_depth=opened["parent_depth"],
            )
        )

    return sorted(blocks, key=lambda block: (block.line_start, block.line_end, block.parent_depth))


def has_content(line: str) -> bool:
    stripped = strip_tex_comments(line).strip()
    if not stripped:
        return False
    if stripped in {r"\noindent", r"\par"}:
        return False
    if re.fullmatch(r"\\label\{[^{}]+\}", stripped):
        return False
    if re.fullmatch(r"\\(smallskip|medskip|bigskip|quad|qquad)\*?", stripped):
        return False
    return bool(re.search(r"[A-Za-z]", stripped))


def trim_content_lines(lines: list[str], line_start: int, line_end: int) -> tuple[int, int] | None:
    while line_start <= line_end and not has_content(lines[line_start - 1]):
        line_start += 1
    while line_end >= line_start and not has_content(lines[line_end - 1]):
        line_end -= 1
    if line_start > line_end:
        return None
    return line_start, line_end


def prose_paragraph_ranges(lines: list[str], line_start: int, line_end: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for idx in range(line_start, line_end + 1):
        if has_content(lines[idx - 1]):
            if current_start is None:
                current_start = idx
            current_end = idx
            continue
        if current_start is not None and current_end is not None:
            ranges.append((current_start, current_end))
            current_start = None
            current_end = None
    if current_start is not None and current_end is not None:
        ranges.append((current_start, current_end))
    return ranges


def region_text_and_chars(
    text: str, lines: list[str], starts: list[int], ends: list[int], line_start: int, line_end: int
) -> tuple[int, int, str]:
    raw_start = starts[line_start - 1]
    raw_end = ends[line_end - 1]
    chunk = text[raw_start:raw_end]
    left_trim = len(chunk) - len(chunk.lstrip())
    right_trim = len(chunk.rstrip())
    char_start = raw_start + left_trim
    char_end = raw_start + right_trim
    return char_start, char_end, text[char_start:char_end]


def section_for_line(sections: list[Section], line: int) -> Section | None:
    for section in sections:
        if section.line_start <= line <= section.line_end:
            return section
    return None


def overlaps_any_block(
    blocks: list[FormalBlock], line_start: int, line_end: int, min_parent_depth: int = 0
) -> bool:
    for block in blocks:
        if block.parent_depth < min_parent_depth:
            continue
        if block.line_start <= line_end and line_start <= block.line_end:
            return True
    return False


def immediate_parent_block(blocks: list[FormalBlock], child: FormalBlock) -> FormalBlock | None:
    candidates = [
        block
        for block in blocks
        if block.parent_depth == child.parent_depth - 1
        and block.line_start <= child.line_start
        and child.line_end <= block.line_end
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda block: (block.parent_depth, block.line_start))


def make_region(
    region_id: str,
    region_type: str,
    section_title: str,
    line_start: int,
    line_end: int,
    text: str,
    lines: list[str],
    starts: list[int],
    ends: list[int],
) -> dict[str, Any]:
    char_start, char_end, region_text = region_text_and_chars(
        text, lines, starts, ends, line_start, line_end
    )
    return {
        "region_id": region_id,
        "type": region_type,
        "section_title": section_title,
        "line_start": line_start,
        "line_end": line_end,
        "char_start": char_start,
        "char_end": char_end,
        "text": region_text,
    }


def is_label_or_blank(line: str) -> bool:
    stripped = strip_tex_comments(line).strip()
    return not stripped or bool(re.fullmatch(r"\\label\{[^{}]+\}", stripped))


def line_overlaps_displayish(blocks: list[FormalBlock], line: int) -> bool:
    return any(
        block.kind in DISPLAYISH_KINDS and block.line_start <= line <= block.line_end
        for block in blocks
    )


def gap_is_math_markup(
    lines: list[str],
    blocks: list[FormalBlock],
    line_start: int,
    line_end: int,
    require_substantive: bool = False,
) -> bool:
    if line_start > line_end:
        return not require_substantive
    if line_end - line_start + 1 > 4:
        return False
    substantive = False
    for idx in range(line_start, line_end + 1):
        stripped = strip_tex_comments(lines[idx - 1]).strip()
        if is_label_or_blank(lines[idx - 1]):
            continue
        if line_overlaps_displayish(blocks, idx):
            substantive = True
            continue
        if re.fullmatch(r"(\\\[|\\\]|\$\$)", stripped):
            substantive = True
            continue
        if re.fullmatch(r"\\(begin|end)\{[^{}]+\}", stripped):
            substantive = True
            continue
        return False
    return substantive or not require_substantive


def has_sentence_terminal(text: str) -> bool:
    stripped = text.rstrip()
    return bool(re.search(r"[.!?](?:[`'\"}\])]*|\s*)$", stripped))


def has_any_sentence_terminal(text: str) -> bool:
    return bool(re.search(r"[.!?]", text))


def starts_sentence(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped.startswith("\\"):
        return True
    return bool(re.match(r"[A-Z0-9({\\$]", stripped))


def rebuild_region(
    old: dict[str, Any],
    line_start: int,
    line_end: int,
    text: str,
    lines: list[str],
    starts: list[int],
    ends: list[int],
) -> dict[str, Any]:
    rebuilt = dict(old)
    char_start, char_end, region_text = region_text_and_chars(
        text, lines, starts, ends, line_start, line_end
    )
    rebuilt.update(
        {
            "line_start": line_start,
            "line_end": line_end,
            "char_start": char_start,
            "char_end": char_end,
            "text": region_text,
        }
    )
    return rebuilt


def coalesce_inflight_regions(
    regions: list[dict[str, Any]],
    text: str,
    lines: list[str],
    starts: list[int],
    ends: list[int],
    blocks: list[FormalBlock],
) -> list[dict[str, Any]]:
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: (r["line_start"], r["line_end"], r["type"]))
    out: list[dict[str, Any]] = []
    current = ordered[0]
    for nxt in ordered[1:]:
        can_merge = (
            current["type"] == "inflight"
            and nxt["type"] == "inflight"
            and current["section_title"] == nxt["section_title"]
            and gap_is_math_markup(
                lines,
                blocks,
                current["line_end"] + 1,
                nxt["line_start"] - 1,
                require_substantive=True,
            )
        )
        if can_merge:
            current = rebuild_region(
                current,
                current["line_start"],
                nxt["line_end"],
                text,
                lines,
                starts,
                ends,
            )
        else:
            out.append(current)
            current = nxt
    out.append(current)
    return out


def expand_unterminated_sentence_starts(
    regions: list[dict[str, Any]],
    text: str,
    lines: list[str],
    starts: list[int],
    ends: list[int],
    blocks: list[FormalBlock],
    body_end: int,
) -> list[dict[str, Any]]:
    ordered = sorted(regions, key=lambda r: (r["line_start"], r["line_end"], r["type"]))
    out: list[dict[str, Any]] = []
    for idx, region in enumerate(ordered):
        if (
            region["type"] != "inflight"
            or has_any_sentence_terminal(region["text"])
            or not starts_sentence(region["text"])
        ):
            out.append(region)
            continue
        next_start = ordered[idx + 1]["line_start"] if idx + 1 < len(ordered) else body_end + 1
        limit = min(next_start - 1, region["line_end"] + 12, body_end)
        new_end = region["line_end"]
        cursor = region["line_end"] + 1
        while cursor <= limit:
            if has_content(lines[cursor - 1]) or line_overlaps_displayish(blocks, cursor):
                new_end = cursor
                if re.search(r"[.!?]", strip_tex_comments(lines[cursor - 1])):
                    break
            elif not is_label_or_blank(lines[cursor - 1]):
                break
            cursor += 1
        if new_end != region["line_end"]:
            region = rebuild_region(
                region, region["line_start"], new_end, text, lines, starts, ends
            )
        out.append(region)
    return out


def fold_bare_fragments(
    regions: list[dict[str, Any]],
    text: str,
    lines: list[str],
    starts: list[int],
    ends: list[int],
    blocks: list[FormalBlock],
) -> list[dict[str, Any]]:
    ordered = sorted(regions, key=lambda r: (r["line_start"], r["line_end"], r["type"]))
    out: list[dict[str, Any]] = []
    for region in ordered:
        bare = region["type"] == "inflight" and (
            not starts_sentence(region["text"]) or not has_sentence_terminal(region["text"])
        )
        if (
            bare
            and out
            and out[-1]["type"] == "inflight"
            and out[-1]["section_title"] == region["section_title"]
            and gap_is_math_markup(
                lines,
                blocks,
                out[-1]["line_end"] + 1,
                region["line_start"] - 1,
                require_substantive=True,
            )
        ):
            out[-1] = rebuild_region(
                out[-1],
                out[-1]["line_start"],
                region["line_end"],
                text,
                lines,
                starts,
                ends,
            )
            continue
        out.append(region)
    return out


def renumber_regions(entity_id: str, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {"leaf-section": 0, "inflight": 0}
    renumbered: list[dict[str, Any]] = []
    for region in sorted(regions, key=lambda r: (r["line_start"], r["line_end"], r["type"])):
        kind = region["type"]
        counts[kind] = counts.get(kind, 0) + 1
        prefix = "leaf" if kind == "leaf-section" else kind
        updated = dict(region)
        updated["region_id"] = f"{entity_id}-{prefix}-{counts[kind]:04d}"
        renumbered.append(updated)
    return renumbered


def cleanup_regions(
    entity_id: str,
    regions: list[dict[str, Any]],
    text: str,
    lines: list[str],
    starts: list[int],
    ends: list[int],
    blocks: list[FormalBlock],
    body_end: int,
) -> list[dict[str, Any]]:
    cleaned = coalesce_inflight_regions(regions, text, lines, starts, ends, blocks)
    cleaned = expand_unterminated_sentence_starts(
        cleaned, text, lines, starts, ends, blocks, body_end
    )
    cleaned = coalesce_inflight_regions(cleaned, text, lines, starts, ends, blocks)
    cleaned = fold_bare_fragments(cleaned, text, lines, starts, ends, blocks)
    cleaned = coalesce_inflight_regions(cleaned, text, lines, starts, ends, blocks)
    return renumber_regions(entity_id, cleaned)


# S1's env/* kinds that are formal blocks, by canonical name (dp_paper_view maps an
# author's \\newtheorem and macro environments onto these).
# The bibliography and posed questions are not exposition either, so they count as blocks.
S1_FORMAL_KINDS = {"theorem", "lemma", "proposition", "corollary", "definition", "remark", "proof",
                   "conjecture", "claim", "notation", "assumption", "hypothesis", "axiom",
                   "question", "problem", "thebibliography"}


def own_span_end(sections: list[Section], pos: int) -> int:
    """Where a section's own text ends: at its first subsection, else at its end."""
    section = sections[pos]
    for later in sections[pos + 1:]:
        if later.line_start > section.line_start and later.level > section.level:
            return later.line_start - 1
    return section.line_end


PROSE_ARGS = re.compile(r"\\(?:label|ref|eqref|cite|begin|end|newtheorem|bibitem)\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})*")


def prose_words(text: str) -> int:
    """Words of running prose, with math, labels, references and bare commands removed."""
    text = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", " ", text, flags=re.S)
    text = re.sub(r"\\\\[A-Za-z]+", " ", PROSE_ARGS.sub(" ", text))
    return len(re.findall(r"[A-Za-z]{2,}", text))


def blocks_from_marks(marks: list[dict[str, Any]], starts: list[int], ends: list[int],
                      body_start: int, body_end: int) -> list[FormalBlock]:
    """Formal blocks as S1 found them, including environments opened by an author's
    macro (\\df{...}, \\prf ... \\eprf) that the fixed FORMAL_ENVS list cannot see."""
    def line_of(offset: int) -> int:
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= offset:
                lo = mid + 1
            else:
                hi = mid
        return lo
    spans = sorted({(m["start"], m["end"], m["kind"][4:]) for m in marks
                    if str(m.get("kind", "")).startswith("env/") and m["kind"][4:] in S1_FORMAL_KINDS},
                   key=lambda x: (x[0], -x[1]))
    out = []
    for a, b, kind in spans:
        lo, hi = line_of(a), line_of(b - 1)
        if hi < body_start or lo > body_end:
            continue
        depth = sum(1 for x, y, _ in spans if x <= a < y and (y - x) > (b - a))
        out.append(FormalBlock(lo, hi, a, b, kind, depth))
    return out


def merge_blocks(found: list[FormalBlock], from_marks: list[FormalBlock]) -> list[FormalBlock]:
    """Union of the two detections; a block's depth counts every block that encloses it."""
    seen = {(b.char_start, b.char_end) for b in found}
    blocks = list(found) + [b for b in from_marks if (b.char_start, b.char_end) not in seen]
    # A block is inside another when it starts inside it: a parsed display ends at the
    # end of its line, so on a line shared with \\eprf it runs past the proof's end.
    def inside(b: FormalBlock, o: FormalBlock) -> bool:
        return o.char_start <= b.char_start < o.char_end and (o.char_end - o.char_start) > (b.char_end - b.char_start)
    def depth(b: FormalBlock) -> int:
        return sum(1 for o in blocks if o is not b and inside(b, o))
    blocks = [FormalBlock(b.line_start, b.line_end, b.char_start, b.char_end, b.kind, depth(b)) for b in blocks]
    return sorted(blocks, key=lambda block: (block.line_start, block.line_end, block.parent_depth))


def extract_regions(entity_id: str, text: str, marks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Expository regions of `text`. Given the paper's S1 marks, formal blocks include
    the environments S1 found (author macros included), and a section's prose before
    its first formal block and after its last becomes section-lead / section-tail."""
    lines = text.splitlines()
    starts, ends = line_offsets(text)
    body_start, body_end = find_body_range(lines)
    sections = parse_sections(lines, body_start, body_end)
    blocks = parse_formal_blocks(lines, starts, ends, body_start, body_end)
    if marks is not None:
        blocks = merge_blocks(blocks, blocks_from_marks(marks, starts, ends, body_start, body_end))
    regions: list[dict[str, Any]] = []
    leaf_section_ranges: list[tuple[int, int]] = []

    region_num = 1
    for section in sections:
        section_blocks = [
            block
            for block in blocks
            if section.line_start <= block.line_start <= section.line_end
        ]
        display_only_intro = (
            bool(LEAF_PROSE_TITLES_RE.search(section.title))
            and section_blocks
            and all(block.kind in DISPLAYISH_KINDS for block in section_blocks)
        )
        if section.has_deeper_section or (section_blocks and not display_only_intro):
            continue
        trimmed = trim_content_lines(lines, section.line_start, section.line_end)
        if not trimmed:
            continue
        line_start, line_end = trimmed
        leaf_section_ranges.append((section.line_start, section.line_end))
        regions.append(
            make_region(
                f"{entity_id}-leaf-{region_num:04d}",
                "leaf-section",
                section.title,
                line_start,
                line_end,
                text,
                lines,
                starts,
                ends,
            )
        )
        region_num += 1

    # Inflight regions: adjacent formal blocks at the same formal-parent depth.
    seen: set[tuple[int, int, str]] = set()
    for pos, section in enumerate(sections):
        if any(start <= section.line_start and section.line_end <= end for start, end in leaf_section_ranges):
            continue
        # With S1 marks, a section's blocks are its own, not its subsections': otherwise
        # each gap is emitted once per enclosing section, and gaps span subsection headings.
        span_end = own_span_end(sections, pos) if marks is not None else section.line_end
        section_blocks = [
            block
            for block in blocks
            if section.line_start <= block.line_start <= span_end
        ]
        by_depth: dict[int, list[FormalBlock]] = {}
        for block in section_blocks:
            by_depth.setdefault(block.parent_depth, []).append(block)
        for depth, depth_blocks in sorted(by_depth.items()):
            depth_blocks = sorted(depth_blocks, key=lambda block: (block.line_start, block.line_end))
            for left, right in zip(depth_blocks, depth_blocks[1:]):
                if depth == 0 and left.kind in DISPLAYISH_KINDS and right.kind in DISPLAYISH_KINDS:
                    continue
                if depth > 0:
                    left_parent = immediate_parent_block(section_blocks, left)
                    right_parent = immediate_parent_block(section_blocks, right)
                    if left_parent is None or right_parent is None or left_parent != right_parent:
                        continue
                if left.line_end >= right.line_start:
                    continue
                gap_start = left.line_end + 1
                gap_end = right.line_start - 1
                trimmed = trim_content_lines(lines, gap_start, gap_end)
                if not trimmed:
                    continue
                line_start, line_end = trimmed
                for para_start, para_end in prose_paragraph_ranges(lines, line_start, line_end):
                    if overlaps_any_block(section_blocks, para_start, para_end, depth):
                        continue
                    key = (para_start, para_end, section.title)
                    if key in seen:
                        continue
                    seen.add(key)
                    regions.append(
                        make_region(
                            f"{entity_id}-inflight-{region_num:04d}",
                            "inflight",
                            section.title,
                            para_start,
                            para_end,
                            text,
                            lines,
                            starts,
                            ends,
                        )
                    )
                    region_num += 1

    # Section lead and tail (with S1 marks only): the prose a section opens with before
    # its first formal block ("We recall the following definition from [E].") and ends
    # with after its last, which neither rule above reaches. A section's own span stops
    # at its first subsection; blocks count at depth 0 only.
    if marks is not None:
        covered = [(r["line_start"], r["line_end"]) for r in regions]
        for pos, section in enumerate(sections):
            if any(start <= section.line_start and section.line_end <= end for start, end in leaf_section_ranges):
                continue
            own_end = own_span_end(sections, pos)
            top = [b for b in blocks if b.parent_depth == 0 and section.line_start < b.line_start <= own_end]
            spans = []
            if top:
                spans.append(("section-lead", section.line_start + 1, top[0].line_start - 1))
                tail_from = max(b.line_end for b in top) + 1
                spans.append(("section-tail", tail_from, own_end))
            elif section.has_deeper_section:
                spans.append(("section-lead", section.line_start + 1, own_end))
            for kind, lo, hi in spans:
                trimmed = trim_content_lines(lines, lo, hi) if lo <= hi else None
                if not trimmed:
                    continue
                for para_start, para_end in prose_paragraph_ranges(lines, *trimmed):
                    if overlaps_any_block(blocks, para_start, para_end, 0) or any(
                            a <= para_start and para_end <= b for a, b in covered):
                        continue
                    regions.append(make_region(f"{entity_id}-{kind}-{region_num:04d}", kind, section.title,
                                               para_start, para_end, text, lines, starts, ends))
                    region_num += 1

    # SCAFFOLD-LESS FALLBACK (Joe's definition): a paper with no sectioning
    # markup is, by definition, one top-level expository section — so every body
    # prose paragraph not inside a formal/display block is expository. This
    # rescues short notes (e.g. 1005.2653) that carry the argument in running
    # prose between display equations, which the depth-0 display<->display
    # inflight rule above deliberately skips.
    has_real_sections = any(
        SECTION_RE.search(lines[i - 1]) for i in range(body_start, body_end + 1)
    )
    if not has_real_sections:
        # A paper with no sectioning is ONE top-level expository section (Joe):
        # emit a SINGLE region spanning the content prose, so the expository
        # (green) scope is CONTINUOUS across the whole passage and the finer
        # scopes (reasoning, displays, environments) nest INSIDE it — rather than
        # fragmenting into a region per paragraph. Front/back matter (centered
        # title/author/date/address, the bibliography) is excluded, and the span
        # runs from the first real-prose line to the last.
        excluded: set[int] = set()
        for env in ("center", "thebibliography", "tabular"):
            depth, opn = 0, None
            for idx in range(body_start, body_end + 1):
                if re.search(r"\\begin\{" + env + r"\}", lines[idx - 1]):
                    if depth == 0:
                        opn = idx
                    depth += 1
                if re.search(r"\\end\{" + env + r"\}", lines[idx - 1]) and depth:
                    depth -= 1
                    if depth == 0 and opn is not None:
                        excluded.update(range(opn, idx + 1))
                        opn = None

        def _is_prose(ln):
            return (has_content(lines[ln - 1])
                    and len(re.sub(r"[^A-Za-z]", "",
                                   re.sub(r"\\[a-zA-Z]+", "", lines[ln - 1]))) >= 3)
        content = [idx for idx in range(body_start, body_end + 1)
                   if idx not in excluded and _is_prose(idx)]
        if content:
            regions = [r for r in regions
                       if not (content[0] <= r["line_start"] and r["line_end"] <= content[-1])]
            regions.append(make_region(
                f"{entity_id}-leaf-{region_num:04d}", "leaf-section", "Body",
                content[0], content[-1], text, lines, starts, ends))
            region_num += 1
        regions.sort(key=lambda r: (r["line_start"], r["line_end"]))

    if marks is not None:
        # A paragraph of macros (\\eeg, \\esetup \\lemma \\label{..}) or of bare formulae
        # gives a scope nothing to read.
        regions = [r for r in regions if prose_words(r["text"]) >= 3]
    regions = cleanup_regions(entity_id, regions, text, lines, starts, ends, blocks, body_end)
    if marks is not None:
        # Prose between displays inside a proof stays a region (the inflight rule wants
        # it), but is typed apart: S3 reads proofs, so S4 can tell exposition from steps.
        proofs = [b for b in blocks if b.kind == "proof"]
        for r in regions:
            if any(b.char_start <= r["char_start"] and r["char_end"] <= b.char_end for b in proofs):
                r["type"] = "in-proof"
        regions = renumber_regions(entity_id, regions)
    expository_lines: set[int] = set()
    for region in regions:
        expository_lines.update(range(region["line_start"], region["line_end"] + 1))
    body_lines = max(0, body_end - body_start + 1)
    pct = round((len(expository_lines) / body_lines * 100.0) if body_lines else 0.0, 2)
    return {
        "entity_id": entity_id,
        "body_line_range": [body_start, body_end],
        "regions": regions,
        "coverage": {
            "expository_lines": len(expository_lines),
            "body_lines": body_lines,
            "pct": pct,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="paper id, e.g. 0905.0595, or fable JSON path")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entity_id, text = load_text(args.input)
    result = extract_regions(entity_id, text)
    if args.compact:
        json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
