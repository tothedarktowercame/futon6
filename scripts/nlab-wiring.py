#!/usr/bin/env python3
"""nLab CT-Backed Wiring Extraction Pipeline.

Extracts category-theory-aware wiring diagrams from nLab pages:
  - Structural environments (definitions, theorems, proofs, remarks, examples)
  - Wiki links typed by enclosing environment
  - Commutative diagrams (tikzcd) as structured records
  - Discourse wiring (scopes, wires, ports) within environments
  - N-ary categorical hyperedges (adjunction, Kan extension, monad, etc.)

Usage:
    python scripts/nlab-wiring.py extract [--pages-dir ...] [--limit N] [--output-dir ...]
    python scripts/nlab-wiring.py reference [--input-dir ...] [--output ...]
    python scripts/nlab-wiring.py evaluate [--reference ...] [--answers ...] [--output ...]

Memory-safe throughout. CPU-only, laptop-scale.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Constants
# ============================================================

NLAB_PAGES = Path(os.environ.get("NLAB_PAGES",
    str(Path(__file__).parent.parent.parent / "nlab-content" / "pages")))
NER_KERNEL = Path("data/ner-kernel/terms.tsv")

# Environment type mapping: CSS class / LaTeX env → canonical type
ENV_WIKI_STYLE = {
    ".num_defn": "env/definition",
    ".num_theorem": "env/theorem",
    ".num_prop": "env/proposition",
    ".num_lemma": "env/lemma",
    ".num_cor": "env/corollary",
    ".num_remark": "env/remark",
    ".num_example": "env/example",
    ".num_note": "env/note",
    ".proof": "env/proof",
    ".un_defn": "env/definition",
    ".un_theorem": "env/theorem",
    ".un_prop": "env/proposition",
    ".un_lemma": "env/lemma",
    ".un_remark": "env/remark",
    ".un_example": "env/example",
    ".standout": "env/remark",
}

ENV_LATEX_STYLE = {
    "defn": "env/definition",
    "definition": "env/definition",
    "theorem": "env/theorem",
    "thm": "env/theorem",
    "prop": "env/proposition",
    "proposition": "env/proposition",
    "lemma": "env/lemma",
    "cor": "env/corollary",
    "corollary": "env/corollary",
    "remark": "env/remark",
    "rmk": "env/remark",
    "example": "env/example",
    "proof": "env/proof",
    "note": "env/note",
    "notation": "env/definition",
}

# Link type derived from environment context
ENV_TO_LINK_TYPE = {
    "env/definition": "link/definition-ref",
    "env/theorem": "link/theorem-ref",
    "env/proposition": "link/theorem-ref",
    "env/lemma": "link/theorem-ref",
    "env/corollary": "link/theorem-ref",
    "env/proof": "link/proof-ref",
    "env/remark": "link/remark-ref",
    "env/example": "link/example-ref",
    "env/note": "link/remark-ref",
}

# Scope detection (from nlab-prevalidate.py / validate-ct.py)
SCOPE_REGEXES = [
    ("let-binding", r"\bLet\s+\$([^$]+)\$\s+(be|denote)\s+([^.,$]+)"),
    ("define", r"\bDefine\s+\$([^$]+)\$\s*(:=|=|\\equiv)\s*([^.,$]+)"),
    ("assume", r"\b(Assume|Suppose)\s+(that\s+)?\$([^$]+)\$"),
    ("consider", r"\bConsider\s+(a|an|the|some)?\s*\$?([^$.]{1,60})"),
    ("for-any", r"\b(?:for\s+)?(any|every|each|all)\s+\$([^$]+)\$"),
    ("where-binding", r"\bwhere\s+\$([^$]+)\$\s+(is|denotes|represents)\s+([^.,$]+)"),
    ("set-notation", r"\$([^$]*\\in\s+[^$]+)\$"),
]

CLASSICAL_TO_METATHEORY = {
    "let-binding": "bind/let",
    "define": "bind/define",
    "assume": "assume/explicit",
    "consider": "assume/consider",
    "for-any": "quant/universal",
    "where-binding": "constrain/where",
    "set-notation": "constrain/such-that",
}

BINDER_TYPE_BY_COMMAND = {
    "sum": "bind/summation",
    "prod": "bind/product",
    "coprod": "bind/coprod",
    "bigcup": "bind/big-union",
    "bigcap": "bind/big-intersection",
}

SYMBOL_RE = re.compile(r"[A-Za-z](?:_[A-Za-z0-9]+)?")
QUANTIFIER_CMD_RE = re.compile(
    r"\\(forall|exists)\s*(?:\{)?\s*([A-Za-z](?:_[A-Za-z0-9]+)?)"
)
AGGREGATE_BINDER_RE = re.compile(
    r"\\(sum|prod|coprod|bigcup|bigcap)\s*(?:_\{([^}]*)\}|_([A-Za-z](?:_[A-Za-z0-9]+)?))?"
)
INTEGRAL_BINDER_RE = re.compile(
    r"\\int(?:\s*(?:_\{[^}]*\}|_[^\s^{}]+))?(?:\s*(?:\^\{[^}]*\}|\^[^\s{}]+))?"
)
PROSE_ENV_TYPE_MAP = {
    "definition": "env/definition",
    "defn": "env/definition",
    "theorem": "env/theorem",
    "lemma": "env/lemma",
    "proposition": "env/proposition",
    "prop": "env/proposition",
    "corollary": "env/corollary",
    "remark": "env/remark",
    "example": "env/example",
    "proof": "env/proof",
    "notation": "env/definition",
}
LATEX_ENV_OPEN_RE = re.compile(r"\\begin\{(\w+)\}")
PROSE_ENV_HEADING_RE = re.compile(
    r"(?m)^\s*(Definition|Defn|Theorem|Lemma|Proposition|Prop|Corollary|Remark|Example|Proof|Notation)\b[:.]?"
)

# Wire detection (from validate-ct.py)
WIRE_REGEXES = [
    ("wire/adversative", r"\b(?:but|however|on the other hand|nevertheless|yet)\b", re.IGNORECASE),
    ("wire/causal", r"\b(?:because|since|the reason is|given that)\b", re.IGNORECASE),
    ("wire/consequential", r"\b(?:therefore|thus|hence|it follows|so that|note that|in fact)\b", re.IGNORECASE),
    ("wire/clarifying", r"\b(?:that is|in other words|namely|more precisely|i\.e\.)\b", re.IGNORECASE),
    ("wire/intuitive", r"\b(?:intuitively|roughly speaking|heuristically)\b", re.IGNORECASE),
]

# Port detection (from validate-ct.py)
PORT_REGEXES = [
    ("port/that-noun", r"\bthat\s+(?:root|function|map|functor|morphism|object|category|space|set|group|arrow|diagram)\b", re.IGNORECASE),
    ("port/this-noun", r"\bthis\s+(?:equation|operator|means|functor|morphism|diagram|category|map|adjunction|construction)\b", re.IGNORECASE),
    ("port/the-above", r"\b(?:the above|the preceding|the previous)\s+\w+", re.IGNORECASE),
    ("port/the-same", r"\bthe same\s+\w+", re.IGNORECASE),
    ("port/such", r"\bsuch (?:a|an)\s+\w+", re.IGNORECASE),
    ("port/similarly", r"\b(?:similarly|analogously)\b", re.IGNORECASE),
    ("port/likewise", r"\b(?:likewise|correspondingly)\b", re.IGNORECASE),
]

# Wire labels (from validate-ct.py)
LABEL_REGEXES = [
    ("explain/meaning", r"\bthis means\b", re.IGNORECASE),
    ("explain/think-of", r"\b(?:think of|can be thought of)\b", re.IGNORECASE),
    ("explain/the-idea", r"\bthe (?:idea|trick|key|point) is\b", re.IGNORECASE),
    ("correct/actually", r"\bactually\b", re.IGNORECASE),
    ("correct/subtlety", r"\b(?:subtlety|subtle)\b", re.IGNORECASE),
    ("epistemic/can-show", r"\bone can (?:show|verify|check)\b", re.IGNORECASE),
    ("epistemic/known", r"\b(?:well known|well-known|it is known)\b", re.IGNORECASE),
    ("construct/exists", r"\bthere (?:is|exists|exist)\b", re.IGNORECASE),
    ("construct/explicit", r"\bexplicitly\b", re.IGNORECASE),
    ("strategy/generalize", r"\b(?:generalize|generalise|more generally)\b", re.IGNORECASE),
    ("strategy/example", r"\b(?:for example|for instance|e\.g\.)\b", re.IGNORECASE),
]


# ============================================================
# Step 1: nLab page iteration (reused from nlab-prevalidate.py)
# ============================================================

def iter_nlab_pages(pages_dir, limit=None):
    """Yield (page_id, name, content_md) from nLab sharded directory."""
    count = 0
    for name_file in sorted(Path(pages_dir).rglob("name")):
        content_file = name_file.parent / "content.md"
        if not content_file.exists():
            continue
        page_id = name_file.parent.name
        name = name_file.read_text().strip()
        content = content_file.read_text()
        yield page_id, name, content
        count += 1
        if limit and count >= limit:
            break


# ============================================================
# Step 1: Environment parser
# ============================================================

def parse_environments(content):
    """Parse nLab structural environments from page content.

    Returns list of environment records with type, text, position, length.
    Handles both wiki-style (+-- {: .class} ... =--) and LaTeX-style
    (\\begin{env} ... \\end{env}) blocks.
    """
    envs = []

    # --- Wiki-style environments ---
    # Pattern: +-- {: .class_name ...}\n ... \n=--
    # These can nest, so we track depth.
    wiki_env_open = re.compile(r'^\+--\s*\{:\s*([^}]+)\}', re.MULTILINE)
    wiki_env_close = re.compile(r'^=--', re.MULTILINE)

    # Find all opens and closes with positions
    opens = [(m.start(), m.end(), m.group(1).strip()) for m in wiki_env_open.finditer(content)]
    closes = [m.start() for m in wiki_env_close.finditer(content)]

    # Match opens to closes respecting nesting
    close_idx = 0
    stack = []
    for open_start, open_end, classes in opens:
        # Consume any closes that come before this open
        while close_idx < len(closes) and closes[close_idx] < open_start:
            if stack:
                entry = stack.pop()
                close_pos = closes[close_idx]
                entry["end"] = close_pos
                entry["text"] = content[entry["text_start"]:close_pos]
                envs.append(entry)
            close_idx += 1

        # Determine environment type from CSS classes
        env_type = None
        for cls in classes.split():
            cls_clean = cls.strip()
            if cls_clean in ENV_WIKI_STYLE:
                env_type = ENV_WIKI_STYLE[cls_clean]
                break

        if env_type:
            stack.append({
                "start": open_start,
                "text_start": open_end + 1,  # skip newline after open tag
                "classes": classes,
                "env_type": env_type,
            })
        else:
            # Navigation/layout block — push a marker to track nesting
            stack.append({
                "start": open_start,
                "text_start": open_end + 1,
                "classes": classes,
                "env_type": None,  # skip, but track for nesting
            })

    # Close remaining stack entries
    while close_idx < len(closes) and stack:
        entry = stack.pop()
        close_pos = closes[close_idx]
        entry["end"] = close_pos
        entry["text"] = content[entry["text_start"]:close_pos]
        envs.append(entry)
        close_idx += 1

    # Filter to only typed environments
    envs = [e for e in envs if e.get("env_type")]

    # --- LaTeX-style environments ---
    latex_env_re = re.compile(
        r'\\begin\{(\w+)\}(?:\s*\\label\{([^}]*)\})?'
        r'(.*?)'
        r'\\end\{\1\}',
        re.DOTALL
    )
    for m in latex_env_re.finditer(content):
        env_name = m.group(1)
        label = m.group(2)
        body = m.group(3)
        env_type = ENV_LATEX_STYLE.get(env_name)
        if env_type:
            envs.append({
                "start": m.start(),
                "end": m.end(),
                "text_start": m.start(3),
                "text": body,
                "env_type": env_type,
                "classes": env_name,
                "label": label,
            })

    # Sort by position
    envs.sort(key=lambda e: e["start"])

    return envs


def envs_to_records(page_id, envs):
    """Convert parsed environments to hx/ records."""
    records = []
    for i, env in enumerate(envs):
        rec = {
            "hx/id": f"nlab-{page_id}:env-{i:03d}",
            "hx/role": "environment",
            "hx/type": env["env_type"],
            "hx/parent": None,
            "hx/content": {
                "position": env["start"],
                "length": env["end"] - env["start"],
                "text_preview": env["text"][:200].strip(),
            },
            "hx/labels": ["environment", env["env_type"].split("/")[1]],
        }
        if env.get("label"):
            rec["hx/content"]["label"] = env["label"]
        records.append(rec)
    return records


# ============================================================
# Step 2: Environment-typed wiki links
# ============================================================

WIKI_LINK_RE = re.compile(r'\[\[([^\]|!]+?)(?:\|([^\]]+))?\]\]')
INCLUDE_RE = re.compile(r'\[\[!include\s+([^\]]+)\]\]')
NAVIGATION_END = None  # set per-page: end of context sidebar

def extract_typed_links(page_id, content, envs):
    """Extract wiki links tagged by enclosing environment type.

    Returns list of typed link records.
    """
    links = []
    link_idx = 0

    # Build environment span index for fast lookup
    env_spans = [(e["start"], e["end"], e["env_type"], i) for i, e in enumerate(envs)]

    # Detect end of navigation boilerplate
    # (typically ends after the last =-- before first ## heading)
    first_heading = re.search(r'^##\s', content, re.MULTILINE)
    nav_end = first_heading.start() if first_heading else 0

    for m in WIKI_LINK_RE.finditer(content):
        target = m.group(1).strip()
        display = m.group(2)
        pos = m.start()

        # Skip redirects and includes
        # (check if preceded by [[! on same line)
        line_start = content.rfind('\n', 0, pos) + 1
        line = content[line_start:pos + len(m.group())]
        if '[[!' in line and '[[!' in content[max(0, pos - 5):pos + 3]:
            continue

        # Determine link type from environment context
        link_type = "link/prose-ref"
        parent_env = None

        if pos < nav_end:
            link_type = "link/navigation"
        else:
            for env_start, env_end, env_type, env_idx in env_spans:
                if env_start <= pos < env_end:
                    link_type = ENV_TO_LINK_TYPE.get(env_type, "link/prose-ref")
                    parent_env = f"nlab-{page_id}:env-{env_idx:03d}"
                    break

        links.append({
            "hx/id": f"nlab-{page_id}:link-{link_idx:03d}",
            "hx/role": "typed-link",
            "hx/type": link_type,
            "hx/source": parent_env or f"nlab-{page_id}",
            "hx/target": f"nlab:{target}",
            "hx/content": {
                "match": m.group(),
                "target_name": target,
                "display": display,
                "position": pos,
            },
            "hx/labels": ["link", link_type.split("/")[1]],
        })
        link_idx += 1

    return links


# ============================================================
# Step 3: Commutative diagram extraction (tikzcd)
# ============================================================

TIKZCD_RE = re.compile(
    r'\\begin\{tikzcd\}(.*?)\\end\{tikzcd\}',
    re.DOTALL
)

# Arrow pattern inside tikzcd: \ar[...] or \arrow[...]
TIKZCD_ARROW_RE = re.compile(
    r'\\ar(?:row)?\[([^\]]+)\]'
)


def parse_tikzcd_direction(direction_str):
    """Parse tikzcd direction codes like 'r', 'dr', 'rr' into (row_delta, col_delta)."""
    row_delta = 0
    col_delta = 0
    for ch in direction_str:
        if ch == 'r':
            col_delta += 1
        elif ch == 'l':
            col_delta -= 1
        elif ch == 'd':
            row_delta += 1
        elif ch == 'u':
            row_delta -= 1
    return row_delta, col_delta


def parse_tikzcd(tikzcd_body):
    """Parse a tikzcd body into objects (nodes) and morphisms (arrows).

    Handles both direction-code arrows (\\ar[r, "label"]) and
    coordinate arrows (\\arrow["label", from=1-1, to=1-2]).

    Returns {"objects": [...], "morphisms": [...]}.
    """
    objects = []
    morphisms = []

    # Split into rows by \\ (row separator)
    rows = re.split(r'\\\\', tikzcd_body)

    node_grid = {}  # (row, col) -> node_index
    node_idx = 0

    # First pass: collect standalone \arrow lines (coordinate format)
    # These appear on their own lines, not inside cells
    standalone_arrows = []
    cell_lines = []
    for row_num, row in enumerate(rows):
        # Check if this row is purely arrow declarations
        stripped = row.strip()
        if re.match(r'^\\ar(?:row)?\[', stripped) and '&' not in stripped:
            # This is a standalone arrow line — may have multiple arrows
            for arrow_m in TIKZCD_ARROW_RE.finditer(stripped):
                standalone_arrows.append(arrow_m.group(1))
        else:
            cell_lines.append((row_num, row))

    # Second pass: parse cells from non-arrow rows
    real_row = 0
    for orig_row_num, row in cell_lines:
        cells = row.split('&')
        for col_num, cell in enumerate(cells):
            cell = cell.strip()
            if not cell:
                continue

            # Extract the node label (everything that's not an \ar/\arrow command)
            label = TIKZCD_ARROW_RE.sub('', cell).strip()
            # Clean up LaTeX noise
            label = re.sub(r'\\(?:ar(?:row)?|mathrm|mathrlap|mathllap|phantom)\b', '', label)
            label = re.sub(r'\{:\s*\.\w+\}', '', label)
            label = re.sub(r'"\{[^"]*\}"', '', label)  # tikzcd label specs
            label = re.sub(r"'\{[^']*\}'", '', label)  # tikzcd swapped label specs
            label = re.sub(r'\{description\}', '', label)
            label = re.sub(r'Rightarrow', '', label)
            label = re.sub(r'no head', '', label)
            # Strip tikzcd options like [sep=20pt] or [row sep=10pt]
            label = re.sub(r'^\[[\w\s=,]+\]\s*', '', label)
            label = label.strip(' \t\n{},')

            if label and label not in ('', '&', '=', ','):
                objects.append({
                    "index": node_idx,
                    "row": real_row,
                    "col": col_num,
                    "label": label,
                })
                node_grid[(real_row, col_num)] = node_idx
                node_idx += 1
            elif (real_row, col_num) not in node_grid:
                node_grid[(real_row, col_num)] = None

            # Extract inline arrows from this cell (direction-code format)
            for arrow_m in TIKZCD_ARROW_RE.finditer(cell):
                arrow_spec = arrow_m.group(1)
                parts = [p.strip() for p in arrow_spec.split(',')]

                # Check for coordinate format (from=r-c, to=r-c)
                from_coord = _parse_coord(arrow_spec)
                to_coord = _parse_to_coord(arrow_spec)

                if from_coord or to_coord:
                    # Coordinate format
                    src_pos = (from_coord[0] - 1, from_coord[1] - 1) if from_coord else (real_row, col_num)
                    tgt_pos = (to_coord[0] - 1, to_coord[1] - 1) if to_coord else (real_row, col_num)
                else:
                    # Direction-code format: first part is direction
                    direction = parts[0] if parts else ''
                    row_delta, col_delta = parse_tikzcd_direction(direction)
                    src_pos = (real_row, col_num)
                    tgt_pos = (real_row + row_delta, col_num + col_delta)

                arrow_label = _extract_arrow_label(parts)

                morphisms.append({
                    "source_pos": src_pos,
                    "target_pos": tgt_pos,
                    "source_idx": node_grid.get(src_pos),
                    "target_idx": node_grid.get(tgt_pos),
                    "label": arrow_label,
                    "direction": parts[0] if parts else '',
                })

        real_row += 1

    # Process standalone arrows (coordinate format)
    for arrow_spec in standalone_arrows:
        parts = [p.strip() for p in arrow_spec.split(',')]
        from_coord = _parse_coord(arrow_spec)
        to_coord = _parse_to_coord(arrow_spec)

        if from_coord and to_coord:
            src_pos = (from_coord[0] - 1, from_coord[1] - 1)
            tgt_pos = (to_coord[0] - 1, to_coord[1] - 1)
        else:
            continue  # Can't resolve without coordinates

        arrow_label = _extract_arrow_label(parts)

        morphisms.append({
            "source_pos": src_pos,
            "target_pos": tgt_pos,
            "source_idx": node_grid.get(src_pos),
            "target_idx": node_grid.get(tgt_pos),
            "label": arrow_label,
            "direction": "coord",
        })

    return {"objects": objects, "morphisms": morphisms}


def _parse_coord(spec):
    """Parse from=row-col coordinate. Returns (row, col) or None."""
    m = re.search(r'from=(\d+)-(\d+)', spec)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _parse_to_coord(spec):
    """Parse to=row-col coordinate. Returns (row, col) or None."""
    m = re.search(r'to=(\d+)-(\d+)', spec)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _extract_arrow_label(parts):
    """Extract label from tikzcd arrow parts."""
    for part in parts:
        # Skip known non-label parts
        if re.match(r'^(from|to)=', part):
            continue
        if part in ('Rightarrow', 'no head', 'dashed', 'hook', 'two heads',
                     'shift left', 'shift right', 'bend left', 'bend right'):
            continue
        # Label in quotes: "label" or "label"'
        lm = re.search(r'"([^"]*)"', part)
        if lm:
            lbl = lm.group(1).strip()
            if lbl and lbl not in (r'\ ', ' ', ''):
                return lbl
    return None


# \array{} diagram start pattern (older nLab pages)
ARRAY_START_RE = re.compile(r'\\array\s*\{')

# Arrow tokens in \array{} diagrams
ARRAY_ARROW_TOKENS = {
    r'\to', r'\rightarrow', r'\longrightarrow', r'\hookrightarrow',
    r'\leftarrow', r'\longleftarrow',
    r'\downarrow', r'\uparrow',
    r'\nearrow', r'\searrow', r'\swarrow', r'\nwarrow',
    r'\Rightarrow', r'\Leftarrow',
    r'\big\downarrow', r'\big\uparrow',
}

def _extract_brace_balanced(text, start):
    """Extract brace-balanced content starting after the opening { at position start.

    Returns the body text (between outer braces) and end position, or (None, start).
    """
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1], i
    return None, start


ARRAY_ARROW_RE = re.compile(
    r'\\(?:stackrel|overset)\{([^}]*)\}\{\\(?:to|rightarrow|longrightarrow)\}'
    r'|\\(?:big)?(?:downarrow|uparrow)\\?(?:mathrlap|mathllap)?\{?\{?\^?\{?([^}]*?)\}?\}?\}?'
    r'|\\(?:to|rightarrow|longrightarrow|leftarrow|nearrow|searrow|swarrow|nwarrow|hookrightarrow)\b'
)


def parse_array_diagram(array_body):
    """Parse a \\array{} body into objects and morphisms.

    These use row/column layout with \\to, \\downarrow etc. for arrows
    and \\stackrel{label}{\\to} for labeled arrows.
    """
    objects = []
    morphisms = []

    rows = re.split(r'\\\\', array_body)
    node_grid = {}
    node_idx = 0

    for row_num, row in enumerate(rows):
        cells = row.split('&')
        for col_num, cell in enumerate(cells):
            cell = cell.strip()
            if not cell:
                continue

            # Check if this cell is an arrow
            cell_stripped = re.sub(r'\\(?:math[rl]lap|big)\b', '', cell).strip(' {}^_')
            is_arrow = False
            arrow_label = None

            # Labeled arrow: \stackrel{label}{\to}
            sm = re.search(r'\\(?:stackrel|overset)\{([^}]*)\}\{\\(?:to|rightarrow|longrightarrow)\}', cell)
            if sm:
                is_arrow = True
                arrow_label = sm.group(1).strip()
            elif any(tok in cell for tok in (r'\to', r'\rightarrow', r'\longrightarrow',
                                              r'\leftarrow', r'\downarrow', r'\uparrow',
                                              r'\nearrow', r'\searrow', r'\swarrow')):
                # Check if cell is primarily an arrow (not an object with \to in it)
                non_arrow = re.sub(r'\\(?:stackrel|overset)\{[^}]*\}\{[^}]*\}', '', cell)
                non_arrow = re.sub(r'\\(?:big)?(?:to|rightarrow|longrightarrow|leftarrow|downarrow|uparrow|nearrow|searrow|swarrow|nwarrow|hookrightarrow)\b', '', non_arrow)
                non_arrow = re.sub(r'\\(?:math[rl]lap|big)\b', '', non_arrow)
                non_arrow = non_arrow.strip(' {}^_\n\t,.')
                if len(non_arrow) < 3:
                    is_arrow = True
                    # Extract label from \mathrlap{{}^label} patterns
                    lm = re.search(r'\{\{\}?\^?\{?([A-Za-z\\]+[^}]*)\}', cell)
                    if lm:
                        arrow_label = lm.group(1).strip()

            if is_arrow:
                # Determine direction from arrow token
                direction = "right"
                if r'\downarrow' in cell or r'\big\downarrow' in cell:
                    direction = "down"
                elif r'\uparrow' in cell or r'\big\uparrow' in cell:
                    direction = "up"
                elif r'\nearrow' in cell:
                    direction = "upright"
                elif r'\searrow' in cell:
                    direction = "downright"
                elif r'\swarrow' in cell:
                    direction = "downleft"
                elif r'\leftarrow' in cell:
                    direction = "left"

                morphisms.append({
                    "source_pos": (row_num, col_num),
                    "target_pos": None,  # resolved later by direction
                    "source_idx": None,
                    "target_idx": None,
                    "label": arrow_label,
                    "direction": direction,
                })
            else:
                # Object node
                label = cell
                label = re.sub(r'\\(?:math[rl]lap|big|mathrm)\b', '', label)
                label = re.sub(r'\{:\s*\.\w+\}', '', label)
                label = label.strip(' \t\n{}$,.')
                if label and len(label) >= 1:
                    objects.append({
                        "index": node_idx,
                        "row": row_num,
                        "col": col_num,
                        "label": label,
                    })
                    node_grid[(row_num, col_num)] = node_idx
                    node_idx += 1

    return {"objects": objects, "morphisms": morphisms}


def extract_diagrams(page_id, content, envs):
    """Extract tikzcd and \\array{} diagrams as structured records."""
    diagrams = []
    diag_idx = 0

    env_spans = [(e["start"], e["end"], e["env_type"], i) for i, e in enumerate(envs)]

    def find_parent_env(pos):
        for env_start, env_end, env_type, env_idx in env_spans:
            if env_start <= pos < env_end:
                return f"nlab-{page_id}:env-{env_idx:03d}"
        return None

    def make_ends(parsed):
        ends = []
        for obj in parsed["objects"]:
            ends.append({
                "role": "object",
                "label": obj["label"],
                "grid": [obj["row"], obj["col"]],
            })
        for morph in parsed["morphisms"]:
            end = {
                "role": "morphism",
                "direction": morph["direction"],
            }
            if morph.get("label"):
                end["label"] = morph["label"]
            if morph.get("source_idx") is not None:
                end["source"] = morph["source_idx"]
            if morph.get("target_idx") is not None:
                end["target"] = morph["target_idx"]
            ends.append(end)
        return ends

    # tikzcd diagrams
    for m in TIKZCD_RE.finditer(content):
        body = m.group(1)
        pos = m.start()
        parsed = parse_tikzcd(body)
        if not parsed["objects"] and not parsed["morphisms"]:
            continue
        ends = make_ends(parsed)
        diagrams.append({
            "hx/id": f"nlab-{page_id}:diag-{diag_idx:03d}",
            "hx/role": "diagram",
            "hx/type": "diagram/commutative",
            "hx/parent": find_parent_env(pos),
            "hx/ends": ends,
            "hx/content": {
                "raw_tikzcd": body[:500],
                "position": pos,
                "n_objects": len(parsed["objects"]),
                "n_morphisms": len(parsed["morphisms"]),
            },
            "hx/labels": ["diagram", "commutative"],
        })
        diag_idx += 1

    # \array{} diagrams (brace-balanced extraction)
    for m in ARRAY_START_RE.finditer(content):
        body, end_pos = _extract_brace_balanced(content, m.end())
        if body is None:
            continue
        pos = m.start()
        # Skip very short arrays (likely inline notation, not diagrams)
        if len(body) < 20 or '\\\\' not in body:
            continue
        parsed = parse_array_diagram(body)
        if len(parsed["objects"]) < 2:
            continue
        ends = make_ends(parsed)
        diagrams.append({
            "hx/id": f"nlab-{page_id}:diag-{diag_idx:03d}",
            "hx/role": "diagram",
            "hx/type": "diagram/array",
            "hx/parent": find_parent_env(pos),
            "hx/ends": ends,
            "hx/content": {
                "raw_array": body[:500],
                "position": pos,
                "n_objects": len(parsed["objects"]),
                "n_morphisms": len(parsed["morphisms"]),
            },
            "hx/labels": ["diagram", "array"],
        })
        diag_idx += 1

    return diagrams


# ============================================================
# Step 4: Discourse wiring within environments
# ============================================================

def strip_nlab_markup(text):
    """Strip nLab markup for prose processing, keeping wiki-link text."""
    plain = text
    plain = re.sub(r'\+-- \{:.*?\}.*?=--', '', plain, flags=re.DOTALL)
    plain = re.sub(r'\[\[!include.*?\]\]', '', plain)
    plain = re.sub(r'\[\[([^\]|!]+?)(?:\|([^\]]+))?\]\]',
                   lambda m: m.group(2) or m.group(1), plain)
    plain = re.sub(r'^#+\s+', '', plain, flags=re.MULTILINE)
    plain = re.sub(r'\{:.*?\}', '', plain)
    plain = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', plain)
    plain = re.sub(r'\\begin\{tikzcd\}.*?\\end\{tikzcd\}', '', plain, flags=re.DOTALL)
    plain = re.sub(r'\\begin\{centre\}', '', plain)
    plain = re.sub(r'\\end\{centre\}', '', plain)
    return plain


def _iter_math_fragments(text):
    """Yield (fragment, absolute_position) for common LaTeX math delimiters."""
    blocked = []

    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        blocked.append((m.start(), m.end()))
        yield m.group(1), m.start(1)

    for m in re.finditer(r"\\\[(.+?)\\\]", text, re.DOTALL):
        blocked.append((m.start(), m.end()))
        yield m.group(1), m.start(1)

    for m in re.finditer(r"\$([^$\n]+?)\$", text):
        if any(a <= m.start() < b for a, b in blocked):
            continue
        yield m.group(1), m.start(1)

    for m in re.finditer(r"\\\((.+?)\\\)", text, re.DOTALL):
        yield m.group(1), m.start(1)


def _extract_bound_symbol(subscript):
    """Best-effort extraction of a bound variable from a subscript."""
    if not subscript:
        return None

    m = re.search(r"([A-Za-z](?:_[A-Za-z0-9]+)?)\s*(?:=|\\in|∈)", subscript)
    if m:
        return m.group(1)

    m = SYMBOL_RE.search(subscript)
    return m.group(0) if m else None


def _extract_integral_symbol(fragment_tail):
    """Best-effort extraction of integration variable from trailing fragment."""
    m = re.search(
        r"(?:\\,|\\;|\\!|\s)*(?:d|\\mathrm\{d\})\s*([A-Za-z](?:_[A-Za-z0-9]+)?)",
        fragment_tail,
    )
    if m:
        return m.group(1)
    return None


def _detect_symbolic_binders(entity_id, text, start_idx=0, parent_env_id=None):
    """Detect binder-like symbolic operators in LaTeX math fragments."""
    scopes = []
    scope_idx = start_idx

    for fragment, frag_pos in _iter_math_fragments(text):
        # Quantifiers: \forall x, \exists y, ...
        for m in QUANTIFIER_CMD_RE.finditer(fragment):
            quant_cmd = m.group(1)
            symbol = m.group(2)
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1

            scope_type = "quant/universal" if quant_cmd == "forall" else "quant/existential"
            scopes.append({
                "hx/id": scope_id,
                "hx/role": "component",
                "hx/type": scope_type,
                "hx/parent": parent_env_id,
                "hx/ends": [
                    {"role": "entity", "ident": entity_id},
                    {"role": "binder", "latex": f"\\{quant_cmd}"},
                    {"role": "symbol", "latex": symbol},
                ],
                "hx/content": {
                    "match": fragment[m.start():m.end()][:120],
                    "position": frag_pos + m.start(),
                },
                "hx/labels": ["scope", "symbolic-binder", quant_cmd],
            })

        # Aggregate binders: \sum, \prod, \coprod, ...
        for m in AGGREGATE_BINDER_RE.finditer(fragment):
            cmd = m.group(1)
            subscript = m.group(2) or m.group(3) or ""
            bound_symbol = _extract_bound_symbol(subscript)
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1

            ends = [
                {"role": "entity", "ident": entity_id},
                {"role": "binder", "latex": f"\\{cmd}"},
            ]
            if bound_symbol:
                ends.append({"role": "symbol", "latex": bound_symbol})
            if subscript:
                ends.append({"role": "subscript", "latex": subscript[:80]})

            scopes.append({
                "hx/id": scope_id,
                "hx/role": "component",
                "hx/type": BINDER_TYPE_BY_COMMAND.get(cmd, "bind/operator"),
                "hx/parent": parent_env_id,
                "hx/ends": ends,
                "hx/content": {
                    "match": fragment[m.start():m.end()][:120],
                    "position": frag_pos + m.start(),
                },
                "hx/labels": ["scope", "symbolic-binder", cmd],
            })

        # Integrals: \int ... dx
        for m in INTEGRAL_BINDER_RE.finditer(fragment):
            tail = fragment[m.end():m.end() + 64]
            symbol = _extract_integral_symbol(tail)
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1

            ends = [
                {"role": "entity", "ident": entity_id},
                {"role": "binder", "latex": "\\int"},
            ]
            if symbol:
                ends.append({"role": "symbol", "latex": symbol})

            scopes.append({
                "hx/id": scope_id,
                "hx/role": "component",
                "hx/type": "bind/integral",
                "hx/parent": parent_env_id,
                "hx/ends": ends,
                "hx/content": {
                    "match": fragment[m.start():min(len(fragment), m.end() + 32)][:120],
                    "position": frag_pos + m.start(),
                },
                "hx/labels": ["scope", "symbolic-binder", "integral"],
            })

    return scopes


def _detect_environment_scopes(entity_id, text, start_idx=0, parent_env_id=None):
    """Detect theorem-like environments as discourse scopes."""
    scopes = []
    scope_idx = start_idx

    for m in LATEX_ENV_OPEN_RE.finditer(text):
        env_name = m.group(1).lower()
        env_type = ENV_LATEX_STYLE.get(env_name)
        if not env_type:
            continue
        scope_id = f"{entity_id}:scope-{scope_idx:03d}"
        scope_idx += 1
        scopes.append({
            "hx/id": scope_id,
            "hx/role": "component",
            "hx/type": env_type,
            "hx/parent": parent_env_id,
            "hx/ends": [
                {"role": "entity", "ident": entity_id},
                {"role": "environment", "name": env_name},
            ],
            "hx/content": {
                "match": m.group()[:120],
                "position": m.start(),
            },
            "hx/labels": ["scope", "environment", env_name],
        })

    for m in PROSE_ENV_HEADING_RE.finditer(text):
        label = m.group(1).lower()
        env_type = PROSE_ENV_TYPE_MAP.get(label)
        if not env_type:
            continue
        scope_id = f"{entity_id}:scope-{scope_idx:03d}"
        scope_idx += 1
        scopes.append({
            "hx/id": scope_id,
            "hx/role": "component",
            "hx/type": env_type,
            "hx/parent": parent_env_id,
            "hx/ends": [
                {"role": "entity", "ident": entity_id},
                {"role": "environment", "name": label},
            ],
            "hx/content": {
                "match": m.group()[:120],
                "position": m.start(),
            },
            "hx/labels": ["scope", "environment", label],
        })

    return scopes


def detect_scopes(entity_id, text, parent_env_id=None):
    """Detect scope bindings in text. Returns hx/ records."""
    scopes = []
    scope_idx = 0
    for stype, pattern in SCOPE_REGEXES:
        for m in re.finditer(pattern, text):
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1
            ends = [{"role": "entity", "ident": entity_id}]
            if stype == "let-binding":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "type", "text": m.group(3).strip()[:80]})
            elif stype == "define":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "value", "text": m.group(3).strip()[:80]})
            elif stype == "assume":
                ends.append({"role": "condition", "latex": m.group(3).strip()})
            elif stype == "consider":
                obj = (m.group(2) or "").strip()
                if obj:
                    ends.append({"role": "object", "text": obj[:80]})
            elif stype == "for-any":
                ends.append({"role": "quantifier", "text": m.group(1)})
                ends.append({"role": "symbol", "latex": m.group(2).strip()})
            elif stype == "where-binding":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "description", "text": m.group(3).strip()[:80]})
            elif stype == "set-notation":
                ends.append({"role": "membership", "latex": m.group(1).strip()})
            meta_type = CLASSICAL_TO_METATHEORY.get(stype, f"scope/{stype}")
            scopes.append({
                "hx/id": scope_id,
                "hx/role": "component",
                "hx/type": meta_type,
                "hx/parent": parent_env_id,
                "hx/ends": ends,
                "hx/content": {"match": m.group()[:120], "position": m.start()},
                "hx/labels": ["scope", stype],
            })
    env_scopes = _detect_environment_scopes(
        entity_id, text, start_idx=scope_idx, parent_env_id=parent_env_id)
    scopes.extend(env_scopes)
    scope_idx += len(env_scopes)

    symbolic = _detect_symbolic_binders(
        entity_id, text, start_idx=scope_idx, parent_env_id=parent_env_id)
    scopes.extend(symbolic)
    return scopes


def detect_wires(entity_id, text, parent_env_id=None):
    """Detect connective wires in text."""
    wires = []
    wire_idx = 0
    for wtype, pattern, flags in WIRE_REGEXES:
        for m in re.finditer(pattern, text, flags):
            wires.append({
                "hx/id": f"{entity_id}:wire-{wire_idx:03d}",
                "hx/role": "wire",
                "hx/type": wtype,
                "hx/parent": parent_env_id,
                "hx/content": {"match": m.group(), "position": m.start()},
                "hx/labels": ["wire", wtype.split("/")[1]],
            })
            wire_idx += 1
    return wires


def detect_ports(entity_id, text, parent_env_id=None):
    """Detect anaphoric port references in text."""
    ports = []
    port_idx = 0
    for ptype, pattern, flags in PORT_REGEXES:
        for m in re.finditer(pattern, text, flags):
            ports.append({
                "hx/id": f"{entity_id}:port-{port_idx:03d}",
                "hx/role": "port",
                "hx/type": ptype,
                "hx/parent": parent_env_id,
                "hx/content": {"match": m.group(), "position": m.start()},
                "hx/labels": ["port", ptype.split("/")[1]],
            })
            port_idx += 1
    return ports


def detect_labels(entity_id, text, parent_env_id=None):
    """Detect wire reasoning labels in text."""
    labels = []
    label_idx = 0
    for ltype, pattern, flags in LABEL_REGEXES:
        for m in re.finditer(pattern, text, flags):
            labels.append({
                "hx/id": f"{entity_id}:label-{label_idx:03d}",
                "hx/role": "label",
                "hx/type": ltype,
                "hx/parent": parent_env_id,
                "hx/content": {"match": m.group(), "position": m.start()},
                "hx/labels": ["label", ltype.split("/")[0]],
            })
            label_idx += 1
    return labels


def extract_discourse_wiring(page_id, content, envs):
    """Run scope/wire/port/label detection within each environment and on prose.

    Returns combined list of all discourse records.
    """
    all_records = []
    entity_id = f"nlab-{page_id}"

    # Process each environment
    for i, env in enumerate(envs):
        env_id = f"nlab-{page_id}:env-{i:03d}"
        text = strip_nlab_markup(env.get("text", ""))
        if len(text) < 10:
            continue
        all_records.extend(detect_scopes(entity_id, text, parent_env_id=env_id))
        all_records.extend(detect_wires(entity_id, text, parent_env_id=env_id))
        all_records.extend(detect_ports(entity_id, text, parent_env_id=env_id))
        all_records.extend(detect_labels(entity_id, text, parent_env_id=env_id))

    # Process prose outside environments
    prose = content
    for env in envs:
        prose = prose[:env["start"]] + " " * (env["end"] - env["start"]) + prose[env["end"]:]
    prose = strip_nlab_markup(prose)
    if len(prose.strip()) > 20:
        all_records.extend(detect_scopes(entity_id, prose, parent_env_id=None))
        all_records.extend(detect_wires(entity_id, prose, parent_env_id=None))
        all_records.extend(detect_ports(entity_id, prose, parent_env_id=None))
        all_records.extend(detect_labels(entity_id, prose, parent_env_id=None))

    return all_records


# ============================================================
# Step 5: N-ary categorical hyperedges
# ============================================================

# Keyword/link signals for each categorical pattern
CAT_PATTERNS = {
    "cat/adjunction": {
        "link_signals": [
            "left adjoint", "right adjoint", "adjoint functor", "adjoint functors",
            "adjunction unit", "adjunction counit", "unit", "counit",
        ],
        "text_signals": [
            r"\bleft adjoint\b", r"\bright adjoint\b",
            r"\\dashv\b", r"\\vdash\b",
            r"\bunit\b.*\bcounit\b",
            r"\btriangle identit",
            r"\bzig-?zag",
        ],
        "min_signals": 2,
        "roles_template": [
            {"role": "left-adjoint", "link_hint": "left adjoint"},
            {"role": "right-adjoint", "link_hint": "right adjoint"},
            {"role": "unit", "link_hint": "adjunction unit"},
            {"role": "counit", "link_hint": "adjunction counit"},
        ],
    },
    "cat/kan-extension": {
        "link_signals": [
            "Kan extension", "left Kan extension", "right Kan extension",
            "pointwise Kan extension",
        ],
        "text_signals": [
            r"\bKan extension\b",
            r"\\mathrm\{Lan\}", r"\\mathrm\{Ran\}",
            r"\bLan_", r"\bRan_",
            r"\buniversal\b.*\bnatural transformation\b",
        ],
        "min_signals": 1,
        "roles_template": [
            {"role": "base-functor", "link_hint": "functor"},
            {"role": "restriction", "link_hint": "functor"},
            {"role": "extension", "link_hint": "Kan extension"},
            {"role": "universal-arrow", "link_hint": "natural transformation"},
        ],
    },
    "cat/limit": {
        "link_signals": [
            "limit", "colimit", "cone", "universal cone",
            "terminal object", "product", "pullback", "equalizer",
        ],
        "text_signals": [
            r"\blim\b", r"\bcolim\b",
            r"\\lim\b", r"\\colim\b",
            r"\buniversal cone\b",
            r"\blimiting cone\b",
        ],
        "min_signals": 2,
        "roles_template": [
            {"role": "diagram-functor", "link_hint": "functor"},
            {"role": "limit-object", "link_hint": "limit"},
            {"role": "cone", "link_hint": "cone"},
            {"role": "universal-property", "link_hint": "universal construction"},
        ],
    },
    "cat/monad": {
        "link_signals": [
            "monad", "Kleisli category", "Eilenberg-Moore category",
            "monadicity theorem",
        ],
        "text_signals": [
            r"\bmonad\b",
            r"\bendofunctor\b",
            r"\\mu\s*:", r"\\eta\s*:",
            r"\bmultiplication\b.*\bunit\b",
        ],
        "min_signals": 2,
        "roles_template": [
            {"role": "endofunctor", "link_hint": "endofunctor"},
            {"role": "unit", "link_hint": "natural transformation"},
            {"role": "multiplication", "link_hint": "natural transformation"},
        ],
    },
    "cat/natural-transformation": {
        "link_signals": [
            "natural transformation", "naturality", "functor category",
        ],
        "text_signals": [
            r"\bnatural transformation\b",
            r"\bnaturality square\b",
            r"\bnatural isomorphism\b",
        ],
        "min_signals": 1,
        "roles_template": [
            {"role": "source-functor", "link_hint": "functor"},
            {"role": "target-functor", "link_hint": "functor"},
            {"role": "components", "link_hint": "natural transformation"},
        ],
    },
    "cat/fibration": {
        "link_signals": [
            "Grothendieck fibration", "fibration", "Cartesian morphism",
            "Cartesian fibration",
        ],
        "text_signals": [
            r"\bfibration\b",
            r"\b[Cc]artesian\s+morphism\b",
            r"\b[Cc]artesian\s+lift\b",
            r"\bGrothendieck\b.*\bfibration\b",
        ],
        "min_signals": 2,
        "roles_template": [
            {"role": "total-category", "link_hint": "category"},
            {"role": "base-category", "link_hint": "category"},
            {"role": "projection", "link_hint": "functor"},
            {"role": "cartesian-lift", "link_hint": "Cartesian morphism"},
        ],
    },
    "cat/equivalence": {
        "link_signals": [
            "equivalence of categories", "equivalence in a 2-category",
            "adjoint equivalence",
        ],
        "text_signals": [
            r"\bequivalence of categories\b",
            r"\badjoint equivalence\b",
            r"\bessentially surjective\b.*\bfully faithful\b",
        ],
        "min_signals": 2,
        "roles_template": [
            {"role": "forward-functor", "link_hint": "functor"},
            {"role": "inverse-functor", "link_hint": "functor"},
            {"role": "unit-iso", "link_hint": "natural isomorphism"},
            {"role": "counit-iso", "link_hint": "natural isomorphism"},
        ],
    },
    "cat/universal-property": {
        "link_signals": [
            "universal property", "universal construction", "representable functor",
            "universal arrow",
        ],
        "text_signals": [
            r"\buniversal property\b",
            r"\buniversal morphism\b",
            r"\brepresentable\b",
            r"\binitial object\b.*\bcomma category\b",
        ],
        "min_signals": 2,
        "roles_template": [
            {"role": "universal-object", "link_hint": "universal construction"},
            {"role": "universal-morphism", "link_hint": "morphism"},
        ],
    },
}


def detect_categorical_patterns(page_id, name, content, envs, typed_links):
    """Detect n-ary categorical patterns from environments + links + text.

    Returns list of categorical hyperedge records.
    """
    hyperedges = []
    entity_id = f"nlab-{page_id}"

    # Collect link targets for this page
    link_targets = set()
    for link in typed_links:
        target_name = link["hx/content"].get("target_name", "")
        link_targets.add(target_name.lower())

    # Collect definition-level link targets (strongest signal)
    def_link_targets = set()
    for link in typed_links:
        if link["hx/type"] == "link/definition-ref":
            target_name = link["hx/content"].get("target_name", "")
            def_link_targets.add(target_name.lower())

    plain = strip_nlab_markup(content)

    for cat_type, pattern in CAT_PATTERNS.items():
        score = 0
        evidence = []

        # Check link signals
        for sig in pattern["link_signals"]:
            if sig.lower() in link_targets:
                score += 1
                evidence.append(f"link:{sig}")
                if sig.lower() in def_link_targets:
                    score += 1  # bonus for definition-level link
                    evidence.append(f"deflink:{sig}")

        # Check text signals
        for sig_re in pattern["text_signals"]:
            if re.search(sig_re, plain, re.IGNORECASE):
                score += 1
                evidence.append(f"text:{sig_re[:30]}")

        # Check page name for direct match
        name_lower = name.lower()
        cat_key = cat_type.split("/")[1]
        if cat_key in name_lower or cat_key.replace("-", " ") in name_lower:
            score += 3
            evidence.append(f"name:{name}")

        if score >= pattern["min_signals"]:
            # Build ends from template, filling in what we can
            ends = []
            for role_tmpl in pattern["roles_template"]:
                end = {"role": role_tmpl["role"]}
                hint = role_tmpl.get("link_hint", "")
                # Check if we have a link to the hinted page
                if hint.lower() in link_targets:
                    end["page"] = f"nlab:{hint}"
                ends.append(end)

            hyperedges.append({
                "hx/id": f"{entity_id}:cat-{cat_type.split('/')[1]}",
                "hx/role": "categorical",
                "hx/type": cat_type,
                "hx/ends": ends,
                "hx/content": {
                    "evidence": evidence,
                    "score": score,
                    "page": entity_id,
                    "page_name": name,
                },
                "hx/labels": ["categorical", cat_type.split("/")[1], f"{len(ends)}-ary"],
            })

    return hyperedges


# ============================================================
# NER kernel (reused from nlab-prevalidate.py)
# ============================================================

def load_ner_kernel(path):
    """Load NER kernel from TSV. Returns (singles, multi_index, count)."""
    singles = {}
    multi_index = {}
    multi_count = 0
    skip_prefixes = ("$", "(", '"', "-")
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            term_lower = parts[0].strip()
            term_orig = parts[1].strip()
            canon = parts[3].strip() if len(parts) > 3 else term_lower
            if not term_lower or any(term_lower.startswith(p) for p in skip_prefixes):
                continue
            if len(term_lower) < 3:
                continue
            words = term_lower.split()
            if len(words) == 1:
                singles[term_lower] = (term_orig, canon)
            else:
                first_key = None
                for w in words:
                    if len(w) >= 3:
                        first_key = w
                        break
                if first_key is None:
                    first_key = words[0]
                if first_key not in multi_index:
                    multi_index[first_key] = []
                multi_index[first_key].append((term_lower, term_orig, canon))
                multi_count += 1
    return singles, multi_index, multi_count


def spot_terms(text, singles, multi_index):
    """Spot NER terms in text."""
    text_lower = text.lower()
    words = text_lower.split()
    hits = {}
    for w in set(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in singles:
            hits[clean] = singles[clean]
    for w in set(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in multi_index:
            for term_lower, term_orig, canon in multi_index[clean]:
                if term_lower not in hits and term_lower in text_lower:
                    hits[term_lower] = (term_orig, canon)
    return [{"term": orig, "term_lower": tl, "canon": canon}
            for tl, (orig, canon) in sorted(hits.items())]


# ============================================================
# Main page processor
# ============================================================

def process_page(page_id, name, content, singles=None, multi_index=None):
    """Process a single nLab page. Returns a complete wiring record."""
    # Step 1: Parse environments
    envs = parse_environments(content)
    env_records = envs_to_records(page_id, envs)

    # Step 2: Extract typed links
    typed_links = extract_typed_links(page_id, content, envs)

    # Step 3: Extract diagrams
    diagrams = extract_diagrams(page_id, content, envs)

    # Step 4: Discourse wiring
    discourse = extract_discourse_wiring(page_id, content, envs)

    # Step 5: Categorical hyperedges
    cat_hyperedges = detect_categorical_patterns(page_id, name, content, envs, typed_links)

    # NER terms (optional, if kernel loaded)
    ner_terms = []
    if singles is not None:
        plain = strip_nlab_markup(content)
        ner_terms = spot_terms(plain, singles, multi_index or {})

    # Assemble page record
    return {
        "page_id": f"nlab-{page_id}",
        "page_name": name,
        "environments": env_records,
        "typed_links": typed_links,
        "diagrams": diagrams,
        "discourse": discourse,
        "categorical": cat_hyperedges,
        "ner_terms": ner_terms,
        "stats": {
            "n_environments": len(env_records),
            "n_typed_links": len(typed_links),
            "n_diagrams": len(diagrams),
            "n_discourse": len(discourse),
            "n_categorical": len(cat_hyperedges),
            "n_ner_terms": len(ner_terms),
            "env_types": dict(Counter(e["hx/type"] for e in env_records)),
            "link_types": dict(Counter(l["hx/type"] for l in typed_links)),
            "cat_types": [h["hx/type"] for h in cat_hyperedges],
        },
    }


# ============================================================
# Subcommand: extract
# ============================================================

def cmd_extract(args):
    """Extract CT-aware wiring diagrams from nLab pages."""
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load NER kernel
    ner_path = Path(args.ner_kernel)
    singles, multi_index, multi_count = None, None, 0
    if ner_path.exists():
        print(f"Loading NER kernel from {ner_path}...")
        singles, multi_index, multi_count = load_ner_kernel(ner_path)
        print(f"  {len(singles)} single + {multi_count} multi-word terms")
    else:
        print(f"NER kernel not found at {ner_path}, skipping term spotting.")

    print(f"\nExtracting wiring from nLab pages (limit={args.limit})...")

    # Aggregate stats
    total = Counter()
    env_type_freq = Counter()
    link_type_freq = Counter()
    cat_type_freq = Counter()
    n = 0

    out_path = outdir / "pages.json"
    with open(out_path, "w") as f:
        f.write("[\n")

        for page_id, name, content in iter_nlab_pages(args.pages_dir, args.limit):
            result = process_page(page_id, name, content, singles, multi_index)

            sep = ",\n" if n > 0 else ""
            f.write(sep + json.dumps(result, ensure_ascii=False))

            # Accumulate stats
            stats = result["stats"]
            total["environments"] += stats["n_environments"]
            total["typed_links"] += stats["n_typed_links"]
            total["diagrams"] += stats["n_diagrams"]
            total["discourse"] += stats["n_discourse"]
            total["categorical"] += stats["n_categorical"]
            total["ner_terms"] += stats["n_ner_terms"]
            env_type_freq.update(stats["env_types"])
            link_type_freq.update(stats["link_types"])
            for ct in stats["cat_types"]:
                cat_type_freq[ct] += 1

            n += 1
            if n % 50 == 0:
                print(f"  [{n}] envs={total['environments']} links={total['typed_links']} "
                      f"diags={total['diagrams']} cat={total['categorical']}")

        f.write("\n]")

    elapsed = time.time() - t0

    # Write manifest
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "nlab",
        "pages_processed": n,
        "limit": args.limit,
        "elapsed_seconds": round(elapsed),
        "totals": dict(total),
        "env_type_freq": dict(env_type_freq.most_common()),
        "link_type_freq": dict(link_type_freq.most_common()),
        "cat_type_freq": dict(cat_type_freq.most_common()),
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"nLab wiring extraction complete in {elapsed:.0f}s")
    print(f"  Pages:        {n}")
    print(f"  Environments: {total['environments']}")
    print(f"  Typed links:  {total['typed_links']}")
    print(f"  Diagrams:     {total['diagrams']}")
    print(f"  Discourse:    {total['discourse']}")
    print(f"  Categorical:  {total['categorical']}")
    print(f"  NER terms:    {total['ner_terms']}")

    if env_type_freq:
        print(f"\n  Environment types:")
        for etype, count in env_type_freq.most_common():
            print(f"    {etype}: {count}")

    if cat_type_freq:
        print(f"\n  Categorical patterns detected:")
        for ctype, count in cat_type_freq.most_common():
            print(f"    {ctype}: {count}")

    print(f"\nOutput: {outdir}/")
    for fp in sorted(outdir.iterdir()):
        if fp.is_file():
            print(f"  {fp.name:30s} {os.path.getsize(fp)/1e6:8.3f} MB")


# ============================================================
# Subcommand: reference
# ============================================================

def cmd_reference(args):
    """Build CT reference dictionary from extracted wiring data."""
    input_dir = Path(args.input_dir)
    pages_path = input_dir / "pages.json"

    if not pages_path.exists():
        print(f"No pages.json found in {input_dir}. Run 'extract' first.")
        sys.exit(1)

    print(f"Building CT reference from {pages_path}...")

    with open(pages_path) as f:
        pages = json.load(f)

    # Aggregate by categorical pattern type
    patterns = defaultdict(lambda: {
        "instances": [],
        "required_links": Counter(),
        "typical_links": Counter(),
        "discourse_components": Counter(),
        "discourse_wires": Counter(),
        "n_diagrams": 0,
    })

    # Aggregate link weights across all pages
    link_weights = defaultdict(lambda: Counter())

    for page in pages:
        page_id = page["page_id"]

        # Collect link weights (strip link/ prefix for cleaner keys)
        for link in page["typed_links"]:
            target = link["hx/content"].get("target_name", "")
            ltype = link["hx/type"]
            if target and ltype != "link/navigation":
                short_type = ltype.replace("link/", "")
                link_weights[target.lower()][short_type] += 1

        # Aggregate categorical patterns
        for cat in page["categorical"]:
            cat_type = cat["hx/type"]
            pat = patterns[cat_type]
            pat["instances"].append(page_id)

            # Collect links from this page's definitions
            for link in page["typed_links"]:
                target = link["hx/content"].get("target_name", "")
                if link["hx/type"] == "link/definition-ref":
                    pat["required_links"][target.lower()] += 1
                elif link["hx/type"] != "link/navigation":
                    pat["typical_links"][target.lower()] += 1

            # Collect discourse signatures
            for disc in page["discourse"]:
                if disc["hx/role"] == "component":
                    pat["discourse_components"][disc["hx/type"]] += 1
                elif disc["hx/role"] == "wire":
                    pat["discourse_wires"][disc["hx/type"]] += 1

            pat["n_diagrams"] += len(page["diagrams"])

    # Build output
    reference = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pages_processed": len(pages),
            "pattern_types": len(patterns),
        },
        "patterns": {},
        "link_weights": {k: dict(v.most_common(20)) for k, v in
                         sorted(link_weights.items(), key=lambda x: -sum(x[1].values()))[:500]},
    }

    for cat_type, pat in sorted(patterns.items()):
        n_inst = len(pat["instances"])
        reference["patterns"][cat_type] = {
            "instances": pat["instances"],
            "instance_count": n_inst,
            "required_links": [link for link, count in pat["required_links"].most_common(10)
                               if count >= max(1, n_inst * 0.1)],
            "typical_links": [link for link, count in pat["typical_links"].most_common(20)
                              if count >= max(1, n_inst * 0.05)],
            "discourse_signature": {
                "components": dict(pat["discourse_components"].most_common(10)),
                "wires": dict(pat["discourse_wires"].most_common(10)),
            },
            "avg_diagrams": pat["n_diagrams"] / max(1, n_inst),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)

    print(f"\nCT reference written to {out_path}")
    print(f"  Pages: {len(pages)}")
    print(f"  Pattern types: {len(patterns)}")
    for cat_type, pat_data in sorted(reference["patterns"].items()):
        print(f"    {cat_type}: {pat_data['instance_count']} instances, "
              f"{len(pat_data['required_links'])} required links")
    print(f"  Link weights: {len(reference['link_weights'])} terms")


# ============================================================
# Subcommand: evaluate
# ============================================================

def load_threads(path):
    """Load threads from JSONL (one thread per line) or JSON array.

    Normalises to a list of dicts with keys:
        id, body, tags, accepted_answer_id, site, topic, answers[]
    where each answer has: id, body, score, is_accepted
    """
    threads = []
    p = Path(path)
    # Try JSONL first
    with open(p) as f:
        first_char = f.read(1)
    if first_char == '{':
        # JSONL
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                threads.append(json.loads(line))
    else:
        # JSON array
        with open(p) as f:
            threads = json.load(f)

    normalised = []
    for t in threads:
        # Handle Codex SE thread format: {question: {...}, answers: [...]}
        if "question" in t:
            q = t["question"]
            accepted_id = q.get("accepted_answer_id")
            answers = []
            for a in t.get("answers", []):
                a_id = a.get("id")
                answers.append({
                    "id": a_id,
                    "body": a.get("body_text", a.get("body", "")),
                    "score": a.get("score", 0),
                    "is_accepted": str(a_id) == str(accepted_id) if accepted_id else False,
                })
            normalised.append({
                "id": q.get("id", t.get("thread_id", "unknown")),
                "body": q.get("body_text", q.get("body", "")),
                "title": q.get("title", ""),
                "tags": q.get("tags", []),
                "site": t.get("site", ""),
                "topic": t.get("topic", ""),
                "answers": answers,
            })
        else:
            # Direct format: {id, body, tags, answers[{id, body, score, is_accepted}]}
            normalised.append(t)
    return normalised


def evaluate_threads(threads, reference, singles, multi_index):
    """Score a list of normalised threads against CT reference.

    Returns (results_list, summary_dict).
    """
    patterns = reference["patterns"]
    results = []

    for question in threads:
        q_id = question.get("id", "unknown")
        q_text = question.get("body", "")
        q_title = question.get("title", "")
        q_full = q_title + " " + q_text
        answers = question.get("answers", [])

        # Determine which categorical patterns are relevant
        relevant_patterns = []
        q_lower = q_full.lower()
        for cat_type, pat_data in patterns.items():
            score = 0
            for link in pat_data.get("required_links", []):
                if link in q_lower:
                    score += 2
            for link in pat_data.get("typical_links", []):
                if link in q_lower:
                    score += 1
            if score > 0:
                relevant_patterns.append((cat_type, score, pat_data))
        relevant_patterns.sort(key=lambda x: -x[1])

        # Score each answer
        scored_answers = []
        for answer in answers:
            a_id = answer.get("id", "unknown")
            a_text = answer.get("body", "")
            is_accepted = answer.get("is_accepted", False)
            a_score = answer.get("score", 0)

            # 1. Term overlap
            term_score = 0
            if singles:
                terms = spot_terms(a_text, singles, multi_index or {})
                term_score = len(terms)

            # 2. Structural match: discourse wiring in answer
            discourse_scopes = detect_scopes(f"ans-{a_id}", a_text)
            discourse_wires = detect_wires(f"ans-{a_id}", a_text)
            discourse_ports = detect_ports(f"ans-{a_id}", a_text)
            struct_score = len(discourse_scopes) + len(discourse_wires) * 0.5

            # 3. Completeness vs best matching pattern
            completeness_score = 0
            a_lower = a_text.lower()
            best_pattern = relevant_patterns[0] if relevant_patterns else None
            if best_pattern:
                cat_type, _, pat_data = best_pattern
                required = pat_data.get("required_links", [])
                covered = sum(1 for link in required if link in a_lower)
                completeness_score = covered / max(1, len(required))

            # 4. Diagram presence
            diagram_score = 1.0 if re.search(
                r'\\begin\{tikzcd\}|\\array\s*\{|\\xymatrix|\\begin\{CD\}|'
                r'\\xrightarrow|\\xleftarrow|\\overset\{.*?\}\{\\to\}',
                a_text) else 0.0

            combined = (
                term_score * 0.3 +
                struct_score * 0.3 +
                completeness_score * 20.0 +
                diagram_score * 5.0
            )

            scored_answers.append({
                "answer_id": a_id,
                "is_accepted": is_accepted,
                "community_score": a_score,
                "ct_score": round(combined, 2),
                "breakdown": {
                    "term_overlap": term_score,
                    "structural": round(struct_score, 2),
                    "completeness": round(completeness_score, 2),
                    "diagram": diagram_score,
                },
            })

        scored_answers.sort(key=lambda x: -x["ct_score"])

        ct_rank_of_accepted = None
        for rank, sa in enumerate(scored_answers):
            if sa["is_accepted"]:
                ct_rank_of_accepted = rank
                break

        results.append({
            "question_id": q_id,
            "title": question.get("title", ""),
            "relevant_patterns": [(p[0], p[1]) for p in relevant_patterns[:3]],
            "n_answers": len(scored_answers),
            "scored_answers": scored_answers,
            "ct_rank_of_accepted": ct_rank_of_accepted,
            "ct_top_is_accepted": scored_answers[0]["is_accepted"] if scored_answers else None,
        })

    n_questions = len(results)
    n_with_accepted = sum(1 for r in results if r["ct_rank_of_accepted"] is not None)
    n_multi_answer = sum(1 for r in results if r["n_answers"] > 1)
    n_multi_with_accepted = sum(1 for r in results
                                 if r["n_answers"] > 1 and r["ct_rank_of_accepted"] is not None)
    n_top_match = sum(1 for r in results if r.get("ct_top_is_accepted"))
    n_top_match_multi = sum(1 for r in results
                            if r.get("ct_top_is_accepted") and r["n_answers"] > 1)

    # Mean CT score for accepted vs non-accepted
    accepted_scores = []
    rejected_scores = []
    for r in results:
        for sa in r["scored_answers"]:
            if sa["is_accepted"]:
                accepted_scores.append(sa["ct_score"])
            else:
                rejected_scores.append(sa["ct_score"])

    summary = {
        "n_questions": n_questions,
        "n_with_accepted": n_with_accepted,
        "n_multi_answer": n_multi_answer,
        "n_multi_with_accepted": n_multi_with_accepted,
        "n_ct_top_is_accepted": n_top_match,
        "n_ct_top_is_accepted_multi": n_top_match_multi,
        "accuracy_all": round(n_top_match / max(1, n_with_accepted), 3),
        "accuracy_multi": round(n_top_match_multi / max(1, n_multi_with_accepted), 3),
        "mean_ct_score_accepted": round(sum(accepted_scores) / max(1, len(accepted_scores)), 2),
        "mean_ct_score_rejected": round(sum(rejected_scores) / max(1, len(rejected_scores)), 2),
    }

    return results, summary


def cmd_evaluate(args):
    """Score SE/MO answers against CT reference structures.

    Accepts a single JSONL file or a directory of JSONL files for 2x2 comparison.
    """
    ref_path = Path(args.reference)
    answers_path = Path(args.answers)

    if not ref_path.exists():
        print(f"Reference file not found: {ref_path}")
        sys.exit(1)

    print(f"Loading reference from {ref_path}...")
    with open(ref_path) as f:
        reference = json.load(f)

    # Load NER kernel
    ner_path = Path(args.ner_kernel)
    singles, multi_index = None, None
    if ner_path.exists():
        singles, multi_index, _ = load_ner_kernel(ner_path)

    # Collect input files
    if answers_path.is_dir():
        input_files = sorted(answers_path.glob("*.jsonl"))
    else:
        input_files = [answers_path]

    if not input_files:
        print(f"No JSONL files found at {answers_path}")
        sys.exit(1)

    all_evaluations = {}
    for input_file in input_files:
        label = input_file.stem  # e.g. "math.stackexchange.com__category-theory"
        print(f"\nEvaluating {label}...")
        threads = load_threads(input_file)
        print(f"  {len(threads)} threads loaded")
        results, summary = evaluate_threads(threads, reference, singles, multi_index)
        all_evaluations[label] = {
            "file": str(input_file),
            "summary": summary,
            "results": results,
        }
        print(f"  Threads: {summary['n_questions']}, "
              f"with accepted: {summary['n_with_accepted']}, "
              f"multi-answer: {summary['n_multi_answer']}")
        print(f"  CT top = accepted (all): {summary['n_ct_top_is_accepted']}"
              f"/{summary['n_with_accepted']}"
              f" ({summary['accuracy_all']:.0%})")
        print(f"  CT top = accepted (multi): {summary['n_ct_top_is_accepted_multi']}"
              f"/{summary['n_multi_with_accepted']}"
              f" ({summary['accuracy_multi']:.0%})")
        print(f"  Mean CT score: accepted={summary['mean_ct_score_accepted']}"
              f" rejected={summary['mean_ct_score_rejected']}")

    # Write combined output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reference": str(ref_path),
            "n_input_files": len(input_files),
        },
        "evaluations": all_evaluations,
    }

    # Print 2x2 comparison table if we have 4 files
    if len(all_evaluations) >= 2:
        print(f"\n{'='*70}")
        print(f"{'2x2 CT Reference Evaluation':^70}")
        print(f"{'='*70}")
        print(f"{'Dataset':<50} {'Acc(all)':>8} {'Acc(2+)':>8} {'CT_acc':>7} {'CT_rej':>7}")
        print(f"{'-'*70}")
        for label, ev in sorted(all_evaluations.items()):
            s = ev["summary"]
            print(f"{label:<50} "
                  f"{s['accuracy_all']:>7.0%} "
                  f"{s['accuracy_multi']:>7.0%} "
                  f"{s['mean_ct_score_accepted']:>7.1f} "
                  f"{s['mean_ct_score_rejected']:>7.1f}")
        print(f"{'='*70}")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation written to {out_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="nLab CT-Backed Wiring Extraction Pipeline")
    sub = parser.add_subparsers(dest="command")

    # extract
    ext_p = sub.add_parser("extract", help="Extract wiring from nLab pages")
    ext_p.add_argument("--pages-dir", default=str(NLAB_PAGES))
    ext_p.add_argument("--output-dir", "-o", default="data/nlab-wiring")
    ext_p.add_argument("--ner-kernel", default=str(NER_KERNEL))
    ext_p.add_argument("--limit", type=int, default=None,
                       help="Max pages to process")

    # reference
    ref_p = sub.add_parser("reference", help="Build CT reference dictionary")
    ref_p.add_argument("--input-dir", default="data/nlab-wiring")
    ref_p.add_argument("--output", default="data/nlab-ct-reference.json")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Score answers against reference")
    eval_p.add_argument("--reference", default="data/nlab-ct-reference.json")
    eval_p.add_argument("--answers", required=True,
                        help="Path to JSONL file or directory of JSONL files")
    eval_p.add_argument("--ner-kernel", default=str(NER_KERNEL))
    eval_p.add_argument("--output", default="data/ct-evaluation.json")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "reference":
        cmd_reference(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
