"""Browser-safety budget for generated SVG/HTML exports.

The greatest-hits export once emitted ~1.6M DOM nodes in a single 189 MB HTML
file (one <circle>+<title>+<line> per scope-MARK across ~194 papers), which
OOM'd Firefox and tripped librsvg's 1M-element cap. The fix there was to
AGGREGATE geometry (one glyph per group); this module is the systemic backstop
so the next export can't ship that hazard silently — it fails at generation
time, on the generating machine, not in the reader's browser.

Usage (call right before writing the file):

    from viz_budget import guard_svg
    n = guard_svg(html, "greatest-hits")     # raises if over budget
    path.write_text(html)
"""
import re

# circle/line/rect/path/polygon/polyline/text/g — the elements that become DOM
# nodes. We count the cheap way (regex on the string) so it's free to call.
_SVG_ELEM_RE = re.compile(r"<(circle|line|rect|path|polygon|polyline|text|image)\b")

# Browsers render ~100k SVG nodes comfortably; librsvg caps at 1M. 150k is a
# generous ceiling for real exports while still catching the ~1.6M disaster
# (and anything within ~6x of it) loudly.
DEFAULT_MAX_ELEMENTS = 150_000


def svg_element_count(svg: str) -> int:
    return len(_SVG_ELEM_RE.findall(svg))


def guard_svg(svg: str, label: str = "export", max_elements: int = DEFAULT_MAX_ELEMENTS) -> int:
    """Return the element count; raise ValueError if it exceeds the budget."""
    n = svg_element_count(svg)
    if n > max_elements:
        raise ValueError(
            f"{label}: {n:,} SVG elements exceeds the browser-safety budget "
            f"({max_elements:,}). An HTML/SVG with this many DOM nodes OOMs "
            f"browsers and trips librsvg's 1M-element cap. Don't ship it: "
            f"AGGREGATE the geometry (one glyph per group, like the per-(paper,"
            f"kind) glyphs in warp_greatest_hits) or rasterise the flat geometry "
            f"to PNG. See viz_budget.py for the history."
        )
    return n
