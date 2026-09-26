"""efe_carpet_controls.py — pure helpers for the EFE carpet global control panel
(M-the-perfect-crime, packet E-kimi-task-45, 2026-09-26).

Standard library only; no reads at import time so tests can load this module
standalone. Used by mission_efe_field.py to stamp per-mission data-band /
data-status attributes that the page's inline JS filters on.
"""
import re

# A status line marks a mission DONE when it opens with one of these words (after any
# markdown emphasis), or says its last lifecycle phase is complete ("DOCUMENT complete").
# Word-boundary matched, case-insensitive: "incomplete" is not "complete". Anchored at the
# start because open missions report finished phases mid-line ("HEAD complete; IDENTIFY
# draft pending" is open): 28 such lines were classed done by an unanchored match
# (claude-12 review of 2ade473, against mission-activity.json status_line values).
DONE_WORDS = ("archived", "complete", "completed", "superseded", "closed", "retired", "folded")
_DONE_RE = re.compile(r"^\W*(" + "|".join(DONE_WORDS) + r")\b|\bdocument\s+complete\b",
                      re.IGNORECASE)


def hub_band(grid, fmax, nb, step, x, y):
    """Level-set band (0..nb-1) of the metric field at the grid cell under (x, y)."""
    gx = min(max(int(round(x / step)), 0), len(grid[0]) - 1)
    gy = min(max(int(round(y / step)), 0), len(grid) - 1)
    return min(nb - 1, int(grid[gy][gx] / fmax * nb))


def status_class(status_line):
    """Classify a mission's free-text status line: "done" | "open" | "unknown"."""
    if not status_line or not str(status_line).strip():
        return "unknown"
    return "done" if _DONE_RE.search(str(status_line)) else "open"
