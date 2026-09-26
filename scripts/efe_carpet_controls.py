"""efe_carpet_controls.py — pure helpers for the EFE carpet global control panel
(M-the-perfect-crime, packet E-kimi-task-45, 2026-09-26).

Standard library only; no reads at import time so tests can load this module
standalone. Used by mission_efe_field.py to stamp per-mission data-band /
data-status attributes that the page's inline JS filters on.
"""
import re

# Words that mark a mission's status line as DONE. Word-boundary matched
# (case-insensitive), never substring matched: "incomplete" must NOT count as
# "complete". Reviewed by claude-12 against mission-activity.json status_line
# values (e.g. "archived", "COMPLETE (SUPERSEDED by …)", "CLOSED 2026-06-12").
DONE_WORDS = ("archived", "complete", "completed", "superseded", "closed", "retired")
_DONE_RE = re.compile(r"\b(" + "|".join(DONE_WORDS) + r")\b", re.IGNORECASE)


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
