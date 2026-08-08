#!/usr/bin/env python3
"""Recover a JSON object from model prose. Never raises.

Fourth shared helper extracted for the reason the other three were (`edn_compat`
reader divergence, `paper_ids` id families, `run_artifacts` sidecar selection): a
rule needed in several places had been written separately in each, and the copies
behaved differently on the same input.

Every LLM caller here asks for one JSON object and gets back something close to
one. The two failure modes actually observed on GLM-4.5-Air:

  {"pattern": null, descriptions: "..."}      <- bare property name
  {"pattern": "x", "confidence": 0.8,}        <- trailing comma

The first is the one that aborted a 98-proof run on 2026-08-07: `json.loads`
raises *"Expecting property name enclosed in double quotes"*, which was
misdiagnosed as a trailing comma because the exception type was read instead of
a response. Both are repaired here.

Two subtler points the individual copies got wrong:

- **Greedy `\\{.*\\}` is wrong.** It spans the first `{` to the *last* `}`, so a
  reply containing commentary plus a verdict yields garbage rather than the
  verdict. This scans balanced-free candidates and prefers the LAST that parses,
  which is the one models put their answer in.
- **An unparseable reply is an outcome, not an error.** "I could not classify
  this" is a legitimate result; raising turns it into a lost run, and silently
  substituting a template turns it into a fabricated one. Callers get `default`
  back and can count how often that happened.
"""
from __future__ import annotations

import json
import re
from typing import Any

_OBJ = re.compile(r"\{[^{}]*\}", re.S)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_BARE_KEY = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)')


def _repairs(candidate: str):
    yield candidate
    c = _TRAILING_COMMA.sub(r"\1", candidate)
    yield c
    yield _BARE_KEY.sub(r'\1"\2"\3', c)


def parse_object(text: str | None, default: Any = None) -> Any:
    """Last well-formed JSON object in `text`, repairing near-misses.

    Returns a *copy* of `default` when nothing parses, so a caller's sentinel
    dict cannot be mutated by whoever receives it.
    """
    for candidate in reversed(_OBJ.findall(text or "")):
        for attempt in _repairs(candidate):
            try:
                value = json.loads(attempt)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return dict(default) if isinstance(default, dict) else default
