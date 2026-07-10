#!/usr/bin/env python3
"""Shared transcript-provenance classifier — the ONE operator/agent/harness test (E-patch-agent-
evidence-leaks DERIVE-1).

The Evidence Landscape (~/.claude/projects/*/*.jsonl) commingles three kinds of `user` turn:
OPERATOR (Joe), AGENT (inter-agent bells/whistles Agency delivers as ordinary user turns), and
HARNESS (task-notifications, system-reminders, autorunner control prompts). Every miner that wants
the operator's voice must exclude the other two — and they were doing it DIVERGENTLY (c_mine_joint
`read_pairs` keyed on promptSource; meme_mine_runner `read_asks` keyed on AUTO_CALLERS+ASK-cue), so a
bell-phrased-as-request leaked into the forward run (~5%+). This module is the single test both call,
seeded by the validated c_mine_joint patch and memory/reference_transcript_operator_provenance.

    classify(record) -> "operator" | "agent" | "harness" | "unknown"
    is_operator(record) -> bool

Precedence — provenance metadata first; body heuristic ONLY for legacy turns predating promptSource:
    Caller: claude-N/codex-N   -> agent      (explicit agent mesh surface)
    Caller: joe                -> operator   (Joe's own mesh input — sdk, but carries Caller:joe)
    Caller in AUTO_CALLERS     -> harness     (auto-bellback / cron / heartbeat)
    promptSource == typed      -> operator   (Joe typed interactively)
    promptSource == system     -> harness    (task-notification / system-reminder inject)
    promptSource == sdk        -> agent      (programmatic bell; Joe's sdk already caught by Caller:joe)
    legacy (<none>/queued/—)   -> body heuristic (HARNESS_AUTHORED / AGENT_AUTHORED / PLUMBING) else operator
KEY: mentioning an agent != being an agent. "codex-3 is idle, whistle it" is Joe in the third person
(KEEP); "claude-1 -> claude-2: SCOPE CHANGE" is an agent bell (DROP). Filter on AUTHORSHIP, not the id.
"""
import re

CALLER = re.compile(r"Caller:\s*(\S+)")
WRAP = re.compile(r"User message:\s*(.*)$", re.S)
AGENT_CALLER = re.compile(r"^(?:claude|codex)-\d+$")
AUTO_CALLERS = {"auto-bellback", "auto", "system", "cron", "heartbeat"}
PLUMBING = re.compile(r"bell delivered|belled to|auto-?bellback|job-id invoke|verdict belled|"
                      r"bell sent|🔔|finished job|surface:\s*bell|\(state:", re.I)
# Legacy body markers, split into the two non-operator kinds (the c_mine_joint seed lumped both into one):
HARNESS_AUTHORED = re.compile(r"^\s*<(?:task-notification|task-id|tool-use-id|system-reminder)"
                              r"|^\s*Reply with exactly:", re.I)              # harness inject / autorunner
AGENT_AUTHORED = re.compile(r"(?:claude|codex)-\d+\s*(?:→|->|=>|»)\s*\S"      # agent→target routing arrow
                            r"|^\s*(?:claude|codex)-\d+\s+(?:here|back|reporting|online|speaking|\()", re.I)


def raw_text(record):
    """The turn's text (handles str | content-block list); '' if none (e.g. a tool_result-only turn)."""
    c = (record.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return ""


def classify(record):
    raw = raw_text(record) or ""
    cm = CALLER.search(raw)
    caller = cm.group(1).lower() if cm else None
    psrc = record.get("promptSource")
    if caller and AGENT_CALLER.match(caller):
        return "agent"
    if caller == "joe":
        return "operator"
    if caller and caller in AUTO_CALLERS:
        return "harness"
    if psrc == "typed":
        return "operator"
    if psrc == "system":
        return "harness"
    if psrc == "sdk":
        return "agent"
    # legacy turns (promptSource <none>/queued/missing): fall back to body authorship markers
    body = raw
    wm = WRAP.search(raw)
    if wm:
        body = wm.group(1)
    if HARNESS_AUTHORED.search(body):
        return "harness"
    if AGENT_AUTHORED.search(body) or PLUMBING.search(body):
        return "agent"
    if not body.strip():
        return "unknown"
    return "operator"


def is_operator(record):
    """True iff the turn was authored by the operator (Joe) — what every belly/methods miner wants."""
    return classify(record) == "operator"
