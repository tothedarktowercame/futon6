r"""Regression tests for the escaped-dollar scan in normalize-math-prose.py.

On 2026-09-23 an unbalanced backtick left TeX source exposed to the prose
rules; the pi_N rule wrapped `\pi_1` without its leading backslash, and the
resulting `\$` made split_inline_math_dollar scan forever, appending an empty
part per pass.  84 KB of input reached 57 GB resident and throttled every
process sharing the cgroup, including the Agency JVM.

Each call that could stall runs in a subprocess under a memory cap and a
timeout, so a regression fails these tests in seconds instead of repeating the
original failure inside the test runner.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import queue as queue_mod
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "normalize-math-prose.py"
SPEC = importlib.util.spec_from_file_location("normalize_math_prose", SCRIPT)
assert SPEC and SPEC.loader
nmp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nmp)

_MEM_CAP_BYTES = 1024 * 1024 * 1024  # 1 GiB: ample for real input, fatal to a runaway
_TIMEOUT_S = 20


def _worker(out_q, fn_name, arg):
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (_MEM_CAP_BYTES, _MEM_CAP_BYTES))
    try:
        out_q.put(("ok", getattr(nmp, fn_name)(arg)))
    except MemoryError:
        out_q.put(("memory", None))


def call_bounded(fn_name, arg, timeout=_TIMEOUT_S):
    """Call nmp.<fn_name>(arg) in a capped subprocess; fail if it runs away."""
    ctx = multiprocessing.get_context("fork")
    out_q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(out_q, fn_name, arg))
    proc.start()
    try:
        status, value = out_q.get(timeout=timeout)
    except queue_mod.Empty:
        status, value = "stalled", None
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)
    if status == "stalled":
        pytest.fail(f"{fn_name}({arg!r}) did not return within {timeout}s")
    if status == "memory":
        pytest.fail(f"{fn_name}({arg!r}) exhausted the {_MEM_CAP_BYTES} byte cap")
    return value


ESCAPED_DOLLAR_CASES = [
    r"\$",
    r"cost is 5\$ per unit",
    r"a \$ b \$ c",
    r"\$\pi_{1}$ trailing",
    r"$x + y$ and \$ literal",
]


@pytest.mark.parametrize("text", ESCAPED_DOLLAR_CASES)
def test_split_inline_math_dollar_terminates_on_escaped_dollar(text):
    """The scan must advance past an escaped '$' instead of stalling on it."""
    parts = call_bounded("split_inline_math_dollar", text)
    assert "".join(p for _, p in parts) == text, "split must be lossless"
    assert all(p != "" for _, p in parts), "an empty part means the scan stalled"


def test_split_inline_math_dollar_still_finds_real_math():
    assert ("math", "$x+y$") in nmp.split_inline_math_dollar("plain $x+y$ text")


@pytest.mark.parametrize("tex", [r"\pi_1", r"\pi_2", r"\mu_k", r"\phi_2", r"\nu_i"])
def test_tex_commands_are_left_alone(tex):
    """A backslash means it is already TeX; the prose rules must not re-wrap it."""
    assert call_bounded("process_plain_text_segment", tex) == tex


@pytest.mark.parametrize(
    "prose,expected",
    [
        ("pi_1 is the fundamental group", r"$\pi_{1}$ is the fundamental group"),
        ("the map y_n converges", "the map $y_{n}$ converges"),
    ],
)
def test_prose_subscripts_still_convert(prose, expected):
    assert call_bounded("process_plain_text_segment", prose) == expected


def test_unbalanced_backtick_line_does_not_emit_escaped_dollar():
    """The exact line that wedged the machine on 2026-09-23."""
    line = r"and by the **lifting criterion** (`f_*(\pi_1\mathbb T) = H \subseteq"
    assert "\\$" not in call_bounded("process_line", line)


def test_pathological_line_completes():
    line = r"(`f_*(\pi_1\mathbb T) = H \subseteq \pi_1(X) \mu_k \phi_2"
    assert call_bounded("process_line", line) is not None
