#!/usr/bin/env python3
"""Concurrent driver for the mark3 IATC loop.

This keeps mark3_iatc_loop's per-paper candidate -> model -> repair/gate ->
rung-2 sidecar -> retry semantics, but schedules multiple papers concurrently
against one OpenAI-compatible vLLM server.  The concurrency cap is the Linode
GPU safety valve: vLLM continuous-batches the in-flight requests, while the cap
keeps KV-cache pressure bounded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import mark3_iatc_loop as loop  # noqa: E402


@dataclass(frozen=True)
class PaperResult:
    paper_id: str
    status: str
    message: str
    graph_path: Path | None = None
    rung2_path: Path | None = None


class InFlightMeter:
    """Small test hook for proving the driver respects --concurrency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.max_seen = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)

    def exit(self) -> None:
        with self._lock:
            self.current -= 1


def candidate_paths(candidates_dir: Path) -> list[Path]:
    return sorted(candidates_dir.glob("*.candidate.json"))


def graph_path(outdir: Path, paper_id: str) -> Path:
    return outdir / f"{paper_id}.edn"


def rung2_path(outdir: Path, paper_id: str) -> Path:
    return outdir / f"{paper_id}.rung2.edn"


def is_complete(outdir: Path, paper_id: str) -> bool:
    return graph_path(outdir, paper_id).exists() and rung2_path(outdir, paper_id).exists()


def repair_attempt(path: Path) -> None:
    subprocess.run(["bb", str(loop.REPAIR), str(path)], capture_output=True, text=True)


def call_backend(prompt: str, cand: dict, attempt: int, *, backend: str, model: str) -> str:
    if backend == "stub":
        return loop.call_stub(prompt, cand, attempt)
    return loop.call_openai(prompt, cand, attempt, model)


def process_candidate(
    cf: Path,
    *,
    outdir: Path,
    tmp: Path,
    seeds: str,
    backend: str,
    model: str,
    rung2_gate: bool,
    meter: InFlightMeter | None = None,
) -> PaperResult:
    cand = json.loads(cf.read_text())
    pid = cand["paper-id"]
    final = graph_path(outdir, pid)
    rung2_report = rung2_path(outdir, pid)
    if is_complete(outdir, pid):
        return PaperResult(pid, "skip", "existing graph+rung2 sidecar", final, rung2_report)

    prompt = loop.build_prompt(cand, seeds)
    status, last_err = "fail", ""
    for attempt in range(loop.MAX_ATTEMPTS):
        attempt_prompt = (
            prompt
            if attempt == 0
            else prompt + f"\n\n# previous attempt failed the gate:\n{last_err}\n# fix it and re-emit ONLY the EDN."
        )
        if meter is not None:
            meter.enter()
        try:
            resp = call_backend(attempt_prompt, cand, attempt, backend=backend, model=model)
        finally:
            if meter is not None:
                meter.exit()

        edn = loop.extract_edn(resp)
        if not edn:
            last_err = "no EDN map found in response"
            continue
        ap = tmp / f"{pid}.attempt{attempt}.edn"
        ap.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(edn)
        repair_attempt(ap)
        edn = ap.read_text()
        ok, err = loop.candidate_check(edn, cand)
        if ok:
            ok, err = loop.gate_one(ap)
        if ok:
            if rung2_gate:
                attempt_report = tmp / f"{pid}.attempt{attempt}.rung2.edn"
                r2_ok, r2_msg = loop.run_rung2(ap, attempt_report, gate=True)
                if not r2_ok:
                    last_err = r2_msg
                    continue
            final.write_text(edn)
            _r2_ok, r2_msg = loop.run_rung2(final, rung2_report, gate=False)
            status = "pass"
            last_err = f"attempt {attempt}; {r2_msg}; report {rung2_report.name}"
            break
        last_err = err
    return PaperResult(pid, status, last_err, final if final.exists() else None, rung2_report if rung2_report.exists() else None)


def batch_substance_gate(paths: list[Path], outdir: Path) -> tuple[bool, str]:
    sub_paths = [str(p) for p in paths] or [str(outdir)]
    sub = subprocess.run(
        [sys.executable, str(loop.SUBSTANCE), *sub_paths, "--kind", "iatc"],
        capture_output=True,
        text=True,
    )
    return sub.returncode == 0, sub.stdout.strip()[-400:]


def run(args: argparse.Namespace, *, meter: InFlightMeter | None = None) -> int:
    cands = candidate_paths(Path(args.candidates))
    if not cands:
        print("no candidates found", file=sys.stderr)
        return 2
    if not loop.require_enriched(cands):
        return 2

    seeds = loop.load_seeds(args.shots)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / ".attempts"
    tmp.mkdir(exist_ok=True)

    max_workers = max(1, int(args.concurrency))
    results: list[PaperResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                process_candidate,
                cf,
                outdir=outdir,
                tmp=tmp,
                seeds=seeds,
                backend=args.backend,
                model=args.model,
                rung2_gate=args.rung2_gate,
                meter=meter,
            )
            for cf in cands
        ]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"  {result.paper_id}: {result.status} ({result.message[:80]})")

    accepted = [r.graph_path for r in results if r.status in {"pass", "skip"} and r.graph_path]
    print("\n=== batch substance gate (cross-item) ===")
    batch_ok, batch_msg = batch_substance_gate(accepted, outdir)
    if batch_msg:
        print(batch_msg)

    n_pass = sum(1 for r in results if r.status == "pass")
    n_skip = sum(1 for r in results if r.status == "skip")
    print(
        f"\nconcurrent-loop: pass={n_pass} skip={n_skip} fail={len(results) - n_pass - n_skip}/"
        f"{len(results)} · batch-substance {'PASS' if batch_ok else 'FAIL'}"
    )
    print("Next: OWNER REVIEW — spot-check faithfulness against source at the anchors.")
    return 0 if (n_pass + n_skip == len(results) and batch_ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(REPO / "data" / "iatc-candidates"))
    ap.add_argument("--out", default=str(REPO / "data" / "iatc-argument-graphs" / "loop-run"))
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--rung2-gate",
        action="store_true",
        help="Hard-gate rung-2 semantic failures; default records the profile/verdict only.",
    )
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
