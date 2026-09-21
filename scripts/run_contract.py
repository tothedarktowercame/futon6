#!/usr/bin/env python3
"""The run contract: everything that decides WHAT a Mark7 run produces.

Two runs are comparable exactly when their contract hashes match. Machine and
serving configuration, advisory or required, is not part of this file: it lives
in futon6_config, and nothing there may switch a behaviour on or off.

Why this is its own module: mark7probe-20260921 ran on code that contained the
span-quotation contract and produced graphs under the ORIGINAL retype contract,
because its candidates carried no spans and the loop fell back without a word.
The run looked current and was not. A contract feature whose input is missing
now refuses the run instead.

Select a contract with FUTON6_RUN_CONTRACT; unset means RECOMMENDED, the latest
behaviour. Anything that departs from the selected contract is recorded as a
deviation next to it, so an experiment is visible rather than mistaken for the
baseline.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

CONTRACT_ENV = "FUTON6_RUN_CONTRACT"

CONTRACTS: dict[str, dict] = {
    # v1 (0919b probe, 20260921 superpod run): the model retyped each node's
    #     mathematics inside a JSON string, whose escape alphabet cannot spell
    #     \Sigma, \alpha or \in -- the decoder substituted legal commands.
    # v2: nodes quoted source LINES; too coarse, a line holds a hypothesis and
    #     its conclusion, so 6.0% of edges read "this text implies itself".
    # v3: nodes select S1's clause-sized marked spans; the text is taken from
    #     the source by offset, never typed by the model.
    "mark7-v3": {
        "model": {
            "checkpoint": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            "served-as": "mark4-70b",
        },
        "decoding": {"temperature": 0, "max-tokens": 8192},
        "candidates": {"schema": "iatc-candidate/v4-proof", "requires": ["spans"]},
        "quotation": "spans",
        "gate-retries": 1,
    },
}

RECOMMENDED = "mark7-v3"


def contract_id() -> str:
    return os.environ.get(CONTRACT_ENV) or RECOMMENDED


def spec(cid: str | None = None) -> dict:
    cid = cid or contract_id()
    if cid not in CONTRACTS:
        raise SystemExit(f"{CONTRACT_ENV}={cid!r} is not a run contract; "
                         f"known: {', '.join(sorted(CONTRACTS))}")
    return copy.deepcopy(CONTRACTS[cid])


def digest(cid: str | None = None) -> str:
    cid = cid or contract_id()
    canonical = json.dumps({"id": cid, **spec(cid)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def active() -> dict:
    """The contract this process runs under, as it goes into a run record."""
    cid = contract_id()
    return {"id": cid, "sha256": digest(cid), "recommended": cid == RECOMMENDED, **spec(cid)}


def deviations(actual: dict, cid: str | None = None) -> list[str]:
    """Contract terms this run departs from. `actual` holds what the run used."""
    want = spec(cid)
    out = []
    if "gate-retries" in actual and actual["gate-retries"] != want["gate-retries"]:
        out.append(f"gate-retries {actual['gate-retries']} (contract {want['gate-retries']})")
    if "max-tokens" in actual and actual["max-tokens"] != want["decoding"]["max-tokens"]:
        out.append(f"max-tokens {actual['max-tokens']} (contract {want['decoding']['max-tokens']})")
    if "model" in actual and actual["model"] != want["model"]["served-as"]:
        out.append(f"model {actual['model']!r} (contract {want['model']['served-as']!r})")
    return out


def missing_inputs(candidate: dict, cid: str | None = None) -> list[str]:
    """What this candidate lacks that the contract needs. Non-empty means refuse."""
    return [field for field in spec(cid)["candidates"]["requires"] if not candidate.get(field)]


if __name__ == "__main__":
    print(json.dumps(active(), indent=2))
