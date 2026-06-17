import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mark4_apm_structure_coverage.py"


def scope(pid, typ, *symbols):
    return {
        "hx/id": f"{pid}:{typ}",
        "hx/type": typ,
        "hx/ends": [{"role": "symbol", "latex": s} for s in symbols],
    }


def test_mark4_structure_coverage_flavours_and_metric(tmp_path):
    proof = {
        "p1": [
            scope("p1", "bind/integral", "x"),
            scope("p1", "quant/universal", "epsilon"),
        ],
    }
    eprint = {
        "e1": [
            scope("e1", "bind/integral", "y"),
            scope("e1", "quant/universal", "epsilon"),
        ],
    }
    proof_path = tmp_path / "proof.json"
    eprint_path = tmp_path / "eprint.json"
    proof_path.write_text(json.dumps(proof))
    eprint_path.write_text(json.dumps(eprint))

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--proof-scopes",
            str(proof_path),
            "--eprint-scopes",
            str(eprint_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    summary = json.loads(proc.stdout)
    assert summary["type_only"]["mean"] == 1.0
    assert summary["type_any_symbol"]["mean"] == 0.5
    assert summary["type_multichar"]["mean"] == 0.5
    assert summary["chosen_metric"] == "type_multichar"
    assert summary["gate"]["metric"] == "type_multichar"
