from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_discovers_xelatex_outside_minimal_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    style_dir = tmp_path / "style"
    fake_bin = tmp_path / "bin"
    tex_bin = tmp_path / "texlive" / "2099" / "bin" / "test-arch"
    for directory in (input_dir, style_dir, fake_bin, tex_bin):
        directory.mkdir(parents=True, exist_ok=True)

    (input_dir / "proof.md").write_text("# Proof\n\n$1+1=2$\n")
    (style_dir / "math-proofread-style.sty").write_text("% fixture\n")
    xelatex = tex_bin / "xelatex"
    xelatex.write_text("#!/bin/sh\nexit 0\n")
    xelatex.chmod(0o755)

    engine_record = tmp_path / "engine.txt"
    pandoc = fake_bin / "pandoc"
    pandoc.write_text(
        "#!/bin/sh\n"
        "out=\n"
        "for arg in \"$@\"; do\n"
        "  case $arg in --pdf-engine=*) printf '%s\\n' \"${arg#*=}\" > \"$ENGINE_RECORD\";; esac\n"
        "done\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = -o ]; then shift; out=$1; fi\n"
        "  shift\n"
        "done\n"
        "printf '%%PDF-1.7\\n' > \"$out\"\n"
    )
    pandoc.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "TEXLIVE_ROOT": str(tmp_path / "texlive"),
            "MATH_PROOFREAD_STYLE_DIR": str(style_dir),
            "ENGINE_RECORD": str(engine_record),
            "JOBS": "1",
        }
    )
    subprocess.run(
        [str(ROOT / "scripts" / "build-apm-highlighted-pdfs.sh"),
         str(input_dir), str(output_dir)],
        check=True,
        env=env,
    )

    assert engine_record.read_text().strip() == str(xelatex)
    assert (output_dir / "proof.pdf").read_bytes().startswith(b"%PDF-1.7")
