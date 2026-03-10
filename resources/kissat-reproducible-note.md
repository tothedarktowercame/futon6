# Kissat Reproducible Research Note

This note captures how we vend the `kissat` SAT solver inside the futon
workspace and how to reproduce the FM-001 `n=5` run that generated the current
witness.

## 1. Installation

We keep `kissat` in the top-level `~/code` tree so every repo can reuse it.

```bash
cd ~/code
git clone https://github.com/arminbiere/kissat.git
cd kissat
./configure              # creates ./build
cd build
make -j$(nproc)          # builds ./kissat (v4.0.4 as of 2026-03-09)
```

The resulting binary lives at `~/code/kissat/build/kissat`. The build uses only
standard GCC, so no extra dependencies are needed beyond `build-essential`.

## 2. Example: FM-001 `n=5`

The harness assets are already committed under
`futon6/data/frontiermath-pilot/harness/`. We keep both CNFs and decoded
witnesses there.

```bash
cd ~/code/futon6/data/frontiermath-pilot/harness
# optional: gunzip if you only have .gz
gunzip -k FM001-n5.cnf.gz

# run kissat with a 1800 s wall-clock cap and tee the log
stdbuf -oL ~/code/kissat/build/kissat --time=1800 FM001-n5.cnf \
  | tee FM001-n5.kissat.log
```

The committed log `FM001-n5.kissat.log` shows a SAT result in 11 seconds and
the solver statistics. To turn the printed assignment into a JSON witness, run
the existing harness decoder:

```bash
cd ~/code/futon6
.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.fm001.ramsey_book_sat import build_instance, decode_model, write_witness

log_path = Path("/home/joe/code/futon6/data/frontiermath-pilot/harness/FM001-n5.kissat.log")
vals = []
with log_path.open() as fh:
    for line in fh:
        if line.startswith("v "):
            vals.extend(int(x) for x in line.split()[1:] if x not in {"0"})

cnf, edges, _ = build_instance(5)
assignment = decode_model(vals, edges)
write_witness(Path("/home/joe/code/futon6/data/frontiermath-pilot/harness/n5-witness.json"),
              assignment, 5)
PY
```

Both the log and the generated witness are tracked in git, so collaborators can
verify hashes listed in `data/frontiermath-pilot/harness/README.md`.

## 3. Tips & Extensions

- Use `--time=<seconds>` to cap runtime; for larger CNFs (e.g. `FM001-n6`) we
  currently use `--time=3600`.
- `kissat` prints assignments by default; add `--no-color` if piping into tools
  that dislike ANSI escapes.
- Always copy solver outputs into the harness directory and update the hash
  table so remote collaborators can confirm integrity.
