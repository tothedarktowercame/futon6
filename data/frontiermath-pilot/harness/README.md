# FM-001 harness artifacts

Solver-facing inputs/outputs that live under `data/frontiermath-pilot/harness/`.
All hashes are SHA512 so remote collaborators can verify downloads before
launching SAT or checker runs.

| File | Description | SHA512 |
|------|-------------|--------|
| `FM001-n5.cnf.gz` | 18-vertex ($n=5$) book-Ramsey SAT instance generated 2026-03-09 with vertex-0 monotone symmetry breaker | `af9d0deadaa72ba55df332ba687e89c64b40167f32c7889460172346058e8083f75fd6c5d4be2b4e66742ea6828348dbca0d48b005956fd493523f153a3f8dbe` |
| `n5-witness.json` | Coloring for $n=5$ produced by `kissat 4.0.4` + decoder | `9a2e1c2cc98b501473dfa3e51e25f401fbadc1860c570afc246d09a7327c9e6b132be1fbc86bdc6ecb0c85ce2fdacfe38a2ae4c73234ae9c3790db727e225fdd` |
| `FM001-n6.cnf.gz` | 22-vertex ($n=6$) instance, unsolved (built with `--no-solve`) | `95e1883090b9f43ab1237f48fdff0d188714d22d8a0e6dc670931a475aa7fdbf9ca2602ec6cfdfcf0f82ead718ec6a1c5375972402e86ccc8e240812ddf03997` |
| `FM001-n7.cnf.gz` | 26-vertex ($n=7$) instance, unsolved (built with `--no-solve`) | `6460b7f8c03ac06da7d274596eb1bfbc4d3391e64b1b7f321e04687a2303c9aeca59a2508aa96f4ce9b4e086efead107003cd71d73b6658be89ea865ae8711e4` |
| `n3-witness.json` | Witness coloring for $n=3$ confirming $R(B_{2},B_3)=11$ | `a2538e6871739844c400f3e2508fd46a49d2c0e17c7dcd5b7d1f2e00b02a7437fe49ceff26f2f8501aa3a74632833a4b716a03a15c28e8792fcf965caf015b36` |
| `n4-witness.json` | Witness coloring for $n=4$ confirming $R(B_{3},B_4)=15$ | `d143e8217afd821c966e3e23ad24033b39a44da2c043ee6ebeb7228d34e36e5f5033eaf21b69603e2ed17a265bd8bee3caf8793b7b78f665e9003902540ff2ef` |

## Scratch directories

- `FM-001b-sat/` preserves intermediate FM-001b SAT scratch artifacts that were
  temporarily produced outside `futon6`. They are kept here as a reminder that
  experiment outputs need a canonical home in this harness.
