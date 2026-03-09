# FM-001 harness artifacts

Solver-facing inputs/outputs that live under `data/frontiermath-pilot/harness/`.
All hashes are SHA512 so remote collaborators can verify downloads before
launching SAT or checker runs.

| File | Description | SHA512 |
|------|-------------|--------|
| `FM001-n5.cnf.gz` | 18-vertex ($n=5$) book-Ramsey SAT instance generated 2026-03-09 | `6e1d2976c0f31e6c1e054deee88941750237f028e2ccc8172592e138acbc76eb09089a1585f22727d18fbbaaadaa92440b16d91217b2b9a35edfe7f6549508e5` |
| `FM001-n6.cnf.gz` | 22-vertex ($n=6$) instance, unsolved (built with `--no-solve`) | `4d237737fdefbab061bcf772525389a8531de8919c47606559360e6267a538108fa4212a830502f55571f81164ae168862484c3845a396cf787a8433d067aac8` |
| `n3-witness.json` | Witness coloring for $n=3$ confirming $R(B_{2},B_3)=11$ | `a2538e6871739844c400f3e2508fd46a49d2c0e17c7dcd5b7d1f2e00b02a7437fe49ceff26f2f8501aa3a74632833a4b716a03a15c28e8792fcf965caf015b36` |
| `n4-witness.json` | Witness coloring for $n=4$ confirming $R(B_{3},B_4)=15$ | `d143e8217afd821c966e3e23ad24033b39a44da2c043ee6ebeb7228d34e36e5f5033eaf21b69603e2ed17a265bd8bee3caf8793b7b78f665e9003902540ff2ef` |

