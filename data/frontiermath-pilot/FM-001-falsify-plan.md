# FM-001/F1-opposite — falsification work plan

Last updated: 2026-03-07

## Goal

Show evidence that $R(B_{n-1}, B_n) < 4n-1$ for some $n$, or exhaust the most
plausible counterexample ranges (small $n$ and number-theoretic edge cases)
before switching to the constructive pipeline.

## Immediate steps

1. **Reproduce published small-$n$ searches.** Translate the 2-block-circulant +
   SAT/DRAT workflow from Wesley (2025) into our toolchain and re-run it for
   $n \in \{3,4,5,6,7,8,9,10\}$. Record whether each $n$ exhaustively confirms
   $R(B_{n-1}, B_n) = 4n-1$.
2. **Identify arithmetic gaps.** For $n$ where $2n-1$ is *not* a prime power
   $\equiv 1 \pmod 4$, catalogue whether the Paley-style constructions fail, and
   target those $n$ for explicit counterexample search.
3. **Define brute-force search envelope.** For $n=3,4$, brute-force all
   $K_{4n-2}$ colorings (with canonical reductions) to ensure we can detect a
   counterexample automatically.
4. **Literature cross-check.** Summarize any upper/lower bounds from Wigderson’s
   “Book Ramsey numbers I” that constrain the falsification space.

## Artifacts

- `data/frontiermath-pilot/FM-001-strategy.md` — primary FM-001 strategy/status note.
- `holes/missions/M-distributed-frontiermath.md` — mission log for solver runs and phase updates.

## Published evidence snapshot (2026-03-08)

- Wesley Chen, *Lower bounds for book Ramsey numbers* (WashU FYS slides, Feb 2025) reproduces the 2-block-circulant + SAT/DRAT workflow and certifies that $R(B_{n-1}, B_n) = 4n-1$ for every $n \le 20$ and for each odd $n$ with $2n-1$ a prime power $\equiv 1 \pmod 4$. Downloaded from <https://www.math.wustl.edu/~dschung/FYS_S25/Slides/Oral%202%20-%20Wesley%20Chen.pdf> on 2026-03-08.
- Wigderson, Fox, and Conlon, *Book Ramsey numbers I* (arXiv:2110.14483) analyze off-diagonal book Ramsey numbers via quasirandom templates, showing that near-extremal colorings must resemble either a balanced multipartite graph or a Paley-type construction when they exist. This constrains how a falsification witness can look and motivates focusing on moduli where Paley inputs fail.

## Candidate $n$ beyond current coverage

Computed on 2026-03-08 via `python3` + `sympy`: the table lists the first batch of $n$ where $n>20$ and $2n-1$ is **not** a prime power (hence Paley templates do not immediately certify the $4n-1$ bound). All cases with $n \le 20$ were already settled by the SAT certificates above.

| $n$ | $2n-1$ | Factorization | Notes |
| --- | --- | --- | --- |
| 23 | 45 | $3^2 \cdot 5$ | Smallest open modulus; composite with repeated prime |
| 26 | 51 | $3 \cdot 17$ | Even $n$ (modulus $\equiv 3 \pmod 4$) |
| 28 | 55 | $5 \cdot 11$ | Even, square-free |
| 29 | 57 | $3 \cdot 19$ | Odd but composite modulus |
| 32 | 63 | $3^2 \cdot 7$ | --- |
| 33 | 65 | $5 \cdot 13$ | --- |
| 35 | 69 | $3 \cdot 23$ | --- |
| 38 | 75 | $3 \cdot 5^2$ | --- |
| 39 | 77 | $7 \cdot 11$ | --- |
| 43 | 85 | $5 \cdot 17$ | First modulus after Paley window |
| 44 | 87 | $3 \cdot 29$ | --- |
| 46 | 91 | $7 \cdot 13$ | --- |
| 47 | 93 | $3 \cdot 31$ | --- |
| 48 | 95 | $5 \cdot 19$ | --- |
| 50 | 99 | $3^2 \cdot 11$ | --- |
| 53 | 105 | $3 \cdot 5 \cdot 7$ | --- |
| 56 | 111 | $3 \cdot 37$ | --- |
| 58 | 115 | $5 \cdot 23$ | --- |
| 59 | 117 | $3^2 \cdot 13$ | --- |
| 60 | 119 | $7 \cdot 17$ | --- |

Extend this list to $n \le 100$ if/when we need a deeper queue.

## Immediate falsification experiment queue

1. **Rebuild the 2-block-circulant SAT harness for composite moduli.** Start with $n=23$ and $n=29$, where $2n-1 \in \{45,57\}$, to see whether the Paley template fails sharply or if a mild perturbation exists.
2. **Add even-$n$ benchmarks.** For $n=26,28$, rerun the SAT search with the `K_{4n-2}` encoding to confirm whether the absence of Paley inputs makes these the first plausibly smaller Ramsey numbers.
3. **Log obstruction notes.** For each $n$ tested, capture whether the search terminates (certificate showing $R(B_{n-1}, B_n) = 4n-1$) or whether we stumble on a candidate coloring; update `data/frontiermath-pilot/FM-001-strategy.md` and `holes/missions/M-distributed-frontiermath.md` accordingly.
4. **General CNF generator.** `scripts/fm001/ramsey_book_sat.py` now emits the unconstrained SAT instance and can run Glucose via `./.venv/bin/python ...`. Verified SAT for $n=3$ and prepared to scale toward the composite-modulus targets.
