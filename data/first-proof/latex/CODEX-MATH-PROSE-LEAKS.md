# Math/Prose Boundary Leaks — Fix Report for Codex

**Date:** 2026-02-13
**Scanner:** Claude (Opus 4.6)
**Files:** `full/problem{1..10}-solution-full.tex`

Two categories of systematic issues remain after the syntax normalization pass.

## Category 1: Prose Words Inside Math Mode (20 instances)

English words appearing inside `\[...\]` or `\(...\)` that should be
restructured — either move the prose outside math mode, or use proper
`\text{...}` blocks only for genuine inline annotations.

### Worst offenders

**problem6-solution-full.tex lines 194-196** — Entire theorem statement
as individual `\text{word}` fragments in display math:
```latex
% BAD:
\[ Assumption V: There \text{exists} c_0 > 0 \text{such} \text{that}
   \text{for} \text{every} ... \]
% GOOD: Move to prose with inline math:
Assumption~V\@: There exists $c_0 > 0$ such that for every weighted
graph $G$ and $\varepsilon \in (0,1)$, one can choose ...
```

**problem8-solution-full.tex lines 230-233** — Descriptive prose
("smooth curve", "reparameterized to match the axes") inside display math:
```latex
% BAD:
\[ C_1^{sm}: smooth curve \in V_1, \text{for} x > \delta \]
% GOOD: Use an aligned environment or move description to prose
```

**problem5-solution-full.tex line 95** — Quantifiers as separate `\text{}`:
```latex
% BAD:
\[ \pi_i(\Phi^H X) = 0 \text{for} \text{all} i < \lceil n/|H| \rceil \]
% GOOD:
\[ \pi_i(\Phi^H X) = 0 \quad\text{for all } i < \lceil n/|H| \rceil \]
```

### Full list

| File | Line | Words in math mode |
|------|------|--------------------|
| problem1-solution-full.tex | 134 | `\text{for}` |
| problem3-solution-full.tex | 67 | `\text{where}` |
| problem5-solution-full.tex | 91 | `\text{is}` |
| problem5-solution-full.tex | 95 | `\text{for} \text{all}` (×2) |
| problem5-solution-full.tex | 184 | `\text{all}` + bare "subgroups" |
| problem6-solution-full.tex | 194 | bare "Assumption V" label |
| problem6-solution-full.tex | 195-196 | `\text{exists/such/that/for/every/and/one/can/with}` |
| problem6-solution-full.tex | 261 | `\text{internal} \text{to}` |
| problem8-solution-full.tex | 64 | `\text{for} \text{each}` |
| problem8-solution-full.tex | 216-217 | `\text{and}` (×2) |
| problem8-solution-full.tex | 230-233 | Entire sentence of prose in display math |

**Fix pattern:** If it reads like English, it goes outside `\[...\]`. If it
must stay (like "for all" after an equation), use `\quad\text{for all }` as
a single block, not separate `\text{for} \text{all}`.

## Category 2: Math Not in Math Mode (37 instances)

### 2a. `\textgreater{}` / `\textless{}` for math comparisons (14 instances)

These are pandoc artifacts. Replace with proper math mode.

| File | Line | Bad | Good |
|------|------|-----|------|
| problem1-solution-full.tex | 8 | `-\textgreater{} R` | `$\to \mathbb{R}$` |
| problem1-solution-full.tex | 54 | `\textless{} infinity` | `$< \infty$` |
| problem1-solution-full.tex | 55 | `\textgreater{} 0` | `$> 0$` |
| problem2-solution-full.tex | 9 | `-\textgreater{}` | `$\to$` |
| problem2-solution-full.tex | 49 | `\textgreater\textgreater{} 0` | `$\gg 0$` |
| problem3-solution-full.tex | 117 | `\textgreater{} 0` | `$> 0$` |
| problem3-solution-full.tex | 170 | `-\textgreater{}` | `$\to$` |
| problem4-solution-full.tex | 110 | `-\textgreater{} infinity` | `$\to \infty$` |
| problem4-solution-full.tex | 213 | `\textgreater{} 0` | `$> 0$` |
| problem6-solution-full.tex | 89 | `\textless= exp(...)` | `$\le \exp(...)$` |
| problem8-solution-full.tex | 337 | `\textless{}` (×2) | `$<$` |
| problem10-solution-full.tex | 13 | `\textless\textless{}` (×2) | `$\ll$` |
| problem10-solution-full.tex | 153 | `\textgreater\textgreater{}` | `$\gg$` |

### 2b. Fragmented math expressions (17 instances)

Expressions split across multiple `\(...\)` fragments with operators,
subscripts, superscripts in bare prose between them.

| File | Line | Issue |
|------|------|-------|
| problem1-solution-full.tex | 33 | `(m^2 - \Delta)^{-1}` fragmented |
| problem1-solution-full.tex | 144 | `1/R \in L^1(T_\psi^* \mu)` fragmented |
| problem4-solution-full.tex | 55 | `p(x) = \prod_i (x - \lambda_i)` half out |
| problem4-solution-full.tex | 125 | `\Phi_n(\mathrm{char}(A)) = \sum_i (...)^2` fragmented |
| problem4-solution-full.tex | 127 | `F_A''(x) = -\sum_i 1/(x-\lambda_i)^2` fragmented |
| problem4-solution-full.tex | 192 | `\mathrm{disc} = \prod_{i<j}(\lambda_i-\lambda_j)^2` fragmented |
| problem4-solution-full.tex | 226 | `\Phi_n \cdot \mathrm{disc} = \mathrm{const} \cdot a_2^2` fragmented |
| problem7-solution-full.tex | 22 | `\chi = 0` with `= 0` outside |
| problem7-solution-full.tex | 383 | `\theta = 0` outside math |
| problem8-solution-full.tex | 75 | `\omega|_{V_1 \times V_2} = 0` fragmented |
| problem8-solution-full.tex | 91 | `L_i = \mathrm{span}(...)` fragmented |
| problem8-solution-full.tex | 95 | `\ell = \ker(\omega|_H)` fragmented |
| problem8-solution-full.tex | 377 | `\mathrm{Ham}_c(M, \omega)` fragmented |
| problem9-solution-full.tex | 8 | `Q^{(\alpha\beta\gamma\delta)}` with Greek as bare words |
| problem9-solution-full.tex | 99 | `\Omega_{mn} = Q^{(...)}_{...}` fragmented |
| problem10-solution-full.tex | 59 | `x = \mathrm{vec}(V), V \in \mathbb{R}^{n \times r}` fragmented |
| problem10-solution-full.tex | 162 | `\kappa = \mathrm{cond}(...)` fragmented |

**Fix pattern:** Unify each mathematical expression into a single `\(...\)` or
`$...$` block. Don't split at operators or subscripts.

### 2c. Unicode math symbols outside math mode (5 instances, all Problem 7)

| File | Line | Unicode | Should be |
|------|------|---------|-----------|
| problem7-solution-full.tex | 391 | `e(ν)⊗Q = 0` | `$e(\nu) \otimes \mathbb{Q} = 0$` |
| problem7-solution-full.tex | 397 | `g ≡ I mod I` | `$g \equiv I \pmod{I}$` |
| problem7-solution-full.tex | 401 | `θ = 0 ∈ L₈(Z[Γ])` | `$\theta = 0 \in L_8(\mathbb{Z}[\Gamma])$` |
| problem7-solution-full.tex | 406 | `∂W = S(ν) = F × S¹` | `$\partial W = S(\nu) = F \times S^1$` |
| problem8-solution-full.tex | 100 | `H/ell ≅ R²` | `$H/\ell \cong \mathbb{R}^2$` |

## Priority

1. **Category 2a** (textgreater/textless) — mechanical, safe bulk replacement
2. **Category 2b** (fragmented expressions) — unify each into one math block
3. **Category 1** (prose in math) — restructure display math blocks
4. **Category 2c** (Unicode) — wrap in math mode with proper LaTeX commands

After fixes, re-run: `python3 scripts/apply-proof-boxes.py` to regenerate boxed files.
