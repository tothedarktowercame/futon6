# P4 Path 2 Cycle 2 (Codex): Closure of C1 and Boundary C0

Date: 2026-02-13  
Script: `scripts/explore-p4-path2-codex-handoff2.py`  
Results: `data/first-proof/p4-path2-codex-handoff2-results.json`

## Setup
Using the exact normalized surplus polynomial `K_red` (from the full `T2+R` derivation), we use
\[
K_{\mathrm{red}} = C_1\,(p+q)^2 + C_0,
\]
with
\[
C_1 = a_{20}p^2 + a_{02}q^2 + a_{12}p^2q^2,
\quad a_{12}=-1296rL,
\]
\[
a_{20}=72r^4(1-y)M_y,
\quad a_{02}=72(1-x)M_x.
\]
On the feasible domain: `r>0`, `0<x<1`, `0<y<1`, `p^2<=2(1-x)/9`, `q^2<=2r^3(1-y)/9`.

## Claim 1: `C1 >= 0`
The script proves exact identities:
\[
M_y-4L=(3x+1)(3y+1)^2\bigl(3r^2y+r^2+4r+3x+1\bigr),
\]
\[
M_x-4rL=(3x+1)^2(3y+1)\bigl(3r^2y+r^2+4r+3x+1\bigr).
\]
Hence `M_y>=4L` and `M_x>=4rL` on the full feasible domain.

Using `q^2<=2r^3(1-y)/9` and `p^2<=2(1-x)/9`:
\[
a_{20}p^2 = 72r^4(1-y)M_y p^2 \ge 324rM_y p^2q^2,
\]
\[
a_{02}q^2 = 72(1-x)M_x q^2 \ge 324M_x p^2q^2.
\]
Therefore
\[
C_1 \ge 324p^2q^2\,(rM_y+M_x-4rL).
\]
The script also verifies `rM_y+M_x-4rL` has 32 monomials and every coefficient is positive (minimum coefficient `1`), so `C1>=0`.

## Claim 2: `C0(p,-p) >= 0`
Set `t=p^2` and `q=-p`. Then
\[
f(t):=C_0(p,-p)=a_{00}+(a_{10}+a_{01}-b_{00})t+\delta_1 t^2.
\]
The script proves exact factorization
\[
\delta_1=24W\cdot P_+(r,x,y),
\quad W=3r^2y-3r^2-4r+3x-3,
\]
with `W<0` on feasible interior and `P_+>0`, so `delta1<0` and `f` is concave.

Let
\[
P_{\max}=\frac{2(1-x)}{9},\quad Q_{\max}=\frac{2r^3(1-y)}{9},
\quad D=r^3(1-y)-(1-x)=\frac{9}{2}(Q_{\max}-P_{\max}).
\]
Exact endpoint identities verified by symbolic equality:
1. `f(P_max) = const * D * W * (positive factors)`.
2. `f(Q_max) = const * (-D) * W * (positive factors)`.

So `sign(f(P_max))=sign(D)` and `sign(f(Q_max))=-sign(D)`. The active endpoint is
`t_max=min(P_max,Q_max)`, therefore the selected endpoint always has nonnegative sign. Also `f(0)=a00>=0` from its exact factorization.

Since `f` is concave, the minimum on `[0,t_max]` is attained at an endpoint; both endpoints relevant to the interval are nonnegative, hence
\[
C_0(p,-p)=f(t)\ge 0\quad\text{for all feasible }t\in[0,t_{\max}].
\]

## Numerical sanity (supporting)
The script ran `300,000` feasible random tests and found:
1. `bad_endpoint_rule = 0`.
2. `bad_interval_rule = 0`.
3. Minimum observed endpoint/interval values were positive up to floating tolerance (`~2.29e-21`).

## Conclusion
Both handoff targets are closed in the script artifacts:
1. `C1>=0` by exact algebraic identities plus domain inequalities.
2. `C0(p,-p)>=0` by concavity and exact endpoint sign identities.

This yields `K_red>=0` for the Path-2 reduction chain under the stated decomposition.
