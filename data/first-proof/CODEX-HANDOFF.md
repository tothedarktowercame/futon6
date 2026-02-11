# Codex Handoff: Problem 10 Proof Polish

## What This Is

An attempt at Problem 10 of the **First Proof** benchmark
(arxiv.org/abs/2602.05192 — Abouzaid, Blumberg, Hairer, Kileel, Kolda,
Nelson, Spielman, Srivastava, Ward, Weinberger, Williams). Problem 10
is by Kolda: show that PCG solves the RKHS-constrained tensor CP mode-k
subproblem without O(N) computation.

We have:
1. A written solution (`problem10-solution.md`)
2. A wiring diagram modeling the proof as an IATC argument graph (`problem10-wiring.json`)
3. Codex prompts for verification and polishing (`problem10-codex-prompts.jsonl`)

## What Codex Should Do

**For each of 14 proof nodes + 1 synthesis prompt:**

1. **Verify** the mathematical claim (correct? gap? error?)
2. **Cross-reference** with math.SE for supporting discussions
3. **Identify gaps** — especially in Section 5 (convergence), the weakest link
4. **Suggest improvements** — tighter bounds, missing assumptions, cleaner statements

**Key verification targets:**

| Node | What to check | Risk |
|------|---------------|------|
| p10-s2a | Kronecker identity (A⊗B)vec(X) = vec(BXAᵀ) | Low — standard identity |
| p10-s4-hadamard | Khatri-Rao Hadamard: ZᵀZ = ∏(AᵢᵀAᵢ) | Low — standard in tensor lit |
| p10-s4-solve | Kronecker inverse (H⊗K)⁻¹ = H⁻¹⊗K⁻¹ | Low — requires both invertible |
| p10-s5 | Convergence bound t = O(r) to O(r√(n/q)) | **High** — weakest part |
| p10-s6 | Final complexity comparison | Medium — aggregation step |

## Math.SE Search Queries

When math.SE data is available, search for:

- "Kronecker product" + "conjugate gradient"
- "tensor decomposition" + "preconditioning"
- "Khatri-Rao" + "Hadamard"
- "RKHS" + "regularization" + "kernel"
- "sparse observation" + "matrix-vector product"
- "PCG convergence" + "condition number"
- "Cholesky" + "Kronecker structure"

## How To Run

```bash
# 1. Generate prompts (already done, but regenerate if solution changes)
python3 scripts/proof-wiring-diagram.py
python3 scripts/run-proof-polish-codex.py --dry-run

# 2. Run through Codex (all 15 prompts)
python3 scripts/run-proof-polish-codex.py --model o3 --limit 15

# 3. Check results
python3 -c "
import json
results = [json.loads(l) for l in open('data/first-proof/problem10-codex-results.jsonl')]
for r in results:
    nid = r.get('node_id', '?')
    v = r.get('claim_verified', 'parse_error')
    c = r.get('confidence', '?')
    refs = len(r.get('math_se_references', []))
    gaps = len(r.get('missing_assumptions', []))
    print(f'{nid:20s}  {v:10s}  conf={c:6s}  refs={refs}  gaps={gaps}')
"
```

## Files

| File | Purpose |
|------|---------|
| `problem10-solution.md` | The written proof (7 sections + algorithm) |
| `problem10-wiring.json` | IATC wiring diagram (14 nodes, 20 edges, 20 hyperedges) |
| `problem10-codex-prompts.jsonl` | 15 verification prompts for Codex |
| `problem10-codex-results.jsonl` | Codex output (created by run) |
| `../../scripts/proof-wiring-diagram.py` | Generates wiring diagram from proof |
| `../../scripts/run-proof-polish-codex.py` | Runs verification prompts through Codex |

## What To Do With Results

After Codex runs:

1. **If all nodes verified**: proof is solid, submit as-is
2. **If convergence (p10-s5) flagged**: this is expected — the bound is
   heuristic. Codex should suggest a tighter statement or cite a theorem
   from math.SE that pins it down
3. **If Kronecker identities flagged**: double-check dimension consistency
4. **Merge improvements back**: edit `problem10-solution.md`, regenerate
   wiring diagram, re-run for a second pass

## Context: First Proof Benchmark

- 10 unpublished research math problems
- Answers encrypted, released Feb 13 2026
- We chose #10 (Kolda) as most tractable given our corpus (PlanetMath has
  conjugate gradient, Kronecker product, positive definite matrices; physics.SE
  has iterative solver discussions)
- The wiring diagram is a meta-demonstration: Stage 7's thread-as-wiring-diagram
  infrastructure modeling the proof itself
