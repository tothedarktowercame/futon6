# AIF+ Full Method Audit (P4/P6)

Scope: Problems 4 and 6 only.
Definitions: I1-I6 from `chapter0-aif-as-wiring-diagram.md`; gates G5-G0 from `gate-pattern-mapping.md`.

Source DAG timestamp: `2026-02-13T15:02:01.244409+00:00`

## Route Telemetry

| Route | Steps | Observe events | Act events | Balance (I2 proxy) | Artifact events | Architecture events | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| P6 A-Only | 4 | 3 | 1 | 0.33 | 1 | 1 | Limited: observation-heavy, weak closure |
| P6 A->E+F | 15 | 10 | 6 | 0.60 | 4 | 6 | Improved: richer loop; still incomplete |
| P4 Main Route | 3 | 1 | 1 | 1.00 | 1 | 0 | Focused partial: closes n<=3, leaves cert branch |

## Problem 4 (Finite Free Convolution)

Overall verdict: **PARTIAL** - n=2 and n=3 are closed; n>=4 remains open with strong evidence.

Coverage score: invariants `9/12`, gates `11/12`.

### Invariants (I1-I6)
| Check | Status | Assessment | Evidence |
|---|---|---|---|
| I1 Boundary integrity | PASS | The theorem boundary and exact target inequality are explicit and stable. | `data/first-proof/problem4-solution.md:3`, `data/first-proof/problem4-solution.md:21`, `data/first-proof/problem4-proof-strategy-skeleton.md:6` |
| I2 Observe/action asymmetry | PARTIAL | The route separates exploration from proof actions, but there is no episode-level instrumentation. | `data/first-proof/problem4-proof-strategy-skeleton.md:17`, `data/first-proof/problem4-proof-strategy-skeleton.md:33`, `data/first-proof/project-flow-dag.md:11` |
| I3 Timescale separation | PASS | Fast checks and slow closure are separated: n<4 is closed while n>=4 remains explicitly open. | `data/first-proof/problem4-proof-strategy-skeleton.md:4`, `data/first-proof/problem4-solution.md:34` |
| I4 Preference exogeneity | PASS | The writeup preserves target integrity under failed approaches instead of rewriting success criteria. | `data/first-proof/problem4-conditional-stam.md:4`, `data/first-proof/problem4-conditional-stam.md:7`, `data/first-proof/problem4-proof-strategy-skeleton.md:62` |
| I5 Model adequacy | PARTIAL | Empirical support is strong, but n>=4 still lacks an unconditional theorem. | `data/first-proof/problem4-solution.md:29`, `data/first-proof/problem4-solution.md:32`, `data/first-proof/problem4-ngeq4-research-brief.md:34` |
| I6 Compositional closure | PARTIAL | The argument closes n=2 and n=3, but does not yet compose to n>=4. | `data/first-proof/problem4-solution.md:35`, `data/first-proof/problem4-solution.md:36`, `data/first-proof/problem4-proof-strategy-skeleton.md:133` |

### Gates (G5-G0)
| Check | Status | Assessment | Evidence |
|---|---|---|---|
| G5 Task specification | PASS | Problem statement, symbols, and success condition are explicit. | `data/first-proof/problem4-solution.md:3`, `data/first-proof/problem4-solution.md:26` |
| G4 Capability/assignment | PARTIAL | Model and run metadata exist for verifier runs, but formal role typing is lightweight. | `data/first-proof/problem4-lt4-verification-summary.md:3`, `data/first-proof/problem4-lt4-verification-summary.md:7` |
| G3 Pattern reference | PASS | Named strategy routes and route eliminations are documented. | `data/first-proof/problem4-proof-strategy-skeleton.md:33`, `data/first-proof/problem4-proof-strategy-skeleton.md:51` |
| G2 Execution | PASS | Execution artifacts include scripts and route-specific verification commands. | `data/first-proof/problem4-solution.md:139`, `data/first-proof/problem4-solution.md:265`, `data/first-proof/problem4-proof-strategy-skeleton.md:75` |
| G1 Validation | PASS | Verifier outcomes and flagged gaps are explicitly recorded. | `data/first-proof/problem4-lt4-verification-summary.md:13`, `data/first-proof/problem4-lt4-verification-summary.md:30` |
| G0 Evidence durability | PASS | Durable route artifacts and summary tables exist in canonical project files. | `data/first-proof/project-flow-dag.md:95`, `data/first-proof/problem4-solution.md:227` |

### Open-Gap Ledger
| Evidence | Excerpt |
|---|---|
| `data/first-proof/problem4-solution.md:28` | **Conjecturally yes, with strong numerical evidence.** The inequality |
| `data/first-proof/problem4-solution.md:32` | Cauchy-Schwarz). An analytic proof for n >= 4 remains open. |
| `data/first-proof/problem4-solution.md:36` | - n >= 4: numerically verified, proof incomplete. The n=3 identity does not |
| `data/first-proof/problem4-solution.md:229` | **The inequality is conjecturally true, with strong numerical evidence and |
| `data/first-proof/problem4-solution.md:248` | **What remains open for n >= 4:** |
| `data/first-proof/problem4-proof-strategy-skeleton.md:4` | Status: n=2 proved (equality), n=3 proved (Cauchy-Schwarz), n>=4 open |
| `data/first-proof/problem4-proof-strategy-skeleton.md:59` | ## Open Lemmas (Current Gaps) |
| `data/first-proof/problem4-proof-strategy-skeleton.md:62` | **ELIMINATED**: F is not Schur-convex or Schur-concave. Majorization alone |
| `data/first-proof/problem4-proof-strategy-skeleton.md:65` | **ELIMINATED**: F is not submodular in root coordinates (~50% violation rate). |
| `data/first-proof/problem4-proof-strategy-skeleton.md:123` | FAILS — the product depends on \(a_3, a_4\). Additionally \(\boxplus_4\) has a |

## Problem 6 (Epsilon-Light Vertex Subsets)

Overall verdict: **PARTIAL** - K_n is closed with c=1/3; general-graph closure has one explicit open bridge.

Coverage score: invariants `10/12`, gates `11/12`.

### Invariants (I1-I6)
| Check | Status | Assessment | Evidence |
|---|---|---|---|
| I1 Boundary integrity | PASS | The reduced theorem target and spectral normalization are explicit. | `data/first-proof/problem6-method-wiring-library.md:5`, `data/first-proof/problem6-method-wiring-library.md:9`, `data/first-proof/problem6-solution.md:40` |
| I2 Observe/action asymmetry | PASS | Probe directions and constructive pivots are both present and separated. | `data/first-proof/problem6-gpl-h-attack-dispatch.md:104`, `data/first-proof/problem6-gpl-h-attack-dispatch.md:111`, `data/first-proof/project-flow-dag.md:25` |
| I3 Timescale separation | PASS | The route shows staged progression from diagnostics to architecture switch to draft and gap correction. | `data/first-proof/project-flow-dag.md:36`, `data/first-proof/project-flow-dag.md:62` |
| I4 Preference exogeneity | PASS | Gap honesty and counterexamples are surfaced explicitly rather than hidden. | `data/first-proof/problem6-solution.md:30`, `data/first-proof/problem6-solution.md:424`, `data/first-proof/problem6-direction-e-f-proof.md:4` |
| I5 Model adequacy | PARTIAL | Evidence is strong (440/440 numeric support), but a theorem-level general-graph bridge is still open. | `data/first-proof/problem6-solution.md:31`, `data/first-proof/problem6-solution.md:284`, `data/first-proof/problem6-gpl-h-attack-paths.md:59` |
| I6 Compositional closure | PARTIAL | K_n is closed with c=1/3; composition to arbitrary graphs is not yet fully discharged. | `data/first-proof/problem6-solution.md:27`, `data/first-proof/problem6-solution.md:30`, `data/first-proof/problem6-direction-e-f-proof.md:15` |

### Gates (G5-G0)
| Check | Status | Assessment | Evidence |
|---|---|---|---|
| G5 Task specification | PASS | The problem, constraints, and universal constant target are explicit. | `data/first-proof/problem6-solution.md:3`, `data/first-proof/problem6-solution.md:22` |
| G4 Capability/assignment | PASS | Explicit student-dispatch and direction assignment are documented. | `data/first-proof/problem6-gpl-h-attack-dispatch.md:4`, `data/first-proof/problem6-gpl-h-attack-dispatch.md:5` |
| G3 Pattern reference | PASS | Method library D1..D10 and bridge statuses are concretely mapped. | `data/first-proof/problem6-method-wiring-library.md:49`, `data/first-proof/problem6-method-wiring-library.md:52` |
| G2 Execution | PASS | Direction artifacts and proof drafts show active execution across routes. | `data/first-proof/problem6-direction-e-f-proof.md:1`, `data/first-proof/project-flow-dag.md:29`, `data/first-proof/project-flow-dag.md:34` |
| G1 Validation | PARTIAL | Validation is extensive, but one theorem-level condition remains open. | `data/first-proof/problem6-solution.md:284`, `data/first-proof/problem6-solution.md:285`, `data/first-proof/problem6-direction-e-f-proof.md:15` |
| G0 Evidence durability | PASS | DAG, route tables, and durable writeups preserve the full trajectory. | `data/first-proof/project-flow-dag.md:3`, `data/first-proof/project-flow-dag.md:95`, `data/first-proof/problem6-solution.md:344` |

### Open-Gap Ledger
| Evidence | Excerpt |
|---|---|
| `data/first-proof/problem6-solution.md:30` | **General graphs: ONE GAP.** The formal bound dbar < 1 at M_t != 0 is |
| `data/first-proof/problem6-solution.md:31` | empirically verified (440/440 steps, 36% margin) but open. The leverage |
| `data/first-proof/problem6-solution.md:87` | ## 4. Concentration setup (gap-fixed formulation) |
| `data/first-proof/problem6-solution.md:342` | concentration proof for general graphs remains open. |
| `data/first-proof/problem6-solution.md:382` | ### Remaining formal gap |
| `data/first-proof/problem6-solution.md:401` | The remaining gap is not "we don't know enough math"; it is "we are |
| `data/first-proof/problem6-solution.md:412` | alignment is <= 0.25 across all 351 Phase 2 steps. Closing the gap |
| `data/first-proof/problem6-solution.md:424` | margin and is the SINGLE remaining gap. |
| `data/first-proof/problem6-direction-e-f-proof.md:4` | Status: E+F reduction proved but F-Lemma has counterexample. Better route: |
| `data/first-proof/problem6-direction-e-f-proof.md:15` | open lemma package. |

## Audit Conclusion

1. Problem 4 is structurally well-audited and partially closed (n<=3 closed, n>=4 open).
2. Problem 6 is structurally well-audited and partially closed (K_n closed, one general-graph bridge open).
3. The P6 A->E+F route is a real improvement over A-only, but still not a full closure.

