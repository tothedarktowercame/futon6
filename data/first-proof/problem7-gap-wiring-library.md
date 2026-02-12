# Problem 7 Gap Wiring Library

Date: 2026-02-12

This file mines proof/method patterns into wiring diagrams for the current
Problem 7 gaps (`G1`, `G2`, `G3`, `U1-U3`) plus dependency branches (`FJ`, `E2`).

Total diagrams: 22

## Shape Schema
- `Q`: target subproblem
- `D`: decomposition/data model
- `M`: core mechanism
- `C`: certificate theorem/tool
- `O`: output statement
- `B`: bridge status back to P7 gaps

Canonical edges:
- `D -> Q` (`clarify`)
- `M -> D` (`assert`)
- `C -> M` (`reference`)
- `O -> Q` (`assert`)
- `B -> O` (`reform` or `challenge`)

## Gap Coverage
- `E2`: 6 diagrams
- `FJ`: 8 diagrams
- `G1`: 7 diagrams
- `G2`: 2 diagrams
- `G3`: 3 diagrams
- `U3`: 2 diagrams

## Diagram Index

- `p7g-d01` — Fowler FH(Q) Fixed-Set Criterion (arXiv:1204.4667)
  tags: G1, E2; bridge: direct
- `p7g-d02` — Avramidi Rational Manifold Model Pipeline (arXiv:1506.06293)
  tags: G2; bridge: partial
- `p7g-d03` — Bredon-Poincare Duality Group Framework (arXiv:1311.7629)
  tags: G1; bridge: support
- `p7g-d04` — Orbifold Translation Groupoid Cohomology Scaffold (arXiv:0705.3249)
  tags: G1; bridge: support
- `p7g-d05` — Classifying Spaces for Families Toolkit (arXiv:math/0312378)
  tags: G1, FJ; bridge: support
- `p7g-d06` — FJ with Coefficients Inheritance Template (arXiv:math/0510602)
  tags: FJ; bridge: support
- `p7g-d07` — K/L-Theory of Group Rings Survey Pattern (arXiv:1003.5002)
  tags: FJ; bridge: support
- `p7g-d08` — Hyperbolic/Virtually Abelian L-Group Computation Pattern (arXiv:1007.0845)
  tags: FJ, U3; bridge: partial
- `p7g-d09` — Assembly-Map Meta-Library (arXiv:1805.00226)
  tags: FJ; bridge: support
- `p7g-d10` — FJ for Cocompact Lattices (arXiv:1101.0469)
  tags: FJ; bridge: direct
- `p7g-d11` — FJ Extension to Arbitrary Lattices (arXiv:1401.0876)
  tags: FJ; bridge: support
- `p7g-d12` — Closed-Manifold vs Inertia Subgroup Interface (arXiv:0905.0104)
  tags: G3; bridge: partial
- `p7g-d13` — Infinite Dihedral Surgery Obstruction Computations (arXiv:math/0306054)
  tags: U3; bridge: direct
- `p7g-d14` — Modern Wall Finiteness Obstruction View (arXiv:1707.07960)
  tags: G1; bridge: support
- `p7g-d15` — Classical Wall Finiteness Obstruction Survey (arXiv:math/0008070)
  tags: G1; bridge: support
- `p7g-d16` — Equivariant Spivak Bundle and Surgery (arXiv:1705.10909)
  tags: G2, G3; bridge: direct
- `p7g-d17` — Non-Simply-Connected Rational Homotopy Models (arXiv:2304.00880)
  tags: G1, G3; bridge: partial
- `p7g-d18` — Smith-Theory Constraints on Periodic Actions (arXiv:1106.1704)
  tags: E2; bridge: support
- `p7g-d19` — Reflections in Congruence Hyperbolic Manifolds (arXiv:2506.23994)
  tags: E2; bridge: direct
- `p7g-d20` — Finite Groups as Isometry Groups of Hyperbolic Manifolds (arXiv:math/0406607)
  tags: E2; bridge: partial
- `p7g-d21` — Finite Subgroup Counting in Lattices (arXiv:1209.2484)
  tags: E2; bridge: support
- `p7g-d22` — Homology Bounds for Hyperbolic Orbifolds (arXiv:2012.15322)
  tags: E2, FJ; bridge: support

## Diagrams

### p7g-d01: Fowler FH(Q) Fixed-Set Criterion
- Paper: `arXiv:1204.4667`
- Gap tags: G1, E2
- Bridge status: `direct`
- `Q`: Place an orbifold extension group Gamma in FH(Q) via finite-action data.
- `D`: Model Gamma as pi_1((EG x Bpi)/G) for finite G-action on finite Bpi.
- `M`: Reduce finiteness obstruction to Euler-characteristic data on fixed components.
- `C`: Main theorem: vanishing chi on nontrivial fixed components kills obstruction.
- `O`: Existence of finite CW with pi_1=Gamma and rationally acyclic universal cover.
- `B`: Directly discharges E2 when concrete Z/2 fixed-set checks are verified.

### p7g-d02: Avramidi Rational Manifold Model Pipeline
- Paper: `arXiv:1506.06293`
- Gap tags: G2
- Bridge status: `partial`
- `Q`: Upgrade rational duality/finiteness input to manifold-level realization.
- `D`: Start from duality-group and finite-classifying-space hypotheses.
- `M`: Construct rational surgery/normal-map pipeline through manifold-with-boundary stage.
- `C`: Rational surgery interface theorems and reflection-group closure methods.
- `O`: Manifold models with rationally acyclic universal covers in controlled ranges.
- `B`: Potential route for G2, but requires careful pi_1-preservation in P7 setting.

### p7g-d03: Bredon-Poincare Duality Group Framework
- Paper: `arXiv:1311.7629`
- Gap tags: G1
- Bridge status: `support`
- `Q`: Interpret Poincare duality when torsion prevents classical PD-group framing.
- `D`: Use proper actions/family language and Bredon module categories.
- `M`: Translate geometric duality into Bredon cohomological duality statements.
- `C`: Bredon-PD theorems and criteria for duality over families of subgroups.
- `O`: Cohomological PD control for torsion-containing groups acting properly.
- `B`: Supports G1 at homology/cohomology level, not chain-level Poincare complex by itself.

### p7g-d04: Orbifold Translation Groupoid Cohomology Scaffold
- Paper: `arXiv:0705.3249`
- Gap tags: G1
- Bridge status: `support`
- `Q`: Relate orbifold quotient geometry to computable equivariant invariants.
- `D`: Encode orbifold actions by translation groupoids.
- `M`: Compute invariants via Bredon-style/orbifold cohomological functors.
- `C`: Groupoid-cohomology comparison theorems for orbifold invariants.
- `O`: Formal bridge between quotient-orbifold language and equivariant homological data.
- `B`: Useful infrastructure for G1 framing, but not a direct Poincare-complex proof.

### p7g-d05: Classifying Spaces for Families Toolkit
- Paper: `arXiv:math/0312378`
- Gap tags: G1, FJ
- Bridge status: `support`
- `Q`: Control transitions among E_Fin, E_VCyc and family-indexed assembly domains.
- `D`: Model group actions with family-classifying spaces and orbit-category functors.
- `M`: Use family filtration and equivariant homology functoriality.
- `C`: General classifying-space theorems and family comparison machinery.
- `O`: Reusable setup for obstruction-domain changes and assembly-map inputs.
- `B`: Supports U1/FJ structure checks; no direct obstruction vanishing.

### p7g-d06: FJ with Coefficients Inheritance Template
- Paper: `arXiv:math/0510602`
- Gap tags: FJ
- Bridge status: `support`
- `Q`: Prove assembly isomorphism robustness under constructions relevant to lattices.
- `D`: Work in additive-category-with-involution coefficient framework.
- `M`: Exploit inheritance/closure properties under group extensions/actions.
- `C`: Coefficient-form Farrell-Jones statements and transitivity principles.
- `O`: Assembly reduction remains valid in broader categorical settings.
- `B`: Strengthens FJ leg of S-branch; does not identify sigma(f).

### p7g-d07: K/L-Theory of Group Rings Survey Pattern
- Paper: `arXiv:1003.5002`
- Gap tags: FJ
- Bridge status: `support`
- `Q`: Organize obstruction computations after assembly reduction.
- `D`: Express target L-groups via equivariant homology and known decomposition pieces.
- `M`: Use high-level computation templates and conjectural implications.
- `C`: Survey-level statements linking FJ to explicit L-group calculations.
- `O`: Roadmap for moving from abstract assembly to computable summands.
- `B`: Method guidance only; needs problem-specific arithmetic/topological input.

### p7g-d08: Hyperbolic/Virtually Abelian L-Group Computation Pattern
- Paper: `arXiv:1007.0845`
- Gap tags: FJ, U3
- Bridge status: `partial`
- `Q`: Extract explicit L-group formulas in groups with VCyc structure.
- `D`: Decompose by finite and virtually cyclic subgroup contributions.
- `M`: Apply assembly plus nil-term analysis to obtain concrete group-ring formulas.
- `C`: Computation theorems for K/L-theory in hyperbolic and related classes.
- `O`: Explicit algebraic targets for obstruction classes in selected cases.
- `B`: Informative for U3-style coefficient behavior but not directly P7 lattice-specific.

### p7g-d09: Assembly-Map Meta-Library
- Paper: `arXiv:1805.00226`
- Gap tags: FJ
- Bridge status: `support`
- `Q`: Choose safe assembly formulations and decorations for obstruction calculations.
- `D`: Track model choices, decorations, and families in a unified assembly framework.
- `M`: Map problem data to the appropriate assembly variant and comparison diagram.
- `C`: Comprehensive assembly map taxonomy and compatibility results.
- `O`: Reduced risk of decoration/family mismatches in L-theory arguments.
- `B`: Quality-control module for S-branch rigor, not an existence theorem.

### p7g-d10: FJ for Cocompact Lattices
- Paper: `arXiv:1101.0469`
- Gap tags: FJ
- Bridge status: `direct`
- `Q`: Establish FJ for the cocompact lattice class used in P7.
- `D`: Identify Gamma as cocompact lattice in virtually connected Lie group.
- `M`: Apply geometric-flow/control machinery to prove assembly isomorphism.
- `C`: Main theorem covering K/L-theoretic FJ with coefficients for this class.
- `O`: Valid reduction from L_n(Z[Gamma]) to equivariant homology of VCyc family.
- `B`: Directly secures core FJ input in p7r-s3b.

### p7g-d11: FJ Extension to Arbitrary Lattices
- Paper: `arXiv:1401.0876`
- Gap tags: FJ
- Bridge status: `support`
- `Q`: Ensure FJ remains available beyond cocompact specialization choices.
- `D`: Embed lattice cases into arbitrary-lattice class in virtually connected Lie groups.
- `M`: Transfer cocompact arguments via broader inheritance and reduction steps.
- `C`: Main theorem for arbitrary lattices.
- `O`: Robust fallback if lattice family variations occur in later revisions.
- `B`: Stability module for scope changes; not itself a gap closer.

### p7g-d12: Closed-Manifold vs Inertia Subgroup Interface
- Paper: `arXiv:0905.0104`
- Gap tags: G3
- Bridge status: `partial`
- `Q`: Relate algebraic surgery obstructions to realizability by closed manifolds.
- `D`: Fix orientation character and compare closed-manifold and inertia subgroups in L-groups.
- `M`: Use surgery exact-sequence interfaces and subgroup-identification theorems.
- `C`: Theorems A/B relating these subgroups under hypotheses.
- `O`: Criteria for when assembly/image elements correspond to closed-manifold realizations.
- `B`: Potentially sharpens final S-branch closure after obstruction class is identified.

### p7g-d13: Infinite Dihedral Surgery Obstruction Computations
- Paper: `arXiv:math/0306054`
- Gap tags: U3
- Bridge status: `direct`
- `Q`: Control UNil/torsion contributions in VCyc and dihedral pieces.
- `D`: Model type-II VCyc subgroups through dihedral group-ring computations.
- `M`: Compute surgery obstruction groups and identify torsion behavior.
- `C`: Explicit calculation theorems for infinite-dihedral L-groups/UNil terms.
- `O`: Evidence that problematic terms are 2-primary and vanish after rationalization.
- `B`: Directly supports the E_Fin to E_VCyc rational comparison step.

### p7g-d14: Modern Wall Finiteness Obstruction View
- Paper: `arXiv:1707.07960`
- Gap tags: G1
- Bridge status: `support`
- `Q`: Understand finiteness obstruction semantics behind FH(Q) style statements.
- `D`: Represent finite-domination questions through Wall obstruction classes.
- `M`: Analyze vanishing criteria and categorical interpretations of obstruction.
- `C`: Survey-style synthesis of Wall obstruction machinery.
- `O`: Improved conceptual control over what FH(Q) does and does not guarantee.
- `B`: Supports precise gap wording for G1 without closing chain-level PD.

### p7g-d15: Classical Wall Finiteness Obstruction Survey
- Paper: `arXiv:math/0008070`
- Gap tags: G1
- Bridge status: `support`
- `Q`: Track legacy finiteness-obstruction tools used in PD/FH transitions.
- `D`: Package classical examples and obstruction calculations in Wall framework.
- `M`: Map group/space data to obstruction-group elements.
- `C`: Foundational survey references and computation templates.
- `O`: Historical/technical base for careful finite-complex claims.
- `B`: Background module; indirect value for G1 language hygiene.

### p7g-d16: Equivariant Spivak Bundle and Surgery
- Paper: `arXiv:1705.10909`
- Gap tags: G2, G3
- Bridge status: `direct`
- `Q`: Construct normal-map data equivariantly so cover restriction is controlled.
- `D`: Work with equivariant Spivak normal bundle for group actions.
- `M`: Lift/compare equivariant normal structures through surgery setup.
- `C`: Equivariant Spivak and equivariant surgery theorems.
- `O`: Potential direct route to solve G2 and enforce G3 compatibility.
- `B`: Highest-value method candidate for unresolved normal-map/transfer gaps.

### p7g-d17: Non-Simply-Connected Rational Homotopy Models
- Paper: `arXiv:2304.00880`
- Gap tags: G1, G3
- Bridge status: `partial`
- `Q`: Justify rational-homotopy comparisons for spaces with nontrivial pi_1.
- `D`: Represent spaces by rational models adapted to non-simply-connected setting.
- `M`: Use algebraic models to compare rational homotopy types beyond nilpotent defaults.
- `C`: Model-construction and comparison statements in non-simply-connected rational homotopy.
- `O`: Candidate citation path for replacing hand-wavy rational-equivalence claims.
- `B`: Promising for G3 caveat cleanup; must verify applicability hypotheses.

### p7g-d18: Smith-Theory Constraints on Periodic Actions
- Paper: `arXiv:1106.1704`
- Gap tags: E2
- Bridge status: `support`
- `Q`: Filter impossible periodic-action scenarios in aspherical/lattice contexts.
- `D`: Separate homotopically trivial periodic actions from geometric action data.
- `M`: Apply Smith/L2-cohomology rigidity arguments.
- `C`: No-homotopically-trivial-periodic-diffeomorphism theorems in specified settings.
- `O`: Constraint layer preventing invalid fixed-point constructions.
- `B`: Useful negative filter; not a positive constructor for P7.

### p7g-d19: Reflections in Congruence Hyperbolic Manifolds
- Paper: `arXiv:2506.23994`
- Gap tags: E2
- Bridge status: `direct`
- `Q`: Produce explicit reflective involutions in cocompact arithmetic families.
- `D`: Use arithmetic lattice with reflection and congruence-cover tower.
- `M`: Induce manifold involutions with totally geodesic fixed hypersurfaces.
- `C`: Theorem/remark package on reflective congruence manifolds and fixed sets.
- `O`: Concrete E2 instantiation substrate for Z/2 action and fixed-set geometry.
- `B`: Direct constructor feeding the successful p7r-s2b route.

### p7g-d20: Finite Groups as Isometry Groups of Hyperbolic Manifolds
- Paper: `arXiv:math/0406607`
- Gap tags: E2
- Bridge status: `partial`
- `Q`: Realize prescribed finite symmetry groups in compact hyperbolic settings.
- `D`: Construct finite-index subgroups with targeted normalizer quotient.
- `M`: Combine subgroup growth and lattice-commensurator control.
- `C`: Main realization theorem for finite groups as full isometry groups.
- `O`: Existence of compact hyperbolic manifolds with chosen finite symmetry group.
- `B`: Good symmetry-existence backup, but fixed-set Euler data not automatic.

### p7g-d21: Finite Subgroup Counting in Lattices
- Paper: `arXiv:1209.2484`
- Gap tags: E2
- Bridge status: `support`
- `Q`: Control isotropy complexity in lattice/orbifold models.
- `D`: Count maximal finite subgroups and isotropy classes versus covolume.
- `M`: Use lattice geometry and subgroup-growth estimates.
- `C`: Linear/sublinear bounds for finite subgroup conjugacy classes.
- `O`: Quantitative isotropy control for families of orbifold quotients.
- `B`: Auxiliary selection/filtering tool, not a direct P7 closure theorem.

### p7g-d22: Homology Bounds for Hyperbolic Orbifolds
- Paper: `arXiv:2012.15322`
- Gap tags: E2, FJ
- Bridge status: `support`
- `Q`: Estimate orbifold homology growth in arithmetic/nonuniform families.
- `D`: Build efficient simplicial thick-part models for orbifolds.
- `M`: Derive linear-volume homology/torsion bounds from geometric models.
- `C`: Theorems bounding Betti numbers and torsion parts by volume.
- `O`: Practical constraints on candidate families for obstruction-space dimensions.
- `B`: Useful quantitative side-information for strategy A/B computations.

