# Problem 7 Approach II/III Wiring Library

Date: 2026-02-12
Updated: 2026-02-12 (triage pass after reading 1705.10909 and 1106.1704)

This library is focused on the new S-branch framing from `0fa4e82`.

## Critical findings from paper reads

**1705.10909 (Costenoble-Waner):** Equivariant surgery requires a
**codimension-2 gap hypothesis**. Our reflection construction has a
codimension-1 fixed set, so Approach II (equivariant surgery on the
reflection action) is **BLOCKED** for the current lattice family.
However, if we switch to **rotational involutions** (codimension-2 fixed
sets), the gap hypothesis is satisfied.

**1106.1704 (Avramidi):** The Smith-theoretic constraints work mod p, not
over Q. The paper does not kill the odd-dim E2 alternative. However,
**Gauss-Bonnet kills it for reflections**: even-dim closed hyperbolic
manifolds always have chi != 0, so reflections in odd ambient dimension
fail Fowler.

**Combined insight:** The dimension-parity tension is inescapable for
reflections. The resolution is to use **rotational involutions (codim-2)
in odd ambient dimension**: fixed set is odd-dim (chi = 0), gap hypothesis
is satisfied, and L-theory parity is favorable. See
`problem7-hypothetical-wirings.md`, Diagram H1.

## Triage

**Read (high value):**
- d01 (`1705.10909`) — READ. Codim-2 gap kills reflection Approach II but
  enables rotation Approach II.
- d15 (`1106.1704`) — READ. Smith/Gauss-Bonnet constraints clarified.
- d02 (`1811.08794`) — Equivariant Pontrjagin-Thom for orbifold cobordism.
  Bridges Approaches II and III. Worth reading.
- d04 (`1112.2104`) — Proper discrete-group bordism with signatures.
  Relevant if rotation route proceeds.

**Skip (wrong setting, already consumed, or infrastructure only):**
- d03 (`math/0412550`) — Equivariant bordism comparison. Too general.
- d05 (`math/9910024`) — Complex equivariant bordism rings. Wrong setting.
- d06 (`math/0512284`) — Positive scalar curvature with symmetry. Wrong problem.
- d07 (`1305.2288`) — Circle actions and scalar curvature. Wrong problem.
- d08 (`2002.02326`) — Corks and involutions. 4-manifold specific.
- d09-d12 — Approach III infrastructure recycled from round 1. No
  construction techniques found.
- d13 (`2506.23994`) — Already consumed in E2.
- d14 (`math/0406607`) — Already consumed. BUT relevant to rotation-route
  lattice existence question.
- d16-d18 — Fixed-set rigidity/finiteness. Constraint data, not constructive.
- d19-d22 — Interface/Approach I. Already consumed.

## Original counts (for reference)
- `approach_ii`: 8 (1 read, 1 worth reading, 6 skip)
- `approach_iii`: 4 (all skip — no construction techniques found)
- `e2_odd_alt`: 6 (1 read, 5 skip/consumed)
- `interface`: 4 (all skip/consumed)

## Diagram Index

- `p7a23-d01` — The equivariant Spivak normal bundle and equivariant surgery for compact Lie groups (`1705.10909`)
  track: `approach_ii`; tags: Approach-II, G2, G3
- `p7a23-d02` — Cobordisms of global quotient orbifolds and an equivariant Pontrjagin-Thom construction (`1811.08794`)
  track: `approach_ii`; tags: Approach-II
- `p7a23-d03` — Geometric versus homotopy theoretic equivariant bordism (`math/0412550`)
  track: `approach_ii`; tags: Approach-II
- `p7a23-d04` — Bordisms of manifolds with proper action of a discrete group: signatures and descriptions of $G$-bundles (`1112.2104`)
  track: `approach_ii`; tags: Approach-II
- `p7a23-d05` — Computations of Complex Equivariant Bordism Rings (`math/9910024`)
  track: `approach_ii`; tags: Approach-II
- `p7a23-d06` — Positive scalar curvature with symmetry (`math/0512284`)
  track: `approach_ii`; tags: Approach-II
- `p7a23-d07` — Circle actions and scalar curvature (`1305.2288`)
  track: `approach_ii`; tags: Approach-II
- `p7a23-d08` — Corks, involutions, and Heegaard Floer homology (`2002.02326`)
  track: `approach_ii`; tags: Approach-II
- `p7a23-d09` — Translation Groupoids and Orbifold Bredon Cohomology (`0705.3249`)
  track: `approach_iii`; tags: Approach-III, G1
- `p7a23-d10` — Survey on Classifying Spaces for Families of Subgroups (`math/0312378`)
  track: `approach_iii`; tags: Approach-III, FJ
- `p7a23-d11` — Homology bounds for hyperbolic orbifolds (`2012.15322`)
  track: `approach_iii`; tags: Approach-III, E2
- `p7a23-d12` — On the number of finite subgroups of a lattice (`1209.2484`)
  track: `approach_iii`; tags: Approach-III, E2
- `p7a23-d13` — On reflections of congruence hyperbolic manifolds (`2506.23994`)
  track: `e2_odd_alt`; tags: E2-odd-alt, E2
- `p7a23-d14` — Finite Groups and Hyperbolic Manifolds (`math/0406607`)
  track: `e2_odd_alt`; tags: E2-odd-alt, E2
- `p7a23-d15` — Smith theory, L2 cohomology, isometries of locally symmetric manifolds and moduli spaces of curves (`1106.1704`)
  track: `e2_odd_alt`; tags: E2-odd-alt, Cross
- `p7a23-d16` — Finiteness of totally geodesic hypersurfaces (`2408.03430`)
  track: `e2_odd_alt`; tags: E2-odd-alt
- `p7a23-d17` — Rigidity of Totally Geodesic Hypersurfaces in Negative Curvature (`2306.01254`)
  track: `e2_odd_alt`; tags: E2-odd-alt
- `p7a23-d18` — Effective virtual and residual properties of some arithmetic hyperbolic 3-manifolds (`1806.02360`)
  track: `e2_odd_alt`; tags: E2-odd-alt
- `p7a23-d19` — Finiteness properties for some rational Poincaré duality groups (`1204.4667`)
  track: `interface`; tags: Approach-I, G1, E2
- `p7a23-d20` — Rational manifold models for duality groups (`1506.06293`)
  track: `interface`; tags: Approach-I, G2
- `p7a23-d21` — Bredon-Poincare Duality Groups (`1311.7629`)
  track: `interface`; tags: Approach-I, G1
- `p7a23-d22` — Surgery obstructions on closed manifolds and the Inertia subgroup (`0905.0104`)
  track: `interface`; tags: Approach-I, G3

## Diagrams

### p7a23-d01: The equivariant Spivak normal bundle and equivariant surgery for compact Lie groups
- Paper: `1705.10909`
- Track: `approach_ii`
- Tags: Approach-II, G2, G3
- Why selected: Core candidate for equivariant Spivak + surgery normal-structure control.
- `Q`: Construct equivariant normal data for group actions so fixed-set surgery can be controlled.
- `D`: Model manifold action with equivariant Spivak normal bundle rather than only ordinary normal data.
- `M`: Use equivariant surgery input to build/compare normal maps compatible with the action.
- `C`: Equivariant Spivak bundle and equivariant surgery theorems for compact Lie group actions.
- `O`: Concrete route to repair Approach II normal-structure and transfer-compatibility gaps.
- `B`: High-priority bridge for Approach II and for G2/G3-style compatibility constraints.

### p7a23-d02: Cobordisms of global quotient orbifolds and an equivariant Pontrjagin-Thom construction
- Paper: `1811.08794`
- Track: `approach_ii`
- Tags: Approach-II
- Why selected: Global quotient orbifold cobordism / equivariant Pontrjagin-Thom framework.
- `Q`: Translate global-quotient orbifold data into equivariant bordism information usable for manifold upgrades.
- `D`: Treat orbifolds as global quotients with equivariant Pontrjagin-Thom models.
- `M`: Compute orbifold cobordism classes via equivariant collapse maps and quotient structures.
- `C`: Equivariant Pontrjagin-Thom construction for global quotient orbifolds.
- `O`: Potential mechanism for Approach III resolution while tracking quotient/fundamental-group behavior.
- `B`: Strong cross-over module between Approach II equivariant surgery and Approach III orbifold route.

### p7a23-d03: Geometric versus homotopy theoretic equivariant bordism
- Paper: `math/0412550`
- Track: `approach_ii`
- Tags: Approach-II
- Why selected: Equivariant bordism comparison tools useful for surgery-entry arguments.
- `Q`: Advance Approach II: remove or control involution fixed-set effects while preserving target quotient group.
- `D`: Represent data in an equivariant bordism/surgery model of manifolds with group action.
- `M`: Apply method pattern from 'Geometric versus homotopy theoretic equivariant bordism' to fixed-set surgery or action-preserving modification steps.
- `C`: Use the paper's equivariant topology/cobordism/surgery theorem package as certificate.
- `O`: Candidate subroutine for action-compatible manifold upgrades in Approach II.
- `B`: Equivariant bordism comparison tools useful for surgery-entry arguments.

### p7a23-d04: Bordisms of manifolds with proper action of a discrete group: signatures and descriptions of $G$-bundles
- Paper: `1112.2104`
- Track: `approach_ii`
- Tags: Approach-II
- Why selected: Bordism with proper discrete-group actions; signatures and bundle control.
- `Q`: Track signatures and bundle data in bordisms with proper discrete-group actions.
- `D`: Model manifolds with proper actions via equivariant bordism and G-bundle descriptors.
- `M`: Use bordism invariants to constrain allowable surgery transformations under group actions.
- `C`: Signature/bundle classification statements for proper-action bordisms.
- `O`: Potential mechanism for proving Approach II surgeries preserve required global invariants.
- `B`: Method candidate for rigorous action-preserving surgery bookkeeping.

### p7a23-d05: Computations of Complex Equivariant Bordism Rings
- Paper: `math/9910024`
- Track: `approach_ii`
- Tags: Approach-II
- Why selected: Computational equivariant bordism ring background.
- `Q`: Advance Approach II: remove or control involution fixed-set effects while preserving target quotient group.
- `D`: Represent data in an equivariant bordism/surgery model of manifolds with group action.
- `M`: Apply method pattern from 'Computations of Complex Equivariant Bordism Rings' to fixed-set surgery or action-preserving modification steps.
- `C`: Use the paper's equivariant topology/cobordism/surgery theorem package as certificate.
- `O`: Candidate subroutine for action-compatible manifold upgrades in Approach II.
- `B`: Computational equivariant bordism ring background.

### p7a23-d06: Positive scalar curvature with symmetry
- Paper: `math/0512284`
- Track: `approach_ii`
- Tags: Approach-II
- Why selected: Surgery-with-symmetry pattern; potentially reusable obstruction mechanics.
- `Q`: Advance Approach II: remove or control involution fixed-set effects while preserving target quotient group.
- `D`: Represent data in an equivariant bordism/surgery model of manifolds with group action.
- `M`: Apply method pattern from 'Positive scalar curvature with symmetry' to fixed-set surgery or action-preserving modification steps.
- `C`: Use the paper's equivariant topology/cobordism/surgery theorem package as certificate.
- `O`: Candidate subroutine for action-compatible manifold upgrades in Approach II.
- `B`: Surgery-with-symmetry pattern; potentially reusable obstruction mechanics.

### p7a23-d07: Circle actions and scalar curvature
- Paper: `1305.2288`
- Track: `approach_ii`
- Tags: Approach-II
- Why selected: Group-action surgery/symmetry techniques; lower direct relevance but still useful.
- `Q`: Advance Approach II: remove or control involution fixed-set effects while preserving target quotient group.
- `D`: Represent data in an equivariant bordism/surgery model of manifolds with group action.
- `M`: Apply method pattern from 'Circle actions and scalar curvature' to fixed-set surgery or action-preserving modification steps.
- `C`: Use the paper's equivariant topology/cobordism/surgery theorem package as certificate.
- `O`: Candidate subroutine for action-compatible manifold upgrades in Approach II.
- `B`: Group-action surgery/symmetry techniques; lower direct relevance but still useful.

### p7a23-d08: Corks, involutions, and Heegaard Floer homology
- Paper: `2002.02326`
- Track: `approach_ii`
- Tags: Approach-II
- Why selected: Involution-centric 4-manifold techniques; heuristic transfer ideas.
- `Q`: Advance Approach II: remove or control involution fixed-set effects while preserving target quotient group.
- `D`: Represent data in an equivariant bordism/surgery model of manifolds with group action.
- `M`: Apply method pattern from 'Corks, involutions, and Heegaard Floer homology' to fixed-set surgery or action-preserving modification steps.
- `C`: Use the paper's equivariant topology/cobordism/surgery theorem package as certificate.
- `O`: Candidate subroutine for action-compatible manifold upgrades in Approach II.
- `B`: Involution-centric 4-manifold techniques; heuristic transfer ideas.

### p7a23-d09: Translation Groupoids and Orbifold Bredon Cohomology
- Paper: `0705.3249`
- Track: `approach_iii`
- Tags: Approach-III, G1
- Why selected: Orbifold groupoid/Bredon infrastructure for quotient-to-manifold translation.
- `Q`: Represent orbifold quotient constructions in a cohomological language compatible with group-action invariants.
- `D`: Use translation groupoids to encode orbifold action data and isotropy structure.
- `M`: Compute orbifold/Bredon invariants via groupoid-level functorial machinery.
- `C`: Translation-groupoid and orbifold Bredon cohomology comparison framework.
- `O`: Formal infrastructure for Approach III pi_1-aware resolution bookkeeping.
- `B`: Framework module: clarifies semantics but does not itself construct the closed manifold.

### p7a23-d10: Survey on Classifying Spaces for Families of Subgroups
- Paper: `math/0312378`
- Track: `approach_iii`
- Tags: Approach-III, FJ
- Why selected: Families/classifying-space machinery for orbifold/family control.
- `Q`: Control family-level classifying spaces needed in orbifold and assembly transitions.
- `D`: Parameterize subgroup families with E_F models and orbit-category functors.
- `M`: Move between family domains (Fin, VCyc, etc.) while preserving homological meaning.
- `C`: Classifying-space-for-families toolkit and comparison results.
- `O`: Reliable domain control for Approach III orbifold/family transitions and cross-checks.
- `B`: Infrastructure bridge supporting both Approach III and fallback assembly verifications.

### p7a23-d11: Homology bounds for hyperbolic orbifolds
- Paper: `2012.15322`
- Track: `approach_iii`
- Tags: Approach-III, E2
- Why selected: Hyperbolic orbifold homology bounds and geometric models.
- `Q`: Advance Approach III: resolve orbifold singular structure without losing pi_1 control.
- `D`: Model quotient/orbifold structure via groupoid/family-level descriptors.
- `M`: Use 'Homology bounds for hyperbolic orbifolds' techniques to connect orbifold invariants and manifold replacement operations.
- `C`: Invoke orbifold/topological-stack/family comparison results as certificate step.
- `O`: Potential resolution component with explicit orbifold bookkeeping.
- `B`: Hyperbolic orbifold homology bounds and geometric models.

### p7a23-d12: On the number of finite subgroups of a lattice
- Paper: `1209.2484`
- Track: `approach_iii`
- Tags: Approach-III, E2
- Why selected: Finite subgroup/isotropy complexity bounds in lattices/orbifolds.
- `Q`: Advance Approach III: resolve orbifold singular structure without losing pi_1 control.
- `D`: Model quotient/orbifold structure via groupoid/family-level descriptors.
- `M`: Use 'On the number of finite subgroups of a lattice' techniques to connect orbifold invariants and manifold replacement operations.
- `C`: Invoke orbifold/topological-stack/family comparison results as certificate step.
- `O`: Potential resolution component with explicit orbifold bookkeeping.
- `B`: Finite subgroup/isotropy complexity bounds in lattices/orbifolds.

### p7a23-d13: On reflections of congruence hyperbolic manifolds
- Paper: `2506.23994`
- Track: `e2_odd_alt`
- Tags: E2-odd-alt, E2
- Why selected: Reflections + fixed hypersurface geometry in congruence hyperbolic manifolds.
- `Q`: Control fixed-hypersurface geometry in congruence hyperbolic manifolds under reflections/involutions.
- `D`: Start from arithmetic reflective lattice and pass to congruence manifold covers.
- `M`: Track induced involutions and geometric properties of fixed totally geodesic hypersurfaces.
- `C`: Reflection/nonseparating fixed-set theorems in congruence hyperbolic settings.
- `O`: Concrete geometric substrate for E2 and for odd-dimension alternative hunts.
- `B`: Directly informs fixed-set side constraints needed before either Approach II or III closure.

### p7a23-d14: Finite Groups and Hyperbolic Manifolds
- Paper: `math/0406607`
- Track: `e2_odd_alt`
- Tags: E2-odd-alt, E2
- Why selected: Finite-group hyperbolic symmetry realization backup.
- `Q`: Realize prescribed finite symmetries in compact hyperbolic manifolds for action-construction flexibility.
- `D`: Encode target finite symmetry as a normalizer quotient in finite-index lattice subgroups.
- `M`: Use subgroup growth and commensurator control to force chosen finite isometry groups.
- `C`: Finite-group realization theorem for compact hyperbolic manifold isometry groups.
- `O`: Symmetry-existence supply for building candidate involution/reflection action models.
- `B`: Useful constructor for alternate E2 branches, but fixed-set Euler behavior is not automatic.

### p7a23-d15: Smith theory, L2 cohomology, isometries of locally symmetric manifolds and moduli spaces of curves
- Paper: `1106.1704`
- Track: `e2_odd_alt`
- Tags: E2-odd-alt, Cross
- Why selected: Smith/L2 constraints on periodic actions in locally symmetric settings.
- `Q`: Find odd-dimensional E2 alternatives where fixed sets can have chi=0 despite even dimension.
- `D`: Parameterize hyperbolic involution/reflection models by fixed-set geometry and arithmetic constraints.
- `M`: Extract fixed-set construction/rigidity behavior from 'Smith theory, L2 cohomology, isometries of locally symmetric manifolds and moduli spaces of curves'.
- `C`: Use geometric fixed-set theorems/constraints as certificate.
- `O`: Constraint map for viable odd-dimensional E2 candidate families.
- `B`: Smith/L2 constraints on periodic actions in locally symmetric settings.

### p7a23-d16: Finiteness of totally geodesic hypersurfaces
- Paper: `2408.03430`
- Track: `e2_odd_alt`
- Tags: E2-odd-alt
- Why selected: Finiteness/rigidity constraints for totally geodesic hypersurfaces.
- `Q`: Find odd-dimensional E2 alternatives where fixed sets can have chi=0 despite even dimension.
- `D`: Parameterize hyperbolic involution/reflection models by fixed-set geometry and arithmetic constraints.
- `M`: Extract fixed-set construction/rigidity behavior from 'Finiteness of totally geodesic hypersurfaces'.
- `C`: Use geometric fixed-set theorems/constraints as certificate.
- `O`: Constraint map for viable odd-dimensional E2 candidate families.
- `B`: Finiteness/rigidity constraints for totally geodesic hypersurfaces.

### p7a23-d17: Rigidity of Totally Geodesic Hypersurfaces in Negative Curvature
- Paper: `2306.01254`
- Track: `e2_odd_alt`
- Tags: E2-odd-alt
- Why selected: Rigidity of totally geodesic hypersurfaces in negative curvature.
- `Q`: Find odd-dimensional E2 alternatives where fixed sets can have chi=0 despite even dimension.
- `D`: Parameterize hyperbolic involution/reflection models by fixed-set geometry and arithmetic constraints.
- `M`: Extract fixed-set construction/rigidity behavior from 'Rigidity of Totally Geodesic Hypersurfaces in Negative Curvature'.
- `C`: Use geometric fixed-set theorems/constraints as certificate.
- `O`: Constraint map for viable odd-dimensional E2 candidate families.
- `B`: Rigidity of totally geodesic hypersurfaces in negative curvature.

### p7a23-d18: Effective virtual and residual properties of some arithmetic hyperbolic 3-manifolds
- Paper: `1806.02360`
- Track: `e2_odd_alt`
- Tags: E2-odd-alt
- Why selected: Arithmetic hyperbolic residual/congruence control with reflection context.
- `Q`: Find odd-dimensional E2 alternatives where fixed sets can have chi=0 despite even dimension.
- `D`: Parameterize hyperbolic involution/reflection models by fixed-set geometry and arithmetic constraints.
- `M`: Extract fixed-set construction/rigidity behavior from 'Effective virtual and residual properties of some arithmetic hyperbolic 3-manifolds'.
- `C`: Use geometric fixed-set theorems/constraints as certificate.
- `O`: Constraint map for viable odd-dimensional E2 candidate families.
- `B`: Arithmetic hyperbolic residual/congruence control with reflection context.

### p7a23-d19: Finiteness properties for some rational Poincaré duality groups
- Paper: `1204.4667`
- Track: `interface`
- Tags: Approach-I, G1, E2
- Why selected: FH(Q) criterion and fixed-set Euler mechanism.
- `Q`: Provide interface support between Approach II/III work and legacy Approach I machinery.
- `D`: Express assumptions/results in family-level or surgery-obstruction language.
- `M`: Leverage 'Finiteness properties for some rational Poincaré duality groups' as an interface theorem linking constructions to obstruction frameworks.
- `C`: Use foundational assembly/PD/surgery statements from the paper.
- `O`: Compatibility checks preventing hidden hypothesis mismatches.
- `B`: FH(Q) criterion and fixed-set Euler mechanism.

### p7a23-d20: Rational manifold models for duality groups
- Paper: `1506.06293`
- Track: `interface`
- Tags: Approach-I, G2
- Why selected: Rational surgery/manifold model route; partial bridge.
- `Q`: Provide interface support between Approach II/III work and legacy Approach I machinery.
- `D`: Express assumptions/results in family-level or surgery-obstruction language.
- `M`: Leverage 'Rational manifold models for duality groups' as an interface theorem linking constructions to obstruction frameworks.
- `C`: Use foundational assembly/PD/surgery statements from the paper.
- `O`: Compatibility checks preventing hidden hypothesis mismatches.
- `B`: Rational surgery/manifold model route; partial bridge.

### p7a23-d21: Bredon-Poincare Duality Groups
- Paper: `1311.7629`
- Track: `interface`
- Tags: Approach-I, G1
- Why selected: Bredon-PD framing for torsion groups.
- `Q`: Provide interface support between Approach II/III work and legacy Approach I machinery.
- `D`: Express assumptions/results in family-level or surgery-obstruction language.
- `M`: Leverage 'Bredon-Poincare Duality Groups' as an interface theorem linking constructions to obstruction frameworks.
- `C`: Use foundational assembly/PD/surgery statements from the paper.
- `O`: Compatibility checks preventing hidden hypothesis mismatches.
- `B`: Bredon-PD framing for torsion groups.

### p7a23-d22: Surgery obstructions on closed manifolds and the Inertia subgroup
- Paper: `0905.0104`
- Track: `interface`
- Tags: Approach-I, G3
- Why selected: Closed-manifold subgroup/inertia interface in surgery obstruction groups.
- `Q`: Provide interface support between Approach II/III work and legacy Approach I machinery.
- `D`: Express assumptions/results in family-level or surgery-obstruction language.
- `M`: Leverage 'Surgery obstructions on closed manifolds and the Inertia subgroup' as an interface theorem linking constructions to obstruction frameworks.
- `C`: Use foundational assembly/PD/surgery statements from the paper.
- `O`: Compatibility checks preventing hidden hypothesis mismatches.
- `B`: Closed-manifold subgroup/inertia interface in surgery obstruction groups.
