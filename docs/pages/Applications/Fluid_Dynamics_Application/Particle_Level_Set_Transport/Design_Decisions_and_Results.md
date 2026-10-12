---
title: Design decisions and results
keywords: [particle level set design decisions results dam break LEoPart beta]
tags: [Design_Decisions_and_Results.md]
sidebar: fluid_dynamics_application
summary: Why the method is set up as it is, measured results on 2D and 3D cases, and known limitations.
---

## Design decisions

### Conserve an indicator, not the signed distance

The PDE-constrained projection conserves $$\int c_h$$ of whatever field it projects. For a 0/1 phase indicator this integral is the volume of one of the fluids, which is the quantity to protect. Conserving the integral of a signed distance has no physical meaning, and the projection of a discontinuous indicator is a poor source of interface geometry.

The method therefore uses a **dual representation**. Particles carry the indicator, which is conserved, and a signed distance, which gives the geometry. The two are linked by a global shift of the nodal `DISTANCE`. This keeps the two-fluid elements, the redistancing, the surface tension and the curvature computations unchanged.

### Use the existing continuous velocity

LEoPart pairs the projection with an H(div)-conforming, pointwise divergence-free HDG velocity. Kratos two-fluid velocities are continuous P1, which already have a single-valued normal component on every face. That is the only property the cell-wise and global conservation proofs need.

No velocity reconstruction is done. The velocity is only weakly divergence-free, and the resulting inconsistency between the particle (material) form and the constraint (conservative) form appears as a small over/undershoot, not as a loss of volume.

### P1 cells, P0 multiplier, P1 facets, static condensation

These follow LEoPart: $$k = 1$$ and $$l = 0$$. The P0 multiplier makes the scheme independent of the θ time parameter, while the P1 facet field keeps $$k+1$$ accuracy for smooth data. The element unknowns are condensed, so the global system only contains facet unknowns.

### Penalties scaled with the element size

LEoPart's β and ζ were tuned on unit-sized meshes. The terms they multiply scale with the element size, so they are scaled as $$\beta_K = \beta\,h_K^{1-d}$$ and $$\zeta_K = \zeta\,h_K^{2-d}$$. The defaults then mean the same on meshes in metres or millimetres.

### Gradient penalty zeta

Without a gradient penalty the projection of a sharp indicator oscillates strongly. In the 2D test below the range was [-1.15, 2.06] with ζ = 0, which is consistent with the [-11.2, 8.04] that LEoPart reports for the slotted disk. ζ close to the number of particles per element, as recommended by LEoPart, reduces it to about [0.00, 1.00] in 2D. The defaults are ζ = 25 in 2D and ζ = 40 in 3D, equal to the initial particles per element.

### Facet penalty beta

LEoPart uses β = 1e-6, and this implementation uses **β = 1e-2**.

With a very small β, the facet unknowns that carry no flux are controlled only by β. In 3D each facet has 3 unknowns but contributes a single flux, and each element imposes a single constraint, so most of the facet space is nearly undetermined. Two consequences were measured:

1. **Mismatch mass spreads over the domain.** The least-squares fit prefers to spread any particle/mesh mismatch thinly over all flow-connected cells, because spreading mass $$m$$ over $$N$$ cells costs about $$m^2/N$$. In the 2D test this left a uniform 1e-4 to 1e-3 layer of indicator everywhere after a few steps, which leaks through open boundaries.
2. **The facet system becomes very ill-conditioned**, with a condition number of order flux/β. Only direct solvers work: LEoPart uses MUMPS. On the 3D dam break, an iterative solve does not converge and the simulation diverges, while a sparse LU does not fit in memory (see [results](#3d-dam-break)).

A larger β makes the facet value closer to the average of the two traces, which is a central flux. That improves the conditioning but increases the over/undershoot. β = 1e-2 converges reliably with a simple iterative solver and keeps the overshoot near the 1-2% LEoPart reports for practical cases in 2D, about 10-20% at the start of the 3D dam break.

The conservation does not depend on β: it holds for any value. With a capable direct solver β = 1e-6 can still be used.

### Facet system solver

The condensed facet system is symmetric positive definite, but it is not elliptic: it is a sum of rank-one flux blocks plus a small penalty. Algebraic multigrid coarsening does not fit it, so the default is **conjugate gradients with a Jacobi relaxation preconditioner** (AMGCL `relaxation` with `damped_jacobi`, tolerance 1e-8). The previous solution is used as initial guess.

In MPI the AMGCL relaxation preconditioner does not read `smoother_type`: the serial wrapper writes it to `precond.relax.type`, a key the MPI relaxation preconditioner does not read. AMGCL prints "Unknown parameter relax" and uses its default single-level relaxation, which still converges to the requested tolerance.

### Boundary fluxes from the level set, and flux-free walls

LEoPart uses the interior trace of the projected field for the outflow. With that choice the thin indicator layer spread over the air leaves through open air boundaries. The dam break lost 5.3e-4 of the water volume in 0.2 s through its top outlet, which only air reaches in that time.

The dual representation already defines where the water crosses the boundary: where `DISTANCE < 0`. Open boundary faces therefore use the level set indicator $$g = H(-d)$$ for both inflow and outflow (`outflow_flux_from_level_set: true`, the default).

Slip walls must then carry **no** flux. Their nodal velocity is tangent to nodal normals averaged over several faces, so at edges and corners it has a normal component relative to each face. With the level set flux this created 6.0e-4 of the water volume in 0.2 s. Faces whose nodes are all slip (`SLIP`) or fixed at zero velocity are classified as walls and excluded from the fluxes, and particles leaving through them are pushed back.

### Volume correction by a global shift

The cell-wise conservation is carried by $$c_h$$, while `DISTANCE` only has to match its total volume. A single global shift, found by bracketing and regula falsi on the exact negative volume, is robust and cheap. Its largest value in the dam break was below 1e-4 m. A local correction could be considered in the future.

### Exact initialization

The initial $$c_h$$ is the exact element-wise L2 projection of $$H(-d)$$, integrated on the cut sub-elements, rather than a projection of the particles. The initial conserved volume is then exactly the negative volume of the initial `DISTANCE`.

### No clipping

$$c_h$$ is not clipped to [0, 1], because clipping would destroy the conservation. It is only used through its integral (the volume target) and for visualization.

### Other choices

* **`mass_source`** is disabled with the particle transport, because the transport conserves the volume itself.
* **FM-ALE** velocity correction of the hydraulic solver is kept, because it only modifies the velocity history.
* **Zero distances.** Elements with nodal distances exactly zero are split with the same criterion as the geometry division utilities (zero values belong to neither side). The criterion of `FluidAuxiliaryUtilities::IsSplit`, which counts zero as negative, made the division utilities throw "non-split element" errors.
* **MPI.** It reuses Kratos' `DataCommunicator` and the Trilinos extension rather than new communication layers. See [MPI parallelization](MPI.html).

## Results

### Unit and integration tests

* **Local system.** For any facet values, the local conservation residual is below 1e-12 times the element volume, and the Schur complement is symmetric.
* **Linear fields.** Particles sampled from a linear field, with zero velocity, reproduce it to 1e-9 in 2D and 3D, so $$P_E \circ P_L = I$$.
* **Conservation.** With a rotation through an open box, the global balance $$\int c_h - \int c^* + \text{outflow}$$ is below 1e-13 and the cell residuals are below 1e-11, in 2D and 3D and for both outflow options.
* **Advection.** A particle rotated once around the centre with RK3 returns to its position within 1.6e-5, the RK3 time error for 100 steps.
* **Sloshing tanks** (2D and 3D, `two_fluids` and hydraulic solvers). The water volume stays constant to round-off, and the indicator range stays within [-0.004, 1.009] in 2D and [-0.09, 1.06] in 3D.

### 2D penalty study

Unit square, 16×16 triangles, about 16 particles per element, a circle advected by a rotation that crosses the open box boundary, Δt = 0.01. The volume drift comes only from the open boundaries.

Gradient penalty, β = 1e-6, after 10 steps:

| ζ | Indicator range |
|---|---|
| 0 | [-1.151, 2.058] |
| 4 | [-0.050, 1.067] |
| 16 | [0.000, 1.001] |

Facet penalty, ζ = 16, after 30 steps:

| β | Volume drift through the boundary | Indicator range | Max far-field indicator |
|---|---|---|---|
| 1e-6 | 4.1e-5 | [0.000, 1.000] | 2.0e-4 |
| 1e-2 | 1.7e-4 | [-0.004, 1.017] | 3.4e-5 |
| 3e-2 | 1.3e-4 | [-0.015, 1.040] | 2.1e-3 |
| 1e-1 | 1.1e-4 | [-0.030, 1.069] | 8.1e-4 |

These runs used the LEoPart outflow (interior trace) and therefore show the leakage of the spread mass. With the default level set boundary flux, the leakage is removed.

### 3D dam break

The case is a 3D dam break solved with the hydraulic solver: 17.4k nodes, 92k tetrahedra, Δt = 0.002 s, 100 steps (0.2 s), 40 particles per element (3.7M particles), with a top outlet and slip walls.

Solver and penalty choices, all with LEoPart's trace outflow:

| Setup | Outcome |
|---|---|
| β = 1e-6, AMGCL GMRES + AMG | Facet solve stalls (relative residual 6e-3, 0.13 and 0.20 at steps 2-4, after 500 iterations). 33-47 s per step in `InitializeSolutionStep`. The indicator explodes ([-50, 8] at step 20, [-4365, 10190] at step 30) and the volume collapses. |
| β = 1e-6, Eigen sparse LU | The first factorization exceeded the available 22 GB of memory and was stopped. |
| β = 1e-6, CG + Jacobi, SPAI0 or BiCGStab + ILU0 | Converges at first, then reaches the 2000-iteration limit at a residual of about 2e-8. 12-49 s per step in `InitializeSolutionStep`. |
| β = 1e-3, CG + Jacobi | Stable, indicator up to 1.053, but the `InitializeSolutionStep` time grows from 2.3 to 6.8 s as the flow accelerates. |
| β = 1e-2, CG + Jacobi | Stable, residual below 1e-8 every step, 2.4-3.4 s per `InitializeSolutionStep` over the first 30 steps, indicator in [-0.10, 1.12]. |

Boundary flux choices, with β = 1e-2 and CG + Jacobi, over 0.2 s:

<a id="boundary-fluxes"></a>

| Boundary flux | Water volume $$\int c_h$$ (initial 0.67419686) |
|---|---|
| Interior trace on outflow (LEoPart) | 0.67383862 (-5.3e-4): the spread indicator leaves through the air outlet |
| Level set on all boundary faces | 0.67460296 (+6.0e-4): spurious flux through the slip walls |
| Level set on open faces, walls flux-free (**default**) | 0.67419686 at every step, zero boundary flux |

With the defaults:

* the cell residuals stay around 5e-16 and the facet solve below 1e-8 every step;
* the indicator stays in [-0.12, 1.20];
* the solver `InitializeSolutionStep` (FM-ALE correction, particle transport and distance modification) takes 2.2-7.3 s, against about 1.1 s with the Eulerian convection;
* the whole run takes 537 s, against 228 s with the default Eulerian transport (SUPG, with the global `mass_source` correction).

The initial conserved volume (0.67420) is slightly larger than the negative volume of the initial `DISTANCE` (0.67404). The distance modification process moves near-zero nodal distances before the first step, and the initialization uses the modified field.

### MPI

On the closed vortex with 32,768 triangles and 820k particles, 4 MPI ranks run the transport step 3.0 times faster than one thread, with the same conservation and accuracy. See [MPI parallelization](MPI.html#results).

## Known limitations

* **Bounds.** The projected indicator is not bounded. ζ limits the over/undershoot, and it is larger at the start of 3D runs and near fast-moving interfaces.
* **Volume correction.** `DISTANCE` matches the conserved volume through a single global shift. Local volume errors of the level set are not corrected.
* **Cost.** The transport is more expensive than the Eulerian convection. In the 3D dam break, `InitializeSolutionStep` took 2 to 7 times longer and the whole run 2.4 times longer. The cost is dominated by the particle advection and the facet solve.
* **Solvers.** No MPI hydraulic solver exists, and the fractional hydraulic solver does not use the particle transport.
* **Meshes.** Remeshing, moving meshes and restarts are not supported.
* **MPI accuracy.** Runge-Kutta stage points beyond a partition use the clipped local velocity, which is first-order accurate there.
* **Partitioning.** MPI runs need a partitioned model part (METIS). The stripe partitioner of the tests is for testing only.

## Related fixes

The following problems in existing code were found and fixed during this work:

* `FluidAuxiliaryUtilities` volume functions (`CalculateFluidVolume`, `CalculateFluidPositive/NegativeVolume` and the cut-element variants) discarded the result of their `SumAll` reduction. In MPI they returned each rank's local volume; they now return the global one, which also affects the `mass_source` correction of the MPI two-fluid solver.
* `python_solvers_wrapper_fluid.py` mapped `solver_type: "two_fluids"` to a non-existent module, so serial and OpenMP `FluidDynamicsAnalysis` runs could not load the two-fluid solver. The `TwoFluidMassConservationTest` cases now run, and still fail on their reference values (issue #9592).
