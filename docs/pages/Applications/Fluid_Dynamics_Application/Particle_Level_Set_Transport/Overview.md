---
title: Overview
keywords: [two fluids level set particles LEoPart PDE-constrained projection mass conservation]
tags: [Overview.md]
sidebar: fluid_dynamics_application
summary: Conservative Lagrangian particle transport of the interface for the two-fluid solvers (LEoPart PDE-constrained particle-mesh projection).
---

## What it is

The two-fluid solvers locate the interface between the fluids with the zero level set of the nodal `DISTANCE`. By default `DISTANCE` is transported with an Eulerian convection (`LevelSetConvectionProcess`, SUPG or algebraic stabilization, optionally with BFECC). This transport is diffusive and does not conserve the volume of each fluid. The only fixes available are global and heuristic: the `mass_source` correction and `MassConservationCheckProcess`.

The **particle level set transport** is an alternative to the Eulerian convection. It is based on the PDE-constrained particle-mesh projection of the LEoPart library (Maljaars et al. 2019, 2021):

* Lagrangian particles carry the interface information, so the advection itself has no numerical diffusion.
* Every time step the particle phase indicator is projected onto a discontinuous P1 field $$c_h$$. The projection is a least-squares fit constrained by a discrete conservation law. The integral of $$c_h$$ therefore changes **exactly by the fluxes through the element faces**, in every element and globally, up to round-off.
* The nodal `DISTANCE` used by the two-fluid elements is rebuilt from the particles. It is then shifted so that the volume where `DISTANCE < 0` equals the conserved volume $$\int c_h$$.

The two-fluid elements, the redistancing and the distance modification process are not modified.

The pages of this section describe:

* [Algorithm](Algorithm.html): the formulation and one time step in detail.
* [Implementation](Implementation.html): the classes, files, Python integration and tests.
* [MPI parallelization](MPI.html): the distributed algorithm (particle migration, distributed facet system).
* [Design decisions and results](Design_Decisions_and_Results.html): why the defaults are what they are, measured results and known limitations.

## When to use it

Use it when the volume of each fluid must be preserved, for example long sloshing or dam break simulations, where the Eulerian transport loses or gains volume. Expect the following:

* **Conservation.** The water volume is conserved to round-off in closed domains. With inlets or outlets it changes only by the fluxes through the open boundaries.
* **Cost.** The transport step is several times more expensive than the Eulerian convection. On the 3D dam break benchmark (92k tetrahedra, 3.7M particles) the whole simulation took about 2.4 times longer (see [results](Design_Decisions_and_Results.html#results)).
* **Over/undershoot.** The projected indicator is not bounded, with typical values in [-0.1, 1.2] next to the interface. It is only used through its integral and for visualization.

## Availability

| Solver | `solver_type` | Serial / OpenMP | MPI |
|---|---|---|---|
| Navier-Stokes two fluids | `two_fluids` / `TwoFluids` | Yes | Yes (Trilinos) |
| Two-fluid hydraulic (FluidDynamicsHydraulicsApplication) | `two_fluid_hydraulic` | Yes | No MPI solver exists |
| Two-fluid hydraulic fractional | `two_fluid_hydraulic_fractional` | No | No |

## Usage

Set `levelset_transport_type` in the `solver_settings`. The default is `"eulerian"`, the existing behaviour:

```json
"solver_settings": {
    "solver_type": "two_fluids",
    "levelset_transport_type": "particle_pde_projection",
    "particle_pde_projection_settings": {
        "echo_level": 1
    }
}
```

The `levelset_convection_settings` are ignored when the particle transport is used. When `mass_source` is `true` it is switched off, with a message, because the transport conserves the volume by itself.

To visualize the projected indicator, add the non-historical variable `PHASE_FRACTION` to the output:

```json
"nodal_data_value_variables": ["PHASE_FRACTION"],
"element_data_value_variables": ["PHASE_FRACTION"]
```

### Settings

All the settings are optional. The defaults depend on the dimension:

| Setting | 2D default | 3D default | Description |
|---|---|---|---|
| `echo_level` | 0 | 0 | 1: volume, conservation residuals, indicator range, distance shift and facet solver residual of every step. 2: also particle counts. |
| `particles_per_element.initial` | 25 | 40 | Particles seeded per element at the first step. |
| `particles_per_element.minimum` | 15 | 25 | Elements with fewer particles are reseeded. Must be at least the number of element nodes. |
| `particles_per_element.maximum` | 50 | 80 | Elements with more particles are thinned. |
| `runge_kutta_order` | 3 | 3 | Order (1, 2 or 3) of the explicit particle integration. |
| `max_CFL` | 1.0 | 1.0 | Particle CFL per substep, which sets the number of substeps of each particle. |
| `beta` | 1e-2 | 1e-2 | Facet penalty β (see [Design decisions](Design_Decisions_and_Results.html#facet-penalty-beta)). |
| `zeta` | 25 | 40 | Gradient penalty ζ, close to the number of particles per element. |
| `volume_correction.enabled` | true | true | Shift `DISTANCE` so that its negative volume equals the conserved volume. |
| `volume_correction.relative_tolerance` | 1e-10 | 1e-10 | Tolerance of the shift, relative to the domain volume. |
| `volume_correction.max_iterations` | 50 | 50 | Maximum number of iterations of the shift. |
| `outflow_flux_from_level_set` | true | true | Outflow boundary flux computed from the level set (true) or from the projected field (false, as in LEoPart). |
| `update_particle_distance_after_redistancing` | true | true | Transfer the change of `DISTANCE` made after the transport (redistancing) to the particles. |
| `random_seed` | 0 | 0 | Seed of the particle sampling. |
| `particles_output.model_part_name` | "" | "" | If set, the particles are copied to a model part with this name, created in the same `Model`, so they can be written to output (see [below](#visualizing-the-particles)). |
| `particles_output.interval` | 1 | 1 | Copy the particles in the first step and every `interval` steps. |
| `particles_output.stride` | 1 | 1 | Copy only every `stride`-th particle, to keep 3D output files small. |
| `linear_solver_settings` | see below | see below | Solver of the global facet system. |

The default facet solver is AMGCL conjugate gradients with a single-level relaxation preconditioner (AMGCL's default SPAI0). Algebraic multigrid does not help for this system; see [Design decisions](Design_Decisions_and_Results.html#facet-system-solver).

```json
"linear_solver_settings": {
    "solver_type"         : "amgcl",
    "krylov_type"         : "cg",
    "preconditioner_type" : "relaxation",
    "tolerance"           : 1e-8,
    "max_iteration"       : 1000,
    "block_size"          : 1
}
```

In MPI runs the same settings are passed to the Trilinos linear solver factory.

## Visualizing the particles

The particles are internal data of the process. To see them, set `particles_output.model_part_name`:

* The process creates a model part with that name. Each particle becomes one node with a point condition, carrying its distance (`DISTANCE`) and indicator (`PHASE_FRACTION`) as non-historical nodal values.
* The model part shares the `ProcessInfo` of the fluid, so any output process can write it.
* In MPI runs, each rank writes its own particles.

```json
"solver_settings": {
    "levelset_transport_type": "particle_pde_projection",
    "particle_pde_projection_settings": {
        "particles_output": {
            "model_part_name": "FluidParticles",
            "interval": 25,
            "stride": 10
        }
    }
},
"output_processes": {
    "vtk_output": [{
        "python_module": "vtk_output_process",
        "kratos_module": "KratosMultiphysics",
        "Parameters": {
            "model_part_name": "FluidParticles",
            "output_control_type": "step",
            "output_interval": 25,
            "output_path": "vtk_output_particles",
            "nodal_data_value_variables": ["DISTANCE", "PHASE_FRACTION"]
        }
    }]
}
```

Use `"output_control_type": "step"` with the same interval as `particles_output.interval`. The particles are copied in the first step and at every multiple of the interval, which are exactly the steps the step-controlled output writes. In ParaView, display the particle files with the *Point Gaussian* representation and colour them by `PHASE_FRACTION`.

The copy can also be triggered from Python at any time with `process.UpdateParticlesModelPart(model_part, stride)`. In MPI runs, call it on every rank.

A 3D case with tens of particles per element produces large particle files (e.g. 3.7M particles for 92k tetrahedra), so use `stride` and a coarse `interval`.

## References

* J.M. Maljaars, R.J. Labeur, N. Trask, D. Sulsky. *Conservative, high-order particle-mesh scheme with applications to advection-dominated flows.* Computer Methods in Applied Mechanics and Engineering 348 (2019) 443-465. arXiv:1806.09916.
* J.M. Maljaars, C.N. Richardson, N. Sime. *LEoPart: A particle library for FEniCS.* Computers & Mathematics with Applications 81 (2021) 289-315. arXiv:1912.13375.
* J.M. Maljaars, R.J. Labeur, M. Möller. *A hybridized discontinuous Galerkin framework for high-order particle-mesh operator splitting of the incompressible Navier-Stokes equations.* Journal of Computational Physics 358 (2018) 150-172.
