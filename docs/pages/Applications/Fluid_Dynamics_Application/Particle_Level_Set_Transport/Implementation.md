---
title: Implementation
keywords: [particle level set implementation classes MeshFacetTable LagrangianParticleContainer PdeConstrainedProjectionUtility]
tags: [Implementation.md]
sidebar: fluid_dynamics_application
summary: Classes, files, Python integration and tests of the particle level set transport.
---

## Overview of the classes

All the C++ code is in the FluidDynamicsApplication. The distributed linear algebra is in its Trilinos extension.

| Class | File | Role |
|---|---|---|
| `MeshFacetTable<TDim>` | `custom_utilities/particle_mesh/mesh_facet_table.h/.cpp` | Cached simplex topology: affine shape functions, area normals, neighbours, global facet numbering (also across MPI partitions). |
| `LagrangianParticleContainer<TDim>` | `custom_utilities/particle_mesh/lagrangian_particle_container.h/.cpp` | Particle storage, seeding, Runge-Kutta advection, point location, population control, MPI migration. |
| `PdeConstrainedProjectionUtility<TDim>` | `custom_utilities/particle_mesh/pde_constrained_projection_utility.h/.cpp` | Local KKT systems, static condensation, assembly, recovery, $$\ell^2$$ projection, initialization, volumes. |
| `FacetLinearSystem` (abstract), `SerialFacetLinearSystem` | `custom_utilities/particle_mesh/facet_linear_system.h/.cpp` | Global system of the facet unknowns. The serial version uses a ublas CSR matrix. |
| `TrilinosFacetLinearSystem` | `trilinos_extension/custom_utilities/trilinos_facet_linear_system.h` | Distributed (Epetra) facet system for MPI runs. |
| `ParticleMeshPdeProjectionProcess<TDim>` | `custom_processes/particle_mesh_pde_projection_process.h/.cpp` | Process that runs one transport step and writes `DISTANCE` and `PHASE_FRACTION`. |

All templates are instantiated for 2D (triangles) and 3D (tetrahedra).

## MeshFacetTable

Built once from the elements of the model part. Elements and nodes are referred to by their **position** in the model part containers (`ElementIndex`, `NodeIndices`), not by Id.

`ElementData` stores, for each element:

* `NodeIndices`, `Centroid`, `DN_DX`, `Volume` and `Size` (the minimum height, $$1/\max_i \vert\nabla N_i\vert$$);
* `FaceAreaNormals`: the outward normal of face $$i$$ (opposite to node $$i$$) scaled by the face measure. It is computed as $$-d\,\vert K\vert\,\nabla N_i$$.
* `Neighbours`: the local element across each face, or `Invalid` (-1);
* `Facets`: the local index of each shared face, interior or MPI partition facet, or `Invalid` for boundary faces;
* `PartitionRanks`: the rank across each partition facet, or `Invalid`.

The shape functions are affine: $$N_i(x) = \tfrac{1}{d+1} + \nabla N_i \cdot (x - x_c)$$. The same evaluation gives the barycentric coordinates used to locate points.

**Facet matching.** All element faces are sorted by their sorted node Ids, so that the faces shared by two elements become consecutive. The area normal of the first element (the owner) is copied, with the opposite sign, to the second one, so both sides use bitwise opposite normals and the fluxes cancel exactly.

**Facet DOFs.** A facet DOF is `global_facet_id * TDim + rank`, where `rank` is the position of the vertex Id among the sorted Ids of the facet. Both elements sharing a facet obtain the same DOFs without any communication.

Main methods:

* `ShapeFunctions(element, x, N)` and `FacetDof(element, face, node)`;
* `FaceMassCoefficient(area, same_node)`;
* `ComputeElementFacetDofs` (the coupled DOF blocks of every element, used to build the sparsity) and `FindPartitionFacet`;
* the owned DOF range: `NumberOfOwnedFacetDofs`, `FirstOwnedFacetDof` and `NumberOfFacetDofs` (global);
* `GetDataCommunicator`, `IsDistributed` and `GetMinimumElementSize` (global minimum).

## LagrangianParticleContainer

**Particles.** A particle is a small struct: `Coordinates`, `Indicator`, `Distance` and `Element` (the host element position).

**Storage.** The particles are kept in a flat `std::vector`, sorted by host element with a counting sort (`Rebucket`) after every operation that moves, adds or removes them. The particles of an element are therefore contiguous and are accessed with `ParticlesBegin(element)` and `ParticlesEnd(element)`.

| Method | Description |
|---|---|
| `Seed(n, nodal_distance)` | Seeds $$n$$ random particles per element. |
| `ClassifyFaces(table, velocity, is_closed_node, is_inlet_node, face_types)` | Static. Classifies the faces as `Interior`, `Wall` (all nodes closed), `Inlet` (all nodes flagged inlet), `Outflow` or `Backflow` (the other open faces, by the sign of the average normal velocity). |
| `Advect(old_velocity, new_velocity, face_types, dt, max_CFL, rk_order)` | Advects the particles (OpenMP over particles). Returns the number of particles that left the domain. Particles crossing MPI partitions are migrated (see [MPI](MPI.html)). |
| `ControlPopulation(min, max, face_types, nodal_distance, c_h, phi_h)` | Keeps the particle count per element between the bounds (OpenMP over elements). |
| `IncrementDistance(nodal_increment)` | FLIP-like update of the particle distances. |
| `Locate(x, start_element)` | Neighbour walk with a bins fallback. Returns the host element or the exit face (and the exit rank for partition facets). |
| `SetParticles(particles)` | Replaces the particles, used by the tests. |

The random generator is a small counter-based SplitMix64, seeded per element and per call from the element **Id**. Results are therefore reproducible and independent of the number of threads and of the MPI partitioning.

## PdeConstrainedProjectionUtility

* `BuildLocalSystem`, `CondenseLocalSystem`, `RecoverLocalSolution` and `ComputeCellResidual` are static and public, so they can be unit tested on a single element.
* `Project(particles, flux_velocity, face_types, nodal_distance, settings, facet_system, c_h)` drives one projection:
  1. Initializes the facet system on the first call.
  2. Assembles the condensed contributions in parallel.
  3. Solves the facet system.
  4. Recovers $$c_h$$ element by element and returns a `ProjectionInfo` with the global integrals, the net boundary outflow, the maximum relative cell residual, the range of $$c_h$$ and the facet solve residual.

  Every element system is built twice, for the assembly and for the recovery, instead of being stored. This is cheap and avoids storing about 0.5 KB per tetrahedron.
* `L2ProjectDistance`, `InitializeIndicatorFromDistance`, `ComputeNegativeVolume` and `ComputeIntegral` are static helpers. The volume and integral helpers are reduced over the MPI ranks.
* `SolveDenseSystem<N, M>` is a small Gaussian elimination with partial pivoting on bounded matrices, used for the saddle-point blocks, which have a zero diagonal entry.

## FacetLinearSystem

An abstract interface that decouples the projection from the linear algebra:

```cpp
virtual void Initialize(NumberOfOwnedDofs, FirstOwnedDof, TotalDofs, rElementDofBlocks);
virtual void BeginAssemble();
virtual void Assemble(Size, pDofs, pLHS, LHSRowStride, pRHS);   // thread-safe
virtual double Solve();                                          // returns the relative residual
virtual double GetValue(GlobalDof) const;                        // any DOF referenced by a local element
```

* **`SerialFacetLinearSystem`** builds a ublas CSR matrix from the element blocks, assembles with atomic additions and solves with any Kratos linear solver. The previous solution is used as initial guess.
* **`TrilinosFacetLinearSystem`** builds an `Epetra_Map` over the owned contiguous range and the graph with `TrilinosSpace::BuildSystemStructure`. It assembles with `AssembleLHS`/`AssembleRHS` behind a mutex, because the Epetra assembly is not thread-safe. Contributions to rows of other ranks are sent by the Epetra global assembly. After solving, it imports the values of all the DOFs the local elements need with an `Epetra_Import`.

## ParticleMeshPdeProjectionProcess

Constructors:

```cpp
// Serial and OpenMP (wraps a SerialFacetLinearSystem)
ParticleMeshPdeProjectionProcess(ModelPart&, LinearSolverType::Pointer, Parameters);
ParticleMeshPdeProjectionProcess(Model&, LinearSolverType::Pointer, Parameters);   // "model_part_name"
// Any facet system, e.g. TrilinosFacetLinearSystem for MPI
ParticleMeshPdeProjectionProcess(ModelPart&, FacetLinearSystem::Pointer, Parameters);
```

* `ExecuteInitialize`: builds the facet table, the particles and the projection utility, and initializes $$c_h$$ and $$\phi_h$$. It is called automatically by the first `Execute`.
* `Execute`: one transport step. `ExecuteFinalizeSolutionStep`: distance update after redistancing.
* Accessors: `GetIndicatorIntegral()`, `GetNumberOfParticles()`, `GetLastVolumeCorrectionShift()` and `GetLastProjectionInfo()`. The first two are **collective in MPI** (call them on all the ranks).
* `UpdateParticlesModelPart(model_part, stride)` copies the particles to a model part for output. The model part is cleared and filled with one node and one `PointCondition2D1N`/`PointCondition3D1N` per particle, with `DISTANCE` and `PHASE_FRACTION` as non-historical values. Node Ids are globally unique: each rank gets a contiguous range from `ScanSum`.
  * With `particles_output.model_part_name`, the process creates that model part in its constructor, before the output processes are built, and makes it share the fluid `ProcessInfo`. It refreshes the model part in the first step and every `interval` steps.
  * In MPI, the Python helper first gives the model part an MPI communicator, so that the VTK output names the files by rank.

Nodal averages are accumulated in the non-historical variable `PARTICLE_MESH_NODAL_SUMS` (registered by the application). It is assembled across ranks with `Communicator::AssembleNonHistoricalData`, which does nothing in serial runs.

The process requires linear simplices, the historical variables `DISTANCE` and `VELOCITY`, and a buffer size of at least 2. It checks that the mesh does not change between steps; remeshing is not supported.

## Python

* **Bindings.**
  * `KratosMultiphysics.FluidDynamicsApplication.ParticleMeshPdeProjectionProcess2D/3D(model_part or model, linear_solver, parameters)`, with `GetLastProjectionInfo()` returning a dict.
  * `KratosMultiphysics.FluidDynamicsApplication.TrilinosExtension.TrilinosParticleMeshPdeProjectionProcess2D/3D(epetra_communicator, model_part, trilinos_linear_solver, parameters)`. These are factory functions returning the same process type, because the process type is already registered by the main module.
* **Helper** `python_scripts/particle_pde_projection_utilities.py`:
  * `GetDefaultLinearSolverSettings()`;
  * `GetLevelSetTransportType(settings, solver_name)` validates the setting;
  * `CreateParticlePdeProjectionProcess(model_part, domain_size, settings, epetra_communicator=None)` builds the serial or the MPI version, depending on `model_part.IsDistributed()`. The facet solver comes from `linear_solver_settings`, or from the defaults.
* **Solvers.** `navier_stokes_two_fluids_solver.py`, `trilinos_navier_stokes_two_fluids_solver.py` and `navier_stokes_two_fluid_hydraulic_solver.py`:
  * add the `levelset_transport_type` and `particle_pde_projection_settings` keys;
  * return the particle process from `_CreateLevelSetConvectionProcess`;
  * call its `ExecuteFinalizeSolutionStep` at the end of `FinalizeSolutionStep`;
  * switch off `mass_source`.

  The FM-ALE velocity correction of the hydraulic solver is compatible with the particle transport; note that with `eulerian_fm_ale: true` the Eulerian convection does not move the interface (see the design decisions).

## Tests

| Test | Suite | What it checks |
|---|---|---|
| `tests/cpp_tests/test_particle_mesh_pde_projection.cpp` (12 tests) | C++ fast suite | Facet table consistency (2D/3D), exact negative volumes, symmetry of the Schur complement, exact local conservation for arbitrary facet values, reproduction of a linear field (2D/3D), conservation with inflow/outflow (2D with both outflow options, 3D), RK3 particle rotation, closed vortex with the process, particle output model part. |
| `tests/test_particle_mesh_pde_projection.py` | FluidDynamics `small` | Slotted disk in a closed vortex (conservation, range, volume correction), particle VTK output, settings validation, `two_fluids` sloshing tank with constant water volume. |
| `FluidDynamicsHydraulicsApplication/tests/two_fluid_hydraulic_particle_transport_test.py` | Hydraulics `small` | 2D and 3D hydraulic sloshing tank with constant water volume. |
| `tests/test_particle_mesh_pde_projection_mpi.py` | FluidDynamics `mpi_small` | Distributed versus serial closed vortex (conservation, particle counts, nodal `DISTANCE`), particle output in MPI (globally unique Ids, one VTK file per rank), Trilinos `two_fluids` sloshing tank. |

Running them:

```bash
bin/Release/test/KratosFluidDynamicsCoreTest --gtest_filter='*ParticleMeshPde*'
python3 applications/FluidDynamicsApplication/tests/test_particle_mesh_pde_projection.py
mpirun -np 3 python3 applications/FluidDynamicsApplication/tests/test_FluidDynamicsApplication_mpi.py --using-mpi -l mpi_small
```
