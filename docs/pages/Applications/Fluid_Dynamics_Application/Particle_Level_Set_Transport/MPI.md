---
title: MPI parallelization
keywords: [particle level set MPI Trilinos particle migration partition facets]
tags: [MPI.md]
sidebar: fluid_dynamics_application
summary: Distributed facet numbering, particle migration between ranks and the distributed facet system.
---

## Starting point

A Kratos MPI model part has the following layout:

* every element belongs to exactly one rank, and there are no ghost elements;
* only the nodes on the partition interfaces are duplicated, and `PARTITION_INDEX` names their owner;
* the `Communicator` lists the neighbour ranks (`NeighbourIndices`, one per colour) and the nodes shared with each of them (`InterfaceMesh(colour)`).

The serial algorithm has three parts that couple elements and therefore cross the partitions:

1. **The facet unknowns.** A face shared by elements of two ranks (a *partition facet*) carries facet unknowns used by both ranks.
2. **The particles.** A particle may leave the local partition during a time step.
3. **The global quantities.** Integrals, volumes, the volume-correction shift, nodal averages and diagnostics.

Everything else (local systems, population control, $$\ell^2$$ projection) is element-local and works unchanged.

The FluidDynamics core code only uses the abstract `DataCommunicator` (`SendRecv`, `SumAll`, `MinAll`, `MaxAll`, `ScanSum`) and the `Communicator`. In serial runs these calls do nothing, so the same code path runs in serial, OpenMP and MPI. Only the distributed linear algebra (`TrilinosFacetLinearSystem`) lives in the Trilinos extension.

## Partition facets and global numbering

`MeshFacetTable` builds the facet numbering as follows:

1. **Local matching.** Faces shared by two local elements are interior facets, as in serial.
2. **Partition candidates.** For each neighbour rank, the locally unmatched faces whose nodes **all** belong to the interface with that neighbour are candidates. Their keys (the sorted node Ids) are exchanged with `SendRecv`. A key present on both sides is a partition facet. A domain boundary face can never match: it belongs to a single element in the whole mesh.
3. **Ownership.** The matched facets are sorted by key, so both ranks list them in the same order. The owner of a partition facet is the lower rank.
4. **Numbering.** Each rank numbers its owned facets contiguously: the local interior facets first, then the owned partition facets. The offset of each rank is `ScanSum(n_owned) - n_owned`, so the owned facet DOFs of a rank form one contiguous global range.
5. **Exchange.** For each partition facet, the owner sends the global index and its area normal. The non-owner uses the negated normal, so the fluxes of the two sides cancel bitwise, exactly as for interior facets.

The facet DOFs keep the serial definition (`global_facet_id * TDim + rank of the vertex Id within the facet`), so they are consistent across ranks without further communication. Each element face stores the rank behind it (`PartitionRanks`). A map from the global index of each partition facet to its local element and face is used to receive particles.

## Distributed facet system

`TrilinosFacetLinearSystem`:

* **Initialization.** Builds an `Epetra_Map` with the owned contiguous range, and the `Epetra_FECrsGraph` and matrix with `TrilinosSpace::BuildSystemStructure`, from the element DOF blocks. A block may reference DOFs owned by another rank.
* **Assembly.** `AssembleLHS`/`AssembleRHS` (`SumIntoGlobalValues`) are serialized with a mutex, because Epetra assembly is not thread-safe. The rows of other ranks are communicated in `GlobalAssemble`.
* **Solution.** Any Trilinos linear solver can be used. The default is AMGCL conjugate gradients with a relaxation preconditioner and an integer `block_size: 1`; there is no DOF set to deduce it from. The relative residual is computed with Epetra.
* **Ghost values.** An `Epetra_Import` brings the values of all the DOFs referenced by local elements, owned or not, to each rank. The import is collective, so ranks that need no values still take part.

The system keeps its own copy of the Epetra communicator, which is reference counted, so it does not depend on the lifetime of the communicator passed from Python.

## Particle migration

`LagrangianParticleContainer::Advect` works in two phases.

1. **Local integration (OpenMP over particles).** Each particle integrates its substeps. If the end point of a substep is not in the local mesh, the neighbour walk stops at the exit face. A bins search confirms that the point is not elsewhere in the local partition, since partitions are often non-convex. If the exit face is a partition facet, the particle is removed locally and queued for the rank behind it, together with the state of its trajectory:
   * position, indicator and distance;
   * the next substep and the number of substeps;
   * the global index of the crossed partition facet.

   The state is packed as 8 doubles.
2. **Exchange rounds.** In each round:
   * every rank exchanges a buffer of doubles with every neighbour with `SendRecv`. Empty buffers are exchanged too, so all ranks take part.
   * The receiver finds the element behind the facet and locates the particle from it. It then integrates the remaining substeps.
   * A particle that reaches yet another partition is queued for the next round.

   The rounds repeat until `SumAll(particles in flight) == 0`. There are at most 50 rounds; any particle still in flight after that is deleted, with a warning.

Runge-Kutta **stage points** that fall beyond the local partition use the velocity of the last local element, with the shape functions clipped to the element. This is first-order accurate near the partition interfaces and is the only difference with the serial algorithm. Only the end point of each substep decides the migration.

Seeding and reseeding use random streams keyed by the element **Id**, so the particles do not depend on the partitioning. On the test cases the distributed and serial particle counts differ by a few particles out of hundreds of thousands, because of the stage-point treatment above.

## Global quantities

* `ComputeIntegral`, `ComputeNegativeVolume`, the projection diagnostics, the minimum element size, the particle counts and the total domain volume are reduced with `SumAll`, `MinAll` and `MaxAll`.
* **Volume correction.** It is an iterative root search, so **every rank must take the same decisions**. All the quantities used in the iteration (residuals, tolerance, domain volume, bracketing step) are global. During development, a domain volume summed only over the local elements made the ranks run different numbers of iterations and deadlock.
* **Nodal averages.** Each rank accumulates $$(\sum \vert K\vert\,v, \sum \vert K\vert)$$ of its elements in `PARTICLE_MESH_NODAL_SUMS`. The variable is assembled across the ranks sharing a node (`AssembleNonHistoricalData`), and the average is identical in every copy of an interface node.
* **Logging.** Only rank 0 writes to the log.

## Using it

The MPI two-fluid solver (`parallel_type: "MPI"`, `solver_type: "two_fluids"`) creates the distributed process when `levelset_transport_type` is `particle_pde_projection`. It passes its Epetra communicator, and the facet solver is built with the Trilinos linear solver factory. Directly from Python:

```python
import KratosMultiphysics.TrilinosApplication as KratosTrilinos
from KratosMultiphysics.FluidDynamicsApplication import particle_pde_projection_utilities

epetra_comm = KratosTrilinos.CreateEpetraCommunicator(model_part.GetCommunicator().GetDataCommunicator())
process = particle_pde_projection_utilities.CreateParticlePdeProjectionProcess(model_part, 2, settings, epetra_comm)
```

`GetNumberOfParticles()` and `GetIndicatorIntegral()` are collective: call them on every rank, also when only rank 0 prints the result.

## Testing without METIS

The MPI tests do not depend on METIS, which is not always available. Instead, `tests/particle_mesh_mpi_test_utilities.py` provides `PartitionInStripes`, a **test-only** partitioner:

* every rank holds the whole mesh;
* each element goes to the stripe of its centroid along one axis;
* each node is owned by the lowest stripe among its elements, so all the ranks agree;
* conditions follow an element that contains them;
* the entities of other stripes are removed, `PARTITION_INDEX` is set and `ParallelFillCommunicator` creates the communicator.

The two-fluid MPI test reads the `.mdpa` in every rank, partitions it this way and passes `"input_type": "use_input_model_part"` to the distributed importer. Real simulations should use the regular METIS partitioning.

## Results

The 2D slotted disk in a closed vortex was run on a 128×128 mesh (32,768 triangles, 820k particles) for 40 steps:

| Run | Time per transport step | Particles | Relative volume drift | max cell residual | Facet solve residual |
|---|---|---|---|---|---|
| 1 rank, 1 thread | 1.29 s | 823,062 | 9.3e-15 | 5.6e-16 | 9.4e-9 |
| 1 rank, 4 threads (OpenMP) | 0.39 s | 823,062 | 2.5e-15 | 5.6e-16 | 9.9e-9 |
| 4 ranks, 1 thread each (MPI) | 0.43 s | 823,066 | 1.6e-15 | 5.1e-16 | 9.7e-9 |

In all three runs the indicator range is [-0.004, 1.006] and the volume correction matches $$\int c_h$$ to 2.4e-12. The MPI tests (`mpi_small`) pass on 2, 3 and 4 ranks.
