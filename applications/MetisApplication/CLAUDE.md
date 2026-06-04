# CLAUDE.md — MetisApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> Metis/partitioning specifics are documented here. Read the root file first for global conventions.

## Purpose

The **MetisApplication** is a thin interface to the
[METIS](https://github.com/KarypisLab/METIS) graph-partitioning library. Within Kratos it
**partitions meshes for MPI runs**: it reads the model, builds the nodal/elemental graph,
calls METIS, and assigns each entity to an MPI partition. It is almost always used together
with the **TrilinosApplication** (which provides the distributed solvers that consume the
partition).

The application is organized as a set of **partitioning processes**, each wrapping a different
algorithm. The most commonly used for FE meshes is
`metis_divide_heterogeneous_input_process` (handles meshes mixing multiple element types).

## Partitioning processes (`custom_processes/`)

- `metis_divide_heterogeneous_input_process.h` — **the default**; partitions a heterogeneous
  mesh from an input (e.g. `.mdpa`) based on the nodal graph.
- `metis_divide_heterogeneous_input_in_memory_process.h` — same, but works on an already-read
  in-memory model (no re-read from disk).
- `metis_divide_submodelparts_heterogeneous_input_process.h` — partition respecting
  sub-model-part structure.
- `morton_divide_input_to_partitions_process.h` / `morton_partitioning_process.h` —
  Morton-order (space-filling-curve) partitioning alternative.

## Dependencies

- **KratosCore** + **KratosMPICore**.
- **METIS** (system library) and its dependency **GKlib**. On Debian/Ubuntu:
  `sudo apt install libmetis-dev`. Point CMake at a custom build with `METIS_ROOT_DIR`.
- Built **only** under MPI (`USE_MPI=ON`); pairs with **TrilinosApplication**.

## Directory layout (application-specific)

```
custom_processes/   # the partitioning processes listed above
custom_utilities/   # graph-construction / partition helpers around the METIS API
custom_python/      # bindings: add_processes_to_python.cpp, kratos_metis_python_application.cpp
test_exemples/      # example input meshes for partitioning
tests/              # test_MetisApplication.py + cpp_tests (metis_fast_suite.{h,cpp})
```

> Note: this application has **no `python_scripts/`** of its own — the processes are exposed
> directly through the C++ bindings and used from other applications' MPI launch scripts.

## Build

- CMake target: `KratosMetisApplication` (pybind11), linked against METIS.
- Requires `USE_MPI=ON` and METIS to be found. Provide `METIS_ROOT_DIR` (or have
  `libmetis-dev` installed) so CMake can locate headers + libs.
- Available via Spack as well (see README) for HPC environments.
- In the **nightly** MPI CI matrix, not the default PR Linux build.

## Usage

Typical MPI preprocessing flow (from an application's MPI main script):

```python
import KratosMultiphysics as KM
import KratosMultiphysics.MetisApplication as KratosMetis
import KratosMultiphysics.mpi as KratosMPI

# read serial-or-distributed input, then partition
partitioner = KratosMetis.MetisDivideHeterogeneousInputProcess(io, number_of_partitions, ...)
partitioner.Execute()
```

In current Kratos the partitioning is usually invoked through the **distributed import**
machinery (`DistributedImportModelPartUtility` in the MPI core), which selects a Metis
partitioning process under the hood. Prefer that path over calling the process manually.

## Testing

- **C++ fixture:** `KratosMetisFastSuite` (`metis_fast_suite.h`); generic tests use
  `KratosCoreFastSuite`. These are MPI GTests.
- **Python:** `test_MetisApplication.py` is the suite entry point — register new `test_*.py`
  there. Run under `mpiexec` with `OMP_NUM_THREADS=1` to match CI.

## Conventions & gotchas

- This app **produces partitions, it does not solve** — the distributed solve is done by
  TrilinosApplication. An MPI run needs both.
- The graph is built from the mesh connectivity; for heterogeneous (mixed-element) meshes use
  the `heterogeneous` process, not a homogeneous-only variant.
- METIS is an **external dependency** — do not vendor or modify it; respect the
  `METIS_ROOT_DIR` discovery path.
- Keep partition results deterministic where CI relies on it (METIS seeding / options).
