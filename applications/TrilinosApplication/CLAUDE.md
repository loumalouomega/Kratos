# CLAUDE.md — TrilinosApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> Trilinos/MPI specifics are documented here. Read the root file first for global conventions.

## Purpose

The **TrilinosApplication** is the backbone of Kratos's **distributed-memory (MPI)**
capabilities. It wraps the [Trilinos](https://trilinos.org/) project to provide
**distributed matrices/vectors** (Epetra) and **parallel linear solvers** (AztecOO, Amesos,
Amesos2, ML), plus the **MPI versions of the core Kratos solving machinery** (builder-and-
solvers, strategies, schemes, convergence criteria) adapted to Epetra. It also exposes an MPI
build of **AMGCL**.

In practice, other applications' `trilinos_*` solvers (Fluid, Structural, Mapping's MPI
extension, …) are built on top of the classes registered here. This application **only builds
under MPI** (`USE_MPI=ON` and Trilinos found).

## Provided components (C++ → Python)

- **Spaces:** `TrilinosSpace` (Epetra-backed `Space` abstraction), experimental space.
- **Linear solvers:** `TrilinosLinearSolver`, `AztecSolver`, `AmesosSolver`, `Amesos2Solver`,
  `MultiLevelSolver` (ML), and the MPI AMGCL solver.
- **Builder-and-solvers:** `TrilinosResidualBasedBuilderAndSolver`,
  `TrilinosBlockBuilderAndSolver(Periodic)`, `TrilinosEliminationBuilderAndSolver`.
- **Strategies:** `TrilinosLinearStrategy`, `TrilinosNewtonRaphsonStrategy`, `TrilinosSolvingStrategy`.
- **Schemes** and **convergence criteria:** MPI counterparts of the core
  (`TrilinosResidualCriteria`, `TrilinosDisplacementCriteria`, `TrilinosAnd/OrCriteria`,
  `TrilinosMixedGenericCriteria`).
- These mirror the serial classes (without the `Trilinos` prefix) — consult the serial docs
  for behavioral semantics.

## Dependencies

- **KratosCore** + **KratosMPICore** (the MPI part of the core).
- **Trilinos** (system install) — Epetra, Teuchos, AztecOO, Amesos, Amesos2, ML.
- **MetisApplication** — almost always used together to partition the mesh for MPI runs.
- Built **only** when `USE_MPI=ON` and `TRILINOS_FOUND`.

## Directory layout (application-specific)

```
custom_factories/    # trilinos linear-solver factory (registers the solvers above)
custom_strategies/   # MPI strategies, schemes, builder-and-solvers, convergence criteria
custom_processes/    # MPI-aware processes
custom_utilities/    # Epetra/Trilinos helpers, parallel fill-comm utilities
external_includes/   # thin wrappers over Aztec/Amesos/ML headers
custom_python/       # bindings, split per category:
  add_trilinos_space_to_python, add_trilinos_linear_solvers_to_python,
  add_trilinos_strategies_to_python, add_trilinos_schemes_to_python,
  add_trilinos_convergence_criterias_to_python, add_trilinos_convergence_accelerators_to_python,
  add_trilinos_processes_to_python, add_custom_utilities_to_python
  trilinos_pointer_wrapper.h   # shared/raw pointer interop helper for Trilinos objects
python_scripts/      # trilinos_linear_solver_factory.py, MonolithicMultiLevelSolver.py,
                     # PressureMultiLevelSolver.py, gid_output_process_mpi.py
tests/               # test_TrilinosApplication_mpi.py + test_trilinos_* (matrix, solvers, redistance, …)
  cpp_tests/         # GTest (MPI) + trilinos_fast_suite.{h,cpp}
```

## Build

- CMake libs: `KratosTrilinosCore` (SHARED) and `KratosTrilinosApplication` (pybind11),
  linked against `KratosMPICore` and the Trilinos libraries.
- Requires `USE_MPI=ON`; CMake must locate Trilinos (`TRILINOS_ROOT` / standard find).
- This application is in the **nightly** MPI CI matrix, not the default PR Linux build.

## Python usage

- `trilinos_linear_solver_factory.py` constructs distributed solvers from
  `linear_solver_settings` (`solver_type`: `amesos`, `aztec`, `multi_level`, `amgcl`, …).
- Other applications select their `trilinos_*` solver wrappers (e.g.
  `trilinos_navier_stokes_solver_vmsmonolithic.py`), which internally use these classes.
- Typical MPI run: partition with MetisApplication → solve with a `Trilinos*Strategy` +
  `TrilinosBlockBuilderAndSolver` + a Trilinos linear solver.

## Testing

- **C++ fixtures:** `KratosMPICoreFastSuite` / `KratosTrilinosApplicationMPITestSuite`
  (`trilinos_fast_suite.h`) — these are **MPI GTest suites** (run under `mpiexec`).
- **Python:** `test_TrilinosApplication_mpi.py` is the MPI suite entry point (there is no
  serial `test_TrilinosApplication.py`). Register new MPI tests there.
- Run MPI tests with `OMP_NUM_THREADS=1` to match CI, under `mpiexec -np <N>`.

## Conventions & gotchas

- All matrix/vector work goes through `TrilinosSpace` (Epetra) — never assume the serial
  `UblasSpace` types in MPI code paths.
- Keep `Trilinos*` classes behaviorally in sync with their serial counterparts; this is a
  parallel re-implementation layer, not a place for new physics.
- Use `trilinos_pointer_wrapper.h` for shared/raw pointer interop with Trilinos objects.
- **Do not modify Trilinos itself** — it is an external system dependency.
- Pair with MetisApplication for partitioning; an MPI run without a partition is ill-defined.
