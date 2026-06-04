# CLAUDE.md — CoSimulationApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> co-simulation specifics are documented here. Read the root file first for global conventions.

## Purpose

The **CoSimulationApplication** couples black-box solvers and external software tools
(Kratos↔Kratos, Kratos↔external) into partitioned multi-physics simulations (FSI, thermal,
etc.). It is **predominantly a Python application**: the orchestration (coupled solvers,
convergence accelerators, predictors, data-transfer operators) lives in `python_scripts/`,
while the C++ core mostly provides the `CoSimIO` communication layer and a few utilities.

Kratos↔Kratos coupling has no data-duplication overhead (same database). Mapping between
non-matching grids is delegated to the **MappingApplication**.

## Dependencies

- **KratosCore** (always).
- **MappingApplication** — for mapping between non-matching interfaces.
- **CoSimIO** — vendored in `custom_external_libraries/` (the detached communication library,
  also usable from non-Kratos codes).
- Domain solvers as needed: **FluidDynamics**, **StructuralMechanics**, etc. (Kratos wrappers),
  or any external solver via its wrapper.
- Optional **MPI** (`mpi_extension/`) — independent of whether the coupled solvers run in MPI.

## Directory layout (application-specific, mostly Python)

```
python_scripts/
  co_simulation_analysis.py        # CoSimulationAnalysis — top-level driver (AnalysisStage)
  MainKratosCoSim.py               # main script entry point
  base_classes/                    # CoSimulationSolverWrapper, CoSimulationCoupledSolver, …
  solver_wrappers/                 # one subpackage per coupled solver/tool:
    kratos/        #   Kratos solver wrappers (fluid, structural, …)
    external/      #   generic external-solver wrapper (via CoSimIO)
    sdof/          #   single-DOF analytical solvers
    rigid_body/    #   rigid-body solver
    cpp_ping_pong/ #   C++ example/test wrapper
  coupled_solvers/                 # coupling algorithms:
    gauss_seidel_strong/weak.py, jacobi_strong/weak.py, block_strong.py,
    feti_dynamic_coupled_solver.py
  convergence_accelerators/        # Aitken, MVQN, IQN-ILS, constant relaxation, …
  convergence_criteria/            # residual / relative criteria for the coupling loop
  predictors/                      # interface predictors (linear, quadratic, …)
  data_transfer_operators/         # kratos_mapping (uses MappingApplication), copy, …
  coupling_operations/             # scaling, reaction computation, etc.
  coupling_interface_data.py       # CouplingInterfaceData — the data exchanged across the interface
  factories/, helpers/, processes/, utilities/
custom_io/        # C++ CoSimIO interface wrappers
custom_processes/ # C++ helper processes
custom_utilities/ # C++ helpers
custom_python/    # bindings incl. add_co_sim_io_to_python.cpp
custom_external_libraries/  # vendored CoSimIO — DO NOT modify
mpi_extension/    # MPI build additions
tests/
  cpp_tests/      # GTest + co_simulation_fast_suite.{h,cpp}
  test_CoSimulationApplication.py
```

## Build

- CMake libs: `KratosCoSimulationCore` (SHARED) and `KratosCoSimulationApplication` (pybind11).
- Compile definition: `CO_SIMULATION_APPLICATION=EXPORT,API`.
- `CoSimIO` is built from `custom_external_libraries/` — treat as vendored (don't modify).
- `mpi_extension/` added under MPI builds.

## Configuration & usage

Co-simulations are driven entirely by a **JSON** config describing solvers, the coupled
solver (algorithm), coupling interfaces, data-transfer operators, convergence accelerator and
criteria. Run via `MainKratosCoSim.py` with that JSON, or instantiate `CoSimulationAnalysis`.

Key extension interface — to couple a **new solver/tool**, implement a `SolverWrapper`
(subclass of `CoSimulationSolverWrapper` in `base_classes/`) exposing at least:
`Initialize`, `AdvanceInTime`, `InitializeSolutionStep`, `Predict`, `SolveSolutionStep`,
`FinalizeSolutionStep`, `Finalize`, plus `ImportCouplingInterface*`/`Export*` data hooks.
External (non-Kratos) tools connect through the **CoSimIO** library / the `external` wrapper.

## Testing

- **C++ fixture:** `KratosCoSimulationFastSuite` (`co_simulation_fast_suite.h`); generic tests
  use `KratosCoreFastSuite`.
- **Python:** `test_CoSimulationApplication.py` is the suite entry point — register new
  `test_*.py` there. Many tests use the `sdof`/`rigid_body` analytical wrappers and the
  `cpp_ping_pong` example to exercise the coupling loop without heavy solvers.

## Conventions & gotchas

- Logic lives in **Python**; keep the C++ side thin (CoSimIO + utilities). New coupling
  features (accelerators, predictors, operators) are Python subclasses registered via the
  corresponding `factories/`.
- Strong vs. weak coupling = with/without inner convergence loop (`*_strong` vs `*_weak`
  coupled solvers); choose the matching convergence accelerator/criteria.
- Data exchange is via `CouplingInterfaceData` (a named variable on a ModelPart interface);
  map mismatched meshes with the `kratos_mapping` data-transfer operator (MappingApplication).
- MPI of the coupling is independent of MPI inside each solver — don't conflate them.
