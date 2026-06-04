# CLAUDE.md — FluidDynamicsApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> fluid-dynamics specifics are documented here. Read the root file first for global
> C++/Python/pybind11/testing conventions.

## Purpose

The **FluidDynamicsApplication** holds the core Computational Fluid Dynamics (CFD)
developments of Kratos: stabilized FEM solvers for **incompressible**,
**weakly-compressible** and **compressible** flow, on linear elements, in 2D/3D
(plus limited 2D axisymmetric). It supports ALE mesh motion (with MeshMovingApplication)
and MPI parallelism (with Metis + Trilinos).

Key formulations: **VMS** (quasi-static & dynamic subscales), **Orthogonal SubScales
(OSS)**, **FIC**; Newtonian and non-Newtonian (Bingham, Herschel-Bulkley) constitutive
models; embedded/two-fluid (level-set) formulations; explicit compressible Navier-Stokes
in conservative variables with shock capturing.

## Dependencies

- **KratosCore** (always).
- **LinearSolversApplication** — practical default for the algebraic solvers.
- **MeshMovingApplication** — ALE / moving-mesh fluid solvers.
- **TrilinosApplication + MetisApplication** — MPI (`trilinos_*` solvers, `trilinos_extension/`).
- Optional companions: **FSIApplication / CoSimulationApplication** (coupled problems),
  **HDF5Application** (I/O), **MappingApplication**.

## Directory layout (application-specific)

```
custom_elements/        # ~35 element .cpp: VMS, QSVMS, DVMS, FIC, fractional-step,
  data_containers/      #   weakly-compressible, two-fluid, embedded, low-Mach, compressible NS
custom_conditions/      # inlet/outlet/wall/slip/Neumann conditions, monolithic wall conditions
custom_constitutive/    # fluid constitutive laws (Newtonian 2D/3D, Bingham, Herschel-Bulkley, …)
custom_processes/       # embedded, distance, mass conservation, drag, slip, wall-law processes
custom_strategies/      # fractional-step & monolithic strategies, schemes, B&S
custom_response_functions/  # adjoint fluid response functions (drag, …)
custom_utilities/       # compiled FIRST (used inside element cpps): fluid math, statistics, drag utils
custom_python/          # pybind11 bindings
automatic_differentiation/  # sympy generators for the symbolic element variants
trilinos_extension/     # MPI bindings/strategies (built only with USE_MPI + TRILINOS_FOUND)
python_scripts/         # analysis stages, solvers, processes (see below)
tests/
  cpp_tests/            # GTest + fluid_dynamics_fast_suite.{h,cpp}
  test_FluidDynamicsApplication.py
```

## Build

- CMake libs: `KratosFluidDynamicsCore` (SHARED) and `KratosFluidDynamicsApplication` (pybind11).
- **Source ordering matters:** `custom_utilities/*.cpp` is globbed *before* `custom_elements/*.cpp`
  in `CMakeLists.txt` because utilities are used inside the element translation units. Keep this order.
- Compile definition: `FLUID_DYNAMICS_APPLICATION=EXPORT,API`.
- `automatic_differentiation/` is installed into the Python package; `python_registry_lists.py`
  is installed alongside `__init__.py`.
- The `trilinos_extension` subdirectory is added only when `USE_MPI=ON AND TRILINOS_FOUND`.

## Python entry points

- **Analysis stage:** `fluid_dynamics_analysis.py` → `FluidDynamicsAnalysis`
  (RVE variant: `fluid_dynamics_analysis_rve.py`; adjoint: `adjoint_fluid_analysis.py`).
- **Solver wrapper:** `python_solvers_wrapper_fluid.py` (and `python_solvers_wrapper_adjoint_fluid.py`)
  dispatch on `solver_type`. Solver base: `fluid_solver.py` (`FluidSolver`).
- **Solver families:**
  - `navier_stokes_solver_vmsmonolithic.py` (monolithic VMS — the workhorse)
  - `navier_stokes_solver_fractionalstep.py` (segregated fractional step, VMS only)
  - `navier_stokes_two_fluids_solver.py` (level-set two-fluid)
  - `navier_stokes_embedded_solver.py` (embedded/immersed boundary)
  - `navier_stokes_compressible_explicit_solver.py`, `navier_stokes_low_mach_solver.py`
  - `navier_stokes_ale_fluid_solver.py` (moving mesh), `navier_stokes_monolithic_iga_solver.py`
  - `stokes_solver_monolithic.py`, `adjoint_monolithic_solver.py`
  - `trilinos_*` variants for MPI
- Boundary conditions / post-processing are applied via the many `apply_*_process.py` and
  `compute_*_process.py` modules (inlet, outlet, slip, no-slip, wall law, drag, CFL, y+, …).

## Testing

- **C++ fixture:** tests primarily use `KratosCoreFastSuite`; the application fixture is
  `fluid_dynamics_fast_suite.{h,cpp}`.
- **Python:** `test_FluidDynamicsApplication.py` builds `small`/`night`/`validation`/`all`
  suites — register any new `test_*.py` there.

## Conventions & gotchas

- Elements rely on `ProcessInfo` for stabilization constants, `OSS_SWITCH`, `DYNAMIC_TAU`,
  time-integration parameters — make sure these are set by the scheme/solver, not the element.
- Many elements are **symbolically generated** from `automatic_differentiation/`; edit the
  generator and regenerate rather than hand-editing the generated `Calculate*` bodies.
- DOFs: `VELOCITY` + `PRESSURE` (monolithic) or staggered (fractional step). Two-fluid/embedded
  use `DISTANCE`. Compressible uses conservative variables (`DENSITY`, `MOMENTUM`, `TOTAL_ENERGY`).
- New variables → `fluid_dynamics_application_variables.h` (define) +
  `fluid_dynamics_application.cpp` (register), where elements/conditions are also registered.
- Keep MPI tests consistent with CI (`OMP_NUM_THREADS=1`).
