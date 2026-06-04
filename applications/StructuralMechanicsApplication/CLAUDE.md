# CLAUDE.md — StructuralMechanicsApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> structural-mechanics specifics are documented here. Read the root file first for global
> C++/Python/pybind11/testing conventions.

## Purpose

The **StructuralMechanicsApplication** provides the finite-element toolbox for solid and
structural mechanics in Kratos: solid elements (small/large strain), structural elements
(trusses, beams, cables, shells, membranes), Neumann/contact conditions, the solving
strategies and schemes that drive static, dynamic, eigenvalue, harmonic, formfinding,
prebuckling and adjoint analyses, plus the supporting utilities and response functions.

The actual material models (plasticity, damage, hyperelasticity, composites, …) live in
the sibling **ConstitutiveLawsApplication**; this application contains only a small set of
"elastic" base laws plus the constitutive-law *infrastructure* (`constitutive_law_utilities.cpp`).

## Dependencies

- **KratosCore** (always).
- **ConstitutiveLawsApplication** — strongly recommended companion; most material models live there.
- **LinearSolversApplication** — for eigenvalue / dense solvers used by eigen, harmonic and prebuckling analyses.
- **TrilinosApplication + MetisApplication** — only for the MPI (`trilinos_*`) solvers and convergence-criteria factories.
- Optional: **MeshingApplication** — adaptive remeshing solvers (`adaptative_remeshing_*`).

## Directory layout (application-specific)

```
custom_elements/
  solid_elements/      # small-displacement, total/updated Lagrangian, mixed (Bbar, U-eps, Q1P0, U-dV/V), SPrism solid-shell
  truss_elements/      # truss + cable (3D)
  beam_elements/       # corotational beam (2D/3D)
  membrane_elements/   # prestressed membrane, formfinding
  shell_elements/      # thin/thick quad & triangle shells
  nodal_elements/      # nodal concentrated mass/stiffness/damping, spring-damper
custom_conditions/     # point/line/surface loads, point moment, distributed, contact
custom_constitutive/   # elastic base laws + constitutive-law plumbing (flat dir, no subdirs)
custom_constraints/    # MPC-style structural constraints (e.g. link constraint)
custom_processes/      # structural processes (see python_scripts wrappers too)
custom_strategies/     # strategies, schemes, convergence criteria, builder-and-solvers
custom_response_functions/  # adjoint sensitivity response functions (mass, displacement, stress, …)
custom_io/             # specialized I/O
custom_utilities/      # constitutive_law_utilities.cpp (compiled FIRST — see CMake note), structural math helpers
custom_python/         # pybind11 bindings (one add_custom_*_to_python.cpp per category)
automatic_differentiation/  # sympy scripts that generate element/constitutive code
python_scripts/        # analysis stages, solvers, processes (see below)
benchmarks/            # Google Benchmark files (built with KRATOS_BUILD_BENCHMARK=ON)
tests/
  cpp_tests/           # GTest sources + structural_mechanics_fast_suite.{h,cpp}
  test_StructuralMechanicsApplication.py  # Python suite entry point
```

## Build

- CMake target libs: `KratosStructuralMechanicsCore` (SHARED C++) and
  `KratosStructuralMechanicsApplication` (pybind11 module).
- Sources are auto-globbed via `file(GLOB_RECURSE ...)` over `custom_*` — **new `.cpp`
  files are picked up automatically**, no CMake edit needed.
- **Gotcha:** `custom_utilities/constitutive_law_utilities.cpp` is explicitly removed and
  re-inserted at position 0 of the source list so it compiles first. Do not reorder this.
- Compile definition: `STRUCTURAL_MECHANICS_APPLICATION=EXPORT,API`.
- C++ tests built when `KRATOS_BUILD_TESTING=ON` via `kratos_add_gtests(TARGET KratosStructuralMechanicsCore ...)`.

## Python entry points

- **Analysis stage:** `structural_mechanics_analysis.py` → `StructuralMechanicsAnalysis`
  (inherit from this for custom drivers). Main script: `kratos_main_structural.py`.
- **Solver wrapper / factory:** `python_solvers_wrapper_structural.py` selects the solver
  from `solver_settings["solver_type"]`. Solver base: `structural_mechanics_solver.py`
  (`MechanicalSolver`).
- **Solver families** (each a `*_solver.py`):
  - `static`, `implicit_dynamic`, `explicit_dynamic`
  - `eigensolver` (modal), `harmonic_analysis_solver`, `prebuckling` (+ `prebuckling_analysis`)
  - `formfinding_solver`, `custom_scipy_base_solver`, `static_shifted_boundary_solver`
  - `adjoint_static_solver` (sensitivities, pairs with `custom_response_functions/`)
  - `trilinos_*` variants for MPI
  - `adaptative_remeshing_*` (needs MeshingApplication)
- Use `Parameters` (JSON `ProjectParameters.json` + `StructuralMaterials.json`) — never hardcode.

## Testing

- **C++ fixture:** `KratosStructuralMechanicsFastSuite` (in `structural_mechanics_fast_suite.h`).
  Some core-only tests use `KratosCoreFastSuite`.
- **Python:** `test_StructuralMechanicsApplication.py` assembles `smallSuite`, `nightSuite`,
  `validationSuite`, `allSuite`. **Adding a new `test_*.py` requires registering it in this runner.**
- Run most-specific first: a single GTest filter or single Python test, then the small suite.

## Conventions & gotchas

- Elements/Conditions use `KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION`; Processes/Utilities/
  Strategies use `KRATOS_CLASS_POINTER_DEFINITION`.
- Override the four required element/condition virtuals (`CalculateLocalSystem`,
  `EquationIdVector`, `GetDofList`, `Check`).
- New global `Variable`s: declare with `KRATOS_DEFINE_APPLICATION_VARIABLE` in
  `structural_mechanics_application_variables.h`, register with `KRATOS_REGISTER_VARIABLE`
  in `structural_mechanics_application.cpp`, and register new elements/conditions/laws in
  the same `.cpp` (`KRATOS_REGISTER_ELEMENT` / `_CONDITION` / `_CONSTITUTIVE_LAW`).
- Several elements/constitutive laws are **symbolically generated** — edit the generator in
  `automatic_differentiation/` (or `symbolic_generation`) and regenerate, do not hand-patch
  the generated body if a generator exists.
- Local axes for orthotropy are set via the `set_*_local_axes_process.py` processes.
- DOFs: `DISPLACEMENT`(+`ROTATION` for beams/shells), `VELOCITY`/`ACCELERATION` for dynamics.
