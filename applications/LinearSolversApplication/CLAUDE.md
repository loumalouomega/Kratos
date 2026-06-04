# CLAUDE.md — LinearSolversApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> linear-solver specifics are documented here. Read the root file first for global conventions.

## Purpose

The **LinearSolversApplication** is a thin Kratos wrapper around the **Eigen** linear algebra
library (and the **Spectra** eigensolver library), exposing direct/iterative sparse solvers,
dense solvers, decompositions and eigenvalue solvers to Kratos. It is the de-facto default
provider of robust direct solvers for most applications, so it is a near-universal dependency.

## Provided solvers (Python `solver_type`)

**Sparse direct/iterative** (`custom_solvers/eigen_sparse_*`): `sparse_lu`, `sparse_qr`,
`sparse_cg`, complex variants (`sparse_lu_complex`), and — with Intel MKL — `pardiso_llt`,
`pardiso_ldlt`, `pardiso_lu` (+ complex). With SuiteSparse: `cholmod`, `umfpack`, `spqr`
(+ complex). (In project parameters these are usually prefixed `eigen_`, e.g. `eigen_sparse_lu`.)

**Dense direct** (`custom_solvers/eigen_dense_*`): `dense_col_piv_householder_qr`,
`dense_householder_qr`, `dense_llt`, `dense_partial_piv_lu` (+ complex variants).

**Eigenvalue** (`eigensystem_solver`, `spectra_sym_g_eigs_shift_solver`,
`feast_eigensystem_solver`, `eigen_dense_eigenvalue_solver`): generalized symmetric eigenvalue
problems (modal analysis), with FEAST available only when built against MKL.

## Dependencies

- **KratosCore** (always).
- **Eigen** — vendored in `external_libraries/eigen3` (compiled with `EIGEN_MPL2_ONLY`).
- **Spectra** — vendored in `external_libraries/spectra1`.
- Optional **Intel MKL** (`USE_EIGEN_MKL=ON`) — Pardiso solvers + FEAST.
- Optional **SuiteSparse** (`USE_EIGEN_SUITESPARSE=ON`) — CHOLMOD/UMFPACK/SPQR.
- Optional **FEAST4** (`USE_EIGEN_FEAST=ON`, Linux + MKL only).

## Directory layout (application-specific)

```
custom_solvers/        # eigen_sparse_*, eigen_dense_*, eigensystem_solver, feast_*, spectra_*
custom_decompositions/ # dense matrix decompositions exposed to Python
custom_factories/      # dense_linear_solver_factory.cpp (registered dense solver factory)
custom_utilities/      # MKL helpers (mkl_utilities.cpp, only with USE_EIGEN_MKL)
custom_python/         # bindings: add_custom_solvers/decompositions/utilities_to_python
external_libraries/    # eigen3, spectra1 (and FEAST when enabled) — DO NOT modify
python_scripts/        # dense_linear_solver_factory.py
tests/                 # Python tests only (no GTest fast-suite in this app)
```

## Build

- CMake libs: `KratosLinearSolversCore` (SHARED) and `KratosLinearSolversApplication` (pybind11).
- Built differently from most apps: sources are added with `target_sources` (not a single
  GLOB of `custom_*`). The core registers solver factories; most solvers are header-only
  (`custom_solvers/*.h`) and instantiated through the factory.
- Compile definitions: `LINEARSOLVERS_APPLICATION=EXPORT,API`, plus `EIGEN_MPL2_ONLY` (PUBLIC).
- **MKL** (`USE_EIGEN_MKL`): needs `MKLROOT` (or a Conda env); adds
  `mkl_smoother_base.cpp`, `mkl_ilu.cpp`, `mkl_utilities.cpp` and defines
  `USE_EIGEN_MKL`/`EIGEN_USE_MKL_ALL`.
- **SuiteSparse** (`USE_EIGEN_SUITESPARSE`): runs the bundled `FindCHOLMOD/SPQR/UMFPACK`
  cmake modules and defines `KRATOS_USE_EIGEN_SUITESPARSE`.
- **FEAST** requires MKL and is Linux-only.

## Python usage

```json
{ "solver_type": "eigen_sparse_lu" }
```

- Standard `linear_solver_settings` blocks accept these `solver_type`s; other applications
  list this app as a dependency to get them.
- `dense_linear_solver_factory.py` builds dense solvers for small/local systems.

## Testing

- **Python only**: `test_LinearSolversApplication.py` is the suite entry point, aggregating
  `test_eigen_direct_solver`, `test_eigen_dense_*`, `test_eigensystem_solver`,
  `test_feast_eigensystem_solver`, `test_mkl`, etc. Register new `test_*.py` there.
- MKL/SuiteSparse/FEAST tests skip themselves when the corresponding backend isn't compiled in.
- There is **no C++ GTest fast-suite** in this application.

## Conventions & gotchas

- **Never modify `external_libraries/`** (Eigen, Spectra, FEAST) — they are vendored.
- Solvers are exposed by **registering them in the factory**; adding a header in
  `custom_solvers/` is not enough — wire it into the (dense or sparse) factory and the
  pybind bindings.
- Complex-domain solvers are separate types (`*_complex`); keep real/complex paths distinct.
- Respect the `EIGEN_MPL2_ONLY` constraint — do not pull in non-MPL2 Eigen modules.
