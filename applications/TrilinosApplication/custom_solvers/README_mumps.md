# TrilinosMumpsSolver — direct MUMPS integration

`TrilinosMumpsSolver` calls the MUMPS sparse direct solver via its native C API,
bypassing the Amesos/Amesos2 Trilinos packages.  The matrix entries are
extracted from the distributed `Epetra_FECrsMatrix` in COO format using MUMPS
distributed assembled input (`ICNTL(18) = 3`), so no global gather of the
matrix is required.

---

## Build instructions

1. Install MUMPS (and its dependencies: BLAS, ScaLAPACK, and optionally
   METIS/Scotch for reordering).  Set the `MUMPS_ROOT` environment variable or
   CMake variable to the MUMPS installation prefix.

2. Configure Kratos with the extra flag:

   ```bash
   cmake -DTRILINOS_APPLICATION_USE_MUMPS_DIRECTLY=ON \
         -DMUMPS_ROOT=/path/to/mumps/install \
         ...
   ```

3. If your MUMPS was compiled with a Fortran compiler, you may also need:

   ```bash
   cmake -DTRILINOS_APPLICATION_LINK_GFORTRAN=ON ...
   ```

The build will fail loudly if `dmumps_c.h` or the MUMPS libraries cannot be
found.  When the option is `OFF` (the default), no MUMPS symbols are referenced
and the build is unaffected.

---

## Python usage

```python
import KratosMultiphysics
import KratosMultiphysics.TrilinosApplication as KratosTrilinos
from KratosMultiphysics.TrilinosApplication import trilinos_linear_solver_factory

solver_settings = KratosMultiphysics.Parameters("""{
    "solver_type"                : "mumps_direct",
    "sym"                        : 0,
    "ordering"                   : 0,
    "iterative_refinement_steps" : 0,
    "out_of_core"                : 0,
    "verbosity"                  : 0
}""")

# Via the factory (recommended)
solver = trilinos_linear_solver_factory.ConstructSolver(solver_settings)
solver.Solve(A, x, b)

# Or directly
solver = KratosTrilinos.TrilinosMumpsSolver(solver_settings)
solver.Solve(A, x, b)
```

### Parameters

| Key | MUMPS control | Default | Notes |
|-----|---------------|---------|-------|
| `sym` | `id.sym` | `0` | 0 = unsymmetric, 1 = symmetric, 2 = SPD |
| `ordering` | `ICNTL(7)` | `0` | 0 = auto, 1 = Scotch, 2 = METIS, … |
| `iterative_refinement_steps` | `ICNTL(10)` | `0` | 0 = disabled |
| `out_of_core` | `ICNTL(22)` | `0` | 1 = out-of-core factorization |
| `verbosity` | `ICNTL(1-4)` | `0` | 0 = silent, 1+ = increasingly verbose |

---

## Sequential vs MPI

- **Sequential** (single rank): works identically to the MPI case; only rank 0
  exists, so no gather/scatter is needed.
- **MPI** (multiple ranks): each rank contributes its locally owned rows via
  `nz_loc`/`irn_loc`/`jcn_loc`/`a_loc`.  The right-hand side is gathered to
  rank 0 before the solve and the solution is scattered back to all ranks.

---

## Known limitations

- **Windows toolchain**: mixing MSVC and MinGW Fortran for MUMPS may require
  careful linkage.  Use a consistent compiler suite.
- **Sequential stub**: when building MUMPS without MPI, link against `libmpiseq`
  (provided in the MUMPS distribution) instead of a real MPI library.  The
  `FindMUMPS.cmake` in `cmake_modules/` searches for `libmpiseq` automatically.
- **64-bit MUMPS**: this wrapper uses the default 32-bit integer interface
  (`MUMPS_INT = int`).  For problems with more than ~2 billion non-zeros, a
  MUMPS build with 64-bit integers is required and this code would need
  adaptation.
- The symbolic factorization (`ICNTL(18) = 3`) is reused across consecutive
  `Solve` calls for the same matrix sparsity pattern.  Set `mReanalyze = true`
  (currently internal) if the sparsity changes between solves.
