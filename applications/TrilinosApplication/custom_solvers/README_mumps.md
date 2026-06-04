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

All MUMPS behaviour is driven from the `Parameters` object passed at construction.

**Core**

| Key | MUMPS control | Default | Notes |
|-----|---------------|---------|-------|
| `sym` | `id.sym` | `0` | 0 = unsymmetric, 1 = symmetric, 2 = SPD |
| `ordering` | `ICNTL(7)` | `0` | 0 = auto, 1 = Scotch, 2 = METIS, … |
| `iterative_refinement_steps` | `ICNTL(10)` | `0` | 0 = disabled |
| `out_of_core` | `ICNTL(22)` | `0` | 1 = out-of-core factorization |
| `verbosity` | `ICNTL(1-4)` | `0` | 0 = silent, 1+ = increasingly verbose |

**Numerical robustness**

| Key | MUMPS control | Default | Notes |
|-----|---------------|---------|-------|
| `scaling` | `ICNTL(8)` | `77` | 77 = automatic, 0 = none, 7/8 = iterative; helps ill-conditioned systems |
| `pivoting_threshold` | `CNTL(1)` | `-1.0` | relative pivoting threshold; `< 0` leaves MUMPS automatic default |
| `null_pivot_detection` | `ICNTL(24)` | `0` | 1 = detect null pivots (singular systems); reports `GetNumNullPivots()` |
| `null_pivot_threshold` | `CNTL(3)` | `0.0` | threshold for null-pivot detection; 0 = MUMPS automatic |

**Memory controls**

| Key | MUMPS control | Default | Notes |
|-----|---------------|---------|-------|
| `memory_relaxation_percent` | `ICNTL(14)` | `-1` | extra workspace (%); `< 0` leaves MUMPS default. Raise if factorization runs out of memory |
| `max_working_memory_mb` | `ICNTL(23)` | `0` | max working memory per MPI process (MB); 0 = MUMPS decides |

**Parallel ordering / analysis**

| Key | MUMPS control | Default | Notes |
|-----|---------------|---------|-------|
| `analysis_type` | `ICNTL(28)` | `0` | 0 = auto, 1 = sequential, 2 = parallel analysis |
| `parallel_ordering` | `ICNTL(29)` | `0` | 0 = auto, 1 = PT-Scotch, 2 = ParMETIS (used when `analysis_type` = 2) |

**Block Low-Rank (approximate / fast factorization)**

| Key | MUMPS control | Default | Notes |
|-----|---------------|---------|-------|
| `block_low_rank` | `ICNTL(35)` | `0` | 0 = off (exact), 1/2/3 = BLR factorization variants |
| `blr_variant` | `ICNTL(36)` | `0` | BLR variant selector |
| `blr_compression_threshold` | `CNTL(7)` | `0.0` | BLR dropping/compression accuracy; 0 = MUMPS default |

**Diagnostics**

| Key | MUMPS control | Default | Notes |
|-----|---------------|---------|-------|
| `error_analysis` | `ICNTL(11)` | `0` | 0 = none, 1 = full statistics + condition number, 2 = backward error only |
| `compute_determinant` | `ICNTL(33)` | `0` | 1 = compute the determinant; read with `GetDeterminant()` |

**Escape hatch (power users)**

Any control not named above can be set directly. Both maps use the **1-based**
MUMPS index as the key and are applied *last*, so they override the named
parameters:

| Key | Type | Notes |
|-----|------|-------|
| `additional_icntl` | object `{ "index": int }` | sets `ICNTL(index)` |
| `additional_cntl` | object `{ "index": double }` | sets `CNTL(index)` |

```python
solver_settings = KratosMultiphysics.Parameters("""{
    "solver_type"      : "mumps_direct",
    "additional_icntl" : { "14": 35 },
    "additional_cntl"  : { "1": 0.001 }
}""")
```

---

## Reading diagnostics

After a `Solve`, the following getters return values gathered from the MUMPS
`INFOG`/`RINFOG` arrays. Each is only meaningful when the corresponding feature
was enabled in the parameters.

| Getter | Requires | Returns |
|--------|----------|---------|
| `GetDeterminant()` | `compute_determinant = 1` | determinant = `RINFOG(12) * 2^INFOG(34)` |
| `GetEstimatedConditionNumber()` | `error_analysis = 1` | estimated condition number `RINFOG(11)` |
| `GetBackwardError()` | `error_analysis >= 1` | scaled backward error `max(RINFOG(7), RINFOG(8))` |
| `GetNumNullPivots()` | `null_pivot_detection = 1` | number of null pivots `INFOG(28)` |
| `GetInfog(i)` | — | raw `INFOG(i)`, 1-based (`i` in `[1, 80]`) |
| `GetRinfog(i)` | — | raw `RINFOG(i)`, 1-based (`i` in `[1, 40]`) |

```python
solver_settings = KratosMultiphysics.Parameters("""{
    "solver_type"         : "mumps_direct",
    "sym"                 : 1,
    "compute_determinant" : 1,
    "error_analysis"      : 1
}""")
solver = KratosTrilinos.TrilinosMumpsSolver(solver_settings)
solver.Solve(A, x, b)
print("det  =", solver.GetDeterminant())
print("cond =", solver.GetEstimatedConditionNumber())
```

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
