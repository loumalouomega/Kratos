# CLAUDE.md — MeshingApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> meshing specifics are documented here. Read the root file first for global conventions.
> The user (Vicente Mataix Ferrándiz) is the main author of the MMG integration.

## Purpose

The **MeshingApplication** provides tools to create, manipulate, remesh and interpolate
between meshes. It wraps several third-party meshers/remeshers (**Triangle**, **TetGen**,
**MMG/ParMMG**) and provides metric-computation processes (level-set, Hessian, error-based)
that drive anisotropic adaptive remeshing, plus utilities to transfer/interpolate nodal and
internal (Gauss-point) variables between old and new meshes.

## Dependencies

- **KratosCore** (always).
- **StructuralMechanicsApplication** — included at configure time (its source dir is on the
  include path); internal-variable interpolation reuses structural constitutive infrastructure.
- **Triangle** — vendored in `external_libraries/triangle` (core), always available.
- **TetGen** — optional, non-free TPL (`USE_TETGEN_NONFREE_TPL=ON`).
- **MMG** (2D/3D/S) — optional serial anisotropic remesher (`INCLUDE_MMG=ON`, needs `MMG_ROOT`).
- **ParMMG** — optional MPI remesher (`INCLUDE_PMMG=ON`, requires `USE_MPI=ON` **and** `INCLUDE_MMG=ON`).

## Directory layout (application-specific)

```
custom_processes/
  metrics_levelset_process.h   # level-set based metric
  metrics_hessian_process.h    # Hessian (solution-based) metric
  metrics_error_process.h      # error-estimation metric
  metric_fast_init_process.h, set_h_map_process.h
  nodal_values_interpolation_process.h / internal_variables_interpolation_process.h
  multiscale_refining_process.h, embedded_mesh_locator_process.h
  mmg/      # MmgProcess (2D/3D/Surface remeshing) — built when INCLUDE_MMG=ON
  parmmg/   # ParMmgProcess (distributed remeshing) — built when INCLUDE_PMMG=ON
custom_utilities/   # mesh transfer (MeshTransfer, BinBasedMeshTransfer), meshers, mmg/ & parmmg/ utilities
custom_includes/    # shared headers / mesher data structures
custom_io/          # PFEMGidIO and mmg/ IO; MMG file readers/writers
custom_external_libraries/tetMeshOpt/  # vendored tet mesh optimizer (built as static lib)
external_includes/  # mesher wrappers (Triangle/TetGen interface headers + .cpp)
custom_python/      # bindings: add_meshers/add_processes/add_custom_io/add_custom_utilities
python_scripts/
  mmg_process.py                       # Python wrapper around the C++ MmgProcess
  multiscale_refining_process.py
  gradual_variable_interpolation_process.py
  modelers/                            # Modeler-based meshing helpers
tests/
  cpp_tests/   # GTest + meshing_fast_suite.{h,cpp}
  test_MeshingApplication.py
```

## Build

- CMake libs: `KratosMeshingCore` (SHARED) and `KratosMeshingApplication` (pybind11).
- Compile definition: `MESHING_APPLICATION=EXPORT,API`.
- `tetMeshOpt` is always built as an extra static lib and linked into `KratosMeshingCore`.
- Conditional sources:
  - `INCLUDE_MMG=ON` adds `custom_utilities/mmg/*`, `custom_processes/mmg/*`,
    `custom_io/mmg/*`, defines `-DINCLUDE_MMG`, and `find_package`s MMG/MMG2D/MMG3D/MMGS.
    Set `MMG_ROOT` to the MMG build dir if CMake can't auto-locate it.
  - `INCLUDE_PMMG=ON` adds `custom_utilities/parmmg/*` and `custom_processes/parmmg/*`,
    defines `-DINCLUDE_PMMG`; **fails configure** if `USE_MPI=OFF` or MMG is not enabled.
- **Unity-build gotcha:** `mmg/mmg_utilities.cpp` and `parmmg/pmmg_utilities.cpp` are excluded
  from unity builds (MMG headers are not unity-compatible). Keep that exclusion.
- C++ tests use `USE_CUSTOM_MAIN`.
- `python_registry_lists.py` is installed alongside the package `__init__.py`.

## Python usage

- **MMG remeshing:** `mmg_process.py` → `MmgProcess` wraps the C++ `MmgProcess2D/3D/Surfaces`.
  Typically combined with a metric process (`ComputeLevelSetSolMetricProcess`,
  `ComputeHessianSolMetricProcess`, `MetricErrorProcess`) that fills `METRIC_TENSOR_*`/`NODAL_H`,
  then the remesher acts via `Execute()`.
- **Multiscale refinement:** `multiscale_refining_process.py`.
- Configure via `Parameters` (JSON); see `mmg_process.py` defaults and `normal_distribution.json`.

## Testing

- **C++ fixture:** `KratosMeshingFastSuite` (`meshing_fast_suite.h`); core-only tests use
  `KratosCoreFastSuite`.
- **Python:** `test_MeshingApplication.py` — register new `test_*.py` there. MMG/ParMMG tests
  are skipped automatically when those libraries are not compiled in.

## Conventions & gotchas

- MMG/ParMMG code is guarded by `INCLUDE_MMG`/`INCLUDE_PMMG` macros — wrap new MMG-dependent
  code accordingly so non-MMG builds keep compiling.
- Interpolation of internal (Gauss-point) variables goes through
  `internal_variables_interpolation_process` — pick the correct interpolation method
  (CPT / LST / SF) for the physics.
- Remeshing rebuilds the `ModelPart`: re-create elements/conditions and re-assign properties;
  historical and non-historical nodal data are interpolated, not preserved by identity.
- New variables → `meshing_application_variables.{h,cpp}`.
