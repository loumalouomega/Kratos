# CLAUDE.md — MappingApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> mapping specifics are documented here. Read the root file first for global conventions.

## Purpose

The **MappingApplication** transfers nodal data between **non-matching grids**. It works in
serial, shared memory (OpenMP) and distributed memory (**MPI**), across 1D/2D/3D domains and
matching or non-matching meshes. It is the mapping backend used by the
CoSimulationApplication (via `KratosMappingDataTransferOperator`).

A `Mapper` maps nodal data from an **Origin** `ModelPart` to a **Destination** `ModelPart`,
configured with `Kratos::Parameters`. Mappers are always built through the **`MapperFactory`**
(serial) or `MPIMapperFactory` (distributed) — never constructed directly.

## Available mappers

Built from `custom_mappers/`:

- `nearest_neighbor_mapper` — nearest neighbor (+ IGA variant `nearest_neighbor_mapper_iga`).
- `nearest_element_mapper` — projects onto the nearest element/condition (shape-function based).
- `barycentric_mapper` — barycentric interpolation.
- `radial_basis_function_mapper` — RBF mapping.
- `beam_mapper` — structural beam mapping (translations + rotations).
- `coupling_geometry_mapper` — mortar-style mapping on a coupling geometry.
- `projection_3D_2D_mapper` — metamapper deriving a 3D destination solution from a 2D origin.
- `interpolative_mapper_base` — shared base for the interpolation-type mappers.

## Dependencies

- **KratosCore** — serial / shared-memory build has no other dependency.
- **TrilinosApplication** (+ MPI) — required for the distributed build (`mpi_extension/`),
  which uses Trilinos distributed matrices/vectors. Most MPI solvers in Kratos depend on it.

## Directory layout (application-specific)

```
custom_mappers/      # the Mapper implementations listed above
custom_searching/    # interface object search (bins/octree based neighbor location)
custom_utilities/    # mapping matrix utilities, interface communicators, vector containers
custom_modelers/     # Modelers that build mapping/coupling geometries
custom_python/       # pybind11 bindings (add_custom_mappers_*, add_custom_utilities_*)
mpi_extension/       # MPI/Trilinos mapper backend (built only with USE_MPI + TRILINOS_FOUND)
python_scripts/      # python_mapper_factory.py, python_mapper.py, helpers
tests/
  cpp_tests/         # GTest + mapping_fast_suite.{h,cpp}
  test_MappingApplication.py
```

## Build

- CMake libs: `KratosMappingCore` (SHARED) and `KratosMappingApplication` (pybind11).
- Core sources use plain `file(GLOB ...)` (non-recursive) over `custom_mappers`,
  `custom_searching`, `custom_utilities`, `custom_modelers`.
- Compile definition: `MAPPING_APPLICATION=EXPORT,API`.
- **Unity-build gotcha:** `custom_utilities/mapping_matrix_utilities.cpp` and
  `interface_vector_container.cpp` are explicitly excluded from unity builds — keep that
  exclusion when touching these files.
- C++ tests use `USE_CUSTOM_MAIN` in `kratos_add_gtests`.
- `mpi_extension` subdir added only when `USE_MPI=ON AND TRILINOS_FOUND`.

## Python usage

```python
from KratosMultiphysics.MappingApplication import python_mapper_factory
mapper = python_mapper_factory.CreateMapper(origin_model_part, destination_model_part, mapper_settings)
mapper.Map(ORIGIN_VARIABLE, DESTINATION_VARIABLE)        # origin -> destination
mapper.InverseMap(DESTINATION_VARIABLE, ORIGIN_VARIABLE) # destination -> origin
```

- `mapper_settings["mapper_type"]` selects the mapper (`"nearest_neighbor"`,
  `"nearest_element"`, `"barycentric"`, `"coupling_geometry"`, …).
- Behavior is customized with **flags** (e.g. `ADD_VALUES`, `SWAP_SIGN`, `USE_TRANSPOSE`,
  `TO_NON_HISTORICAL`/`FROM_NON_HISTORICAL`) passed to `Map`/`InverseMap`.
- For MPI, use the MPI factory; ModelParts may live on a subset of ranks.

## Testing

- **C++ fixture:** `KratosMappingFastSuite` (`mapping_fast_suite.h`); core-only tests use
  `KratosCoreFastSuite`.
- **Python:** `test_MappingApplication.py` is the suite entry point — register new
  `test_*.py` there. MPI tests run separately.

## Conventions & gotchas

- Always go through the factory; the mapper builds an interface search + a mapping matrix.
- The mapping matrix is reusable: rebuild it (`UpdateInterface`) when geometry changes,
  otherwise reuse it for repeated `Map` calls.
- New variables → `mapping_application_variables.{h,cpp}`.
- Keep serial and MPI mapper behavior in sync; the MPI variant lives in `mpi_extension/`.
