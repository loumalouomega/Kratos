# CLAUDE.md — Kratos Multiphysics

## Project Overview

**Kratos Multiphysics** is an open-source C++20/Python framework for building parallel, multi-disciplinary finite-element simulation software. The core (`kratos/`) provides containers, solvers, geometries, I/O, and Python bindings via **pybind11**. The `applications/` directory holds 40+ domain-specific extensions (structural, fluid, contact, DEM, etc.) that add Elements, Conditions, ConstitutiveLaws, Processes, and Strategies without touching the core.

---

## Repository Layout

```
Kratos/
├── kratos/                  # Core framework
│   ├── includes/            # Public headers: checks.h, expect.h, define.h, …
│   ├── sources/             # Core implementation
│   ├── containers/          # ModelPart, Model, Node, Element, Variable, …
│   ├── elements/            # Base element types
│   ├── conditions/          # Base condition types
│   ├── processes/           # Base Process and core processes
│   ├── utilities/           # Math, geometry, parallel utilities
│   ├── linear_solvers/      # Solver abstractions
│   ├── solving_strategies/  # Strategy/scheme/builder-solver hierarchy
│   ├── geometries/          # Geometry implementations
│   ├── integration/         # Quadrature rules
│   ├── spatial_containers/  # Octree, bins, kd-tree
│   ├── input_output/        # GiD, HDF5, vtk I/O
│   ├── python/              # pybind11 bindings for core
│   ├── python_scripts/      # Core Python modules
│   ├── testing/             # testing.h, GTest fixture setup
│   ├── tests/               # Core C++ and Python tests
│   ├── mpi/                 # MPI-aware core components
│   └── benchmarks/          # Core Google Benchmark files
├── applications/            # 40+ domain-specific applications
│   └── <ApplicationName>/   # e.g. StructuralMechanicsApplication
│       ├── <app>_application.h/.cpp
│       ├── <app>_application_variables.h/.cpp
│       ├── CMakeLists.txt
│       ├── custom_elements/
│       ├── custom_conditions/
│       ├── custom_constitutive/
│       ├── custom_processes/
│       ├── custom_utilities/
│       ├── custom_strategies/
│       ├── custom_python/
│       │   ├── <app>_python_application.cpp  # PYBIND11_MODULE entry
│       │   └── add_custom_*_to_python.cpp    # Per-category bindings
│       ├── python_scripts/
│       └── tests/
│           ├── test_<ApplicationName>.py     # Python suite entry point
│           └── cpp_tests/
│               ├── <app>_fast_suite.h/.cpp   # GTest fixture
│               └── test_*.cpp
├── external_libraries/      # Vendored third-party deps (amgcl, Boost headers, …)
├── scripts/                 # Configure script templates
│   ├── standard_configure.sh
│   ├── standard_configure.bat
│   └── ...
├── cmake_modules/           # Custom CMake Find/utility modules
├── CMakeLists.txt           # Root CMake entry
├── INSTALL.md               # Build instructions
└── CONTRIBUTING.md          # Contribution guidelines
```

---

## Build System

### Configure Scripts

Never invoke CMake directly. Use the wrapper scripts:

| Platform | Template | Personalized copy |
|----------|----------|------------------|
| Linux | `scripts/standard_configure.sh` | `build/configure.sh` |
| Windows | `scripts/standard_configure.bat` | `build/configure.bat` |

Copy the template, customize compilers, Python path, and `KRATOS_APPLICATIONS`, then use that copy. The VS Code `Build` task calls `build/configure.*`.

### Key CMake Variables

| Variable | Description |
|----------|-------------|
| `KRATOS_BUILD_TYPE` | `Release` \| `RelWithDebInfo` \| `FullDebug` \| `Custom` |
| `KRATOS_APPLICATIONS` | Semicolon-separated list of application paths |
| `KRATOS_BUILD_TESTING` | `ON` to build C++ GTest binaries |
| `KRATOS_BUILD_BENCHMARK` | `ON` to build Google Benchmark binaries |
| `USE_MPI` | `ON` for MPI-parallel builds |
| `USE_EIGEN_MKL` | `ON` to link Eigen against Intel MKL |
| `PYTHON_EXECUTABLE` | Path to Python 3 interpreter |

### Runtime Environment

After a successful build, before running Python scripts:

```bash
export PYTHONPATH=/path/to/Kratos/bin/Release   # or RelWithDebInfo, FullDebug
export LD_LIBRARY_PATH=/path/to/Kratos/bin/Release/libs:$LD_LIBRARY_PATH
```

### VS Code Tasks

Prefer `.vscode/tasks.json` tasks over ad-hoc commands:

| Task | What it does |
|------|-------------|
| `Build` | configure + compile |
| `MPI Build` | build with `USE_MPI=ON` |
| `Run Tests` | full Python test suite |
| `Run CurrentFile` | run active Python file with correct env |
| `Run C++ Tests` | all C++ GTest suites |
| `Run C++ Test Suite` | specific GTest executable |
| `Run C++ Test Suite Filtered` | `--gtest_filter=*<pattern>*` |
| `Run Current Benchmark file to JSON` | benchmark → JSON output |

---

## C++ Conventions

### Naming

| Symbol | Style | Example |
|--------|-------|---------|
| Classes / types | `PascalCase` | `TotalLagrangianElement` |
| Methods | `PascalCase` | `CalculateLocalSystem`, `GetDofList` |
| File names | `snake_case` | `total_lagrangian_element.h` |
| Member variables | `m` prefix | `mThickness`, `mConstitutiveLaw` |
| Reference parameters | `r` prefix | `rModelPart`, `rCurrentProcessInfo` |
| Pointer parameters | `p` prefix | `pElement`, `pNode` |
| Local variables / free functions | `snake_case` | `local_stiffness` |
| Kratos Variables (global) | `UPPER_SNAKE_CASE` | `DISPLACEMENT`, `TEMPERATURE` |

### Error Handling

Always wrap method bodies with `KRATOS_TRY`/`KRATOS_CATCH("")`. Never use `std::cout` in production code.

```cpp
void MyProcess::Execute() {
    KRATOS_TRY

    KRATOS_ERROR_IF(mrModelPart.NumberOfNodes() == 0)
        << "ModelPart has no nodes." << std::endl;

    KRATOS_INFO("MyProcess") << "Starting execution." << std::endl;

    // implementation

    KRATOS_CATCH("")
}
```

| Macro | Location | Purpose |
|-------|----------|---------|
| `KRATOS_TRY` / `KRATOS_CATCH("")` | method body | exception context |
| `KRATOS_ERROR_IF(cond)` | `includes/exception.h` | throw on condition |
| `KRATOS_ERROR_IF_NOT(cond)` | `includes/exception.h` | throw if false |
| `KRATOS_ERROR` | `includes/exception.h` | unconditional throw |
| `KRATOS_INFO("tag")` | `includes/logger.h` | info logging |
| `KRATOS_WARNING("tag")` | `includes/logger.h` | warning logging |

### Memory Management

- Never use raw `new` / `delete` — use `Kratos::shared_ptr`, `Kratos::unique_ptr`.
- Use `KRATOS_CLASS_POINTER_DEFINITION(ClassName)` for Process/Utility/Strategy classes.
- Use `KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(ClassName)` for Element/Condition classes.
- Prefer Kratos type aliases: `IndexType`, `SizeType`, `MatrixType`, `VectorType`.
- Never `using namespace std;` at global scope.

### Header Layout

```cpp
#pragma once   // always use this — no macro include guards

// System includes
// External includes
// Project includes
```

### Elements / Conditions

Always override the four required virtual methods:

```cpp
void CalculateLocalSystem(MatrixType& rLHS, VectorType& rRHS,
                           const ProcessInfo& rCurrentProcessInfo) override;
void EquationIdVector(EquationIdVectorType& rResult,
                      const ProcessInfo& rCurrentProcessInfo) const override;
void GetDofList(DofsVectorType& rDofList,
                const ProcessInfo& rCurrentProcessInfo) const override;
int Check(const ProcessInfo& rCurrentProcessInfo) const override;
```

### Processes

```cpp
class MyProcess : public Process {
public:
    KRATOS_CLASS_POINTER_DEFINITION(MyProcess);

    MyProcess(ModelPart& rModelPart, Parameters rParameters);

    void ExecuteInitialize() override;
    void ExecuteBeforeSolutionLoop() override;
    void ExecuteInitializeSolutionStep() override;
    void Execute() override;
    void ExecuteFinalizeSolutionStep() override;
    void ExecuteFinalize() override;

private:
    ModelPart& mrModelPart;
    Parameters mParameters;
};
```

Use `Parameters` (JSON-backed) for configuration — prefer data-driven design over hardcoded values.

---

## Python Conventions

- Follow PEP 8; `snake_case` for all identifiers.
- Import pattern:

```python
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanics
```

- Analysis stages and processes inherit from appropriate Kratos base classes.
- All test classes inherit from `KratosMultiphysics.KratosUnittest.TestCase`.

---

## pybind11 Bindings

Bindings live in `custom_python/` (application) or `kratos/python/` (core).

- Keep binding files thin — logic belongs in C++.
- File naming: `add_custom_<category>_to_python.cpp`.
- Class exposure:

```cpp
py::class_<MyProcess, MyProcess::Pointer, Process>(m, "MyProcess")
    .def(py::init<ModelPart&, Parameters>(), py::arg("model_part"), py::arg("parameters"))
    .def("Execute", &MyProcess::Execute)
    .def("ExecuteInitialize", &MyProcess::ExecuteInitialize);
```

- Provide docstrings for all newly introduced public bindings.
- Set `py::return_value_policy` explicitly when returning references or pointers.

---

## Testing

### C++ Tests

Framework: **GTest wrapped by Kratos macros** (`testing/testing.h`).

```cpp
#include "testing/testing.h"
#include "structural_mechanics_fast_suite.h"
#include "containers/model.h"

namespace Kratos::Testing {

KRATOS_TEST_CASE_IN_SUITE(MyFeature_ShouldDoX, KratosStructuralMechanicsFastSuite)
{
    Model model;
    auto& r_mp = model.CreateModelPart("Test", 1);

    // Arrange / Act
    KRATOS_EXPECT_NEAR(result, expected, 1e-10);
    KRATOS_EXPECT_VECTOR_NEAR(v1, v2, 1e-10);
    KRATOS_EXPECT_MATRIX_NEAR(m1, m2, 1e-10);
    KRATOS_EXPECT_EQ(a, b);
}

} // namespace Kratos::Testing
```

Use `KRATOS_EXPECT_*` (from `includes/expect.h`) in tests. Use `KRATOS_CHECK_*` (from `includes/checks.h`) in production-code preconditions.

Test files: `tests/cpp_tests/test_<feature_name>.cpp`.  
Fast suite fixture: `tests/cpp_tests/<app>_fast_suite.h/.cpp`.

### Python Tests

Framework: `KratosMultiphysics.KratosUnittest` (wraps `unittest`).

```python
import KratosMultiphysics
import KratosMultiphysics.KratosUnittest as KratosUnittest

class TestMyFeature(KratosUnittest.TestCase):

    def setUp(self):
        self.model = KratosMultiphysics.Model()
        self.model_part = self.model.CreateModelPart("TestPart")

    def test_something(self):
        self.assertAlmostEqual(expected, actual, places=6)
        self.assertVectorAlmostEqual(expected_vec, actual_vec)

if __name__ == "__main__":
    KratosUnittest.main()
```

Helpers on `KratosUnittest.TestCase`:
- `skipTestIfApplicationsNotAvailable(*apps)`
- `assertVectorAlmostEqual(v1, v2, places=7)`
- `assertMatrixAlmostEqual(m1, m2, places=7)`

Suite entry point per application: `tests/test_<ApplicationName>.py` — assembles `small`, `nightly`, `validation`, `all` (and `mpi_*`) suites.

### Benchmarks

Framework: **Google Benchmark**. Files: `benchmarks/benchmark_<feature_name>.cpp`. Built when `KRATOS_BUILD_BENCHMARK=ON`. Always end with `BENCHMARK_MAIN()`.

---

## CMake Application Pattern

```cmake
file(GLOB_RECURSE MY_APP_SOURCES
    custom_elements/*.cpp custom_conditions/*.cpp
    custom_processes/*.cpp custom_utilities/*.cpp
    my_app_application.cpp my_app_application_variables.cpp
)

add_library(KratosMyAppCore SHARED ${MY_APP_SOURCES})
target_link_libraries(KratosMyAppCore PUBLIC KratosCore)

pybind11_add_module(KratosMyAppApplication MODULE THIN_LTO
    custom_python/my_app_python_application.cpp
    custom_python/add_custom_processes_to_python.cpp
)
target_link_libraries(KratosMyAppApplication PRIVATE KratosMyAppCore)

if(KRATOS_BUILD_TESTING)
    kratos_add_gtests(TARGET KratosMyAppCore
        SOURCES tests/cpp_tests/test_my_feature.cpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

kratos_python_install(${INSTALL_PYTHON_USING_LINKS}
    "${CMAKE_CURRENT_SOURCE_DIR}/python_scripts/"
    KratosMultiphysics/MyAppApplication)

install(TARGETS KratosMyAppCore KratosMyAppApplication DESTINATION libs)
```

Use `kratos_add_gtests` and `kratos_add_benchmark` (from `cmake_modules/`) rather than bare `add_executable`.

---

## CI/CD

GitHub Actions exclusively — `.github/workflows/`:

| Workflow | Trigger | What it runs |
|----------|---------|-------------|
| `ci.yml` | PR to `master`, `workflow_dispatch` | Ubuntu matrix: gcc+clang × Custom+FullDebug |
| `nightly_build.yml` | Scheduled nightly | Broader: Windows, Rocky Linux, wider app set |

Changed-file detection (`get_files_changed_in_pr.py`) skips builds for unaffected applications.

To add an application to CI, add it to the appropriate JSON file in `.github/workflows/`:
- `ci_apps_linux.json` — Ubuntu PR builds
- `ci_apps_windows.json`, `ci_apps_rocky.json`, `ci_apps_intel.json` — nightly builds

---

## Key Kratos Concepts

| Concept | Description |
|---------|-------------|
| `Model` | Top-level container owning all `ModelPart` instances |
| `ModelPart` | Container for nodes, elements, conditions, sub-model-parts |
| `Node` | Geometric point with DOFs and historical/non-historical data |
| `Element` | FE entity — owns nodes and implements `CalculateLocalSystem` |
| `Condition` | Boundary entity — same interface as Element |
| `Process` | Encapsulates an operation on a `ModelPart` |
| `Variable` | Typed data field (e.g., `DISPLACEMENT`, `TEMPERATURE`) |
| `ConstitutiveLaw` | Material law abstraction |
| `ProcessInfo` | Solver-level metadata: time step, iteration count, etc. |
| `Parameters` | JSON-backed configuration object |
| `Kernel` | Bootstraps the framework and loads applications |
| `DataCommunicator` | MPI communication abstraction |
| `Strategy` | Top-level solver orchestration |
| `Scheme` | Time integration / linearization |
| `BuilderAndSolver` | Assembles the global system and solves |

---

## Contribution Workflow

Branch naming: `subject/short-description` (e.g., `core/adding-xxx-utility`, `structural/fix-xxx-element`).

Change checklist before submitting a PR:
1. Do **not** modify `external_libraries/` unless explicitly required.
2. Maintain style consistency with neighboring files.
3. Update CMake, variable registration, and pybind11 binding hooks when adding new entities.
4. Register new components in `RegisterComponents()` in `<app>_application.cpp`.
5. Run the most specific available test first, then broader ones.
6. Avoid editing generated build artifacts (`compile_commands.json`, CMake cache files).

---

## Scope Boundaries

- `kratos/` and `applications/` are both first-class — changes to either are valid.
- `external_libraries/` contains vendored third-party code — do not modify unless explicitly asked.
- Core (`kratos/`) changes affect all applications; be conservative and test broadly.
- Application changes are scoped — target only the affected application unless a cross-cutting fix is needed.

---

## Common Gotchas

- **Variable registration**: new `Variable`s must be declared with `KRATOS_DEFINE_APPLICATION_VARIABLE` in `<app>_application_variables.h` and registered via `KRATOS_REGISTER_VARIABLE` in `<app>_application.cpp`.
- **CMake globbing**: new `.cpp` files under `custom_*/` are auto-collected by `file(GLOB_RECURSE ...)` only if that pattern already covers the directory. Check the application's `CMakeLists.txt`.
- **Python install path**: after adding a new `.py` file to `python_scripts/`, it is installed via `kratos_python_install` — no manual install step needed.
- **MPI tests**: keep `OMP_NUM_THREADS=1` consistent with CI when running MPI test tasks.
- **Test suites**: adding a new `test_*.py` file does not automatically include it in the suite. Add it to the `test_<ApplicationName>.py` suite runner.
- **Pointer macros**: Elements/Conditions use `KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION`; Processes/Utilities use `KRATOS_CLASS_POINTER_DEFINITION`.
