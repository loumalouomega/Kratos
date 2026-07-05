# Eigen Sparse Linear-Algebra Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Deliverable note:** this plan itself must be saved as `Plan.md` at the repo root (Task 0) — the user requested a `Plan.md` to drive the implementation.

**Goal:** Make Eigen available as a configure-time alternative to Boost uBLAS for the **sparse** system linear algebra in the Kratos core (system matrix, RHS vector, builders, linear solvers), selected via `KRATOS_LINEAR_ALGEBRA_BACKEND=ublas|eigen` in the configure script, with `ublas` as the default so nothing breaks.

**Architecture:** Follow the proven in-tree precedents. (1) The removed `KRATOS_USE_AMATRIX` switch (commit `87b4f59fb89^`) shows the pattern: a Kratos-owned wrapper type adds the ublas-style member surface (`size1/size2`, `value_data()/index1_data()/index2_data()`, `set_filled`, 3-arg `resize`) on top of the new backend, so existing code compiles unchanged. (2) `kratos/spaces/kratos_space.h` (native `CsrMatrix` space) is the structural blueprint for a new `EigenSpace` mirroring the full `UblasSpace` static API. (3) A SFINAE-constrained free-function compat layer (`prod`, `inner_prod`, `noalias`, `trans`, `norm_2`, …) makes Eigen types usable with the ublas idiom without colliding with the `using namespace boost::numeric::ublas;` in `ublas_interface.h`. The backend switch happens only at the space-instantiation boundary (python bindings / registrations) through a new `default_spaces.h` alias header.

**Tech Stack:** C++20, Eigen 3.5.0 (already vendored in LinearSolversApplication, to be moved to `external_libraries/`), Boost uBLAS, CMake, pybind11, GTest (`KRATOS_TEST_CASE_IN_SUITE` in `KratosCoreFastSuite`).

## Context

Kratos hardcodes Boost uBLAS: `kratos/includes/ublas_interface.h` does `using namespace boost::numeric::ublas;` and defines `Kratos::Matrix/Vector/CompressedMatrix`, injecting `prod/noalias/inner_prod/...` into tens of thousands of call sites (~1,700 in core, ~17,000 in applications). `kratos/spaces/ublas_space.h` is the sparse/dense "space" used by all strategies/builders/solvers. Long-term the project wants `std::linalg` (see existing `KRATOS_FUTURE_STDBLAS` flag), but that is far off; Eigen is the practical intermediate backend.

**Key verified constraints that shaped this design:**
- `kratos/includes/element.h:87-89` hardcodes ublas `Matrix`/`Vector` in the virtual interface of every Element/Condition, and `scheme.h` passes `TDenseSpace::MatrixType&` straight into `CalculateLocalSystem`. → **The dense/local space cannot switch**; this plan switches the sparse space only (user-confirmed scope). `TEigenDenseSpace` is still implemented and tested, ready for a future dense swap.
- `residualbased_block_builder_and_solver.h` builds and assembles the system matrix by writing raw CSR arrays (`value_data().begin()`, `index1_data()`, `set_filled`, `AtomicAdd` into values). This maps 1:1 onto `Eigen::SparseMatrix<T, RowMajor, Index>` in compressed mode (`valuePtr/outerIndexPtr/innerIndexPtr/resizeNonZeros`) — so a wrapper type that exposes the ublas member names makes the existing assembly algorithm work verbatim, with identical thread-safety and performance.
- Eigen requires a **signed** sparse index; ublas uses `std::size_t`. → configurable `KratosEigenIndexType`, default `std::ptrdiff_t` (user-confirmed).
- Eigen is safe-by-default on aliasing (opposite of ublas): `x = A*x` makes a temporary unless `.noalias()`. The compat `noalias()` restores the fast path; correctness is never at risk.

## Global Constraints

- Default build (`KRATOS_LINEAR_ALGEBRA_BACKEND=ublas` or flag absent) must be **behavior- and ABI-identical** to today. All refactors that land before the switch must be backend-neutral.
- `UblasSpace` stays fully functional in both modes forever — 124 application files reference it directly and must keep compiling unchanged.
- Never call `cmake` directly — always build via `bash build/configure.sh` (it wipes the cache, reconfigures, builds, installs). Runtime env: `export PYTHONPATH=$HOME/src/Kratos-test/bin/Release` (or the active build type) and `export LD_LIBRARY_PATH=$HOME/src/Kratos-test/bin/Release/libs:$LD_LIBRARY_PATH`.
- C++ conventions per CLAUDE.md: `#pragma once`, `KRATOS_TRY/KRATOS_CATCH("")` in method bodies, `PascalCase` methods, `r`/`p` parameter prefixes, `KRATOS_EXPECT_*` in tests, no raw `new/delete`.
- Do not modify `external_libraries/` *content* (moving eigen3 there is explicitly requested; do not patch Eigen itself).
- Branch name: `core/eigen-sparse-backend` (split per-PR as `core/eigen-sparse-backend-pr<N>` if landing separately).
- Out of scope (state in docs): swapping `Kratos::Matrix`/`Vector`/`array_1d`/`BoundedMatrix` dense typedefs repo-wide (future AMatrix-style `#ifdef` phase), `std::linalg`, MPI/Trilinos spaces, applications with bespoke builders.

## PR sequencing

```
PR1 (Task 1)      Relocate Eigen to external_libraries/          [independent, land first]
PR2 (Tasks 2-4)   kratos_eigen_interface.h + compat ops + tests  [needs PR1]
PR3 (Tasks 5-6)   eigen_space.h + parity tests                   [needs PR2]
PR4 (Tasks 7-8)   backend-neutral builder genericization         [independent — pure ublas-safe refactor, can go in parallel]
PR5 (Tasks 9-13)  CMake switch + default_spaces.h + bindings + solvers + parity test  [needs PR3 + PR4]
PR6 (Task 14)     configure scripts + CI job + docs              [needs PR5]
```

---

### Task 0: Save this plan into the repo

**Files:**
- Create: `Plan.md` (repo root)

- [ ] **Step 1:** Copy this document verbatim to `/home/ubuntu/src/Kratos-test/Plan.md`.
- [ ] **Step 2:** `git checkout -b core/eigen-sparse-backend && git add Plan.md && git commit -m "[Core] Add plan for Eigen sparse linear-algebra backend"` (append the Claude co-author trailer per repo instructions).

---

### Task 1 (PR1): Relocate Eigen to top-level `external_libraries/`

**Files:**
- Move: `applications/LinearSolversApplication/external_libraries/eigen3` → `external_libraries/eigen3`
- Modify: `CMakeLists.txt` (root, around line 656), `applications/LinearSolversApplication/CMakeLists.txt` (lines ~30-41 include dirs, ~100-102 SuiteSparse Find modules, install-interface strings), `applications/RomApplication/CMakeLists.txt` (line ~11)

**Interfaces:**
- Produces: `#include <Eigen/Core>` / `<Eigen/Sparse>` resolvable from **any** core or application target; `EIGEN_MPL2_ONLY` defined globally.

- [ ] **Step 1:** `git mv applications/LinearSolversApplication/external_libraries/eigen3 external_libraries/eigen3`
- [ ] **Step 2:** Root `CMakeLists.txt`, next to the existing blanket include (line 656 `include_directories( SYSTEM ${KRATOS_SOURCE_DIR}/external_libraries )`):

```cmake
include_directories( SYSTEM ${KRATOS_SOURCE_DIR}/external_libraries/eigen3 )
add_definitions( -DEIGEN_MPL2_ONLY )
```

Global `add_definitions` (not per-target) so every TU agrees — avoids ODR drift with LinearSolversApplication, which currently sets it `PUBLIC` on its own target. Add an install rule for the headers next to how the app installed them, targeting `include/kratos/external_libraries/eigen3`.
- [ ] **Step 3:** `applications/LinearSolversApplication/CMakeLists.txt`: remove the app-local eigen3 include paths (keep `spectra1`), remove its `EIGEN_MPL2_ONLY` (now global), update the `INSTALL_INTERFACE`/`BUILD_INTERFACE` generator expressions and the `include(...Find{CHOLMOD,SPQR,UMFPACK}.cmake)` lines to `${KRATOS_SOURCE_DIR}/external_libraries/eigen3/cmake/...`.
- [ ] **Step 4:** `applications/RomApplication/CMakeLists.txt`: drop the hardcoded `LinearSolversApplication/external_libraries/eigen3` include path (root-level include now covers it; keep the spectra1 path).
- [ ] **Step 5:** `grep -rn "LinearSolversApplication/external_libraries/eigen3" --include="*.txt" --include="*.cmake" --include="*.sh" --include="*.py" --include="*.md" .` — fix any remaining references (packaging scripts, docs, `.github/`).
- [ ] **Step 6:** Build: `bash build/configure.sh`. Expected: configures and builds cleanly (working copy enables LinearSolversApplication, which exercises the moved headers).
- [ ] **Step 7:** Run the LinearSolvers C++ suite (VS Code task `Run C++ Test Suite` or the gtest binary in `bin/<type>/test/`): expected all PASS.
- [ ] **Step 8:** Commit: `git commit -m "[External] Move eigen3 from LinearSolversApplication to external_libraries"`.

---

### Task 2 (PR2): `kratos/includes/kratos_eigen_interface.h` — wrapper types

**Files:**
- Create: `kratos/includes/kratos_eigen_interface.h`

**Interfaces:**
- Produces (consumed by Tasks 3, 5, 9-11):
  - `Kratos::KratosEigenIndexType` (default `std::ptrdiff_t`, overridable via `-DKRATOS_EIGEN_INDEX_TYPE=<type>` compile definition)
  - `Kratos::EigenMatrix<TDataType>` — row-major dense, ublas member surface
  - `Kratos::EigenVector<TDataType>` — dense column vector, ublas member surface
  - `Kratos::EigenCompressedMatrix<TDataType, TIndexType = KratosEigenIndexType>` — row-major CSR, ublas member surface

This is additive (like `csr_matrix.h`) — it compiles in every build, independent of the backend switch. Header layout sketch (implement fully; the AMatrix precedent `kratos/includes/amatrix_interface.h` shows the wrapper idiom):

```cpp
#pragma once

// System includes
#include <cstddef>
#include <type_traits>

// External includes
#include <Eigen/Core>
#include <Eigen/Sparse>

// Project includes
#include "includes/define.h"

namespace Kratos {

#ifdef KRATOS_EIGEN_INDEX_TYPE
using KratosEigenIndexType = KRATOS_EIGEN_INDEX_TYPE;
#else
using KratosEigenIndexType = std::ptrdiff_t;
#endif
static_assert(std::is_signed_v<KratosEigenIndexType>,
              "Eigen requires a signed sparse StorageIndex.");

namespace Internals {
// Iterable view over a raw backend array; gives Eigen's CSR arrays the
// same begin()/end()/operator[] surface ublas storage arrays expose.
template<class T>
class EigenArrayProxy {
public:
    EigenArrayProxy(T* pData, std::size_t Size) : mpData(pData), mSize(Size) {}
    T* begin() { return mpData; }
    T* end() { return mpData + mSize; }
    const T* begin() const { return mpData; }
    const T* end() const { return mpData + mSize; }
    T& operator[](std::size_t Index) { return mpData[Index]; }
    const T& operator[](std::size_t Index) const { return mpData[Index]; }
    std::size_t size() const { return mSize; }
private:
    T* mpData;
    std::size_t mSize;
};
} // namespace Internals

template<class TDataType>
class EigenMatrix : public Eigen::Matrix<TDataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> {
public:
    using BaseType = Eigen::Matrix<TDataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using value_type = TDataType;
    using size_type = std::size_t;

    EigenMatrix() = default;
    EigenMatrix(std::size_t Size1, std::size_t Size2) : BaseType(Size1, Size2) {}
    template<class TDerived>
    EigenMatrix(const Eigen::MatrixBase<TDerived>& rOther) : BaseType(rOther) {}
    template<class TDerived>
    EigenMatrix& operator=(const Eigen::MatrixBase<TDerived>& rOther) {
        BaseType::operator=(rOther); return *this;
    }

    std::size_t size1() const { return static_cast<std::size_t>(this->rows()); }
    std::size_t size2() const { return static_cast<std::size_t>(this->cols()); }
    void resize(std::size_t Size1, std::size_t Size2, bool Preserve = false) {
        if (Preserve) this->conservativeResize(Size1, Size2);
        else BaseType::resize(Size1, Size2);
    }
};

template<class TDataType>
class EigenVector : public Eigen::Matrix<TDataType, Eigen::Dynamic, 1> {
public:
    using BaseType = Eigen::Matrix<TDataType, Eigen::Dynamic, 1>;
    using value_type = TDataType;
    using size_type = std::size_t;

    EigenVector() = default;
    explicit EigenVector(std::size_t Size) : BaseType(Size) {}
    EigenVector(std::size_t Size, const TDataType Value) : BaseType(BaseType::Constant(Size, Value)) {}
    template<class TDerived>
    EigenVector(const Eigen::MatrixBase<TDerived>& rOther) : BaseType(rOther) {}
    template<class TDerived>
    EigenVector& operator=(const Eigen::MatrixBase<TDerived>& rOther) {
        BaseType::operator=(rOther); return *this;
    }

    // rows()/size()/operator[]/operator() inherited; add ublas-style resize
    void resize(std::size_t NewSize, bool Preserve = false) {
        if (Preserve) this->conservativeResize(NewSize);
        else BaseType::resize(NewSize);
    }
};

template<class TDataType, class TIndexType = KratosEigenIndexType>
class EigenCompressedMatrix : public Eigen::SparseMatrix<TDataType, Eigen::RowMajor, TIndexType> {
public:
    using BaseType = Eigen::SparseMatrix<TDataType, Eigen::RowMajor, TIndexType>;
    using value_type = TDataType;
    using size_type = std::size_t;

    EigenCompressedMatrix() = default;
    EigenCompressedMatrix(std::size_t Size1, std::size_t Size2) : BaseType(Size1, Size2) {}
    // ublas-style (rows, cols, nnz): allocate compressed storage up front so
    // the builder can write outer/inner/value arrays directly.
    EigenCompressedMatrix(std::size_t Size1, std::size_t Size2, std::size_t NNZ)
        : BaseType(Size1, Size2) {
        this->resizeNonZeros(NNZ);
    }

    std::size_t size1() const { return static_cast<std::size_t>(this->rows()); }
    std::size_t size2() const { return static_cast<std::size_t>(this->cols()); }
    std::size_t nnz() const { return static_cast<std::size_t>(this->nonZeros()); }

    auto value_data()  { return Internals::EigenArrayProxy<TDataType>(this->valuePtr(), nnz()); }
    auto index1_data() { return Internals::EigenArrayProxy<TIndexType>(this->outerIndexPtr(), size1() + 1); }
    auto index2_data() { return Internals::EigenArrayProxy<TIndexType>(this->innerIndexPtr(), nnz()); }
    auto value_data()  const { return Internals::EigenArrayProxy<const TDataType>(this->valuePtr(), nnz()); }
    auto index1_data() const { return Internals::EigenArrayProxy<const TIndexType>(this->outerIndexPtr(), size1() + 1); }
    auto index2_data() const { return Internals::EigenArrayProxy<const TIndexType>(this->innerIndexPtr(), nnz()); }

    void set_filled(std::size_t FilledSize1, std::size_t FilledNNZ) {
        // Storage was pre-sized by the (rows, cols, nnz) ctor; validate only.
        KRATOS_DEBUG_ERROR_IF(FilledSize1 != size1() + 1 || FilledNNZ != nnz())
            << "set_filled inconsistent with allocated compressed storage." << std::endl;
    }

    void resize(std::size_t Size1, std::size_t Size2, bool Preserve = false) {
        if (Preserve) this->conservativeResize(Size1, Size2);
        else BaseType::resize(Size1, Size2); // drops values and structure, like ublas resize(m,n,false)
    }

    // ublas-style element access; inserting into a compressed matrix is the
    // same O(nnz-in-row) slow path as ublas operator() insertion.
    TDataType& operator()(std::size_t I, std::size_t J) { return this->coeffRef(I, J); }
    TDataType operator()(std::size_t I, std::size_t J) const { return this->coeff(I, J); }
};

} // namespace Kratos
```

- [ ] **Step 1:** Write a failing test first — create `kratos/tests/cpp_tests/includes/test_kratos_eigen_interface.cpp` in `KratosCoreFastSuite` covering: `(rows, cols, nnz)` ctor leaves the matrix compressed (`isCompressed()`) with writable `index1_data()/index2_data()/value_data()` of the right sizes; a hand-built 3×3 CSR round-trips through `operator()` reads; dense `resize(m, n, true)` preserves values; `EigenVector(n, value)` fills. Use `KRATOS_EXPECT_EQ`/`KRATOS_EXPECT_NEAR`.
- [ ] **Step 2:** Build (`bash build/configure.sh`) — expected: test file fails to compile (header missing). That is the "red" state for a header-only unit.
- [ ] **Step 3:** Implement `kratos/includes/kratos_eigen_interface.h` per the sketch.
- [ ] **Step 4:** Rebuild and run `KratosCoreFastSuite` filtered (`Run C++ Test Suite Filtered` task, pattern `KratosEigenInterface`): expected PASS.
- [ ] **Step 5:** Commit: `git commit -m "[Core] Add Eigen interface wrapper types (kratos_eigen_interface.h)"`.

---

### Task 3 (PR2): `kratos/includes/eigen_compat_operations.h` — ublas-idiom free functions for Eigen types

**Files:**
- Create: `kratos/includes/eigen_compat_operations.h` (included at the end of `kratos_eigen_interface.h`)

**Interfaces:**
- Produces, in `namespace Kratos`, deduction-constrained on `Eigen::MatrixBase<D>` / `Eigen::SparseMatrixBase<D>` so they can never collide with the ublas overloads injected by `using namespace boost::numeric::ublas;` (ublas deduces on `matrix_expression<E>`/`vector_expression<E>`, and its `row/subrange` require `typename M::size_type` — each library's overloads drop out of the candidate set for the other's types):

```cpp
namespace Kratos {

template<class TDerived1, class TDerived2>
inline auto prod(const Eigen::MatrixBase<TDerived1>& rA, const Eigen::MatrixBase<TDerived2>& rB) { return rA * rB; }

template<class TDerived1, class TDerived2>
inline auto prod(const Eigen::SparseMatrixBase<TDerived1>& rA, const Eigen::MatrixBase<TDerived2>& rX) { return rA * rX; }

template<class TDerived1, class TDerived2>
inline auto inner_prod(const Eigen::MatrixBase<TDerived1>& rX, const Eigen::MatrixBase<TDerived2>& rY) { return rX.dot(rY); }

template<class TDerived1, class TDerived2>
inline auto outer_prod(const Eigen::MatrixBase<TDerived1>& rX, const Eigen::MatrixBase<TDerived2>& rY) { return rX * rY.transpose(); }

template<class TDerived> inline auto trans(const Eigen::MatrixBase<TDerived>& rM) { return rM.transpose(); }
template<class TDerived> inline auto norm_1(const Eigen::MatrixBase<TDerived>& rX) { return rX.template lpNorm<1>(); }
template<class TDerived> inline auto norm_2(const Eigen::MatrixBase<TDerived>& rX) { return rX.norm(); }
template<class TDerived> inline auto norm_inf(const Eigen::MatrixBase<TDerived>& rX) { return rX.template lpNorm<Eigen::Infinity>(); }
template<class TDerived> inline auto sum(const Eigen::MatrixBase<TDerived>& rX) { return rX.sum(); }

// Eigen's own NoAlias proxy supports =, +=, -= — exactly the ublas contract.
template<class TDerived> inline auto noalias(Eigen::MatrixBase<TDerived>& rM) { return rM.noalias(); }

template<class TDerived> inline auto row(Eigen::MatrixBase<TDerived>& rM, std::size_t I) { return rM.row(I); }
template<class TDerived> inline auto row(const Eigen::MatrixBase<TDerived>& rM, std::size_t I) { return rM.row(I); }
template<class TDerived> inline auto column(Eigen::MatrixBase<TDerived>& rM, std::size_t J) { return rM.col(J); }
template<class TDerived> inline auto column(const Eigen::MatrixBase<TDerived>& rM, std::size_t J) { return rM.col(J); }

template<class TDerived>
inline auto subrange(Eigen::MatrixBase<TDerived>& rV, std::size_t Low, std::size_t High) { return rV.segment(Low, High - Low); }
template<class TDerived>
inline auto subrange(const Eigen::MatrixBase<TDerived>& rV, std::size_t Low, std::size_t High) { return rV.segment(Low, High - Low); }

} // namespace Kratos
```

(Extend during implementation only if a Phase-5 consumer needs more — YAGNI; `axpy_prod`/`project` only if the eigen-space build actually hits them.)

- [ ] **Step 1:** Write the failing test `kratos/tests/cpp_tests/includes/test_eigen_compat_operations.cpp`. **Critical property:** the TU must include BOTH `includes/ublas_interface.h` and `includes/kratos_eigen_interface.h`, and each `KRATOS_TEST_CASE_IN_SUITE` computes the same quantity with ublas types and Eigen types and compares with `KRATOS_EXPECT_NEAR` — this is the permanent overload-collision regression test. Cover: `prod` (dense×dense, dense×vector, sparse×vector), `inner_prod`, `outer_prod`, `trans`, `norm_1/2/inf`, `noalias(y) = prod(A, x)` and `noalias(y) += prod(A, x)`, `row/column/subrange` read and write-through.
- [ ] **Step 2:** Build — expected compile failure (header missing).
- [ ] **Step 3:** Implement the header.
- [ ] **Step 4:** Rebuild, run filtered suite (`EigenCompat`): expected PASS.
- [ ] **Step 5:** Commit: `git commit -m "[Core] Add ublas-idiom compat operations for Eigen types"`.

---

### Task 4 (PR2): PR2 wrap-up

- [ ] **Step 1:** Full core suite: run `KratosCoreFastSuite` (VS Code task `Run C++ Test Suite`). Expected: all PASS, zero behavior change (headers are additive).
- [ ] **Step 2:** If gcc and clang are both available locally, compile the two new test TUs with both (overload-resolution edge cases); otherwise note that CI covers it.
- [ ] **Step 3:** Push PR2.

---

### Task 5 (PR3): `kratos/spaces/eigen_space.h`

**Files:**
- Create: `kratos/spaces/eigen_space.h`
- Reference (contract): `kratos/spaces/ublas_space.h`; blueprint: `kratos/spaces/kratos_space.h`; reuse: `kratos/utilities/dof_updater.h` (space-generic), `kratos/utilities/parallel_utilities.h` (`IndexPartition`), `kratos/utilities/atomic_utilities.h`, `kratos/utilities/reduction_utilities.h`

**Interfaces:**
- Produces:

```cpp
template<class TDataType, class TMatrixType, class TVectorType> class EigenSpace;

template<class TDataType, class TIndexType = KratosEigenIndexType>
using TEigenSparseSpace = EigenSpace<TDataType, EigenCompressedMatrix<TDataType, TIndexType>, EigenVector<TDataType>>;

template<class TDataType>
using TEigenDenseSpace = EigenSpace<TDataType, EigenMatrix<TDataType>, EigenVector<TDataType>>;
```

Mirror the **complete** `UblasSpace` static API — every public static method in `kratos/spaces/ublas_space.h` gets an `EigenSpace` equivalent with identical name, signature shape, and semantics: `CreateEmptyMatrixPointer/CreateEmptyVectorPointer`, `IsNull`, `Size/Size1/Size2`, `GetColumn/SetColumn`, `Copy` (both), `Assign`, `Set/SetValue/GetValue`, `Dot`, `TwoNorm` (vector / dense / compressed-Frobenius), `JacobiNorm` (dense + compressed), `Mult` (dense + sparse), `TransposeMult`, `RowDot`, `GraphDegree/GraphNeighbors`, `InplaceMult`, `UnaliasedAdd`, `ScaleAndAdd` (both overloads), `Resize` (matrix/matrix-pointer/vector/vector-pointer), `Clear`, `ResizeData`, `SetToZero` (matrix + vector), `CheckAndCorrectZeroDiagonalValues`, `GetScaleNorm/GetDiagonalNorm/GetAveragevalueDiagonal/GetMaxDiagonal/GetMinDiagonal`, `GatherValues`, `WriteMatrixMarketMatrix/WriteMatrixMarketVector`, `IsDistributed() → false`, `FastestDirectSolverList`, `CreateDofUpdater`, `Info/PrintInfo/PrintData`, plus the typedefs `DataType/MatrixType/VectorType/IndexType/SizeType/MatrixPointerType/VectorPointerType/DofUpdaterType/DofUpdaterPointerType` and `KRATOS_CLASS_POINTER_DEFINITION(EigenSpace)`. Must instantiate for `double` and `std::complex<double>` (complex space bindings exist in `add_factories_to_python.cpp` and friends).

Representative implementations (port the rest from `ublas_space.h`, translating CSR internals `value_data/index1_data/index2_data` → `valuePtr/outerIndexPtr/innerIndexPtr`):

```cpp
// Sparse SpMV. Deterministic manual CSR loop (same as kratos_space.h) rather
// than Eigen's internally-parallelized product, to avoid nested-OpenMP
// surprises under KRATOS_SMP_OPENMP and keep behavior identical to UblasSpace.
static void Mult(const MatrixType& rA, const VectorType& rX, VectorType& rY)
{
    if (static_cast<SizeType>(rY.size()) != Size1(rA)) rY.resize(Size1(rA), false);
    const auto* row_ptr = rA.outerIndexPtr();
    const auto* col_idx = rA.innerIndexPtr();
    const auto* values  = rA.valuePtr();
    IndexPartition<SizeType>(Size1(rA)).for_each([&](SizeType i) {
        TDataType acc = TDataType();
        for (auto k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            acc += values[k] * rX[col_idx[k]];
        rY[i] = acc;
    });
}

// Keep the sparsity pattern, zero the values (NOT setZero(), which drops the graph).
static void SetToZero(MatrixType& rA)
{
    IndexPartition<SizeType>(rA.nnz()).for_each([&](SizeType i) { rA.valuePtr()[i] = TDataType(); });
}

static double TwoNorm(const EigenCompressedMatrix<TDataType>& rA) // Frobenius
{
    return Eigen::Map<const Eigen::Matrix<TDataType, Eigen::Dynamic, 1>>(rA.valuePtr(), rA.nonZeros()).norm();
}

// Direct port of UblasSpace::CheckAndCorrectZeroDiagonalValues over CSR arrays.
static double CheckAndCorrectZeroDiagonalValues(const ProcessInfo& rProcessInfo,
    MatrixType& rA, VectorType& rb, const SCALING_DIAGONAL ScalingDiagonal)
{
    const auto* row_ptr = rA.outerIndexPtr();
    const auto* col_idx = rA.innerIndexPtr();
    auto* values = rA.valuePtr();
    const double zero_tolerance = std::numeric_limits<double>::epsilon();
    const double scale_factor = GetScaleNorm(rProcessInfo, rA, ScalingDiagonal);
    IndexPartition<SizeType>(Size1(rA)).for_each([&](SizeType i) {
        bool empty = true;
        for (auto k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            if (static_cast<SizeType>(col_idx[k]) == i && std::abs(values[k]) > zero_tolerance) {
                empty = false; break;
            }
        }
        if (empty) { rA.coeffRef(i, i) = scale_factor; rb[i] = 0.0; }
    });
    return scale_factor;
}

// ublas Resize(A,m,n) == resize(m,n,false): discard values AND structure. Matches Eigen resize.
static void Resize(MatrixType& rA, SizeType m, SizeType n) { rA.resize(m, n, false); }
static void Resize(VectorType& rX, SizeType n) { rX.resize(n, false); } // uninitialized, like ublas
```

Matrix-market note: the vendored eigen3 has **no** `unsupported/` module (no `saveMarket`). Check whether `kratos/includes/matrix_market_interface.h` templates are generic enough to instantiate on `EigenCompressedMatrix` (it iterates CSR-style members); if yes, reuse; if not, write a small manual writer producing identical output.

- [ ] **Step 1:** Create `kratos/tests/cpp_tests/spaces/test_eigen_space.cpp` first — mirror every test in `kratos/tests/cpp_tests/spaces/test_ublas_space.cpp` and `test_kratos_space.cpp` (the existing per-backend parity convention), suite `KratosCoreFastSuite`.
- [ ] **Step 2:** Build — expected compile failure (`spaces/eigen_space.h` missing).
- [ ] **Step 3:** Implement `eigen_space.h` (full API port).
- [ ] **Step 4:** Rebuild, run filtered (`EigenSpace`): expected PASS.
- [ ] **Step 5:** Commit: `git commit -m "[Core] Add EigenSpace mirroring the UblasSpace static API"`.

### Task 6 (PR3): cross-backend parity tests

**Files:**
- Modify: `kratos/tests/cpp_tests/spaces/test_eigen_space.cpp`

- [ ] **Step 1:** Add parity cases that build identical data in `TUblasSparseSpace<double>` and `TEigenSparseSpace<double>` types and `KRATOS_EXPECT_NEAR`-compare results of: `Dot`, `TwoNorm` (all three), `Mult`, `TransposeMult`, `ScaleAndAdd` (both), `UnaliasedAdd`, `InplaceMult`, `CheckAndCorrectZeroDiagonalValues`, `GetScaleNorm/GetDiagonalNorm/GetMaxDiagonal/GetMinDiagonal`, and a **CSR round-trip test that does exactly what the block builder does**: construct `EigenCompressedMatrix(rows, cols, nnz)`, write `index1_data()/index2_data()/value_data()` through `.begin()` pointers, `set_filled`, then verify `Mult` and `operator()` reads.
- [ ] **Step 2:** Run filtered suite: expected PASS. Run full `KratosCoreFastSuite`: expected PASS.
- [ ] **Step 3:** Commit and push PR3: `git commit -m "[Core] Add Eigen/uBLAS space parity tests"`.

---

### Task 7 (PR4): backend-neutral genericization of the block builder

**Files:**
- Modify: `kratos/solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h`

**Interfaces:**
- Consumes: nothing new — this is a pure refactor that must be behavior-identical under ublas.
- Produces: a builder whose matrix access compiles for any `TSparseSpace::MatrixType` exposing the ublas CSR member surface (which both ublas `compressed_matrix` and `EigenCompressedMatrix` do).

The four verified coupling points and their fixes:

- [ ] **Step 1:** Line 122 `typedef boost::numeric::ublas::compressed_matrix<double> CompressedMatrixType;` → `typedef typename TSparseSpace::MatrixType CompressedMatrixType;` (and line ~1510's construction `A = CompressedMatrixType(rows, cols, nnz)` keeps working via the wrapper's 3-arg ctor).
- [ ] **Step 2:** All raw-pointer locals over CSR arrays become deduced. Pattern (applies at lines ~885-887, ~1314-1316, ~1512-1514, ~1603-1605):

```cpp
// before
double* Avalues = rA.value_data().begin();
std::size_t* Arow_indices = rA.index1_data().begin();
std::size_t* Acol_indices = rA.index2_data().begin();
// after
auto* Avalues = rA.value_data().begin();
auto* Arow_indices = rA.index1_data().begin();
auto* Acol_indices = rA.index2_data().begin();
```

Keep loop-bound variables `std::size_t` and cast once at the pointer read to contain `-Wsign-compare` fallout (Eigen indices are signed).
- [ ] **Step 3:** `ForwardFind`/`BackwardFind` (lines ~1729-1745) take `const std::size_t*` — template them on the index type:

```cpp
template<class TIndexPointerType>
inline unsigned int ForwardFind(const unsigned int Id, const unsigned int Start, const TIndexPointerType Index2Vector)
```

(same for `BackwardFind`; call sites unchanged thanks to deduction).
- [ ] **Step 4:** `AssembleRowContribution` and the graph-construction loops: verify no other hardcoded `std::size_t*`/`double*` remains (`grep -n "value_data\|index1_data\|index2_data\|set_filled" <file>`).
- [ ] **Step 5:** Build with the default ublas backend; run `KratosCoreFastSuite` plus the builder-and-solver tests (`kratos/tests/cpp_tests/strategies/builder_and_solvers/`): expected all PASS, proving backend-neutrality.
- [ ] **Step 6:** Commit: `git commit -m "[Core] Genericize block builder-and-solver CSR access over the sparse space matrix type"`.

### Task 8 (PR4): same treatment for the remaining CSR-internal users

**Files:**
- Modify: `kratos/solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver.h`, `residualbased_elimination_builder_and_solver_with_constraints.h`, `residualbased_block_builder_and_solver_with_lagrange_multiplier.h` (and siblings found by grep)
- Modify: `kratos/utilities/sparse_matrix_multiplication_utility.h` (typed `const IndexType*` locals at ~142-155, 385-390, 590-605 → deduced; the index-templated variants at lines 261+ already show the target pattern)

- [ ] **Step 1:** `grep -rln "index1_data\|value_data().begin()\|set_filled" kratos/ --include="*.h" --include="*.cpp"` — enumerate the actual list (expected: the builders above, `sparse_matrix_multiplication_utility.h`, `amgcl` adaptor code, `add_matrix_to_python.cpp` (ublas-only binding — leave), `p_multigrid/*`, `monotonicity_preserving_solver.h` if present).
- [ ] **Step 2:** Apply the Task-7 pattern (deduced pointer types, `typename TSparseSpace::MatrixType` instead of hardcoded `compressed_matrix`) to each file that is instantiated with `TSparseSpace`. Files only ever used with ublas types (python matrix binding) stay untouched.
- [ ] **Step 3:** Build + run `KratosCoreFastSuite` and, with the working-copy configure, the LinearSolvers suite: expected all PASS.
- [ ] **Step 4:** Commit and push PR4: `git commit -m "[Core] Make sparse-matrix internal access index-type generic in builders and utilities"`.

---

### Task 9 (PR5): CMake backend switch + `default_spaces.h`

**Files:**
- Modify: `CMakeLists.txt` (root — next to the `KRATOS_SHARED_MEMORY_PARALLELIZATION` block at lines ~476-507, whose if/elseif → `add_definitions` idiom this copies)
- Create: `kratos/spaces/default_spaces.h`

**Interfaces:**
- Produces: CMake cache var `KRATOS_LINEAR_ALGEBRA_BACKEND` (`ublas` default | `eigen`); compile definition `KRATOS_USE_EIGEN_BACKEND`; and:

```cpp
// kratos/spaces/default_spaces.h
#pragma once
#include "spaces/ublas_space.h"
#ifdef KRATOS_USE_EIGEN_BACKEND
#include "spaces/eigen_space.h"
namespace Kratos {
template<class TDataType> using TDefaultSparseSpace = TEigenSparseSpace<TDataType>;
}
#else
namespace Kratos {
template<class TDataType> using TDefaultSparseSpace = TUblasSparseSpace<TDataType>;
}
#endif
namespace Kratos {
// Dense/local space stays ublas: Element/Condition virtual interfaces hardcode
// Kratos::Matrix/Vector (includes/element.h:87-89) and schemes pass
// TDenseSpace::MatrixType directly into CalculateLocalSystem.
template<class TDataType> using TDefaultDenseSpace = TUblasDenseSpace<TDataType>;

using DefaultSparseSpaceType        = TDefaultSparseSpace<double>;
using DefaultLocalSpaceType         = TDefaultDenseSpace<double>;
using DefaultComplexSparseSpaceType = TDefaultSparseSpace<std::complex<double>>;
using DefaultComplexLocalSpaceType  = TDefaultDenseSpace<std::complex<double>>;
}
```

- [ ] **Step 1:** Root CMake block:

```cmake
if(NOT DEFINED KRATOS_LINEAR_ALGEBRA_BACKEND)
  message(STATUS "\"KRATOS_LINEAR_ALGEBRA_BACKEND\" not defined, defaulting to \"ublas\"")
  set(KRATOS_LINEAR_ALGEBRA_BACKEND "ublas")
endif()
if(KRATOS_LINEAR_ALGEBRA_BACKEND STREQUAL "ublas")
  # default backend — no definition needed
elseif(KRATOS_LINEAR_ALGEBRA_BACKEND STREQUAL "eigen")
  add_definitions(-DKRATOS_USE_EIGEN_BACKEND)
else()
  message(FATAL_ERROR "Invalid KRATOS_LINEAR_ALGEBRA_BACKEND \"${KRATOS_LINEAR_ALGEBRA_BACKEND}\". Options: \"ublas\", \"eigen\"")
endif()
```

Global `add_definitions` on purpose: the define changes mangled names of instantiated strategies, so core and every application in one build must agree (same rule as `KRATOS_SMP_*`). Never mix binaries built with different flags.
- [ ] **Step 2:** Create `default_spaces.h` as above.
- [ ] **Step 3:** Build default (ublas): expected unchanged. Commit: `git commit -m "[Core] Add KRATOS_LINEAR_ALGEBRA_BACKEND switch and default_spaces.h"`.

### Task 10 (PR5): swap the space typedefs at the binding/registration boundary

**Files (the pattern is one edit repeated; representative list):**
- Modify: `kratos/python/add_strategies_to_python.cpp` (lines 78-82 — the canonical site), `add_linear_solvers_to_python.cpp`, `add_amgcl_solver_to_python.cpp`, `add_convergence_accelerators_to_python.cpp`, `add_processes_to_python.cpp`, `add_factories_to_python.cpp`, `add_other_utilities_to_python.cpp`
- Modify: `kratos/sources/kratos_components.cpp` (lines ~155-166 — factory/ExplicitBuilder registrations)

- [ ] **Step 1:** In each file, replace the local space typedefs:

```cpp
// before (add_strategies_to_python.cpp:78-82)
typedef UblasSpace<double, CompressedMatrix, boost::numeric::ublas::vector<double>> SparseSpaceType;
typedef UblasSpace<double, Matrix, Vector> LocalSpaceType;
// after
#include "spaces/default_spaces.h"
typedef DefaultSparseSpaceType SparseSpaceType;
typedef DefaultLocalSpaceType LocalSpaceType;
```

(complex variants → `DefaultComplexSparseSpaceType`/`DefaultComplexLocalSpaceType`). In `kratos_components.cpp`, register the default-space instantiations via the aliases; **also keep the explicit ublas registrations** so `UblasSpace`-hardcoding applications keep finding their factories in the eigen build.
- [ ] **Step 2:** Build default (ublas): the aliases resolve to exactly the old types — binary-identical behavior. Run `KratosCoreFastSuite` + core Python tests (`Run Tests` task): expected PASS.
- [ ] **Step 3:** Commit: `git commit -m "[Core] Route space instantiation at the python/registration boundary through default_spaces.h"`.

### Task 11 (PR5): make the linear solvers work with `TEigenSparseSpace`

**Files:**
- Modify: `kratos/linear_solvers/amgcl_solver_impl.cpp` (the `AMGCLAdaptor` trait extension point at line ~39) + the explicit `AMGCLSolver` instantiations
- Create: `kratos/linear_solvers/linear_solver_eigen.h/.cpp` (sibling of `linear_solver_ublas.h/.cpp` — explicit `LinearSolver<TEigenSparseSpace<double>, TUblasDenseSpace<double>>` instantiations, compiled unconditionally so both backends' symbols always exist and CI always compiles them)
- Modify: `kratos/linear_solvers/skyline_lu_factorization_solver.h` / `skyline_lu_custom_scalar_solver.h` (verify member-API usage; the wrapper's `size1/size2/operator()`/iterator surface should cover it — fix what grep finds)
- Modify: `kratos/python/add_matrix_to_python.cpp` — add (under `#ifdef KRATOS_USE_EIGEN_BACKEND`) a `CreateMatrixInterface<EigenCompressedMatrix<double>>(m, "CompressedMatrix")`-style binding so `strategy.GetSystemMatrix()` stays usable from Python; the existing binder template mostly works because the wrapper provides `size1/size2/operator()`

- [ ] **Step 1:** AMGCL adaptor specialization (zero-copy — `std::ptrdiff_t` indices are exactly what amgcl's CRS wants):

```cpp
template<class TValue>
struct AMGCLAdaptor<TEigenSparseSpace<TValue>> {
    template<class TMatrix>
    static auto Adapt(const TMatrix& rA) {
        return amgcl::adapter::zero_copy(rA.rows(), rA.outerIndexPtr(), rA.innerIndexPtr(), rA.valuePtr());
    }
};
```

(match the exact trait shape found at `amgcl_solver_impl.cpp:39` during implementation) and add `template class AMGCLSolver<TEigenSparseSpace<double>, TUblasDenseSpace<double>>;`.
- [ ] **Step 2:** Create `linear_solver_eigen.h/.cpp` mirroring `linear_solver_ublas.h/.cpp` line-for-line with the Eigen sparse space; add both files to the core sources (they're under `kratos/linear_solvers/`, covered by the existing glob — verify in `kratos/CMakeLists.txt`).
- [ ] **Step 3:** Grep skyline/iterative solvers for member-API gaps (`grep -n "index1_data\|\.size1()\|filled1" kratos/linear_solvers/*.h`) and fix with the Task-7 pattern where needed.
- [ ] **Step 4:** Build **eigen backend**: edit `build/configure.sh` temporarily to add `-DKRATOS_LINEAR_ALGEBRA_BACKEND="eigen"`, run `bash build/configure.sh`. Expected: full compile. This step is the forcing function that surfaces every remaining coupling — iterate compile errors using the Task-7/8 patterns until green.
- [ ] **Step 5:** Run `KratosCoreFastSuite` under the eigen build: expected PASS.
- [ ] **Step 6:** Rebuild with default ublas (revert the configure edit) and re-run the suite: expected PASS (no regression).
- [ ] **Step 7:** Commit: `git commit -m "[Core] Enable AMGCL, skyline and LinearSolver instantiations for the Eigen sparse space"`.

### Task 12 (PR5): LinearSolversApplication under the eigen backend

**Files:**
- Modify: `applications/LinearSolversApplication/linear_solvers_define.h` and `custom_solvers/eigen_direct_solver.h` (the `SpaceType` mapping + `UblasWrapper` usage at line ~58)

- [ ] **Step 1:** Route the app's `SpaceType<Scalar>::Global` through `spaces/default_spaces.h`. Where the solver currently builds a `UblasWrapper` (ublas → `Eigen::Map`), add a compile-time branch: when `TSparseSpaceType::MatrixType` is already an Eigen sparse type, construct the solver input via `Eigen::Map` directly with an index-converting copy `ptrdiff_t → int` (`Eigen::SparseMatrix<T,RowMajor,int>` requires it; same cost as today's `UblasWrapper` index copy). Keep the `UblasWrapper` path untouched for the ublas backend.
- [ ] **Step 2:** Build both backends with LinearSolversApplication enabled; run `KratosLinearSolversFastSuite` + the app's Python tests under both. Expected: PASS ×2.
- [ ] **Step 3:** Commit: `git commit -m "[LinearSolversApplication] Support the core Eigen sparse space directly"`.

### Task 13 (PR5): builder-level parity test + smoke verification

**Files:**
- Create: `kratos/tests/cpp_tests/strategies/builder_and_solvers/test_eigen_ublas_builder_parity.cpp` (next to the existing builder tests — copy their ModelPart setup)

- [ ] **Step 1:** Write the test: assemble the same small ModelPart with `ResidualBasedBlockBuilderAndSolver<TUblasSparseSpace<double>, TUblasDenseSpace<double>, ...>` and `<TEigenSparseSpace<double>, TUblasDenseSpace<double>, ...>`; compare CSR arrays (`index1_data/index2_data/value_data`) entry-by-entry and the solved `Dx` with `KRATOS_EXPECT_NEAR` / `KRATOS_EXPECT_VECTOR_NEAR`. This test compiles both instantiations unconditionally, so **both backends are compile-checked in every CI run regardless of the flag**.
- [ ] **Step 2:** Run it under both backend builds: expected PASS ×2.
- [ ] **Step 3:** Python smoke test under the eigen build: run the core Python test suite (`Run Tests` task); if StructuralMechanics/FluidDynamics are enabled in the local configure, run one small patch-test case per app and compare converged results with a ublas run (tolerance 1e-10).
- [ ] **Step 4:** Commit and push PR5: `git commit -m "[Core] Add uBLAS/Eigen builder-and-solve parity test"`.

---

### Task 14 (PR6): configure scripts, CI, docs

**Files:**
- Modify: `scripts/standard_configure.sh`, `scripts/standard_configure.bat`, `scripts/standard_configure_mac.sh`, MINGW variants (add next to `-DUSE_EIGEN_MKL=OFF`, line ~49)
- Modify: `.github/workflows/ci.yml` + `.github/workflows/configure_core.sh`
- Create: docs page under `docs/` (match existing docs layout) describing the switch

- [ ] **Step 1:** Add to every configure template, defaulted and documented:

```sh
-DKRATOS_LINEAR_ALGEBRA_BACKEND="ublas" \
```

with a comment: `# Options: "ublas" (default) | "eigen" — selects the sparse system linear-algebra backend`.
- [ ] **Step 2:** CI: add one job (linux/gcc, `Custom` build type) that copies `configure_core.sh` and injects `-DKRATOS_LINEAR_ALGEBRA_BACKEND="eigen"`, builds core + StructuralMechanics + FluidDynamics, runs the C++ FastSuites and the small Python suite.
- [ ] **Step 3:** Docs: what the flag does, sparse-only scope (dense element-local algebra stays ublas and why — `element.h:87-89`), the `KRATOS_EIGEN_INDEX_TYPE` override, the "never mix binaries across flags" rule, migration guidance for applications (`UblasSpace` hardcoding keeps working; migrate to `spaces/default_spaces.h` aliases to opt in), and the future roadmap (dense typedef swap following the removed `KRATOS_USE_AMATRIX` pattern, `std::linalg` via `KRATOS_FUTURE_STDBLAS`).
- [ ] **Step 4:** Verify `bash scripts/standard_configure.sh` syntax by configuring both modes locally via `build/configure.sh`. Commit and push PR6: `git commit -m "[Core] Expose KRATOS_LINEAR_ALGEBRA_BACKEND in configure templates, CI and docs"`.

---

## Verification (end-to-end)

1. **Default-build invariance (every PR):** `bash build/configure.sh` with no new flags → `KratosCoreFastSuite` + LinearSolvers suite + core Python suite all PASS. PRs 1, 2, 3, 4 must be provably no-ops for behavior.
2. **Eigen build:** add `-DKRATOS_LINEAR_ALGEBRA_BACKEND="eigen"` to `build/configure.sh`, rebuild → same suites PASS; `test_eigen_ublas_builder_parity` PASSes (CSR arrays and solutions match ublas bit-for-bit within 1e-12).
3. **Overload-collision guard:** `test_eigen_compat_operations.cpp` compiles ublas + Eigen calls in one TU — permanent regression test.
4. **Performance sanity:** run `kratos/benchmarks/builder_and_solver_benchmark.cpp` (needs `KRATOS_BUILD_BENCHMARK=ON`) under both backends; assembly must be within noise (identical algorithm by construction), SpMV comparable.
5. **Python level:** under the eigen build, run one nonlinear structural case end-to-end and confirm identical convergence history vs ublas.

## Risks (mitigations baked into tasks above)

- **ODR/ABI across the flag** → global `add_definitions`, whole-build consistency, ublas instantiations always compiled (Tasks 9-11).
- **124 app files hardcoding `UblasSpace`** → `UblasSpace` untouched and its factory registrations kept in both modes (Task 10); apps migrate opportunistically.
- **Eigen aliasing differences** → safe-by-default (temporary) in Eigen; compat `noalias()` restores performance; only genuinely-aliased `noalias` misuse is UB in *both* libraries (behavior parity).
- **Signed/unsigned index fallout** → deduced pointer types + `size_t` loop bounds with single casts (Tasks 7-8); watch `-Wsign-compare`.
- **Accidental slow-path `coeffRef` insertion** → `KRATOS_DEBUG_ERROR_IF` in the wrapper when `operator()` would insert into a compressed matrix can be added if profiling shows hits; graph construction guarantees diagonals exist for the known write sites.
- **Nested OpenMP in Eigen products** → `EigenSpace::Mult` uses the manual `IndexPartition` CSR loop (Task 5), sidestepping Eigen's internal threading entirely.


---

# Implementation Notes (post-execution)

All tasks were executed on this branch. Deviations and discoveries relative to the plan above:

- **Complex spaces stay uBLAS in both modes** (encoded in `spaces/default_spaces.h` via a selector on `TDefaultSparseSpace`): they are only used by the eigenvalue-related solvers and switching them buys nothing.
- **`noalias`/`row`/`column`/`subrange` compat overloads take the concrete wrapper types** (`EigenMatrix`/`EigenVector`), not `Eigen::MatrixBase`: the ublas counterparts of these are fully generic templates (`template<class C> noalias(C&)`), so an exact match on the wrapper is required to win overload resolution. The expression-argument functions (`prod`, `trans`, norms, ...) are `MatrixBase`-constrained as planned.
- **`EigenCompressedMatrix` proxies span the allocated capacity** (`data().size()`), matching ublas storage arrays; `nnz()` is the filled count derived from the row pointers (0 until `set_filled`-style filling), matching ublas semantics.
- **`EigenSpace` normalizes uncompressed matrices lazily** (`EnsureCompressed`): element insertion through `operator()` (python, `AssembleLHS`) puts an Eigen sparse matrix into uncompressed mode, which the CSR-array loops cannot read.
- **Registration/typedef boundary was wider than the plan's list**: `factories/register_factories.h`, `factories/linear_solver_factory.h` and `factories/preconditioner_factory.h` define namespace-scope `SparseSpaceType` typedefs and the `KRATOS_REGISTER_(LINEAR_SOLVER|PRECONDITIONER)` macros; all now follow the default spaces. `solving_strategies/builder_and_solvers/explicit_builder.cpp` registers a static prototype and follows the default spaces too (its previous mismatch caused a static-init-order segfault under eigen).
- **Backend-neutral fixes to previously hidden uBLAS couplings**: `explicit_builder.h` zeroed the (sparse-space) lumped-mass vector through the dense space; the power-iteration/Rayleigh eigenvalue solvers initialized vectors from `ublas::zero_vector`; `RandomInitializeUtility` hardcoded ublas types (now templated); `adaptive_residualbased_newton_raphson_strategy.h` called the free `WriteMatrixMarketMatrix` instead of the space one; ILU0/skyline-LU copied matrices via ublas iterators (now CSR-array loops); `scaling_solver.h` iterator typedefs are now deduced; the mortar-mapper and condition-number python bindings now use those classes' own (uBLAS) solver types.
- **Python surface under the eigen backend**: `Kernel.LinearAlgebraBackend()` reports the backend; `UblasSparseSpace` is always bound to the actual uBLAS space; `EigenSparseSpace`, `EigenCompressedMatrix` and `EigenVector` are additionally bound. A uBLAS-space `LinearSolver` base class (`UblasLinearSolver`) is registered so application solvers bound to uBLAS types (eigensystem/spectra) keep their pybind base.
- **Excluded from the eigen backend for now** (uBLAS-only sparse construction, gated + documented): `deflated_cg`, `PMultigridBuilderAndSolver`, `ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier`. The `LinearSolversApplication` direct-solver python tests skip under eigen (they hand-build uBLAS containers; coverage comes from the C++ backend parity tests).

Verification executed: full `KratosCoreFastSuite` and `LinearSolversApplication` python suite under both backends (identical pass/fail sets; the only failures are 6 pre-existing geometry/platform tests unrelated to linear algebra, failing identically in both modes), the uBLAS/Eigen builder-and-solve parity tests (CSR arrays, RHS and solution match), and a python smoke of the new bindings.
