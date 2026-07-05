//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

// System includes
#include <cmath>

// External includes
#include <benchmark/benchmark.h>

// Project includes
#include "spaces/ublas_space.h"
#include "spaces/eigen_space.h"
#include "linear_solvers/cg_solver.h"
#include "utilities/parallel_utilities.h"

// Side-by-side performance comparison of the sparse linear-algebra backends
// (uBLAS vs Eigen). Both space implementations are always compiled, so a
// single binary benchmarks both under identical compiler flags — the
// KRATOS_LINEAR_ALGEBRA_BACKEND option only selects which one the default
// space aliases point to, and this file names both spaces explicitly.
//
// Every benchmark is registered twice through BENCHMARK_TEMPLATE, once per
// space, on identical data. The system matrix is a synthetic banded
// diagonally-dominant (SPD) matrix mimicking a FEM stencil, built by writing
// the CSR arrays directly (exactly as the builder-and-solvers do).
//
// The parallel behavior follows the usual Kratos shared-memory settings: run
// with OMP_NUM_THREADS=1 for serial numbers and higher values for the
// threaded ones.

namespace Kratos
{

namespace
{

using UblasSparse = TUblasSparseSpace<double>;
using EigenSparse = TEigenSparseSpace<double>;
using DenseSpace = TUblasDenseSpace<double>;

constexpr std::size_t BandHalfWidth = 4; // 9 entries per interior row, FEM-stencil-like

template <class TSpaceType>
struct BandedSystem
{
    typename TSpaceType::MatrixType A;
    typename TSpaceType::VectorType x;
    typename TSpaceType::VectorType y;
};

/// Builds a banded, diagonally dominant matrix by writing the CSR arrays
/// directly (the same access pattern the block builder-and-solver uses).
template <class TSpaceType>
BandedSystem<TSpaceType> MakeBandedSystem(const std::size_t Size)
{
    BandedSystem<TSpaceType> system;

    // Count the nonzeros
    std::size_t nnz = 0;
    for (std::size_t i = 0; i < Size; ++i) {
        const std::size_t begin = (i < BandHalfWidth) ? 0 : i - BandHalfWidth;
        const std::size_t end = (i + BandHalfWidth + 1 > Size) ? Size : i + BandHalfWidth + 1;
        nnz += end - begin;
    }

    system.A = typename TSpaceType::MatrixType(Size, Size, nnz);

    auto* row_indices = system.A.index1_data().begin();
    auto* col_indices = system.A.index2_data().begin();
    auto* values = system.A.value_data().begin();

    std::size_t counter = 0;
    row_indices[0] = 0;
    for (std::size_t i = 0; i < Size; ++i) {
        const std::size_t begin = (i < BandHalfWidth) ? 0 : i - BandHalfWidth;
        const std::size_t end = (i + BandHalfWidth + 1 > Size) ? Size : i + BandHalfWidth + 1;
        for (std::size_t j = begin; j < end; ++j) {
            col_indices[counter] = j;
            values[counter] = (i == j) ? 2.0 * BandHalfWidth + 1.0 : -0.5;
            ++counter;
        }
        row_indices[i + 1] = counter;
    }
    system.A.set_filled(Size + 1, nnz);

    system.x = typename TSpaceType::VectorType(Size);
    system.y = typename TSpaceType::VectorType(Size);
    for (std::size_t i = 0; i < Size; ++i) {
        system.x[i] = 1.0 + 0.001 * static_cast<double>(i % 17);
        system.y[i] = 0.0;
    }

    return system;
}

} // namespace

// --- Sparse matrix kernels -------------------------------------------------

template <class TSpaceType>
static void BM_SpMV(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    for (auto _ : rState) {
        TSpaceType::Mult(system.A, system.x, system.y);
        benchmark::DoNotOptimize(system.y[0]);
    }
}

template <class TSpaceType>
static void BM_TransposeSpMV(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    for (auto _ : rState) {
        TSpaceType::TransposeMult(system.A, system.x, system.y);
        benchmark::DoNotOptimize(system.y[0]);
    }
}

template <class TSpaceType>
static void BM_MatrixFrobeniusNorm(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    for (auto _ : rState) {
        benchmark::DoNotOptimize(TSpaceType::TwoNorm(system.A));
    }
}

template <class TSpaceType>
static void BM_SetToZeroMatrix(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    for (auto _ : rState) {
        TSpaceType::SetToZero(system.A);
        benchmark::DoNotOptimize(system.A.value_data().begin());
    }
}

// Graph construction the way the block builder-and-solver performs it:
// (rows, cols, nnz) construction plus direct CSR array filling.
template <class TSpaceType>
static void BM_CSRConstruction(benchmark::State& rState)
{
    for (auto _ : rState) {
        auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
        benchmark::DoNotOptimize(system.A.value_data().begin());
    }
}

// --- Vector kernels ----------------------------------------------------------

template <class TSpaceType>
static void BM_Dot(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    TSpaceType::Mult(system.A, system.x, system.y);
    for (auto _ : rState) {
        benchmark::DoNotOptimize(TSpaceType::Dot(system.x, system.y));
    }
}

template <class TSpaceType>
static void BM_VectorTwoNorm(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    for (auto _ : rState) {
        benchmark::DoNotOptimize(TSpaceType::TwoNorm(system.x));
    }
}

template <class TSpaceType>
static void BM_ScaleAndAdd(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    typename TSpaceType::VectorType z(rState.range(0));
    for (auto _ : rState) {
        TSpaceType::ScaleAndAdd(1.5, system.x, -0.5, system.y, z); // z = 1.5 x - 0.5 y
        benchmark::DoNotOptimize(z[0]);
    }
}

template <class TSpaceType>
static void BM_UnaliasedAdd(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));
    for (auto _ : rState) {
        TSpaceType::UnaliasedAdd(system.y, 0.001, system.x); // y += 0.001 x
        benchmark::DoNotOptimize(system.y[0]);
    }
}

// --- End-to-end iterative solve ---------------------------------------------
// CG chains SpMV, Dot and the vector updates through the space, so it is a
// representative aggregate of the backend performance. The matrix and the
// starting point are identical for both backends, so the iteration counts are
// identical too.

template <class TSpaceType>
static void BM_CGSolve(benchmark::State& rState)
{
    auto system = MakeBandedSystem<TSpaceType>(rState.range(0));

    typename TSpaceType::VectorType b(rState.range(0));
    TSpaceType::Mult(system.A, system.x, b); // manufactured solution: x

    for (auto _ : rState) {
        rState.PauseTiming();
        typename TSpaceType::VectorType solution(rState.range(0));
        TSpaceType::SetToZero(solution);
        typename TSpaceType::VectorType rhs = b;
        CGSolver<TSpaceType, DenseSpace> solver(1e-10, 500);
        rState.ResumeTiming();

        solver.Solve(system.A, solution, rhs);
        benchmark::DoNotOptimize(solution[0]);
    }
}

// --- Registration -------------------------------------------------------------

#define KRATOS_REGISTER_BACKEND_BENCHMARK(name)                                       \
    BENCHMARK_TEMPLATE(name, UblasSparse)->Name(#name "/ublas")->Arg(1<<14)->Arg(1<<20); \
    BENCHMARK_TEMPLATE(name, EigenSparse)->Name(#name "/eigen")->Arg(1<<14)->Arg(1<<20);

KRATOS_REGISTER_BACKEND_BENCHMARK(BM_SpMV)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_TransposeSpMV)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_MatrixFrobeniusNorm)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_SetToZeroMatrix)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_CSRConstruction)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_Dot)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_VectorTwoNorm)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_ScaleAndAdd)
KRATOS_REGISTER_BACKEND_BENCHMARK(BM_UnaliasedAdd)

BENCHMARK_TEMPLATE(BM_CGSolve, UblasSparse)->Name("BM_CGSolve/ublas")->Arg(1<<14)->Arg(1<<18)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_CGSolve, EigenSparse)->Name("BM_CGSolve/eigen")->Arg(1<<14)->Arg(1<<18)->Unit(benchmark::kMillisecond);

#undef KRATOS_REGISTER_BACKEND_BENCHMARK

} // namespace Kratos

BENCHMARK_MAIN();
