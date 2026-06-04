//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#ifdef KRATOS_TRILINOS_USE_MUMPS_DIRECTLY

// System includes

// External includes

// Project includes
#include "testing/testing.h"
#include "trilinos_space.h"
#include "spaces/ublas_space.h"
#include "tests/cpp_tests/trilinos_cpp_test_utilities.h"
#include "tests/cpp_tests/trilinos_fast_suite.h"
#include "custom_solvers/trilinos_mumps_solver.h"

namespace Kratos::Testing
{

using TrilinosSparseSpaceType = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;
using TrilinosLocalSpaceType  = UblasSpace<double, Matrix, Vector>;
using SparseMatrixType        = TrilinosSparseSpaceType::MatrixType;
using VectorType              = TrilinosSparseSpaceType::VectorType;
using MumpsSolverType         = TrilinosMumpsSolver<TrilinosSparseSpaceType, TrilinosLocalSpaceType>;

/// Computes ||A*x - b|| / ||b||. A and b are not modified.
static double RelativeResidual(SparseMatrixType& rA, VectorType& rX, VectorType& rB)
{
    VectorType tmp(rX.Map());
    tmp.PutScalar(0.0);
    rA.Multiply(false, rX, tmp);    // tmp  = A*x
    tmp.Update(1.0, rB, -1.0);     // tmp  = b - A*x
    double residual_norm, b_norm;
    tmp.Norm2(&residual_norm);
    rB.Norm2(&b_norm);
    return (b_norm > 0.0) ? residual_norm / b_norm : residual_norm;
}

/**
 * Solve a distributed diagonal SPD system in unsymmetric mode (sym=0).
 *
 * Matrix : A(i,i) = 1 + i  (global 0-based index, Offset=1)
 * RHS    : b[i]   = 1 + i
 * Exact solution : x[i] = 1.0  for all i
 */
KRATOS_DISTRIBUTED_TEST_CASE_IN_SUITE(TrilinosMumpsSolverDiagonalSystem, KratosTrilinosApplicationMPITestSuite)
{
    const auto& r_comm = Testing::GetDefaultDataCommunicator();
    const int size = 4 * r_comm.Size();

    auto A = TrilinosCPPTestUtilities::GenerateDummySparseMatrix(r_comm, size, 1.0);
    auto b = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 1.0);
    auto x = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 0.0);

    MumpsSolverType solver(Parameters(R"({"solver_type":"mumps_direct","sym":0})"));
    KRATOS_EXPECT_TRUE(solver.Solve(A, x, b));

    // x[i] must be 1.0 on every local entry
    const int n_local = x.Map().NumMyElements();
    for (int i = 0; i < n_local; ++i) {
        KRATOS_EXPECT_NEAR(x[0][i], 1.0, 1e-10);
    }
}

/**
 * Solve the same diagonal SPD system using MUMPS symmetric mode (sym=1).
 *
 * MUMPS only receives the lower triangular entries; the factorization uses
 * symmetry to reconstruct the full matrix.
 */
KRATOS_DISTRIBUTED_TEST_CASE_IN_SUITE(TrilinosMumpsSolverSymmetricMode, KratosTrilinosApplicationMPITestSuite)
{
    const auto& r_comm = Testing::GetDefaultDataCommunicator();
    const int size = 4 * r_comm.Size();

    auto A = TrilinosCPPTestUtilities::GenerateDummySparseMatrix(r_comm, size, 1.0);
    auto b = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 1.0);
    auto x = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 0.0);

    MumpsSolverType solver(Parameters(R"({"solver_type":"mumps_direct","sym":1})"));
    KRATOS_EXPECT_TRUE(solver.Solve(A, x, b));

    const int n_local = x.Map().NumMyElements();
    for (int i = 0; i < n_local; ++i) {
        KRATOS_EXPECT_NEAR(x[0][i], 1.0, 1e-10);
    }
}

/**
 * Solve a tridiagonal diagonally-dominant system in unsymmetric mode and
 * verify the residual norm.
 *
 * Matrix : A(i,i) = 2 + i,  A(i,i±1) = -1  (Offset=2, off-diagonal)
 * RHS    : b[i]   = 1 + i
 * The exact solution is not trivial, but the direct solver must satisfy
 * ||A*x - b|| / ||b|| < 1e-10.
 */
KRATOS_DISTRIBUTED_TEST_CASE_IN_SUITE(TrilinosMumpsSolverTridiagonalResidual, KratosTrilinosApplicationMPITestSuite)
{
    const auto& r_comm = Testing::GetDefaultDataCommunicator();
    const int size = 4 * r_comm.Size();

    auto A = TrilinosCPPTestUtilities::GenerateDummySparseMatrix(r_comm, size, 2.0, true);
    auto b = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 1.0);
    auto x = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 0.0);

    MumpsSolverType solver(Parameters(R"({"solver_type":"mumps_direct","sym":0})"));
    KRATOS_EXPECT_TRUE(solver.Solve(A, x, b));

    KRATOS_EXPECT_LT(RelativeResidual(A, x, b), 1e-10);
}

/**
 * Call Solve twice on the same matrix.
 *
 * After the first call, TrilinosMumpsSolver sets mReanalyze = false so the
 * second call reuses the symbolic factorization (MUMPS job=1 is skipped).
 * Both solutions must satisfy x[i] = 1.0.
 */
KRATOS_DISTRIBUTED_TEST_CASE_IN_SUITE(TrilinosMumpsSolverRepeatSolve, KratosTrilinosApplicationMPITestSuite)
{
    const auto& r_comm = Testing::GetDefaultDataCommunicator();
    const int size = 4 * r_comm.Size();

    auto A  = TrilinosCPPTestUtilities::GenerateDummySparseMatrix(r_comm, size, 1.0);
    auto b1 = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 1.0);
    auto b2 = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 1.0);
    auto x1 = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 0.0);
    auto x2 = TrilinosCPPTestUtilities::GenerateDummySparseVector(r_comm, size, 0.0);

    MumpsSolverType solver(Parameters(R"({"solver_type":"mumps_direct","sym":0})"));

    KRATOS_EXPECT_TRUE(solver.Solve(A, x1, b1)); // analysis + factorize + solve
    KRATOS_EXPECT_TRUE(solver.Solve(A, x2, b2)); // factorize + solve  (analysis reused)

    const int n_local = x1.Map().NumMyElements();
    for (int i = 0; i < n_local; ++i) {
        KRATOS_EXPECT_NEAR(x1[0][i], 1.0, 1e-10);
        KRATOS_EXPECT_NEAR(x2[0][i], 1.0, 1e-10);
    }
}

} // namespace Kratos::Testing

#endif // KRATOS_TRILINOS_USE_MUMPS_DIRECTLY
