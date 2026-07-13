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

// System includes

// External includes

// Project includes
#include "testing/testing.h"
#include "trilinos_space_experimental.h"
#include "tests/cpp_tests/trilinos_cpp_test_experimental_utilities.h"
#include "tests/cpp_tests/trilinos_fast_suite.h"
#include "custom_factories/trilinos_linear_solver_factory.h"
#include "external_includes/amgcl_mpi_solver.h"
#ifdef HAVE_BELOS
#include "external_includes/belos_solver.h"
#endif
#if defined(HAVE_BELOS) && defined(HAVE_MUELU)
#include "external_includes/muelu_solver.h"
#endif

namespace Kratos::Testing
{

// Define the types of spaces
using TrilinosSparseSpaceType = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;
using TrilinosLocalSpaceType = UblasSpace<double, Matrix, Vector>;

namespace
{

/**
 * @brief Solves a diagonally dominant SPD tridiagonal system whose solution is the ones vector
 * and checks the solution, using the given solver.
 */
template<class TSolverType>
void TestExperimentalIterativeSolver(TSolverType& rSolver, const double Tolerance = 1.0e-6)
{
    const auto& r_comm = Testing::GetDefaultDataCommunicator();
    const int size = 4 * r_comm.Size();

    // SPD tridiagonal matrix: diagonal(i) = 10 + i, off-diagonal (i, i +- 1) = -1
    auto p_A = TrilinosCPPTestExperimentalUtilities::GenerateDummySparseMatrix(r_comm, size, 10.0, true);

    // Reference solution: ones vector; RHS b = A * ones
    auto p_ref = TrilinosSparseSpaceType::CreateVector(p_A->getDomainMap());
    p_ref->putScalar(1.0);
    auto p_b = TrilinosSparseSpaceType::CreateVector(p_A->getRangeMap());
    TrilinosSparseSpaceType::Mult(*p_A, *p_ref, *p_b);

    // Initial guess: zeros
    auto p_x = TrilinosSparseSpaceType::CreateVector(p_A->getDomainMap());

    // Solve and check
    const bool converged = rSolver.Solve(*p_A, *p_x, *p_b);
    KRATOS_EXPECT_TRUE(converged);

    const auto data = p_x->getData(0);
    for (std::size_t i = 0; i < static_cast<std::size_t>(data.size()); ++i) {
        KRATOS_EXPECT_NEAR(static_cast<double>(data[i]), 1.0, Tolerance);
    }
}

} // namespace

KRATOS_TEST_CASE_IN_SUITE(TrilinosExperimentalAmgclMPISolver, KratosTrilinosApplicationMPITestSuite)
{
    Parameters parameters = Parameters(R"({
        "solver_type"   : "amgcl",
        "tolerance"     : 1.0e-12,
        "max_iteration" : 500,
        "verbosity"     : 0
    })");
    AmgclMPISolver<TrilinosSparseSpaceType, TrilinosLocalSpaceType> solver(parameters);
    TestExperimentalIterativeSolver(solver);
}

#ifdef HAVE_BELOS

KRATOS_TEST_CASE_IN_SUITE(TrilinosExperimentalBelosSolverCG, KratosTrilinosApplicationMPITestSuite)
{
    Parameters parameters = Parameters(R"({
        "solver_type"         : "cg",
        "preconditioner_type" : "jacobi",
        "tolerance"           : 1.0e-12,
        "max_iteration"       : 500
    })");
    BelosSolver<TrilinosSparseSpaceType, TrilinosLocalSpaceType> solver(parameters);
    TestExperimentalIterativeSolver(solver);
}

KRATOS_TEST_CASE_IN_SUITE(TrilinosExperimentalBelosSolverGMRES, KratosTrilinosApplicationMPITestSuite)
{
    Parameters parameters = Parameters(R"({
        "solver_type"         : "gmres",
        "preconditioner_type" : "ilut",
        "tolerance"           : 1.0e-12,
        "max_iteration"       : 500
    })");
    BelosSolver<TrilinosSparseSpaceType, TrilinosLocalSpaceType> solver(parameters);
    TestExperimentalIterativeSolver(solver);
}

KRATOS_TEST_CASE_IN_SUITE(TrilinosExperimentalBelosSolverBICGSTABNoPreconditioner, KratosTrilinosApplicationMPITestSuite)
{
    Parameters parameters = Parameters(R"({
        "solver_type"         : "bicgstab",
        "preconditioner_type" : "none",
        "tolerance"           : 1.0e-12,
        "max_iteration"       : 500
    })");
    BelosSolver<TrilinosSparseSpaceType, TrilinosLocalSpaceType> solver(parameters);
    TestExperimentalIterativeSolver(solver);
}

KRATOS_TEST_CASE_IN_SUITE(TrilinosExperimentalBelosSolverFromFactory, KratosTrilinosApplicationMPITestSuite)
{
    // The experimental factory must resolve the Aztec-compatible names to the Belos wrapper
    using FactoryType = LinearSolverFactory<TrilinosSparseSpaceType, TrilinosLocalSpaceType>;
    Parameters parameters = Parameters(R"({
        "solver_type" : "cg"
    })");
    auto p_solver = FactoryType().Create(parameters);
    KRATOS_EXPECT_NE(nullptr, p_solver.get());
    TestExperimentalIterativeSolver(*p_solver);
}

#endif // HAVE_BELOS

#if defined(HAVE_BELOS) && defined(HAVE_MUELU)

KRATOS_TEST_CASE_IN_SUITE(TrilinosExperimentalMueLuSolver, KratosTrilinosApplicationMPITestSuite)
{
    // Small test matrices: keep the hierarchy shallow so aggregation does not degenerate
    Parameters parameters = Parameters(R"({
        "solver_type"      : "multi_level",
        "krylov_type"      : "cg",
        "tolerance"        : 1.0e-12,
        "max_iteration"    : 500,
        "muelu_parameters" : {
            "max levels"       : 2,
            "coarse: max size" : 1000
        }
    })");
    MueLuSolver<TrilinosSparseSpaceType, TrilinosLocalSpaceType> solver(parameters);
    TestExperimentalIterativeSolver(solver);
}

#endif // HAVE_BELOS && HAVE_MUELU

} // namespace Kratos::Testing
