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
//  IMPORTANT: This translation unit must NOT include any Kratos header that (directly or
//  indirectly) includes includes/mmio.h (e.g. spaces/ublas_space.h): the MueLu include chain
//  pulls MatrixMarket_Tpetra.hpp whose C-linkage readers conflict with it. See
//  muelu_solver_internals.h for details.
//

#if defined(HAVE_TPETRA) && defined(HAVE_BELOS) && defined(HAVE_MUELU)

// System includes
#include <stdexcept>

// External includes
#include <Teuchos_RCP.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosLinearProblem.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

// Project includes
#include "muelu_solver_internals.h"

namespace Kratos
{
namespace MueLuSolverInternals
{

bool SolveWithMueLuPreconditionedBelos(
    Tpetra::CrsMatrix<>& rA,
    Tpetra::MultiVector<>& rX,
    Tpetra::MultiVector<>& rB,
    const Teuchos::ParameterList& rBelosParameters,
    const Teuchos::ParameterList& rMueLuParameters,
    const std::string& rSolverName,
    int& rIterationCount)
{
    using Teuchos::RCP;
    using Teuchos::rcp;

    using MatrixType = Tpetra::CrsMatrix<>;
    using ST = typename MatrixType::scalar_type;
    using LO = typename MatrixType::local_ordinal_type;
    using GO = typename MatrixType::global_ordinal_type;
    using NT = typename MatrixType::node_type;
    using MultiVectorType = Tpetra::MultiVector<ST, LO, GO, NT>;
    using OperatorType = Tpetra::Operator<ST, LO, GO, NT>;

    RCP<OperatorType> rcp_A = Teuchos::rcpFromRef(static_cast<OperatorType&>(rA));
    RCP<MultiVectorType> rcp_X = Teuchos::rcpFromRef(rX);
    RCP<MultiVectorType> rcp_B = Teuchos::rcpFromRef(rB);

    auto p_problem = rcp(new Belos::LinearProblem<ST, MultiVectorType, OperatorType>(rcp_A, rcp_X, rcp_B));

    // Build the MueLu multigrid hierarchy as right preconditioner
    Teuchos::ParameterList muelu_parameters(rMueLuParameters);
    RCP<MueLu::TpetraOperator<ST, LO, GO, NT>> p_preconditioner =
        MueLu::CreateTpetraPreconditioner<ST, LO, GO, NT>(rcp_A, muelu_parameters);
    p_problem->setRightPrec(p_preconditioner);

    if (!p_problem->setProblem()) {
        throw std::runtime_error("MueLuSolver: Belos::LinearProblem::setProblem() failed");
    }

    Belos::SolverFactory<ST, MultiVectorType, OperatorType> factory;
    RCP<Teuchos::ParameterList> rcp_params = rcp(new Teuchos::ParameterList(rBelosParameters));
    auto p_solver = factory.create(rSolverName, rcp_params);
    p_solver->setProblem(p_problem);

    const Belos::ReturnType return_type = p_solver->solve();
    rIterationCount = static_cast<int>(p_solver->getNumIters());

    return return_type == Belos::Converged;
}

} // namespace MueLuSolverInternals
} // namespace Kratos

#endif // HAVE_TPETRA && HAVE_BELOS && HAVE_MUELU
