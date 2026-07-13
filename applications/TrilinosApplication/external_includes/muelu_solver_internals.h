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

#pragma once

// System includes
#include <string>

// External includes
// NOTE: Only Tpetra/Teuchos core headers here. The MueLu (and Belos) headers are deliberately
// confined to muelu_solver_internals.cpp: the MueLu include chain pulls MatrixMarket_Tpetra.hpp,
// whose C-linkage MatrixMarket readers (mmio_Tpetra.h) conflict with Kratos' includes/mmio.h,
// so MueLu headers can never share a translation unit with Kratos headers that use it (e.g.
// spaces/ublas_space.h via includes/matrix_market_interface.h).
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Teuchos_ParameterList.hpp>

// Project includes

namespace Kratos
{
namespace MueLuSolverInternals
{

/**
 * @brief Solves the system with a Belos Krylov solver right-preconditioned by a MueLu multigrid hierarchy.
 * @param rA The system matrix (any Tpetra::CrsMatrix, e.g. the FECrsMatrix of the experimental space)
 * @param rX The solution vector (initial guess on input)
 * @param rB The right hand side vector
 * @param rBelosParameters The Belos solver parameter list
 * @param rMueLuParameters The MueLu hierarchy parameter list
 * @param rSolverName The Belos solver name ("CG", "GMRES", "BICGSTAB")
 * @param rIterationCount On output, the number of iterations performed
 * @return True if the solver converged
 */
bool SolveWithMueLuPreconditionedBelos(
    Tpetra::CrsMatrix<>& rA,
    Tpetra::MultiVector<>& rX,
    Tpetra::MultiVector<>& rB,
    const Teuchos::ParameterList& rBelosParameters,
    const Teuchos::ParameterList& rMueLuParameters,
    const std::string& rSolverName,
    int& rIterationCount);

} // namespace MueLuSolverInternals
} // namespace Kratos
