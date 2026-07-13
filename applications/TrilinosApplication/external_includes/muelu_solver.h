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
#include <unordered_map>

// External includes
#include <Teuchos_ParameterList.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>

// Project includes
#include "linear_solvers/linear_solver.h"
#include "custom_utilities/trilinos_solver_utilities.h"
#include "external_includes/muelu_solver_internals.h"
#include "trilinos_application.h"

namespace Kratos
{
///@name Kratos Classes
///@{

/// Wrapper for the Trilinos-MueLu algebraic multigrid preconditioned Krylov solvers.
/** MueLu is the Tpetra-era algebraic multigrid package of Trilinos (successor of ML).
 * The multigrid hierarchy is used as a right preconditioner of a Belos Krylov solver
 * (CG/GMRES/BiCGStab), replicating what the Epetra MultiLevelSolver does with AztecOO+ML.
 * https://trilinos.github.io/muelu.html
 * @note This solver only works with the Tpetra-based TrilinosSpaceExperimental.
*/
template< class TSparseSpaceType, class TDenseSpaceType,
          class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType> >
class MueLuSolver : public LinearSolver< TSparseSpaceType,
    TDenseSpaceType, TReordererType>
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of MueLuSolver
    KRATOS_CLASS_POINTER_DEFINITION(MueLuSolver);

    static_assert(TSparseSpaceType::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA,
                  "MueLuSolver requires the Tpetra-based experimental Trilinos space");

    using SparseMatrixType = typename TSparseSpaceType::MatrixType;

    using VectorType = typename TSparseSpaceType::VectorType;

    using DenseMatrixType = typename TDenseSpaceType::MatrixType;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Constructor with Parameters.
    MueLuSolver(Parameters settings)
    {
        Parameters default_settings( R"({
            "solver_type"                  : "multi_level",
            "krylov_type"                  : "cg",
            "tolerance"                    : 1.0e-9,
            "max_iteration"                : 500,
            "gmres_krylov_space_dimension" : 100,
            "symmetric"                    : true,
            "verbosity"                    : 0,
            "trilinos_parameters"          : { },
            "muelu_parameters"             : { }
        }  )" );

        settings.ValidateAndAssignDefaults(default_settings);

        // Map from Kratos Krylov solver names to Belos internal names
        const std::unordered_map<std::string, std::string> kratos_to_belos_names = {
            {"cg",       "CG"},
            {"bicgstab", "BICGSTAB"},
            {"gmres",    "GMRES"}
        };

        const std::string krylov_type = settings["krylov_type"].GetString();
        const auto iter_belos_name = kratos_to_belos_names.find(krylov_type);
        KRATOS_ERROR_IF(iter_belos_name == kratos_to_belos_names.end())
            << "The Krylov type specified: \"" << krylov_type << "\" is not supported by the MueLu wrapper";
        mSolverName = iter_belos_name->second;

        // Fill the Belos parameter list
        mBelosParameterList.set("Convergence Tolerance", settings["tolerance"].GetDouble());
        mBelosParameterList.set("Maximum Iterations", settings["max_iteration"].GetInt());
        if (mSolverName == "GMRES") {
            mBelosParameterList.set("Num Blocks", settings["gmres_krylov_space_dimension"].GetInt());
        }
        if (settings["verbosity"].GetInt() > 0) {
            mBelosParameterList.set("Verbosity", Belos::Errors | Belos::Warnings | Belos::IterationDetails | Belos::FinalSummary);
            mBelosParameterList.set("Output Frequency", 1);
            mMueLuParameterList.set("verbosity", std::string("medium"));
        }

        // Default MueLu settings: smoothed aggregation for SPD systems, unsmoothed otherwise
        mMueLuParameterList.set("multigrid algorithm", settings["symmetric"].GetBool() ? std::string("sa") : std::string("unsmoothed"));

        // Assign the user-provided parameters, which may contain parameters IN TRILINOS INTERNAL FORMAT
        // NOTE: this will OVERWRITE PREVIOUS SETTINGS TO GIVE FULL CONTROL
        TrilinosSolverUtilities::SetTeuchosParameters(settings["trilinos_parameters"], mBelosParameterList);
        TrilinosSolverUtilities::SetTeuchosParameters(settings["muelu_parameters"], mMueLuParameterList);
    }

    /// Copy constructor.
    MueLuSolver(const MueLuSolver& Other) = delete;

    /// Destructor.
    ~MueLuSolver() override = default;

    ///@}
    ///@name Operators
    ///@{

    /// Assignment operator.
    MueLuSolver& operator=(const MueLuSolver& Other) = delete;

    ///@}
    ///@name Operations
    ///@{

    /**
     * Normal solve method.
     * Solves the linear system Ax=b and puts the result on SystemVector& rX.
     * rX is also the initial guess for iterative methods.
     * @param rA. System matrix
     * @param rX. Solution vector.
     * @param rB. Right hand side vector.
     */
    bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB) override
    {
        KRATOS_TRY

        // The actual solve lives in muelu_solver_internals.cpp: the MueLu headers cannot be
        // included here (see muelu_solver_internals.h for the rationale). The FE matrix and
        // vector bind to their Tpetra::CrsMatrix / Tpetra::MultiVector base classes.
        int iteration_count = 0;
        const bool converged = MueLuSolverInternals::SolveWithMueLuPreconditionedBelos(
            rA, rX, rB, mBelosParameterList, mMueLuParameterList, mSolverName, iteration_count);

        KRATOS_WARNING_IF("MueLuSolver", !converged)
            << "MueLu-preconditioned Belos solver \"" << mSolverName << "\" did not converge within "
            << iteration_count << " iterations" << std::endl;

        return converged;

        KRATOS_CATCH("");
    }

    /**
     * Multi solve method for solving a set of linear systems with same coefficient matrix.
     * @param rA. System matrix
     * @param rX. Solution vector.
     * @param rB. Right hand side vector.
     */
    bool Solve(SparseMatrixType& rA, DenseMatrixType& rX, DenseMatrixType& rB) override
    {
        return false;
    }

    ///@}
    ///@name Input and output
    ///@{

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "Trilinos MueLu-Solver (" << mSolverName << ")";
    }

    ///@}

private:
    ///@name Member Variables
    ///@{

    Teuchos::ParameterList mBelosParameterList;
    Teuchos::ParameterList mMueLuParameterList;
    std::string mSolverName;

    ///@}

}; // Class MueLuSolver

/// output stream function
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream,
                                  const MueLuSolver<TSparseSpaceType,
                                  TDenseSpaceType, TReordererType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

}  // namespace Kratos.
