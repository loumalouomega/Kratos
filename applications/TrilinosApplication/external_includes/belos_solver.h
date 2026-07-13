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
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosLinearProblem.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_MultiVector.hpp>
#ifdef HAVE_IFPACK2
#include <Ifpack2_Factory.hpp>
#endif

// Project includes
#include "linear_solvers/linear_solver.h"
#include "custom_utilities/trilinos_solver_utilities.h"
#include "trilinos_application.h"

namespace Kratos
{
///@name Kratos Classes
///@{

/// Wrapper for the Trilinos-Belos iterative solvers (Tpetra-era replacement for AztecOO).
/** Belos provides next-generation iterative linear solvers (CG, GMRES, BiCGStab, ...) operating
 * on Tpetra objects, optionally preconditioned with Ifpack2 incomplete factorizations/relaxations.
 * https://trilinos.github.io/belos.html
 * @note This solver only works with the Tpetra-based TrilinosSpaceExperimental.
*/
template< class TSparseSpaceType, class TDenseSpaceType,
          class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType> >
class BelosSolver : public LinearSolver< TSparseSpaceType,
    TDenseSpaceType, TReordererType>
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of BelosSolver
    KRATOS_CLASS_POINTER_DEFINITION(BelosSolver);

    static_assert(TSparseSpaceType::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA,
                  "BelosSolver requires the Tpetra-based experimental Trilinos space");

    using SparseMatrixType = typename TSparseSpaceType::MatrixType;

    using VectorType = typename TSparseSpaceType::VectorType;

    using DenseMatrixType = typename TDenseSpaceType::MatrixType;

    /// Tpetra scalar/ordinal definitions (taken from the space)
    using ST = typename TSparseSpaceType::ST;
    using LO = typename TSparseSpaceType::LO;
    using GO = typename TSparseSpaceType::GO;
    using NT = typename TSparseSpaceType::NT;

    /// Belos operates on the Tpetra base classes of the FE-matrix/vector types
    using MultiVectorType = Tpetra::MultiVector<ST, LO, GO, NT>;
    using OperatorType = Tpetra::Operator<ST, LO, GO, NT>;
    using RowMatrixType = Tpetra::RowMatrix<ST, LO, GO, NT>;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Constructor with Parameters.
    BelosSolver(Parameters settings)
    {
        Parameters default_settings( R"({
            "solver_type"                  : "gmres",
            "tolerance"                    : 1.0e-9,
            "max_iteration"                : 500,
            "gmres_krylov_space_dimension" : 100,
            "preconditioner_type"          : "jacobi",
            "verbosity"                    : 0,
            "trilinos_parameters"          : { }
        }  )" );

        settings.ValidateAndAssignDefaults(default_settings);

        // Map from Kratos solver names to Belos internal names
        const std::unordered_map<std::string, std::string> kratos_to_belos_names = {
            {"cg",       "CG"},
            {"bicgstab", "BICGSTAB"},
            {"gmres",    "GMRES"},
            {"belos",    "GMRES"}
        };

        const std::string solver_type = settings["solver_type"].GetString();
        const auto iter_belos_name = kratos_to_belos_names.find(solver_type);
        KRATOS_ERROR_IF(iter_belos_name == kratos_to_belos_names.end())
            << "The solver type specified: \"" << solver_type << "\" is not supported by the Belos wrapper";
        mSolverName = iter_belos_name->second;

        // Map from Kratos preconditioner names to Ifpack2 internal names
        const std::unordered_map<std::string, std::string> kratos_to_ifpack2_names = {
            {"none",     ""},
            {"jacobi",   "RELAXATION"},
            {"diagonal", "RELAXATION"},
            {"ilut",     "ILUT"},
            {"ilu0",     "RILUK"},
            {"riluk",    "RILUK"},
            {"schwarz",  "SCHWARZ"}
        };

        const std::string preconditioner_type = settings["preconditioner_type"].GetString();
        const auto iter_ifpack2_name = kratos_to_ifpack2_names.find(preconditioner_type);
        KRATOS_ERROR_IF(iter_ifpack2_name == kratos_to_ifpack2_names.end())
            << "The preconditioner type specified: \"" << preconditioner_type << "\" is not supported by the Belos wrapper";
        mPreconditionerName = iter_ifpack2_name->second;
#ifndef HAVE_IFPACK2
        KRATOS_ERROR_IF_NOT(mPreconditionerName.empty())
            << "The current compilation of Trilinos does not include Ifpack2, only \"none\" is available as preconditioner" << std::endl;
#endif

        // Fill the Belos parameter list
        mParameterList.set("Convergence Tolerance", settings["tolerance"].GetDouble());
        mParameterList.set("Maximum Iterations", settings["max_iteration"].GetInt());
        if (mSolverName == "GMRES") {
            mParameterList.set("Num Blocks", settings["gmres_krylov_space_dimension"].GetInt());
        }
        if (settings["verbosity"].GetInt() > 0) {
            mParameterList.set("Verbosity", Belos::Errors | Belos::Warnings | Belos::IterationDetails | Belos::FinalSummary);
            mParameterList.set("Output Frequency", 1);
        }

        // Assign the user-provided parameters, which may contain parameters IN TRILINOS INTERNAL FORMAT
        // NOTE: this will OVERWRITE PREVIOUS SETTINGS TO GIVE FULL CONTROL
        TrilinosSolverUtilities::SetTeuchosParameters(settings["trilinos_parameters"], mParameterList);
    }

    /// Constructor with solver-name and Teuchos::ParameterList.
    BelosSolver(
        const std::string& SolverName,
        const std::string& PreconditionerName,
        Teuchos::ParameterList& rParameterList)
    {
        mParameterList = rParameterList;
        mSolverName = SolverName;
        mPreconditionerName = PreconditionerName;
    }

    /// Copy constructor.
    BelosSolver(const BelosSolver& Other) = delete;

    /// Destructor.
    ~BelosSolver() override = default;

    ///@}
    ///@name Operators
    ///@{

    /// Assignment operator.
    BelosSolver& operator=(const BelosSolver& Other) = delete;

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

        using Teuchos::RCP;
        using Teuchos::rcp;

        // Belos is missing specializations for FE-matrices and vectors,
        // hence here we need to use the corresponding base classes
        RCP<OperatorType> rcp_A = rcp(dynamic_cast<OperatorType*>(&rA), false);
        RCP<MultiVectorType> rcp_X = rcp(dynamic_cast<MultiVectorType*>(&rX), false);
        RCP<MultiVectorType> rcp_B = rcp(dynamic_cast<MultiVectorType*>(&rB), false);

        auto p_problem = rcp(new Belos::LinearProblem<ST, MultiVectorType, OperatorType>(rcp_A, rcp_X, rcp_B));

#ifdef HAVE_IFPACK2
        if (!mPreconditionerName.empty()) {
            RCP<const RowMatrixType> rcp_row_A = rcp(dynamic_cast<const RowMatrixType*>(&rA), false);
            auto p_preconditioner = Ifpack2::Factory::create<RowMatrixType>(mPreconditionerName, rcp_row_A);
            p_preconditioner->initialize();
            p_preconditioner->compute();
            p_problem->setRightPrec(p_preconditioner);
        }
#endif

        KRATOS_ERROR_IF_NOT(p_problem->setProblem()) << "Belos::LinearProblem::setProblem() failed" << std::endl;

        Belos::SolverFactory<ST, MultiVectorType, OperatorType> factory;
        RCP<Teuchos::ParameterList> rcp_params = rcp(new Teuchos::ParameterList(mParameterList));
        auto p_solver = factory.create(mSolverName, rcp_params);
        p_solver->setProblem(p_problem);

        const Belos::ReturnType return_type = p_solver->solve();

        KRATOS_WARNING_IF("BelosSolver", return_type != Belos::Converged)
            << "Belos solver \"" << mSolverName << "\" did not converge within "
            << p_solver->getNumIters() << " iterations" << std::endl;

        return return_type == Belos::Converged;

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
        rOStream << "Trilinos Belos-Solver (" << mSolverName << ")";
    }

    ///@}

private:
    ///@name Member Variables
    ///@{

    Teuchos::ParameterList mParameterList;
    std::string mSolverName;
    std::string mPreconditionerName;

    ///@}

}; // Class BelosSolver

/// output stream function
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream,
                                  const BelosSolver<TSparseSpaceType,
                                  TDenseSpaceType, TReordererType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

}  // namespace Kratos.
