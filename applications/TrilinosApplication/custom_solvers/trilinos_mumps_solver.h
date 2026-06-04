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

#ifdef KRATOS_TRILINOS_USE_MUMPS_DIRECTLY

// System includes
#include <vector>

// External includes
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Vector.h>
#include <mpi.h>
#include <dmumps_c.h>

// Project includes
#include "includes/define.h"
#include "includes/kratos_parameters.h"
#include "linear_solvers/linear_solver.h"

namespace Kratos
{
///@name Kratos Classes
///@{

/// Direct solver that calls MUMPS via its native C API, bypassing Amesos/Amesos2.
/** The Epetra_FECrsMatrix entries are extracted in COO format using distributed
 *  assembled input (ICNTL(18)=3), so no global gather of the matrix is required.
 *  The right-hand side is gathered to rank 0 before the solve and the solution
 *  is scattered back to all ranks afterwards.
 */
template<class TSparseSpaceType, class TDenseSpaceType,
         class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType>>
class TrilinosMumpsSolver
    : public LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType>
{
public:
    ///@name Type Definitions
    ///@{

    using BaseType         = LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType>;
    using SparseMatrixType = typename TSparseSpaceType::MatrixType;   // Epetra_FECrsMatrix
    using VectorType       = typename TSparseSpaceType::VectorType;   // Epetra_FEVector
    using DenseMatrixType  = typename TDenseSpaceType::MatrixType;

    KRATOS_CLASS_POINTER_DEFINITION(TrilinosMumpsSolver);

    ///@}
    ///@name Life Cycle
    ///@{

    explicit TrilinosMumpsSolver(Parameters settings = Parameters(R"({})"))
    {
        Parameters default_settings(R"({
            "solver_type"                : "mumps_direct",
            "sym"                        : 0,
            "verbosity"                  : 0,
            "ordering"                   : 0,
            "iterative_refinement_steps" : 0,
            "out_of_core"                : 0
        })");
        settings.ValidateAndAssignDefaults(default_settings);

        mSym                 = settings["sym"].GetInt();
        mVerbosity           = settings["verbosity"].GetInt();
        mOrdering            = settings["ordering"].GetInt();
        mIterativeRefinement = settings["iterative_refinement_steps"].GetInt();
        mOutOfCore           = settings["out_of_core"].GetInt();
    }

    TrilinosMumpsSolver(const TrilinosMumpsSolver&) = delete;
    TrilinosMumpsSolver& operator=(const TrilinosMumpsSolver&) = delete;

    ~TrilinosMumpsSolver() override
    {
        if (mInitialized) {
            FinalizeMumps();
        }
    }

    ///@}
    ///@name Operations
    ///@{

    /**
     * Solves the linear system Ax=b using MUMPS directly.
     * @param rA Distributed sparse system matrix (Epetra_FECrsMatrix).
     * @param rX Solution vector (distributed). Overwritten with the solution.
     * @param rB Right-hand side vector (distributed). Not modified.
     */
    bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB) override
    {
        KRATOS_TRY

        const auto& r_epetra_comm = dynamic_cast<const Epetra_MpiComm&>(rA.Comm());
        MPI_Comm mpi_comm         = r_epetra_comm.Comm();
        const int rank            = r_epetra_comm.MyPID();
        const int n_global        = rA.NumGlobalRows();

        if (!mInitialized) {
            InitializeMumps(mpi_comm);
        }

        mId.n = n_global;

        // Extract local COO data (no copy of the full sparsity pattern)
        std::vector<MUMPS_INT> irn_loc, jcn_loc;
        std::vector<double>    a_loc;
        ExtractCOO(rA, irn_loc, jcn_loc, a_loc);

        // Distributed assembled matrix input (ICNTL(18) = 3)
        mId.nz_loc  = static_cast<MUMPS_INT>(irn_loc.size());
        mId.irn_loc = irn_loc.data();
        mId.jcn_loc = jcn_loc.data();
        mId.a_loc   = a_loc.data();

        // Symbolic analysis — only on first call or when forced
        if (mReanalyze) {
            mId.job = 1;
            dmumps_c(&mId);
            KRATOS_ERROR_IF(mId.infog[0] != 0)
                << "MUMPS analysis phase failed with error code " << mId.infog[0]
                << " (infog[1] = " << mId.infog[1] << ")" << std::endl;
            mReanalyze = false;
        }

        // Numeric factorization
        mId.job = 2;
        dmumps_c(&mId);
        KRATOS_ERROR_IF(mId.infog[0] != 0)
            << "MUMPS factorization phase failed with error code " << mId.infog[0]
            << " (infog[1] = " << mId.infog[1] << ")" << std::endl;

        // Gather RHS to rank 0 via a serial (all-on-root) map
        // Serial map: rank 0 owns all n_global elements, others own none
        Epetra_Map serial_map(n_global, rank == 0 ? n_global : 0, 0, r_epetra_comm);
        Epetra_Vector b_serial(serial_map);
        {
            Epetra_Import rhs_importer(serial_map, rB.Map());
            b_serial.Import(rB, rhs_importer, Insert);
        }

        // Solve — rank 0 provides the dense RHS; solution overwrites it in-place
        mId.job  = 3;
        mId.nrhs = 1;
        mId.lrhs = n_global;
        mId.rhs  = (rank == 0) ? b_serial.Values() : nullptr;
        dmumps_c(&mId);
        KRATOS_ERROR_IF(mId.infog[0] != 0)
            << "MUMPS solve phase failed with error code " << mId.infog[0]
            << " (infog[1] = " << mId.infog[1] << ")" << std::endl;

        // Scatter solution (now in b_serial on rank 0) back to distributed rX
        {
            Epetra_Import x_importer(rX.Map(), serial_map);
            rX.Import(b_serial, x_importer, Insert);
        }

        return true;

        KRATOS_CATCH("")
    }

    bool Solve(SparseMatrixType& rA, DenseMatrixType& rX, DenseMatrixType& rB) override
    {
        return false;
    }

    ///@}
    ///@name Input and Output
    ///@{

    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "TrilinosMumpsSolver"
                 << " [sym=" << mSym
                 << ", ordering=" << mOrdering
                 << ", iterative_refinement=" << mIterativeRefinement
                 << ", out_of_core=" << mOutOfCore
                 << ", verbosity=" << mVerbosity << "]";
        KRATOS_INFO("TrilinosMumpsSolver")
            << "sym=" << mSym
            << " ordering=" << mOrdering
            << " iterative_refinement=" << mIterativeRefinement
            << " out_of_core=" << mOutOfCore
            << " verbosity=" << mVerbosity << std::endl;
    }

    ///@}
    ///@name Static Methods
    ///@{

    static Parameters GetDefaultParameters()
    {
        return Parameters(R"({
            "solver_type"                : "mumps_direct",
            "sym"                        : 0,
            "verbosity"                  : 0,
            "ordering"                   : 0,
            "iterative_refinement_steps" : 0,
            "out_of_core"                : 0
        })");
    }

    ///@}

private:
    ///@name Private Operations
    ///@{

    void InitializeMumps(MPI_Comm mpi_comm)
    {
        mId.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(mpi_comm);
        mId.job = -1;   // initialize
        mId.par = 1;    // host processor participates in factorization
        mId.sym = mSym;
        dmumps_c(&mId);

        // ICNTL(18) = 3: distributed assembled input — must be set before analysis
        mId.icntl[17] = 3;

        if (mVerbosity == 0) {
            mId.icntl[0] = -1;  // error messages: suppressed
            mId.icntl[1] = -1;  // diagnostic messages: suppressed
            mId.icntl[2] = -1;  // global information: suppressed
            mId.icntl[3] = 0;   // no printing level
        } else {
            mId.icntl[0] = 6;   // errors to stdout
            mId.icntl[1] = 6;   // diagnostics to stdout
            mId.icntl[2] = 6;   // global info to stdout
            mId.icntl[3] = (mVerbosity > 1) ? 2 : 1;
        }

        mId.icntl[6]  = mOrdering;           // ICNTL(7): ordering
        mId.icntl[9]  = mIterativeRefinement; // ICNTL(10): iterative refinement steps
        mId.icntl[21] = mOutOfCore;           // ICNTL(22): out-of-core

        mInitialized = true;
    }

    void FinalizeMumps()
    {
        mId.job = -2;
        dmumps_c(&mId);
        mInitialized = false;
    }

    /// Extracts the matrix entries from rA in COO format with 1-based global indices.
    /// For symmetric matrices (mSym > 0) only the lower triangular part is stored.
    void ExtractCOO(const SparseMatrixType& rA,
                    std::vector<MUMPS_INT>& rIRN,
                    std::vector<MUMPS_INT>& rJCN,
                    std::vector<double>&    rVals)
    {
        const int n_local_rows    = rA.NumMyRows();
        const Epetra_Map& row_map = rA.RowMap();
        const Epetra_Map& col_map = rA.ColMap();

        rIRN.clear();
        rJCN.clear();
        rVals.clear();

        for (int i_local = 0; i_local < n_local_rows; ++i_local) {
            // Convert to 1-based global index for MUMPS
            const MUMPS_INT global_row = static_cast<MUMPS_INT>(row_map.GID(i_local)) + 1;

            int     num_entries;
            double* values;
            int*    col_inds;
            rA.ExtractMyRowView(i_local, num_entries, values, col_inds);

            for (int k = 0; k < num_entries; ++k) {
                const MUMPS_INT global_col = static_cast<MUMPS_INT>(col_map.GID(col_inds[k])) + 1;

                if (mSym > 0) {
                    // Lower triangular only for symmetric matrices
                    if (global_row >= global_col) {
                        rIRN.push_back(global_row);
                        rJCN.push_back(global_col);
                        rVals.push_back(values[k]);
                    }
                } else {
                    rIRN.push_back(global_row);
                    rJCN.push_back(global_col);
                    rVals.push_back(values[k]);
                }
            }
        }
    }

    ///@}
    ///@name Member Variables
    ///@{

    DMUMPS_STRUC_C mId;
    bool           mInitialized         = false;
    bool           mReanalyze           = true;
    int            mSym                 = 0;
    int            mVerbosity           = 0;
    int            mOrdering            = 0;
    int            mIterativeRefinement = 0;
    int            mOutOfCore           = 0;

    ///@}

}; // class TrilinosMumpsSolver

/// output stream function
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator<<(std::ostream& rOStream,
                                const TrilinosMumpsSolver<TSparseSpaceType,
                                TDenseSpaceType, TReordererType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    return rOStream;
}

} // namespace Kratos

#endif // KRATOS_TRILINOS_USE_MUMPS_DIRECTLY
