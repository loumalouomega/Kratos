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
#include <cmath>
#include <string>
#include <utility>
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
        settings.ValidateAndAssignDefaults(GetDefaultParameters());

        // Existing controls
        mSym                 = settings["sym"].GetInt();
        mVerbosity           = settings["verbosity"].GetInt();
        mOrdering            = settings["ordering"].GetInt();
        mIterativeRefinement = settings["iterative_refinement_steps"].GetInt();
        mOutOfCore           = settings["out_of_core"].GetInt();

        // Numerical robustness
        mScaling             = settings["scaling"].GetInt();
        mPivotingThreshold   = settings["pivoting_threshold"].GetDouble();
        mNullPivotDetection  = settings["null_pivot_detection"].GetInt();
        mNullPivotThreshold  = settings["null_pivot_threshold"].GetDouble();

        // Memory controls
        mMemoryRelaxation    = settings["memory_relaxation_percent"].GetInt();
        mMaxWorkingMemoryMB  = settings["max_working_memory_mb"].GetInt();

        // Parallel ordering / analysis
        mAnalysisType        = settings["analysis_type"].GetInt();
        mParallelOrdering    = settings["parallel_ordering"].GetInt();

        // Block Low-Rank
        mBlockLowRank        = settings["block_low_rank"].GetInt();
        mBlrVariant          = settings["blr_variant"].GetInt();
        mBlrThreshold        = settings["blr_compression_threshold"].GetDouble();

        // Diagnostics
        mErrorAnalysis       = settings["error_analysis"].GetInt();
        mComputeDeterminant  = settings["compute_determinant"].GetInt();

        // Escape hatch: parse free-form override maps (index -> value, 1-based)
        for (auto it = settings["additional_icntl"].begin(); it != settings["additional_icntl"].end(); ++it) {
            mAdditionalIcntl.emplace_back(std::stoi(it.name()), it->GetInt());
        }
        for (auto it = settings["additional_cntl"].begin(); it != settings["additional_cntl"].end(); ++it) {
            mAdditionalCntl.emplace_back(std::stoi(it.name()), it->GetDouble());
        }
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
                 << ", scaling=" << mScaling
                 << ", null_pivot_detection=" << mNullPivotDetection
                 << ", memory_relaxation_percent=" << mMemoryRelaxation
                 << ", max_working_memory_mb=" << mMaxWorkingMemoryMB
                 << ", analysis_type=" << mAnalysisType
                 << ", parallel_ordering=" << mParallelOrdering
                 << ", block_low_rank=" << mBlockLowRank
                 << ", error_analysis=" << mErrorAnalysis
                 << ", compute_determinant=" << mComputeDeterminant
                 << ", verbosity=" << mVerbosity << "]";

        KRATOS_INFO("TrilinosMumpsSolver")
            << "sym=" << mSym
            << " ordering=" << mOrdering
            << " iterative_refinement=" << mIterativeRefinement
            << " out_of_core=" << mOutOfCore
            << " scaling=" << mScaling
            << " null_pivot_detection=" << mNullPivotDetection
            << " memory_relaxation_percent=" << mMemoryRelaxation
            << " max_working_memory_mb=" << mMaxWorkingMemoryMB
            << " analysis_type=" << mAnalysisType
            << " parallel_ordering=" << mParallelOrdering
            << " block_low_rank=" << mBlockLowRank
            << " error_analysis=" << mErrorAnalysis
            << " compute_determinant=" << mComputeDeterminant
            << " verbosity=" << mVerbosity << std::endl;

        // Report diagnostics gathered during the last solve (when enabled)
        if (mInitialized) {
            if (mComputeDeterminant) {
                KRATOS_INFO("TrilinosMumpsSolver") << "determinant=" << GetDeterminant() << std::endl;
            }
            if (mNullPivotDetection) {
                KRATOS_INFO("TrilinosMumpsSolver") << "num_null_pivots=" << GetNumNullPivots() << std::endl;
            }
            if (mErrorAnalysis == 1) {
                KRATOS_INFO("TrilinosMumpsSolver")
                    << "estimated_condition_number=" << GetEstimatedConditionNumber() << std::endl;
            }
            if (mErrorAnalysis == 1 || mErrorAnalysis == 2) {
                KRATOS_INFO("TrilinosMumpsSolver") << "backward_error=" << GetBackwardError() << std::endl;
            }
        }
    }

    ///@}
    ///@name Inquiry (diagnostics from the last solve)
    ///@{

    /// Raw access to MUMPS global integer info: returns INFOG(Index) (1-based, as in the MUMPS docs).
    int GetInfog(int Index) const
    {
        KRATOS_ERROR_IF(Index < 1 || Index > 80) << "INFOG index out of range [1, 80]: " << Index << std::endl;
        return mId.infog[Index - 1];
    }

    /// Raw access to MUMPS global real info: returns RINFOG(Index) (1-based, as in the MUMPS docs).
    double GetRinfog(int Index) const
    {
        KRATOS_ERROR_IF(Index < 1 || Index > 40) << "RINFOG index out of range [1, 40]: " << Index << std::endl;
        return mId.rinfog[Index - 1];
    }

    /// Determinant of the matrix. Only meaningful when "compute_determinant" was enabled.
    /// MUMPS returns it as det = RINFOG(12) * 2^INFOG(34).
    double GetDeterminant() const
    {
        return mId.rinfog[11] * std::ldexp(1.0, mId.infog[33]); // RINFOG(12) * 2^INFOG(34)
    }

    /// Estimated condition number RINFOG(11). Only meaningful when "error_analysis" == 1.
    double GetEstimatedConditionNumber() const
    {
        return mId.rinfog[10]; // RINFOG(11)
    }

    /// Scaled backward error max(RINFOG(7), RINFOG(8)). Only meaningful when "error_analysis" >= 1.
    double GetBackwardError() const
    {
        return std::max(mId.rinfog[6], mId.rinfog[7]); // max(RINFOG(7), RINFOG(8))
    }

    /// Number of null pivots detected INFOG(28). Only meaningful when "null_pivot_detection" was enabled.
    int GetNumNullPivots() const
    {
        return mId.infog[27]; // INFOG(28)
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
            "out_of_core"                : 0,
            "scaling"                    : 77,
            "pivoting_threshold"         : -1.0,
            "null_pivot_detection"       : 0,
            "null_pivot_threshold"       : 0.0,
            "memory_relaxation_percent"  : -1,
            "max_working_memory_mb"      : 0,
            "analysis_type"              : 0,
            "parallel_ordering"          : 0,
            "block_low_rank"             : 0,
            "blr_variant"                : 0,
            "blr_compression_threshold"  : 0.0,
            "error_analysis"             : 0,
            "compute_determinant"        : 0,
            "additional_icntl"           : {},
            "additional_cntl"            : {}
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

        // Ordering / analysis (must be set before the analysis phase, job=1)
        mId.icntl[6]  = mOrdering;            // ICNTL(7):  sequential ordering
        mId.icntl[27] = mAnalysisType;        // ICNTL(28): sequential vs parallel analysis
        mId.icntl[28] = mParallelOrdering;    // ICNTL(29): parallel ordering tool

        // Numerical robustness
        mId.icntl[7]  = mScaling;             // ICNTL(8):  scaling strategy
        mId.icntl[23] = mNullPivotDetection;  // ICNTL(24): null pivot detection
        if (mPivotingThreshold >= 0.0) {
            mId.cntl[0] = mPivotingThreshold; // CNTL(1): relative pivoting threshold
        }
        mId.cntl[2] = mNullPivotThreshold;    // CNTL(3): null pivot threshold

        // Solve-time controls
        mId.icntl[9]  = mIterativeRefinement; // ICNTL(10): iterative refinement steps
        mId.icntl[10] = mErrorAnalysis;       // ICNTL(11): error analysis
        mId.icntl[21] = mOutOfCore;           // ICNTL(22): out-of-core
        mId.icntl[32] = mComputeDeterminant;  // ICNTL(33): compute determinant

        // Memory controls
        if (mMemoryRelaxation >= 0) {
            mId.icntl[13] = mMemoryRelaxation; // ICNTL(14): memory relaxation (%)
        }
        mId.icntl[22] = mMaxWorkingMemoryMB;  // ICNTL(23): max working memory per MPI process (MB)

        // Block Low-Rank (must be set before the analysis phase)
        mId.icntl[34] = mBlockLowRank;        // ICNTL(35): BLR activation
        mId.icntl[35] = mBlrVariant;          // ICNTL(36): BLR variant
        mId.cntl[6]   = mBlrThreshold;        // CNTL(7):  BLR compression threshold

        // Escape hatch: user overrides applied last so they win over named controls
        for (const auto& r_pair : mAdditionalIcntl) {
            mId.icntl[r_pair.first - 1] = r_pair.second;
        }
        for (const auto& r_pair : mAdditionalCntl) {
            mId.cntl[r_pair.first - 1] = r_pair.second;
        }

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

    // Existing controls
    int            mSym                 = 0;
    int            mVerbosity           = 0;
    int            mOrdering            = 0;
    int            mIterativeRefinement = 0;
    int            mOutOfCore           = 0;

    // Numerical robustness
    int            mScaling             = 77;
    double         mPivotingThreshold   = -1.0;
    int            mNullPivotDetection  = 0;
    double         mNullPivotThreshold  = 0.0;

    // Memory controls
    int            mMemoryRelaxation    = -1;
    int            mMaxWorkingMemoryMB  = 0;

    // Parallel ordering / analysis
    int            mAnalysisType        = 0;
    int            mParallelOrdering    = 0;

    // Block Low-Rank
    int            mBlockLowRank        = 0;
    int            mBlrVariant          = 0;
    double         mBlrThreshold        = 0.0;

    // Diagnostics
    int            mErrorAnalysis       = 0;
    int            mComputeDeterminant  = 0;

    // Escape hatch: index (1-based) -> value
    std::vector<std::pair<int, int>>    mAdditionalIcntl;
    std::vector<std::pair<int, double>> mAdditionalCntl;

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
