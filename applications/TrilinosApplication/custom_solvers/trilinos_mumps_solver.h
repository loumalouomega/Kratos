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
#include <mpi.h>       ///< MPI_Comm, MPI_Comm_c2f
#include <dmumps_c.h>  ///< DMUMPS_STRUC_C, dmumps_c, MUMPS_INT, DMUMPS_COMPLEX

// Project includes
#include "includes/define.h"
#include "includes/kratos_parameters.h"
#include "linear_solvers/linear_solver.h"

namespace Kratos
{

///@addtogroup TrilinosApplication
///@{

///@name Kratos Classes
///@{

/**
 * @class TrilinosMumpsSolver
 * @ingroup TrilinosApplication
 * @brief Direct linear solver that calls MUMPS via its native C API.
 *
 * @details
 * This solver bypasses the Amesos/Amesos2 Trilinos wrappers and drives MUMPS
 * directly, which gives full control over all MUMPS integer and real controls
 * (ICNTL/CNTL arrays) and post-solve diagnostics (INFOG/RINFOG arrays).
 *
 * ### Matrix input
 * The distributed matrix entries are extracted in COO format using MUMPS
 * *distributed assembled input* (`ICNTL(18) = 3`): every MPI rank contributes
 * its locally owned rows via `irn_loc`/`jcn_loc`/`a_loc`, so no global gather
 * of the matrix is ever required.
 *
 * ### RHS / solution handling
 * The right-hand side is gathered to rank 0 before the solve (using
 * `TSparseSpaceType::GatherToBuffer`) and the solution is scattered back to all
 * ranks afterwards (using `TSparseSpaceType::ScatterFromBuffer`).  The calls are
 * collective so every rank participates even though only rank 0 drives MUMPS'
 * dense RHS phase.
 *
 * ### Parameters (constructor)
 * All knobs are exposed through the `Parameters` object passed at construction.
 * An *escape hatch* (`additional_icntl` / `additional_cntl`) lets the user set
 * any ICNTL/CNTL index that does not have a friendly named parameter.
 * Named parameters are applied first; the escape hatch is applied last and
 * therefore overrides them.
 *
 * See `GetDefaultParameters()` and the `README_mumps.md` for the full parameter
 * table.
 *
 * ### Symbolic analysis caching
 * The first `Solve` call performs the full MUMPS pipeline:
 *   - JOB = −1 (initialize)
 *   - JOB =  1 (symbolic analysis)
 *   - JOB =  2 (numeric factorization)
 *   - JOB =  3 (back substitution)
 *
 * Subsequent calls on a matrix with the **same sparsity pattern** skip the
 * analysis phase (JOB = 1) so that the expensive fill-reducing ordering is
 * reused.  The internal flag `mReanalyze` is set to `false` after the first
 * successful analysis.
 *
 * MUMPS is finalized (JOB = −2) in the destructor.
 *
 * @tparam TSparseSpaceType Distributed sparse-space type (e.g. `TrilinosSpace<…>`).
 *         Must provide: `GetCommunicator`, `GetMpiComm`, `GetRank`, `Size1`,
 *         `GetLocalCOO`, `GatherToBuffer`, `ScatterFromBuffer`.
 * @tparam TDenseSpaceType  Dense (local) space type.
 * @tparam TReordererType   Reorderer (default `Reorderer<…>`).
 *
 * @author Vicente Mataix Ferrandiz
 */
template<class TSparseSpaceType, class TDenseSpaceType,
         class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType>>
class TrilinosMumpsSolver
    : public LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType>
{
public:
    ///@name Type Definitions
    ///@{

    /// Base class alias.
    using BaseType = LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType>;

    /// Distributed sparse matrix type (e.g. `Epetra_FECrsMatrix`).
    using SparseMatrixType = typename TSparseSpaceType::MatrixType;

    /// Distributed vector type (e.g. `Epetra_FEVector`).
    using VectorType = typename TSparseSpaceType::VectorType;

    /// Dense matrix type (local).
    using DenseMatrixType = typename TDenseSpaceType::MatrixType;

    /// Space index type.
    using IndexType = typename TSparseSpaceType::IndexType;

    /// Pointer definition following Kratos conventions.
    KRATOS_CLASS_POINTER_DEFINITION(TrilinosMumpsSolver);

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Constructs the solver from a JSON-backed Parameters object.
     *
     * @details All unrecognised keys are rejected by `ValidateAndAssignDefaults`.
     *          The raw ICNTL/CNTL escape hatches (`additional_icntl` and
     *          `additional_cntl`) are parsed into internal vectors during
     *          construction and applied in `InitializeMumps`.
     *
     * @param settings JSON Parameters.  Use `GetDefaultParameters()` to obtain
     *                 the full schema with documented defaults.
     */
    explicit TrilinosMumpsSolver(Parameters settings = Parameters(R"({})"))
    {
        settings.ValidateAndAssignDefaults(GetDefaultParameters());

        // Existing core controls
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

        // Escape hatches: map "1-based_index" → value
        for (auto it = settings["additional_icntl"].begin();
                  it != settings["additional_icntl"].end(); ++it) {
            mAdditionalIcntl.emplace_back(std::stoi(it.name()), it->GetInt());
        }
        for (auto it = settings["additional_cntl"].begin();
                  it != settings["additional_cntl"].end(); ++it) {
            mAdditionalCntl.emplace_back(std::stoi(it.name()), it->GetDouble());
        }
    }

    /// Not copyable.
    TrilinosMumpsSolver(const TrilinosMumpsSolver&) = delete;

    /// Not copy-assignable.
    TrilinosMumpsSolver& operator=(const TrilinosMumpsSolver&) = delete;

    /**
     * @brief Destructor.  Finalizes the MUMPS instance (JOB = −2) if it was
     *        previously initialized.
     */
    ~TrilinosMumpsSolver() override
    {
        if (mInitialized) {
            FinalizeMumps();
        }
    }

    ///@}
    ///@name LinearSolver interface
    ///@{

    /**
     * @brief Solves the linear system @f$ A x = b @f$ in-place using MUMPS.
     *
     * @details The solve pipeline for the first call:
     *   1. Extract local COO data from @p rA via `TSparseSpaceType::GetLocalCOO`.
     *   2. Initialize MUMPS (JOB = −1) and set all ICNTL/CNTL controls.
     *   3. Symbolic analysis (JOB = 1, skipped on subsequent calls).
     *   4. Numeric factorization (JOB = 2).
     *   5. Gather @p rB to rank 0 via `TSparseSpaceType::GatherToBuffer`.
     *   6. Back-substitution (JOB = 3) on rank 0.
     *   7. Scatter solution to @p rX via `TSparseSpaceType::ScatterFromBuffer`.
     *
     * @p rA and @p rB are **not modified**.
     *
     * @param rA Distributed sparse system matrix.
     * @param rX Solution vector (distributed).  Overwritten with the result.
     * @param rB Right-hand side vector (distributed).  Not modified.
     * @return @c true on success (MUMPS INFOG(1) == 0 for each phase).
     * @throws Kratos::Exception if any MUMPS phase reports a non-zero error code.
     */
    bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB) override
    {
        KRATOS_TRY

        const auto& r_comm    = TSparseSpaceType::GetCommunicator(rA);
        MPI_Comm mpi_comm     = TSparseSpaceType::GetMpiComm(r_comm);
        const int rank        = TSparseSpaceType::GetRank(r_comm);
        const int n_global    = static_cast<int>(TSparseSpaceType::Size1(rA));

        if (!mInitialized) {
            InitializeMumps(mpi_comm);
        }

        mId.n = n_global;

        // --- 1. Extract local COO data (0-based global indices) ---
        std::vector<IndexType> row_0based, col_0based;
        std::vector<double>    a_loc;
        TSparseSpaceType::GetLocalCOO(rA, row_0based, col_0based, a_loc, mSym > 0);

        // Convert to MUMPS 1-based MUMPS_INT
        const std::size_t nnz_loc = row_0based.size();
        std::vector<MUMPS_INT> irn_loc(nnz_loc), jcn_loc(nnz_loc);
        for (std::size_t k = 0; k < nnz_loc; ++k) {
            irn_loc[k] = static_cast<MUMPS_INT>(row_0based[k]) + 1;
            jcn_loc[k] = static_cast<MUMPS_INT>(col_0based[k]) + 1;
        }

        // Distributed assembled matrix input (ICNTL(18) = 3)
        mId.nz_loc  = static_cast<MUMPS_INT>(nnz_loc);
        mId.irn_loc = irn_loc.data();
        mId.jcn_loc = jcn_loc.data();
        mId.a_loc   = a_loc.data();

        // --- 2. Symbolic analysis (first call only, or when forced) ---
        if (mReanalyze) {
            mId.job = 1;
            dmumps_c(&mId);
            KRATOS_ERROR_IF(mId.infog[0] != 0)
                << "MUMPS analysis phase failed with error code " << mId.infog[0]
                << " (infog[1] = " << mId.infog[1] << ")" << std::endl;
            mReanalyze = false;
        }

        // --- 3. Numeric factorization ---
        mId.job = 2;
        dmumps_c(&mId);
        KRATOS_ERROR_IF(mId.infog[0] != 0)
            << "MUMPS factorization phase failed with error code " << mId.infog[0]
            << " (infog[1] = " << mId.infog[1] << ")" << std::endl;

        // --- 4. Gather RHS to rank 0 as a flat buffer ---
        std::vector<double> rhs_buffer;
        TSparseSpaceType::GatherToBuffer(rB, r_comm, rhs_buffer);

        // --- 5. Back-substitution (RHS is centralized on rank 0) ---
        mId.job  = 3;
        mId.nrhs = 1;
        mId.lrhs = n_global;
        mId.rhs  = (rank == 0) ? rhs_buffer.data() : nullptr;
        dmumps_c(&mId);
        KRATOS_ERROR_IF(mId.infog[0] != 0)
            << "MUMPS solve phase failed with error code " << mId.infog[0]
            << " (infog[1] = " << mId.infog[1] << ")" << std::endl;

        // --- 6. Scatter solution (now in rhs_buffer on rank 0) to rX ---
        TSparseSpaceType::ScatterFromBuffer(r_comm, rhs_buffer, rX);

        return true;

        KRATOS_CATCH("")
    }

    /**
     * @brief Dense multi-RHS overload — not supported; always returns @c false.
     */
    bool Solve(SparseMatrixType& rA, DenseMatrixType& rX, DenseMatrixType& rB) override
    {
        return false;
    }

    ///@}
    ///@name Post-solve diagnostics
    ///@{

    /**
     * @brief Returns INFOG(@p Index) from the last solve (1-based index, range 1–80).
     * @param Index MUMPS 1-based INFOG index.
     * @return Integer value of INFOG(Index).
     * @throws Kratos::Exception if @p Index is out of range.
     */
    int GetInfog(int Index) const
    {
        KRATOS_ERROR_IF(Index < 1 || Index > 80)
            << "INFOG index " << Index << " out of valid range [1, 80]" << std::endl;
        return mId.infog[Index - 1];
    }

    /**
     * @brief Returns RINFOG(@p Index) from the last solve (1-based index, range 1–40).
     * @param Index MUMPS 1-based RINFOG index.
     * @return Real value of RINFOG(Index).
     * @throws Kratos::Exception if @p Index is out of range.
     */
    double GetRinfog(int Index) const
    {
        KRATOS_ERROR_IF(Index < 1 || Index > 40)
            << "RINFOG index " << Index << " out of valid range [1, 40]" << std::endl;
        return mId.rinfog[Index - 1];
    }

    /**
     * @brief Returns the determinant of the factored matrix.
     * @details Only meaningful when `compute_determinant = 1` was set.
     *          MUMPS returns det = RINFOG(12) × 2^INFOG(34).
     * @return The determinant.
     */
    double GetDeterminant() const
    {
        return mId.rinfog[11] * std::ldexp(1.0, mId.infog[33]);
    }

    /**
     * @brief Returns the estimated condition number RINFOG(11).
     * @details Only meaningful when `error_analysis = 1` was set.
     * @return Estimated condition number.
     */
    double GetEstimatedConditionNumber() const
    {
        return mId.rinfog[10];
    }

    /**
     * @brief Returns the scaled backward error max(RINFOG(7), RINFOG(8)).
     * @details Only meaningful when `error_analysis >= 1` was set.
     * @return Scaled backward error.
     */
    double GetBackwardError() const
    {
        return std::max(mId.rinfog[6], mId.rinfog[7]);
    }

    /**
     * @brief Returns the count of null pivots detected INFOG(28).
     * @details Only meaningful when `null_pivot_detection = 1` was set.
     * @return Number of null (near-zero) pivots encountered during factorization.
     */
    int GetNumNullPivots() const
    {
        return mId.infog[27];
    }

    ///@}
    ///@name Input and Output
    ///@{

    /**
     * @brief Prints solver configuration and (when enabled) post-solve diagnostics.
     * @param rOStream Output stream to write to.
     */
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "TrilinosMumpsSolver"
                 << " [sym="                     << mSym
                 << ", ordering="                << mOrdering
                 << ", iterative_refinement="    << mIterativeRefinement
                 << ", out_of_core="             << mOutOfCore
                 << ", scaling="                 << mScaling
                 << ", null_pivot_detection="    << mNullPivotDetection
                 << ", memory_relaxation_pct="   << mMemoryRelaxation
                 << ", max_working_memory_mb="   << mMaxWorkingMemoryMB
                 << ", analysis_type="           << mAnalysisType
                 << ", parallel_ordering="       << mParallelOrdering
                 << ", block_low_rank="          << mBlockLowRank
                 << ", error_analysis="          << mErrorAnalysis
                 << ", compute_determinant="     << mComputeDeterminant
                 << ", verbosity="               << mVerbosity << "]";

        KRATOS_INFO("TrilinosMumpsSolver")
            << "sym=" << mSym
            << " ordering=" << mOrdering
            << " iterative_refinement=" << mIterativeRefinement
            << " out_of_core=" << mOutOfCore
            << " scaling=" << mScaling
            << " null_pivot_detection=" << mNullPivotDetection
            << " memory_relaxation_pct=" << mMemoryRelaxation
            << " max_working_memory_mb=" << mMaxWorkingMemoryMB
            << " analysis_type=" << mAnalysisType
            << " parallel_ordering=" << mParallelOrdering
            << " block_low_rank=" << mBlockLowRank
            << " error_analysis=" << mErrorAnalysis
            << " compute_determinant=" << mComputeDeterminant
            << " verbosity=" << mVerbosity << std::endl;

        if (mInitialized) {
            if (mComputeDeterminant) {
                KRATOS_INFO("TrilinosMumpsSolver")
                    << "determinant=" << GetDeterminant() << std::endl;
            }
            if (mNullPivotDetection) {
                KRATOS_INFO("TrilinosMumpsSolver")
                    << "num_null_pivots=" << GetNumNullPivots() << std::endl;
            }
            if (mErrorAnalysis == 1) {
                KRATOS_INFO("TrilinosMumpsSolver")
                    << "estimated_condition_number=" << GetEstimatedConditionNumber() << std::endl;
            }
            if (mErrorAnalysis >= 1) {
                KRATOS_INFO("TrilinosMumpsSolver")
                    << "backward_error=" << GetBackwardError() << std::endl;
            }
        }
    }

    ///@}
    ///@name Static Utilities
    ///@{

    /**
     * @brief Returns a Parameters object containing all recognised keys with their
     *        default values.
     *
     * @details The schema is also used by the constructor to validate input through
     *          `ValidateAndAssignDefaults`.
     *
     * @par Parameter table (subset)
     * | Key | MUMPS control | Default | Notes |
     * |-----|---------------|---------|-------|
     * | `sym` | `id.sym` | 0 | 0=unsymmetric, 1=symmetric, 2=SPD |
     * | `ordering` | ICNTL(7) | 0 | 0=auto, 1=Scotch, 2=METIS, … |
     * | `scaling` | ICNTL(8) | 77 | 77=auto |
     * | `iterative_refinement_steps` | ICNTL(10) | 0 | 0=disabled |
     * | `null_pivot_detection` | ICNTL(24) | 0 | 1=enabled |
     * | `out_of_core` | ICNTL(22) | 0 | 1=out-of-core |
     * | `memory_relaxation_percent` | ICNTL(14) | -1 | <0=keep MUMPS default |
     * | `max_working_memory_mb` | ICNTL(23) | 0 | 0=MUMPS decides |
     * | `analysis_type` | ICNTL(28) | 0 | 0=auto, 1=seq, 2=parallel |
     * | `parallel_ordering` | ICNTL(29) | 0 | 0=auto, 1=PT-Scotch, 2=ParMETIS |
     * | `block_low_rank` | ICNTL(35) | 0 | 0=off, 1/2/3=BLR variants |
     * | `error_analysis` | ICNTL(11) | 0 | 1=full, 2=backward error only |
     * | `compute_determinant` | ICNTL(33) | 0 | 1=compute det |
     * | `additional_icntl` | `{}` | — | raw overrides: `{"14": 35}` |
     * | `additional_cntl` | `{}` | — | raw overrides: `{"1": 0.001}` |
     *
     * @return Default Parameters object.
     */
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

    /**
     * @brief Initializes the MUMPS instance (JOB = −1) and applies all ICNTL/CNTL
     *        settings from the constructor parameters.
     *
     * @details Called once on the first `Solve` invocation.  The MPI communicator
     *          is obtained from the matrix via `TSparseSpaceType::GetMpiComm` and
     *          passed to MUMPS as a Fortran handle via `MPI_Comm_c2f`.
     *
     *          All controls that affect the analysis phase (ordering, BLR, input
     *          format) must be set *before* JOB = 1; they are set here so that the
     *          subsequent analysis call in `Solve` picks them up correctly.
     *
     * @param mpi_comm The MPI communicator to pass to MUMPS.
     */
    void InitializeMumps(MPI_Comm mpi_comm)
    {
        mId.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(mpi_comm);
        mId.job = -1;  // MUMPS initialize
        mId.par = 1;   // host participates in factorization
        mId.sym = mSym;
        dmumps_c(&mId);

        // ICNTL(18) = 3: distributed assembled input — must be set before analysis
        mId.icntl[17] = 3;

        // Verbosity (ICNTL(1–4))
        if (mVerbosity == 0) {
            mId.icntl[0] = -1;  // suppress error stream
            mId.icntl[1] = -1;  // suppress diagnostic stream
            mId.icntl[2] = -1;  // suppress global information stream
            mId.icntl[3] = 0;   // no statistics
        } else {
            mId.icntl[0] = 6;   // errors → stdout
            mId.icntl[1] = 6;   // diagnostics → stdout
            mId.icntl[2] = 6;   // global info → stdout
            mId.icntl[3] = (mVerbosity > 1) ? 2 : 1;
        }

        // Ordering / analysis (must precede JOB = 1)
        mId.icntl[6]  = mOrdering;         // ICNTL(7):  sequential fill-reducing ordering
        mId.icntl[27] = mAnalysisType;     // ICNTL(28): sequential vs parallel analysis
        mId.icntl[28] = mParallelOrdering; // ICNTL(29): parallel ordering tool

        // Numerical robustness
        mId.icntl[7]  = mScaling;             // ICNTL(8):  scaling strategy
        mId.icntl[23] = mNullPivotDetection;  // ICNTL(24): null pivot detection
        if (mPivotingThreshold >= 0.0) {
            mId.cntl[0] = mPivotingThreshold; // CNTL(1): relative pivoting threshold
        }
        mId.cntl[2] = mNullPivotThreshold;    // CNTL(3): null pivot detection threshold

        // Solve-time controls
        mId.icntl[9]  = mIterativeRefinement; // ICNTL(10): iterative refinement steps
        mId.icntl[10] = mErrorAnalysis;       // ICNTL(11): error analysis / condition number
        mId.icntl[21] = mOutOfCore;           // ICNTL(22): out-of-core factorization
        mId.icntl[32] = mComputeDeterminant;  // ICNTL(33): compute determinant

        // Memory controls
        if (mMemoryRelaxation >= 0) {
            mId.icntl[13] = mMemoryRelaxation; // ICNTL(14): workspace relaxation (%)
        }
        mId.icntl[22] = mMaxWorkingMemoryMB;  // ICNTL(23): per-process max working memory (MB)

        // Block Low-Rank (must be set before analysis)
        mId.icntl[34] = mBlockLowRank;        // ICNTL(35): BLR activation level
        mId.icntl[35] = mBlrVariant;          // ICNTL(36): BLR variant
        mId.cntl[6]   = mBlrThreshold;        // CNTL(7):   BLR compression threshold

        // Escape hatch: user overrides applied last so they win over named controls
        for (const auto& r_pair : mAdditionalIcntl) {
            mId.icntl[r_pair.first - 1] = r_pair.second; // convert 1-based → 0-based
        }
        for (const auto& r_pair : mAdditionalCntl) {
            mId.cntl[r_pair.first - 1] = r_pair.second;
        }

        mInitialized = true;
    }

    /**
     * @brief Finalizes the MUMPS instance (JOB = −2), releasing all internal memory.
     * @details Called from the destructor when `mInitialized` is @c true.
     */
    void FinalizeMumps()
    {
        mId.job = -2;
        dmumps_c(&mId);
        mInitialized = false;
    }

    ///@}
    ///@name Member Variables
    ///@{

    /// MUMPS internal data structure; holds all control arrays and output info.
    DMUMPS_STRUC_C mId;

    /// True after JOB = −1 has been called; false before or after JOB = −2.
    bool mInitialized = false;

    /**
     * @brief When @c true the symbolic analysis (JOB = 1) is (re-)run on the next
     *        `Solve` call.  Set to @c false after the first successful analysis and
     *        kept @c false for subsequent calls with the same sparsity pattern.
     */
    bool mReanalyze = true;

    // ----- Named ICNTL/CNTL parameters (read from Parameters in constructor) -----

    int    mSym                 = 0;     ///< id.sym: 0=unsymmetric, 1=symmetric, 2=SPD
    int    mVerbosity           = 0;     ///< Controls ICNTL(1–4) output streams
    int    mOrdering            = 0;     ///< ICNTL(7): fill-reducing ordering method
    int    mIterativeRefinement = 0;     ///< ICNTL(10): max iterative refinement steps
    int    mOutOfCore           = 0;     ///< ICNTL(22): 1 = out-of-core factorization
    int    mScaling             = 77;    ///< ICNTL(8): scaling strategy (77=automatic)
    double mPivotingThreshold   = -1.0; ///< CNTL(1): pivoting threshold (<0 = MUMPS auto)
    int    mNullPivotDetection  = 0;     ///< ICNTL(24): 1 = detect null pivots
    double mNullPivotThreshold  = 0.0;  ///< CNTL(3): null pivot detection threshold
    int    mMemoryRelaxation    = -1;    ///< ICNTL(14): workspace relaxation % (<0 = default)
    int    mMaxWorkingMemoryMB  = 0;     ///< ICNTL(23): per-process working memory limit (MB)
    int    mAnalysisType        = 0;     ///< ICNTL(28): 0=auto, 1=sequential, 2=parallel
    int    mParallelOrdering    = 0;     ///< ICNTL(29): 0=auto, 1=PT-Scotch, 2=ParMETIS
    int    mBlockLowRank        = 0;     ///< ICNTL(35): BLR level (0=off)
    int    mBlrVariant          = 0;     ///< ICNTL(36): BLR variant selector
    double mBlrThreshold        = 0.0;  ///< CNTL(7): BLR compression threshold
    int    mErrorAnalysis       = 0;     ///< ICNTL(11): 0=none, 1=full, 2=backward error only
    int    mComputeDeterminant  = 0;     ///< ICNTL(33): 1 = compute the determinant

    /// Raw ICNTL overrides from `additional_icntl`; pairs of (1-based index, value).
    std::vector<std::pair<int, int>>    mAdditionalIcntl;

    /// Raw CNTL overrides from `additional_cntl`; pairs of (1-based index, value).
    std::vector<std::pair<int, double>> mAdditionalCntl;

    ///@}

}; // class TrilinosMumpsSolver

///@}

/// Output stream operator.
template<class TSparseSpaceType, class TDenseSpaceType, class TReordererType>
inline std::ostream& operator<<(
    std::ostream& rOStream,
    const TrilinosMumpsSolver<TSparseSpaceType, TDenseSpaceType, TReordererType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    return rOStream;
}

///@}

} // namespace Kratos

#endif // KRATOS_TRILINOS_USE_MUMPS_DIRECTLY
