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

// External includes

/* Trilinos includes */
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_FEMultiVector.hpp>
#include <Tpetra_FECrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#include <TpetraExt_TripleMatrixMultiply.hpp>

//#include <MatrixMarket_Tpetra.hpp>


// Project includes
#include "includes/ublas_interface.h"
#include "spaces/ublas_space.h"
#include "includes/data_communicator.h"
#include "mpi/includes/mpi_data_communicator.h"


namespace Kratos
{
///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/**
 * @class TrilinosSpaceExperimental
 * @ingroup TrilinosApplication
 * @brief The space adapted for Trilinos vectors and matrices (TPetra)
 * @details This class is experimental and aims to bring support for new TPetra matrices and vectors to TrilinosApplication
 * @author Vicente Mataix Ferrandiz
 * @tparam TMatrixType The matrix type considered
 * @tparam TVectorType the vector type considered
 */
template<class TMatrixType, class TVectorType>
class TrilinosSpaceExperimental
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of TrilinosSpaceExperimental
    KRATOS_CLASS_POINTER_DEFINITION(TrilinosSpaceExperimental);

    /// Definition of the data type
    using DataType = typename TMatrixType::scalar_type;

    /// Definition of the matrix type
    using MatrixType = TMatrixType;

    /// Definition of the vector type
    using VectorType = TVectorType;

    /// Definition of the index type
    using IndexType = std::size_t;

    /// Definition of the size type
    using SizeType = std::size_t;

    /// Class definition
    using ClassType = TrilinosSpaceExperimental<TMatrixType, TVectorType>;

    /// Tpetra definitions
    // Your scalar type; the type of sparse matrix entries. e.g., double.
    using ST = typename MatrixType::scalar_type;
    // Your local ordinal type; the signed integer type used to store local sparse matrix indices.  e.g., int.
    using LO = typename MatrixType::local_ordinal_type;
    // Your global ordinal type; the signed integer type used to index the matrix globally, over all processes. e.g., int, long, ptrdif_t, int64_t, ...
    using GO = typename MatrixType::global_ordinal_type;
    // The Node type.  e.g., Kokkos::DefaultNode::DefaultNodeType, defined in KokkosCompat_DefaultNode.hpp.
    using NT = typename MatrixType::node_type;

    /// Definition of the CrsMatrix type
    using CrsMatrixType = Tpetra::CrsMatrix<ST, LO, GO, NT>;

    /// Define the import/export types
    using ImportType = Tpetra::Import<LO, GO, NT>;
    using ExportType = Tpetra::Export<LO, GO, NT>;

    /// Define the map type
    using MapType = Tpetra::Map<LO, GO, NT>;
    using MapPointerType = Teuchos::RCP<const MapType>;

    /// Define the graph type
    using GraphType = Tpetra::FECrsGraph<LO, GO, NT>;
    using GraphPointerType = Teuchos::RCP<const GraphType>;

    // Define TPetra communicator
    using CommunicatorType = Teuchos::MpiComm<int>;
    using CommunicatorPointerType = Teuchos::RCP<const CommunicatorType>;

    /// Definition of the pointer types
    using MatrixPointerType = Teuchos::RCP<MatrixType>;
    using VectorPointerType = Teuchos::RCP<VectorType>;

    /// Column indices of row non-zero values type
    using ColumnViewType = typename MatrixType::nonconst_global_inds_host_view_type;

    /// Row non-zero values type
    using ValueViewType = typename MatrixType::nonconst_values_host_view_type;

    /// Some other definitions
    using DofUpdaterType = DofUpdater<ClassType>;
    using DofUpdaterPointerType = typename DofUpdater<ClassType>::UniquePointer;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    TrilinosSpaceExperimental()
    {
    }

    /// Destructor.
    virtual ~TrilinosSpaceExperimental()
    {
    }

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief This method returns the rank of the communicator
     * @param rComm The communicator considered
     * @return The rank of the communicator
     */
    inline static int GetRank(const CommunicatorType& rComm)
    {
        return rComm.getRank();
    }

    /**
     * @brief This method returns true if the pointer is null
     * @param pPointer The pointer considered
     * @return True if the pointer is null
     */
    template<class TPointerType>
    inline static bool IsNull(const TPointerType& pPointer)
    {
        return pPointer == Teuchos::null;
    }

    /**
     * @brief This method creates an empty pointer to a map
     * @return The pointer to the map
     */
    inline static MapPointerType CreateEmptyMapPointer()
    {
        return Teuchos::null;
    }

    /**
     * @brief This method creates an empty pointer to a matrix
     * @return The pointer to the matrix
     */
    inline static MatrixPointerType CreateEmptyMatrixPointer()
    {
        return MatrixPointerType(nullptr);
    }

    /**
     * @brief This method creates an empty pointer to a vector
     * @return The pointer to the vector
     */
    inline static VectorPointerType CreateEmptyVectorPointer()
    {
        return VectorPointerType(nullptr);
    }

    /**
     * @brief This method creates an empty pointer to a matrix using TPetra communicator
     * @param rComm The Tpetra communicator
     * @return The pointer to the matrix
     */
    inline static MatrixPointerType CreateEmptyMatrixPointer(CommunicatorPointerType pComm)
    {
        const int global_elems = 0;
        MapPointerType map = Teuchos::rcp(new MapType(global_elems, 0, pComm));
        // Use non-const RCP so we can call endAssembly() (FE variant of fillComplete)
        Teuchos::RCP<GraphType> graph = Teuchos::rcp(new GraphType(map, map, 0));
        if (graph->isFillActive()) {
            graph->endAssembly();
        }
        return Teuchos::rcp(new MatrixType(Teuchos::rcp_const_cast<const GraphType>(graph)));
    }

    /**
     * @brief This method creates an empty pointer to a vector using Tpetra communicator
     * @param pComm The Tpetra communicator
     * @return The pointer to the vector
     */
    inline static VectorPointerType CreateEmptyVectorPointer(CommunicatorPointerType pComm)
    {
        const int global_elems = 0;
        MapPointerType map = Teuchos::rcp(new MapType(global_elems, 0, pComm));
        return CreateVector(map);
    }
    /**
     * @brief This method creates an empty pointer to a matrix using TPetra communicator
     * @param rComm The Tpetra communicator
     * @return The pointer to the matrix
     */
    inline static MatrixPointerType CreateEmptyMatrixPointer(CommunicatorType& rComm)
    {
        return CreateEmptyMatrixPointer(Teuchos::rcp(&rComm, false));
    }

    /**
     * @brief This method creates an empty pointer to a vector using Tpetra communicator
     * @param rComm The Tpetra communicator
     * @return The pointer to the vector
     */
    inline static VectorPointerType CreateEmptyVectorPointer(CommunicatorType& rComm)
    {
        return CreateEmptyVectorPointer(Teuchos::rcp(&rComm, false));
    }


    /**
     * @brief Returns size of vector rV
     * @param rV The vector considered
     * @return The size of the vector
     */
    inline static IndexType Size(const VectorType& rV)
    {
        const int size = rV.getGlobalLength();
        return size;
    }

    /**
     * @brief Returns number of rows of rM
     * @param rM The matrix considered
     * @return The number of rows of rM
     */
    inline static IndexType Size1(MatrixType const& rM)
    {
        const int size1 = rM.getGlobalNumRows();
        return size1;
    }

    /**
     * @brief Returns number of columns of rM
     * @param rM The matrix considered
     * @return The number of columns of rM
     */
    inline static IndexType Size2(MatrixType const& rM)
    {
        const int size1 = rM.getGlobalNumCols();
        return size1;
    }

    /**
     * @brief Returns the column of the matrix in the given position
     * @details rXi = rMij
     * @param j The position of the column
     * @param rM The matrix considered
     * @param rX The column considered
     * @todo Implement this method
     */
    inline static void GetColumn(
        const unsigned int j,
        const MatrixType& rM,
        VectorType& rX
        )
    {
        KRATOS_ERROR << "GetColumn method is not currently implemented" << std::endl;
    }

    /**
     * @brief Returns a copy of the matrix rX
     * @details rY = rX
     * @param rX The matrix considered
     * @param rY The copy of the matrix rX
     */
    inline static void Copy(
        const MatrixType& rX,
        MatrixType& rY
        )
    {
        rY = rX;
    }

    /**
     * @brief Returns a copy of the vector rX
     * @details rY = rX
     * @param rX The vector considered
     * @param rY The copy of the vector rX
     */
    inline static void Copy(
        const VectorType& rX,
        VectorType& rY
        )
    {
        rY = rX;
    }

    /**
     * @brief Returns the product of two vectors
     * @details rX * rY
     * @param rX The first vector considered
     * @param rY The second vector considered
     */
    inline static double Dot(
        const VectorType& rX,
        const VectorType& rY
        )
    {
        // FEMultiVector doesn't have a single-value dot(); compute via local views
        auto localViewX = rX.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto localViewY = rY.getLocalViewHost(Tpetra::Access::ReadOnly);
        ST localDot = Teuchos::ScalarTraits<ST>::zero();
        const auto n = rX.getLocalLength();
        for (std::size_t i = 0; i < n; ++i) {
            localDot += localViewX(i, 0) * localViewY(i, 0);
        }
        ST globalDot = Teuchos::ScalarTraits<ST>::zero();
        Teuchos::reduceAll(*rX.getMap()->getComm(), Teuchos::REDUCE_SUM, localDot, Teuchos::outArg(globalDot));
        return static_cast<double>(globalDot);
    }

    /**
     * @brief Returns the maximum value of the vector rX
     * @param rX The vector considered
     * @return The maximum value of the vector rX
     */
    inline static double Max(const VectorType& rX)
    {
        // Access the local data
        auto localVec = rX.getLocalViewHost(Tpetra::Access::ReadOnly);

        // Find the local maximum
        ST localMax = localVec(0,0);
        auto localLength = rX.getLocalLength();
        for (std::size_t i = 1; i < localLength; ++i) {
            localMax = std::max(localMax, localVec(i,0));
        }

        // Perform a global maximum reduction
        ST globalMax;
        Teuchos::reduceAll(*rX.getMap()->getComm(), Teuchos::REDUCE_MAX, localMax, Teuchos::outArg(globalMax));

        return globalMax;
    }

    /**
     * @brief Returns the minimum value of the vector rX
     * @param rX The vector considered
     * @return The minimum value of the vector rX
     */
    inline static double Min(const VectorType& rX)
    {
        // Access the local data
        auto localVec = rX.getLocalViewHost(Tpetra::Access::ReadOnly);

        // Find the local minimum
        ST localMin = localVec(0,0);
        auto localLength = rX.getLocalLength();
        for (std::size_t i = 1; i < localLength; ++i) {
            localMin = std::min(localMin, localVec(i,0));
        }

        // Perform a global minimum reduction
        ST globalMin;
        Teuchos::reduceAll(*rX.getMap()->getComm(), Teuchos::REDUCE_MIN, localMin, Teuchos::outArg(globalMin));

        return globalMin;
    }

    /**
     * @brief Returns the norm of the vector rX
     * @details ||rX||2
     * @param rX The vector considered
     * @return The norm of the vector rX
     */
    inline static double TwoNorm(const VectorType& rX)
    {
        // FEMultiVector doesn't have a scalar-returning norm2(); compute via local views
        auto localView = rX.getLocalViewHost(Tpetra::Access::ReadOnly);
        ST localSumSq = Teuchos::ScalarTraits<ST>::zero();
        const auto n = rX.getLocalLength();
        for (std::size_t i = 0; i < n; ++i) {
            localSumSq += localView(i, 0) * localView(i, 0);
        }
        ST globalSumSq = Teuchos::ScalarTraits<ST>::zero();
        Teuchos::reduceAll(*rX.getMap()->getComm(), Teuchos::REDUCE_SUM, localSumSq, Teuchos::outArg(globalSumSq));
        return std::sqrt(static_cast<double>(globalSumSq));
    }

    /**
     * @brief Returns the Frobenius norm of the matrix rA
     * @details ||rA||2
     * @param rA The matrix considered
     * @return The Frobenius norm of the matrix rA
     */
    inline static double TwoNorm(const MatrixType& rA)
    {
        // Use the built-in getFrobeniusNorm() method to compute the Frobenius norm
        return rA.getFrobeniusNorm();
    }

    /**
     * @brief Returns the multiplication of a matrix by a vector
     * @details y = A*x
     * @param rA The matrix considered
     * @param rX The vector considered
     * @param rY The result of the multiplication
     */
    /**
     * @brief Returns the multiplication of a matrix by a vector
     * @details y = A*x
     * @param rA The matrix considered
     * @param rX The vector considered
     * @param rY The result of the multiplication
     */
    inline static void Mult(
        const MatrixType& rA,
        const VectorType& rX,
        VectorType& rY
        )
    {
        rA.apply(rX, rY);
    }

    /**
     * @brief Returns the multiplication matrix-matrix
     * @details C = A*B
     * @param rA The first matrix considered
     * @param rB The second matrix considered
     * @param rC The result of the multiplication
     * @param CallFillCompleteOnResult Whether to call fillComplete on the result matrix
     * @param KeepAllHardZeros If true, keeps all hard zeros in the result matrix
     */
    inline static void Mult(
        const MatrixType& rA,
        const MatrixType& rB,
        MatrixType& rC,
        const bool CallFillCompleteOnResult = true,
        const bool KeepAllHardZeros = false
        )
    {
        KRATOS_TRY

        // Use a temporary CrsMatrix to allow for dynamic sparsity
        Teuchos::RCP<CrsMatrixType> aux_C = Teuchos::rcp(new CrsMatrixType(rC.getRowMap(), 16));
        Tpetra::MatrixMatrix::Multiply(rA, false, rB, false, *aux_C, CallFillCompleteOnResult);

        // Copy values back to rC
        // Inline copy logic from CrsMatrixType to MatrixType
        if (!rC.isFillActive()) rC.resumeFill();
        auto p_fe_rC = dynamic_cast<MatrixType*>(&rC);
        if (p_fe_rC) p_fe_rC->beginAssembly();
        for (LO i = 0; i < static_cast<LO>(aux_C->getNodeNumRows()); ++i) {
            const auto global_row_index = aux_C->getRowMap()->getGlobalElement(i);
            typename MatrixType::local_inds_host_view_type local_cols;
            typename MatrixType::values_host_view_type vals;
            aux_C->getLocalRowView(i, local_cols, vals);
            if (vals.extent(0) > 0) {
                Teuchos::Array<GO> global_cols(local_cols.extent(0));
                for (std::size_t j = 0; j < static_cast<std::size_t>(local_cols.extent(0)); ++j) {
                    global_cols[j] = aux_C->getColMap()->getGlobalElement(local_cols(j));
                }
                rC.sumIntoGlobalValues(global_row_index, static_cast<LO>(global_cols.size()), vals.data(), global_cols.data());
            }
        }
        if (p_fe_rC) p_fe_rC->endAssembly();
        if (rC.isFillActive()) rC.fillComplete();

        KRATOS_CATCH("")
    }

    /**
     * @brief Returns the transpose multiplication of a matrix by a vector
     * @details y = AT*x
     * @param rA The matrix considered
     * @param rX The vector considered
     * @param rY The result of the multiplication
     */
    inline static void TransposeMult(
        const MatrixType& rA,
        const VectorType& rX,
        VectorType& rY
        )
    {
        // Use the apply method with the transpose flag set to true
        rA.apply(rX, rY, Teuchos::TRANS);
    }

    /**
     * @brief Returns the transpose multiplication matrix-matrix
     * @details C = A*B
     * @param rA The first matrix considered
     * @param rB The second matrix considered
     * @param rC The result of the multiplication
     * @param TransposeFlag Flags to transpose the matrices
     * @param CallFillCompleteOnResult	Optional argument, defaults to true. Power users may specify this argument to be false if they DON'T want this function to call C.FillComplete. (It is often useful to allow this function to call C.FillComplete, in cases where one or both of the input matrices are rectangular and it is not trivial to know which maps to use for the domain- and range-maps.)
     * @param KeepAllHardZeros	Optional argument, defaults to false. If true, Multiply, keeps all entries in C corresponding to hard zeros. If false, the following happens by case: A*B^T, A^T*B^T - Does not store entries caused by hard zeros in C. A^T*B (unoptimized) - Hard zeros are always stored (this option has no effect) A*B, A^T*B (optimized) - Hard zeros in corresponding to hard zeros in A are not stored, There are certain cases involving reuse of C, where this can be useful.
     */
    inline static void TransposeMult(
        const MatrixType& rA,
        const MatrixType& rB,
        MatrixType& rC,
        const std::pair<bool, bool> TransposeFlag = {false, false},
        const bool CallFillCompleteOnResult = true,
        const bool KeepAllHardZeros = false
        )
    {
        KRATOS_TRY

        // Use a temporary CrsMatrix to allow for dynamic sparsity
        Teuchos::RCP<CrsMatrixType> aux_C = Teuchos::rcp(new CrsMatrixType(rC.getRowMap(), 16));
        Tpetra::MatrixMatrix::Multiply(rA, TransposeFlag.first, rB, TransposeFlag.second, *aux_C, CallFillCompleteOnResult);

        // Copy values back to rC
        // Inline copy logic from CrsMatrixType to MatrixType
        if (!rC.isFillActive()) rC.resumeFill();
        auto p_fe_rC = dynamic_cast<MatrixType*>(&rC);
        if (p_fe_rC) p_fe_rC->beginAssembly();
        for (LO i = 0; i < static_cast<LO>(aux_C->getNodeNumRows()); ++i) {
            const auto global_row_index = aux_C->getRowMap()->getGlobalElement(i);
            typename MatrixType::local_inds_host_view_type local_cols;
            typename MatrixType::values_host_view_type vals;
            aux_C->getLocalRowView(i, local_cols, vals);
            if (vals.extent(0) > 0) {
                Teuchos::Array<GO> global_cols(local_cols.extent(0));
                for (std::size_t j = 0; j < static_cast<std::size_t>(local_cols.extent(0)); ++j) {
                    global_cols[j] = aux_C->getColMap()->getGlobalElement(local_cols(j));
                }
                rC.sumIntoGlobalValues(global_row_index, static_cast<LO>(global_cols.size()), vals.data(), global_cols.data());
            }
        }
        if (p_fe_rC) p_fe_rC->endAssembly();
        if (rC.isFillActive()) rC.fillComplete();

        KRATOS_CATCH("")
    }

    /**
     * @brief Calculates the product operation B'DB
     * @param rA The resulting matrix
     * @param rD The "center" matrix
     * @param rB The matrices to be transposed
     * @param CallFillCompleteOnResult	Optional argument, defaults to true. Power users may specify this argument to be false if they DON'T want this function to call C.FillComplete. (It is often useful to allow this function to call C.FillComplete, in cases where one or both of the input matrices are rectangular and it is not trivial to know which maps to use for the domain- and range-maps.)
     * @param KeepAllHardZeros	Optional argument, defaults to false. If true, Multiply, keeps all entries in C corresponding to hard zeros. If false, the following happens by case: A*B^T, A^T*B^T - Does not store entries caused by hard zeros in C. A^T*B (unoptimized) - Hard zeros are always stored (this option has no effect) A*B, A^T*B (optimized) - Hard zeros in corresponding to hard zeros in A are not stored, There are certain cases involving reuse of C, where this can be useful.
     * @param EnforceInitialGraph If the initial graph is enforced, or a new one is generated
     * @todo TpetraExt_TripleMatrixMultiply_def is failing compilation in the version of Trilinos available in Ubuntu 20.04. We cannot use TripleMatrixMultiply::MultiplyRAP
     */
    inline static void BtDBProductOperation(
        MatrixType& rA,
        const MatrixType& rD,
        const MatrixType& rB,
        const bool CallFillCompleteOnResult = true,
        const bool KeepAllHardZeros = false,
        const bool EnforceInitialGraph = false
        )
    {
        // Use temporary CrsMatrix for intermediate steps
        Teuchos::RCP<CrsMatrixType> aux_1 = Teuchos::rcp(new CrsMatrixType(rB.getDomainMap(), 16));
        Tpetra::MatrixMatrix::Multiply(rB, true, rD, false, *aux_1);

        Teuchos::RCP<CrsMatrixType> aux_2 = Teuchos::rcp(new CrsMatrixType(aux_1->getRowMap(), 16));
        Tpetra::MatrixMatrix::Multiply(*aux_1, false, rB, false, *aux_2);

        // We must ensure rA has enough space. 
        // If it is an FECrsMatrix, we might need to recreate it if the graph is too small.
        // But we try to copy values directly.
        // Inline copy logic from CrsMatrixType to MatrixType
        if (!rA.isFillActive()) rA.resumeFill();
        auto p_fe_A = dynamic_cast<MatrixType*>(&rA);
        if (p_fe_A) p_fe_A->beginAssembly();
        for (LO i = 0; i < static_cast<LO>(aux_2->getNodeNumRows()); ++i) {
            const auto global_row_index = aux_2->getRowMap()->getGlobalElement(i);
            typename MatrixType::local_inds_host_view_type local_cols;
            typename MatrixType::values_host_view_type vals;
            aux_2->getLocalRowView(i, local_cols, vals);
            if (vals.extent(0) > 0) {
                Teuchos::Array<GO> global_cols(local_cols.extent(0));
                for (std::size_t j = 0; j < static_cast<std::size_t>(local_cols.extent(0)); ++j) {
                    global_cols[j] = aux_2->getColMap()->getGlobalElement(local_cols(j));
                }
                rA.sumIntoGlobalValues(global_row_index, static_cast<LO>(global_cols.size()), vals.data(), global_cols.data());
            }
        }
        if (p_fe_A) p_fe_A->endAssembly();
        if (rA.isFillActive()) rA.fillComplete();
    }

    /**
     * @brief Calculates the product operation BDB'
     * @param rA The resulting matrix
     * @param rD The "center" matrix
     * @param rB The matrices to be transposed
     * @param CallFillCompleteOnResult	Optional argument, defaults to true. Power users may specify this argument to be false if they DON'T want this function to call C.FillComplete. (It is often useful to allow this function to call C.FillComplete, in cases where one or both of the input matrices are rectangular and it is not trivial to know which maps to use for the domain- and range-maps.)
     * @param KeepAllHardZeros	Optional argument, defaults to false. If true, Multiply, keeps all entries in C corresponding to hard zeros. If false, the following happens by case: A*B^T, A^T*B^T - Does not store entries caused by hard zeros in C. A^T*B (unoptimized) - Hard zeros are always stored (this option has no effect) A*B, A^T*B (optimized) - Hard zeros in corresponding to hard zeros in A are not stored, There are certain cases involving reuse of C, where this can be useful.
     * @param EnforceInitialGraph If the initial graph is enforced, or a new one is generated
     */
    inline static void BDBtProductOperation(
        MatrixType& rA,
        const MatrixType& rD,
        const MatrixType& rB,
        const bool CallFillCompleteOnResult = true,
        const bool KeepAllHardZeros = false,
        const bool EnforceInitialGraph = false
        )
    {
        // Use temporary CrsMatrix for intermediate steps
        Teuchos::RCP<CrsMatrixType> aux_1 = Teuchos::rcp(new CrsMatrixType(rB.getRowMap(), 16));
        Tpetra::MatrixMatrix::Multiply(rB, false, rD, false, *aux_1);

        Teuchos::RCP<CrsMatrixType> aux_2 = Teuchos::rcp(new CrsMatrixType(aux_1->getRowMap(), 16));
        Tpetra::MatrixMatrix::Multiply(*aux_1, false, rB, true, *aux_2);

        // Inline copy logic from CrsMatrixType to MatrixType
        if (!rA.isFillActive()) rA.resumeFill();
        auto p_fe_A = dynamic_cast<MatrixType*>(&rA);
        if (p_fe_A) p_fe_A->beginAssembly();
        for (LO i = 0; i < static_cast<LO>(aux_2->getNodeNumRows()); ++i) {
            const auto global_row_index = aux_2->getRowMap()->getGlobalElement(i);
            typename MatrixType::local_inds_host_view_type local_cols;
            typename MatrixType::values_host_view_type vals;
            aux_2->getLocalRowView(i, local_cols, vals);
            if (vals.extent(0) > 0) {
                Teuchos::Array<GO> global_cols(local_cols.extent(0));
                for (std::size_t j = 0; j < static_cast<std::size_t>(local_cols.extent(0)); ++j) {
                    global_cols[j] = aux_2->getColMap()->getGlobalElement(local_cols(j));
                }
                rA.sumIntoGlobalValues(global_row_index, static_cast<LO>(global_cols.size()), vals.data(), global_cols.data());
            }
        }
        if (p_fe_A) p_fe_A->endAssembly();
        if (rA.isFillActive()) rA.fillComplete();
    }

    /**
     * @brief Returns the multiplication of a vector by a scalar
     * @details y = A*x
     * Checks if a multiplication is needed and tries to do otherwise
     * @param rX The vector considered
     * @param A The scalar considered
     */
    inline static void InplaceMult(
        VectorType& rX,
        const double A
        )
    {
        if (A != 1.00) {
            // Scale the vector: x = A * x
            rX.scale(A);
        }
    }

    /**
     * @brief Returns the multiplication of a vector by a scalar
     * @details x = A*y
     * Checks if a multiplication is needed and tries to do otherwise
     * @note ATTENTION it is assumed no aliasing between rX and rY
     * @param rX The resulting vector considered
     * @param A The scalar considered
     * @param rY The multiplied vector considered
     */
    inline static void Assign(
        VectorType& rX,
        const double A,
        const VectorType& rY
        )
    {
        if (A != 1.00) {
            // Perform the operation x = A * y
            rX.update(A, rY, 0.0);
        } else {
            // FEMultiVector has deleted copy assignment; use update to copy
            rX.update(1.0, rY, 0.0);
        }
    }

    /**
     * @brief Returns the unaliased addition of a vector by a scalar times a vector
     * @details X += A*y;
     * Checks if a multiplication is needed and tries to do otherwise
     * @note ATTENTION it is assumed no aliasing between rX and rY
     * @param rX The resulting vector considered
     * @param A The scalar considered
     * @param rY The multiplied vector considered
     */
    inline static void UnaliasedAdd(
        VectorType& rX,
        const double A,
        const VectorType& rY
        )
    {
        rX.update(A, rY, 1.0);
    }

    /**
     * @brief Returns the unaliased addition of two vectors by a scalar
     * @details rZ = (A * rX) + (B * rY)
     * @param A The scalar considered
     * @param rX The first vector considered
     * @param B The scalar considered
     * @param rY The second vector considered
     * @param rZ The resulting vector considered
     */
    inline static void ScaleAndAdd(
        const double A,
        const VectorType& rX,
        const double B,
        const VectorType& rY,
        VectorType& rZ
        )
    {
        // Compute rZ = A * rX + B * rY
        rZ.update(A, rX, B, rY, 0.0);
    }

    /**
     * @brief Returns the unaliased addition of two vectors by a scalar
     * @details rY = (A * rX) + (B * rY)
     * @param A The scalar considered
     * @param rX The first vector considered
     * @param B The scalar considered
     * @param rY The resulting vector considered
     */
    inline static void ScaleAndAdd(
        const double A,
        const VectorType& rX,
        const double B,
        VectorType& rY
        )
    {
        // Compute rY = A * rX + B * rY
        rY.update(A, rX, B);
    }

    /**
     * @brief Returns the unaliased addition of two matrices by a scalar
     * @details rY = (A * rX) + (B * rY)
     * @param A The scalar considered
     * @param rX The first matrix considered
     * @param B The scalar considered
     * @param rY The resulting matrix considered
     */
    inline static void ScaleAndAdd(
        const double A,
        const MatrixType& rX,
        const double B,
        MatrixType& rY
        )
    {
        // Use a temporary CrsMatrix to allow for dynamic sparsity
        Teuchos::RCP<CrsMatrixType> aux_Y = Teuchos::rcp(new CrsMatrixType(rY.getRowMap(), 100));
        Tpetra::MatrixMatrix::Add(rX, false, A, rY, false, B, aux_Y);
        aux_Y->fillComplete();

        // Inline copy logic from CrsMatrixType to MatrixType
        MatrixType& rDest = rY;
        const CrsMatrixType& rSrc = *aux_Y;
        if (!rDest.isFillActive()) rDest.resumeFill();
        auto p_fe_Dest = dynamic_cast<MatrixType*>(&rDest);
        if (p_fe_Dest) {
            p_fe_Dest->beginAssembly();
        }
        for (LO i = 0; i < static_cast<LO>(rSrc.getNodeNumRows()); ++i) {
            const auto global_row_index = rSrc.getRowMap()->getGlobalElement(i);
            typename MatrixType::local_inds_host_view_type local_cols;
            typename MatrixType::values_host_view_type vals;
            rSrc.getLocalRowView(i, local_cols, vals);
            if (vals.extent(0) > 0) {
                Teuchos::Array<GO> global_cols(local_cols.extent(0));
                for (std::size_t j = 0; j < static_cast<std::size_t>(local_cols.extent(0)); ++j) {
                    global_cols[j] = rSrc.getColMap()->getGlobalElement(local_cols(j));
                }
                rDest.sumIntoGlobalValues(global_row_index, static_cast<LO>(global_cols.size()), vals.data(), global_cols.data());
            }
        }
        if (p_fe_Dest) {
            p_fe_Dest->endAssembly();
        }
        if (rDest.isFillActive()) rDest.fillComplete();
    }

    /**
     * @brief Sets a value in a vector
     * @param rX The vector considered
     * @param i The index of the value considered
     * @param value The value considered
     */
    inline static void SetValue(
        VectorType& rX,
        IndexType i,
        const double value
        )
    {
        // Get the local index corresponding to the global index `i`
        auto map = rX.getMap();
        IndexType localIndex = map->getLocalElement(i);

        // Check if the index is on this process
        if (localIndex != Tpetra::Details::OrdinalTraits<IndexType>::invalid()) {
            // Set the value at the specified local index
            rX.replaceLocalValue(localIndex, size_t(0), value);
        }
        // If the index `i` is not local, it is ignored (handled by Tpetra's parallel distribution)
    }

    /**
     * @brief Sets a value in a matrix
     * @param rA The matrix considered
     * @param i The row index
     * @param j The column index
     * @param value The value considered
     */
    inline static void SetValue(
        MatrixType& rA,
        IndexType i,
        IndexType j,
        const double value
        )
    {
        if (!rA.isFillActive()) {
            rA.resumeFill();
        }
        rA.beginAssembly();
        const GO globalRow = static_cast<GO>(i);
        const GO globalCol = static_cast<GO>(j);
        const ST val = static_cast<ST>(value);
        rA.replaceGlobalValues(globalRow, 1, &val, &globalCol);
    }

    /**
     * @brief assigns a scalar to a vector
     * @details rX = A
     * @param rX The vector considered
     * @param A The scalar considered
     */
    inline static void Set(
        VectorType& rX,
        const DataType A
        )
    {
        rX.putScalar(static_cast<ST>(A));
    }

    /**
     * @brief Resizes a matrix
     * @param rA The matrix to be resized
     * @param m The new number of rows
     * @param n The new number of columns
     */
    inline static void Resize(
        MatrixType& rA,
        const SizeType m,
        const SizeType n
        )
    {
        KRATOS_ERROR << "Resize is not defined for Trilinos Sparse Matrix" << std::endl;
    }

    /**
     * @brief Resizes a vector
     * @param rX The vector to be resized
     * @param n The new size
     */
    inline static void Resize(
        VectorType& rX,
        const SizeType n
        )
    {
        KRATOS_ERROR << "Resize is not defined for a reference to Trilinos Vector - need to use the version passing a Pointer" << std::endl;
    }

    /**
     * @brief Resizes a vector
     * @param pA The pointer to the vector to be resized
     * @param n The new size
    */
    inline static void Resize(
        VectorPointerType pX,
        const SizeType n
        )
    {
        //KRATOS_ERROR_IF(pX != Teuchos::null) << "Trying to resize a null pointer" << std::endl;
        int global_elems = n;
        auto map = Teuchos::rcp(new MapType(0, 0, pX->getMap()->getComm()));
        VectorPointerType pNewEmptyX = CreateVector(map);
        pX.swap(pNewEmptyX);
    }

    /**
     * @brief Clears a matrix
     * @param pA The pointer to the matrix to be cleared
     */
    inline static void Clear(MatrixPointerType pA)
    {
        if(pA != Teuchos::null) {
            auto map = Teuchos::rcp(new MapType(0, 0, pA->getMap()->getComm()));
            GraphPointerType graph = Teuchos::rcp(new GraphType(map, map, 0));
            MatrixPointerType pNewEmptyA = Teuchos::rcp(new MatrixType(graph));
            pA.swap(pNewEmptyA);
        }
    }

    /**
     * @brief Clears a vector
     * @param pX The pointer to the vector to be cleared
     */
    inline static void Clear(VectorPointerType pX)
    {
        if(pX != Teuchos::null) {
            auto map = Teuchos::rcp(new MapType(0, 0, pX->getMap()->getComm()));
            VectorPointerType pNewEmptyX = CreateVector(map);
            pX.swap(pNewEmptyX);
        }
    }

    /**
     * @brief Sets a matrix to zero
     * @param rX The matrix to be set
     */
    inline static void SetToZero(MatrixType& rA)
    {
        // Set all values in the matrix to zero.
        if (!rA.isFillActive()) {
            rA.resumeFill();
        }
        rA.beginAssembly();
        rA.setAllToScalar(0.0);
    }

    /**
     * @brief Sets a vector to zero
     * @param rX The vector to be set
     */
    inline static void SetToZero(VectorType& rX)
    {
        auto p_fe_rX = dynamic_cast<Tpetra::FEMultiVector<ST, LO, GO, NT>*>(&rX);
        if (p_fe_rX) {
            p_fe_rX->beginAssembly();
        }
        rX.putScalar(static_cast<ST>(0));
    }

    /// TODO: creating the the calculating reaction version
    // 	template<class TOtherMatrixType, class TEquationIdVectorType>

    /**
     * @brief Assembles the LHS of the system
     * @param rA The LHS matrix
     * @param rLHSContribution The contribution to the LHS
     * @param rEquationId The equation ids
     */
    inline static void AssembleLHS(
        MatrixType& rA,
        const Matrix& rLHSContribution,
        const std::vector<std::size_t>& rEquationId
        )
    {
        const std::size_t system_size = rA.getGlobalNumRows();

        // Count active indices
        std::vector<LO> indices;
        for (std::size_t i = 0; i < rEquationId.size(); ++i) {
            if (rEquationId[i] < system_size) {
                indices.push_back(static_cast<LO>(rEquationId[i]));
            }
        }

        if (!indices.empty()) {
            std::vector<GO> global_indices(indices.size());
            for (std::size_t i = 0; i < indices.size(); ++i) {
                global_indices[i] = static_cast<GO>(indices[i]);
            }

            for (std::size_t i = 0; i < indices.size(); ++i) {
                const GO globalRow = global_indices[i];
                std::vector<ST> row_values(indices.size());
                for (std::size_t j = 0; j < indices.size(); ++j) {
                    row_values[j] = rLHSContribution(i, j);
                }
                const int ierr = rA.sumIntoGlobalValues(globalRow, static_cast<LO>(global_indices.size()), row_values.data(), global_indices.data());
                // Note: sumIntoGlobalValues might return the number of values successfully summed instead of an error code 0 or -1. 
                // Epetra returns 0, Tpetra returns the number of values (indices.size()) if successful.
                KRATOS_ERROR_IF(ierr != static_cast<int>(indices.size())) << "Tpetra failure found" << std::endl;
            }
        }
    }

    //***********************************************************************
    /// TODO: creating the the calculating reaction version
    // 	template<class TOtherVectorType, class TEquationIdVectorType>

    /**
     * @brief Assembles the RHS of the system
     * @param rb The RHS vector
     * @param rRHSContribution The RHS contribution
     * @param rEquationId The equation ids
     */
    inline static void AssembleRHS(
        VectorType& rb,
        const Vector& rRHSContribution,
        const std::vector<std::size_t>& rEquationId
        )
    {
        const std::size_t system_size = rb.getGlobalLength();

        // Count active indices
        std::vector<LO> indices;
        for (std::size_t i = 0; i < rEquationId.size(); ++i) {
            if (rEquationId[i] < system_size) {
                indices.push_back(static_cast<LO>(rEquationId[i]));
            }
        }

        if (!indices.empty()) {
            std::vector<GO> global_indices(indices.size());
            std::vector<ST> values(indices.size());
            for (std::size_t i = 0; i < indices.size(); ++i) {
                global_indices[i] = static_cast<GO>(indices[i]);
                values[i] = rRHSContribution[i];
            }
            for (std::size_t i = 0; i < global_indices.size(); ++i) {
                rb.sumIntoGlobalValue(global_indices[i], size_t(0), values[i]);
            }
        }
    }

    /**
     * @brief This function returns if we are in a distributed system
     * @return True if we are in a distributed system, false otherwise (always true in this case)
     */
    inline static constexpr bool IsDistributed()
    {
        return true;
    }

    /**
     * @brief Returns a list of the fastest direct solvers.
     * @details This function returns a vector of strings representing the names of the fastest direct solvers. The order of the solvers in the list may need to be updated and reordered depending on the size of the equation system.
     * @return A vector of strings containing the names of the fastest direct solvers.
     */
    inline static std::vector<std::string> FastestDirectSolverList()
    {
        // May need to be updated and reordered. In fact I think it depends of the size of the equation system
        std::vector<std::string> faster_direct_solvers({
            "mumps2",         // Amesos2 (if compiled with MUMPS-support)
            "mumps",          // Amesos (if compiled with MUMPS-support)
            "super_lu_dist2", // Amesos2 SuperLUDist (if compiled with MPI-support)
            "super_lu_dist",  // Amesos SuperLUDist (if compiled with MPI-support)
            "amesos2",        // Amesos2
            "amesos",         // Amesos
            "klu2",           // Amesos2 KLU
            "klu",            // Amesos KLU
            "basker"          // Amesos2 Basker
        });
        return faster_direct_solvers;
    }

    /**
     * @brief This function returns a value from a given vector according to a given index
     * @param rX The vector from which values are to be gathered
     * @param I The index of the value to be gathered
     * @return The value of the vector corresponding to the index I
     */
    inline static double GetValue(
        const VectorType& rX,
        const std::size_t I
        )
    {
        // Get the local index corresponding to the global index `I`
        auto map = rX.getMap();
        IndexType localIndex = map->getLocalElement(static_cast<IndexType>(I));

        // Index must be local to this proc
        KRATOS_ERROR_IF(localIndex == Tpetra::Details::OrdinalTraits<IndexType>::invalid()) << " non-local id: " << I << "." << std::endl;

        // Get the value at the specified local index via local view (FEMultiVector-compatible)
        auto localView = rX.getLocalViewHost(Tpetra::Access::ReadOnly);
        return static_cast<double>(localView(localIndex, 0));
    }

    /**
     * @brief This function gathers the values of a given vector according to a given index array
     * @param rX The vector from which values are to be gathered
     * @param IndexArray The array containing the indices of the values to be gathered
     * @param pValues The array containing the gathered values
     */
    inline static void GatherValues(
        const VectorType& rX,
        const std::vector<int>& IndexArray,
        double* pValues
        )
    {
        KRATOS_TRY

        // Get the total size of the index array
        const std::size_t tot_size = IndexArray.size();

        // Create a Map with the desired indices
        Teuchos::ArrayView<const IndexType> indexArrayView(IndexArray.data(), IndexArray.size());
        MapPointerType dof_update_map = Tpetra::createNonContigMapWithNode<IndexType, IndexType>(indexArrayView, rX.getMap()->getComm());

        // Define the Importer
        Tpetra::Import<IndexType, IndexType> importer(dof_update_map, rX.getMap());

        // Create a temporary vector to gather the values
        VectorType temp(dof_update_map);

        // Import the values from rX into the temp vector
        temp.doImport(rX, importer, Tpetra::INSERT);

        // Extract the values from the temp vector
        temp.get1dCopy(Teuchos::ArrayView<double>(pValues, tot_size));

        // Synchronize processes
        rX.getMap()->getComm()->barrier();

        KRATOS_CATCH("")
    }

    /**
     * @brief Read a matrix from a MatrixMarket file
     * @param rFileName The name of the file to read
     * @param rComm The MPI communicator
     * @return The matrix read from the file
     */
    inline static MatrixPointerType ReadMatrixMarket(const std::string& FileName, CommunicatorType& rComm)
    {
        KRATOS_ERROR << "MatrixMarket not built due to internal conflicts" << std::endl;
        return CreateEmptyMatrixPointer();
    }

    /**
     * @brief Read a vector from a MatrixMarket file
     * @param rFileName The name of the file to read
     * @param pComm The MPI communicator
     * @param N The size of the vector
     */
    inline static VectorPointerType ReadMatrixMarketVector(const std::string& FileName, CommunicatorPointerType pComm, const int n)
    {
        KRATOS_ERROR << "MatrixMarket not built due to internal conflicts" << std::endl;
        return CreateEmptyVectorPointer();
    }

    /**
    * @brief Generates a graph combining the graphs of two matrices
    * @param rA The first matrix
    * @param rB The second matrix
    */

    static GraphPointerType CombineMatricesGraphs(
        const MatrixType& rA,
        const MatrixType& rB
        )
    {
        // Row maps must be the same
        KRATOS_ERROR_IF(!rA.getRowMap()->isSameAs(*rB.getRowMap())) << "Row maps are not compatible" << std::endl;

        // Getting the graphs
        auto p_graph_a = rA.getCrsGraph();
        auto p_graph_b = rB.getCrsGraph();

        // Getting the maps
        const auto& r_row_map = rA.getRowMap();

        // New graph with large capacity
        Teuchos::RCP<GraphType> graph = Teuchos::rcp(new GraphType(r_row_map, r_row_map, 100));

        const auto numLocalRows = r_row_map->getNodeNumElements();

        // Combine graphs using global indexing
        for (LO i = 0; i < static_cast<LO>(numLocalRows); ++i) {
            const auto global_row_index = r_row_map->getGlobalElement(i);
            std::set<GO> combined_indices;

            if (p_graph_a->isLocallyIndexed()) {
                typename MatrixType::local_inds_host_view_type cols_a;
                p_graph_a->getLocalRowView(i, cols_a);
                for (std::size_t j = 0; j < static_cast<std::size_t>(cols_a.extent(0)); ++j) combined_indices.insert(p_graph_a->getColMap()->getGlobalElement(cols_a(j)));
            } else {
                typename MatrixType::global_inds_host_view_type cols_a;
                p_graph_a->getGlobalRowView(global_row_index, cols_a);
                for (std::size_t j = 0; j < static_cast<std::size_t>(cols_a.extent(0)); ++j) combined_indices.insert(cols_a(j));
            }

            if (p_graph_b->isLocallyIndexed()) {
                typename MatrixType::local_inds_host_view_type cols_b;
                p_graph_b->getLocalRowView(i, cols_b);
                for (std::size_t j = 0; j < static_cast<std::size_t>(cols_b.extent(0)); ++j) combined_indices.insert(p_graph_b->getColMap()->getGlobalElement(cols_b(j)));
            } else {
                typename MatrixType::global_inds_host_view_type cols_b;
                p_graph_b->getGlobalRowView(global_row_index, cols_b);
                for (std::size_t j = 0; j < static_cast<std::size_t>(cols_b.extent(0)); ++j) combined_indices.insert(cols_b(j));
            }

            std::vector<GO> combined_indices_vector(combined_indices.begin(), combined_indices.end());
            graph->insertGlobalIndices(global_row_index, Teuchos::ArrayView<const GO>(combined_indices_vector));
        }

        if (graph->isFillActive()) graph->fillComplete(r_row_map, r_row_map);
        return graph;
    }

    /**
     * @brief Copy values from one matrix to another
     * @details It is assumed that the sparsity of both matrices is compatible
     * @param rA The matrix where assigning values
     * @param rB The matrix to be copied
     */
    inline static void CopyMatrixValues(
        MatrixType& rA,
        const MatrixType& rB
        )
    {
        // Cleaning destination matrix
        SetToZero(rA);

        // Begin matrix assembly if FECrsMatrix
        auto p_fe_A = dynamic_cast<MatrixType*>(&rA);
        if (p_fe_A) {
            p_fe_A->beginAssembly();
        } else {
            if (!rA.isFillActive()) rA.resumeFill();
        }

        for (LO i = 0; i < static_cast<LO>(rB.getNodeNumRows()); ++i) {
            const auto global_row_index = rB.getRowMap()->getGlobalElement(i);
            typename MatrixType::local_inds_host_view_type local_cols_b;
            typename MatrixType::values_host_view_type vals;
            rB.getLocalRowView(i, local_cols_b, vals);

            if (vals.extent(0) > 0) {
                Teuchos::Array<GO> global_cols(local_cols_b.extent(0));
                for (std::size_t j = 0; j < static_cast<std::size_t>(local_cols_b.extent(0)); ++j) {
                    global_cols[j] = rB.getColMap()->getGlobalElement(local_cols_b(j));
                }

                // Sum values into global matrix using global row and column indices
                rA.sumIntoGlobalValues(global_row_index, static_cast<LO>(global_cols.size()), vals.data(), global_cols.data());
            }
        }

        // Finalizing the fill process
        if (p_fe_A) {
            p_fe_A->endAssembly();
        }
        if (rA.isFillActive()) rA.fillComplete();
    }

    /**
     * @brief This method checks and corrects the zero diagonal values
     * @details This method returns the scale norm considering scaling the diagonal
     * @param rProcessInfo The problem process info
     * @param rA The LHS matrix
     * @param rb The RHS vector
     * @param ScalingDiagonal The type of scaling diagonal considered
     * @return The scale norm
     */
    inline static double CheckAndCorrectZeroDiagonalValues(
        const ProcessInfo& rProcessInfo,
        MatrixType& rA,
        VectorType& rb,
        const SCALING_DIAGONAL ScalingDiagonal = SCALING_DIAGONAL::NO_SCALING
        )
    {
        KRATOS_TRY

        // Define zero value tolerance
        const double zero_tolerance = std::numeric_limits<double>::epsilon();

        // The diagonal considered
        const double scale_factor = GetScaleNorm(rProcessInfo, rA, ScalingDiagonal);

        auto localMatrix = rA.getLocalMatrixHost();
        auto rowMap = rA.getRowMap();
        auto colMap = rA.getColMap();
        auto localRhs = rb.getLocalViewHost(Tpetra::Access::ReadWrite);

        for (int i = 0; i < localMatrix.numRows(); ++i) {
            auto localRow = localMatrix.row(i);
            const auto row_gid = rowMap->getGlobalElement(i);
            bool empty = true;
            int j;
            for (j = 0; j < localRow.length; ++j) {
                const auto col_gid = colMap->getGlobalElement(localRow.colidx(j));
                // Check diagonal value
                if (col_gid == row_gid) {
                    if (std::abs(localRow.value(j)) > zero_tolerance) {
                        empty = false;
                    }
                    break;
                }
            }

            // If diagonal empty assign scale factor
            if (empty) {
                const int row_gid_int = static_cast<int>(row_gid);  // Casting to int
                localMatrix.replaceValues(i, &row_gid_int, 1, &scale_factor, false, true);
                localRhs(i, 0) = 0.0;
            }
        }

        return scale_factor;

        KRATOS_CATCH("")
    }

    /**
     * @brief This method returns the scale norm considering for scaling the diagonal
     * @param rProcessInfo The problem process info
     * @param rA The LHS matrix
     * @param ScalingDiagonal The type of scaling diagonal considered
     * @return The scale norm
     */
    inline static double GetScaleNorm(
        const ProcessInfo& rProcessInfo,
        const MatrixType& rA,
        const SCALING_DIAGONAL ScalingDiagonal = SCALING_DIAGONAL::NO_SCALING
        )
    {
        KRATOS_TRY

        switch (ScalingDiagonal) {
            case SCALING_DIAGONAL::NO_SCALING:
                return 1.0;
            case SCALING_DIAGONAL::CONSIDER_PRESCRIBED_DIAGONAL: {
                KRATOS_ERROR_IF_NOT(rProcessInfo.Has(BUILD_SCALE_FACTOR)) << "Scale factor not defined at process info" << std::endl;
                return rProcessInfo.GetValue(BUILD_SCALE_FACTOR);
            }
            case SCALING_DIAGONAL::CONSIDER_NORM_DIAGONAL:
                return GetDiagonalNorm(rA)/static_cast<double>(Size1(rA));
            case SCALING_DIAGONAL::CONSIDER_MAX_DIAGONAL:
                return GetMaxDiagonal(rA);
            default:
                return GetMaxDiagonal(rA);
        }

        KRATOS_CATCH("");
    }

    /**
    * @brief This method returns the diagonal norm considering for scaling the diagonal
    * @param rA The LHS matrix
    * @return The diagonal norm
    */
    inline static double GetDiagonalNorm(const MatrixType& rA)
    {
        KRATOS_TRY

        // Create a plain Vector (not FEMultiVector) for diagonal copy — getLocalDiagCopy requires Vector
        Tpetra::Vector<ST, LO, GO, NT> diag(rA.getRowMap());

        // Extract the diagonal entries
        rA.getLocalDiagCopy(diag);

        // Get the local view of the diagonal
        auto diagLocalView = diag.getLocalViewHost(Tpetra::Access::ReadOnly);

        // Compute the local sum of squares of the diagonal
        ST localSumOfSquares = Teuchos::ScalarTraits<ST>::zero();  // Initialize to 0

        auto numLocalEntries = diag.getLocalLength();
        ST value = 0.0;
        for (std::size_t i = 0; i < numLocalEntries; ++i) {
            value = diagLocalView(i, 0);
            localSumOfSquares += value * value;  // Sum of squares
        }

        // Perform a global reduction to sum the squares across all processes
        ST globalSumOfSquares = 0.0;
        Teuchos::reduceAll(*rA.getMap()->getComm(), Teuchos::REDUCE_SUM, localSumOfSquares, Teuchos::outArg(globalSumOfSquares));

        // Compute the two-norm by taking the square root of the global sum of squares
        return std::sqrt(globalSumOfSquares);

        KRATOS_CATCH("");
    }

    /**
    * @brief This method returns the diagonal max value
    * @param rA The LHS matrix
    * @return The diagonal max value
    */
    inline static double GetAveragevalueDiagonal(const MatrixType& rA)
    {
        KRATOS_TRY

        return 0.5 * (GetMaxDiagonal(rA) + GetMinDiagonal(rA));

        KRATOS_CATCH("");
    }

    /**
    * @brief This method returns the diagonal max value
    * @param rA The LHS matrix
    * @return The diagonal max value
    */
    inline static double GetMaxDiagonal(const MatrixType& rA)
    {
        KRATOS_TRY

        // Create a plain Vector (not FEMultiVector) for diagonal copy — getLocalDiagCopy requires Vector
        Tpetra::Vector<ST, LO, GO, NT> diag(rA.getRowMap());

        // Extract the diagonal entries
        rA.getLocalDiagCopy(diag);

        // Get the local view of the diagonal
        auto diagLocalView = diag.getLocalViewHost(Tpetra::Access::ReadOnly);

        // Find the local maximum value
        ST localMax = Teuchos::ScalarTraits<ST>::zero();  // Initialize to 0

        auto numLocalEntries = diag.getLocalLength();
        ST value = 0.0;
        for (std::size_t i = 0; i < numLocalEntries; ++i) {
            value = diagLocalView(i, 0);
            if (value > localMax) {
                localMax = value;
            }
        }

        // Perform a global reduction to find the global maximum
        double globalMax = 0.0;
        Teuchos::reduceAll(*rA.getMap()->getComm(), Teuchos::REDUCE_MAX, localMax, Teuchos::outArg(globalMax));

        return globalMax;

        KRATOS_CATCH("");
    }

    /**
    * @brief This method returns the diagonal min value
    * @param rA The LHS matrix
    * @return The diagonal min value
    */
    inline static double GetMinDiagonal(const MatrixType& rA)
    {
        KRATOS_TRY

        // Create a plain Vector (not FEMultiVector) for diagonal copy — getLocalDiagCopy requires Vector
        Tpetra::Vector<ST, LO, GO, NT> diag(rA.getRowMap());

        // Extract the diagonal entries
        rA.getLocalDiagCopy(diag);

        // Get the local view of the diagonal
        auto diagLocalView = diag.getLocalViewHost(Tpetra::Access::ReadOnly);

        // Find the local minimum value
        ST localMin = Teuchos::ScalarTraits<ST>::rmax();  // Initialize to the max possible value

        auto numLocalEntries = diag.getLocalLength();
        ST value = 0.0;
        for (std::size_t i = 0; i < numLocalEntries; ++i) {
            value = diagLocalView(i, 0);
            if (value < localMin) {
                localMin = value;
            }
        }

        // Perform a global reduction to find the global minimum
        double globalMin = 0.0;
        Teuchos::reduceAll(*rA.getMap()->getComm(), Teuchos::REDUCE_MIN, localMin, Teuchos::outArg(globalMin));

        return globalMin;

        KRATOS_CATCH("");
    }

   /**
    * @brief Check if the TrilinosSpaceExperimental is distributed.
    * @details This static member function checks whether the TrilinosSpaceExperimental is distributed or not.
    * @return True if the space is distributed, false otherwise.
    */
    static constexpr bool IsDistributedSpace()
    {
        return true;
    }

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Input and output
    ///@{

    /**
     * @brief Turn back information as a string.
     * @return Info as a string.
     */
    virtual std::string Info() const
    {
        return "TrilinosSpaceExperimental";
    }

    /**
     * @brief Print information about this object.
     * @param rOStream The output stream to print on.
     */
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << "TrilinosSpaceExperimental";
    }

    /**
     * @brief Print object's data.
     * @param rOStream The output stream to print on.
     */
    virtual void PrintData(std::ostream& rOStream) const
    {
    }

    /**
     * @brief Writes a matrix to a file in MatrixMarket format
     * @param pFileName The name of the file to be written
     * @param rM The matrix to be written
     * @param Symmetric If the matrix is symmetric
     * @return True if the file was successfully written, false otherwise
     */
    static void WriteMatrixMarketMatrix(const char* FileName, const MatrixType& rA, const bool symmetric)
    {
        KRATOS_ERROR << "MatrixMarket not built due to internal conflicts" << std::endl;
    }

    /**
     * @brief Writes a vector to a file in MatrixMarket format
     * @param pFileName The name of the file to be written
     * @param rV The vector to be written
     * @return True if the file was successfully written, false otherwise
     */
    static void WriteMatrixMarketVector(
        const char* pFileName,
        const VectorType& rV
        )
    {
        KRATOS_ERROR << "MatrixMarket not built due to internal conflicts" << std::endl;
    }

    /**
     * @brief Creates a new dof updater
     * @return The new dof updater
     */
    inline static DofUpdaterPointerType CreateDofUpdater()
    {
        return DofUpdaterPointerType(new DofUpdater<TrilinosSpaceExperimental<TMatrixType, TVectorType>>());
    }

    /**
     * @brief Returns a Tpetra map for this rank's local rows starting at FirstMyId.
     */
    static MapPointerType GetOrCreateTpetraMap(
        CommunicatorType& rComm,
        const IndexType LocalSize,
        const int FirstMyId)
    {
        std::vector<GO> local_ids(LocalSize);
        for (IndexType i = 0; i < LocalSize; ++i) {
            local_ids[i] = static_cast<GO>(FirstMyId + static_cast<int>(i));
        }
        return Teuchos::rcp(new MapType(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            Teuchos::ArrayView<const GO>(local_ids.data(), static_cast<int>(local_ids.size())),
            0,
            Teuchos::rcp(&rComm, false)));
    }

    /**
     * @brief This method returns the map of the vector
     * @param rV The vector considered
     * @return The map of the vector
     */
    inline static const MapType& GetMap(const VectorType& rV)
    {
        return *(rV.getMap());
    }

    /**
     * @brief This method returns the communicator of the vector
     * @param rV The vector considered
     * @return The communicator of the vector
     */
    inline static const CommunicatorType& GetCommunicator(const VectorType& rV)
    {
        return dynamic_cast<const CommunicatorType&>(*(rV.getMap()->getComm()));
    }

    /**
     * @brief This method returns the communicator of the matrix
     * @param rA The matrix considered
     * @return The communicator of the matrix
     */
    inline static const CommunicatorType& GetCommunicator(const MatrixType& rA)
    {
        return dynamic_cast<const CommunicatorType&>(*(rA.getMap()->getComm()));
    }

    /// @brief Global assembly on a Tpetra FECrsMatrix - no-op (lifecycle managed by structure builders).
    static void GlobalAssemble(MatrixType& rA)
    {
    }

    /// @brief Global assembly on a Tpetra Vector.
    static void GlobalAssemble(VectorType& rV)
    {
        auto p_fe_rb = dynamic_cast<Tpetra::FEMultiVector<ST, LO, GO, NT>*>(&rV);
        if (p_fe_rb) {
            p_fe_rb->endAssembly();
        }
    }

    /**
     * @brief Manually finalizes matrix assembly.
     */
    static void ManualFinalize(MatrixType& rA)
    {
        rA.endAssembly();
        if (rA.isFillActive()) {
            rA.fillComplete();
        }
    }

    /**
     * @brief Build Tpetra FECrsGraph and create new system matrix + vectors.
     */
    static void BuildSystemStructure(
        CommunicatorType& rComm,
        const IndexType LocalSize,
        const int FirstMyId,
        const int GuessRowSize,
        const std::vector<std::vector<int>>& rAllEquationIds,
        MatrixPointerType& rpA,
        VectorPointerType& rpb,
        VectorPointerType& rpDx,
        VectorPointerType& rpReactions,
        const IndexType equationSystemSize,
        MapPointerType pMap)
    {
        Teuchos::RCP<GraphType> graph = Teuchos::rcp(new GraphType(pMap, pMap, GuessRowSize));
        std::vector<GO> gids;
        for (const auto& eq_ids : rAllEquationIds) {
            if (eq_ids.empty()) continue;
            gids.resize(eq_ids.size());
            for (std::size_t k = 0; k < eq_ids.size(); ++k) gids[k] = static_cast<GO>(eq_ids[k]);
            for (std::size_t row = 0; row < gids.size(); ++row) {
                graph->insertGlobalIndices(gids[row],
                    Teuchos::ArrayView<const GO>(gids.data(), static_cast<int>(gids.size())));
            }
        }
        graph->endAssembly();
        graph->fillComplete();
        rpA = Teuchos::rcp(new MatrixType(Teuchos::rcp_const_cast<const GraphType>(graph)));
        if (!rpb || Size(*rpb) != equationSystemSize)
            rpb = CreateVector(pMap);
        if (!rpDx || Size(*rpDx) != equationSystemSize)
            rpDx = CreateVector(pMap);
        if (!rpReactions)
            rpReactions = CreateVector(pMap);
    }

    /**
     * @brief Apply Dirichlet conditions on a Tpetra system using local row iteration.
     */
    static void ApplyDirichletConditionsTpetra(
        MatrixType& rA,
        VectorType& rb,
        const std::vector<int>& rGlobalIds,
        const std::vector<int>& rIsFixed,
        const ProcessInfo& rProcessInfo,
        const SCALING_DIAGONAL scalingDiagonal,
        double& rScaleFactor)
    {
        rScaleFactor = ClassType::CheckAndCorrectZeroDiagonalValues(rProcessInfo, rA, rb, scalingDiagonal);
        std::unordered_map<GO, int> is_fixed_map;
        for (std::size_t i = 0; i < rGlobalIds.size(); ++i)
            is_fixed_map[static_cast<GO>(rGlobalIds[i])] = rIsFixed[i];
        auto p_row_map = rA.getRowMap();
        const LO num_local_rows = static_cast<LO>(p_row_map->getNodeNumElements());
        auto rb_view = rb.getLocalViewHost(Tpetra::Access::ReadWrite);
        for (LO local_row = 0; local_row < num_local_rows; ++local_row) {
            const GO global_row = p_row_map->getGlobalElement(local_row);
            const bool row_is_fixed = is_fixed_map.count(global_row) > 0 && is_fixed_map.at(global_row) != 0;
            typename MatrixType::local_inds_host_view_type cols_view;
            typename MatrixType::values_host_view_type vals_view;
            rA.getLocalRowView(local_row, cols_view, vals_view);
            const LO num_entries = static_cast<LO>(cols_view.size());
            if (num_entries == 0) continue;
            auto col_map = rA.getColMap();
            std::vector<ST> new_vals(num_entries);
            if (!row_is_fixed) {
                for (LO j = 0; j < num_entries; ++j) {
                    const GO global_col = col_map->getGlobalElement(cols_view(j));
                    new_vals[j] = (is_fixed_map.count(global_col) > 0 && is_fixed_map.at(global_col) != 0) ? ST(0.0) : vals_view(j);
                }
            } else {
                rb_view(local_row, 0) = ST(0.0);
                for (LO j = 0; j < num_entries; ++j) {
                    const GO global_col = col_map->getGlobalElement(cols_view(j));
                    new_vals[j] = (global_col == global_row) ? vals_view(j) : ST(0.0);
                }
            }
            rA.replaceLocalValues(local_row, num_entries, new_vals.data(), cols_view.data());
        }
    }

    /**
     * @brief Build a Tpetra FE constraint graph and create T matrix + constant vector.
     */
    static void BuildConstraintsStructure(
        CommunicatorType& rComm,
        const IndexType LocalSize,
        const int FirstMyId,
        const int GuessRowSize,
        const std::vector<std::vector<int>>& rSlaveEquationIds,
        const std::vector<std::vector<int>>& rMasterEquationIds,
        MatrixPointerType& rpT,
        VectorPointerType& rpConstantVector,
        MapPointerType pMap)
    {
        Teuchos::RCP<GraphType> graph = Teuchos::rcp(new GraphType(pMap, pMap, GuessRowSize));
        for (IndexType i = 0; i < LocalSize; ++i) {
            const GO gid = static_cast<GO>(FirstMyId + static_cast<int>(i));
            graph->insertGlobalIndices(gid, Teuchos::ArrayView<const GO>(&gid, 1));
        }
        for (std::size_t c = 0; c < rSlaveEquationIds.size(); ++c) {
            const auto& slave_ids = rSlaveEquationIds[c];
            const auto& master_ids = rMasterEquationIds[c];
            if (slave_ids.empty() || master_ids.empty()) continue;
            std::vector<GO> master_gids(master_ids.size());
            for (std::size_t k = 0; k < master_ids.size(); ++k) master_gids[k] = static_cast<GO>(master_ids[k]);
            for (int slave_id : slave_ids) {
                const GO slave_gid = static_cast<GO>(slave_id);
                graph->insertGlobalIndices(slave_gid,
                    Teuchos::ArrayView<const GO>(master_gids.data(), static_cast<int>(master_gids.size())));
            }
        }
        graph->endAssembly();
        graph->fillComplete();
        rpT = Teuchos::rcp(new MatrixType(Teuchos::rcp_const_cast<const GraphType>(graph)));
        rpConstantVector = CreateVector(pMap);
    }

    ///@}

    /// @brief Creates an empty VectorType from a map. Handles both Tpetra::FEMultiVector (needs importer+numVecs) and plain Vector/MultiVector.
    inline static VectorPointerType CreateVector(const Teuchos::RCP<const MapType>& pMap)
    {
        if constexpr (std::is_same_v<VectorType, Tpetra::FEMultiVector<ST, LO, GO, NT>>) {
            return Teuchos::rcp(new VectorType(pMap, Teuchos::null, 1));
        } else {
            return Teuchos::rcp(new VectorType(pMap));
        }
    }

private:
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
    TrilinosSpaceExperimental & operator=(TrilinosSpaceExperimental const& rOther);

    /// Copy constructor.
    TrilinosSpaceExperimental(TrilinosSpaceExperimental const& rOther);

}; // Class TrilinosSpaceExperimental

///@}

} // namespace Kratos.