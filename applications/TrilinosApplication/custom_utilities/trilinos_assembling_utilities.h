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

// Project includes
#include "includes/define.h"

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
 * @class TrilinosAssemblingUtilities
 * @ingroup TrilinosApplication
 * @brief The Trilinos assembling utilities
 * @author Vicente Mataix Ferrandiz
 */
template<class TSparseSpace>
class TrilinosAssemblingUtilities
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of TrilinosAssemblingUtilities
    KRATOS_CLASS_POINTER_DEFINITION(TrilinosAssemblingUtilities);

    /// Definition of the matrix type
    using MatrixType = typename TSparseSpace::MatrixType;

    /// Definition of the vector type
    using VectorType = typename TSparseSpace::VectorType;

    /// Definition of the index type
    using IndexType = std::size_t;

    /// Definition of the size type
    using SizeType = std::size_t;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    TrilinosAssemblingUtilities() = delete;

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Assembles the relation matrix T of the system with MPC
     * @param rT The T relation matrix
     * @param rTContribution The contribution to the T
     * @param rSlaveEquationId The slave equation ids
     * @param rMasterEquationId The master equation ids
     */
    inline static void AssembleRelationMatrixT(
        MatrixType& rT,
        const Matrix& rTContribution,
        const std::vector<std::size_t>& rSlaveEquationId,
        const std::vector<std::size_t>& rMasterEquationId
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            const unsigned int system_size = rT.NumGlobalRows();

            // Count active indices
            int slave_active_indices = 0;
            for (unsigned int i = 0; i < rSlaveEquationId.size(); i++) {
                if (rSlaveEquationId[i] < system_size) {
                    ++slave_active_indices;
                }
            }
            int master_active_indices = 0;
            for (unsigned int i = 0; i < rMasterEquationId.size(); i++) {
                if (rMasterEquationId[i] < system_size) {
                    ++master_active_indices;
                }
            }

            if (slave_active_indices > 0 && master_active_indices > 0) {
                std::vector<int> indices(master_active_indices);
                std::vector<double> values(master_active_indices);

                // Fill Epetra vectors
                for (unsigned int i = 0; i < rSlaveEquationId.size(); i++) {
                    if (rSlaveEquationId[i] < system_size) {
                        const int current_global_row = rSlaveEquationId[i];

                        unsigned int loc_j = 0;
                        for (unsigned int j = 0; j < rMasterEquationId.size(); j++) {
                            if (rMasterEquationId[j] < system_size) {
                                indices[loc_j] = rMasterEquationId[j];
                                values[loc_j] = rTContribution(i, j);
                                ++loc_j;
                            }
                        }

                        const int ierr = rT.SumIntoGlobalValues(current_global_row, master_active_indices, values.data(), indices.data());
                        KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
                    }
                }
            }
        } else {
#ifdef HAVE_TPETRA
            using GO = typename MatrixType::global_ordinal_type;
            using ST = typename MatrixType::scalar_type;
            const std::size_t system_size = rT.getGlobalNumRows();

            for (std::size_t i = 0; i < rSlaveEquationId.size(); ++i) {
                if (rSlaveEquationId[i] < system_size) {
                    const GO global_id = static_cast<GO>(rSlaveEquationId[i]);
                    std::vector<GO> indices;
                    std::vector<ST> values;
                    for (std::size_t j = 0; j < rMasterEquationId.size(); ++j) {
                        if (rMasterEquationId[j] < system_size) {
                            indices.push_back(static_cast<GO>(rMasterEquationId[j]));
                            values.push_back(static_cast<ST>(rTContribution(i, j)));
                        }
                    }
                    if (indices.size() > 0) {
                        rT.sumIntoGlobalValues(global_id, indices.size(), values.data(), indices.data());
                    }
                }
            }
#endif
        }
    }

    /**
     * @brief Assembles the Constant vector of the system with MPC
     * @param rC The constant vector
     * @param rConstantContribution The RHS contribution
     * @param rEquationId The equation ids
     */
    inline static void AssembleConstantVector(
        VectorType& rC,
        const Vector& rConstantContribution,
        const std::vector<std::size_t>& rSlaveEquationId
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            const unsigned int system_size = rC.GlobalLength();

            // Count active indices
            unsigned int slave_active_indices = 0;
            for (unsigned int i = 0; i < rSlaveEquationId.size(); i++)
                if (rSlaveEquationId[i] < system_size)
                    ++slave_active_indices;

            if (slave_active_indices > 0) {
                // Size Epetra vectors
                Epetra_IntSerialDenseVector indices(slave_active_indices);
                Epetra_SerialDenseVector values(slave_active_indices);

                // Fill Epetra vectors
                unsigned int loc_i = 0;
                for (unsigned int i = 0; i < rSlaveEquationId.size(); i++) {
                    if (rSlaveEquationId[i] < system_size) {
                        indices[loc_i] = rSlaveEquationId[i];
                        values[loc_i] = rConstantContribution[i];
                        ++loc_i;
                    }
                }

                const int ierr = rC.SumIntoGlobalValues(indices, values);
                KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
            }
        } else {
#ifdef HAVE_TPETRA
            using GO = typename VectorType::global_ordinal_type;
            using ST = typename VectorType::scalar_type;
            const std::size_t system_size = rC.getGlobalLength();

            for (std::size_t i = 0; i < rSlaveEquationId.size(); ++i) {
                if (rSlaveEquationId[i] < system_size) {
                    const GO global_id = static_cast<GO>(rSlaveEquationId[i]);
                    const ST val = static_cast<ST>(rConstantContribution[i]);
                    rC.sumIntoGlobalValue(global_id, val);
                }
            }
#endif
        }
    }

    /**
     * @brief Sets a value in a vector
     * @param rX The vector considered
     * @param i The index of the value considered
     * @param Value The value considered
     */
    static inline void SetGlobalValue(
        VectorType& rX,
        IndexType i,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            Epetra_IntSerialDenseVector indices(1);
            Epetra_SerialDenseVector values(1);
            indices[0] = i;
            values[0] = Value;
            int ierr = rX.ReplaceGlobalValues(indices, values);
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

            ierr = rX.GlobalAssemble(Insert,true); //Epetra_CombineMode mode=Add);
            KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            rX.replaceGlobalValue(static_cast<typename VectorType::global_ordinal_type>(i), Value);
#endif
        }
    }

    /**
     * @brief Sets a value in a vector
     * @param rX The vector considered
     * @param i The index of the value considered
     * @param Value The value considered
     */
    static inline void SetGlobalValueWithoutGlobalAssembly(
        VectorType& rX,
        IndexType i,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            Epetra_IntSerialDenseVector indices(1);
            Epetra_SerialDenseVector values(1);
            indices[0] = i;
            values[0] = Value;
            const int ierr = rX.ReplaceGlobalValues(indices, values);
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            rX.replaceGlobalValue(static_cast<typename VectorType::global_ordinal_type>(i), Value);
#endif
        }
    }

    /**
     * @brief Sets a value in a vector (local)
     * @param rX The vector considered
     * @param i The index of the value considered
     * @param Value The value considered
     */
    static inline void SetLocalValue(
        VectorType& rX,
        IndexType i,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            int ierr = rX.ReplaceMyValue(static_cast<int>(i), 0, Value);
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
            ierr = rX.GlobalAssemble(Insert,true); //Epetra_CombineMode mode=Add);
            KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            rX.replaceLocalValue(static_cast<typename VectorType::local_ordinal_type>(i), Value);
#endif
        }
    }

    /**
     * @brief Sets a value in a vector (local without global assembly)
     * @param rX The vector considered
     * @param i The index of the value considered
     * @param Value The value considered
     */
    static inline void SetLocalValueWithoutGlobalAssembly(
        VectorType& rX,
        IndexType i,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            const int ierr = rX.ReplaceMyValue(static_cast<int>(i), 0, Value);
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            rX.replaceLocalValue(static_cast<typename VectorType::local_ordinal_type>(i), Value);
#endif
        }
    }

    /**
     * @brief Sets a value in a matrix
     * @param rA The matrix considered
     * @param i The first index of the value considered
     * @param j The second index of the value considered
     * @param Value The value considered
     */
    static inline void SetGlobalValue(
        MatrixType& rA,
        IndexType i,
        IndexType j,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            std::vector<double> values(1, Value);
            std::vector<int> indices(1, j);

            int ierr = rA.ReplaceGlobalValues(static_cast<int>(i), 1, values.data(), indices.data());
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

            ierr = rA.GlobalAssemble();
            KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            using GO = typename MatrixType::global_ordinal_type;
            using ST = typename MatrixType::scalar_type;
            const GO col = static_cast<GO>(j);
            const ST val = static_cast<ST>(Value);
            rA.replaceGlobalValues(static_cast<GO>(i), 1, &val, &col);
#endif
        }
    }

    /**
     * @brief Sets a value in a matrix
     * @param rA The matrix considered
     * @param i The first index of the value considered
     * @param j The second index of the value considered
     * @param Value The value considered
     */
    static inline void SetGlobalValueWithoutGlobalAssembly(
        MatrixType& rA,
        IndexType i,
        IndexType j,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            std::vector<double> values(1, Value);
            std::vector<int> indices(1, j);

            const int ierr = rA.ReplaceGlobalValues(static_cast<int>(i), 1, values.data(), indices.data());
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            using GO = typename MatrixType::global_ordinal_type;
            using ST = typename MatrixType::scalar_type;
            const GO col = static_cast<GO>(j);
            const ST val = static_cast<ST>(Value);
            rA.replaceGlobalValues(static_cast<GO>(i), 1, &val, &col);
#endif
        }
    }

    /**
     * @brief Sets a value in a matrix
     * @param rA The matrix considered
     * @param i The first index of the value considered
     * @param j The second index of the value considered
     * @param Value The value considered
     */
    static inline void SetLocalValue(
        MatrixType& rA,
        IndexType i,
        IndexType j,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            std::vector<double> values(1, Value);
            std::vector<int> indices(1, j);

            int ierr = rA.ReplaceMyValues(static_cast<int>(i), 1, values.data(), indices.data());
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

            ierr = rA.GlobalAssemble();
            KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            using LO = typename MatrixType::local_ordinal_type;
            using ST = typename MatrixType::scalar_type;
            const LO col = static_cast<LO>(j);
            const ST val = static_cast<ST>(Value);
            rA.replaceLocalValues(static_cast<LO>(i), 1, &val, &col);
#endif
        }
    }

    /**
     * @brief Sets a value in a matrix
     * @param rA The matrix considered
     * @param i The first index of the value considered
     * @param j The second index of the value considered
     * @param Value The value considered
     */
    static inline void SetLocalValueWithoutGlobalAssembly(
        MatrixType& rA,
        IndexType i,
        IndexType j,
        const double Value
        )
    {
        if constexpr (std::is_same_v<typename TSparseSpace::CommunicatorType, Epetra_MpiComm>) {
            std::vector<double> values(1, Value);
            std::vector<int> indices(1, j);

            const int ierr = rA.ReplaceMyValues(static_cast<int>(i), 1, values.data(), indices.data());
            KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
        } else {
#ifdef HAVE_TPETRA
            using LO = typename MatrixType::local_ordinal_type;
            using ST = typename MatrixType::scalar_type;
            const LO col = static_cast<LO>(j);
            const ST val = static_cast<ST>(Value);
            rA.replaceLocalValues(static_cast<LO>(i), 1, &val, &col);
#endif
        }
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
        return "TrilinosAssemblingUtilities";
    }

    /**
     * @brief Print information about this object.
     * @param rOStream The output stream to print on.
     */
    virtual void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << "TrilinosAssemblingUtilities";
    }

    /**
     * @brief Print object's data.
     * @param rOStream The output stream to print on.
     */
    virtual void PrintData(std::ostream& rOStream) const
    {
    }

    ///@}
private:
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
    TrilinosAssemblingUtilities & operator=(TrilinosAssemblingUtilities const& rOther);

    /// Copy constructor.
    TrilinosAssemblingUtilities(TrilinosAssemblingUtilities const& rOther);

    ///@}
}; // Class TrilinosAssemblingUtilities

///@}

} // namespace Kratos.
