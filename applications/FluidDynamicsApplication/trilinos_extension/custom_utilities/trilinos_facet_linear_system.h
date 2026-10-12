//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#pragma once

// System includes
#include <algorithm>
#include <mutex>
#include <vector>

// External includes
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "Epetra_Import.h"
#include "Epetra_Map.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Vector.h"

// Project includes
#include "includes/define.h"
#include "linear_solvers/linear_solver.h"
#include "spaces/default_spaces.h"

// TrilinosApplication includes
#include "trilinos_space.h"

// Application includes
#include "custom_utilities/particle_mesh/facet_linear_system.h"

namespace Kratos
{
///@addtogroup FluidDynamicsApplication
///@{

///@name Kratos Classes
///@{

/**
 * @class TrilinosFacetLinearSystem
 * @ingroup FluidDynamicsApplication
 * @brief Distributed (MPI) facet system of the PDE-constrained particle-mesh projection stored in Epetra objects
 * @details Each rank owns a contiguous range of facet unknowns (rows). The contributions to the rows of other ranks
 * (partition facets) are communicated by the Epetra global assembly. After the solution, the values of all the unknowns
 * referenced by the local elements are imported.
 * @author Vicente Mataix Ferrandiz
 */
class TrilinosFacetLinearSystem : public FacetLinearSystem
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of TrilinosFacetLinearSystem
    KRATOS_CLASS_POINTER_DEFINITION(TrilinosFacetLinearSystem);

    /// Linear algebra
    using SparseSpaceType = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;
    using LocalSpaceType = TDefaultDenseSpace<double>;
    using LinearSolverType = LinearSolver<SparseSpaceType, LocalSpaceType>;

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Construct a new distributed facet system
     * @param rComm The Epetra communicator
     * @param pLinearSolver The (Trilinos) linear solver
     */
    TrilinosFacetLinearSystem(
        const Epetra_MpiComm& rComm,
        LinearSolverType::Pointer pLinearSolver
        ) : mComm(rComm),
            mpLinearSolver(pLinearSolver)
    {
        KRATOS_ERROR_IF(mpLinearSolver == nullptr) << "A linear solver is required for the facet system." << std::endl;
    }

    ///@}
    ///@name Operations
    ///@{

    void Initialize(
        const IndexType NumberOfOwnedDofs,
        const IndexType FirstOwnedDof,
        const IndexType TotalDofs,
        const std::vector<std::vector<IndexType>>& rElementDofBlocks
        ) override
    {
        KRATOS_TRY

        // Map of the owned (contiguous) unknowns
        std::vector<int> owned_ids(NumberOfOwnedDofs);
        for (IndexType i = 0; i < NumberOfOwnedDofs; ++i) {
            owned_ids[i] = static_cast<int>(FirstOwnedDof + i);
        }
        mpMap = Kratos::make_shared<Epetra_Map>(-1, static_cast<int>(NumberOfOwnedDofs), owned_ids.data(), 0, mComm);

        // Graph from the element blocks (the unknowns of an element are all coupled)
        std::vector<std::vector<int>> blocks(rElementDofBlocks.size());
        std::vector<int> needed_dofs;
        for (IndexType i_block = 0; i_block < rElementDofBlocks.size(); ++i_block) {
            const auto& r_block = rElementDofBlocks[i_block];
            blocks[i_block].assign(r_block.begin(), r_block.end());
            needed_dofs.insert(needed_dofs.end(), r_block.begin(), r_block.end());
        }
        typename SparseSpaceType::VectorPointerType p_reactions;
        constexpr int guess_row_size = 40;
        SparseSpaceType::BuildSystemStructure(mComm, NumberOfOwnedDofs, static_cast<int>(FirstOwnedDof), guess_row_size,
            blocks, blocks, mpA, mpRHS, mpSolution, p_reactions, TotalDofs, mpMap);

        // Unknowns whose values are needed by the local elements
        std::sort(needed_dofs.begin(), needed_dofs.end());
        needed_dofs.erase(std::unique(needed_dofs.begin(), needed_dofs.end()), needed_dofs.end());
        mNeededDofs = needed_dofs;
        mNeededValues.assign(mNeededDofs.size(), 0.0);
        mIsInitialized = true;

        KRATOS_CATCH("")
    }

    bool IsInitialized() const override
    {
        return mIsInitialized;
    }

    void BeginAssemble() override
    {
        mpA->PutScalar(0.0);
        mpRHS->PutScalar(0.0);
    }

    void Assemble(
        const IndexType Size,
        const IndexType* pDofs,
        const double* pLHS,
        const IndexType LHSRowStride,
        const double* pRHS
        ) override
    {
        Matrix lhs(Size, Size);
        Vector rhs(Size);
        std::vector<std::size_t> dofs(pDofs, pDofs + Size);
        for (IndexType i = 0; i < Size; ++i) {
            rhs[i] = pRHS[i];
            for (IndexType j = 0; j < Size; ++j) {
                lhs(i, j) = pLHS[i * LHSRowStride + j];
            }
        }

        // The Epetra assembly is not thread-safe
        std::lock_guard<std::mutex> lock(mAssemblyMutex);
        SparseSpaceType::AssembleLHS(*mpA, lhs, dofs);
        SparseSpaceType::AssembleRHS(*mpRHS, rhs, dofs);
    }

    double Solve() override
    {
        KRATOS_TRY

        SparseSpaceType::GlobalAssemble(*mpA);
        SparseSpaceType::GlobalAssemble(*mpRHS);

        mpLinearSolver->Solve(*mpA, *mpSolution, *mpRHS);

        // Relative residual
        Epetra_Vector residual(*mpMap);
        mpA->Multiply(false, *mpSolution, residual);
        residual.Update(1.0, *mpRHS, -1.0);
        double residual_norm = 0.0;
        double rhs_norm = 0.0;
        residual.Norm2(&residual_norm);
        mpRHS->Norm2(&rhs_norm);

        // Import the values needed by the local elements (collective, also for ranks that need no values)
        const Epetra_Map needed_map(-1, static_cast<int>(mNeededDofs.size()), mNeededDofs.empty() ? nullptr : mNeededDofs.data(), 0, mComm);
        const Epetra_Import importer(needed_map, mpSolution->Map());
        Epetra_Vector needed_values(needed_map);
        const int error_code = needed_values.Import(*mpSolution, importer, Insert);
        KRATOS_ERROR_IF(error_code != 0) << "Epetra failure importing the facet values (error code " << error_code << ")." << std::endl;
        for (IndexType i = 0; i < mNeededDofs.size(); ++i) {
            mNeededValues[i] = needed_values[i];
        }

        return rhs_norm > 0.0 ? residual_norm / rhs_norm : 0.0;

        KRATOS_CATCH("")
    }

    double GetValue(const IndexType GlobalDof) const override
    {
        const auto it = std::lower_bound(mNeededDofs.begin(), mNeededDofs.end(), static_cast<int>(GlobalDof));
        KRATOS_DEBUG_ERROR_IF(it == mNeededDofs.end() || *it != static_cast<int>(GlobalDof)) << "Facet unknown " << GlobalDof << " is not available in this rank." << std::endl;
        return mNeededValues[std::distance(mNeededDofs.begin(), it)];
    }

    ///@}
private:
    ///@name Member Variables
    ///@{

    Epetra_MpiComm mComm; // Copy (reference counted), so the system does not depend on the lifetime of the given communicator

    LinearSolverType::Pointer mpLinearSolver;

    bool mIsInitialized = false;

    typename SparseSpaceType::MapPointerType mpMap;

    typename SparseSpaceType::MatrixPointerType mpA;

    typename SparseSpaceType::VectorPointerType mpRHS;

    typename SparseSpaceType::VectorPointerType mpSolution;

    std::vector<int> mNeededDofs;

    std::vector<double> mNeededValues;

    std::mutex mAssemblyMutex;

    ///@}
}; // Class TrilinosFacetLinearSystem

///@}
///@} addtogroup block

} // namespace Kratos
