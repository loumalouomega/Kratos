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

// System includes
#include <algorithm>

// External includes

// Project includes
#include "utilities/atomic_utilities.h"
#include "utilities/parallel_utilities.h"

// Application includes
#include "facet_linear_system.h"

namespace Kratos
{

SerialFacetLinearSystem::SerialFacetLinearSystem(LinearSolverType::Pointer pLinearSolver)
    : mpLinearSolver(pLinearSolver)
{
    KRATOS_ERROR_IF(mpLinearSolver == nullptr) << "A linear solver is required for the facet system." << std::endl;
}

/***********************************************************************************/
/***********************************************************************************/

void SerialFacetLinearSystem::Initialize(
    const IndexType NumberOfOwnedDofs,
    const IndexType FirstOwnedDof,
    const IndexType TotalDofs,
    const std::vector<std::vector<IndexType>>& rElementDofBlocks
    )
{
    KRATOS_TRY

    KRATOS_ERROR_IF(FirstOwnedDof != 0 || NumberOfOwnedDofs != TotalDofs)
        << "The serial facet system cannot be used with a distributed facet numbering." << std::endl;

    // Sparsity pattern: the unknowns of an element are all coupled
    std::vector<std::vector<IndexType>> rows(TotalDofs);
    for (const auto& r_block : rElementDofBlocks) {
        for (const IndexType row : r_block) {
            rows[row].insert(rows[row].end(), r_block.begin(), r_block.end());
        }
    }
    IndexPartition<IndexType>(TotalDofs).for_each([&rows](const IndexType iRow) {
        auto& r_row = rows[iRow];
        std::sort(r_row.begin(), r_row.end());
        r_row.erase(std::unique(r_row.begin(), r_row.end()), r_row.end());
    });

    IndexType nnz = 0;
    for (const auto& r_row : rows) {
        nnz += r_row.size();
    }
    mMatrix = SparseMatrixType(TotalDofs, TotalDofs, nnz);
    auto* p_values = mMatrix.value_data().begin();
    auto* p_row_indices = mMatrix.index1_data().begin();
    auto* p_col_indices = mMatrix.index2_data().begin();
    p_row_indices[0] = 0;
    for (IndexType i_row = 0; i_row < TotalDofs; ++i_row) {
        p_row_indices[i_row + 1] = p_row_indices[i_row] + rows[i_row].size();
    }
    IndexPartition<IndexType>(TotalDofs).for_each([&](const IndexType iRow) {
        IndexType k = p_row_indices[iRow];
        for (const IndexType col : rows[iRow]) {
            p_col_indices[k] = col;
            p_values[k] = 0.0;
            ++k;
        }
    });
    mMatrix.set_filled(TotalDofs + 1, nnz);

    mRHS = ZeroVector(TotalDofs);
    mSolution = ZeroVector(TotalDofs);
    mIsInitialized = true;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

void SerialFacetLinearSystem::BeginAssemble()
{
    auto& r_values = mMatrix.value_data();
    IndexPartition<IndexType>(r_values.size()).for_each([&r_values](const IndexType i) {r_values[i] = 0.0;});
    IndexPartition<IndexType>(mRHS.size()).for_each([this](const IndexType i) {mRHS[i] = 0.0;});
}

/***********************************************************************************/
/***********************************************************************************/

void SerialFacetLinearSystem::Assemble(
    const IndexType Size,
    const IndexType* pDofs,
    const double* pLHS,
    const IndexType LHSRowStride,
    const double* pRHS
    )
{
    auto& r_values = mMatrix.value_data();
    const auto& r_row_indices = mMatrix.index1_data();
    const auto& r_col_indices = mMatrix.index2_data();
    for (IndexType i = 0; i < Size; ++i) {
        const IndexType row = pDofs[i];
        AtomicAdd(mRHS[row], pRHS[i]);
        const auto it_row_begin = r_col_indices.begin() + r_row_indices[row];
        const auto it_row_end = r_col_indices.begin() + r_row_indices[row + 1];
        for (IndexType j = 0; j < Size; ++j) {
            const auto it_col = std::lower_bound(it_row_begin, it_row_end, pDofs[j]);
            AtomicAdd(r_values[std::distance(r_col_indices.begin(), it_col)], pLHS[i * LHSRowStride + j]);
        }
    }
}

/***********************************************************************************/
/***********************************************************************************/

double SerialFacetLinearSystem::Solve()
{
    KRATOS_TRY

    if (mRHS.size() == 0) {
        return 0.0;
    }

    mpLinearSolver->Solve(mMatrix, mSolution, mRHS);

    // Relative residual (the local conservation holds for any facet values, but the optimality does not)
    SparseVectorType residual(mRHS.size());
    SparseSpaceType::Mult(mMatrix, mSolution, residual);
    SparseSpaceType::ScaleAndAdd(1.0, mRHS, -1.0, residual);
    const double rhs_norm = SparseSpaceType::TwoNorm(mRHS);
    return rhs_norm > 0.0 ? SparseSpaceType::TwoNorm(residual) / rhs_norm : 0.0;

    KRATOS_CATCH("")
}

} // namespace Kratos
