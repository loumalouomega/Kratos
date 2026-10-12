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
#include <vector>

// External includes

// Project includes
#include "includes/define.h"
#include "includes/data_communicator.h"
#include "linear_solvers/linear_solver.h"
#include "spaces/default_spaces.h"

namespace Kratos
{
///@addtogroup FluidDynamicsApplication
///@{

///@name Kratos Classes
///@{

/**
 * @class FacetLinearSystem
 * @ingroup FluidDynamicsApplication
 * @brief Global linear system of the facet unknowns of the PDE-constrained particle-mesh projection
 * @details The facet unknowns are numbered globally. Each rank owns a contiguous range of them, and the elements of a rank
 * may also reference facet unknowns owned by other ranks (partition facets). Derived classes provide the storage, the
 * assembly and the solution for a given linear algebra (serial ublas or distributed Trilinos).
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) FacetLinearSystem
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of FacetLinearSystem
    KRATOS_CLASS_POINTER_DEFINITION(FacetLinearSystem);

    /// Index type
    using IndexType = std::size_t;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Destructor.
    virtual ~FacetLinearSystem() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Builds the structure of the system
     * @param NumberOfOwnedDofs Number of unknowns owned by this rank
     * @param FirstOwnedDof First global unknown owned by this rank (the owned unknowns are contiguous)
     * @param TotalDofs Global number of unknowns
     * @param rElementDofBlocks Global unknowns of each local element (all of them are coupled)
     */
    virtual void Initialize(
        const IndexType NumberOfOwnedDofs,
        const IndexType FirstOwnedDof,
        const IndexType TotalDofs,
        const std::vector<std::vector<IndexType>>& rElementDofBlocks
        ) = 0;

    /// Returns true if the structure has been built
    virtual bool IsInitialized() const = 0;

    /// Sets the matrix and the right hand side to zero
    virtual void BeginAssemble() = 0;

    /**
     * @brief Adds a local contribution (thread-safe)
     * @param Size Number of local unknowns
     * @param pDofs Global unknowns
     * @param pLHS Local matrix (row-major)
     * @param LHSRowStride Distance between consecutive rows of the local matrix
     * @param pRHS Local right hand side
     */
    virtual void Assemble(
        const IndexType Size,
        const IndexType* pDofs,
        const double* pLHS,
        const IndexType LHSRowStride,
        const double* pRHS
        ) = 0;

    /**
     * @brief Solves the system (the previous solution is used as initial guess)
     * @return double Relative residual of the solution
     */
    virtual double Solve() = 0;

    /// Returns the solution value of a global unknown referenced by a local element (valid after Solve)
    virtual double GetValue(const IndexType GlobalDof) const = 0;

    ///@}
}; // Class FacetLinearSystem

/**
 * @class SerialFacetLinearSystem
 * @ingroup FluidDynamicsApplication
 * @brief Serial (shared memory) facet system stored in a ublas CSR matrix
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) SerialFacetLinearSystem : public FacetLinearSystem
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of SerialFacetLinearSystem
    KRATOS_CLASS_POINTER_DEFINITION(SerialFacetLinearSystem);

    /// Linear algebra
    using SparseSpaceType = DefaultSparseSpaceType;
    using LocalSpaceType = DefaultLocalSpaceType;
    using LinearSolverType = LinearSolver<SparseSpaceType, LocalSpaceType>;
    using SparseMatrixType = typename SparseSpaceType::MatrixType;
    using SparseVectorType = typename SparseSpaceType::VectorType;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Constructor with the linear solver
    explicit SerialFacetLinearSystem(LinearSolverType::Pointer pLinearSolver);

    ///@}
    ///@name Operations
    ///@{

    void Initialize(
        const IndexType NumberOfOwnedDofs,
        const IndexType FirstOwnedDof,
        const IndexType TotalDofs,
        const std::vector<std::vector<IndexType>>& rElementDofBlocks
        ) override;

    bool IsInitialized() const override
    {
        return mIsInitialized;
    }

    void BeginAssemble() override;

    void Assemble(
        const IndexType Size,
        const IndexType* pDofs,
        const double* pLHS,
        const IndexType LHSRowStride,
        const double* pRHS
        ) override;

    double Solve() override;

    double GetValue(const IndexType GlobalDof) const override
    {
        return mSolution[GlobalDof];
    }

    ///@}
private:
    ///@name Member Variables
    ///@{

    LinearSolverType::Pointer mpLinearSolver;

    bool mIsInitialized = false;

    SparseMatrixType mMatrix;

    SparseVectorType mRHS;

    SparseVectorType mSolution;

    ///@}
}; // Class SerialFacetLinearSystem

///@}
///@} addtogroup block

} // namespace Kratos
