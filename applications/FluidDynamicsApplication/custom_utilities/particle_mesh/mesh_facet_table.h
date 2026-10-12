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
#include <array>
#include <unordered_map>
#include <vector>

// External includes

// Project includes
#include "includes/data_communicator.h"
#include "includes/define.h"
#include "includes/model_part.h"

namespace Kratos
{
///@addtogroup FluidDynamicsApplication
///@{

///@name Kratos Classes
///@{

/**
 * @class MeshFacetTable
 * @ingroup FluidDynamicsApplication
 * @brief Cached simplex mesh topology for particle-mesh projections
 * @details This class stores, for each simplex element of a model part, the affine shape function data,
 * the outward area-weighted face normals, the face neighbours and a global numbering of the facets shared by two elements.
 * Face i of an element is the face opposite to its local node i. Each shared facet carries TDim degrees of
 * freedom (a discontinuous P1 facet field, one value per facet vertex). The facet DOF of a vertex is
 * global_facet_index * TDim + rank of the vertex Id among the sorted Ids of the facet nodes, so both elements sharing
 * a facet agree on it. The area normal of a shared facet is computed once from its owner and negated for the other side,
 * so the facet fluxes of both sides cancel exactly.
 * In MPI runs (elements are not duplicated, only the interface nodes) the facets shared by elements of two ranks
 * (partition facets) are found by exchanging the keys of the locally unmatched faces with the neighbour ranks. The owner
 * of a partition facet is the lowest rank. Each rank numbers its owned facets contiguously, so the facet DOFs owned by a rank
 * form the range [FirstOwnedFacetDof, FirstOwnedFacetDof + NumberOfOwnedFacetDofs).
 * @tparam TDim Working space dimension (2 for triangles, 3 for tetrahedra)
 * @author Vicente Mataix Ferrandiz
 */
template<std::size_t TDim>
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) MeshFacetTable
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of MeshFacetTable
    KRATOS_CLASS_POINTER_DEFINITION(MeshFacetTable);

    /// Number of nodes of the simplex
    static constexpr std::size_t NumNodes = TDim + 1;

    /// Number of faces of the simplex
    static constexpr std::size_t NumFaces = TDim + 1;

    /// Number of nodes of each face
    static constexpr std::size_t NumFaceNodes = TDim;

    /// Value used to flag a missing neighbour or a boundary face
    static constexpr int Invalid = -1;

    /// Shape functions values container
    using ShapeFunctionsType = array_1d<double, NumNodes>;

    /// Cached data of each element
    struct ElementData
    {
        std::array<std::size_t, NumNodes> NodeIndices;                  /// Positions of the element nodes in the model part nodes container
        array_1d<double, 3> Centroid = ZeroVector(3);                   /// Element centroid
        BoundedMatrix<double, NumNodes, TDim> DN_DX = ZeroMatrix(NumNodes, TDim); /// Constant shape functions gradients
        double Volume = 0.0;                                            /// Element domain size
        double Size = 0.0;                                              /// Element characteristic size (minimum height)
        std::array<array_1d<double, 3>, NumFaces> FaceAreaNormals;      /// Outward normals scaled by the face measure (face i is opposite to node i)
        std::array<int, NumFaces> Neighbours;                           /// Local neighbour element index across each face (Invalid at the boundary and at partition facets)
        std::array<int, NumFaces> Facets;                               /// Local facet index of each shared face, interior or partition facet (Invalid at the boundary)
        std::array<int, NumFaces> PartitionRanks;                       /// Rank of the element across each partition facet (Invalid otherwise)
    };

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Construct the facet table from the elements of a model part
     * @param rModelPart Model part made of linear simplices (triangles in 2D, tetrahedra in 3D)
     */
    explicit MeshFacetTable(const ModelPart& rModelPart);

    /// Destructor.
    ~MeshFacetTable() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Evaluates the (affine) shape functions of an element at a given point
     * @details The point does not need to be inside the element (negative values mean it is outside)
     * @param ElementIndex Element position in the model part elements container
     * @param rCoordinates Point coordinates
     * @param rN Shape functions values
     */
    void ShapeFunctions(
        const std::size_t ElementIndex,
        const array_1d<double, 3>& rCoordinates,
        ShapeFunctionsType& rN
        ) const;

    /**
     * @brief Returns the global facet DOF associated to a vertex of an interior face
     * @param ElementIndex Element position in the model part elements container
     * @param Face Local face index (opposite node)
     * @param LocalNode Local node index of the vertex (must be different from Face)
     * @return std::size_t The global facet DOF
     */
    std::size_t FacetDof(
        const std::size_t ElementIndex,
        const std::size_t Face,
        const std::size_t LocalNode
        ) const
    {
        return mFacetDofs[ElementIndex][Face][LocalNode];
    }

    /**
     * @brief Computes the global facet DOFs of each element (the DOFs of an element are all coupled in the facet system)
     * @param rElementDofs Global facet DOFs of each element
     */
    void ComputeElementFacetDofs(std::vector<std::vector<std::size_t>>& rElementDofs) const;

    /**
     * @brief Finds the local element and face of a partition facet
     * @param GlobalFacet Global facet index
     * @param rElementIndex Local element index
     * @param rFace Local face index
     * @return true if the partition facet is local
     */
    bool FindPartitionFacet(
        const std::size_t GlobalFacet,
        std::size_t& rElementIndex,
        std::size_t& rFace
        ) const;

    /**
     * @brief Computes the facet mass matrix coefficient of a face
     * @details For P1 functions on a face of measure |F| and dimension m = TDim - 1 the face mass matrix is |F| (1 + delta_ab) / ((m + 1) (m + 2))
     * @param FaceMeasure The face measure
     * @param SameNode True for the diagonal entries
     * @return double The face mass matrix entry
     */
    static double FaceMassCoefficient(
        const double FaceMeasure,
        const bool SameNode
        )
    {
        constexpr double denominator = static_cast<double>(TDim * (TDim + 1));
        return SameNode ? 2.0 * FaceMeasure / denominator : FaceMeasure / denominator;
    }

    ///@}
    ///@name Access
    ///@{

    /// Returns the number of elements
    std::size_t NumberOfElements() const
    {
        return mElementData.size();
    }

    /// Returns the number of nodes
    std::size_t NumberOfNodes() const
    {
        return mNumberOfNodes;
    }

    /// Returns the number of local shared facets (interior and partition facets)
    std::size_t NumberOfFacets() const
    {
        return mFacetGlobalIds.size();
    }

    /// Returns the global number of facet DOFs (TDim per shared facet)
    std::size_t NumberOfFacetDofs() const
    {
        return mTotalNumberOfFacets * NumFaceNodes;
    }

    /// Returns the number of facet DOFs owned by this rank
    std::size_t NumberOfOwnedFacetDofs() const
    {
        return mNumberOfOwnedFacets * NumFaceNodes;
    }

    /// Returns the first facet DOF owned by this rank
    std::size_t FirstOwnedFacetDof() const
    {
        return mFirstOwnedFacet * NumFaceNodes;
    }

    /// Returns the global index of a local facet
    std::size_t GetFacetGlobalId(const std::size_t LocalFacet) const
    {
        return mFacetGlobalIds[LocalFacet];
    }

    /// Returns the Id of the element at a given position
    IndexType GetElementId(const std::size_t ElementIndex) const
    {
        return mElementIds[ElementIndex];
    }

    /// Returns the data communicator of the model part
    const DataCommunicator& GetDataCommunicator() const
    {
        return *mpDataCommunicator;
    }

    /// Returns true if the facets are distributed among several ranks
    bool IsDistributed() const
    {
        return mIsDistributed;
    }

    /// Returns the cached data of an element
    const ElementData& GetElementData(const std::size_t ElementIndex) const
    {
        return mElementData[ElementIndex];
    }

    /// Returns the position in the elements container of the element with the given Id
    std::size_t GetElementIndex(const IndexType ElementId) const;

    /// Returns the minimum element size of the (global) mesh
    double GetMinimumElementSize() const
    {
        return mMinimumElementSize;
    }

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const
    {
        return "MeshFacetTable" + std::to_string(TDim) + "D";
    }

    ///@}
private:
    ///@name Member Variables
    ///@{

    std::size_t mNumberOfNodes = 0;

    std::size_t mNumberOfOwnedFacets = 0;

    std::size_t mFirstOwnedFacet = 0;

    std::size_t mTotalNumberOfFacets = 0;

    bool mIsDistributed = false;

    const DataCommunicator* mpDataCommunicator = nullptr;

    std::vector<std::size_t> mFacetGlobalIds;

    std::vector<IndexType> mElementIds;

    std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>> mPartitionFacets;

    double mMinimumElementSize = 0.0;

    std::vector<ElementData> mElementData;

    std::vector<std::array<std::array<std::size_t, NumNodes>, NumFaces>> mFacetDofs;

    std::unordered_map<IndexType, std::size_t> mElementIdToIndex;

    ///@}
}; // Class MeshFacetTable

///@}
///@} addtogroup block

} // namespace Kratos
