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

// External includes

// Project includes
#include "includes/node.h"
#include "utilities/octree_hybrid_mesh_utility.h"

namespace Kratos::Internals 
{

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridMesherData
 * @ingroup KratosCore
 * @brief Shared state of the @ref OctreeHybridMesherModeler, accessed by its components.
 * @details Holds the built octree, the surface triangle soup, and the extracted
 * in-memory dual/primal hex mesh (node coordinates, hex connectivity, per-cell
 * refinement level and inside/outside colour), plus the de-duplication map from a
 * mesh-node index to the created ModelPart node.  This is the octree analog of the
 * voxel mesher's `CartesianMeshColors` + `CartesianData`.
 * @author Vicente Mataix Ferrandiz
 */
class OctreeHybridMesherData
{
public:
    ///@name Type Definitions
    ///@{

    using OctreeType        = OctreeHybridMeshUtility::OctreeType;
    using TriangleSoup      = OctreeHybridMeshUtility::TriangleSoup;
    using HangingConstraint = OctreeHybridMeshUtility::HangingConstraint;

    ///@}
    ///@name Member Variables
    ///@{

    /// The built (and 2:1-balanced) octree.
    std::unique_ptr<OctreeType> mpOctree;

    /// World-space surface triangles the mesh is carved/projected against.
    TriangleSoup mTriangles;

    /// Dual/primal hex node coordinates (world space).
    std::vector<std::array<double, 3>> mNodes;

    /// Hexahedra connectivity (indices into @ref mNodes).
    std::vector<std::array<int, 8>> mCells;

    /// Per-cell refinement level (-1 for transition-template hexes).
    std::vector<int> mCellLevel;

    /// Per-cell inside(1)/outside(0) classification (empty until coloured).
    std::vector<int> mCellColor;

    /// Hanging-node constraints at 2:1 transitions (primal mesh only).
    std::vector<HangingConstraint> mHanging;

    /// Mesh-node index -> created ModelPart node (lazy de-duplication map).
    std::vector<Node::Pointer> mNodePtrs;

    /// Whether the surface projection has already been applied to @ref mNodes.
    bool mProjected = false;

    ///@}
    ///@name Inquiry
    ///@{

    /// Returns true once the hex mesh has been extracted.
    bool IsExtracted() const { return !mCells.empty(); }

    ///@}
};

///@}

} // namespace Kratos::Internals
