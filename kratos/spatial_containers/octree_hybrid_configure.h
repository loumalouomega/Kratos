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

// Project includes
#include "includes/node.h"
#include "geometries/geometry.h"

namespace Kratos {

/**
 * @brief OctreeHybrid configuration for Kratos surface-mesh-driven refinement.
 *
 * pointer_type is Geometry<Node>* so that StlIO-produced triangles can be
 * inserted into the octree.  IsIntersected delegates the AABB test to
 * Geometry::HasIntersection, which expects *world-space* bounds.  The bounds
 * passed inside OctreeHybrid are in *normalised* space, so for workflows where
 * the octree's ScaleBackToOriginalCoordinate is called before IsIntersected the
 * test is correct.
 *
 * For OctreeHybridMeshUtility, which drives refinement manually (without using
 * Insert()), this configuration acts only as a type-level contract; its
 * IsIntersected is never invoked.
 */
struct OctreeHybridKratosConfiguration {

    /// Unused user-payload type.
    struct data_type {};

    /// Pointer type inserted into octree cells (triangles from StlIO).
    using pointer_type = Geometry<Node>*;

    static constexpr std::size_t MAX_DEPTH = 10; ///< Maximum allowed refinement depth.
    static constexpr std::size_t MIN_DEPTH = 1;  ///< Minimum leaf depth.
    static constexpr std::size_t DIMENSION  = 3;  ///< Must be 3 for an octree.

    static void DeleteData(data_type*) {}

    /**
     * @brief AABB-triangle intersection using Geometry::HasIntersection.
     *
     * @param p_geom     Triangle geometry pointer.
     * @param tolerance  Ignored (HasIntersection uses its own tolerance).
     * @param lo         Lower bound of the box (world-space coordinates).
     * @param hi         Upper bound of the box (world-space coordinates).
     */
    static bool IsIntersected(
        pointer_type p_geom,
        double /*tolerance*/,
        const double* lo,
        const double* hi)
    {
        Point lo_pt(lo[0], lo[1], lo[2]);
        Point hi_pt(hi[0], hi[1], hi[2]);
        return p_geom->HasIntersection(lo_pt, hi_pt);
    }
};

} // namespace Kratos
