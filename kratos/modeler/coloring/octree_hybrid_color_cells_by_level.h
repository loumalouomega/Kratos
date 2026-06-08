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
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridColorCellsByLevel
 * @ingroup KratosCore
 * @brief Colouring stage that labels hex cells by their octree refinement level.
 * @details Iterates `OctreeHybridMesherData::mCellLevel` and sets `mCellColor[i] = color`
 *          for every cell whose level satisfies `min_level <= mCellLevel[i] <= max_level`.
 *
 *          This operation has no direct voxel-mesher counterpart; it exploits the
 *          multi-resolution structure of the octree to allow level-selective entity
 *          generation (e.g. emit only the finest-level cells, or exclude the coarsest
 *          interior blocks).
 *
 *          ### Level convention
 *          | Value | Meaning |
 *          |-------|---------|
 *          | 1 … N | Octree leaf level (1 = coarsest, N = finest) |
 *          | -1    | Transition-template hex (dual mesh only) |
 *          | -2    | Buffer-layer hex added during surface projection |
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised (or its
 *          size differs from `mCells.size()`), it is resized and filled with 0 before
 *          any colour is written.
 *
 * ### Usage
 * @code{.json}
 * {
 *     "type"      : "OctreeHybridColorCellsByLevel",
 *     "color"     : 2,
 *     "min_level" : 4,
 *     "max_level" : 4
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridClassifyCellsInsideOutside
 * @see OctreeHybridMesherData::mCellLevel
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellsByLevel : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellsByLevel() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellsByLevel(OctreeHybridColorCellsByLevel const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours cells whose refinement level is within `[min_level, max_level]`.
     * @param rModeler            The owning modeler; `mCellColor` in its shared data is
     *                            updated in-place.
     * @param ColoringParameters  Validated JSON parameters for this step (see
     *                            @ref GetDefaultParameters for the schema).
     */
    void Apply(OctreeHybridMesherModeler& rModeler, Parameters ColoringParameters) const override;

    /**
     * @brief Returns the default parameter schema.
     * @return Parameters object:
     * @code{.json}
     * {
     *     "type"      : "OctreeHybridColorCellsByLevel",
     *     "color"     : 1,
     *     "min_level" : 1,
     *     "max_level" : 100
     * }
     * @endcode
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsByLevel.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellsByLevel)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellsByLevel.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellsByLevel)

    ///@}
};

///@}

} // namespace Kratos
