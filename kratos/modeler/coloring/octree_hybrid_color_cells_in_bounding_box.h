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
 * @class OctreeHybridColorCellsInBoundingBox
 * @ingroup KratosCore
 * @brief Colouring stage that labels hex cells whose centre lies inside (or outside)
 *        an axis-aligned bounding box.
 * @details For every cell, the centroid of its 8 corner nodes
 *          (`OctreeHybridMesherData::mNodes`, via `OctreeHybridMesherData::mCells`) is
 *          tested against an axis-aligned bounding box. The box is either given
 *          explicitly via `min_point`/`max_point`, or — when either array does not
 *          have exactly 3 components — computed as the AABB of the nodes of the
 *          ModelPart `model_part_name`.
 *
 *          A cell centroid is "inside" the box when, for every axis `d`,
 *          `min_point[d] <= centroid[d] <= max_point[d]`. When `inside_bounding_box`
 *          is `true`, cells whose centroid is inside the box receive `color`;
 *          when `false`, cells whose centroid is **outside** the box receive `color`.
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised yet (or its
 *          size differs from `mCells.size()`), it is resized and filled with 0 before
 *          any colour is written.
 *
 * ### Usage
 * Add the following block to the `"coloring_settings_list"` in the modeler JSON:
 * @code{.json}
 * {
 *     "type"                : "OctreeHybridColorCellsInBoundingBox",
 *     "color"               : 2,
 *     "min_point"           : [0.3, 0.3, 0.3],
 *     "max_point"           : [0.7, 0.7, 0.7],
 *     "inside_bounding_box" : true
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridColorCellsFacesInBoundingBox
 * @see OctreeHybridColorCellsWithInsideCenter
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellsInBoundingBox : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellsInBoundingBox() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellsInBoundingBox(OctreeHybridColorCellsInBoundingBox const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours every hex cell whose centroid is inside (or outside) the
     *        requested bounding box.
     * @param rModeler            The owning modeler; `mCellColor` in its shared data is
     *                            updated in-place.
     * @param ColoringParameters  Validated JSON parameters for this step (see
     *                            @ref GetDefaultParameters for the schema).
     */
    void Apply(OctreeHybridMeshGeneratorModeler& rModeler, Parameters ColoringParameters) const override;

    /**
     * @brief Returns the default parameter schema.
     * @return Parameters object:
     * @code{.json}
     * {
     *     "type"                : "OctreeHybridColorCellsInBoundingBox",
     *     "color"               : -1,
     *     "model_part_name"     : "Undefined",
     *     "min_point"           : [],
     *     "max_point"           : [],
     *     "inside_bounding_box" : false
     * }
     * @endcode
     * `"min_point"`/`"max_point"` take precedence when both have exactly 3
     * components; otherwise the bounding box is computed from the nodes of
     * `"model_part_name"`.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsInBoundingBox.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellsInBoundingBox)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellsInBoundingBox.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellsInBoundingBox)

    ///@}
};

///@}

} // namespace Kratos
