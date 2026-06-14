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
 * @class OctreeHybridColorCellsInTouch
 * @ingroup KratosCore
 * @brief Colouring stage that labels every extracted hex cell whose axis-aligned
 *        bounding box (AABB) intersects any geometry in the input ModelPart.
 * @details For each geometry in the requested ModelPart (selected by `input_entities`
 *          as `"elements"`, `"conditions"`, or `"geometries"`; when empty — the
 *          default — the first non-empty of `Conditions()`, `Elements()`,
 *          `Geometries()`, in that order, is used — see
 *          @ref OctreeHybridMeshUtility::ResolveInputEntityGeometries), the operation
 *          computes the geometry's AABB, quick-rejects cells whose own AABB does not overlap,
 *          then calls `Geometry::HasIntersection(min_pt, max_pt)` for the remaining
 *          candidates.  Cells that pass are assigned the requested `color`.
 *
 *          The per-cell AABB is the axis-aligned bounding box of the cell's 8 node
 *          coordinates from `OctreeHybridMesherData::mNodes`.
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised yet (or
 *          its size differs from `mCells.size()`), it is resized and filled with 0
 *          before any colour is written.  Already-coloured cells are updated when they
 *          touch a geometry.
 *
 * ### Colour convention
 * | Value | Typical meaning |
 * |-------|-----------------|
 * |  `0`  | Uncoloured / outside (default) |
 * |  `1`  | Inside the main surface |
 * | other | User-defined sub-region |
 *
 * ### Typical pipeline position
 * Can be used as the sole coloring stage for selecting interface cells,
 * or stacked after `OctreeHybridClassifyCellsInsideOutside` to further refine labels.
 *
 * ### Usage
 * Add the following block to the `"coloring_settings_list"` in the modeler JSON:
 * @code{.json}
 * {
 *     "type"              : "OctreeHybridColorCellsInTouch",
 *     "model_part_name"   : "MySurface",
 *     "color"             : 2,
 *     "input_entities"    : ""
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridClassifyCellsInsideOutside
 * @see OctreeHybridColorConnectedCellsInTouch
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellsInTouch : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellsInTouch() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellsInTouch(OctreeHybridColorCellsInTouch const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours every hex cell whose AABB intersects a geometry in the input ModelPart.
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
     *     "type"           : "OctreeHybridColorCellsInTouch",
     *     "model_part_name": "",
     *     "color"          : 1,
     *     "input_entities" : ""
     * }
     * @endcode
     * `"input_entities" == ""` (the default) auto-detects the first non-empty of
     * `Conditions()`, `Elements()`, `Geometries()`; `"geometries"`, `"elements"`,
     * or `"conditions"` select that container explicitly. Any other value throws.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsInTouch.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellsInTouch)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellsInTouch.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellsInTouch)

    ///@}
};

///@}

} // namespace Kratos
