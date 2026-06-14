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
 * @class OctreeHybridColorCellsWithInsideCenter
 * @ingroup KratosCore
 * @brief Colouring stage that labels every hex cell whose centre lies inside the
 *        surface defined by another ModelPart.
 * @details For each cell, the centroid of its 8 corner nodes (`OctreeHybridMesherData::mNodes`,
 *          via `OctreeHybridMesherData::mCells`) is tested against the triangle soup
 *          extracted from the ModelPart `model_part_name` (selected by `input_entities`
 *          as `"geometries"`, `"elements"`, or `"conditions"`; when empty — the
 *          default — the first non-empty of `Conditions()`, `Elements()`,
 *          `Geometries()`, in that order, is used — see
 *          @ref OctreeHybridMeshUtility::ResolveInputEntityGeometries).  The test combines a
 *          ray-cast crossing-parity sign with the closest-triangle distance
 *          (`OctreeHybridMeshUtility::ComputeNodeSignedDistance`, positive = inside).
 *          A cell centre is considered inside when its signed distance is greater than
 *          or equal to `-tolerance`, i.e.\ strictly inside or within `tolerance` of the
 *          surface.  Cells that pass receive `color`.
 *
 *          ### Bounding-box pre-filter
 *          When `bounding_box.min_point` and `bounding_box.max_point` are both 3-vectors,
 *          only cells whose centre lies within this axis-aligned box are tested at all;
 *          cells outside the box are left unchanged.  When either array is empty
 *          (the default), every cell is a candidate.
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised yet (or its
 *          size differs from `mCells.size()`), it is resized and filled with 0 before
 *          any colour is written.  Already-coloured cells may be overwritten when their
 *          centre is found inside.
 *
 * ### Colour convention
 * | Value | Typical meaning |
 * |-------|-----------------|
 * |  `0`  | Uncoloured / outside (default) |
 * |  `1`  | Inside the main surface |
 * | other | User-defined sub-region |
 *
 * ### Typical pipeline position
 * Can be used as an alternative to `OctreeHybridClassifyCellsInsideOutside` for
 * labelling cells inside a (possibly different) reference surface, or to mark a
 * sub-region inside an auxiliary ModelPart.
 *
 * ### Usage
 * Add the following block to the `"coloring_settings_list"` in the modeler JSON:
 * @code{.json}
 * {
 *     "type"            : "OctreeHybridColorCellsWithInsideCenter",
 *     "model_part_name" : "MySurface",
 *     "color"           : 1,
 *     "input_entities"  : "",
 *     "tolerance"       : 1.0e-12,
 *     "bounding_box"  : {
 *         "min_point" : [],
 *         "max_point" : []
 *     }
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridClassifyCellsInsideOutside
 * @see OctreeHybridColorCellsInTouch
 * @see OctreeHybridMeshUtility::ComputeNodeSignedDistance
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellsWithInsideCenter : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellsWithInsideCenter() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellsWithInsideCenter(OctreeHybridColorCellsWithInsideCenter const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours every hex cell whose centre lies inside the geometry of the
     *        input ModelPart.
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
     *     "type"            : "OctreeHybridColorCellsWithInsideCenter",
     *     "model_part_name" : "Undefined",
     *     "color"           : -1,
     *     "input_entities"  : "",
     *     "tolerance"       : 1.0e-12,
     *     "bounding_box"  : {
     *         "min_point" : [],
     *         "max_point" : []
     *     }
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

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsWithInsideCenter.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellsWithInsideCenter)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellsWithInsideCenter.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellsWithInsideCenter)

    ///@}
};

///@}

} // namespace Kratos
