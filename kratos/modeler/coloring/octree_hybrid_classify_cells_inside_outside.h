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
 * @class OctreeHybridClassifyCellsInsideOutside
 * @ingroup KratosCore
 * @brief Colouring stage that labels every extracted hex cell as inside (1) or outside (0)
 *        the input surface mesh.
 * @details The classification is performed by
 *          `OctreeHybridMeshUtility::ClassifyInsideOutside`, which combines a ray-cast
 *          parity test (sign) with the closest-triangle signed distance (magnitude) to
 *          produce a robust inside/outside decision for each cell's representative point.
 *
 *          ### Projected-mesh classification
 *          When `OctreeHybridMesherData::mProjected == true`, the surface-projection pass
 *          is non-destructive: `OctreeHybridMesherData::mCells` retains the outside cells
 *          alongside the core (projected) and buffer-shell cells, classified per-cell in
 *          `OctreeHybridMesherData::mCarveStatus` (0 = outside, 1 = core, 2 = buffer
 *          shell). In that case the ray-caster is skipped and `mCellColor[i]` is set to 1
 *          for every cell with `mCarveStatus[i] != 0` (core or buffer shell) and to 0
 *          otherwise (outside).
 *
 *          ### Colour convention
 *          | Value | Meaning  |
 *          |-------|----------|
 *          |   1   | Inside   |
 *          |   0   | Outside  |
 *
 *          Results are written into `OctreeHybridMesherData::mCellColor`; the vector is
 *          resized to match `OctreeHybridMesherData::mCells.size()` before any entry is set.
 *
 * ### Usage
 * Add the following block to the `"colorings"` list in the modeler JSON parameters:
 * @code{.json}
 * { "type" : "OctreeHybridClassifyCellsInsideOutside" }
 * @endcode
 *
 * @note This stage must run **after** the octree has been built and the hex mesh
 *       extracted (i.e.\ after `SetupModelPart` has called `BuildOctreeAndExtract`),
 *       and **before** any entity-generation stage that filters by colour.
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridMesherEntityGeneration
 * @see GenerateOctreeHybridHexahedraElementsWithCellColor
 * @see OctreeHybridMeshUtility::ClassifyInsideOutside
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridClassifyCellsInsideOutside : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridClassifyCellsInsideOutside() = default;

    /**
     * @brief Copy constructor (no-op body — this class carries no state).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridClassifyCellsInsideOutside(OctreeHybridClassifyCellsInsideOutside const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Classifies every hex cell in `rModeler.GetData()` as inside (1) or outside (0).
     * @details If `rModeler.GetData().mProjected` is `true`, sets `mCellColor[i] = 1` for
     *          every cell with `mCarveStatus[i] != 0` (core or buffer shell) and `0`
     *          otherwise, without invoking the ray-caster.
     *          Otherwise delegates to `OctreeHybridMeshUtility::ClassifyInsideOutside`,
     *          passing the triangle soup, node coordinates, connectivity and the colour
     *          array to fill.
     *
     * @param rModeler            The owning modeler; its `OctreeHybridMesherData` is modified
     *                            in-place (`mCellColor` is populated).
     * @param ColoringParameters  Validated JSON parameters for this step (all keys are
     *                            defaults from @ref GetDefaultParameters; the base
     *                            `"type"` key is the only one in the schema).
     */
    void Apply(OctreeHybridMeshGeneratorModeler& rModeler, Parameters ColoringParameters) const override;

    /**
     * @brief Returns the default parameter schema for this colouring stage.
     * @details The schema contains only the mandatory `"type"` key set to
     *          `"OctreeHybridClassifyCellsInsideOutside"`.  No additional configuration is
     *          needed because the classification is fully driven by the shared
     *          octree data already stored in the modeler.
     * @return Parameters object:
     * @code{.json}
     * { "type" : "OctreeHybridClassifyCellsInsideOutside" }
     * @endcode
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class at path "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridClassifyCellsInsideOutside.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridClassifyCellsInsideOutside)
    /// Registers this class at path "OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridClassifyCellsInsideOutside)

    ///@}
};

///@}

} // namespace Kratos
