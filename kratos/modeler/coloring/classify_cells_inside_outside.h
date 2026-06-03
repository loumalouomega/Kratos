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
#include "modeler/coloring/octree_mesher_coloring.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class ClassifyCellsInsideOutside
 * @ingroup KratosCore
 * @brief Colouring stage that labels every extracted hex cell as inside (1) or outside (0)
 *        the input surface mesh.
 * @details The classification is performed by
 *          `OctreeHybridMeshUtility::ClassifyInsideOutside`, which combines a ray-cast
 *          parity test (sign) with the closest-triangle signed distance (magnitude) to
 *          produce a robust inside/outside decision for each cell's representative point.
 *
 *          ### Projected-mesh short-circuit
 *          When `OctreeMesherData::mProjected == true`, the surface-projection pass has
 *          already carved away all outside cells during the extraction phase.  In that
 *          case every surviving cell is definitively inside, so the classification loop
 *          is replaced by a single `std::vector::assign` that sets all entries to 1
 *          without touching the ray-caster.
 *
 *          ### Colour convention
 *          | Value | Meaning  |
 *          |-------|----------|
 *          |   1   | Inside   |
 *          |   0   | Outside  |
 *
 *          Results are written into `OctreeMesherData::mCellColor`; the vector is
 *          resized to match `OctreeMesherData::mCells.size()` before any entry is set.
 *
 * ### Usage
 * Add the following block to the `"colorings"` list in the modeler JSON parameters:
 * @code{.json}
 * { "type" : "ClassifyCellsInsideOutside" }
 * @endcode
 *
 * @note This stage must run **after** the octree has been built and the hex mesh
 *       extracted (i.e.\ after `SetupModelPart` has called `BuildOctreeAndExtract`),
 *       and **before** any entity-generation stage that filters by colour.
 *
 * @see OctreeMesherColoring
 * @see OctreeMesherEntityGeneration
 * @see GenerateHexesByCellColor
 * @see OctreeHybridMeshUtility::ClassifyInsideOutside
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) ClassifyCellsInsideOutside : public OctreeMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    ClassifyCellsInsideOutside() = default;

    /**
     * @brief Copy constructor (no-op body — this class carries no state).
     * @param rOther Source instance; no data is copied.
     */
    ClassifyCellsInsideOutside(ClassifyCellsInsideOutside const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Classifies every hex cell in `rModeler.GetData()` as inside (1) or outside (0).
     * @details If `rModeler.GetData().mProjected` is `true`, assigns 1 to all entries in
     *          `mCellColor` directly (the projection pass already removed outside cells).
     *          Otherwise delegates to `OctreeHybridMeshUtility::ClassifyInsideOutside`,
     *          passing the triangle soup, node coordinates, connectivity and the colour
     *          array to fill.
     *
     * @param rModeler            The owning modeler; its `OctreeMesherData` is modified
     *                            in-place (`mCellColor` is populated).
     * @param ColoringParameters  Validated JSON parameters for this step (all keys are
     *                            defaults from @ref GetDefaultParameters; the base
     *                            `"type"` key is the only one in the schema).
     */
    void Apply(OctreeMesherModeler& rModeler, Parameters ColoringParameters) const override;

    /**
     * @brief Returns the default parameter schema for this colouring stage.
     * @details The schema contains only the mandatory `"type"` key set to
     *          `"ClassifyCellsInsideOutside"`.  No additional configuration is
     *          needed because the classification is fully driven by the shared
     *          octree data already stored in the modeler.
     * @return Parameters object:
     * @code{.json}
     * { "type" : "ClassifyCellsInsideOutside" }
     * @endcode
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class at path "OctreeMesherColoring.KratosMultiphysics.ClassifyCellsInsideOutside.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.KratosMultiphysics", OctreeMesherColoring, ClassifyCellsInsideOutside)
    /// Registers this class at path "OctreeMesherColoring.All.ClassifyCellsInsideOutside.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.All", OctreeMesherColoring, ClassifyCellsInsideOutside)

    ///@}
};

///@}

} // namespace Kratos
