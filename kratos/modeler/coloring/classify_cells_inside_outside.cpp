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

// System includes

// External includes

// Project includes
#include "modeler/coloring/classify_cells_inside_outside.h"
#include "modeler/octree_mesher_modeler.h"
#include "modeler/internals/octree_mesher_data.h"
#include "utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

/**
 * @brief Returns the default JSON parameter schema for @ref ClassifyCellsInsideOutside.
 * @details The schema contains a single key:
 *   - `"type"` (`string`, default `"ClassifyCellsInsideOutside"`): the Registry type
 *     token consumed by `OctreeMesherModeler::Dispatch`.
 *
 *   No further configuration keys are required because all classification data
 *   (triangle soup, node coordinates, hex connectivity) are already stored in the
 *   shared @ref OctreeMesherData object accessible through the modeler.
 * @return Parameters object with key `"type"` set to `"ClassifyCellsInsideOutside"`.
 */
const Parameters ClassifyCellsInsideOutside::GetDefaultParameters() const
{
    return Parameters(R"({
        "type" : "ClassifyCellsInsideOutside"
    })");
}

/**
 * @brief Classifies every hex cell in the modeler's shared data as inside (1) or outside (0).
 * @details The method operates on `rModeler.GetData()` (an @ref OctreeMesherData reference)
 *          and populates `OctreeMesherData::mCellColor` with one integer per cell:
 *
 *          - **Projected-mesh path** (`mProjected == true`): The surface-projection pass
 *            that runs during octree extraction has already removed all cells whose
 *            centroid fell outside the surface.  The remaining cells are all inside, so
 *            `mCellColor` is filled with `1` in a single `std::vector::assign` call —
 *            no ray casting is performed.
 *
 *          - **Non-projected path** (`mProjected == false`): Delegates to
 *            `OctreeHybridMeshUtility::ClassifyInsideOutside`, which uses a ray-cast
 *            parity test (odd crossing count == inside) together with the signed
 *            closest-triangle distance to produce a robust classification.  The four
 *            arguments passed are:
 *              - `mTriangles`: world-space surface triangle soup (the carving geometry).
 *              - `mNodes`: world-space hex node coordinates.
 *              - `mCells`: hex connectivity (8 node indices per cell).
 *              - `mCellColor`: output vector, resized and filled by the utility.
 *
 * @param rModeler           The owning modeler; its `OctreeMesherData::mCellColor` is
 *                           populated in-place.  The method does not modify the octree,
 *                           the node coordinates, or the connectivity.
 * @param ColoringParameters Validated JSON parameters (unused beyond validation; the
 *                           only key is `"type"`).
 *
 * @note `ColoringParameters` is accepted by value (copied from the JSON stage array)
 *       to match the virtual-function signature of @ref OctreeMesherColoring::Apply.
 *       The parameter is intentionally unnamed in the definition because it is not
 *       consumed inside the body.
 *
 * @see OctreeHybridMeshUtility::ClassifyInsideOutside
 * @see OctreeMesherData::mCellColor
 * @see OctreeMesherData::mProjected
 */
void ClassifyCellsInsideOutside::Apply(OctreeMesherModeler& rModeler, Parameters) const
{
    auto& r_data = rModeler.GetData();

    // After projection the outside cells were already removed, so everything left
    // is inside; otherwise classify against the surface (carve decision only).
    if (r_data.mProjected) {
        r_data.mCellColor.assign(r_data.mCells.size(), 1);
        return;
    }

    OctreeHybridMeshUtility::ClassifyInsideOutside(
        r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellColor);
}

} // namespace Kratos
