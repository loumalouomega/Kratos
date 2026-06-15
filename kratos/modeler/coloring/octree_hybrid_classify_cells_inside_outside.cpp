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
#include "modeler/coloring/octree_hybrid_classify_cells_inside_outside.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos 
{

const Parameters OctreeHybridClassifyCellsInsideOutside::GetDefaultParameters() const
{
    return Parameters(R"({
        "type" : "OctreeHybridClassifyCellsInsideOutside"
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridClassifyCellsInsideOutside::Apply(OctreeHybridMeshGeneratorModeler& rModeler, Parameters) const
{
    auto& r_data = rModeler.GetData();

    // When the dual mesh has been projected onto the surface, the carve/project
    // pipeline is non-destructive: mCells retains outside cells alongside the core
    // and buffer-shell cells, classified per-cell in mCarveStatus (0 = outside,
    // 1 = core, 2 = buffer shell). Map that to the inside (1) / outside (0)
    // convention directly, without invoking the ray-caster again.
    if (r_data.mProjected) {
        const std::size_t n_cells = r_data.mCells.size();
        r_data.mCellColor.assign(n_cells, 0);
        for (std::size_t i = 0; i < n_cells; ++i) {
            if (r_data.mCarveStatus[i] != 0) {
                r_data.mCellColor[i] = 1;
            }
        }
        return;
    }

    // Non-projected path: classify each cell by the sign of its corner distances.
    // 1 = inside (at least one corner inside and the outside penetration is within
    // the OUT_IN_RATIO tolerance), 0 = outside.
    OctreeHybridMeshUtility::ClassifyInsideOutside(
        r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellColor);
}

} // namespace Kratos
