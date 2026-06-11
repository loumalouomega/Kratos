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

    // When the dual mesh has been projected onto the surface, RemoveOutsideElement
    // already discarded every cell that was predominantly outside the closed surface
    // before ProjectToIsoSurface ran.  Every surviving cell is therefore inside by
    // construction — running the ray-cast classifier again would be redundant and
    // could mis-classify the buffer-layer cells (level == -2) whose centroids have
    // been moved to the surface during projection.
    if (r_data.mProjected) {
        r_data.mCellColor.assign(r_data.mCells.size(), 1);
        return;
    }

    // Non-projected path: classify each cell by the sign of its corner distances.
    // 1 = inside (at least one corner inside and the outside penetration is within
    // the OUT_IN_RATIO tolerance), 0 = outside.
    OctreeHybridMeshUtility::ClassifyInsideOutside(
        r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellColor);
}

} // namespace Kratos
