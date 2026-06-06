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
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters OctreeHybridClassifyCellsInsideOutside::GetDefaultParameters() const
{
    return Parameters(R"({
        "type" : "OctreeHybridClassifyCellsInsideOutside"
    })");
}

void OctreeHybridClassifyCellsInsideOutside::Apply(OctreeHybridMesherModeler& rModeler, Parameters) const
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
