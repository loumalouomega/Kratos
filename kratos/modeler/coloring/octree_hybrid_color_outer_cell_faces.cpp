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
#include <array>

// External includes

// Project includes
#include "modeler/coloring/octree_hybrid_color_outer_cell_faces.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos
{

const Parameters OctreeHybridColorOuterCellFaces::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"            : "OctreeHybridColorOuterCellFaces",
        "model_part_name" : "Undefined",
        "color"           : -1,
        "cell_color"      : -1
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridColorOuterCellFaces::Apply(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters ColoringParameters) const
{
    KRATOS_TRY

    auto& r_data = rModeler.GetData();
    const std::size_t n_cells = r_data.mCells.size();

    // Only reset the color vector when it has the wrong size so that prior
    // coloring steps' assignments are preserved (see OctreeHybridColorCellsByLevel
    // for the same rationale).
    if (r_data.mCellColor.size() != n_cells)
        r_data.mCellColor.assign(n_cells, 0);

    if (r_data.mCellFaceColor.size() != n_cells)
        r_data.mCellFaceColor.assign(n_cells, std::array<int, 6>{0, 0, 0, 0, 0, 0});

    const int face_color = ColoringParameters["color"].GetInt();
    const int cell_color = ColoringParameters["cell_color"].GetInt();

    const auto neighbors = OctreeHybridMeshUtility::ComputeCellFaceNeighbors(r_data.mCells);

    for (std::size_t c = 0; c < n_cells; ++c) {
        if (r_data.mCellColor[c] != cell_color) continue;
        for (int f = 0; f < 6; ++f) {
            if (neighbors[c][f] == -1)
                r_data.mCellFaceColor[c][f] = face_color;
        }
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
