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
#include "modeler/coloring/octree_hybrid_color_cells_by_carve_status.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos
{

const Parameters OctreeHybridColorCellsByCarveStatus::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"       : "OctreeHybridColorCellsByCarveStatus",
        "color"      : 1,
        "min_status" : 0,
        "max_status" : 2
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridColorCellsByCarveStatus::Apply(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters ColoringParameters) const
{
    KRATOS_TRY

    auto& r_data = rModeler.GetData();
    const std::size_t n_cells = r_data.mCells.size();

    KRATOS_ERROR_IF_NOT(r_data.mCarveStatus.size() == n_cells)
        << "OctreeHybridColorCellsByCarveStatus: mCarveStatus has size "
        << r_data.mCarveStatus.size() << " but there are " << n_cells
        << " cells. This stage requires a refinement entry with "
        << "\"project_to_surface\": true to have run first." << std::endl;

    // Only reset the color vector when it has the wrong size.  A prior coloring
    // step may have already assigned colors to some cells; resizing only on a
    // size mismatch preserves those prior assignments while still initialising
    // the vector when this is the first coloring operation.
    if (r_data.mCellColor.size() != n_cells)
        r_data.mCellColor.assign(n_cells, 0);

    const int color      = ColoringParameters["color"].GetInt();
    const int min_status = ColoringParameters["min_status"].GetInt();
    const int max_status = ColoringParameters["max_status"].GetInt();

    KRATOS_ERROR_IF(min_status > max_status)
        << "OctreeHybridColorCellsByCarveStatus: min_status (" << min_status
        << ") must be <= max_status (" << max_status << ")." << std::endl;

    for (std::size_t i = 0; i < n_cells; ++i) {
        const int status = r_data.mCarveStatus[i];
        if (status >= min_status && status <= max_status)
            r_data.mCellColor[i] = color;
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
