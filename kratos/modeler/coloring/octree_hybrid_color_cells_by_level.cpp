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
#include "modeler/coloring/octree_hybrid_color_cells_by_level.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos
{

const Parameters OctreeHybridColorCellsByLevel::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"      : "OctreeHybridColorCellsByLevel",
        "color"     : 1,
        "min_level" : 1,
        "max_level" : 100
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridColorCellsByLevel::Apply(
    OctreeHybridMesherModeler& rModeler,
    Parameters ColoringParameters) const
{
    KRATOS_TRY

    auto& r_data = rModeler.GetData();
    const std::size_t n_cells = r_data.mCells.size();

    // Only reset the color vector when it has the wrong size.  A prior coloring
    // step may have already assigned colors to some cells; resizing only on a
    // size mismatch preserves those prior assignments while still initialising
    // the vector when this is the first coloring operation.
    if (r_data.mCellColor.size() != n_cells)
        r_data.mCellColor.assign(n_cells, 0);

    const int color     = ColoringParameters["color"].GetInt();
    const int min_level = ColoringParameters["min_level"].GetInt();
    const int max_level = ColoringParameters["max_level"].GetInt();

    KRATOS_ERROR_IF(min_level > max_level)
        << "OctreeHybridColorCellsByLevel: min_level (" << min_level
        << ") must be <= max_level (" << max_level << ")." << std::endl;

    // Cells with mCellLevel == -1 (13-element transition template hexes) or
    // -2 (buffer-layer hexes added during surface projection) are never colored
    // by this operation because negative levels fall below any valid min_level.
    // This is intentional: transition and buffer cells are mesh-quality
    // infrastructure and should not participate in level-based entity generation.
    for (std::size_t i = 0; i < n_cells; ++i) {
        const int level = r_data.mCellLevel[i];
        if (level >= min_level && level <= max_level)
            r_data.mCellColor[i] = color;
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
