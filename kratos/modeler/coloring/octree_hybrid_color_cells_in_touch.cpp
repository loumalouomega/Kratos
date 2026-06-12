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
#include <limits>

// External includes

// Project includes
#include "modeler/coloring/octree_hybrid_color_cells_in_touch.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "geometries/point.h"

namespace Kratos
{

const Parameters OctreeHybridColorCellsInTouch::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"            : "OctreeHybridColorCellsInTouch",
        "model_part_name" : "",
        "color"           : 1,
        "input_entities"  : "geometries"
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridColorCellsInTouch::Apply(
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

    const int color                    = ColoringParameters["color"].GetInt();
    const std::string model_part_name  = ColoringParameters["model_part_name"].GetString();
    const std::string input_entities   = ColoringParameters["input_entities"].GetString();

    KRATOS_ERROR_IF_NOT(rModeler.GetModel().HasModelPart(model_part_name))
        << "OctreeHybridColorCellsInTouch: ModelPart '" << model_part_name
        << "' not found." << std::endl;

    const ModelPart& r_mp = rModeler.GetModel().GetModelPart(model_part_name);

    // Precompute per-cell AABB to avoid recomputing it for each geometry.
    // Layout: {min_x, min_y, min_z, max_x, max_y, max_z}
    std::vector<std::array<double, 6>> cell_aabb(n_cells);
    for (std::size_t i = 0; i < n_cells; ++i) {
        double mn[3] = {std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max()};
        double mx[3] = {-std::numeric_limits<double>::max(),
                        -std::numeric_limits<double>::max(),
                        -std::numeric_limits<double>::max()};
        for (int j = 0; j < 8; ++j) {
            const auto& nd = r_data.mNodes[r_data.mCells[i][j]];
            for (int d = 0; d < 3; ++d) {
                if (nd[d] < mn[d]) mn[d] = nd[d];
                if (nd[d] > mx[d]) mx[d] = nd[d];
            }
        }
        cell_aabb[i] = {mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]};
    }

    // Tracks cells already coloured by THIS stage so that the SAT test is not
    // repeated for them against later geometries. Cannot reuse `mCellColor[i] ==
    // color` for this purpose: ApplyColoring pre-fills mCellColor with
    // `default_outside_color` (1 by default), which may already equal `color`
    // before this stage examines any geometry.
    std::vector<bool> touched(n_cells, false);

    auto color_cells = [&](const Geometry<Node>& rGeometry) {
        // Geometry AABB
        double g_min[3] = {std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max()};
        double g_max[3] = {-std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max()};
        for (std::size_t k = 0; k < rGeometry.PointsNumber(); ++k) {
            const auto& p = rGeometry[k];
            if (p.X() < g_min[0]) g_min[0] = p.X();
            if (p.Y() < g_min[1]) g_min[1] = p.Y();
            if (p.Z() < g_min[2]) g_min[2] = p.Z();
            if (p.X() > g_max[0]) g_max[0] = p.X();
            if (p.Y() > g_max[1]) g_max[1] = p.Y();
            if (p.Z() > g_max[2]) g_max[2] = p.Z();
        }

        for (std::size_t i = 0; i < n_cells; ++i) {
            // Skip cells already coloured by this stage — once coloured, a cell
            // cannot be un-coloured by subsequent geometries in the same pass,
            // and the SAT test below would be redundant.
            if (touched[i]) continue;
            const auto& bb = cell_aabb[i];
            // AABB quick-reject: bb layout is {min_x,min_y,min_z,max_x,max_y,max_z}.
            // Rejecting here costs six comparisons and avoids the SAT call for the
            // vast majority of cells that don't overlap the geometry's bounding box.
            if (bb[3] < g_min[0] || bb[0] > g_max[0]) continue;
            if (bb[4] < g_min[1] || bb[1] > g_max[1]) continue;
            if (bb[5] < g_min[2] || bb[2] > g_max[2]) continue;
            // Precise SAT (Separating Axis Theorem) test: handles non-box
            // geometries (triangles, quads) that pass the AABB overlap but do
            // not actually intersect the cell.
            const Point cell_min(bb[0], bb[1], bb[2]);
            const Point cell_max(bb[3], bb[4], bb[5]);
            if (rGeometry.HasIntersection(cell_min, cell_max)) {
                r_data.mCellColor[i] = color;
                touched[i] = true;
            }
        }
    };

    if (input_entities == "geometries") {
        for (const auto& r_geom : r_mp.Geometries())
            color_cells(r_geom);
    } else if (input_entities == "elements") {
        for (const auto& r_elem : r_mp.Elements())
            color_cells(r_elem.GetGeometry());
    } else if (input_entities == "conditions") {
        for (const auto& r_cond : r_mp.Conditions())
            color_cells(r_cond.GetGeometry());
    } else {
        KRATOS_ERROR << "OctreeHybridColorCellsInTouch: unsupported input_entities '"
                     << input_entities
                     << "'. Valid values: \"geometries\", \"elements\", \"conditions\"."
                     << std::endl;
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
