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
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stack>
#include <vector>

// External includes

// Project includes
#include "modeler/coloring/octree_hybrid_color_connected_cells_in_touch.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "geometries/point.h"

namespace Kratos
{

const Parameters OctreeHybridColorConnectedCellsInTouch::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"            : "OctreeHybridColorConnectedCellsInTouch",
        "model_part_name" : "",
        "color"           : 1,
        "cell_color"      : 0,
        "input_entities"  : "geometries"
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridColorConnectedCellsInTouch::Apply(
    OctreeHybridMesherModeler& rModeler,
    Parameters ColoringParameters) const
{
    KRATOS_TRY

    auto& r_data = rModeler.GetData();
    const std::size_t n_cells = r_data.mCells.size();

    // Only reset when the vector has the wrong size so prior coloring steps'
    // assignments are preserved (see OctreeHybridColorCellsByLevel for rationale).
    if (r_data.mCellColor.size() != n_cells)
        r_data.mCellColor.assign(n_cells, 0);

    const int color                   = ColoringParameters["color"].GetInt();
    const int cell_color              = ColoringParameters["cell_color"].GetInt();
    const std::string model_part_name = ColoringParameters["model_part_name"].GetString();
    const std::string input_entities  = ColoringParameters["input_entities"].GetString();

    KRATOS_ERROR_IF_NOT(rModeler.GetModel().HasModelPart(model_part_name))
        << "OctreeHybridColorConnectedCellsInTouch: ModelPart '" << model_part_name
        << "' not found." << std::endl;

    const ModelPart& r_mp = rModeler.GetModel().GetModelPart(model_part_name);

    // --- Precompute per-cell AABB ---
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

    // --- Build face-adjacency list ---
    // Six faces of each hex by local node indices (Hexahedra3D8 ordering).
    static const int face_nodes[6][4] = {
        {0,1,2,3}, {4,5,6,7}, {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7}
    };

    // std::map is used instead of unordered_map because array<int,4> has no
    // standard hash; the lexicographic ordering of std::map suffices here and
    // avoids a custom hash implementation.
    using FaceKey = std::array<int, 4>;
    std::map<FaceKey, std::vector<int>> face_to_cells;
    face_to_cells.clear();

    for (int ci = 0; ci < static_cast<int>(n_cells); ++ci) {
        for (int fi = 0; fi < 6; ++fi) {
            FaceKey key;
            for (int k = 0; k < 4; ++k)
                key[k] = r_data.mCells[ci][face_nodes[fi][k]];
            // Sort the four global node indices so the key is orientation-independent:
            // the same face seen from two different cells will produce the same sorted key.
            std::sort(key.begin(), key.end());
            face_to_cells[key].push_back(ci);
        }
    }

    // adjacency[i] = list of cells sharing a face with cell i.
    // A face shared by exactly 2 cells is an interior face — only those form
    // valid adjacency edges.  A face seen by 1 cell is a boundary; by >2 is
    // non-manifold and is ignored to avoid propagating across degenerate geometry.
    std::vector<std::vector<int>> adjacency(n_cells);
    for (const auto& [key, cells] : face_to_cells) {
        if (cells.size() == 2) {
            adjacency[cells[0]].push_back(cells[1]);
            adjacency[cells[1]].push_back(cells[0]);
        }
    }

    // --- Geometry intersection check (same AABB + HasIntersection as ColorCellsInTouch) ---
    auto touches_geometry = [&](std::size_t ci, const Geometry<Node>& rGeometry,
                                 const double g_min[3], const double g_max[3]) -> bool {
        const auto& bb = cell_aabb[ci];
        if (bb[3] < g_min[0] || bb[0] > g_max[0]) return false;
        if (bb[4] < g_min[1] || bb[1] > g_max[1]) return false;
        if (bb[5] < g_min[2] || bb[2] > g_max[2]) return false;
        const Point cell_min(bb[0], bb[1], bb[2]);
        const Point cell_max(bb[3], bb[4], bb[5]);
        return rGeometry.HasIntersection(cell_min, cell_max);
    };

    // --- Find seeds: cells with cell_color that touch a geometry ---
    // `visited` is shared between the seed-finding pass and the BFS so that a
    // cell pushed as a seed is not pushed again by a second overlapping geometry.
    std::vector<bool> visited(n_cells, false);
    std::stack<int> work;

    auto seed_from_geometry = [&](const Geometry<Node>& rGeometry) {
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
        for (std::size_t ci = 0; ci < n_cells; ++ci) {
            if (r_data.mCellColor[ci] != cell_color) continue;
            // Mark as visited before pushing so that a cell touched by multiple
            // seed geometries is only pushed once.
            if (visited[ci]) continue;
            if (touches_geometry(ci, rGeometry, g_min, g_max)) {
                visited[ci] = true;
                work.push(static_cast<int>(ci));
            }
        }
    };

    if (input_entities == "geometries") {
        for (const auto& r_geom : r_mp.Geometries())
            seed_from_geometry(r_geom);
    } else if (input_entities == "elements") {
        for (const auto& r_elem : r_mp.Elements())
            seed_from_geometry(r_elem.GetGeometry());
    } else if (input_entities == "conditions") {
        for (const auto& r_cond : r_mp.Conditions())
            seed_from_geometry(r_cond.GetGeometry());
    } else {
        KRATOS_ERROR << "OctreeHybridColorConnectedCellsInTouch: unsupported input_entities '"
                     << input_entities
                     << "'. Valid values: \"geometries\", \"elements\", \"conditions\"."
                     << std::endl;
    }

    // --- BFS flood-fill through cells with cell_color ---
    // The color check (mCellColor[nb] == cell_color) gates the flood at region
    // boundaries: cells with a different color (e.g. already-colored interior
    // cells from a prior pass) act as walls that stop propagation.  This lets
    // the connected exterior be labeled with a distinct color without spilling
    // into inner pockets that were separately assigned a different color.
    while (!work.empty()) {
        const int ci = work.top();
        work.pop();
        r_data.mCellColor[ci] = color;
        for (int nb : adjacency[ci]) {
            if (!visited[nb] && r_data.mCellColor[nb] == cell_color) {
                visited[nb] = true;
                work.push(nb);
            }
        }
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
