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
#include "modeler/coloring/octree_hybrid_color_cells_with_inside_center.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"
#include "geometries/geometry.h"

namespace Kratos
{

const Parameters OctreeHybridColorCellsWithInsideCenter::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"            : "OctreeHybridColorCellsWithInsideCenter",
        "model_part_name" : "Undefined",
        "color"           : -1,
        "input_entities"  : "",
        "tolerance"       : 1.0e-12,
        "bounding_box"  : {
            "min_point" : [],
            "max_point" : []
        }
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridColorCellsWithInsideCenter::Apply(
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

    const int color                   = ColoringParameters["color"].GetInt();
    const std::string model_part_name = ColoringParameters["model_part_name"].GetString();
    const std::string input_entities  = ColoringParameters["input_entities"].GetString();
    const double tolerance            = ColoringParameters["tolerance"].GetDouble();

    KRATOS_ERROR_IF_NOT(rModeler.GetModel().HasModelPart(model_part_name))
        << "OctreeHybridColorCellsWithInsideCenter: ModelPart '" << model_part_name
        << "' not found." << std::endl;

    const ModelPart& r_mp = rModeler.GetModel().GetModelPart(model_part_name);

    // Build the triangle soup defining the "inside" surface from the requested entities.
    OctreeHybridMeshUtility::TriangleSoup triangles;
    auto append_geometry = [&triangles](const Geometry<Node>& rGeometry) {
        if (rGeometry.PointsNumber() < 3) return;
        triangles.push_back({{
            {{ rGeometry[0].X(), rGeometry[0].Y(), rGeometry[0].Z() }},
            {{ rGeometry[1].X(), rGeometry[1].Y(), rGeometry[1].Z() }},
            {{ rGeometry[2].X(), rGeometry[2].Y(), rGeometry[2].Z() }}
        }});
    };

    const auto geometries = OctreeHybridMeshUtility::ResolveInputEntityGeometries(
        r_mp, input_entities, "OctreeHybridColorCellsWithInsideCenter");
    triangles.reserve(geometries.size());
    for (const Geometry<Node>* p_geom : geometries)
        append_geometry(*p_geom);

    // Optional AABB pre-filter: only cells whose centre lies within bounding_box are tested.
    const Parameters bounding_box = ColoringParameters["bounding_box"];
    const bool has_bounding_box = bounding_box["min_point"].size() == 3 && bounding_box["max_point"].size() == 3;
    array_1d<double, 3> bb_min, bb_max;
    if (has_bounding_box) {
        bb_min = bounding_box["min_point"].GetVector();
        bb_max = bounding_box["max_point"].GetVector();
    }

    // Cell centres (centroid of the 8 corner nodes) for every candidate cell.
    std::vector<std::size_t> candidate_cells;
    std::vector<std::array<double, 3>> centers;
    candidate_cells.reserve(n_cells);
    centers.reserve(n_cells);
    for (std::size_t i = 0; i < n_cells; ++i) {
        std::array<double, 3> center{0.0, 0.0, 0.0};
        for (int j = 0; j < 8; ++j) {
            const auto& r_node = r_data.mNodes[r_data.mCells[i][j]];
            for (int d = 0; d < 3; ++d) center[d] += r_node[d];
        }
        for (int d = 0; d < 3; ++d) center[d] /= 8.0;

        if (has_bounding_box) {
            if (center[0] < bb_min[0] || center[0] > bb_max[0] ||
                center[1] < bb_min[1] || center[1] > bb_max[1] ||
                center[2] < bb_min[2] || center[2] > bb_max[2])
                continue;
        }

        candidate_cells.push_back(i);
        centers.push_back(center);
    }

    // Positive = inside, negative = outside (see OctreeHybridMeshUtility::ComputeNodeSignedDistance).
    const std::vector<double> signed_distance =
        OctreeHybridMeshUtility::ComputeNodeSignedDistance(triangles, centers);

    for (std::size_t k = 0; k < candidate_cells.size(); ++k) {
        if (signed_distance[k] >= -tolerance)
            r_data.mCellColor[candidate_cells[k]] = color;
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
