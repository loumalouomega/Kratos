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
#include <cmath>

// External includes

// Project includes
#include "modeler/coloring/octree_hybrid_color_cell_faces.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"
#include "geometries/geometry.h"

namespace Kratos
{

namespace
{
    /// Local-node indices of the 6 hex faces, FACE_FIDC/Hexahedra3D8 ordering
    /// (matches the convention used by OctreeHybridMeshUtility::ComputeCellFaceNeighbors
    /// and OctreeHybridMesherData::mCellFaceColor).
    constexpr int FaceFidc[6][4] = {
        {0,1,2,3}, {4,5,1,0}, {4,0,3,7}, {5,6,2,1}, {6,7,3,2}, {7,6,5,4}
    };
} // anonymous namespace

const Parameters OctreeHybridColorCellFaces::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"            : "OctreeHybridColorCellFaces",
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

void OctreeHybridColorCellFaces::Apply(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters ColoringParameters) const
{
    KRATOS_TRY

    auto& r_data = rModeler.GetData();
    const std::size_t n_cells = r_data.mCells.size();

    if (r_data.mCellFaceColor.size() != n_cells)
        r_data.mCellFaceColor.assign(n_cells, std::array<int, 6>{0, 0, 0, 0, 0, 0});

    const int color                   = ColoringParameters["color"].GetInt();
    const std::string model_part_name = ColoringParameters["model_part_name"].GetString();
    const std::string input_entities  = ColoringParameters["input_entities"].GetString();
    const double tolerance            = ColoringParameters["tolerance"].GetDouble();

    // Optional AABB pre-filter: only faces whose centroid lies within bounding_box
    // are candidates. An empty bounding_box (the default) means no restriction.
    const Parameters bounding_box = ColoringParameters["bounding_box"];
    const bool has_bounding_box = bounding_box["min_point"].size() == 3 && bounding_box["max_point"].size() == 3;
    array_1d<double, 3> bb_min, bb_max;
    if (has_bounding_box) {
        bb_min = bounding_box["min_point"].GetVector();
        bb_max = bounding_box["max_point"].GetVector();
    }

    // Collect candidate (cell, local face) pairs and their centroids.
    std::vector<std::pair<std::size_t, int>> candidate_faces;
    std::vector<std::array<double, 3>> centers;
    for (std::size_t c = 0; c < n_cells; ++c) {
        for (int f = 0; f < 6; ++f) {
            std::array<double, 3> center{0.0, 0.0, 0.0};
            for (int k = 0; k < 4; ++k) {
                const auto& r_node = r_data.mNodes[r_data.mCells[c][FaceFidc[f][k]]];
                for (int d = 0; d < 3; ++d) center[d] += r_node[d];
            }
            for (int d = 0; d < 3; ++d) center[d] /= 4.0;

            if (has_bounding_box) {
                if (center[0] < bb_min[0] || center[0] > bb_max[0] ||
                    center[1] < bb_min[1] || center[1] > bb_max[1] ||
                    center[2] < bb_min[2] || center[2] > bb_max[2])
                    continue;
            }

            candidate_faces.emplace_back(c, f);
            centers.push_back(center);
        }
    }

    if (input_entities.empty()) {
        // Manual override: color every candidate face directly, no geometry test.
        for (const auto& [c, f] : candidate_faces)
            r_data.mCellFaceColor[c][f] = color;
    } else {
        // Automatic strategy: build the triangle soup defining the reference surface.
        KRATOS_ERROR_IF_NOT(rModeler.GetModel().HasModelPart(model_part_name))
            << "OctreeHybridColorCellFaces: ModelPart '" << model_part_name
            << "' not found." << std::endl;

        const ModelPart& r_mp = rModeler.GetModel().GetModelPart(model_part_name);

        OctreeHybridMeshUtility::TriangleSoup triangles;
        auto append_geometry = [&triangles](const Geometry<Node>& rGeometry) {
            if (rGeometry.PointsNumber() < 3) return;
            triangles.push_back({{
                {{ rGeometry[0].X(), rGeometry[0].Y(), rGeometry[0].Z() }},
                {{ rGeometry[1].X(), rGeometry[1].Y(), rGeometry[1].Z() }},
                {{ rGeometry[2].X(), rGeometry[2].Y(), rGeometry[2].Z() }}
            }});
        };

        if (input_entities == "geometries") {
            triangles = OctreeHybridMeshUtility::ExtractTriangleSoup(r_mp);
        } else if (input_entities == "elements") {
            triangles.reserve(r_mp.NumberOfElements());
            for (const auto& r_elem : r_mp.Elements())
                append_geometry(r_elem.GetGeometry());
        } else if (input_entities == "conditions") {
            triangles.reserve(r_mp.NumberOfConditions());
            for (const auto& r_cond : r_mp.Conditions())
                append_geometry(r_cond.GetGeometry());
        } else {
            KRATOS_ERROR << "OctreeHybridColorCellFaces: unsupported input_entities '"
                         << input_entities
                         << "'. Valid values: \"\", \"geometries\", \"elements\", \"conditions\"."
                         << std::endl;
        }

        const std::vector<double> signed_distance =
            OctreeHybridMeshUtility::ComputeNodeSignedDistance(triangles, centers);

        for (std::size_t k = 0; k < candidate_faces.size(); ++k) {
            if (std::abs(signed_distance[k]) <= tolerance) {
                const auto& [c, f] = candidate_faces[k];
                r_data.mCellFaceColor[c][f] = color;
            }
        }
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
