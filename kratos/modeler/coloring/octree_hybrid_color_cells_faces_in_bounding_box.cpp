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
#include "modeler/coloring/octree_hybrid_color_cells_faces_in_bounding_box.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "geometries/point.h"
#include "geometries/bounding_box.h"

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

    /**
     * @brief Resolves the axis-aligned bounding box for the *InBoundingBox colouring stages.
     * @details Uses `min_point`/`max_point` from @p rColoringParameters when both have
     *          exactly 3 components; otherwise computes the AABB of the nodes of
     *          `model_part_name`.
     */
    void ResolveBoundingBox(
        OctreeHybridMeshGeneratorModeler& rModeler,
        const Parameters& rColoringParameters,
        const std::string& rCallerName,
        array_1d<double, 3>& rMinPoint,
        array_1d<double, 3>& rMaxPoint)
    {
        if (rColoringParameters["min_point"].size() == 3 && rColoringParameters["max_point"].size() == 3) {
            rMinPoint = rColoringParameters["min_point"].GetVector();
            rMaxPoint = rColoringParameters["max_point"].GetVector();
            return;
        }

        const std::string model_part_name = rColoringParameters["model_part_name"].GetString();
        KRATOS_ERROR_IF_NOT(rModeler.GetModel().HasModelPart(model_part_name))
            << rCallerName << ": ModelPart '" << model_part_name
            << "' not found and no explicit \"min_point\"/\"max_point\" was given." << std::endl;

        const ModelPart& r_mp = rModeler.GetModel().GetModelPart(model_part_name);
        KRATOS_ERROR_IF(r_mp.NumberOfNodes() == 0)
            << rCallerName << ": ModelPart '" << model_part_name
            << "' has no nodes to compute a bounding box from." << std::endl;

        const BoundingBox<Point> bb(r_mp.NodesBegin(), r_mp.NodesEnd());
        rMinPoint = bb.GetMinPoint();
        rMaxPoint = bb.GetMaxPoint();
    }
} // anonymous namespace

const Parameters OctreeHybridColorCellsFacesInBoundingBox::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                : "OctreeHybridColorCellsFacesInBoundingBox",
        "color"               : -1,
        "model_part_name"     : "Undefined",
        "min_point"           : [],
        "max_point"           : [],
        "inside_bounding_box" : false
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridColorCellsFacesInBoundingBox::Apply(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters ColoringParameters) const
{
    KRATOS_TRY

    auto& r_data = rModeler.GetData();
    const std::size_t n_cells = r_data.mCells.size();

    if (r_data.mCellFaceColor.size() != n_cells)
        r_data.mCellFaceColor.assign(n_cells, std::array<int, 6>{0, 0, 0, 0, 0, 0});

    const int color                = ColoringParameters["color"].GetInt();
    const bool inside_bounding_box = ColoringParameters["inside_bounding_box"].GetBool();

    array_1d<double, 3> bb_min, bb_max;
    ResolveBoundingBox(rModeler, ColoringParameters, "OctreeHybridColorCellsFacesInBoundingBox", bb_min, bb_max);

    for (std::size_t c = 0; c < n_cells; ++c) {
        for (int f = 0; f < 6; ++f) {
            std::array<double, 3> center{0.0, 0.0, 0.0};
            for (int k = 0; k < 4; ++k) {
                const auto& r_node = r_data.mNodes[r_data.mCells[c][FaceFidc[f][k]]];
                for (int d = 0; d < 3; ++d) center[d] += r_node[d];
            }
            for (int d = 0; d < 3; ++d) center[d] /= 4.0;

            const bool is_inside = center[0] >= bb_min[0] && center[0] <= bb_max[0] &&
                                    center[1] >= bb_min[1] && center[1] <= bb_max[1] &&
                                    center[2] >= bb_min[2] && center[2] <= bb_max[2];

            if (is_inside == inside_bounding_box)
                r_data.mCellFaceColor[c][f] = color;
        }
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
