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
#include "includes/condition.h"
#include "includes/kratos_components.h"
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/octree_hybrid_generate_boundary_conditions_by_face.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters OctreeHybridGenerateBoundaryConditionsByFace::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"               : "OctreeHybridGenerateBoundaryConditionsByFace",
        "model_part_name"    : "Undefined",
        "color"              : 1,
        "properties_id"      : 1,
        "generated_entity"   : "SurfaceCondition3D4N"
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridGenerateBoundaryConditionsByFace::Generate(
    OctreeHybridMesherModeler& rModeler,
    Parameters GenerationParameters) const
{
    auto& r_data = rModeler.GetData();
    KRATOS_ERROR_IF(!r_data.IsExtracted())
        << "OctreeHybridGenerateBoundaryConditionsByFace: hex mesh not yet extracted." << std::endl;

    ModelPart& r_mp = rModeler.CreateAndGetModelPart(GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_mp);

    const int want_color = GenerationParameters["color"].GetInt();
    const std::size_t properties_id = GenerationParameters["properties_id"].GetInt();
    Properties::Pointer p_props = r_mp.HasProperties(properties_id)
        ? r_mp.pGetProperties(properties_id)
        : r_mp.CreateNewProperties(properties_id);
    const Condition& r_proto = KratosComponents<Condition>::Get(
        GenerationParameters["generated_entity"].GetString());

    // Build a colour-filtered cell list matching the hex generator's skip logic.
    // ExtractBoundaryFaces needs only connectivity (same indices as mNodes).
    std::vector<std::array<int,8>> active_cells;
    active_cells.reserve(r_data.mCells.size());
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_color) continue;
        active_cells.push_back(r_data.mCells[c]);
    }

    const auto bfaces = OctreeHybridMeshUtility::ExtractBoundaryFaces(active_cells);

    ModelPart::NodesContainerType new_nodes;
    ModelPart::ConditionsContainerType new_conditions;
    Condition::NodesArrayType face_nodes(4);
    for (const auto& bf : bfaces) {
        for (int k = 0; k < 4; ++k) {
            Node::Pointer p_node = rModeler.GenerateOrRetrieveNode(r_mp, new_nodes, bf[k]);
            face_nodes(k) = p_node;
            // If the node was created by a prior stage (e.g. hex generator) it won't
            // be in new_nodes, so push it explicitly so it lands in this ModelPart too.
            new_nodes.push_back(p_node);
        }
        new_conditions.push_back(r_proto.Create(rModeler.NextConditionId(), face_nodes, p_props));
    }

    new_nodes.Unique();
    ModelPartUtils::AddNodesFromOrderedContainer(r_mp, new_nodes.begin(), new_nodes.end());
    r_mp.AddConditions(new_conditions.begin(), new_conditions.end());
}

} // namespace Kratos
