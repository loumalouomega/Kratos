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
#include <map>
#include <unordered_map>

// External includes

// Project includes
#include "utilities/parallel_utilities.h"
#include "modeler/operation/octree_hybrid_find_contacts_in_skin_model_part.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos {

namespace {

/// Local copy of the hex-face-to-corner-index table (see OctreeHybridMeshUtility::ExtractBoundaryFaces).
/// The actual winding does not matter here: the face key is sorted before lookup.
constexpr int FaceLocalNodes[6][4] = {
    {0,1,2,3}, {4,5,1,0}, {4,0,3,7}, {5,6,2,1}, {6,7,3,2}, {7,6,5,4}
};

/// Per-thread accumulator: outside_color -> (condition ids, node ids).
struct ContactContainer
{
    std::unordered_map<int, std::vector<std::size_t>> mNodeMap;
    std::unordered_map<int, std::vector<std::size_t>> mConditionsMap;
    std::unordered_map<int, ModelPart*> mModelPartMap;
    std::vector<int> mColorVector;
};

} // anonymous namespace

const Parameters OctreeHybridFindContactsInSkinModelPart::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                : "OctreeHybridFindContactsInSkinModelPart",
        "model_part_name"     : "Undefined",
        "contact_model_parts" : [],
        "cell_color"          : -1
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridFindContactsInSkinModelPart::Execute(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters OperationParameters) const
{
    KRATOS_TRY

    auto& r_data = rModeler.GetData();
    KRATOS_ERROR_IF_NOT(r_data.IsExtracted())
        << "OctreeHybridFindContactsInSkinModelPart: hex mesh not yet extracted." << std::endl;

    const std::string skin_model_part_name = OperationParameters["model_part_name"].GetString();
    KRATOS_ERROR_IF_NOT(rModeler.GetModel().HasModelPart(skin_model_part_name))
        << "OctreeHybridFindContactsInSkinModelPart: ModelPart '" << skin_model_part_name
        << "' not found." << std::endl;
    ModelPart& r_skin_model_part = rModeler.GetModel().GetModelPart(skin_model_part_name);

    const int cell_color = OperationParameters["cell_color"].GetInt();

    // Validate the per-entry schema of "contact_model_parts" and create/retrieve the
    // target ModelParts.
    Parameters contact_default_parameters(R"({
        "outside_color"           : -1,
        "contact_model_part_name" : "Undefined"
    })");

    ContactContainer contact_container;
    Parameters contact_model_parts = OperationParameters["contact_model_parts"];
    for (auto contact_parameters : contact_model_parts) {
        contact_parameters.ValidateAndAssignDefaults(contact_default_parameters);

        const int outside_color = contact_parameters["outside_color"].GetInt();
        ModelPart& r_contact_model_part = rModeler.CreateAndGetModelPart(
            contact_parameters["contact_model_part_name"].GetString());

        contact_container.mNodeMap.emplace(outside_color, std::vector<std::size_t>());
        contact_container.mConditionsMap.emplace(outside_color, std::vector<std::size_t>());
        contact_container.mModelPartMap.emplace(outside_color, &r_contact_model_part);
        contact_container.mColorVector.push_back(outside_color);
    }

    if (contact_container.mColorVector.empty()) {
        KRATOS_INFO("OctreeHybridFindContactsInSkinModelPart")
            << "No contact model parts requested for " << r_skin_model_part.FullName() << "." << std::endl;
        return;
    }

    // Reverse map: Kratos Node::Id() -> mesh-node index in r_data.mNodes.
    std::unordered_map<std::size_t, int> node_id_to_index;
    node_id_to_index.reserve(r_data.mNodePtrs.size());
    for (std::size_t i = 0; i < r_data.mNodePtrs.size(); ++i) {
        if (r_data.mNodePtrs[i]) {
            node_id_to_index.emplace(r_data.mNodePtrs[i]->Id(), static_cast<int>(i));
        }
    }

    // Face (sorted node-index tuple) -> owning cell indices, over the full cell set.
    std::map<std::array<int,4>, std::vector<int>> face_to_cells;
    for (int c = 0; c < static_cast<int>(r_data.mCells.size()); ++c) {
        for (int f = 0; f < 6; ++f) {
            std::array<int,4> key = {
                r_data.mCells[c][FaceLocalNodes[f][0]], r_data.mCells[c][FaceLocalNodes[f][1]],
                r_data.mCells[c][FaceLocalNodes[f][2]], r_data.mCells[c][FaceLocalNodes[f][3]]
            };
            std::sort(key.begin(), key.end());
            face_to_cells[key].push_back(c);
        }
    }

    const bool has_cell_color = !r_data.mCellColor.empty();

    KRATOS_INFO("OctreeHybridFindContactsInSkinModelPart")
        << "Finding contacts for " << r_skin_model_part.FullName() << " model part" << std::endl;

    block_for_each(r_skin_model_part.Conditions(), [&](Condition& rCondition) {
        const auto& r_geom = rCondition.GetGeometry();
        if (r_geom.PointsNumber() != 4) return;

        std::array<int,4> face_indices;
        for (unsigned int k = 0; k < 4; ++k) {
            auto it_node = node_id_to_index.find(r_geom[k].Id());
            if (it_node == node_id_to_index.end()) return;
            face_indices[k] = it_node->second;
        }

        std::array<int,4> key = face_indices;
        std::sort(key.begin(), key.end());
        auto it_face = face_to_cells.find(key);
        if (it_face == face_to_cells.end()) return;

        for (const int cell_index : it_face->second) {
            const int neighbour_color = has_cell_color ? r_data.mCellColor[cell_index] : 0;
            if (neighbour_color == cell_color) continue;

            for (const int outside_color : contact_container.mColorVector) {
                if (neighbour_color != outside_color) continue;

                KRATOS_CRITICAL_SECTION
                {
                    contact_container.mConditionsMap[outside_color].push_back(rCondition.Id());
                    for (const auto& r_node : r_geom) {
                        contact_container.mNodeMap[outside_color].push_back(r_node.Id());
                    }
                }
            }
        }
    });

    for (const int outside_color : contact_container.mColorVector) {
        ModelPart& r_contact_model_part = *(contact_container.mModelPartMap[outside_color]);
        r_contact_model_part.AddConditions(contact_container.mConditionsMap[outside_color]);
        r_contact_model_part.AddNodes(contact_container.mNodeMap[outside_color]);
    }

    KRATOS_CATCH("")
}

} // namespace Kratos
