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

// Project includes
#include "includes/kratos_components.h"
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/generate_hybrid_octree_triangular_conditions_with_face_color.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos
{

const Parameters GenerateHybridOctreeTriangularConditionsWithFaceColor::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                  : "GenerateHybridOctreeTriangularConditionsWithFaceColor",
        "model_part_name"       : "Undefined",
        "color"                 : 1,
        "properties_id"         : 1,
        "generated_entity"      : "SurfaceCondition3D3N",
        "initial_node_id"       : 0,
        "initial_condition_id"  : 0,
        "echo_level"            : 0
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void GenerateHybridOctreeTriangularConditionsWithFaceColor::Generate(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters GenerationParameters) const
{
    // Build triangular boundary conditions from octree boundary faces of cells with the requested color.
    auto& r_data = rModeler.GetData();
    KRATOS_ERROR_IF(!r_data.IsExtracted())
        << "GenerateHybridOctreeTriangularConditionsWithFaceColor: hex mesh not yet extracted."
        << std::endl;

    // Create the output ModelPart
    ModelPart& r_model_part = rModeler.CreateAndGetModelPart(GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_model_part);
    rModeler.OverrideStartNodeId(GenerationParameters["initial_node_id"].GetInt());
    rModeler.OverrideStartConditionId(GenerationParameters["initial_condition_id"].GetInt());

    // Extract parameters.
    const int echo_level = GenerationParameters["echo_level"].GetInt();
    const int want_color = GenerationParameters["color"].GetInt();
    const std::size_t properties_id = GenerationParameters["properties_id"].GetInt();

    // Reuse an existing Properties container if present in the hierarchy, otherwise create it.
    Properties::Pointer p_props = r_model_part.RecursivelyHasProperties(properties_id)
        ? r_model_part.pGetProperties(properties_id)
        : r_model_part.CreateNewProperties(properties_id);

    // Validate the requested condition type exists in KratosComponents.
    const std::string entity_name = GenerationParameters["generated_entity"].GetString();
    KRATOS_ERROR_IF(!KratosComponents<Condition>::Has(entity_name))
        << "GenerateHybridOctreeTriangularConditionsWithFaceColor: condition type '"
        << entity_name << "' is not registered in KratosComponents." << std::endl;
    const Condition& r_proto = KratosComponents<Condition>::Get(entity_name);

    // Keep only cells matching the requested color before boundary extraction.
    std::vector<std::array<int, 8>> active_cells;
    active_cells.reserve(r_data.mCells.size());
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_color) continue;
        active_cells.push_back(r_data.mCells[c]);
    }

    // Extract only external quad faces from the selected hexahedral cells.
    const auto bfaces = OctreeHybridMeshUtility::ExtractBoundaryFaces(active_cells);

    ModelPart::NodesContainerType new_nodes;
    ModelPart::ConditionsContainerType new_conditions;
    Condition::NodesArrayType tri_nodes(3);

    // Each boundary face is a quad {n0, n1, n2, n3}.  Split into two triangles {n0, n1, n2} and {n0, n2, n3}.
    for (const auto& bf : bfaces) {
        // Split quad {n0, n1, n2, n3} along the (n0, n2) diagonal.
        // This diagonal is consistent with the (0,6)-main-diagonal Freudenthal
        // tet decomposition so boundary triangles are faces of interior tetrahedra.
        const int quad[4] = {bf[0], bf[1], bf[2], bf[3]};

        for (int split = 0; split < 2; ++split) {
            // Triangle 0: {n0, n1, n2}  — Triangle 1: {n0, n2, n3}
            const int idx[3] = {quad[0], quad[1 + split], quad[2 + split]};
            for (int v = 0; v < 3; ++v) {
                Node::Pointer p_node = rModeler.GenerateOrRetrieveNode(r_model_part, new_nodes, idx[v]);
                tri_nodes(v) = p_node;
                new_nodes.push_back(p_node);
            }
            new_conditions.push_back(r_proto.Create(rModeler.NextConditionId(), tri_nodes, p_props));
        }
    }

    // Remove duplicates introduced by shared face vertices before final insertion.
    new_nodes.Unique();
    const std::size_t n_new_nodes = new_nodes.size();
    ModelPartUtils::AddNodesFromOrderedContainer(r_model_part, new_nodes.begin(), new_nodes.end());
    r_model_part.AddConditions(new_conditions.begin(), new_conditions.end());

    KRATOS_INFO_IF("GenerateHybridOctreeTriangularConditionsWithFaceColor", echo_level > 0)
        << "Generated " << new_conditions.size() << " conditions and " << n_new_nodes
        << " nodes in ModelPart \"" << r_model_part.FullName() << "\"." << std::endl;
}

} // namespace Kratos
