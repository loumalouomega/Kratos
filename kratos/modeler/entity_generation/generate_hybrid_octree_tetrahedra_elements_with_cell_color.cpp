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
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/generate_hybrid_octree_tetrahedra_elements_with_cell_color.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos
{

const Parameters GenerateHybridOctreeTetrahedraElementsWithCellColor::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                 : "GenerateHybridOctreeTetrahedraElementsWithCellColor",
        "model_part_name"      : "Undefined",
        "color"                : 1,
        "properties_id"        : 1,
        "generated_entity"     : "Element3D4N",
        "tag_refinement_level" : true,
        "initial_node_id"      : 0,
        "initial_element_id"   : 0,
        "echo_level"           : 0
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void GenerateHybridOctreeTetrahedraElementsWithCellColor::Generate(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters GenerationParameters) const
{
    auto& r_data = rModeler.GetData();
    ModelPart& r_model_part = rModeler.CreateAndGetModelPart(
        GenerationParameters["model_part_name"].GetString());
    rModeler.OverrideStartNodeId(GenerationParameters["initial_node_id"].GetInt());
    rModeler.OverrideStartElementId(GenerationParameters["initial_element_id"].GetInt());

    const int echo_level = GenerationParameters["echo_level"].GetInt();

    const int want_color  = GenerationParameters["color"].GetInt();
    const std::size_t properties_id = GenerationParameters["properties_id"].GetInt();
    Properties::Pointer p_properties = r_model_part.RecursivelyHasProperties(properties_id)
        ? r_model_part.pGetProperties(properties_id)
        : r_model_part.CreateNewProperties(properties_id);

    const std::string entity_name = GenerationParameters["generated_entity"].GetString();
    KRATOS_ERROR_IF(!KratosComponents<Element>::Has(entity_name))
        << "GenerateHybridOctreeTetrahedraElementsWithCellColor: element type '"
        << entity_name << "' is not registered in KratosComponents." << std::endl;
    const Element& r_proto = KratosComponents<Element>::Get(entity_name);

    const bool tag_level = GenerationParameters["tag_refinement_level"].GetBool();

    // Freudenthal 6-tet decomposition of a Hexahedra3D8 along the (0,6) main diagonal.
    // See TACS1de1.pdf §3.3.3 for the BCC-lattice rationale.
    static constexpr int TET_CONN[6][4] = {
        {0, 3, 6, 2},
        {3, 6, 7, 0},
        {4, 7, 6, 0},
        {0, 4, 5, 6},
        {0, 1, 2, 6},
        {1, 5, 6, 0}
    };

    ModelPart::NodesContainerType new_nodes;
    ModelPart::ElementsContainerType new_elements;
    Element::NodesArrayType tet_nodes(4);

    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_color) continue;

        // Materialise all 8 hex corner nodes (deduplication via mNodePtrs cache).
        Node::Pointer hex_ptrs[8];
        for (int k = 0; k < 8; ++k)
            hex_ptrs[k] = rModeler.GenerateOrRetrieveNode(
                r_model_part, new_nodes, r_data.mCells[c][k]);

        const int cell_level = (!r_data.mCellLevel.empty()) ? r_data.mCellLevel[c] : -1;

        for (int t = 0; t < 6; ++t) {
            for (int v = 0; v < 4; ++v)
                tet_nodes(v) = hex_ptrs[TET_CONN[t][v]];
            auto p_el = r_proto.Create(rModeler.NextElementId(), tet_nodes, p_properties);
            if (tag_level && cell_level >= 0)
                p_el->SetValue(REFINEMENT_LEVEL, cell_level);
            new_elements.push_back(p_el);
        }
    }

    new_nodes.Unique();
    const std::size_t n_new_nodes = new_nodes.size();
    ModelPartUtils::AddNodesFromOrderedContainer(r_model_part, new_nodes.begin(), new_nodes.end());
    r_model_part.AddElements(new_elements.begin(), new_elements.end());

    KRATOS_INFO_IF("GenerateHybridOctreeTetrahedraElementsWithCellColor", echo_level > 0)
        << "Generated " << new_elements.size() << " elements and " << n_new_nodes
        << " nodes in ModelPart \"" << r_model_part.FullName() << "\"." << std::endl;
}

} // namespace Kratos
