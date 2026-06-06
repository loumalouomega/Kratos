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
#include "includes/variables.h"
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/octree_hybrid_generate_hexes_by_cell_color.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos 
{

const Parameters OctreeHybridGenerateHexesByCellColor::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                 : "OctreeHybridGenerateHexesByCellColor",
        "model_part_name"      : "Undefined",
        "color"                : 1,
        "properties_id"        : 1,
        "generated_entity"     : "Element3D8N",
        "tag_refinement_level" : true
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridGenerateHexesByCellColor::Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const
{
    auto& r_data = rModeler.GetData();
    ModelPart& r_model_part = rModeler.CreateAndGetModelPart(GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_model_part);

    const int want_color = GenerationParameters["color"].GetInt();
    const std::size_t properties_id = GenerationParameters["properties_id"].GetInt();
    Properties::Pointer p_properties = r_model_part.HasProperties(properties_id)
        ? r_model_part.pGetProperties(properties_id)
        : r_model_part.CreateNewProperties(properties_id);
    const Element& r_prototype_element = KratosComponents<Element>::Get(GenerationParameters["generated_entity"].GetString());

    const bool tag_level = GenerationParameters["tag_refinement_level"].GetBool();

    ModelPart::NodesContainerType new_nodes;
    ModelPart::ElementsContainerType new_elements;
    Element::NodesArrayType cell_nodes(8);

    // The engine corner ordering (CX/CY/CZ) already matches the Kratos
    // Hexahedra3D8 node order, so corner k maps directly to local node k.
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_color) continue;
        for (int k = 0; k < 8; ++k) {
            cell_nodes(k) = rModeler.GenerateOrRetrieveNode(r_model_part, new_nodes, r_data.mCells[c][k]);
        }
        auto p_el = r_prototype_element.Create(rModeler.NextElementId(), cell_nodes, p_properties);
        if (tag_level && !r_data.mCellLevel.empty())
            p_el->SetValue(REFINEMENT_LEVEL, r_data.mCellLevel[c]);
        new_elements.push_back(p_el);
    }

    new_nodes.Unique();
    ModelPartUtils::AddNodesFromOrderedContainer(r_model_part, new_nodes.begin(), new_nodes.end());
    r_model_part.AddElements(new_elements.begin(), new_elements.end());
}

} // namespace Kratos
