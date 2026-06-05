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
#include "includes/element.h"
#include "includes/kratos_components.h"
#include "includes/variables.h"
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/generate_hexes_by_cell_color.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos {

/**
 * @brief Returns the default JSON parameter schema for @ref GenerateHexesByCellColor.
 * @details The schema defines all configuration keys accepted by the @ref Generate
 *          method with their default values:
 *
 *   | Key                    | Type    | Default              | Description |
 *   |------------------------|---------|----------------------|-------------|
 *   | `"type"`               | string  | `"GenerateHexesByCellColor"` | Registry type token. |
 *   | `"model_part_name"`    | string  | `"Undefined"`        | Target ModelPart; created if absent. |
 *   | `"color"`              | int     | `1`                  | Cell-colour value to select (1 == inside). |
 *   | `"properties_id"`      | int     | `1`                  | Properties block ID for generated elements. |
 *   | `"generated_entity"`   | string  | `"Element3D8N"`      | Registered Element type name. |
 *   | `"tag_refinement_level"` | bool  | `true`               | Store `REFINEMENT_LEVEL` on each element. |
 *
 * @return Parameters object with all keys set to their defaults.
 */
const Parameters GenerateHexesByCellColor::GetDefaultParameters() const
{
    return Parameters(R"({
        "type" : "GenerateHexesByCellColor",
        "model_part_name" : "Undefined",
        "color" : 1,
        "properties_id" : 1,
        "generated_entity" : "Element3D8N",
        "tag_refinement_level" : true
    })");
}

/**
 * @brief Generates one hexahedral element per cell whose colour matches the requested value.
 * @details The method iterates over every cell in `rModeler.GetData().mCells` (indexed by `c`).
 *          Cells are skipped when:
 *            - `OctreeHybridMesherData::mCellColor` is non-empty **and**
 *              `mCellColor[c] != want_color`.
 *
 *          For each selected cell the following steps are performed:
 *
 *          1. **Node de-duplication** — `OctreeHybridMesherModeler::GenerateOrRetrieveNode` is
 *             called for each of the 8 corners (`mCells[c][k]`, k = 0…7).  On the first
 *             call for a given mesh-node index the node is created from the world-space
 *             coordinates in `OctreeHybridMesherData::mNodes` and cached in
 *             `OctreeHybridMesherData::mNodePtrs`; subsequent calls return the cached pointer.
 *             New nodes are accumulated in the local `new_nodes` container.
 *
 *          2. **Element creation** — `Element::Create(id, nodes, properties)` is invoked
 *             on the prototype element retrieved from the KratosComponents registry.  The
 *             element ID is produced by `OctreeHybridMesherModeler::NextElementId`.
 *
 *          3. **Refinement-level tagging** — when `tag_level` is `true` and
 *             `OctreeHybridMesherData::mCellLevel` is non-empty, the non-historical variable
 *             `REFINEMENT_LEVEL` is set on the element to `mCellLevel[c]`.
 *
 *          4. **Bulk insertion** — after all cells have been processed, duplicate node
 *             pointers are removed via `ModelPart::NodesContainerType::Unique`, nodes are
 *             added to the ModelPart with `ModelPartUtils::AddNodesFromOrderedContainer`
 *             (preserves sort order), and elements are added with
 *             `ModelPart::AddElements`.
 *
 *          ### Node-ordering note
 *          The hybrid octree engine stores corner indices in the Kratos `Hexahedra3D8`
 *          local-node order, so corner `k` maps directly to local node `k` — no
 *          permutation array is required.
 *
 * @param rModeler              The owning modeler.  Provides access to the shared
 *                              `OctreeHybridMesherData` (cells, nodes, colours, levels) and
 *                              the helper methods `CreateAndGetModelPart`,
 *                              `SetStartIds`, `GenerateOrRetrieveNode`, and
 *                              `NextElementId`.
 * @param GenerationParameters  Validated JSON parameters (see @ref GetDefaultParameters
 *                              for the full schema).  Expected keys consumed here:
 *   - `"model_part_name"` — target sub-ModelPart name.
 *   - `"color"` — integer colour value to match.
 *   - `"properties_id"` — Properties block ID; created on demand.
 *   - `"generated_entity"` — registered Element type name looked up from KratosComponents.
 *   - `"tag_refinement_level"` — whether to store `REFINEMENT_LEVEL` on each element.
 *
 * @note If `mCellColor` is empty (no colouring stage ran), **all** cells are emitted
 *       regardless of the requested colour, because the colour-filter condition is
 *       guarded by `!mCellColor.empty()`.
 * @note Properties with the given `properties_id` are retrieved from the ModelPart if
 *       they already exist, or created fresh otherwise.
 *
 * @see OctreeHybridMesherModeler::GenerateOrRetrieveNode
 * @see OctreeHybridMesherData::mCells
 * @see OctreeHybridMesherData::mCellColor
 * @see OctreeHybridMesherData::mCellLevel
 * @see ModelPartUtils::AddNodesFromOrderedContainer
 */
void GenerateHexesByCellColor::Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const
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
