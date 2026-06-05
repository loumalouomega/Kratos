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

/**
 * @brief Returns the default parameter schema for @ref OctreeHybridGenerateBoundaryConditionsByFace.
 * @details The schema contains:
 * - `"type"` — the Registry lookup key, fixed to `"OctreeHybridGenerateBoundaryConditionsByFace"`.
 * - `"model_part_name"` — name of the ModelPart in which conditions are created.
 * - `"color"` — integer cell-colour label identifying inside cells (default `1`).
 * - `"properties_id"` — ID of the Properties object assigned to each new condition.
 * - `"generated_entity"` — registered Condition type name to instantiate per face.
 * @return Parameters with all accepted keys and their default values.
 */
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

/**
 * @brief Generates quadrilateral boundary conditions on the outer surface of the hex mesh.
 * @details Detailed algorithm:
 *
 * 1. **Prerequisite check** — raises an error if `mData.mCells` is empty, which means
 *    the hex-extraction step has not yet been run.
 *
 * 2. **ModelPart setup** — calls `OctreeHybridMesherModeler::CreateAndGetModelPart` to create
 *    (or retrieve) the target ModelPart, then initialises its entity-ID counters via
 *    `OctreeHybridMesherModeler::SetStartIds`.
 *
 * 3. **Cell filtering** — iterates `mData.mCells` and keeps only those cells whose
 *    entry in `mData.mCellColor` matches `want_color`.  If `mCellColor` is empty all
 *    cells pass the filter.  The connectivity of the filtered cells is copied into
 *    `active_cells` (indices reference the same `mData.mNodes` array as the full mesh).
 *
 * 4. **Boundary face extraction** — passes `active_cells` to
 *    `OctreeHybridMeshUtility::ExtractBoundaryFaces`, which returns every quad face
 *    (as a 4-element node-index array) that is shared by exactly one hex.  These faces
 *    form the closed exterior boundary of the carved mesh.
 *
 * 5. **Node and condition creation** — for each boundary quad:
 *    - The four corner node pointers are obtained via
 *      `OctreeHybridMesherModeler::GenerateOrRetrieveNode`.  This call creates a new ModelPart
 *      node the first time a mesh-node index is seen, or returns the existing pointer
 *      from `mData.mNodePtrs` on subsequent references (de-duplication).
 *    - Each pointer is pushed into `new_nodes` regardless of creation order, so nodes
 *      produced by prior generators (e.g. the hex generator) are also registered in the
 *      boundary ModelPart.
 *    - A new Condition is created from the registered prototype using
 *      `OctreeHybridMesherModeler::NextConditionId()` for the ID.
 *
 * 6. **Finalisation** — `new_nodes` is de-duplicated with `Unique()`, then added to the
 *    ModelPart along with all new conditions via `ModelPartUtils::AddNodesFromOrderedContainer`
 *    and `ModelPart::AddConditions`.
 *
 * @param rModeler              The owning @ref OctreeHybridMesherModeler.  Provides the Model,
 *                              `OctreeHybridMesherData`, the node de-duplication map, and the
 *                              entity-ID counters.
 * @param GenerationParameters  Validated JSON parameters; see @ref GetDefaultParameters
 *                              for the full schema.
 */
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
