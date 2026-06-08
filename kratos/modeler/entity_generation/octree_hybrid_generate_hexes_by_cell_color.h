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

#pragma once

// System includes

// External includes

// Project includes
#include "modeler/entity_generation/octree_hybrid_mesher_entity_generation.h"

namespace Kratos 
{

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridGenerateHexesByCellColor
 * @ingroup KratosCore
 * @brief Entity-generation stage that emits one hexahedral element per cell whose
 *        colour matches the requested value.
 * @details For every entry `c` in `OctreeHybridMesherData::mCells` whose associated colour
 *          `OctreeHybridMesherData::mCellColor[c]` equals the parameter `"color"`, the stage:
 *
 *          1. De-duplicates the eight corner nodes by calling
 *             `OctreeHybridMesherModeler::GenerateOrRetrieveNode` — a node is created in the
 *             ModelPart only on its first encounter; subsequent cells sharing the same
 *             mesh-node index reuse the cached `Node::Pointer` stored in
 *             `OctreeHybridMesherData::mNodePtrs`.
 *          2. Creates a new element via the prototype returned by
 *             `KratosComponents<Element>::Get("generated_entity")` (default
 *             `"Element3D8N"`).
 *          3. Optionally tags the element with the octree refinement level
 *             `OctreeHybridMesherData::mCellLevel[c]` under the variable `REFINEMENT_LEVEL`
 *             when the parameter `"tag_refinement_level"` is `true` (default).
 *          4. Adds the new nodes and elements to the target ModelPart in bulk.
 *
 *          ### Node-ordering compatibility
 *          The hybrid octree engine already stores corner indices in the order that
 *          matches the Kratos `Hexahedra3D8` local-node numbering, so corner index
 *          `k` maps directly to local node `k` without any remapping.
 *
 *          ### Dual vs. primal mesh — optional hanging-node constraints
 *          The stage is topology-agnostic for element generation: it operates on the flat
 *          cell arrays (`mCells`, `mNodes`, `mCellColor`) regardless of mesh type.
 *          For **primal** meshes, when the `"variables"` array is non-empty and
 *          `OctreeHybridMesherData::mHanging` contains 2:1 transition records, the stage
 *          additionally creates one `LinearMasterSlaveConstraint` per
 *          (hanging node × master node × variable) triple in the same ModelPart —
 *          eliminating the need for a separate `OctreeHybridGenerateHangingNodeConstraints`
 *          entry in `entities_generator_list`.
 *
 * ### Typical JSON configuration (dual mesh)
 * @code{.json}
 * {
 *     "type"                : "OctreeHybridGenerateHexesByCellColor",
 *     "model_part_name"     : "FluidDomain",
 *     "color"               : 1,
 *     "properties_id"       : 1,
 *     "generated_entity"    : "Element3D8N",
 *     "tag_refinement_level": true
 * }
 * @endcode
 *
 * ### Primal mesh with hanging-node constraints
 * @code{.json}
 * {
 *     "type"                : "OctreeHybridGenerateHexesByCellColor",
 *     "model_part_name"     : "StructureDomain",
 *     "color"               : 1,
 *     "variables"           : ["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"]
 * }
 * @endcode
 *
 * @note The ModelPart is created via `OctreeHybridMesherModeler::CreateAndGetModelPart` if it
 *       does not exist yet.  Properties with `"properties_id"` are likewise created on
 *       demand.
 *
 * @see OctreeHybridMesherEntityGeneration
 * @see OctreeHybridMesherModeler::GenerateOrRetrieveNode
 * @see OctreeHybridClassifyCellsInsideOutside
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridGenerateHexesByCellColor : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridGenerateHexesByCellColor() = default;

    /**
     * @brief Copy constructor (no-op body — this class carries no state).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridGenerateHexesByCellColor(OctreeHybridGenerateHexesByCellColor const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

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
     *           4. **Bulk insertion** — after all cells have been processed, duplicate node
     *              pointers are removed via `ModelPart::NodesContainerType::Unique`, nodes are
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
     *        guarded by `!mCellColor.empty()`.
     * @note Properties with the given `properties_id` are retrieved from the ModelPart if
     *       they already exist, or created fresh otherwise.
     *
     * @see OctreeHybridMesherModeler::GenerateOrRetrieveNode
     * @see OctreeHybridMesherData::mCells
     * @see OctreeHybridMesherData::mCellColor
     * @see OctreeHybridMesherData::mCellLevel
     * @see ModelPartUtils::AddNodesFromOrderedContainer
     */
    void Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const override;

    /**
     * @brief Returns the default JSON parameter schema for @ref OctreeHybridGenerateHexesByCellColor.
     * @details The schema defines all configuration keys accepted by the @ref Generate
     *          method with their default values:
     *
     *   | Key                    | Type    | Default              | Description |
     *   |------------------------|---------|----------------------|-------------|
     *   | `"type"`               | string  | `"OctreeHybridGenerateHexesByCellColor"` | Registry type token. |
     *   | `"model_part_name"`    | string  | `"Undefined"`        | Target ModelPart; created if absent. |
     *   | `"color"`              | int     | `1`                  | Cell-colour value to select (1 == inside). |
     *   | `"properties_id"`      | int     | `1`                  | Properties block ID for generated elements. |
     *   | `"generated_entity"`   | string  | `"Element3D8N"`      | Registered Element type name. |
     *   | `"tag_refinement_level"` | bool  | `true`               | Store `REFINEMENT_LEVEL` on each element. |
     *   | `"constraint_name"`    | string  | `"LinearMasterSlaveConstraint"` | Constraint type for hanging-node MPC; used only when `"variables"` is non-empty. |
     *   | `"variables"`          | array   | `[]`                 | Scalar DOF variable names to constrain at 2:1 transitions.  Empty (default) = no constraints generated. |
     *
     * @return Parameters object with all keys set to their defaults.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class at path "OctreeHybridMesherEntityGeneration.KratosMultiphysics.OctreeHybridGenerateHexesByCellColor.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics", OctreeHybridMesherEntityGeneration, OctreeHybridGenerateHexesByCellColor)
    /// Registers this class at path "OctreeHybridMesherEntityGeneration.All.OctreeHybridGenerateHexesByCellColor.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All", OctreeHybridMesherEntityGeneration, OctreeHybridGenerateHexesByCellColor)

    ///@}
};

///@}

} // namespace Kratos
