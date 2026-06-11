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
 * @class GenerateOctreeHybridConstraintsBetweenColors
 * @ingroup KratosCore
 * @brief Entity-generation stage that ties together two non-conforming colour blocks
 *        of the hybrid octree mesh through master-slave constraints.
 * @details `OctreeHybridMeshGeneratorModeler::GenerateOrRetrieveNode` de-duplicates nodes
 * through a single shared `OctreeHybridMesherData::mNodePtrs` map keyed by mesh-node
 * index, so two cells of different colours that share a mesh-node index already share
 * the very same `Node`. There is nothing to tie in that (conforming) case.
 *
 * This stage instead targets **non-conforming** interfaces: pairs of mesh-node indices
 * that belong to different colour blocks (`"color_block_a"` / `"color_block_b"`), are
 * **not** shared (different indices), but are **geometrically coincident** (same
 * `OctreeHybridMesherData::mNodes` coordinates within a fixed tolerance). For every such
 * pair, one `"generated_entity"` constraint per `"constrained_variables"` entry is
 * created, with the `"color_block_a"` node as master and the `"color_block_b"` node as
 * slave (weight `1.0`, constant `0.0`, i.e. `u_slave = u_master`).
 *
 * Mesh-node indices shared by both blocks (the conforming case) and indices belonging to
 * only one of the two blocks are ignored — they produce zero constraints.
 *
 * ### Optional region filtering
 * `"color"` (default `-1` = no filter): when not `-1`, both the master and the slave
 * candidate nodes of a pair must additionally be corners of at least one cell coloured
 * `"color"` (e.g. to restrict matching to a known interface region).
 *
 * ### Prerequisite
 * `OctreeHybridMesherData::mCellColor` must be populated (a colouring stage must have
 * run) so that `"color_block_a"` / `"color_block_b"` / `"color"` can be resolved.
 *
 * ### Registry paths
 * Registered at:
 * - `OctreeHybridMesherEntityGeneration.KratosMultiphysics`
 * - `OctreeHybridMesherEntityGeneration.All`
 *
 * ### Typical JSON configuration
 * @code{.json}
 * {
 *     "type"                  : "GenerateOctreeHybridConstraintsBetweenColors",
 *     "model_part_name"       : "Interface",
 *     "color_block_a"         : 1,
 *     "color_block_b"         : 2,
 *     "generated_entity"      : "LinearMasterSlaveConstraint",
 *     "constrained_variables" : ["TEMPERATURE"]
 * }
 * @endcode
 *
 * @see OctreeHybridMesherEntityGeneration
 * @see GenerateOctreeHybridConstraints
 * @see OctreeHybridMeshGeneratorModeler::GenerateOrRetrieveNode
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateOctreeHybridConstraintsBetweenColors : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    GenerateOctreeHybridConstraintsBetweenColors() = default;

    /**
     * @brief Copy constructor (no-op body — this class carries no state).
     * @param rOther Source instance; no data is copied.
     */
    GenerateOctreeHybridConstraintsBetweenColors(GenerateOctreeHybridConstraintsBetweenColors const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Generates master-slave constraints between geometrically coincident,
     *        non-conforming nodes of two colour blocks.
     * @details Algorithm:
     *
     *  1. **Membership sets** — for every cell `c`, the indices of its 8 corner nodes
     *     are added to `nodes_a` if `mCellColor[c] == "color_block_a"`, and to
     *     `nodes_b` if `mCellColor[c] == "color_block_b"`. If `"color"` is not `-1`,
     *     a third set `nodes_region` collects corners of cells coloured `"color"`.
     *  2. **Candidate filtering** — `nodes_a_only = nodes_a \ nodes_b` and
     *     `nodes_b_only = nodes_b \ nodes_a` (indices in both sets are already shared
     *     `Node`s and are dropped). When `"color"` is not `-1`, both sets are further
     *     intersected with `nodes_region`.
     *  3. **Geometric matching** — `nodes_a_only` and `nodes_b_only` indices are bucketed
     *     by their (rounded) `mNodes` coordinates. For every coincident pair
     *     `(ia, ib)` with `ia in nodes_a_only`, `ib in nodes_b_only`, and
     *     `||mNodes[ia] - mNodes[ib]|| < tolerance`, the corresponding nodes are
     *     retrieved (or created) via `GenerateOrRetrieveNode`.
     *  4. **Constraint creation** — for each variable in `"constrained_variables"`,
     *     registers the DOF on both nodes (`Node::AddDof`) and creates one
     *     `"generated_entity"` constraint with `ia`'s node as master, `ib`'s node as
     *     slave, weight `1.0` and constant `0.0`.
     *  5. **Finalisation** — any newly created nodes are bulk-inserted into the target
     *     ModelPart via `ModelPartUtils::AddNodesFromOrderedContainer`.
     *
     * @param rModeler              The owning modeler. Provides `OctreeHybridMesherData`
     *                              (`mCells`, `mCellColor`, `mNodes`, `mNodePtrs`) and the
     *                              helper methods `CreateAndGetModelPart`, `SetStartIds`,
     *                              `OverrideStartNodeId`, `OverrideStartConstraintId`,
     *                              `GenerateOrRetrieveNode`, and `NextConstraintId`.
     * @param GenerationParameters  Validated JSON parameters (see @ref GetDefaultParameters
     *                              for the full schema).
     *
     * @note `"generated_entity"` has no default and must be set explicitly.
     * @note If `mCellColor` is empty or `"constrained_variables"` is empty, the method
     *       logs (when `echo_level > 0`) and returns without creating any constraint.
     */
    void Generate(OctreeHybridMeshGeneratorModeler& rModeler, Parameters GenerationParameters) const override;

    /**
     * @brief Returns the default JSON parameter schema for @ref GenerateOctreeHybridConstraintsBetweenColors.
     * @details The schema defines all configuration keys accepted by the @ref Generate
     *          method with their default values:
     *
     *   | Key                    | Type    | Default                                         | Description |
     *   |------------------------|---------|-------------------------------------------------|-------------|
     *   | `"type"`               | string  | `"GenerateOctreeHybridConstraintsBetweenColors"` | Registry type token. |
     *   | `"model_part_name"`    | string  | `"Undefined"`                                   | Target ModelPart; created if absent. |
     *   | `"color"`              | int     | `-1`                                             | Restrict matching to nodes that are also corners of cells with this colour (`-1` = no filter). |
     *   | `"color_block_a"`      | int     | `-1`                                             | Colour of the master-side block. |
     *   | `"color_block_b"`      | int     | `-1`                                             | Colour of the slave-side block. |
     *   | `"generated_entity"`   | string  | `""`                                             | Registered MasterSlaveConstraint type name (required, no default). |
     *   | `"constrained_variables"` | array | `[]`                                           | Scalar DOF variable names to constrain across the interface. |
     *   | `"initial_node_id"`    | int     | `0`                                              | Explicit first node ID; `0` = auto. |
     *   | `"initial_constraint_id"` | int  | `0`                                              | Explicit first constraint ID; `0` = auto. |
     *   | `"echo_level"`         | int     | `0`                                              | Verbosity of `KRATOS_INFO` logging (`0` = silent). |
     *
     * @return Parameters object with all keys set to their defaults.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class at path "OctreeHybridMesherEntityGeneration.KratosMultiphysics.GenerateOctreeHybridConstraintsBetweenColors.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics", OctreeHybridMesherEntityGeneration, GenerateOctreeHybridConstraintsBetweenColors)
    /// Registers this class at path "OctreeHybridMesherEntityGeneration.All.GenerateOctreeHybridConstraintsBetweenColors.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All", OctreeHybridMesherEntityGeneration, GenerateOctreeHybridConstraintsBetweenColors)

    ///@}
};

///@}

} // namespace Kratos
