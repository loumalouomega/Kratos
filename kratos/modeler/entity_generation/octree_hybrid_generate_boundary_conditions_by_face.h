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
 * @class OctreeHybridGenerateBoundaryConditionsByFace
 * @ingroup KratosCore
 * @brief Entity-generation component that creates quadrilateral boundary conditions on
 *        the exterior faces of the coloured hex mesh.
 * @details This generator extracts the outer boundary of all hex cells that have been
 * assigned the configured `color` (typically `1` for inside-cells) and creates one
 * condition per boundary quad.  A boundary quad is a quad face that is owned by exactly
 * one hex, i.e. it lies on the outer closed surface of the carved mesh.  This is
 * accomplished by calling `OctreeHybridMeshUtility::ExtractBoundaryFaces` on the
 * colour-filtered cell list.
 *
 * Nodes for the conditions are retrieved — or created if not yet present — through
 * `OctreeHybridMesherModeler::GenerateOrRetrieveNode`, which de-duplicates via the modeler's
 * `mData.mNodePtrs` map.  Nodes that were already created during a prior hex-generation
 * step are pushed explicitly into the new ModelPart so that each boundary ModelPart
 * owns all of its nodes regardless of creation order.
 *
 * Works with both dual-mesh and primal-mesh topologies, and with any condition type
 * registered in the Kratos component database (default: `"SurfaceCondition3D4N"`).
 *
 * ### Optional hanging-node constraints (primal mesh)
 * When `"variables"` is non-empty and `OctreeHybridMesherData::mHanging` contains 2:1
 * transition records (set by the primal extraction path), the stage additionally creates
 * one `LinearMasterSlaveConstraint` per (hanging node × master node × variable) triple
 * in the same ModelPart — ensuring that boundary DOFs at refinement transitions are
 * properly constrained without a separate entity-generation entry.  This mirrors the
 * behaviour of @ref OctreeHybridGenerateHexesByCellColor.
 *
 * ### Prerequisite
 * The hex-generation step (e.g. @ref OctreeHybridGenerateHexesByCellColor) must have run first so
 * that `OctreeHybridMesherData::mCells` and `OctreeHybridMesherData::mCellColor` are populated.
 *
 * ### Registry paths
 * Registered at:
 * - `OctreeHybridMesherEntityGeneration.KratosMultiphysics`
 * - `OctreeHybridMesherEntityGeneration.All`
 *
 * ### Parameters schema
 * | Key                  | Type         | Default                    | Description                                |
 * |----------------------|--------------|----------------------------|--------------------------------------------|
 * | `type`               | string       | `"OctreeHybridGenerateBoundaryConditionsByFace"` | Registry lookup key.         |
 * | `model_part_name`    | string       | `"Undefined"`              | ModelPart to create conditions in.         |
 * | `color`              | int          | `1`                        | Cell color (inside label) to use.          |
 * | `properties_id`      | int          | `1`                        | Properties ID assigned to each condition.  |
 * | `generated_entity`   | string       | `"SurfaceCondition3D4N"`   | Registered condition type to instantiate.  |
 * | `constraint_name`    | string       | `"LinearMasterSlaveConstraint"` | Constraint type for hanging-node MPC; used only when `"variables"` is non-empty. |
 * | `variables`          | string array | `[]`                       | Scalar DOF variable names to constrain at 2:1 transitions.  Empty (default) = no constraints generated. |
 *
 * @note Outward winding of the boundary quads follows the convention of
 *       `OctreeHybridMeshUtility::ExtractBoundaryFaces`.
 * @see OctreeHybridMesherEntityGeneration
 * @see OctreeHybridMeshUtility::ExtractBoundaryFaces
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridGenerateBoundaryConditionsByFace : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridGenerateBoundaryConditionsByFace() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy in this class.
     */
    OctreeHybridGenerateBoundaryConditionsByFace(OctreeHybridGenerateBoundaryConditionsByFace const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

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
    void Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const override;

    /**
     * @brief Returns the default parameter schema for this generator.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"              : "OctreeHybridGenerateBoundaryConditionsByFace",
     *     "model_part_name"   : "Undefined",
     *     "color"             : 1,
     *     "properties_id"     : 1,
     *     "generated_entity"  : "SurfaceCondition3D4N",
     *     "constraint_name"   : "LinearMasterSlaveConstraint",
     *     "variables"         : []
     * }
     * @endcode
     * The `"variables"` array lists scalar DOF names (e.g. `"PRESSURE"`) for which
     * a `LinearMasterSlaveConstraint` is generated at every 2:1 hanging node on the
     * boundary.  Leave empty (default) when no constraints are needed.
     * @return A Parameters object with all accepted keys and their default values.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class as a prototype under the KratosMultiphysics sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics", OctreeHybridMesherEntityGeneration, OctreeHybridGenerateBoundaryConditionsByFace)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All", OctreeHybridMesherEntityGeneration, OctreeHybridGenerateBoundaryConditionsByFace)

    ///@}
};

///@}

} // namespace Kratos
