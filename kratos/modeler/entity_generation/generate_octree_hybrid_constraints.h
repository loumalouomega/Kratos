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
 * @class GenerateOctreeHybridConstraints
 * @ingroup KratosCore
 * @brief Standalone entity-generation stage that creates the 2:1 transition
 *        hanging-node master-slave constraints of the primal hybrid octree mesh.
 * @details This stage extracts the hanging-node-constraint logic that the per-entity
 * generators (e.g. `GenerateHybridOctreeHexahedraElementsWithCellColor`) optionally run
 * inline, into its own pipeline entry. For every record in
 * `OctreeHybridMesherData::mHanging` (populated only when `mMeshType == "primal"`), it
 * creates one `"generated_entity"` constraint per (master node x constrained variable)
 * pair, with the interpolation weight taken from `HangingConstraint::Weights`.
 *
 * ### Optional colour filtering
 * - `"color"` (default `-1` = no filter): when `OctreeHybridMesherData::mCellColor` is
 *   non-empty and `"color"` is not `-1`, only hanging records whose slave node is a
 *   corner of at least one cell with that colour are processed.
 * - `"face_color"` (default `-1` = no filter): when not `-1`, the boundary of the
 *   cells matching that colour (or all cells if `mCellColor` is empty) is computed via
 *   `OctreeHybridMeshUtility::ExtractBoundaryFaces`, and only hanging records whose
 *   slave node lies on one of those boundary faces are processed.
 *
 * Both filters apply to the slave node index of each `HangingConstraint` and combine
 * with logical AND.
 *
 * ### Prerequisite
 * The hex-generation step(s) must have already created the slave and master nodes in
 * the target ModelPart (or another ModelPart sharing `OctreeHybridMesherData::mNodePtrs`).
 * Hanging records whose slave or any master node has not yet been materialised
 * (`mNodePtrs` entry is null) are skipped silently.
 *
 * ### Registry paths
 * Registered at:
 * - `OctreeHybridMesherEntityGeneration.KratosMultiphysics`
 * - `OctreeHybridMesherEntityGeneration.All`
 *
 * ### Typical JSON configuration
 * @code{.json}
 * {
 *     "type"                  : "GenerateOctreeHybridConstraints",
 *     "model_part_name"       : "StructureDomain",
 *     "color"                 : 1,
 *     "face_color"            : -1,
 *     "generated_entity"      : "LinearMasterSlaveConstraint",
 *     "constrained_variables" : ["TEMPERATURE"]
 * }
 * @endcode
 *
 * @see OctreeHybridMesherEntityGeneration
 * @see GenerateHybridOctreeHexahedraElementsWithCellColor
 * @see OctreeHybridMesherData::mHanging
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateOctreeHybridConstraints : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    GenerateOctreeHybridConstraints() = default;

    /**
     * @brief Copy constructor (no-op body — this class carries no state).
     * @param rOther Source instance; no data is copied.
     */
    GenerateOctreeHybridConstraints(GenerateOctreeHybridConstraints const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Generates the 2:1 transition hanging-node constraints of the primal mesh.
     * @details For every entry `hc` in `rModeler.GetData().mHanging`:
     *
     *  1. **Colour filtering** — if `"color"` is not `-1` and `mCellColor` is non-empty,
     *     `hc.SlaveNode` must be a corner of at least one cell coloured `"color"`.
     *     If `"face_color"` is not `-1`, `hc.SlaveNode` must additionally lie on a
     *     boundary face (per `OctreeHybridMeshUtility::ExtractBoundaryFaces`) of the
     *     cells coloured `"face_color"`.
     *  2. **Existence check** — `hc.SlaveNode` and every entry of `hc.MasterNodes` must
     *     already have a non-null pointer in `mNodePtrs`; otherwise the record is
     *     skipped.
     *  3. **Constraint creation** — for each variable in `"constrained_variables"` and
     *     each master `m`, registers `hc.SlaveNode`'s and master `m`'s DOF for that
     *     variable (via `Node::AddDof`, a no-op if already present) and creates one
     *     `"generated_entity"` constraint via `ModelPart::CreateNewMasterSlaveConstraint`
     *     with weight `hc.Weights[m]` and constant `0.0`.
     *
     * @param rModeler              The owning modeler. Provides `OctreeHybridMesherData`
     *                              (`mHanging`, `mCellColor`, `mCells`, `mNodePtrs`) and
     *                              the helper methods `CreateAndGetModelPart`,
     *                              `SetStartIds`, `OverrideStartNodeId`,
     *                              `OverrideStartConstraintId`, and `NextConstraintId`.
     * @param GenerationParameters  Validated JSON parameters (see @ref GetDefaultParameters
     *                              for the full schema).
     *
     * @note If `mHanging` is empty (dual mesh, or no 2:1 transitions) or
     *       `"constrained_variables"` is empty, the method logs (when `echo_level > 0`)
     *       and returns without creating any constraint.
     */
    void Generate(OctreeHybridMeshGeneratorModeler& rModeler, Parameters GenerationParameters) const override;

    /**
     * @brief Returns the default JSON parameter schema for @ref GenerateOctreeHybridConstraints.
     * @details The schema defines all configuration keys accepted by the @ref Generate
     *          method with their default values:
     *
     *   | Key                    | Type    | Default                          | Description |
     *   |------------------------|---------|----------------------------------|-------------|
     *   | `"type"`               | string  | `"GenerateOctreeHybridConstraints"` | Registry type token. |
     *   | `"model_part_name"`    | string  | `"Undefined"`                    | Target ModelPart; created if absent. |
     *   | `"color"`              | int     | `-1`                              | Restrict to hanging slaves belonging to cells with this colour (`-1` = no filter). |
     *   | `"face_color"`         | int     | `-1`                              | Restrict to hanging slaves on the boundary of cells with this colour (`-1` = no filter). |
     *   | `"generated_entity"`   | string  | `"LinearMasterSlaveConstraint"`  | Registered MasterSlaveConstraint type name. |
     *   | `"constrained_variables"` | array | `["TEMPERATURE"]`              | Scalar DOF variable names to constrain at 2:1 transitions. |
     *   | `"initial_node_id"`    | int     | `0`                               | Explicit first node ID; `0` = auto. |
     *   | `"initial_constraint_id"` | int  | `0`                               | Explicit first constraint ID; `0` = auto. |
     *   | `"echo_level"`         | int     | `0`                               | Verbosity of `KRATOS_INFO` logging (`0` = silent). |
     *
     * @return Parameters object with all keys set to their defaults.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class at path "OctreeHybridMesherEntityGeneration.KratosMultiphysics.GenerateOctreeHybridConstraints.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics", OctreeHybridMesherEntityGeneration, GenerateOctreeHybridConstraints)
    /// Registers this class at path "OctreeHybridMesherEntityGeneration.All.GenerateOctreeHybridConstraints.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All", OctreeHybridMesherEntityGeneration, GenerateOctreeHybridConstraints)

    ///@}
};

///@}

} // namespace Kratos
