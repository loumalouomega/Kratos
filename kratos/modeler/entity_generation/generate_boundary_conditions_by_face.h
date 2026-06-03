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
#include "modeler/entity_generation/octree_mesher_entity_generation.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class GenerateBoundaryConditionsByFace
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
 * `OctreeMesherModeler::GenerateOrRetrieveNode`, which de-duplicates via the modeler's
 * `mData.mNodePtrs` map.  Nodes that were already created during a prior hex-generation
 * step are pushed explicitly into the new ModelPart so that each boundary ModelPart
 * owns all of its nodes regardless of creation order.
 *
 * Works with both dual-mesh and primal-mesh topologies, and with any condition type
 * registered in the Kratos component database (default: `"SurfaceCondition3D4N"`).
 *
 * ### Prerequisite
 * The hex-generation step (e.g. @ref GenerateHexesByCellColor) must have run first so
 * that `OctreeMesherData::mCells` and `OctreeMesherData::mCellColor` are populated.
 *
 * ### Registry paths
 * Registered at:
 * - `OctreeMesherEntityGeneration.KratosMultiphysics`
 * - `OctreeMesherEntityGeneration.All`
 *
 * ### Parameters schema
 * | Key                | Type   | Default                    | Description                                |
 * |--------------------|--------|----------------------------|--------------------------------------------|
 * | `type`             | string | `"GenerateBoundaryConditionsByFace"` | Registry lookup key.             |
 * | `model_part_name`  | string | `"Undefined"`              | ModelPart to create conditions in.         |
 * | `color`            | int    | `1`                        | Cell color (inside label) to use.          |
 * | `properties_id`    | int    | `1`                        | Properties ID assigned to each condition.  |
 * | `generated_entity` | string | `"SurfaceCondition3D4N"`   | Registered condition type to instantiate.  |
 *
 * @note Outward winding of the boundary quads follows the convention of
 *       `OctreeHybridMeshUtility::ExtractBoundaryFaces`.
 * @see OctreeMesherEntityGeneration
 * @see OctreeHybridMeshUtility::ExtractBoundaryFaces
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateBoundaryConditionsByFace : public OctreeMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    GenerateBoundaryConditionsByFace() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy in this class.
     */
    GenerateBoundaryConditionsByFace(GenerateBoundaryConditionsByFace const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Generates quadrilateral boundary conditions on the exterior faces of the
     *        hex mesh for the configured cell color.
     * @details The method:
     * 1. Checks that the hex mesh has already been extracted (`mData.IsExtracted()`).
     * 2. Creates (or retrieves) the target ModelPart and initialises its ID counters.
     * 3. Filters `mData.mCells` to retain only cells with the requested `color`.
     * 4. Calls `OctreeHybridMeshUtility::ExtractBoundaryFaces` on the filtered cell
     *    list to obtain the set of boundary quads (each face owned by exactly one hex).
     * 5. For each boundary quad, retrieves or creates the four corner nodes via
     *    `OctreeMesherModeler::GenerateOrRetrieveNode` and pushes them into a local
     *    nodes container (ensuring nodes created by prior steps are also added).
     * 6. Creates the condition using the registered prototype identified by
     *    `generated_entity`.
     * 7. De-duplicates the node list and adds all nodes and conditions to the ModelPart.
     *
     * @param rModeler              The owning @ref OctreeMesherModeler; provides access
     *                              to the Model, mesh data, and ID generators.
     * @param GenerationParameters  Validated JSON parameters (see class-level schema).
     */
    void Generate(OctreeMesherModeler& rModeler, Parameters GenerationParameters) const override;

    /**
     * @brief Returns the default parameter schema for this generator.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"             : "GenerateBoundaryConditionsByFace",
     *     "model_part_name"  : "Undefined",
     *     "color"            : 1,
     *     "properties_id"    : 1,
     *     "generated_entity" : "SurfaceCondition3D4N"
     * }
     * @endcode
     * @return A Parameters object with all accepted keys and their default values.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class as a prototype under the KratosMultiphysics sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.KratosMultiphysics", OctreeMesherEntityGeneration, GenerateBoundaryConditionsByFace)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.All", OctreeMesherEntityGeneration, GenerateBoundaryConditionsByFace)

    ///@}
};

///@}

} // namespace Kratos
