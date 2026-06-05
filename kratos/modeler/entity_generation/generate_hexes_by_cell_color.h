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

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class GenerateHexesByCellColor
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
 *          ### Dual vs. primal mesh
 *          The stage is topology-agnostic: it operates on the flat cell arrays
 *          (`mCells`, `mNodes`, `mCellColor`) regardless of whether the mesh was
 *          extracted as a conforming dual mesh or as a primal leaf-hex mesh with
 *          hanging-node constraints.
 *
 * ### Typical JSON configuration
 * @code{.json}
 * {
 *     "type"                : "GenerateHexesByCellColor",
 *     "model_part_name"     : "FluidDomain",
 *     "color"               : 1,
 *     "properties_id"       : 1,
 *     "generated_entity"    : "Element3D8N",
 *     "tag_refinement_level": true
 * }
 * @endcode
 *
 * @note The ModelPart is created via `OctreeHybridMesherModeler::CreateAndGetModelPart` if it
 *       does not exist yet.  Properties with `"properties_id"` are likewise created on
 *       demand.
 *
 * @see OctreeHybridMesherEntityGeneration
 * @see OctreeHybridMesherModeler::GenerateOrRetrieveNode
 * @see ClassifyCellsInsideOutside
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateHexesByCellColor : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    GenerateHexesByCellColor() = default;

    /**
     * @brief Copy constructor (no-op body — this class carries no state).
     * @param rOther Source instance; no data is copied.
     */
    GenerateHexesByCellColor(GenerateHexesByCellColor const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Generates hexahedral elements for all cells matching the requested colour.
     * @details Iterates over `rModeler.GetData().mCells`, skips cells whose entry in
     *          `mCellColor` differs from the `"color"` parameter, and for matching cells
     *          constructs an element using the eight de-duplicated corner nodes.
     *
     *          Nodes are added to the ModelPart via
     *          `ModelPartUtils::AddNodesFromOrderedContainer` (bulk, sorted insertion) and
     *          elements via `ModelPart::AddElements`.  Entity IDs are consumed from
     *          `OctreeHybridMesherModeler::NextElementId`.
     *
     * @param rModeler              The owning modeler; nodes and elements are added to the
     *                              ModelPart whose name is given by
     *                              `GenerationParameters["model_part_name"]`.
     * @param GenerationParameters  Validated JSON parameters.  Expected keys:
     *   - `"model_part_name"` (`string`): target ModelPart (created if absent).
     *   - `"color"` (`int`): cell-colour value to select (e.g.\ 1 for inside).
     *   - `"properties_id"` (`int`): Properties block assigned to every new element.
     *   - `"generated_entity"` (`string`): registered Element type name.
     *   - `"tag_refinement_level"` (`bool`): whether to store `REFINEMENT_LEVEL` on each element.
     */
    void Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const override;

    /**
     * @brief Returns the default parameter schema for this entity-generation stage.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"                : "GenerateHexesByCellColor",
     *     "model_part_name"     : "Undefined",
     *     "color"               : 1,
     *     "properties_id"       : 1,
     *     "generated_entity"    : "Element3D8N",
     *     "tag_refinement_level": true
     * }
     * @endcode
     * @return Parameters object with all keys set to their defaults.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class at path "OctreeHybridMesherEntityGeneration.KratosMultiphysics.GenerateHexesByCellColor.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics", OctreeHybridMesherEntityGeneration, GenerateHexesByCellColor)
    /// Registers this class at path "OctreeHybridMesherEntityGeneration.All.GenerateHexesByCellColor.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All", OctreeHybridMesherEntityGeneration, GenerateHexesByCellColor)

    ///@}
};

///@}

} // namespace Kratos
