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
#include "modeler/refine_operations/refine_hybrid_octree.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridRefineInterfaceCells
 * @ingroup KratosCore
 * @brief Refinement operation that selectively subdivides octree cells near a surface geometry.
 * @details For each vertex in the triangle soup of the specified input model part this
 * operation refines every cell along the root-to-leaf path that contains that vertex,
 * up to `"refinement_depth"`.  Only cells in the vicinity of the input geometry are
 * subdivided; interior and far-field cells are left at their current level, keeping
 * the octree graded without a globally fine resolution.
 *
 * The surface geometry is taken from the model part named by `"input_model_part_name"`.
 * When that key is empty the modeler's main input surface (already extracted as
 * `OctreeHybridMesherData::mTriangles`) is reused, avoiding a second triangle extraction.
 *
 * Multiple `OctreeHybridRefineInterfaceCells` entries may be chained in
 * `refine_operations_list`, each targeting a **different model part**, to achieve
 * feature-specific resolution without redundant passes:
 *
 * @code{.json}
 * "refine_operations_list": [
 *   { "type": "OctreeHybridRefineInterfaceCells",
 *     "input_model_part_name": "InnerWall", "refinement_depth": 6 },
 *   { "type": "OctreeHybridRefineInterfaceCells",
 *     "input_model_part_name": "OuterWall", "refinement_depth": 4 }
 * ]
 * @endcode
 *
 * After all refinement operations in `refine_operations_list` complete, the modeler
 * calls `StrongConstrain2To1()` once to produce the globally 2:1-balanced octree
 * before mesh extraction.
 *
 * ### Parameters schema
 * | Key                      | Type   | Default | Description                         |
 * |--------------------------|--------|---------|-------------------------------------|
 * | `type`                   | string | `"OctreeHybridRefineInterfaceCells"` | Registry key. |
 * | `refinement_depth`       | int    | `5`     | Maximum depth for interface cells.  Used when `element_size` is 0. |
 * | `element_size`           | double | `0.0`   | Desired maximum cell size in world-space units.  When > 0, overrides `refinement_depth` and the equivalent depth is computed via `OctreeHybridMeshUtility::ElementSizeToDepth`. |
 * | `input_model_part_name`  | string | `""`    | Model part whose surface triangles drive refinement.  Empty → use the modeler's main input surface. |
 *
 * @see OctreeHybridRefineOperation
 * @see OctreeHybridRefineUniform
 * @see OctreeHybridMeshUtility::RefineInterfaceCells
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridRefineInterfaceCells : public OctreeHybridRefineOperation
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridRefineInterfaceCells() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy.
     */
    OctreeHybridRefineInterfaceCells(OctreeHybridRefineInterfaceCells const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Refines octree cells near the specified surface to the requested depth.
     * @details If `RefineParameters["input_model_part_name"]` is non-empty, the triangle
     * soup is extracted from that model part via
     * `OctreeHybridMeshUtility::ExtractTriangleSoup`; otherwise the triangle soup already
     * stored in `rModeler.GetData().mTriangles` (the main input surface) is used.
     * `OctreeHybridMeshUtility::RefineInterfaceCells` is then called with the chosen soup
     * and the target depth.
     *
     * @param rModeler         Reference to the owning @ref OctreeHybridMesherModeler.
     * @param RefineParameters Validated JSON parameters containing `"refinement_depth"`
     *                         and optionally `"input_model_part_name"`.
     */
    void Refine(OctreeHybridMesherModeler& rModeler, Parameters RefineParameters) const override;

    /**
     * @brief Returns the default parameter schema for this operation.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"                  : "OctreeHybridRefineInterfaceCells",
     *     "refinement_depth"      : 5,
     *     "element_size"          : 0.0,
     *     "input_model_part_name" : ""
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
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridRefineOperation.KratosMultiphysics", OctreeHybridRefineOperation, OctreeHybridRefineInterfaceCells)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridRefineOperation.All", OctreeHybridRefineOperation, OctreeHybridRefineInterfaceCells)

    ///@}
};

///@}

} // namespace Kratos
