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
 * @brief Builds the initial octree from a surface (first call) and selectively refines
 *        cells near it on subsequent calls.
 * @details **First call** (no octree in `OctreeHybridMesherData` yet):
 * 1. Extracts a triangle soup from the input surface model part.
 * 2. Builds the adaptive (or uniform) octree via `OctreeHybridMeshUtility::BuildFromSurfaceMesh`.
 * 3. Records `mesh_type`, `project_to_surface`, `projection_iterations`, and
 *    `projection_smoothing` into @ref Internals::OctreeHybridMesherData for use during
 *    the later balancing and extraction step.
 *
 * **Subsequent calls** (octree already built):
 * For each vertex in the triangle soup of the specified input model part, every cell
 * along the root-to-leaf path that contains that vertex is subdivided up to
 * `"refinement_depth"`.  Only cells in the vicinity of the input geometry are
 * subdivided; interior and far-field cells are left at their current level.
 *
 * When `"input_model_part_name"` is empty on a subsequent call, the triangle soup already
 * stored in `OctreeHybridMesherData::mTriangles` (the main input surface) is reused.
 * Multiple entries may be chained in `refine_operations_list`, each targeting a
 * **different model part**, to achieve feature-specific resolution:
 *
 * @code{.json}
 * "refine_operations_list": [
 *   { "type": "OctreeHybridRefineInterfaceCells",
 *     "refinement_depth": 3, "adaptive": true, "mesh_type": "dual" },
 *   { "type": "OctreeHybridRefineInterfaceCells",
 *     "input_model_part_name": "InnerWall", "refinement_depth": 6 }
 * ]
 * @endcode
 *
 * After all operations complete, the modeler calls `StrongConstrain2To1()` once to
 * produce a globally 2:1-balanced octree before mesh extraction.
 *
 * ### Parameters schema
 * | Key                      | Type   | Default | Description                         |
 * |--------------------------|--------|---------|-------------------------------------|
 * | `type`                   | string | `"OctreeHybridRefineInterfaceCells"` | Registry key. |
 * | `input_model_part_name`  | string | `""`    | Surface model part.  On first call, empty → falls back to the modeler's top-level `input_model_part_name`.  On subsequent calls, empty → reuse the already-extracted main triangle soup. |
 * | `refinement_depth`       | int    | `5`     | Build depth (first call) or maximum refinement depth for interface cells (subsequent calls).  Overridden by `element_size` > 0 on subsequent calls. |
 * | `element_size`           | double | `0.0`   | Desired maximum cell size (world-space) for subsequent calls.  When > 0, overrides `refinement_depth` via `OctreeHybridMeshUtility::ElementSizeToDepth`. |
 * | `adaptive`               | bool   | `true`  | First call only: adaptive subdivision near the surface; globally uniform when false. |
 * | `mesh_type`              | string | `"dual"` | First call only: `"dual"` (conforming all-hex) or `"primal"` (leaf-hex with hanging-node constraints). |
 * | `project_to_surface`     | bool   | `false` | First call only: project extracted dual mesh onto the iso-surface. |
 * | `projection_iterations`  | int    | `20000` | First call only: maximum projection iterations. |
 * | `projection_smoothing`   | int    | `1000`  | First call only: smoothing iterations in the projection step. |
 *
 * @see OctreeHybridRefineOperation
 * @see OctreeHybridRefineUniform
 * @see OctreeHybridMeshUtility::BuildFromSurfaceMesh
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
     * @brief Builds the octree (first call) or refines interface cells (subsequent calls).
     * @details On the first call (`rModeler.GetData().mpOctree == nullptr`): resolves the
     * surface model part (falling back to the modeler's top-level `input_model_part_name`
     * when the operation's own key is empty), extracts the triangle soup, builds the octree,
     * and writes `mesh_type`, `project_to_surface`, `projection_iterations`, and
     * `projection_smoothing` into @ref Internals::OctreeHybridMesherData.
     *
     * On subsequent calls: if `RefineParameters["input_model_part_name"]` is non-empty, the
     * triangle soup is extracted from that model part; otherwise the soup already stored in
     * `rModeler.GetData().mTriangles` is reused.  `OctreeHybridMeshUtility::RefineInterfaceCells`
     * is then called with the chosen soup and the target depth.
     *
     * @param rModeler         Reference to the owning @ref OctreeHybridMesherModeler.
     * @param RefineParameters Validated JSON parameters (see schema in the class Doxygen).
     */
    void Refine(OctreeHybridMesherModeler& rModeler, Parameters RefineParameters) const override;

    /**
     * @brief Returns the default parameter schema for this operation.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"                  : "OctreeHybridRefineInterfaceCells",
     *     "input_model_part_name" : "",
     *     "refinement_depth"      : 5,
     *     "element_size"          : 0.0,
     *     "adaptive"              : true,
     *     "mesh_type"             : "dual",
     *     "project_to_surface"    : false,
     *     "projection_iterations" : 20000,
     *     "projection_smoothing"  : 1000
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
