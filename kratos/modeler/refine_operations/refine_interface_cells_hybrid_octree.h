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
 * @class RefineInterfaceCellsOctreeHybrid
 * @ingroup KratosCore
 * @brief Builds the initial octree from a surface (if not built yet) and selectively
 *        refines cells near it.
 * @details **If no octree exists yet** (`rModeler.GetData().mpOctree == nullptr`), this
 * operation delegates to `OctreeHybridMeshGeneratorModeler::EnsureOctreeBuilt`, which:
 * 1. Extracts a triangle soup from `"model_part_name"` (falling back to the modeler's
 *    top-level `input_model_part_name` when left empty).
 * 2. Builds the adaptive (or uniform) octree via `OctreeHybridMeshUtility::BuildFromSurfaceMesh`.
 * 3. Records `mesh_type`, `project_to_surface`, `projection_iterations`, and
 *    `projection_smoothing` into @ref Internals::OctreeHybridMesherData for use during
 *    the later balancing and extraction step.
 *
 * The operation then returns without performing any additional refinement on that call.
 *
 * **If an octree already exists**:
 * For each vertex in the triangle soup of the specified input model part, every cell
 * along the root-to-leaf path that contains that vertex is subdivided up to
 * `"refinement_depth"`.  Only cells in the vicinity of the input geometry are
 * subdivided; interior and far-field cells are left at their current level.
 *
 * When `"model_part_name"` is empty in this case, the triangle soup already
 * stored in `OctreeHybridMesherData::mTriangles` (the main input surface) is reused.
 * Multiple entries may be chained in `refinement_settings_list`, each targeting a
 * **different model part**, to achieve feature-specific resolution:
 *
 * @code{.json}
 * "refinement_settings_list": [
 *   { "type": "RefineInterfaceCellsOctreeHybrid",
 *     "refinement_depth": 3, "adaptive": true, "mesh_type": "dual" },
 *   { "type": "RefineInterfaceCellsOctreeHybrid",
 *     "model_part_name": "InnerWall", "refinement_depth": 6 }
 * ]
 * @endcode
 *
 * After all operations complete, the modeler calls `StrongConstrain2To1()` once to
 * produce a globally 2:1-balanced octree before mesh extraction.
 *
 * ### Parameters schema
 * | Key                      | Type   | Default | Description                         |
 * |--------------------------|--------|---------|-------------------------------------|
 * | `type`                   | string | `"RefineInterfaceCellsOctreeHybrid"` | Registry key. |
 * | `model_part_name`        | string | `""`    | Surface model part.  If the octree has not been built yet, empty → falls back to the modeler's top-level `input_model_part_name`.  If the octree already exists, empty → reuse the already-extracted main triangle soup. |
 * | `refinement_depth`       | int    | `5`     | Build depth (octree not yet built) or maximum refinement depth for interface cells (octree already built).  Overridden by `refined_cell_size` > 0 in the latter case. |
 * | `refined_cell_size`      | double | `0.0`   | Desired maximum cell size (world-space), used only if the octree already exists.  When > 0, overrides `refinement_depth` via `OctreeHybridMeshUtility::ElementSizeToDepth`. |
 * | `adaptive`               | bool   | `true`  | Only used to build the octree: adaptive subdivision near the surface; globally uniform when false. |
 * | `mesh_type`              | string | `"dual"` | Only used to build the octree: `"dual"` (conforming all-hex) or `"primal"` (leaf-hex with hanging-node constraints). |
 * | `project_to_surface`     | bool   | `false` | Only used to build the octree: project extracted dual mesh onto the iso-surface. |
 * | `projection_iterations`  | int    | `20000` | Only used to build the octree: maximum projection iterations. |
 * | `projection_smoothing`   | int    | `1000`  | Only used to build the octree: smoothing iterations in the projection step. |
 *
 * @see RefineOctreeHybrid
 * @see RefineUniformOctreeHybrid
 * @see OctreeHybridMeshGeneratorModeler::EnsureOctreeBuilt
 * @see OctreeHybridMeshUtility::BuildFromSurfaceMesh
 * @see OctreeHybridMeshUtility::RefineInterfaceCells
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) RefineInterfaceCellsOctreeHybrid : public RefineOctreeHybrid
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    RefineInterfaceCellsOctreeHybrid() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy.
     */
    RefineInterfaceCellsOctreeHybrid(RefineInterfaceCellsOctreeHybrid const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Builds the octree if it does not exist yet, otherwise refines interface cells.
     * @details If `rModeler.GetData().mpOctree == nullptr`, delegates to
     * `rModeler.EnsureOctreeBuilt(RefineParameters)` (which resolves `"model_part_name"`,
     * falling back to the modeler's top-level `input_model_part_name` when empty, builds
     * the octree, and writes `mesh_type`, `project_to_surface`, `projection_iterations`,
     * and `projection_smoothing` into @ref Internals::OctreeHybridMesherData) and returns.
     *
     * Otherwise: if `RefineParameters["model_part_name"]` is non-empty, the triangle soup
     * is extracted from that model part; otherwise the soup already stored in
     * `rModeler.GetData().mTriangles` is reused.  `OctreeHybridMeshUtility::RefineInterfaceCells`
     * is then called with the chosen soup and the target depth.
     *
     * @param rModeler         Reference to the owning @ref OctreeHybridMeshGeneratorModeler.
     * @param RefineParameters Validated JSON parameters (see schema in the class Doxygen).
     */
    void Refine(OctreeHybridMeshGeneratorModeler& rModeler, Parameters RefineParameters) const override;

    /**
     * @brief Returns the default parameter schema for this operation.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"                  : "RefineInterfaceCellsOctreeHybrid",
     *     "model_part_name"       : "",
     *     "refinement_depth"      : 5,
     *     "refined_cell_size"     : 0.0,
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
    KRATOS_REGISTRY_ADD_PROTOTYPE("RefineOctreeHybrid.KratosMultiphysics", RefineOctreeHybrid, RefineInterfaceCellsOctreeHybrid)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("RefineOctreeHybrid.All", RefineOctreeHybrid, RefineInterfaceCellsOctreeHybrid)

    ///@}
};

///@}

} // namespace Kratos
