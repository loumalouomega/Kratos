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
 * @class RefineUniformOctreeHybrid
 * @ingroup KratosCore
 * @brief Refinement operation that subdivides **all** octree leaves to a prescribed depth.
 * @details This operation iterates over the current leaf set and repeatedly subdivides
 * every leaf whose level is below `"refinement_depth"` until the entire octree reaches
 * the requested resolution.  The result is a uniformly refined octree regardless of the
 * proximity of cells to the input surface.
 *
 * It is typically used when a uniform background mesh is desired, either as a standalone
 * refinement pass or in combination with @ref RefineInterfaceCellsOctreeHybrid (which
 * adds extra resolution near specific surfaces).
 *
 * If no octree has been built yet (e.g. this is the first entry in
 * `refinement_settings_list`), `Refine` delegates to
 * `OctreeHybridMeshGeneratorModeler::EnsureOctreeBuilt`, which builds the octree from
 * `model_part_name` (falling back to the modeler's top-level `input_model_part_name`
 * when left as `"Undefined"`) using `refinement_depth` and default values for the
 * remaining build options (`adaptive`, `mesh_type`, `project_to_surface`, ...), before
 * proceeding with the uniform refinement below.
 *
 * ### Parameters schema
 * | Key                | Type   | Default | Description                              |
 * |--------------------|--------|---------|------------------------------------------|
 * | `type`             | string | `"RefineUniformOctreeHybrid"` | Registry lookup key. |
 * | `refinement_depth`  | int    | `5`     | Target refinement depth.  Used when `max_voxel_size` is 0. |
 * | `model_part_name`  | string | `"Undefined"` | Surface model part used to build the octree if it has not been built yet.  `"Undefined"` falls back to the modeler's top-level `input_model_part_name`.  Ignored if the octree already exists. |
 * | `max_voxel_size` | double | `0.0`   | Desired maximum cell size in world-space units.  When > 0, overrides `refinement_depth` and the equivalent depth is computed via `OctreeHybridMeshUtility::ElementSizeToDepth`. |
 *
 * ### Example JSON usage
 * @code{.json}
 * "refinement_settings_list": [
 *   { "type": "RefineUniformOctreeHybrid", "refinement_depth": 4 }
 * ]
 * @endcode
 *
 * @see RefineOctreeHybrid
 * @see RefineInterfaceCellsOctreeHybrid
 * @see OctreeHybridMeshUtility::RefineAllCells
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) RefineUniformOctreeHybrid : public RefineOctreeHybrid
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    RefineUniformOctreeHybrid() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy.
     */
    RefineUniformOctreeHybrid(RefineUniformOctreeHybrid const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Refines all octree leaves to the depth specified in @p RefineParameters.
     * @details If `rModeler.GetData().mpOctree` is null, first calls
     * `rModeler.EnsureOctreeBuilt(RefineParameters)` to build the octree.  Then calls
     * `OctreeHybridMeshUtility::RefineAllCells()` with the target depth from
     * `RefineParameters["refinement_depth"]` (or `"max_voxel_size"` if > 0).
     *
     * @param rModeler         Reference to the owning @ref OctreeHybridMeshGeneratorModeler.
     * @param RefineParameters Validated JSON parameters containing `"refinement_depth"`.
     */
    void Refine(OctreeHybridMeshGeneratorModeler& rModeler, Parameters RefineParameters) const override;

    /**
     * @brief Returns the default parameter schema for this operation.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"              : "RefineUniformOctreeHybrid",
     *     "refinement_depth"  : 5,
     *     "model_part_name"   : "Undefined",
     *     "max_voxel_size"    : 0.0
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
    KRATOS_REGISTRY_ADD_PROTOTYPE("RefineOctreeHybrid.KratosMultiphysics", RefineOctreeHybrid, RefineUniformOctreeHybrid)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("RefineOctreeHybrid.All", RefineOctreeHybrid, RefineUniformOctreeHybrid)

    ///@}
};

///@}

} // namespace Kratos
