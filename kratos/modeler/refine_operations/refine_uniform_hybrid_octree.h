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
 * @class OctreeHybridRefineUniform
 * @ingroup KratosCore
 * @brief Refinement operation that subdivides **all** octree leaves to a prescribed depth.
 * @details This operation iterates over the current leaf set and repeatedly subdivides
 * every leaf whose level is below `"refinement_depth"` until the entire octree reaches
 * the requested resolution.  The result is a uniformly refined octree regardless of the
 * proximity of cells to the input surface.
 *
 * It is typically used when a uniform background mesh is desired, either as a standalone
 * refinement pass or in combination with @ref OctreeHybridRefineInterfaceCells (which
 * adds extra resolution near specific surfaces).
 *
 * ### Parameters schema
 * | Key                | Type   | Default | Description                              |
 * |--------------------|--------|---------|------------------------------------------|
 * | `type`             | string | `"OctreeHybridRefineUniform"` | Registry lookup key. |
 * | `refinement_depth`  | int    | `5`     | Target refinement depth.  Used when `refined_cell_size` is 0. |
 * | `refined_cell_size` | double | `0.0`   | Desired maximum cell size in world-space units.  When > 0, overrides `refinement_depth` and the equivalent depth is computed via `OctreeHybridMeshUtility::ElementSizeToDepth`. |
 *
 * ### Example JSON usage
 * @code{.json}
 * "refinement_settings_list": [
 *   { "type": "OctreeHybridRefineUniform", "refinement_depth": 4 }
 * ]
 * @endcode
 *
 * @see OctreeHybridRefineOperation
 * @see OctreeHybridRefineInterfaceCells
 * @see OctreeHybridMeshUtility::RefineAllCells
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridRefineUniform : public OctreeHybridRefineOperation
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridRefineUniform() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy.
     */
    OctreeHybridRefineUniform(OctreeHybridRefineUniform const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Refines all octree leaves to the depth specified in @p RefineParameters.
     * @details Retrieves the octree from `rModeler.GetData().mpOctree` and calls
     * `OctreeHybridMeshUtility::RefineAllCells()` with the target depth from
     * `RefineParameters["refinement_depth"]`.
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
     *     "type"              : "OctreeHybridRefineUniform",
     *     "refinement_depth"  : 5,
     *     "refined_cell_size" : 0.0
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
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridRefineOperation.KratosMultiphysics", OctreeHybridRefineOperation, OctreeHybridRefineUniform)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridRefineOperation.All", OctreeHybridRefineOperation, OctreeHybridRefineUniform)

    ///@}
};

///@}

} // namespace Kratos
