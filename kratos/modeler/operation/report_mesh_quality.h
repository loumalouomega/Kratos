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
#include "modeler/operation/octree_hybrid_mesher_operation.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class ReportMeshQuality
 * @ingroup KratosCore
 * @brief Post-processing operation that logs scaled-Jacobian statistics for the hex mesh.
 * @details Iterates every eight-noded hexahedral element (`Element3D8N` or any element
 * with exactly 8 geometry nodes) in the ModelPart identified by `model_part_name`.
 * For each element it computes the minimum scaled Jacobian over its 8 corners and the
 * body centre using `OctreeHybridMeshUtility::ScaledJacobianMin`, which applies the
 * `SJ_ADJ` edge-triple convention from the underlying octree engine.
 *
 * At the end of the loop the following statistics are emitted at `INFO` level via
 * `KRATOS_INFO("ReportMeshQuality")`:
 * - **minSJ**   — minimum scaled Jacobian across all elements (worst element quality).
 * - **meanSJ**  — arithmetic mean of per-element minimum scaled Jacobians.
 * - **inverted** — count (and percentage) of elements whose minimum scaled Jacobian is
 *                  `<= 0`, i.e. inverted or degenerate elements.
 *
 * This operation is purely read-only: it does not create, modify, or remove any mesh
 * entity.  It is intended as a diagnostic pass run after element generation to confirm
 * that the produced mesh meets quality requirements.
 *
 * ### Registry paths
 * Registered at:
 * - `OctreeHybridMesherOperation.KratosMultiphysics`
 * - `OctreeHybridMesherOperation.All`
 *
 * ### Parameters schema
 * | Key                | Type   | Default           | Description                       |
 * |--------------------|--------|-------------------|-----------------------------------|
 * | `type`             | string | `"ReportMeshQuality"` | Registry lookup key.          |
 * | `model_part_name`  | string | `"Undefined"`     | Name of the ModelPart to analyse. |
 *
 * @note Elements with fewer or more than 8 nodes are silently skipped.
 * @see OctreeHybridMesherOperation
 * @see OctreeHybridMeshUtility::ScaledJacobianMin
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) ReportMeshQuality : public OctreeHybridMesherOperation
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    ReportMeshQuality() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy in this class.
     */
    ReportMeshQuality(ReportMeshQuality const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Computes and logs scaled-Jacobian statistics for the configured ModelPart.
     * @details Retrieves the ModelPart by the name stored in
     * `OperationParameters["model_part_name"]`, then iterates all elements.  For each
     * hexahedral element (8 nodes) it evaluates the minimum scaled Jacobian with
     * `OctreeHybridMeshUtility::ScaledJacobianMin`, accumulates the global minimum,
     * running sum, and inverted count, and finally logs the aggregate statistics at
     * the `INFO` log level.  If the ModelPart contains no elements the method logs a
     * short notice and returns immediately without error.
     *
     * @param rModeler            The @ref OctreeHybridMesherModeler that holds the Model
     *                            from which the target ModelPart is retrieved.
     * @param OperationParameters Validated JSON parameters.  Must contain the key
     *                            `"model_part_name"` with the name of an existing
     *                            ModelPart.
     */
    void Execute(OctreeHybridMesherModeler& rModeler, Parameters OperationParameters) const override;

    /**
     * @brief Returns the default parameter schema for this operation.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"            : "ReportMeshQuality",
     *     "model_part_name" : "Undefined"
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
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherOperation.KratosMultiphysics", OctreeHybridMesherOperation, ReportMeshQuality)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherOperation.All", OctreeHybridMesherOperation, ReportMeshQuality)

    ///@}
};

///@}

} // namespace Kratos
