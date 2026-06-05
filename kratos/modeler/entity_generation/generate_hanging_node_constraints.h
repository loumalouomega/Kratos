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
 * @class GenerateHangingNodeConstraints
 * @ingroup KratosCore
 * @brief Entity-generation component that creates `LinearMasterSlaveConstraint`s for
 *        hanging nodes in the primal (leaf-hex) octree mesh.
 * @details In a non-uniform octree mesh, cells of different refinement level share faces
 * at 2:1 transitions.  Finer cells introduce *hanging nodes* — primal-mesh nodes that
 * lie on an edge or face of a coarser neighbour but are not corner nodes of that
 * neighbour.  Without additional constraints these nodes are kinematically independent
 * of the coarser mesh, which breaks displacement compatibility.
 *
 * This generator reads the `OctreeHybridMeshUtility::HangingConstraint` records stored
 * in `OctreeHybridMesherData::mHanging` (populated by the hex-generation step) and creates
 * one `LinearMasterSlaveConstraint` per **(hanging node × configured DOF variable)**
 * pair.  The linear relation imposed is:
 *
 * @f[
 *   u_s = \sum_{m=0}^{N_m - 1} w_m \, u_{M_m}
 * @f]
 *
 * where @f$u_s@f$ is the slave (hanging-node) DOF, @f$u_{M_m}@f$ are the master
 * (coarse-corner) DOFs, and @f$w_m@f$ are the bilinear interpolation weights stored in
 * `HangingConstraint::Weights`.  @f$N_m@f$ is 2 for an edge-midpoint hanging node
 * and 4 for a face-centre hanging node.
 *
 * ### Prerequisite
 * The primal hex-generation step (i.e. `GenerateHexesByCellColor` with
 * `mesh_type = "primal"`) must have already run so that `mData.mNodePtrs` is fully
 * populated.  A hanging node or master node that has no entry in `mNodePtrs` (e.g.
 * because it belongs to a carved-away outside cell) is silently skipped together with
 * its entire constraint record, so no error is thrown for partially carved meshes.
 *
 * If the mesh has no 2:1 transitions (all cells at the same refinement level)
 * `mData.mHanging` is empty and the method returns immediately without creating any
 * constraints.
 *
 * ### DOF management
 * For each constraint this generator calls `Node::AddDof` on both the slave and all
 * master nodes, ensuring the required DOFs exist even if they have not been added by
 * an upstream step.
 *
 * ### Registry paths
 * Registered at:
 * - `OctreeHybridMesherEntityGeneration.KratosMultiphysics`
 * - `OctreeHybridMesherEntityGeneration.All`
 *
 * ### Parameters schema
 * | Key                | Type         | Default                                          | Description                                       |
 * |--------------------|--------------|--------------------------------------------------|---------------------------------------------------|
 * | `type`             | string       | `"GenerateHangingNodeConstraints"`               | Registry lookup key.                              |
 * | `model_part_name`  | string       | `"Undefined"`                                    | ModelPart to add constraints to.                  |
 * | `constraint_name`  | string       | `"LinearMasterSlaveConstraint"`                  | Registered constraint type to instantiate.        |
 * | `variables`        | string array | `["DISPLACEMENT_X","DISPLACEMENT_Y","DISPLACEMENT_Z"]` | Scalar DOF variables to constrain.          |
 *
 * @note All variable names listed in `variables` must be registered as
 *       `Variable<double>` in the Kratos component database at the time of execution.
 * @see OctreeHybridMesherEntityGeneration
 * @see OctreeHybridMeshUtility::HangingConstraint
 * @see OctreeHybridMesherData::mHanging
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateHangingNodeConstraints : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    GenerateHangingNodeConstraints() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy in this class.
     */
    GenerateHangingNodeConstraints(GenerateHangingNodeConstraints const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Creates `LinearMasterSlaveConstraint`s for all hanging nodes in the
     *        primal mesh, for each configured DOF variable.
     * @details For each `HangingConstraint` record in `mData.mHanging`:
     * 1. The slave node pointer is looked up in `mData.mNodePtrs`; the record is
     *    skipped if the pointer is null (node was not created).
     * 2. All master node pointers are looked up; the record is skipped if any
     *    master pointer is null.
     * 3. For each variable in the `"variables"` list:
     *    - `Node::AddDof` is called on the slave and all master nodes.
     *    - A 1×N_m relation matrix is built from `HangingConstraint::Weights`.
     *    - A zero constant vector of size 1 is created.
     *    - `ModelPart::CreateNewMasterSlaveConstraint` is called with the registered
     *      constraint type name, a fresh constraint ID from
     *      `OctreeHybridMesherModeler::NextConstraintId`, and the assembled DOF lists.
     *
     * @param rModeler              The owning @ref OctreeHybridMesherModeler.  Provides the
     *                              Model, `OctreeHybridMesherData::mHanging`,
     *                              `OctreeHybridMesherData::mNodePtrs`, and the constraint
     *                              ID counter.
     * @param GenerationParameters  Validated JSON parameters; see @ref GetDefaultParameters
     *                              for the full schema.
     */
    void Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const override;

    /**
     * @brief Returns the default parameter schema for this generator.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"            : "GenerateHangingNodeConstraints",
     *     "model_part_name" : "Undefined",
     *     "constraint_name" : "LinearMasterSlaveConstraint",
     *     "variables"       : ["DISPLACEMENT_X","DISPLACEMENT_Y","DISPLACEMENT_Z"]
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
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics", OctreeHybridMesherEntityGeneration, GenerateHangingNodeConstraints)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All", OctreeHybridMesherEntityGeneration, GenerateHangingNodeConstraints)

    ///@}
};

///@}

} // namespace Kratos
