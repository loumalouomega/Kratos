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
 * @class GenerateHangingNodeConstraints
 * @ingroup KratosCore
 * @brief Generates `LinearMasterSlaveConstraint`s for hanging nodes in the primal (leaf-hex) mesh.
 * @details For every @ref OctreeHybridMeshUtility::HangingConstraint produced by
 * @ref OctreeHybridMeshUtility::ExtractPrimalHexMesh, one `LinearMasterSlaveConstraint`
 * is created per requested DOF variable: the slave hanging node is tied to its 2 or 4
 * master coarse-corner nodes with the stored bilinear weights.  Nodes must exist in
 * @ref OctreeMesherModeler::mData::mNodePtrs (so the hex-generation step must run first).
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateHangingNodeConstraints : public OctreeMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    GenerateHangingNodeConstraints() = default;

    GenerateHangingNodeConstraints(GenerateHangingNodeConstraints const&) {}

    ///@}
    ///@name Operations
    ///@{

    void Generate(OctreeMesherModeler& rModeler, Parameters GenerationParameters) const override;

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.KratosMultiphysics", OctreeMesherEntityGeneration, GenerateHangingNodeConstraints)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.All", OctreeMesherEntityGeneration, GenerateHangingNodeConstraints)

    ///@}
};

///@}

} // namespace Kratos
