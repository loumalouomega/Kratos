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
#include "modeler/operation/octree_mesher_operation.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class ReportMeshQuality
 * @ingroup KratosCore
 * @brief Reports per-element scaled-Jacobian statistics on the finished ModelPart.
 * @details Iterates every `Element3D8N` (or whatever element was generated) in the
 * configured `model_part_name`, computes the minimum scaled Jacobian of each element,
 * and logs min, mean, and inverted-count at INFO level.  The computation is read-only;
 * no mesh topology is changed.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) ReportMeshQuality : public OctreeMesherOperation
{
public:
    ///@name Life Cycle
    ///@{

    ReportMeshQuality() = default;

    ReportMeshQuality(ReportMeshQuality const&) {}

    ///@}
    ///@name Operations
    ///@{

    void Execute(OctreeMesherModeler& rModeler, Parameters OperationParameters) const override;

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherOperation.KratosMultiphysics", OctreeMesherOperation, ReportMeshQuality)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherOperation.All", OctreeMesherOperation, ReportMeshQuality)

    ///@}
};

///@}

} // namespace Kratos
