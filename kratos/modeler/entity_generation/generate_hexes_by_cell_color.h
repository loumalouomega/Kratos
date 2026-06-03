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
 * @class GenerateHexesByCellColor
 * @ingroup KratosCore
 * @brief Generates hexahedral elements from the extracted hex cells of the requested colour.
 * @details One element (default `Element3D8N`) per cell whose colour matches `color`; nodes are
 * de-duplicated through the modeler's @ref OctreeMesherModeler::GenerateOrRetrieveNode.  Works
 * for both the dual (conforming) and primal (leaf-hex) meshes since it only consumes the shared
 * node/cell arrays.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateHexesByCellColor : public OctreeMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    GenerateHexesByCellColor() = default;

    GenerateHexesByCellColor(GenerateHexesByCellColor const&) {}

    ///@}
    ///@name Operations
    ///@{

    void Generate(OctreeMesherModeler& rModeler, Parameters GenerationParameters) const override;

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.KratosMultiphysics", OctreeMesherEntityGeneration, GenerateHexesByCellColor)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.All", OctreeMesherEntityGeneration, GenerateHexesByCellColor)

    ///@}
};

///@}

} // namespace Kratos
