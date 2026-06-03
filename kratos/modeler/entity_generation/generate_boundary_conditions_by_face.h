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
 * @class GenerateBoundaryConditionsByFace
 * @ingroup KratosCore
 * @brief Generates quadrilateral boundary conditions on the exterior faces of the hex mesh.
 * @details Extracts the outer boundary quads (those owned by exactly one hex) and creates
 * one condition per quad.  Operates on cells that have been coloured with the configured
 * @p color (all inside-coloured cells by default), so the conditions represent the
 * closed boundary of the carved mesh.  The hex generator must run first so the nodes
 * exist.  Works for both dual and primal mesh topologies.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateBoundaryConditionsByFace : public OctreeMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    GenerateBoundaryConditionsByFace() = default;

    GenerateBoundaryConditionsByFace(GenerateBoundaryConditionsByFace const&) {}

    ///@}
    ///@name Operations
    ///@{

    void Generate(OctreeMesherModeler& rModeler, Parameters GenerationParameters) const override;

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.KratosMultiphysics", OctreeMesherEntityGeneration, GenerateBoundaryConditionsByFace)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.All", OctreeMesherEntityGeneration, GenerateBoundaryConditionsByFace)

    ///@}
};

///@}

} // namespace Kratos
