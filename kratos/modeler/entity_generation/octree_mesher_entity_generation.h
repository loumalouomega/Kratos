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
#include "includes/define.h"
#include "includes/kratos_parameters.h"
#include "includes/registry.h"
#include "includes/registry_item.h"
#include "includes/define_registry.h"

namespace Kratos {

///@name Kratos Classes
///@{

class OctreeMesherModeler; // forward declaration (breaks the include cycle)

/**
 * @class OctreeMesherEntityGeneration
 * @ingroup KratosCore
 * @brief Base class of an entity-generation component of the @ref OctreeMesherModeler.
 * @details Entity generators turn the extracted hex mesh into ModelPart entities:
 * hexahedral elements by cell colour, boundary conditions by face, and hanging-node
 * master-slave constraints at 2:1 transitions (primal mesh).  Components are Registry
 * prototypes (`KRATOS_REGISTRY_ADD_PROTOTYPE`); the do-work method @ref Generate is
 * `const` and receives the modeler and parameters as arguments.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeMesherEntityGeneration
{
public:
    ///@name Type Definitions
    ///@{

    KRATOS_CLASS_POINTER_DEFINITION(OctreeMesherEntityGeneration);

    ///@}
    ///@name Life Cycle
    ///@{

    OctreeMesherEntityGeneration() = default;

    OctreeMesherEntityGeneration(OctreeMesherEntityGeneration const&) {}

    virtual ~OctreeMesherEntityGeneration() = default;

    ///@}
    ///@name Operations
    ///@{

    /// Generates the entities. Override in derived classes.
    virtual void Generate(OctreeMesherModeler& rModeler, Parameters GenerationParameters) const
    {
        KRATOS_ERROR << "Calling base OctreeMesherEntityGeneration::Generate. Please override it in the derived class." << std::endl;
    }

    /// Validates @p rGenerationParameters against @ref GetDefaultParameters (assigning defaults).
    void ValidateParameters(Parameters& rGenerationParameters) const
    {
        rGenerationParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    }

    /// The default parameters (schema) of the generator.
    virtual const Parameters GetDefaultParameters() const
    {
        return Parameters(R"({"type" : ""})");
    }

    ///@}
    ///@name Input and output
    ///@{

    virtual std::string Info() const { return "OctreeMesherEntityGeneration"; }

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.KratosMultiphysics", OctreeMesherEntityGeneration, OctreeMesherEntityGeneration)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherEntityGeneration.All", OctreeMesherEntityGeneration, OctreeMesherEntityGeneration)

    ///@}
};

///@}
///@name Input and output
///@{

inline std::ostream& operator<<(std::ostream& rOStream, const OctreeMesherEntityGeneration& rThis)
{
    rOStream << rThis.Info();
    return rOStream;
}

///@}

} // namespace Kratos
