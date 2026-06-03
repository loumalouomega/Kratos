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
 * @class OctreeMesherOperation
 * @ingroup KratosCore
 * @brief Base class of a post-processing operation of the @ref OctreeMesherModeler.
 * @details Operations run after entity generation, on the finished ModelPart
 * (e.g. a mesh-quality report).  Components are registered as Registry prototypes
 * (`KRATOS_REGISTRY_ADD_PROTOTYPE`) and retrieved by the modeler as a single shared
 * stateless object, so the do-work method @ref Execute is `const` and receives the
 * modeler and its own parameters as arguments rather than storing them.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeMesherOperation
{
public:
    ///@name Type Definitions
    ///@{

    KRATOS_CLASS_POINTER_DEFINITION(OctreeMesherOperation);

    ///@}
    ///@name Life Cycle
    ///@{

    OctreeMesherOperation() = default;

    OctreeMesherOperation(OctreeMesherOperation const&) {}

    virtual ~OctreeMesherOperation() = default;

    ///@}
    ///@name Operations
    ///@{

    /// Runs the operation. Override in derived classes.
    virtual void Execute(OctreeMesherModeler& rModeler, Parameters OperationParameters) const
    {
        KRATOS_ERROR << "Calling base OctreeMesherOperation::Execute. Please override it in the derived class." << std::endl;
    }

    /// Validates @p rOperationParameters against @ref GetDefaultParameters (assigning defaults).
    void ValidateParameters(Parameters& rOperationParameters) const
    {
        rOperationParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    }

    /// The default parameters (schema) of the operation.
    virtual const Parameters GetDefaultParameters() const
    {
        return Parameters(R"({"type" : ""})");
    }

    ///@}
    ///@name Input and output
    ///@{

    virtual std::string Info() const { return "OctreeMesherOperation"; }

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherOperation.KratosMultiphysics", OctreeMesherOperation, OctreeMesherOperation)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherOperation.All", OctreeMesherOperation, OctreeMesherOperation)

    ///@}
};

///@}
///@name Input and output
///@{

inline std::ostream& operator<<(std::ostream& rOStream, const OctreeMesherOperation& rThis)
{
    rOStream << rThis.Info();
    return rOStream;
}

///@}

} // namespace Kratos
