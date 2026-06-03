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
 * @class OctreeMesherColoring
 * @ingroup KratosCore
 * @brief Base class of a colouring component of the @ref OctreeMesherModeler.
 * @details Colouring classifies the extracted dual/primal hex cells (e.g. inside
 * vs. outside the surface), writing into the shared per-cell colour array; later
 * entity generation emits only cells of the requested colour.  Components are
 * Registry prototypes (`KRATOS_REGISTRY_ADD_PROTOTYPE`); the do-work method
 * @ref Apply is `const` and receives the modeler and parameters as arguments.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeMesherColoring
{
public:
    ///@name Type Definitions
    ///@{

    KRATOS_CLASS_POINTER_DEFINITION(OctreeMesherColoring);

    ///@}
    ///@name Life Cycle
    ///@{

    OctreeMesherColoring() = default;

    OctreeMesherColoring(OctreeMesherColoring const&) {}

    virtual ~OctreeMesherColoring() = default;

    ///@}
    ///@name Operations
    ///@{

    /// Applies the colouring. Override in derived classes.
    virtual void Apply(OctreeMesherModeler& rModeler, Parameters ColoringParameters) const
    {
        KRATOS_ERROR << "Calling base OctreeMesherColoring::Apply. Please override it in the derived class." << std::endl;
    }

    /// Validates @p rColoringParameters against @ref GetDefaultParameters (assigning defaults).
    void ValidateParameters(Parameters& rColoringParameters) const
    {
        rColoringParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    }

    /// The default parameters (schema) of the colouring.
    virtual const Parameters GetDefaultParameters() const
    {
        return Parameters(R"({"type" : ""})");
    }

    ///@}
    ///@name Input and output
    ///@{

    virtual std::string Info() const { return "OctreeMesherColoring"; }

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.KratosMultiphysics", OctreeMesherColoring, OctreeMesherColoring)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.All", OctreeMesherColoring, OctreeMesherColoring)

    ///@}
};

///@}
///@name Input and output
///@{

inline std::ostream& operator<<(std::ostream& rOStream, const OctreeMesherColoring& rThis)
{
    rOStream << rThis.Info();
    return rOStream;
}

///@}

} // namespace Kratos
