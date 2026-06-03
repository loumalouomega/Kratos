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

/// Forward declaration — avoids pulling the heavy octree header into every translation unit.
class OctreeMesherModeler;

/**
 * @class OctreeMesherColoring
 * @ingroup KratosCore
 * @brief Abstract base class for cell-colouring stages of the @ref OctreeMesherModeler pipeline.
 * @details Colouring stages classify the extracted dual/primal hexahedral cells, writing an
 * integer label into the shared per-cell colour array
 * (`OctreeMesherData::mCellColor`, indexed by cell).  Downstream entity-generation stages
 * emit only cells whose colour matches the requested value (e.g.\ inside == 1, outside == 0).
 *
 * The class follows the *Registry prototype* pattern: concrete sub-classes register themselves
 * through `KRATOS_REGISTRY_ADD_PROTOTYPE` so that `OctreeMesherModeler::Dispatch` can
 * instantiate them by name at run time.  The do-work method @ref Apply is `const` and
 * stateless — all mutable state lives in the @ref OctreeMesherModeler passed as argument.
 *
 * ### Typical pipeline position
 * 1. `SetupModelPart` builds + balances the octree and extracts the hex mesh.
 * 2. **Colouring** (this class) assigns per-cell labels.
 * 3. Entity generation emits only cells with the desired colour.
 * 4. Optional post-processing operations.
 *
 * @see ClassifyCellsInsideOutside
 * @see OctreeMesherEntityGeneration
 * @see OctreeMesherModeler
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeMesherColoring
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of OctreeMesherColoring.
    KRATOS_CLASS_POINTER_DEFINITION(OctreeMesherColoring);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeMesherColoring() = default;

    /**
     * @brief Copy constructor (no-op body — derived classes carry no state).
     * @param rOther Source instance; no data is copied.
     */
    OctreeMesherColoring(OctreeMesherColoring const& rOther) {}

    /// Destructor.
    virtual ~OctreeMesherColoring() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Executes the colouring stage on the given modeler.
     * @details Derived classes must override this method to populate
     * `rModeler.GetData().mCellColor`.  The base implementation unconditionally
     * raises a `KRATOS_ERROR` to prevent accidental calls through the prototype.
     *
     * The method is intentionally `const` so that a single Registry prototype
     * object can be shared across many modeler invocations without data races.
     * All mutable state must be read from / written to @p rModeler.
     *
     * @param rModeler          The owning modeler whose shared data is modified
     *                          in-place (cell-colour array, octree, etc.).
     * @param ColoringParameters Validated and default-merged JSON parameters for
     *                          this specific colouring step (see
     *                          @ref GetDefaultParameters for the schema).
     */
    virtual void Apply(OctreeMesherModeler& rModeler, Parameters ColoringParameters) const
    {
        KRATOS_ERROR << "Calling base OctreeMesherColoring::Apply. Please override it in the derived class." << std::endl;
    }

    /**
     * @brief Validates @p rColoringParameters against the schema returned by
     *        @ref GetDefaultParameters, filling in missing keys with their defaults.
     * @param rColoringParameters Parameters object to validate and complete in-place.
     */
    void ValidateParameters(Parameters& rColoringParameters) const
    {
        rColoringParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    }

    /**
     * @brief Returns the default parameter schema for this colouring stage.
     * @details The base class schema only requires the `"type"` key (empty string).
     * Derived classes should override this and add their specific keys.  The returned
     * object is used by @ref ValidateParameters to assign defaults.
     * @return A @ref Parameters object describing the accepted keys and their defaults.
     */
    virtual const Parameters GetDefaultParameters() const
    {
        return Parameters(R"({"type" : ""})");
    }

    ///@}
    ///@name Input and output
    ///@{

    /**
     * @brief Returns a string identifying this component.
     * @return Human-readable name; derived classes should override to return their own name.
     */
    virtual std::string Info() const { return "OctreeMesherColoring"; }

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers the base prototype at path "OctreeMesherColoring.KratosMultiphysics.OctreeMesherColoring.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.KratosMultiphysics", OctreeMesherColoring, OctreeMesherColoring)
    /// Registers the base prototype at path "OctreeMesherColoring.All.OctreeMesherColoring.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.All", OctreeMesherColoring, OctreeMesherColoring)

    ///@}
};

///@}
///@name Input and output
///@{

/**
 * @brief Stream insertion operator for @ref OctreeMesherColoring.
 * @param rOStream Output stream.
 * @param rThis    Instance to print (delegates to @ref OctreeMesherColoring::Info).
 * @return Reference to @p rOStream for chaining.
 */
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeMesherColoring& rThis)
{
    rOStream << rThis.Info();
    return rOStream;
}

///@}

} // namespace Kratos
