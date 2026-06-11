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
#include "includes/kratos_parameters.h"
#include "includes/registry.h"
#include "includes/registry_item.h"
#include "includes/define_registry.h"

namespace Kratos {

///@name Kratos Classes
///@{

/// Forward declaration to break the include cycle with octree_hybrid_mesh_generator_modeler.h.
class OctreeHybridMeshGeneratorModeler;

/**
 * @class OctreeHybridMesherOperation
 * @ingroup KratosCore
 * @brief Base class for post-processing operations executed by @ref OctreeHybridMeshGeneratorModeler.
 * @details Operations run after the entity-generation stage, acting on the finished
 * ModelPart.  Typical uses include mesh-quality reporting, post-smoothing diagnostics,
 * and any read-only or topology-preserving pass over the generated mesh.
 *
 * The design follows a Registry-prototype pattern: each concrete derived class registers
 * itself via `KRATOS_REGISTRY_ADD_PROTOTYPE` under two paths —
 * `"OctreeHybridMesherOperation.KratosMultiphysics"` and `"OctreeHybridMesherOperation.All"`.
 * The modeler retrieves a shared, stateless prototype and calls @ref Execute on it.
 * Because all instances may be shared, @ref Execute is declared `const` and receives
 * the modeler reference and its own Parameters as arguments rather than storing state.
 *
 * ### Derived-class contract
 * - Override @ref Execute with the actual work.
 * - Override @ref GetDefaultParameters to provide the JSON schema (used for validation
 *   and default assignment by @ref ValidateParameters).
 * - Register with the two `KRATOS_REGISTRY_ADD_PROTOTYPE` macros in the `private` section.
 *
 * @see OctreeHybridMeshGeneratorModeler
 * @see OctreeHybridReportMeshQuality
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridMesherOperation
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of OctreeHybridMesherOperation.
    KRATOS_CLASS_POINTER_DEFINITION(OctreeHybridMesherOperation);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridMesherOperation() = default;

    /**
     * @brief Copy constructor (explicitly defined so that derived classes may also
     *        be copy-constructed without special effort).
     * @param rOther The operation to copy.  No data members need copying in the base.
     */
    OctreeHybridMesherOperation(OctreeHybridMesherOperation const& rOther) {}

    /// Destructor (virtual so that base pointers are correctly destroyed).
    virtual ~OctreeHybridMesherOperation() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Executes the post-processing operation on the modeler's finished ModelPart.
     * @details This is the main do-work entry point.  It is `const` because operations
     * are retrieved as shared stateless prototypes; all required context is passed via
     * @p rModeler and @p OperationParameters.
     *
     * The base implementation raises an error to force derived classes to override it.
     *
     * @param rModeler            Reference to the @ref OctreeHybridMeshGeneratorModeler that owns
     *                            the Model, the mesh data, and the ID counters.
     * @param OperationParameters JSON parameters for this specific invocation, already
     *                            validated and default-filled by @ref ValidateParameters.
     */
    virtual void Execute(OctreeHybridMeshGeneratorModeler& rModeler, Parameters OperationParameters) const
    {
        KRATOS_ERROR << "Calling base OctreeHybridMesherOperation::Execute. Please override it in the derived class." << std::endl;
    }

    /**
     * @brief Validates @p rOperationParameters against this operation's default schema.
     * @details Calls `Parameters::ValidateAndAssignDefaults` using the result of
     * @ref GetDefaultParameters, so missing optional keys are filled with their defaults
     * and unknown keys trigger an error.
     * @param rOperationParameters The parameter set to validate in-place.
     */
    void ValidateParameters(Parameters& rOperationParameters) const
    {
        rOperationParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    }

    /**
     * @brief Returns the default (and schema) parameters of this operation.
     * @details Derived classes should override this to enumerate every accepted key
     * together with a sensible default value.  The base provides only the mandatory
     * `"type"` key, which is the Registry lookup string used to instantiate the
     * operation.
     * @return A Parameters object with all accepted keys and their defaults.
     */
    virtual const Parameters GetDefaultParameters() const
    {
        return Parameters(R"({"type" : ""})");
    }

    ///@}
    ///@name Input and output
    ///@{

    /**
     * @brief Returns a string identifying this operation.
     * @return The string `"OctreeHybridMesherOperation"` in the base class; derived classes
     *         typically return their own class name.
     */
    virtual std::string Info() const { return "OctreeHybridMesherOperation"; }

    ///@}
private:
    ///@name Registry
    ///@{

    /// Self-registers the base class prototype at the KratosMultiphysics sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherOperation.KratosMultiphysics", OctreeHybridMesherOperation, OctreeHybridMesherOperation)
    /// Self-registers the base class prototype at the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherOperation.All", OctreeHybridMesherOperation, OctreeHybridMesherOperation)

    ///@}
};

///@}
///@name Input and output
///@{

/**
 * @brief Stream insertion operator for @ref OctreeHybridMesherOperation.
 * @param rOStream Output stream to write to.
 * @param rThis    The operation whose @ref OctreeHybridMesherOperation::Info string is printed.
 * @return Reference to @p rOStream for chaining.
 */
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeHybridMesherOperation& rThis)
{
    rOStream << rThis.Info();
    return rOStream;
}

///@}

} // namespace Kratos
