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
 * @class OctreeHybridRefineOperation
 * @ingroup KratosCore
 * @brief Base class for octree refinement operations dispatched by @ref OctreeHybridMeshGeneratorModeler.
 * @details Refinement operations run inside `BuildOctreeAndExtract()` **after** the initial
 * octree is built from the surface mesh and **before** `StrongConstrain2To1()` and mesh
 * extraction.  This placement lets multiple refinement passes accumulate without
 * repeatedly balancing and extracting.
 *
 * The design mirrors the Registry-prototype pattern used by @ref OctreeHybridMesherColoring,
 * @ref OctreeHybridMesherEntityGeneration, and @ref OctreeHybridMesherOperation: each
 * concrete derived class registers itself via `KRATOS_REGISTRY_ADD_PROTOTYPE` under two paths —
 * `"OctreeHybridRefineOperation.KratosMultiphysics"` and `"OctreeHybridRefineOperation.All"`.
 * The modeler retrieves a shared, stateless prototype and calls @ref Refine on it.
 * Because instances may be shared, @ref Refine is declared `const` and all context is
 * passed through the `OctreeHybridMeshGeneratorModeler&` and `Parameters` arguments.
 *
 * ### Derived-class contract
 * - Override @ref Refine with the actual work.
 * - Override @ref GetDefaultParameters to provide the JSON schema.
 * - Register with the two `KRATOS_REGISTRY_ADD_PROTOTYPE` macros in the `private` section.
 *
 * ### Pipeline position
 * ```
 * BuildOctreeAndExtract():
 *   1. BuildFromSurfaceMesh()          ← initial octree build
 *   2. Dispatch refine_operations_list ← this base class is invoked here
 *   3. StrongConstrain2To1()
 *   4. ExtractDualHexMesh / ExtractPrimalHexMesh
 * ```
 *
 * @see OctreeHybridMeshGeneratorModeler
 * @see OctreeHybridRefineUniform
 * @see OctreeHybridRefineInterfaceCells
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridRefineOperation
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of OctreeHybridRefineOperation.
    KRATOS_CLASS_POINTER_DEFINITION(OctreeHybridRefineOperation);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridRefineOperation() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The operation to copy.  No data members need copying in the base.
     */
    OctreeHybridRefineOperation(OctreeHybridRefineOperation const& rOther) {}

    /// Virtual destructor.
    virtual ~OctreeHybridRefineOperation() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Performs the refinement on the modeler's octree.
     * @details This is the main do-work entry point called by the modeler's
     * `refine_operations_list` dispatch.  It is `const` because operations are
     * retrieved as shared stateless prototypes; all required context is passed via
     * @p rModeler and @p RefineParameters.
     *
     * The base implementation raises an error to force derived classes to override it.
     *
     * @param rModeler         Reference to the @ref OctreeHybridMeshGeneratorModeler that owns
     *                         the Model, the mesh data (octree, triangle soup), and the
     *                         ID counters.
     * @param RefineParameters JSON parameters for this specific invocation, already
     *                         validated and default-filled by @ref ValidateParameters.
     */
    virtual void Refine(OctreeHybridMeshGeneratorModeler& rModeler, Parameters RefineParameters) const
    {
        KRATOS_ERROR << "Calling base OctreeHybridRefineOperation::Refine. "
                     << "Please override it in the derived class." << std::endl;
    }

    /**
     * @brief Validates @p rRefineParameters against this operation's default schema.
     * @details Calls `Parameters::ValidateAndAssignDefaults` using the result of
     * @ref GetDefaultParameters, so missing optional keys are filled with defaults
     * and unknown keys trigger an error.
     * @param rRefineParameters The parameter set to validate in-place.
     */
    void ValidateParameters(Parameters& rRefineParameters) const
    {
        rRefineParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    }

    /**
     * @brief Returns the default (and schema) parameters of this operation.
     * @details Derived classes should override this to enumerate every accepted key
     * with a sensible default.  The base provides only the mandatory `"type"` key.
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
     * @return `"OctreeHybridRefineOperation"` in the base; derived classes return their name.
     */
    virtual std::string Info() const { return "OctreeHybridRefineOperation"; }

    ///@}
private:
    ///@name Registry
    ///@{

    /// Self-registers the base class prototype at the KratosMultiphysics sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridRefineOperation.KratosMultiphysics", OctreeHybridRefineOperation, OctreeHybridRefineOperation)
    /// Self-registers the base class prototype at the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridRefineOperation.All", OctreeHybridRefineOperation, OctreeHybridRefineOperation)

    ///@}
};

///@}
///@name Input and output
///@{

/**
 * @brief Stream insertion operator for @ref OctreeHybridRefineOperation.
 * @param rOStream Output stream to write to.
 * @param rThis    The operation whose @ref Info string is printed.
 * @return Reference to @p rOStream for chaining.
 */
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeHybridRefineOperation& rThis)
{
    rOStream << rThis.Info();
    return rOStream;
}

///@}

} // namespace Kratos
