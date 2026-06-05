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
class OctreeHybridMesherModeler;

/**
 * @class OctreeHybridMesherEntityGeneration
 * @ingroup KratosCore
 * @brief Abstract base class for entity-generation stages of the @ref OctreeHybridMesherModeler pipeline.
 * @details Entity-generation stages transform the in-memory hex mesh stored in
 * `OctreeHybridMesherData` into Kratos ModelPart entities:
 *
 *   - **Hexahedral elements** — one `Element3D8N` (or a user-chosen type) per cell
 *     whose colour matches the requested value (@ref OctreeHybridGenerateHexesByCellColor).
 *   - **Boundary conditions** — conditions on boundary faces (future extensions).
 *   - **Hanging-node constraints** — master–slave `MasterSlaveConstraint` objects at
 *     2:1 refinement transitions in the primal mesh.
 *
 * The class follows the *Registry prototype* pattern: concrete sub-classes register
 * themselves through `KRATOS_REGISTRY_ADD_PROTOTYPE` so that
 * `OctreeHybridMesherModeler::Dispatch` can instantiate them by name at run time.  The
 * do-work method @ref Generate is `const` and stateless — all mutable state lives in
 * the @ref OctreeHybridMesherModeler passed as argument.
 *
 * ### Typical pipeline position
 * 1. `SetupModelPart` builds + balances the octree and extracts the hex mesh.
 * 2. Colouring assigns per-cell labels.
 * 3. **Entity generation** (this class) emits ModelPart nodes, elements and constraints.
 * 4. Optional post-processing operations.
 *
 * @see OctreeHybridGenerateHexesByCellColor
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridMesherModeler
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridMesherEntityGeneration
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of OctreeHybridMesherEntityGeneration.
    KRATOS_CLASS_POINTER_DEFINITION(OctreeHybridMesherEntityGeneration);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridMesherEntityGeneration() = default;

    /**
     * @brief Copy constructor (no-op body — derived classes carry no state).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridMesherEntityGeneration(OctreeHybridMesherEntityGeneration const& rOther) {}

    /// Destructor.
    virtual ~OctreeHybridMesherEntityGeneration() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Executes the entity-generation stage on the given modeler.
     * @details Derived classes must override this method to create Kratos entities
     * (nodes, elements, conditions, constraints) inside the relevant ModelPart.  The
     * base implementation unconditionally raises a `KRATOS_ERROR` to prevent
     * accidental calls through the prototype.
     *
     * The method is intentionally `const` so that a single Registry prototype object
     * can be shared across many modeler invocations without data races.  All mutable
     * state must be read from / written to @p rModeler.
     *
     * @param rModeler              The owning modeler; its shared data and ModelParts
     *                              are modified in-place.
     * @param GenerationParameters  Validated and default-merged JSON parameters for
     *                              this specific generation step (see
     *                              @ref GetDefaultParameters for the schema).
     */
    virtual void Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const
    {
        KRATOS_ERROR << "Calling base OctreeHybridMesherEntityGeneration::Generate. Please override it in the derived class." << std::endl;
    }

    /**
     * @brief Validates @p rGenerationParameters against the schema returned by
     *        @ref GetDefaultParameters, filling in missing keys with their defaults.
     * @param rGenerationParameters Parameters object to validate and complete in-place.
     */
    void ValidateParameters(Parameters& rGenerationParameters) const
    {
        rGenerationParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    }

    /**
     * @brief Returns the default parameter schema for this entity-generation stage.
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
    virtual std::string Info() const { return "OctreeHybridMesherEntityGeneration"; }

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers the base prototype at path "OctreeHybridMesherEntityGeneration.KratosMultiphysics.OctreeHybridMesherEntityGeneration.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics", OctreeHybridMesherEntityGeneration, OctreeHybridMesherEntityGeneration)
    /// Registers the base prototype at path "OctreeHybridMesherEntityGeneration.All.OctreeHybridMesherEntityGeneration.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All", OctreeHybridMesherEntityGeneration, OctreeHybridMesherEntityGeneration)

    ///@}
};

///@}
///@name Input and output
///@{

/**
 * @brief Stream insertion operator for @ref OctreeHybridMesherEntityGeneration.
 * @param rOStream Output stream.
 * @param rThis    Instance to print (delegates to @ref OctreeHybridMesherEntityGeneration::Info).
 * @return Reference to @p rOStream for chaining.
 */
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeHybridMesherEntityGeneration& rThis)
{
    rOStream << rThis.Info();
    return rOStream;
}

///@}

} // namespace Kratos
