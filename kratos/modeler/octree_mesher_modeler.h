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
#include <memory>
#include <string>

// External includes

// Project includes
#include "includes/model_part.h"
#include "includes/registry.h"
#include "modeler/modeler.h"
#include "utilities/string_utilities.h"

namespace Kratos {

namespace Internals { class OctreeMesherData; } // forward declaration (keeps the heavy
                                                // octree utility out of this core header)

///@name Kratos Classes
///@{

/**
 * @class OctreeMesherModeler
 * @ingroup KratosCore
 * @brief Builds a hexahedral ModelPart from a surface mesh using the hybrid octree mesher.
 * @details Mirrors the staged architecture of @ref VoxelMeshGeneratorModeler but drives the
 * @ref OctreeHybridMeshUtility engine and dispatches its components through the Kratos
 * Registry (prototype pattern) instead of hand-written factories.  `SetupModelPart` (1) builds
 * and 2:1-balances the octree from the input surface and extracts the dual (conforming) or
 * primal (leaf-hex + hanging-node constraints) mesh into shared state, then runs the
 * registered (2) colouring, (3) entity-generation and (4) operation stages in order.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeMesherModeler
    : public Modeler
{
public:
    ///@name Type Definitions
    ///@{

    KRATOS_CLASS_POINTER_DEFINITION(OctreeMesherModeler);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor (used for the registered prototype).
    OctreeMesherModeler();

    /// Constructor.
    OctreeMesherModeler(
        Model& rModel,
        Parameters ModelerParameters = Parameters());

    /// Destructor.
    ~OctreeMesherModeler() override;

    /// Creates the Modeler Pointer.
    Modeler::Pointer Create(
        Model& rModel, const Parameters ModelParameters) const override
    {
        return Kratos::make_shared<OctreeMesherModeler>(rModel, ModelParameters);
    }

    ///@}
    ///@name Stages
    ///@{

    void SetupModelPart() override;

    ///@}
    ///@name Access (used by the registered components through the passed-in modeler)
    ///@{

    /// The shared mesher state (octree, extracted hex mesh, colours, constraints).
    Internals::OctreeMesherData& GetData();

    /// The owning model.
    Model& GetModel() { return *mpModel; }

    /// Returns (creating if necessary) the model part with the given full name.
    ModelPart& CreateAndGetModelPart(const std::string& rFullName);

    /// Returns the ModelPart node for mesh-node index @p NodeIndex, creating it once
    /// (from the shared node coordinates) and caching it for reuse.
    Node::Pointer GenerateOrRetrieveNode(
        ModelPart& rModelPart,
        ModelPart::NodesContainerType& rNewNodes,
        int NodeIndex);

    /// Seeds the next entity ids from the current contents of @p rModelPart's root.
    void SetStartIds(ModelPart& rModelPart);

    std::size_t NextElementId()    { return mStartElementId++; }
    std::size_t NextConditionId()  { return mStartConditionId++; }
    std::size_t NextConstraintId() { return mStartConstraintId++; }

    ///@}
    ///@name Input and output
    ///@{

    std::string Info() const override { return "OctreeMesherModeler"; }

    void PrintInfo(std::ostream& rOStream) const override { rOStream << Info(); }

    void PrintData(std::ostream& rOStream) const override {}

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Member Variables
    ///@{

    Model* mpModel = nullptr;
    std::unique_ptr<Internals::OctreeMesherData> mpData;
    std::size_t mStartNodeId = 0;
    std::size_t mStartElementId = 0;
    std::size_t mStartConditionId = 0;
    std::size_t mStartConstraintId = 0;

    ///@}
    ///@name Private Operations
    ///@{

    /// Builds + balances the octree and extracts the (dual/primal) hex mesh into the shared data.
    void BuildOctreeAndExtract();

    /**
     * @brief Runs every stage in @p StageList by resolving its `type` against the Registry.
     * @details A `type` of four dot-separated segments is taken as a full registry path;
     * otherwise it is wrapped as `<rRegistryRoot>.All.<type>.Prototype`.  The retrieved
     * prototype is a shared stateless object, so @p Invoke passes the modeler and the
     * (defaults-assigned) stage parameters to it.
     */
    template<class TBase, class TInvoke>
    void Dispatch(const std::string& rRegistryRoot, Parameters StageList, TInvoke&& Invoke)
    {
        for (Parameters stage_params : StageList) {
            std::string type = stage_params["type"].GetString();
            const auto segments = StringUtilities::SplitStringByDelimiter(type, '.');
            const std::string full_path = (segments.size() == 4)
                ? type
                : rRegistryRoot + ".All." + type + ".Prototype";
            KRATOS_ERROR_IF_NOT(Registry::HasValue(full_path))
                << "The component '" << full_path << "' is not registered." << std::endl;
            const TBase& r_prototype = Registry::GetValue<TBase>(full_path);
            r_prototype.ValidateParameters(stage_params);
            Invoke(r_prototype, stage_params);
        }
    }

    ///@}
};

///@}
///@name Input and output
///@{

inline std::ostream& operator<<(std::ostream& rOStream, const OctreeMesherModeler& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);
    return rOStream;
}

///@}

} // namespace Kratos
