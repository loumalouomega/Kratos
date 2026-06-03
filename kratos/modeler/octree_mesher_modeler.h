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

/// Forward declaration: keeps the heavy OctreeHybridMeshUtility header out of this
/// universally-included core header (PIMPL idiom on the shared mesh data).
namespace Internals { class OctreeMesherData; }

///@name Kratos Classes
///@{

/**
 * @class OctreeMesherModeler
 * @ingroup KratosCore
 * @brief Builds an all-hexahedral `ModelPart` from a closed triangular surface mesh
 *        using the hybrid octree mesher engine.
 *
 * @details `OctreeMesherModeler` mirrors the staged architecture of
 * @ref VoxelMeshGeneratorModeler but drives the @ref OctreeHybridMeshUtility engine and
 * dispatches its components through the Kratos **Registry prototype pattern** instead of
 * hand-written factory classes.
 *
 * ### Pipeline overview
 * `SetupModelPart` executes four sequential stages:
 *
 * | Step | What happens |
 * |------|-------------|
 * | 1 | **Octree generation** (internal): builds and 2:1-balances the adaptive octree from the input surface, then extracts either the conforming *dual* hex mesh or the non-conforming *primal* leaf-hex mesh into shared state (`OctreeMesherData`). |
 * | 2 | **Colouring** (`coloring_settings_list`): classifies cells as inside (1) or outside (0) the surface. |
 * | 3 | **Entity generation** (`entities_generator_list`): emits `Element3D8N` hexes, boundary `SurfaceCondition3D4N`, and/or `LinearMasterSlaveConstraint` hanging-node constraints. |
 * | 4 | **Operations** (`model_part_operations`): post-processing passes (e.g. mesh-quality reports). |
 *
 * Each component in stages 2–4 is a Registry prototype: the modeler resolves its `"type"`
 * string to the registered prototype via `Registry::GetValue<Base>(path)` and calls the
 * stateless `const` do-work virtual with `(*this, parameters)`.  No factory files are
 * needed; components self-register at static-init time through
 * `KRATOS_REGISTRY_ADD_PROTOTYPE`.
 *
 * ### Mesh topologies
 * The `"mesh_type"` key in the `"octree_generator"` block selects the topology:
 * - **`"dual"`** (default): a fully *conforming* mesh produced by the dual extraction +
 *   transition-template algorithm (see @ref OctreeHybridMeshUtility).  No hanging nodes.
 * - **`"primal"`**: one hexahedron per octree leaf, sharing finest-grid corner nodes.
 *   Non-conforming at 2:1 transition faces; hanging nodes on those faces are tied to
 *   the four (or two) coarse-face master corners by bilinear weights using
 *   `LinearMasterSlaveConstraint` objects.
 *
 * ### Shared state
 * All components access the mesh data through `GetData()`, which returns a reference to
 * the internal @ref Internals::OctreeMesherData struct.  The PIMPL pattern keeps the
 * heavy `octree_hybrid_mesh_utility.h` header out of this file.
 *
 * ### Minimal Python usage
 * @code{.python}
 * import KratosMultiphysics as KM
 *
 * model = KM.Model()
 * # ... populate "Surface" ModelPart from an STL ...
 * settings = KM.Parameters('''{
 *     "input_model_part_name"  : "Surface",
 *     "output_model_part_name" : "Volume",
 *     "octree_generator"  : { "refinement_depth": 5 },
 *     "coloring_settings_list" : [{ "type": "ClassifyCellsInsideOutside" }],
 *     "entities_generator_list": [{ "type": "GenerateHexesByCellColor",
 *                                   "model_part_name": "Volume", "color": 1 }],
 *     "model_part_operations"  : [{ "type": "ReportMeshQuality",
 *                                   "model_part_name": "Volume" }]
 * }''')
 * mod = KM.OctreeMesherModeler(model, settings)
 * mod.SetupModelPart()
 * @endcode
 *
 * @note The modeler is registered with `KRATOS_REGISTER_MODELER("OctreeMesherModeler", …)`
 *       and can therefore also be instantiated via
 *       `KratosModelParametersFactory.ConstructListOfItems(modelers_list)` from a JSON
 *       `"modelers"` array entry with `"name": "KratosMultiphysics.OctreeMesherModeler"`.
 *
 * @see OctreeHybridMeshUtility
 * @see VoxelMeshGeneratorModeler
 * @see Internals::OctreeMesherData
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeMesherModeler
    : public Modeler
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of OctreeMesherModeler.
    KRATOS_CLASS_POINTER_DEFINITION(OctreeMesherModeler);

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Default constructor.
     * @details Used internally to create the prototype instance that is stored in the
     *          `KRATOS_REGISTER_MODELER` factory and cloned whenever the modeler is
     *          constructed from a JSON `"modelers"` list.  Direct use is not intended;
     *          prefer the `(Model&, Parameters)` constructor.
     */
    OctreeMesherModeler();

    /**
     * @brief Main constructor.
     * @param rModel              The owning `Model`; model parts are created or retrieved
     *                            from it during `SetupModelPart`.
     * @param ModelerParameters   JSON configuration block.  Keys are validated and defaults
     *                            assigned via `GetDefaultParameters()` on construction.
     */
    OctreeMesherModeler(
        Model& rModel,
        Parameters ModelerParameters = Parameters());

    /**
     * @brief Destructor.
     * @details Defined out-of-line so that the `unique_ptr<OctreeMesherData>` (incomplete
     *          type in the header) is correctly destroyed.
     */
    ~OctreeMesherModeler() override;

    /**
     * @brief Factory method required by the `Modeler` base-class contract.
     * @param rModel          Target model.
     * @param ModelParameters JSON parameters forwarded to the new instance.
     * @return A `shared_ptr` to a freshly constructed `OctreeMesherModeler`.
     */
    Modeler::Pointer Create(
        Model& rModel, const Parameters ModelParameters) const override
    {
        return Kratos::make_shared<OctreeMesherModeler>(rModel, ModelParameters);
    }

    ///@}
    ///@name Modeler stages
    ///@{

    /**
     * @brief Runs the full meshing pipeline.
     * @details Executes the four pipeline stages in order:
     * 1. @ref BuildOctreeAndExtract — octree construction, 2:1 balancing, dual/primal
     *    mesh extraction (and optional surface projection).
     * 2. Dispatch over `coloring_settings_list` via @ref OctreeMesherColoring.
     * 3. Dispatch over `entities_generator_list` via @ref OctreeMesherEntityGeneration.
     * 4. Dispatch over `model_part_operations` via @ref OctreeMesherOperation.
     */
    void SetupModelPart() override;

    ///@}
    ///@name Component access API
    ///@{
    /// These methods are part of the public interface that registered components call
    /// through the `OctreeMesherModeler&` reference passed to their do-work virtuals.

    /**
     * @brief Returns the shared mesh data struct.
     * @details Provides read/write access to the @ref Internals::OctreeMesherData that
     *          holds the octree pointer, extracted node/cell arrays, per-cell colours,
     *          hanging-node constraint descriptors, and the node-pointer cache.
     * @return Reference to the internal `OctreeMesherData`.
     */
    Internals::OctreeMesherData& GetData();

    /**
     * @brief Returns the owning model.
     * @return Reference to the `Model` that was passed to the constructor.
     */
    Model& GetModel() { return *mpModel; }

    /**
     * @brief Returns or creates the ModelPart identified by @p rFullName.
     * @details Queries `Model::HasModelPart`; creates via `Model::CreateModelPart` if
     *          absent.  Supports nested names (e.g. `"Root.SubPart"`).
     * @param rFullName Fully qualified ModelPart name.
     * @return Non-const reference to the requested ModelPart.
     */
    ModelPart& CreateAndGetModelPart(const std::string& rFullName);

    /**
     * @brief Returns (or creates) the Kratos `Node` corresponding to the mesh-node at
     *        index @p NodeIndex in the shared `OctreeMesherData::mNodes` array.
     * @details On the first call for a given @p NodeIndex the node is created with the
     *          world-space coordinates `mNodes[NodeIndex]`, assigned the ModelPart's
     *          variable list and buffer size, cached in `mNodePtrs[NodeIndex]`, and
     *          appended to @p rNewNodes.  Subsequent calls for the same index return
     *          the cached pointer without touching @p rNewNodes, so the same node can
     *          be shared by multiple elements or conditions.
     * @param rModelPart  ModelPart that will own the node (solution-step variable list
     *                    and buffer size are copied from it).
     * @param rNewNodes   Accumulator for newly created nodes; handed to
     *                    `ModelPartUtils::AddNodesFromOrderedContainer` after the loop.
     * @param NodeIndex   Zero-based index into `OctreeMesherData::mNodes`.
     * @return `Node::Pointer` to the (possibly newly created) node.
     */
    Node::Pointer GenerateOrRetrieveNode(
        ModelPart& rModelPart,
        ModelPart::NodesContainerType& rNewNodes,
        int NodeIndex);

    /**
     * @brief Synchronises the internal ID counters with the current state of
     *        @p rModelPart's root model part.
     * @details Computes the first available ID for nodes, elements, conditions, and
     *          master-slave constraints from the root model part's existing entity
     *          arrays, then keeps the running maximum so IDs are never reused across
     *          multiple generator calls.  Must be called at the start of each entity-
     *          generation stage that creates new entities.
     * @param rModelPart Any sub-part of the hierarchy; the root is resolved internally.
     */
    void SetStartIds(ModelPart& rModelPart);

    /**
     * @brief Returns and advances the next unique element ID.
     * @return The next available element ID (post-incremented).
     */
    std::size_t NextElementId()    { return mStartElementId++; }

    /**
     * @brief Returns and advances the next unique condition ID.
     * @return The next available condition ID (post-incremented).
     */
    std::size_t NextConditionId()  { return mStartConditionId++; }

    /**
     * @brief Returns and advances the next unique master-slave constraint ID.
     * @return The next available constraint ID (post-incremented).
     */
    std::size_t NextConstraintId() { return mStartConstraintId++; }

    ///@}
    ///@name Input and output
    ///@{

    /// @return The string `"OctreeMesherModeler"`.
    std::string Info() const override { return "OctreeMesherModeler"; }

    /// Prints `Info()` to @p rOStream.
    void PrintInfo(std::ostream& rOStream) const override { rOStream << Info(); }

    /// No data to print; provided to satisfy the `Modeler` interface.
    void PrintData(std::ostream& rOStream) const override {}

    /**
     * @brief Returns the default parameter schema for this modeler.
     * @details Full schema:
     * @code{.json}
     * {
     *     "echo_level"              : 0,
     *     "input_model_part_name"   : "",
     *     "output_model_part_name"  : "",
     *     "octree_generator" : {
     *         "type"                  : "generate_octree_from_surface",
     *         "input_model_part_name" : "",
     *         "refinement_depth"      : 5,
     *         "adaptive"              : true,
     *         "mesh_type"             : "dual",
     *         "project_to_surface"    : false,
     *         "projection_iterations" : 20000,
     *         "projection_smoothing"  : 1000
     *     },
     *     "coloring_settings_list"   : [],
     *     "entities_generator_list"  : [],
     *     "model_part_operations"    : []
     * }
     * @endcode
     * @return Parameters object with all keys set to their defaults.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Member Variables
    ///@{

    /// Pointer to the owning Model (non-owning; lifetime guaranteed by the caller).
    Model* mpModel = nullptr;

    /// PIMPL handle to the shared mesh data.  Defined out-of-line because
    /// `OctreeMesherData` includes `octree_hybrid_mesh_utility.h` which is heavy.
    std::unique_ptr<Internals::OctreeMesherData> mpData;

    /// Running node ID counter (seeded from the root ModelPart by SetStartIds).
    std::size_t mStartNodeId = 0;

    /// Running element ID counter.
    std::size_t mStartElementId = 0;

    /// Running condition ID counter.
    std::size_t mStartConditionId = 0;

    /// Running master-slave constraint ID counter.
    std::size_t mStartConstraintId = 0;

    ///@}
    ///@name Private Operations
    ///@{

    /**
     * @brief Builds and 2:1-balances the octree, then extracts the hex mesh.
     * @details Reads `mParameters["octree_generator"]`, resolves the input surface
     *          ModelPart, builds the adaptive (or uniform) octree via
     *          `OctreeHybridMeshUtility::BuildFromSurfaceMesh`, calls
     *          `StrongConstrain2To1`, and then:
     *          - **`mesh_type == "dual"`**: calls `ExtractDualHexMesh` and optionally
     *            `RemoveOutsideElement` + `ClearBufferZone` + `ProjectToIsoSurface`
     *            when `"project_to_surface"` is true.
     *          - **`mesh_type == "primal"`**: calls `ExtractPrimalHexMesh` which also
     *            fills `OctreeMesherData::mHanging` with the 2:1 transition constraints.
     *          After extraction, `mNodePtrs` is resized and zeroed.
     */
    void BuildOctreeAndExtract();

    /**
     * @brief Resolves a stage-list from JSON and invokes @p Invoke on each prototype.
     * @details For each entry in @p StageList:
     * 1. Reads `"type"` and, if it is not already a four-segment dot-separated Registry
     *    path, prepends `<rRegistryRoot>.All.` and appends `.Prototype`.
     * 2. Checks `Registry::HasValue`; errors if missing.
     * 3. Retrieves the stateless prototype via `Registry::GetValue<TBase>`.
     * 4. Calls `prototype.ValidateParameters(stage_params)` in-place.
     * 5. Calls `Invoke(prototype, stage_params)`.
     *
     * @tparam TBase    Base component type (`OctreeMesherColoring`,
     *                  `OctreeMesherEntityGeneration`, or `OctreeMesherOperation`).
     * @tparam TInvoke  Callable with signature `(const TBase&, Parameters)`.
     * @param rRegistryRoot  Root path prefix (e.g. `"OctreeMesherColoring"`).
     * @param StageList      Iterable JSON array of stage-parameter objects.
     * @param Invoke         Lambda that calls the component's do-work virtual.
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

/**
 * @brief Stream insertion operator for @ref OctreeMesherModeler.
 * @param rOStream Output stream.
 * @param rThis    Modeler whose `Info()` string is written.
 * @return Reference to @p rOStream for chaining.
 */
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeMesherModeler& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);
    return rOStream;
}

///@}

} // namespace Kratos
