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
#include "containers/model.h"
#include "includes/registry.h"
#include "modeler/modeler.h"
#include "utilities/string_utilities.h"

namespace Kratos {

/// Forward declaration: keeps the heavy OctreeHybridMeshUtility header out of this
/// universally-included core header (PIMPL idiom on the shared mesh data).
namespace Internals { class OctreeHybridMesherData; }

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridMesherModeler
 * @ingroup KratosCore
 * @brief Builds an all-hexahedral `ModelPart` from a closed triangular surface mesh
 *        using the hybrid octree mesher engine.
 *
 * @details `OctreeHybridMesherModeler` mirrors the staged architecture of
 * @ref VoxelMeshGeneratorModeler but drives the @ref OctreeHybridMeshUtility engine and
 * dispatches its components through the Kratos **Registry prototype pattern** instead of
 * hand-written factory classes.
 *
 * ### Pipeline overview
 * `SetupModelPart` delegates to four sequential private methods:
 *
 * | Step | Private method | What happens |
 * |------|---------------|-------------|
 * | 1 | @ref ExecuteRefinementOperations | Dispatches each @ref OctreeHybridRefineOperation in `refine_operations_list` (the first must be @ref OctreeHybridRefineInterfaceCells, which builds the initial octree and records `mesh_type` / projection settings; further entries add deeper refinement).  Then calls `StrongConstrain2To1` and extracts the conforming *dual* or non-conforming *primal* hex mesh.  For `"dual"` with `project_to_surface`, also runs `RemoveOutsideElement`, `ClearBufferZone`, and `ProjectToIsoSurface`. |
 * | 2 | @ref ExecuteColoringOperations | Dispatches `coloring_settings_list`; classifies cells as inside (1) or outside (0) the surface. |
 * | 3 | @ref ExecuteEntityGenerationOperations | Dispatches `entities_generator_list`; emits `Element3D8N` hexes, boundary `SurfaceCondition3D4N`, and/or `LinearMasterSlaveConstraint` hanging-node constraints. |
 * | 4 | @ref ExecuteModelPartOperations | Dispatches `model_part_operations`; post-processing passes (e.g. mesh-quality reports). |
 *
 * Every component in `refine_operations_list` and stages 3–5 is a Registry prototype: the
 * modeler resolves its `"type"` string to the registered prototype via
 * `Registry::GetValue<Base>(path)` and calls the stateless `const` do-work virtual with
 * `(*this, parameters)`.  No factory files are needed; components self-register at
 * static-init time through `KRATOS_REGISTRY_ADD_PROTOTYPE`.
 *
 * ### Mesh topologies
 * The `"mesh_type"` key in the first @ref OctreeHybridRefineInterfaceCells operation selects the topology:
 * - **`"dual"`** (default): a fully *conforming* mesh produced by the dual extraction +
 *   transition-template algorithm (see @ref OctreeHybridMeshUtility).  No hanging nodes.
 * - **`"primal"`**: one hexahedron per octree leaf, sharing finest-grid corner nodes.
 *   Non-conforming at 2:1 transition faces; hanging nodes on those faces are tied to
 *   the four (or two) coarse-face master corners by bilinear weights using
 *   `LinearMasterSlaveConstraint` objects.
 *
 * ### Shared state
 * All components access the mesh data through `GetData()`, which returns a reference to
 * the internal @ref Internals::OctreeHybridMesherData struct.  The PIMPL pattern keeps the
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
 *     "refine_operations_list" : [
 *         { "type": "OctreeHybridRefineInterfaceCells",
 *           "refinement_depth": 3 },
 *         { "type": "OctreeHybridRefineInterfaceCells",
 *           "input_model_part_name": "Surface",
 *           "refinement_depth": 5 }
 *     ],
 *     "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
 *     "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
 *                                   "model_part_name": "Volume", "color": 1 }],
 *     "model_part_operations"  : [{ "type": "OctreeHybridReportMeshQuality",
 *                                   "model_part_name": "Volume" }]
 * }''')
 * mod = KM.OctreeHybridMesherModeler(model, settings)
 * mod.SetupModelPart()
 * @endcode
 *
 * @note The modeler is registered with `KRATOS_REGISTER_MODELER("OctreeHybridMesherModeler", …)`
 *       and can therefore also be instantiated via
 *       `KratosModelParametersFactory.ConstructListOfItems(modelers_list)` from a JSON
 *       `"modelers"` array entry with `"name": "KratosMultiphysics.OctreeHybridMesherModeler"`.
 *
 * @see OctreeHybridMeshUtility
 * @see VoxelMeshGeneratorModeler
 * @see Internals::OctreeHybridMesherData
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridMesherModeler
    : public Modeler
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of OctreeHybridMesherModeler.
    KRATOS_CLASS_POINTER_DEFINITION(OctreeHybridMesherModeler);

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
    OctreeHybridMesherModeler();

    /**
     * @brief Main constructor.
     * @param rModel              The owning `Model`; model parts are created or retrieved
     *                            from it during `SetupModelPart`.
     * @param ModelerParameters   JSON configuration block.  Keys are validated and defaults
     *                            assigned via `GetDefaultParameters()` on construction.
     */
    OctreeHybridMesherModeler(
        Model& rModel,
        Parameters ModelerParameters = Parameters());

    /**
     * @brief Destructor.
     * @details Defined out-of-line so that the `unique_ptr<OctreeHybridMesherData>` (incomplete
     *          type in the header) is correctly destroyed.
     */
    ~OctreeHybridMesherModeler() override;

    /**
     * @brief Factory method required by the `Modeler` base-class contract.
     * @param rModel          Target model.
     * @param ModelParameters JSON parameters forwarded to the new instance.
     * @return A `shared_ptr` to a freshly constructed `OctreeHybridMesherModeler`.
     */
    Modeler::Pointer Create(
        Model& rModel, const Parameters ModelParameters) const override
    {
        return Kratos::make_shared<OctreeHybridMesherModeler>(rModel, ModelParameters);
    }

    ///@}
    ///@name Modeler stages
    ///@{

    /**
     * @brief Runs the full meshing pipeline.
     * @details Delegates to four sequential private stage methods:
     * 1. @ref ExecuteRefinementOperations — dispatches `refine_operations_list`, then
     *    balances and extracts the hex mesh.
     * 2. @ref ExecuteColoringOperations — dispatches `coloring_settings_list`.
     * 3. @ref ExecuteEntityGenerationOperations — dispatches `entities_generator_list`.
     * 4. @ref ExecuteModelPartOperations — dispatches `model_part_operations`.
     */
    void SetupModelPart() override;

    ///@}
    ///@name Component access API
    ///@{
    /// These methods are part of the public interface that registered components call
    /// through the `OctreeHybridMesherModeler&` reference passed to their do-work virtuals.

    /**
     * @brief Returns the shared mesh data struct.
     * @details Provides read/write access to the @ref Internals::OctreeHybridMesherData that
     *          holds the octree pointer, extracted node/cell arrays, per-cell colours,
     *          hanging-node constraint descriptors, and the node-pointer cache.
     * @return Reference to the internal `OctreeHybridMesherData`.
     */
    Internals::OctreeHybridMesherData& GetData();

    /**
     * @brief Returns the owning model.
     * @return Reference to the `Model` that was passed to the constructor.
     */
    Model& GetModel() { return *mpModel; }

    /**
     * @brief Returns the top-level `input_model_part_name` from the modeler parameters.
     * @details Used by @ref OctreeHybridRefineInterfaceCells as a fallback when its own
     *          `input_model_part_name` is empty on the first call (octree build).
     * @return The model part name string (may be empty if not set).
     */
    std::string GetInputModelPartName() const { return mParameters["input_model_part_name"].GetString(); }

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
     *        index @p NodeIndex in the shared `OctreeHybridMesherData::mNodes` array.
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
     * @param NodeIndex   Zero-based index into `OctreeHybridMesherData::mNodes`.
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

    /// @return The string `"OctreeHybridMesherModeler"`.
    std::string Info() const override { return "OctreeHybridMesherModeler"; }

    /// Prints `Info()` to @p rOStream.
    void PrintInfo(std::ostream& rOStream) const override { rOStream << Info(); }

    /// No data to print; provided to satisfy the `Modeler` interface.
    void PrintData(std::ostream& rOStream) const override {}

    /**
     * @brief Returns the default parameter schema for this modeler.
     * @details Full schema:
     * @code{.json}
     * {
     *     "echo_level"             : 0,
     *     "input_model_part_name"  : "",
     *     "output_model_part_name" : "",
     *     "refine_operations_list"  : [],
     *     "coloring_settings_list"  : [],
     *     "entities_generator_list" : [],
     *     "model_part_operations"   : []
     * }
     * @endcode
     * The `refine_operations_list` must start with an @ref OctreeHybridRefineInterfaceCells
     * entry (which builds the octree and records `mesh_type` / projection settings) and may be
     * followed by any number of @ref OctreeHybridRefineUniform or additional
     * @ref OctreeHybridRefineInterfaceCells entries.
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
    /// `OctreeHybridMesherData` includes `octree_hybrid_mesh_utility.h` which is heavy.
    std::unique_ptr<Internals::OctreeHybridMesherData> mpData;

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
     * @brief Dispatches `refine_operations_list`, then balances and extracts the hex mesh.
     * @details Calls @ref OctreeHybridRefineOperation::Refine on every entry (the first must
     *          be @ref OctreeHybridRefineInterfaceCells).  After all refinement passes, calls
     *          `StrongConstrain2To1` and then:
     *          - **`mMeshType == "dual"`**: `ExtractDualHexMesh`; optionally
     *            `RemoveOutsideElement` + `ClearBufferZone` + `ProjectToIsoSurface`.
     *          - **`mMeshType == "primal"`**: `ExtractPrimalHexMesh`, which also fills
     *            `OctreeHybridMesherData::mHanging` with 2:1 transition constraints.
     *          Initialises `mNodePtrs` to null after extraction.
     */
    void ExecuteRefinementOperations();

    /**
     * @brief Dispatches `coloring_settings_list` via @ref OctreeHybridMesherColoring.
     * @details Each entry's `"type"` is resolved to a registered prototype and its
     *          `Apply(*this, parameters)` virtual is called in list order.
     */
    void ExecuteColoringOperations();

    /**
     * @brief Dispatches `entities_generator_list` via @ref OctreeHybridMesherEntityGeneration.
     * @details Each entry's `"type"` is resolved to a registered prototype and its
     *          `Generate(*this, parameters)` virtual is called in list order.
     */
    void ExecuteEntityGenerationOperations();

    /**
     * @brief Dispatches `model_part_operations` via @ref OctreeHybridMesherOperation.
     * @details Each entry's `"type"` is resolved to a registered prototype and its
     *          `Execute(*this, parameters)` virtual is called in list order.
     */
    void ExecuteModelPartOperations();

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
     * @tparam TBase    Base component type (`OctreeHybridMesherColoring`,
     *                  `OctreeHybridMesherEntityGeneration`, or `OctreeHybridMesherOperation`).
     * @tparam TInvoke  Callable with signature `(const TBase&, Parameters)`.
     * @param rRegistryRoot  Root path prefix (e.g. `"OctreeHybridMesherColoring"`).
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
 * @brief Stream insertion operator for @ref OctreeHybridMesherModeler.
 * @param rOStream Output stream.
 * @param rThis    Modeler whose `Info()` string is written.
 * @return Reference to @p rOStream for chaining.
 */
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeHybridMesherModeler& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);
    return rOStream;
}

///@}

} // namespace Kratos
