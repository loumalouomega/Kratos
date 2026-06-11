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
#include "includes/define_registry.h"
#include "modeler/modeler.h"
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"
#include "utilities/string_utilities.h"

namespace Kratos 
{
///@name Type Definitions
///@{

/// Forward declaration: keeps the heavy OctreeHybridMeshUtility header out of this
/// universally-included core header (PIMPL idiom on the shared mesh data).
namespace Internals { class OctreeHybridMesherData; }

///@}
///@name Kratos Classes
///@{

/**
 * @class OctreeHybridMeshGeneratorModeler
 * @ingroup KratosCore
 * @brief Builds an all-hexahedral `ModelPart` from a closed triangular surface mesh
 *        using the hybrid octree mesher engine.
 *
 * @details `OctreeHybridMeshGeneratorModeler` mirrors the staged architecture of
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
 *     "refine_operations_list" : [
 *         { "type": "OctreeHybridRefineInterfaceCells",
 *           "refinement_depth": 3 },
 *         { "type": "OctreeHybridRefineInterfaceCells",
 *           "input_model_part_name": "Surface",
 *           "refinement_depth": 5 }
 *     ],
 *     "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
 *     "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
 *                                   "model_part_name": "Volume", "color": 1 }],
 *     "model_part_operations"  : [{ "type": "OctreeHybridReportMeshQuality",
 *                                   "model_part_name": "Volume" }]
 * }''')
 * mod = KM.OctreeHybridMeshGeneratorModeler(model, settings)
 * mod.SetupModelPart()
 * @endcode
 *
 * @note The modeler self-registers in the Kratos `Registry` under
 *       `"Modelers.KratosMultiphysics.OctreeHybridMeshGeneratorModeler"` and
 *       `"Modelers.All.OctreeHybridMeshGeneratorModeler"` via `KRATOS_REGISTRY_ADD_PROTOTYPE`.
 *       It can also be instantiated via
 *       `KratosModelParametersFactory.ConstructListOfItems(modelers_list)` from a JSON
 *       `"modelers"` array entry with `"name": "KratosMultiphysics.OctreeHybridMeshGeneratorModeler"`,
 *       which resolves the class directly from the `KratosMultiphysics` Python module.
 *
 * @see OctreeHybridMeshUtility
 * @see VoxelMeshGeneratorModeler
 * @see Internals::OctreeHybridMesherData
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridMeshGeneratorModeler
    : public Modeler
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of OctreeHybridMeshGeneratorModeler.
    KRATOS_CLASS_POINTER_DEFINITION(OctreeHybridMeshGeneratorModeler);

    /// Index type for mesh nodes, cells, etc.  Matches the index type used by OctreeHybridMeshUtility.
    using IndexType = std::size_t;

    /// Add to the Kratos registry
    KRATOS_REGISTRY_ADD_PROTOTYPE("Modelers.KratosMultiphysics", Modeler, OctreeHybridMeshGeneratorModeler)
    KRATOS_REGISTRY_ADD_PROTOTYPE("Modelers.All", Modeler, OctreeHybridMeshGeneratorModeler)

    /// Define operation enum
    enum class OperationType
    {
        UNDEFINED,
        Refine,
        Coloring,
        GenerateEntities,
        ModelPartOperation
    };

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Default constructor.
     * @details Used internally to create the prototype instance registered in the Kratos
     *          `Registry` via `KRATOS_REGISTRY_ADD_PROTOTYPE` and cloned (through `Create`)
     *          whenever the modeler is constructed from a JSON `"modelers"` list.  Direct
     *          use is not intended; prefer the `(Model&, Parameters)` constructor.
     */
    OctreeHybridMeshGeneratorModeler();

    /**
     * @brief Main constructor.
     * @param rModel              The owning `Model`; model parts are created or retrieved
     *                            from it during `SetupModelPart`.
     * @param ModelerParameters   JSON configuration block.  Keys are validated and defaults
     *                            assigned via `GetDefaultParameters()` on construction.
     */
    OctreeHybridMeshGeneratorModeler(
        Model& rModel,
        Parameters ModelerParameters = Parameters()
        );

    /**
     * @brief Destructor.
     * @details Defined out-of-line so that the `unique_ptr<OctreeHybridMesherData>` (incomplete
     *          type in the header) is correctly destroyed.
     */
    ~OctreeHybridMeshGeneratorModeler() override;

    /**
     * @brief Factory method required by the `Modeler` base-class contract.
     * @param rModel          Target model.
     * @param ModelParameters JSON parameters forwarded to the new instance.
     * @return A `shared_ptr` to a freshly constructed `OctreeHybridMeshGeneratorModeler`.
     */
    Modeler::Pointer Create(
        Model& rModel, const Parameters ModelParameters) const override
    {
        return Kratos::make_shared<OctreeHybridMeshGeneratorModeler>(rModel, ModelParameters);
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
     *
     * Afterwards, iterates `output_files` and calls @ref WriteOctreeVTK for every entry
     * whose `"type"` is `"octree_vtk"` (pure debug output).
     */
    void SetupModelPart() override;

    /**
     * @brief Read the model parts
     */
    void ReadModelParts();

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Initializes the octree hybrid meshing modeler.
     */
    void Initialize();

    /**
     * @brief Returns the default parameter schema for this modeler.
     * @details Full schema:
     * @code{.json}
     * {
     *     "refine_operations_list"  : [],
     *     "coloring_settings_list"  : [],
     *     "entities_generator_list" : [],
     *     "model_part_operations"   : [],
     *     "mdpa_file_name"          : "",
     *     "input_model_part_name"   : "",
     *     "default_outside_color"   : 1,
     *     "output_files"            : [],
     *     "remove_orphan_nodes"     : true,
     *     "echo_level"              : 1
     * }
     * @endcode
     * The `refine_operations_list` must start with an @ref OctreeHybridRefineInterfaceCells
     * entry (which builds the octree and records `mesh_type` / projection settings) and may be
     * followed by any number of @ref OctreeHybridRefineUniform or additional
     * @ref OctreeHybridRefineInterfaceCells entries.
     *
     * Each entry of `output_files` is a debug-output request with a `"type"` key. The only
     * currently supported type is `"octree_vtk"`, which dumps the raw octree leaves via
     * @ref WriteOctreeVTK and requires a `"file_name"` key, e.g.
     * `{ "type": "octree_vtk", "file_name": "octree.vtk" }`.
     * @return Parameters object with all keys set to their defaults.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
    ///@name Component access API
    ///@{

    /// These methods are part of the public interface that registered components call
    /// through the `OctreeHybridMeshGeneratorModeler&` reference passed to their do-work virtuals.

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
    Model& GetModel() 
    { 
        return *mpModel; 
    }

    /**
     * @brief Returns the input model part
     * @return The input model part
     */
    ModelPart& GetInputModelPart()
    {
        KRATOS_ERROR_IF_NOT(mpInputModelPart) << "Input model part not set.  Ensure that \"input_model_part_name\" is set in the parameters and that the model part exists in the Model." << std::endl;
        return *mpInputModelPart;
    }

    /**
     * @brief Returns the top-level `input_model_part_name` from the modeler parameters.
     * @details Used by @ref OctreeHybridRefineInterfaceCells as a fallback when its own
     *          `input_model_part_name` is empty on the first call (octree build).
     * @return The model part name string (may be empty if not set).
     */
    std::string GetInputModelPartName() const 
    { 
        return mpInputModelPart ? mpInputModelPart->FullName() : mParameters["input_model_part_name"].GetString(); 
    }

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
        IndexType NodeIndex
        );

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
     * @brief Overrides the running node ID counter, if @p Id is non-zero.
     * @details Used by entity-generation stages to honour an explicit
     *          `"initial_node_id"` parameter.  A value of `0` means "keep the
     *          value computed by @ref SetStartIds" and is therefore a no-op.
     * @param Id Requested first node ID, or `0` to leave the counter untouched.
     */
    void OverrideStartNodeId(IndexType Id) 
    { 
        if (Id > 0) {
            mStartNodeId = Id;
        }
    }

    /**
     * @brief Overrides the running element ID counter, if @p Id is non-zero.
     * @details Used by entity-generation stages to honour an explicit
     *          `"initial_element_id"` parameter.  A value of `0` means "keep the
     *          value computed by @ref SetStartIds" and is therefore a no-op.
     * @param Id Requested first element ID, or `0` to leave the counter untouched.
     */
    void OverrideStartElementId(IndexType Id) 
    { 
        if (Id > 0) {
            mStartElementId = Id;
        }
    }

    /**
     * @brief Overrides the running condition ID counter, if @p Id is non-zero.
     * @details Used by entity-generation stages to honour an explicit
     *          `"initial_condition_id"` parameter.  A value of `0` means "keep the
     *          value computed by @ref SetStartIds" and is therefore a no-op.
     * @param Id Requested first condition ID, or `0` to leave the counter untouched.
     */
    void OverrideStartConditionId(IndexType Id) 
    { 
        if (Id > 0) {
            mStartConditionId = Id;
        }
    }

    /**
     * @brief Overrides the running master-slave constraint ID counter, if @p Id is non-zero.
     * @details Used by entity-generation stages to honour an explicit
     *          `"initial_constraint_id"` parameter.  A value of `0` means "keep the
     *          value computed by @ref SetStartIds" and is therefore a no-op.
     * @param Id Requested first constraint ID, or `0` to leave the counter untouched.
     */
    void OverrideStartConstraintId(IndexType Id) 
    { 
        if (Id > 0) {
            mStartConstraintId = Id;
        }
    }

    /**
     * @brief Returns and advances the next unique element ID.
     * @return The next available element ID (post-incremented).
     */
    IndexType NextElementId()
    { 
        return mStartElementId++; 
    }

    /**
     * @brief Returns and advances the next unique condition ID.
     * @return The next available condition ID (post-incremented).
     */
    IndexType NextConditionId()  
    { 
        return mStartConditionId++; 
    }

    /**
     * @brief Returns and advances the next unique master-slave constraint ID.
     * @return The next available constraint ID (post-incremented).
     */
    IndexType NextConstraintId() 
    { 
        return mStartConstraintId++; 
    }

    ///@}
    ///@name Input and output
    ///@{

    /// @return The string `"OctreeHybridMeshGeneratorModeler"`.
    std::string Info() const override 
    { 
        return "OctreeHybridMeshGeneratorModeler"; 
    }

    /// Prints `Info()` to @p rOStream.
    void PrintInfo(std::ostream& rOStream) const override 
    { 
        rOStream << Info(); 
    }

    /// No data to print; provided to satisfy the `Modeler` interface.
    void PrintData(std::ostream& rOStream) const override 
    {
        rOStream << "No data to print.";
    }

    ///@}
private:
    ///@name Member Variables
    ///@{

    /// Pointer to the owning Model (non-owning; lifetime guaranteed by the caller).
    Model* mpModel = nullptr;

    /// The input model part
    ModelPart* mpInputModelPart = nullptr;

    /// The names of the root model parts to be used in the output (for some post-process operations)
    std::unordered_set<std::string> mRootModelPartsNames; 

    /// PIMPL handle to the shared mesh data.  Defined out-of-line because
    /// `OctreeHybridMesherData` includes `octree_hybrid_mesh_utility.h` which is heavy.
    std::unique_ptr<Internals::OctreeHybridMesherData> mpData;

    /// Running node ID counter (seeded from the root ModelPart by SetStartIds).
    IndexType mStartNodeId = 0;

    /// Running element ID counter.
    IndexType mStartElementId = 0;

    /// Running condition ID counter.
    IndexType mStartConditionId = 0;

    /// Running master-slave constraint ID counter.
    IndexType mStartConstraintId = 0;

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
     * @brief  Writes the entire octree leaves as hex mesh in VTK format
     * @details Pure debug/visualization helper, called from @ref SetupModelPart for every
     *          entry of `output_files` whose `"type"` is `"octree_vtk"`. Delegates to
     *          @ref OctreeHybridMeshUtility::WritePrimalVtk, which writes the raw,
     *          non-conforming primal octree leaves (one hexahedron per leaf cell, with a
     *          cell-data field `"level"` encoding the refinement level) of `GetData().mpOctree`.
     * @param ThisParameters Parameters for the output. Required key: `"file_name"` (output .vtk path).
     */
    void WriteOctreeVTK(Parameters ThisParameters);

    /**
     * @brief Generates a percentage bar string based on the given percentage value.
     * @param Percentage The percentage value for which the bar string is to be generated.
     * @return A string representing the percentage bar.
     */
    std::string GeneratePercentageBar(const double Percentage);

    /**
     * @brief Generates a start message string based on the given operation type.
     * @param Operation The type of operation for which the start message is to be generated.
     * @return A string representing the start message for the specified operation.
     */
    std::string GenerateStartMessage(const OperationType Operation)
    {
        switch (Operation) {
            case OperationType::Refine:
                return "Refinement operation";
            case OperationType::Coloring:
                return "Coloring operation";
            case OperationType::GenerateEntities:
                return "Entity generation operation";
            case OperationType::ModelPartOperation:
                return "Model part operation";
            default:
                return "Operation";
        }
    }

    /**
     * @brief Generates an end message string based on the given operation type.
     * @param Operation The type of operation for which the end message is to be generated.
     * @return A string representing the end message for the specified operation.
     */
    std::string GenerateEndMessage(const OperationType Operation)
    {
        switch (Operation) {
            case OperationType::Refine:
                return "Refinement operation";
            case OperationType::Coloring:
                return "Coloring operation";
            case OperationType::GenerateEntities:
                return "Entity generation operation";
            case OperationType::ModelPartOperation:
                return "Model part operation";
            default:
                return "Operation";
        }
    }

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
    void Dispatch(
        const std::string& rRegistryRoot,
        Parameters StageList,
        TInvoke&& Invoke,
        const OperationType Operation = OperationType::UNDEFINED
        )
    {
        // Define the messages to be printed at the start and end of each operation, depending on the echo level
        const std::string start_message = GenerateStartMessage(Operation);
        const std::string end_message = GenerateEndMessage(Operation);

        // Saving the start ids (4 integers, node, element, condition, constraint)
        std::unordered_map<std::string, std::array<std::size_t, 4>> start_ids;

        // Iterate over the operations parameters
        const unsigned int number_of_operations = StageList.size();
        unsigned int operation_counter = 0;
        for (Parameters stage_params : StageList) {
            // Resolve the component from the registry
            std::string type = stage_params["type"].GetString();
            const auto segments = StringUtilities::SplitStringByDelimiter(type, '.');
            const std::string full_path = (segments.size() == 4) ? type : rRegistryRoot + ".All." + type + ".Prototype";
            KRATOS_ERROR_IF_NOT(Registry::HasValue(full_path)) << "The component '" << full_path << "' is not registered." << std::endl;
            const TBase& r_prototype = Registry::GetValue<TBase>(full_path);

            // Get the default parameters of the component (for potential use in specific pre-processing steps)
            Parameters default_parameters = r_prototype.GetDefaultParameters();

            // Defining the model part name if not defined and the component supports it
            if (default_parameters.Has("model_part_name") && !stage_params.Has("model_part_name")) {
                stage_params.AddString("model_part_name", GetInputModelPartName());
            }

            // Print the operation parameters
            ++operation_counter;
            KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << start_message << ", number: " << operation_counter << " with the following parameters:\n" << stage_params << "\n" << GeneratePercentageBar(static_cast<double>(operation_counter) / static_cast<double>(number_of_operations)) << std::endl;

            // Validate the parameters in-place against the component's defaults
            r_prototype.ValidateParameters(stage_params);

            // Specific settings or pre-processing steps depending on the operation type
            if (Operation == OperationType::Coloring) {
                if (default_parameters.Has("default_outside_color")) {
                    if (!stage_params.Has("default_outside_color")) {
                        stage_params.AddValue("default_outside_color", mParameters["default_outside_color"]);
                    } else {
                        stage_params["default_outside_color"].SetInt(mParameters["default_outside_color"].GetInt());
                    }
                }
            } else if (Operation == OperationType::GenerateEntities) {
                // Get the model part
                auto& r_model_part = CreateAndGetModelPart(stage_params["model_part_name"].GetString());

                // Get the root model part
                auto& r_root_model_part = r_model_part.GetRootModelPart();
                mRootModelPartsNames.insert(r_root_model_part.Name());

                // Set the start ids
                auto it_find = start_ids.find(r_root_model_part.Name());
                // If the model part is not in the map, we add it and we calculate the start ids, otherwise we just update the start ids for the elements, conditions and constraints (the nodes are not updated because they are generated first and we want to keep the same start node id for all the entity generations of the same model part)
                if (it_find == start_ids.end()) {
                    auto& r_ids = start_ids[r_root_model_part.Name()];
                    r_ids[0] = block_for_each<MaxReduction<std::size_t>>(r_root_model_part.Nodes(), [](Node& rNode) {
                        return rNode.Id();
                    });
                }
                auto& r_ids = start_ids[r_root_model_part.Name()];
                r_ids[1] = block_for_each<MaxReduction<std::size_t>>(r_root_model_part.Elements(), [](Element& rElement) {
                    return rElement.Id();
                });
                r_ids[2] = block_for_each<MaxReduction<std::size_t>>(r_root_model_part.Conditions(), [](Condition& rCondition) {
                    return rCondition.Id();
                });
                r_ids[3] = block_for_each<MaxReduction<std::size_t>>(r_root_model_part.MasterSlaveConstraints(), [](MasterSlaveConstraint& rConstraint) {
                    return rConstraint.Id();
                });

                // Set the initial node id
                if (default_parameters.Has("initial_node_id") && r_ids[0] > 0) {
                    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Setting the initial node id to " << r_ids[0] << std::endl;
                    // If defined, update the initial node id
                    if (stage_params.Has("initial_node_id")) {
                        stage_params["initial_node_id"].SetInt(r_ids[0]);
                    } else { // If not defined, set the initial node id
                        stage_params.AddInt("initial_node_id", r_ids[0]);
                    }
                }

                // Set the initial element id
                if (default_parameters.Has("initial_element_id") && r_ids[1] > 0) {
                    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Setting the initial element id to " << r_ids[1] << std::endl;
                    // If defined, update the initial element id
                    if (stage_params.Has("initial_element_id")) {
                        stage_params["initial_element_id"].SetInt(r_ids[1]);
                    } else { // If not defined, set the initial element id
                        stage_params.AddInt("initial_element_id", r_ids[1]);
                    }
                }

                // Set the initial condition id
                if (default_parameters.Has("initial_condition_id") && r_ids[2] > 0) {
                    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Setting the initial condition id to " << r_ids[2] << std::endl;
                    // If defined, update the initial condition id
                    if (stage_params.Has("initial_condition_id")) {
                        stage_params["initial_condition_id"].SetInt(r_ids[2]);
                    } else { // If not defined, set the initial condition id
                        stage_params.AddInt("initial_condition_id", r_ids[2]);
                    }
                }

                // Set the initial constraint id
                if (default_parameters.Has("initial_constraint_id") && r_ids[3] > 0) {
                    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Setting the initial constraint id to " << r_ids[3] << std::endl;
                    // If defined, update the initial constraint id
                    if (stage_params.Has("initial_constraint_id")) {
                        stage_params["initial_constraint_id"].SetInt(r_ids[3]);
                    } else { // If not defined, set the initial constraint id
                        stage_params.AddInt("initial_constraint_id", r_ids[3]);
                    }
                }
            }

            // Ensure consistent echo level
            if (default_parameters.Has("echo_level")) {
                if (!stage_params.Has("echo_level")) {
                    stage_params.AddValue("echo_level", default_parameters["echo_level"]);
                }
                stage_params["echo_level"].SetInt(mEchoLevel);
            }

            // Invoke the component's do-work virtual
            Invoke(r_prototype, stage_params);

            // Post-processing steps depending on the operation type
            if (Operation == OperationType::GenerateEntities) {
                // Get the model part
                auto& r_model_part = CreateAndGetModelPart(stage_params["model_part_name"].GetString());

                // Get the root model part
                auto& r_root_model_part = r_model_part.GetRootModelPart();

                // Get the ids from the map
                auto& r_ids = start_ids[r_root_model_part.Name()];

                // Update the entities counters (not nodes). NOTE: Check this is valid for hybrid meshes with multiple entity generation stages, and that the nodes are not generated in the entity generation stages (otherwise we should update also the node counter).
                r_ids[1] = r_root_model_part.NumberOfElements();
                r_ids[2] = r_root_model_part.NumberOfConditions();
                r_ids[3] = r_root_model_part.NumberOfMasterSlaveConstraints();
            }

            // Print the end of the operation
            KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << end_message << " " << operation_counter << "/" << number_of_operations << " finished" << std::endl;
        }
    }

    /**
     * @brief Returns the label of the modeler.
     * @return The label as a string.
     */
    const std::string GetLabel() const 
    {
        return "::[OctreeHybridMeshGeneratorModeler]::"; 
    }

    ///@}
};

///@}
///@name Input and output
///@{

/**
 * @brief Stream insertion operator for @ref OctreeHybridMeshGeneratorModeler.
 * @param rOStream Output stream.
 * @param rThis    Modeler whose `Info()` string is written.
 * @return Reference to @p rOStream for chaining.
 */
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeHybridMeshGeneratorModeler& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);
    return rOStream;
}

///@}

} // namespace Kratos
