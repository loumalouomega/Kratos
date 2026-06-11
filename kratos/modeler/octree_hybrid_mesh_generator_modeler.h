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
#include "includes/define_registry.h"
#include "modeler/modeler.h"
#include "geometries/bounding_box.h"

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
 * | 1 | @ref ApplyRefinement | Dispatches each @ref OctreeHybridRefineOperation in `refinement_settings_list` (the first must be @ref OctreeHybridRefineInterfaceCells, which builds the initial octree and records `mesh_type` / projection settings; further entries add deeper refinement).  Then calls `StrongConstrain2To1` and extracts the conforming *dual* or non-conforming *primal* hex mesh.  For `"dual"` with `project_to_surface`, also runs `RemoveOutsideElement`, `ClearBufferZone`, and `ProjectToIsoSurface`. |
 * | 2 | @ref ApplyColoring | Dispatches `coloring_settings_list`; classifies cells as inside (1) or outside (0) the surface. |
 * | 3 | @ref GenerateEntities | Dispatches `entities_generator_list`; emits `Element3D8N` hexes, boundary `SurfaceCondition3D4N`, and/or `LinearMasterSlaveConstraint` hanging-node constraints. |
 * | 4 | @ref ApplyOperations | Dispatches `model_part_operations`; post-processing passes (e.g. mesh-quality reports). |
 *
 * Every component in `refinement_settings_list` and stages 3–5 is a Registry prototype: the
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
 *     "refinement_settings_list" : [
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

    /**
     * @brief Operation kind executed in each meshing stage.
     */
    enum class OperationType
    {
        UNDEFINED,          /// No operation assigned.
        Refine,             /// Octree refinement and adaptation.
        Coloring,           /// Coloring pass for parallel-safe partitioning/grouping.
        GenerateEntities,   /// Creation of mesh entities (nodes/elements/conditions).
        ModelPartOperation  /// Operations that move/assign entities between model parts.
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
        Model& rModel, const Parameters ModelParameters) const override;

    ///@}
    ///@name Modeler stages
    ///@{

    /**
     * @brief Runs the full meshing pipeline.
     * @details Delegates to four sequential private stage methods:
     * 1. @ref ApplyRefinement — dispatches `refinement_settings_list`, then
     *    balances and extracts the hex mesh.
     * 2. @ref ApplyColoring — dispatches `coloring_settings_list`.
     * 3. @ref GenerateEntities — dispatches `entities_generator_list`.
     * 4. @ref ApplyOperations — dispatches `model_part_operations`.
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
     *     "refinement_settings_list"  : [],
     *     "coloring_settings_list"  : [],
     *     "entities_generator_list" : [],
     *     "model_part_operations"   : [],
     *     "mdpa_file_name"          : "",
     *     "input_model_part_name"   : "",
     *     "bounding_box_model_part" : "",
     *     "bounding_box"            : { "min_point" : [], "max_point" : [] },
     *     "default_outside_color"   : 1,
     *     "output_files"            : [],
     *     "remove_orphan_nodes"     : true,
     *     "echo_level"              : 1
     * }
     * @endcode
     * `"bounding_box"` (both `min_point` and `max_point` given as 3-vectors) and
     * `"bounding_box_model_part"` (the name of another ModelPart in the `Model`) are mutually
     * exclusive ways to override the octree's domain; see @ref ResolveOctreeBoundingBox. When
     * neither is set, the domain is computed automatically from the input surface, as before.
     * The `refinement_settings_list` must start with an @ref OctreeHybridRefineInterfaceCells
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
    ///@name Access
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
    Model& GetModel();

    /**
     * @brief Returns the input model part
     * @return The input model part
     */
    ModelPart& GetInputModelPart();

    /**
     * @brief Returns the top-level `input_model_part_name` from the modeler parameters.
     * @details Used by @ref OctreeHybridRefineInterfaceCells as a fallback when its own
     *          `input_model_part_name` is empty on the first call (octree build).
     * @return The model part name string (may be empty if not set).
     */
    std::string GetInputModelPartName() const;

    /**
     * @brief Returns the bounding box of the octree
     * @return The bounding box of the octree
     */
     BoundingBox<Point>& GetOctreeBoundingBox();

    /**
     * @brief Returns the bounding box of the octree (const version)
     * @return The bounding box of the octree
     */
    const BoundingBox<Point>& GetOctreeBoundingBox() const;

    /**
     * @brief Returns whether an explicit octree bounding box override was resolved
     *        from the `"bounding_box"` or `"bounding_box_model_part"` parameters.
     * @details Set by @ref ResolveOctreeBoundingBox during @ref Initialize. When `false`,
     *          @ref GetOctreeBoundingBox returns a default-constructed (degenerate) box and
     *          the octree build falls back to its auto-computed domain.
     * @return `true` if an override is in effect.
     */
    bool HasOctreeBoundingBox() const;

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
    void OverrideStartNodeId(IndexType Id);

    /**
     * @brief Overrides the running element ID counter, if @p Id is non-zero.
     * @details Used by entity-generation stages to honour an explicit
     *          `"initial_element_id"` parameter.  A value of `0` means "keep the
     *          value computed by @ref SetStartIds" and is therefore a no-op.
     * @param Id Requested first element ID, or `0` to leave the counter untouched.
     */
    void OverrideStartElementId(IndexType Id);

    /**
     * @brief Overrides the running condition ID counter, if @p Id is non-zero.
     * @details Used by entity-generation stages to honour an explicit
     *          `"initial_condition_id"` parameter.  A value of `0` means "keep the
     *          value computed by @ref SetStartIds" and is therefore a no-op.
     * @param Id Requested first condition ID, or `0` to leave the counter untouched.
     */
    void OverrideStartConditionId(IndexType Id);

    /**
     * @brief Overrides the running master-slave constraint ID counter, if @p Id is non-zero.
     * @details Used by entity-generation stages to honour an explicit
     *          `"initial_constraint_id"` parameter.  A value of `0` means "keep the
     *          value computed by @ref SetStartIds" and is therefore a no-op.
     * @param Id Requested first constraint ID, or `0` to leave the counter untouched.
     */
    void OverrideStartConstraintId(IndexType Id);

    /**
     * @brief Returns and advances the next unique element ID.
     * @return The next available element ID (post-incremented).
     */
    IndexType NextElementId();

    /**
     * @brief Returns and advances the next unique condition ID.
     * @return The next available condition ID (post-incremented).
     */
    IndexType NextConditionId();

    /**
     * @brief Returns and advances the next unique master-slave constraint ID.
     * @return The next available constraint ID (post-incremented).
     */
    IndexType NextConstraintId();

    ///@}
    ///@name Input and output
    ///@{

    /// @return The string `"OctreeHybridMeshGeneratorModeler"`.
    std::string Info() const override;

    /// Prints `Info()` to @p rOStream.
    void PrintInfo(std::ostream& rOStream) const override;

    /**
     * @brief Prints a summary of the modeler's current state to @p rOStream.
     * @details Reports the input ModelPart name, the `mesh_type` ("dual"/"primal")
     *          recorded by @ref ApplyRefinement, whether the hex mesh has
     *          already been extracted (and if so its node/hexahedra/hanging-constraint
     *          counts and whether it was projected onto the input surface), and the
     *          configured echo level. Defined out-of-line because the relevant data
     *          lives in the PIMPL @ref Internals::OctreeHybridMesherData struct.
     * @param rOStream Output stream.
     */
    void PrintData(std::ostream& rOStream) const override;

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

    /// The bounding box of the octree
    BoundingBox<Kratos::Point> mOctreeBoundingBox;

    /// The bounding box of the input model part
    BoundingBox<Kratos::Point> mInputBoundingBox;

    /// Whether @ref mOctreeBoundingBox was set from `"bounding_box"` /
    /// `"bounding_box_model_part"` (vs. left default for the auto-computed domain).
    bool mOctreeBoundingBoxSet = false;

    ///@}
    ///@name Private Operations
    ///@{

    /**
     * @brief Resolves the octree bounding box override from `"bounding_box"` or
     *        `"bounding_box_model_part"`, validating it against the input model part.
     * @details Called from @ref Initialize, after @ref ReadModelParts (so
     *          @ref GetInputModelPart is available).
     *
     * - `KRATOS_ERROR` if both `"bounding_box"` (non-empty `min_point`/`max_point`) and a
     *   non-empty `"bounding_box_model_part"` are provided.
     * - If `"bounding_box"` has both `min_point` and `max_point` of size 3, builds
     *   @ref mOctreeBoundingBox directly from them.
     * - Else if `"bounding_box_model_part"` is non-empty, looks it up in the `Model`
     *   (`KRATOS_ERROR` if missing or empty) and builds @ref mOctreeBoundingBox from its
     *   nodes.
     * - Else, leaves @ref mOctreeBoundingBoxSet `false` (no override; existing
     *   auto-computed-domain behaviour is unchanged).
     *
     * When an override is resolved, computes @ref mInputBoundingBox from
     * @ref GetInputModelPart's nodes (`KRATOS_ERROR` if it has none) and `KRATOS_ERROR`s if
     * @ref mOctreeBoundingBox does not fully contain it (within a relative tolerance of
     * `1e-6` of the override box's diagonal).
     */
    void ResolveOctreeBoundingBox();

    /**
     * @brief This initializes de internal cartesian mesh data structure to be used for coloring
     * @param rTheInputModelPart The input model part
     */
    void PreparingTheInternalDataStructure(ModelPart& rTheInputModelPart);

    /**
     * @brief Dispatches `refinement_settings_list`, then balances and extracts the hex mesh.
     * @details Calls @ref OctreeHybridRefineOperation::Refine on every entry (the first must
     *          be @ref OctreeHybridRefineInterfaceCells).  After all refinement passes, calls
     *          `StrongConstrain2To1` and then:
     *          - **`mMeshType == "dual"`**: `ExtractDualHexMesh`; optionally
     *            `RemoveOutsideElement` + `ClearBufferZone` + `ProjectToIsoSurface`.
     *          - **`mMeshType == "primal"`**: `ExtractPrimalHexMesh`, which also fills
     *            `OctreeHybridMesherData::mHanging` with 2:1 transition constraints.
     *          Initialises `mNodePtrs` to null after extraction.
     * @param RefinementParameters Parameters for the refinement
     */
    void ApplyRefinement(Parameters RefinementParameters);

    /**
     * @brief Dispatches `coloring_settings_list` via @ref OctreeHybridMesherColoring.
     * @details Each entry's `"type"` is resolved to a registered prototype and its
     *          `Apply(*this, parameters)` virtual is called in list order.
     * @param ColoringParameters Parameters for the coloring
     * @param OutsideColor The color for the outside
     */
    void ApplyColoring(
        Parameters ColoringParameters,
        const int OutsideColor
        );

    /**
     * @brief Dispatches `entities_generator_list` via @ref OctreeHybridMesherEntityGeneration.
     * @details Each entry's `"type"` is resolved to a registered prototype and its
     *          `Generate(*this, parameters)` virtual is called in list order.
     * @param rTheVolumeModelPart The model part in which the entities will be created
     * @param EntityGeneratorParameters Parameters for the entity generator
     */
    void GenerateEntities(
        ModelPart& rTheVolumeModelPart,
        Parameters EntityGeneratorParameters
        );

    /**
     * @brief Dispatches `model_part_operations` via @ref OctreeHybridMesherOperation.
     * @details Each entry's `"type"` is resolved to a registered prototype and its
     *          `Execute(*this, parameters)` virtual is called in list order.
     * @param OperationParameters Parameters for the operation
     */
    void ApplyOperations(Parameters OperationParameters);

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
    std::string GenerateStartMessage(const OperationType Operation);

    /**
     * @brief Generates an end message string based on the given operation type.
     * @param Operation The type of operation for which the end message is to be generated.
     * @return A string representing the end message for the specified operation.
     */
    std::string GenerateEndMessage(const OperationType Operation);

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
        );

    /**
     * @brief Returns the label of the modeler.
     * @return The label as a string.
     */
    const std::string GetLabel() const;

    ///@}
    ///@name Serializer
    ///@{

    friend class Serializer;
    // TODO: Add serialization methods

    ///@}
};

///@}
///@name Input and output
///@{

/**
 * @brief Stream extraction operator for @ref OctreeHybridMeshGeneratorModeler.
 * @param rIStream Input stream.
 * @param rThis    Modeler to read into.
 * @return Reference to @p rIStream for chaining.
 */
inline std::istream& operator >> (
    std::istream& rIStream,
    OctreeHybridMeshGeneratorModeler& rThis
    );

/**
 * @brief Stream insertion operator for @ref OctreeHybridMeshGeneratorModeler.
 * @param rOStream Output stream.
 * @param rThis    Modeler whose `Info()` string is written.
 * @return Reference to @p rOStream for chaining.
 */
std::ostream& operator<<(
    std::ostream& rOStream,
    const OctreeHybridMeshGeneratorModeler& rThis
    );

///@}

} // namespace Kratos
