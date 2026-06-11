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

// System includes
#include <cmath>

// External includes

// Project includes
#include "utilities/timer.h"
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"
#include "utilities/string_utilities.h"
#include "includes/model_part_io.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/refine_operations/refine_hybrid_octree.h"
#include "modeler/operation/octree_hybrid_mesher_operation.h"
#include "modeler/entity_generation/octree_hybrid_mesher_entity_generation.h"
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"

namespace Kratos 
{

OctreeHybridMeshGeneratorModeler::OctreeHybridMeshGeneratorModeler()
    : Modeler()
    , mpData(Kratos::make_unique<Internals::OctreeHybridMesherData>())
{
}

/***********************************************************************************/
/***********************************************************************************/

OctreeHybridMeshGeneratorModeler::OctreeHybridMeshGeneratorModeler(
    Model& rModel,
    Parameters ModelerParameters)
    : Modeler(rModel, ModelerParameters)
    , mpModel(&rModel)
    , mpData(Kratos::make_unique<Internals::OctreeHybridMesherData>())
{
    mParameters.ValidateAndAssignDefaults(GetDefaultParameters());
}

/***********************************************************************************/
/***********************************************************************************/

OctreeHybridMeshGeneratorModeler::~OctreeHybridMeshGeneratorModeler() = default;

/***********************************************************************************/
/***********************************************************************************/

Modeler::Pointer OctreeHybridMeshGeneratorModeler::Create(
    Model& rModel, const Parameters ModelParameters) const
{
    return Kratos::make_shared<OctreeHybridMeshGeneratorModeler>(rModel, ModelParameters);
}

/***********************************************************************************/
/***********************************************************************************/

Internals::OctreeHybridMesherData& OctreeHybridMeshGeneratorModeler::GetData()
{
    return *mpData;
}

/***********************************************************************************/
/***********************************************************************************/

Model& OctreeHybridMeshGeneratorModeler::GetModel()
{
    return *mpModel;
}

/***********************************************************************************/
/***********************************************************************************/

ModelPart& OctreeHybridMeshGeneratorModeler::GetInputModelPart()
{
    KRATOS_ERROR_IF_NOT(mpInputModelPart) << "Input model part not set.  Ensure that \"input_model_part_name\" is set in the parameters and that the model part exists in the Model." << std::endl;
    return *mpInputModelPart;
}

/***********************************************************************************/
/***********************************************************************************/

std::string OctreeHybridMeshGeneratorModeler::GetInputModelPartName() const
{
    return mpInputModelPart ? mpInputModelPart->FullName() : mParameters["input_model_part_name"].GetString();
}

/***********************************************************************************/
/***********************************************************************************/

BoundingBox<Point>& OctreeHybridMeshGeneratorModeler::GetOctreeBoundingBox()
{
    return mOctreeBoundingBox;
}

/***********************************************************************************/
/***********************************************************************************/

const BoundingBox<Point>& OctreeHybridMeshGeneratorModeler::GetOctreeBoundingBox() const
{
    return mOctreeBoundingBox;
}

/***********************************************************************************/
/***********************************************************************************/

bool OctreeHybridMeshGeneratorModeler::HasOctreeBoundingBox() const
{
    return mOctreeBoundingBoxSet;
}

/***********************************************************************************/
/***********************************************************************************/

const Parameters OctreeHybridMeshGeneratorModeler::GetDefaultParameters() const
{
    return Parameters(R"({
        "refinement_settings_list"   : [],
        "coloring_settings_list"     : [],
        "entities_generator_list"    : [],
        "model_part_operations"      : [],
        "mdpa_file_name"             : "",
        "input_model_part_name"      : "",
        "bounding_box_model_part"    : "",
        "bounding_box"  : {
            "min_point" : [],
            "max_point" : []
        },
        "default_outside_color"      : 1,
        "output_files"               : [],
        "remove_orphan_nodes"        : true,
        "echo_level"                 : 1
    })");
}

/***********************************************************************************/
/***********************************************************************************/

ModelPart& OctreeHybridMeshGeneratorModeler::CreateAndGetModelPart(const std::string& rFullName)
{
    std::istringstream iss(rFullName);
    std::string name;
    std::getline(iss, name, '.');
    if(!mpModel->HasModelPart(name))
        mpModel->CreateModelPart(name);
    ModelPart* p_current_model_part = &mpModel->GetModelPart(name);
    while (std::getline(iss, name, '.')) {
        if(!p_current_model_part->HasSubModelPart(name)){
            p_current_model_part->CreateSubModelPart(name);
        }
        p_current_model_part = &(p_current_model_part->GetSubModelPart(name));
    }

    return *p_current_model_part;
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::SetStartIds(ModelPart& rModelPart)
{
    // Propose start IDs based on the existing entities in the root ModelPart, but allow overrides from the parameters.
    ModelPart& r_root_model_part = rModelPart.GetRootModelPart();
    const std::size_t node_proposal = r_root_model_part.NodesArray().empty() ? 1 : r_root_model_part.NodesArray().back()->Id() + 1;
    mStartNodeId = std::max(node_proposal, mStartNodeId == 0 ? std::size_t(1) : mStartNodeId);
    const std::size_t elem_proposal = r_root_model_part.ElementsArray().empty() ? 1 : r_root_model_part.ElementsArray().back()->Id() + 1;
    mStartElementId = std::max(elem_proposal, mStartElementId == 0 ? std::size_t(1) : mStartElementId);
    const std::size_t cond_proposal = r_root_model_part.ConditionsArray().empty() ? 1 : r_root_model_part.ConditionsArray().back()->Id() + 1;
    mStartConditionId = std::max(cond_proposal, mStartConditionId == 0 ? std::size_t(1) : mStartConditionId);
    const std::size_t mpc_proposal = r_root_model_part.MasterSlaveConstraints().empty() ? 1 : r_root_model_part.MasterSlaveConstraints().back().Id() + 1;
    mStartConstraintId = std::max(mpc_proposal, mStartConstraintId == 0 ? std::size_t(1) : mStartConstraintId);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::OverrideStartNodeId(IndexType Id)
{
    if (Id > 0) {
        mStartNodeId = Id;
    }
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::OverrideStartElementId(IndexType Id)
{
    if (Id > 0) {
        mStartElementId = Id;
    }
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::OverrideStartConditionId(IndexType Id)
{
    if (Id > 0) {
        mStartConditionId = Id;
    }
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::OverrideStartConstraintId(IndexType Id)
{
    if (Id > 0) {
        mStartConstraintId = Id;
    }
}

/***********************************************************************************/
/***********************************************************************************/

OctreeHybridMeshGeneratorModeler::IndexType OctreeHybridMeshGeneratorModeler::NextElementId()
{
    return mStartElementId++;
}

/***********************************************************************************/
/***********************************************************************************/

OctreeHybridMeshGeneratorModeler::IndexType OctreeHybridMeshGeneratorModeler::NextConditionId()
{
    return mStartConditionId++;
}

/***********************************************************************************/
/***********************************************************************************/

OctreeHybridMeshGeneratorModeler::IndexType OctreeHybridMeshGeneratorModeler::NextConstraintId()
{
    return mStartConstraintId++;
}

/***********************************************************************************/
/***********************************************************************************/

Node::Pointer OctreeHybridMeshGeneratorModeler::GenerateOrRetrieveNode(
    ModelPart& rModelPart,
    ModelPart::NodesContainerType& rNewNodes,
    IndexType NodeIndex
    )
{
    Internals::OctreeHybridMesherData& r_data = *mpData;
    if (r_data.mNodePtrs[NodeIndex]) return r_data.mNodePtrs[NodeIndex];

    const auto& r_coord = r_data.mNodes[NodeIndex];
    
    // Generate a new node with the next available ID and the coordinates from r_data.mNodes.
    Node::Pointer p_node = Kratos::make_intrusive< Node >( mStartNodeId++, r_coord[0], r_coord[1], r_coord[2]);
    
    // Giving model part's variables list to the node
    p_node->SetSolutionStepVariablesList(rModelPart.pGetNodalSolutionStepVariablesList());

    // Set buffer size
    p_node->SetBufferSize(rModelPart.GetBufferSize());

    r_data.mNodePtrs[NodeIndex] = p_node;
    rNewNodes.push_back(p_node);
    return p_node;
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::Initialize()
{
    // Get the echo level
    mEchoLevel = mParameters["echo_level"].GetInt();

    // Read the model parts
    ReadModelParts();

    // Prepare the internal data structure
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Preparing Internal Data Structure" << std::endl;
    PreparingTheInternalDataStructure(GetInputModelPart());
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Internal Data Structure prepared" << std::endl;
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::SetupModelPart()
{
    // Initialize the model parts
    Initialize();

    // Apply the refinement
    if(mParameters.Has("refinement_settings_list")) {
        ApplyRefinement(mParameters["refinement_settings_list"]);
    }

    // Apply the coloring
    if(mParameters.Has("coloring_settings_list")) {
        ApplyColoring(mParameters["coloring_settings_list"], mParameters["default_outside_color"].GetInt());
    }

    // Generate the entities
    if(mParameters.Has("entities_generator_list")) {
        GenerateEntities(GetInputModelPart(), mParameters["entities_generator_list"]);
    }

    // Apply the operations
    if(mParameters.Has("model_part_operations")) {
        ApplyOperations(mParameters["model_part_operations"]);
    }

    // Remove orphan nodes (nodes not belonging to any element or condition not constraint generated)
    if (mParameters["remove_orphan_nodes"].GetBool()) {
        Parameters remove_orphan_nodes_parameters = Parameters(R"({"model_part_name" : ""})");
        for (auto& r_name : mRootModelPartsNames) {
            KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Removing orphan nodes from model part: " << r_name << std::endl;
            remove_orphan_nodes_parameters["model_part_name"].SetString(r_name);
            // RemoveOrphanNodesModeler(*mpModel, remove_orphan_nodes_parameters).SetupModelPart();
        }
    }

    // Write the output files
    for (const auto& single_output_settings : mParameters["output_files"]){
        const std::string type = single_output_settings["type"].GetString();
        if( type == "octree_vtk"){
            WriteOctreeVTK(single_output_settings);
        }
    }
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ReadModelParts()
{
    KRATOS_ERROR_IF_NOT( mParameters.Has("input_model_part_name") ) << "Missing \"input_model_part_name\" in OctreeMeshGeneratorModeler Parameters." << std::endl;

    // Get the input model part
    if (mpInputModelPart == nullptr) {
        const std::string input_model_part_name = mParameters["input_model_part_name"].GetString();
        mpInputModelPart = &CreateAndGetModelPart(input_model_part_name);
    }

    KRATOS_ERROR_IF_NOT(mParameters.Has("mdpa_file_name")) << "mdpa_file_name not defined" << std::endl;

    const std::string data_file_name =  mParameters["mdpa_file_name"].GetString();

    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Importing Cad Model from: " << data_file_name << std::endl;

    // Load the mdpa
    if(data_file_name!="") {
        ModelPartIO(data_file_name).ReadModelPart(*mpInputModelPart);
    }
}

/***********************************************************************************/
/***********************************************************************************/

const std::string OctreeHybridMeshGeneratorModeler::GetLabel() const
{
    return "::[OctreeHybridMeshGeneratorModeler]::";
}

/***********************************************************************************/
/***********************************************************************************/

template<class TBase, class TInvoke>
void OctreeHybridMeshGeneratorModeler::Dispatch(
    const std::string& rRegistryRoot,
    Parameters StageList,
    TInvoke&& Invoke,
    const OperationType Operation
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

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ResolveOctreeBoundingBox()
{
    KRATOS_TRY

    const Parameters bounding_box = mParameters["bounding_box"];
    const bool has_explicit_bounding_box = bounding_box["min_point"].size() == 3 && bounding_box["max_point"].size() == 3;
    const std::string bounding_box_model_part_name = mParameters["bounding_box_model_part"].GetString();
    const bool has_bounding_box_model_part = !bounding_box_model_part_name.empty();

    KRATOS_ERROR_IF(has_explicit_bounding_box && has_bounding_box_model_part)
        << "OctreeHybridMeshGeneratorModeler: \"bounding_box\" and \"bounding_box_model_part\" "
        << "cannot both be defined. Choose one." << std::endl;

    if (has_explicit_bounding_box) {
        const array_1d<double, 3> min_point = bounding_box["min_point"].GetVector();
        const array_1d<double, 3> max_point = bounding_box["max_point"].GetVector();
        mOctreeBoundingBox = BoundingBox<Point>(Point(min_point), Point(max_point));
        mOctreeBoundingBoxSet = true;
    } else if (has_bounding_box_model_part) {
        KRATOS_ERROR_IF_NOT(mpModel->HasModelPart(bounding_box_model_part_name))
            << "OctreeHybridMeshGeneratorModeler: \"bounding_box_model_part\" '"
            << bounding_box_model_part_name << "' was not found in the Model." << std::endl;
        ModelPart& r_bounding_box_model_part = mpModel->GetModelPart(bounding_box_model_part_name);
        KRATOS_ERROR_IF(r_bounding_box_model_part.NumberOfNodes() == 0)
            << "OctreeHybridMeshGeneratorModeler: \"bounding_box_model_part\" '"
            << bounding_box_model_part_name << "' has no nodes." << std::endl;
        mOctreeBoundingBox = BoundingBox<Point>(r_bounding_box_model_part.NodesBegin(), r_bounding_box_model_part.NodesEnd());
        mOctreeBoundingBoxSet = true;
    } else {
        mOctreeBoundingBoxSet = false;
        return;
    }

    // Validate that the resolved octree bounding box fully contains the input model part.
    ModelPart& r_input_model_part = GetInputModelPart();
    KRATOS_ERROR_IF(r_input_model_part.NumberOfNodes() == 0)
        << "OctreeHybridMeshGeneratorModeler: input model part '" << GetInputModelPartName()
        << "' has no nodes; cannot validate the octree bounding box against it." << std::endl;
    mInputBoundingBox = BoundingBox<Point>(r_input_model_part.NodesBegin(), r_input_model_part.NodesEnd());

    const auto& r_octree_min = mOctreeBoundingBox.GetMinPoint();
    const auto& r_octree_max = mOctreeBoundingBox.GetMaxPoint();
    double diagonal_squared = 0.0;
    for (unsigned int i = 0; i < 3; ++i) {
        const double extent = r_octree_max[i] - r_octree_min[i];
        diagonal_squared += extent * extent;
    }
    const double tolerance = 1e-6 * std::sqrt(diagonal_squared);

    KRATOS_ERROR_IF_NOT(
        mOctreeBoundingBox.IsInside(mInputBoundingBox.GetMinPoint(), tolerance) &&
        mOctreeBoundingBox.IsInside(mInputBoundingBox.GetMaxPoint(), tolerance))
        << "OctreeHybridMeshGeneratorModeler: the octree bounding box "
        << "[(" << r_octree_min[0] << ", " << r_octree_min[1] << ", " << r_octree_min[2] << "), ("
        << r_octree_max[0] << ", " << r_octree_max[1] << ", " << r_octree_max[2] << ")] "
        << "does not contain the input model part '" << GetInputModelPartName() << "' bounding box "
        << "[(" << mInputBoundingBox.GetMinPoint()[0] << ", " << mInputBoundingBox.GetMinPoint()[1] << ", " << mInputBoundingBox.GetMinPoint()[2] << "), ("
        << mInputBoundingBox.GetMaxPoint()[0] << ", " << mInputBoundingBox.GetMaxPoint()[1] << ", " << mInputBoundingBox.GetMaxPoint()[2] << ")]."
        << std::endl;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::PreparingTheInternalDataStructure(ModelPart& rTheInputModelPart)
{
    // Resolve and validate the octree bounding box override, if any
    ResolveOctreeBoundingBox();
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ApplyRefinement(Parameters RefinementParameters)
{
    Timer::Start("Refinement");

    // Retrieve the shared data struct that refinement operations read from and write to.  It holds the octree pointer, extracted node/cell arrays, per-cell colours, hanging-node constraint descriptors, and the node-pointer cache.
    Internals::OctreeHybridMesherData& r_data = *mpData;

    // Dispatch every entry in refinement_settings_list.
    // The first entry must be OctreeHybridRefineInterfaceCells, which builds the
    // initial octree from the surface and records mesh_type / projection settings
    // in r_data.  Subsequent entries add deeper local or uniform refinement.
    Dispatch<OctreeHybridRefineOperation>(
        "OctreeHybridRefineOperation", RefinementParameters,
        [&](const OctreeHybridRefineOperation& rProto, Parameters rParams) {
            rProto.Refine(*this, rParams); }, OperationType::Refine);

    KRATOS_ERROR_IF_NOT(r_data.mpOctree)
        << "OctreeHybridMeshGeneratorModeler: no octree was built. "
        << "Ensure 'refinement_settings_list' starts with an OctreeHybridRefineInterfaceCells entry."
        << std::endl;

    // Ensure the octree is 2:1 balancing + mesh extraction.
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Ensuring the octree is 2-to-1 conforming starting" << std::endl;
    r_data.mpOctree->StrongConstrain2To1();
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Ensuring the octree is 2-to-1 conforming finished" << std::endl;

    // If we consider a dual mesh, the elements are defined by the octree nodes.  If we consider a primal mesh, the elements are defined by the octree cells and the hanging nodes are stored as constraints.
    if (r_data.mMeshType == "dual") {
        // Extract the dual mesh, which is fully conforming with transition templates and no hanging nodes.  Projection settings are fixed at octree build time and must not be changed by subsequent refinement entries in the list.
        OctreeHybridMeshUtility::ExtractDualHexMesh(*r_data.mpOctree, r_data.mNodes, r_data.mCells, r_data.mCellLevel);

        // If projection to surface is on and we have a triangle soup, remove outside elements, clear the buffer zone, and project to the surface.
        if (r_data.mProjectToSurface && !r_data.mTriangles.empty()) {
            OctreeHybridMeshUtility::RemoveOutsideElement(r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellLevel);
            OctreeHybridMeshUtility::ClearBufferZone(r_data.mNodes, r_data.mCells, r_data.mCellLevel);
            OctreeHybridMeshUtility::ProjectToIsoSurface(r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellLevel, r_data.mProjectionIterations, r_data.mProjectionSmoothing);
            r_data.mProjected = true;
        }
    } else if (r_data.mMeshType == "primal") { // If we consider a primal mesh, the elements are defined by the octree cells and the hanging nodes are stored as constraints.
        OctreeHybridMeshUtility::ExtractPrimalHexMesh(
            *r_data.mpOctree, r_data.mNodes, r_data.mCells, r_data.mCellLevel, r_data.mHanging);
    } else { // Unknown mesh type.
        KRATOS_ERROR << "OctreeHybridMeshGeneratorModeler: unknown mesh_type '" << r_data.mMeshType << "'. Use 'dual' or 'primal'." << std::endl;
    }

    // Initialise the node pointer cache to null.  This allows refinement operations to call GenerateOrRetrieveNode in any order without checking if the octree has been extracted yet.
    r_data.mNodePtrs.assign(r_data.mNodes.size(), nullptr);
    
    Timer::Stop("Refinement");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ApplyColoring(
    Parameters ColoringParameters,
    const int OutsideColor
    )
{
    Timer::Start("MeshColoring");

    // Initialize the cell colors to the default outside color before dispatching the coloring operations, so that custom coloring operations can assume a known initial state and only update cells they want to mark as inside or belonging to a feature.  This also ensures that any cells left uncolored by the coloring operations are safely classified as outside.
    Internals::OctreeHybridMesherData& r_data = *mpData;
    r_data.mCellColor.assign(r_data.mCells.size(), OutsideColor);

    // Dispatch every entry in coloring_settings_list.
    Dispatch<OctreeHybridMesherColoring>(
        "OctreeHybridMesherColoring", ColoringParameters,
        [&](const OctreeHybridMesherColoring& rProto, Parameters rParams) {
            rProto.Apply(*this, rParams); }, OperationType::Coloring);

    Timer::Stop("MeshColoring");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::GenerateEntities(
    ModelPart& rTheVolumeModelPart,
    Parameters EntityGeneratorParameters
    )
{
    Timer::Start("EntityGeneration");

    // Dispatch every entry in entities_generator_list.
    Dispatch<OctreeHybridMesherEntityGeneration>(
        "OctreeHybridMesherEntityGeneration", EntityGeneratorParameters,
        [&](const OctreeHybridMesherEntityGeneration& rProto, Parameters rParams) {
            rProto.Generate(*this, rParams); }, OperationType::GenerateEntities);

    Timer::Stop("EntityGeneration");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ApplyOperations(Parameters OperationParameters)
{
    Timer::Start("ApplyOperations");

    // Dispatch every entry in model_part_operations_list.
    Dispatch<OctreeHybridMesherOperation>(
        "OctreeHybridMesherOperation", OperationParameters,
        [&](const OctreeHybridMesherOperation& rProto, Parameters rParams) {
            rProto.Execute(*this, rParams); }, OperationType::ModelPartOperation);

    Timer::Stop("ApplyOperations");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::WriteOctreeVTK(Parameters ThisParameters)
{
    KRATOS_TRY

    KRATOS_ERROR_IF_NOT(ThisParameters.Has("file_name")) << "Missing \"file_name\" in \"octree_vtk\" output settings." << std::endl;

    Internals::OctreeHybridMesherData& r_data = *mpData;
    KRATOS_ERROR_IF_NOT(r_data.mpOctree) << "No octree has been built yet. Cannot write the octree VTK output." << std::endl;

    const std::string file_name = ThisParameters["file_name"].GetString();
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Writing octree leaves to: " << file_name << std::endl;
    OctreeHybridMeshUtility::WritePrimalVtk(*r_data.mpOctree, file_name);

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

std::string OctreeHybridMeshGeneratorModeler::GenerateStartMessage(const OperationType Operation)
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

/***********************************************************************************/
/***********************************************************************************/

std::string OctreeHybridMeshGeneratorModeler::GenerateEndMessage(const OperationType Operation)
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

/***********************************************************************************/
/***********************************************************************************/

std::string OctreeHybridMeshGeneratorModeler::Info() const
{
    return "OctreeHybridMeshGeneratorModeler";
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::PrintInfo(std::ostream& rOStream) const
{
    rOStream << Info();
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::PrintData(std::ostream& rOStream) const
{
    const Internals::OctreeHybridMesherData& r_data = *mpData;

    rOStream << "Input model part: " << GetInputModelPartName() << "\n";
    rOStream << "Mesh type: " << r_data.mMeshType << "\n";
    if (r_data.IsExtracted()) {
        rOStream << "Extracted mesh: " << r_data.mNodes.size() << " nodes, " << r_data.mCells.size() << " hexahedra";
        if (!r_data.mCellColor.empty()) {
            rOStream << ", coloured";
        }
        if (!r_data.mHanging.empty()) {
            rOStream << ", " << r_data.mHanging.size() << " hanging-node constraints";
        }
        rOStream << "\n";
        rOStream << "Projected to surface: " << (r_data.mProjected ? "yes" : "no") << "\n";
    } else {
        rOStream << "Mesh not yet extracted.\n";
    }
    rOStream << "Echo level: " << mEchoLevel;
}

/***********************************************************************************/
/***********************************************************************************/

std::string OctreeHybridMeshGeneratorModeler::GeneratePercentageBar(const double Percentage)
{
    const int bar_width = 50;
    int pos = static_cast<int>(bar_width * Percentage);
    std::string bar = "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) bar += "=";
        else if (i == pos) bar += ">";
        else bar += " ";
    }
    bar += "] " + std::to_string(static_cast<int>(Percentage * 100.0)) + "%";
    return bar;
}

/***********************************************************************************/
/***********************************************************************************/

std::ostream& operator<<(
    std::ostream& rOStream,
    const OctreeHybridMeshGeneratorModeler& rThis
    )
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

} // namespace Kratos
