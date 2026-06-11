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

// External includes

// Project includes
#include "utilities/timer.h"
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

Internals::OctreeHybridMesherData& OctreeHybridMeshGeneratorModeler::GetData()
{
    return *mpData;
}

/***********************************************************************************/
/***********************************************************************************/

const Parameters OctreeHybridMeshGeneratorModeler::GetDefaultParameters() const
{
    return Parameters(R"({
        "refine_operations_list"  : [],
        "coloring_settings_list"  : [],
        "entities_generator_list" : [],
        "model_part_operations"   : [],
        "mdpa_file_name"          : "",
        "input_model_part_name"   : "",
        "default_outside_color"   : 1,
        "remove_orphan_nodes"     : true,
        "echo_level"              : 1
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
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::SetupModelPart()
{
    // Initialize the model parts
    Initialize();

    // Apply the refinement
    if(mParameters.Has("refine_operations_list")) {
        ExecuteRefinementOperations();
    }

    // Apply the coloring
    if(mParameters.Has("coloring_settings_list")) {
        ExecuteColoringOperations();
    }

    // Generate the entities
    if(mParameters.Has("entities_generator_list")) {
        ExecuteEntityGenerationOperations();
    }

    // Apply the operations
    if(mParameters.Has("model_part_operations")) {
        ExecuteModelPartOperations();
    }

    // Remove orphan nodes (nodes not belonging to any element or condition not constraint generated)
    if (mParameters["remove_orphan_nodes"].GetBool()) {
        // TODO
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

    KRATOS_INFO_IF("::[OctreeHybridMeshGeneratorModeler]::", mEchoLevel > 0) << "Importing Cad Model from: " << data_file_name << std::endl;

    // Load the mdpa
    if(data_file_name!="") {
        ModelPartIO(data_file_name).ReadModelPart(*mpInputModelPart);
    }
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ExecuteRefinementOperations()
{
    Timer::Start("Refinement");

    // Retrieve the shared data struct that refinement operations read from and write to.  It holds the octree pointer, extracted node/cell arrays, per-cell colours, hanging-node constraint descriptors, and the node-pointer cache.
    Internals::OctreeHybridMesherData& r_data = *mpData;

    // Dispatch every entry in refine_operations_list.
    // The first entry must be OctreeHybridRefineInterfaceCells, which builds the
    // initial octree from the surface and records mesh_type / projection settings
    // in r_data.  Subsequent entries add deeper local or uniform refinement.
    Dispatch<OctreeHybridRefineOperation>(
        "OctreeHybridRefineOperation", mParameters["refine_operations_list"],
        [&](const OctreeHybridRefineOperation& rProto, Parameters rParams) {
            rProto.Refine(*this, rParams); }, OperationType::Refine);

    KRATOS_ERROR_IF_NOT(r_data.mpOctree)
        << "OctreeHybridMeshGeneratorModeler: no octree was built. "
        << "Ensure 'refine_operations_list' starts with an OctreeHybridRefineInterfaceCells entry."
        << std::endl;

    // 2:1 balancing + mesh extraction.
    r_data.mpOctree->StrongConstrain2To1();

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

void OctreeHybridMeshGeneratorModeler::ExecuteColoringOperations()
{
    Timer::Start("MeshColoring");

    // Dispatch every entry in coloring_settings_list.
    Dispatch<OctreeHybridMesherColoring>(
        "OctreeHybridMesherColoring", mParameters["coloring_settings_list"],
        [&](const OctreeHybridMesherColoring& rProto, Parameters rParams) {
            rProto.Apply(*this, rParams); }, OperationType::Coloring);

    Timer::Stop("MeshColoring");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ExecuteEntityGenerationOperations()
{
    Timer::Start("EntityGeneration");

    // Dispatch every entry in entities_generator_list.
    Dispatch<OctreeHybridMesherEntityGeneration>(
        "OctreeHybridMesherEntityGeneration", mParameters["entities_generator_list"],
        [&](const OctreeHybridMesherEntityGeneration& rProto, Parameters rParams) {
            rProto.Generate(*this, rParams); }, OperationType::GenerateEntities);

    Timer::Stop("EntityGeneration");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ExecuteModelPartOperations()
{
    Timer::Start("ApplyOperations");

    // Dispatch every entry in model_part_operations_list.
    Dispatch<OctreeHybridMesherOperation>(
        "OctreeHybridMesherOperation", mParameters["model_part_operations"],
        [&](const OctreeHybridMesherOperation& rProto, Parameters rParams) {
            rProto.Execute(*this, rParams); }, OperationType::ModelPartOperation);

    Timer::Stop("ApplyOperations");
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

} // namespace Kratos
