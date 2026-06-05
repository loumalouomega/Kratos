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
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"
#include "modeler/entity_generation/octree_hybrid_mesher_entity_generation.h"
#include "modeler/operation/octree_hybrid_mesher_operation.h"
#include "utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

OctreeHybridMesherModeler::OctreeHybridMesherModeler()
    : Modeler()
    , mpData(Kratos::make_unique<Internals::OctreeHybridMesherData>())
{
}

OctreeHybridMesherModeler::OctreeHybridMesherModeler(
    Model& rModel,
    Parameters ModelerParameters)
    : Modeler(rModel, ModelerParameters)
    , mpModel(&rModel)
    , mpData(Kratos::make_unique<Internals::OctreeHybridMesherData>())
{
    mParameters.ValidateAndAssignDefaults(GetDefaultParameters());
    mParameters["octree_generator"].ValidateAndAssignDefaults(
        GetDefaultParameters()["octree_generator"]);
}

OctreeHybridMesherModeler::~OctreeHybridMesherModeler() = default;

Internals::OctreeHybridMesherData& OctreeHybridMesherModeler::GetData()
{
    return *mpData;
}

const Parameters OctreeHybridMesherModeler::GetDefaultParameters() const
{
    return Parameters(R"({
        "echo_level" : 0,
        "input_model_part_name" : "",
        "output_model_part_name" : "",
        "octree_generator" : {
            "type" : "generate_octree_from_surface",
            "input_model_part_name" : "",
            "refinement_depth" : 5,
            "adaptive" : true,
            "mesh_type" : "dual",
            "project_to_surface" : false,
            "projection_iterations" : 20000,
            "projection_smoothing" : 1000
        },
        "coloring_settings_list" : [],
        "entities_generator_list" : [],
        "model_part_operations" : []
    })");
}

ModelPart& OctreeHybridMesherModeler::CreateAndGetModelPart(const std::string& rFullName)
{
    return mpModel->HasModelPart(rFullName)
        ? mpModel->GetModelPart(rFullName)
        : mpModel->CreateModelPart(rFullName);
}

void OctreeHybridMesherModeler::SetStartIds(ModelPart& rModelPart)
{
    ModelPart& r_root = rModelPart.GetRootModelPart();
    const std::size_t node_proposal = r_root.NodesArray().empty() ? 1 : r_root.NodesArray().back()->Id() + 1;
    mStartNodeId = std::max(node_proposal, mStartNodeId == 0 ? std::size_t(1) : mStartNodeId);
    const std::size_t elem_proposal = r_root.ElementsArray().empty() ? 1 : r_root.ElementsArray().back()->Id() + 1;
    mStartElementId = std::max(elem_proposal, mStartElementId == 0 ? std::size_t(1) : mStartElementId);
    const std::size_t cond_proposal = r_root.ConditionsArray().empty() ? 1 : r_root.ConditionsArray().back()->Id() + 1;
    mStartConditionId = std::max(cond_proposal, mStartConditionId == 0 ? std::size_t(1) : mStartConditionId);
    const std::size_t mpc_proposal = r_root.MasterSlaveConstraints().empty() ? 1 : r_root.MasterSlaveConstraints().back().Id() + 1;
    mStartConstraintId = std::max(mpc_proposal, mStartConstraintId == 0 ? std::size_t(1) : mStartConstraintId);
}

Node::Pointer OctreeHybridMesherModeler::GenerateOrRetrieveNode(
    ModelPart& rModelPart,
    ModelPart::NodesContainerType& rNewNodes,
    int NodeIndex)
{
    Internals::OctreeHybridMesherData& r_data = *mpData;
    if (r_data.mNodePtrs[NodeIndex]) return r_data.mNodePtrs[NodeIndex];

    const auto& r_coord = r_data.mNodes[NodeIndex];
    Node::Pointer p_node = Kratos::make_intrusive<Node>(
        mStartNodeId++, r_coord[0], r_coord[1], r_coord[2]);
    p_node->SetSolutionStepVariablesList(rModelPart.pGetNodalSolutionStepVariablesList());
    p_node->SetBufferSize(rModelPart.GetBufferSize());

    r_data.mNodePtrs[NodeIndex] = p_node;
    rNewNodes.push_back(p_node);
    return p_node;
}

void OctreeHybridMesherModeler::BuildOctreeAndExtract()
{
    Parameters gen = mParameters["octree_generator"];

    std::string surface_name = gen["input_model_part_name"].GetString();
    if (surface_name.empty()) surface_name = mParameters["input_model_part_name"].GetString();
    KRATOS_ERROR_IF(surface_name.empty())
        << "OctreeHybridMesherModeler: no input surface model part specified "
        << "(set 'input_model_part_name' on the modeler or its 'octree_generator')." << std::endl;
    ModelPart& r_surface = mpModel->GetModelPart(surface_name);

    Internals::OctreeHybridMesherData& r_data = *mpData;
    r_data.mTriangles = OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
    r_data.mpOctree = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
        r_surface, gen["refinement_depth"].GetInt(), gen["adaptive"].GetBool());
    r_data.mpOctree->StrongConstrain2To1();

    const std::string mesh_type = gen["mesh_type"].GetString();
    if (mesh_type == "dual") {
        OctreeHybridMeshUtility::ExtractDualHexMesh(
            *r_data.mpOctree, r_data.mNodes, r_data.mCells, r_data.mCellLevel);

        if (gen["project_to_surface"].GetBool() && !r_data.mTriangles.empty()) {
            OctreeHybridMeshUtility::RemoveOutsideElement(
                r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellLevel);
            OctreeHybridMeshUtility::ClearBufferZone(
                r_data.mNodes, r_data.mCells, r_data.mCellLevel);
            OctreeHybridMeshUtility::ProjectToIsoSurface(
                r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellLevel,
                gen["projection_iterations"].GetInt(), gen["projection_smoothing"].GetInt());
            r_data.mProjected = true;
        }
    } else if (mesh_type == "primal") {
        OctreeHybridMeshUtility::ExtractPrimalHexMesh(
            *r_data.mpOctree, r_data.mNodes, r_data.mCells, r_data.mCellLevel, r_data.mHanging);
    } else {
        KRATOS_ERROR << "OctreeHybridMesherModeler: unknown mesh_type '" << mesh_type
                     << "'. Use 'dual' or 'primal'." << std::endl;
    }

    r_data.mNodePtrs.assign(r_data.mNodes.size(), nullptr);
}

void OctreeHybridMesherModeler::SetupModelPart()
{
    // 1. Internal octree-generation step: build, balance and extract the hex mesh.
    BuildOctreeAndExtract();

    // 2. Colouring (classify cells).
    Dispatch<OctreeHybridMesherColoring>(
        "OctreeHybridMesherColoring", mParameters["coloring_settings_list"],
        [&](const OctreeHybridMesherColoring& rProto, Parameters rParams) { rProto.Apply(*this, rParams); });

    // 3. Entity generation (elements, conditions, constraints).
    Dispatch<OctreeHybridMesherEntityGeneration>(
        "OctreeHybridMesherEntityGeneration", mParameters["entities_generator_list"],
        [&](const OctreeHybridMesherEntityGeneration& rProto, Parameters rParams) { rProto.Generate(*this, rParams); });

    // 4. Operations (post-processing on the finished ModelPart).
    Dispatch<OctreeHybridMesherOperation>(
        "OctreeHybridMesherOperation", mParameters["model_part_operations"],
        [&](const OctreeHybridMesherOperation& rProto, Parameters rParams) { rProto.Execute(*this, rParams); });
}

} // namespace Kratos
