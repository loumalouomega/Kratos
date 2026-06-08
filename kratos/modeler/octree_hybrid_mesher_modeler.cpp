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
#include "modeler/refine_operations/refine_hybrid_octree.h"
#include "modeler/refine_operations/refine_uniform_hybrid_octree.h"
#include "modeler/refine_operations/refine_interface_cells_hybrid_octree.h"
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"
#include "modeler/entity_generation/octree_hybrid_mesher_entity_generation.h"
#include "modeler/operation/octree_hybrid_mesher_operation.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

OctreeHybridMesherModeler::OctreeHybridMesherModeler()
    : Modeler()
    , mpData(Kratos::make_unique<Internals::OctreeHybridMesherData>())
{
}


/***********************************************************************************/
/***********************************************************************************/

OctreeHybridMesherModeler::OctreeHybridMesherModeler(
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

OctreeHybridMesherModeler::~OctreeHybridMesherModeler() = default;


/***********************************************************************************/
/***********************************************************************************/

Internals::OctreeHybridMesherData& OctreeHybridMesherModeler::GetData()
{
    return *mpData;
}


/***********************************************************************************/
/***********************************************************************************/

const Parameters OctreeHybridMesherModeler::GetDefaultParameters() const
{
    return Parameters(R"({
        "echo_level" : 0,
        "input_model_part_name" : "",
        "output_model_part_name" : "",
        "refine_operations_list" : [],
        "coloring_settings_list" : [],
        "entities_generator_list" : [],
        "model_part_operations" : []
    })");
}


/***********************************************************************************/
/***********************************************************************************/

ModelPart& OctreeHybridMesherModeler::CreateAndGetModelPart(const std::string& rFullName)
{
    return mpModel->HasModelPart(rFullName)
        ? mpModel->GetModelPart(rFullName)
        : mpModel->CreateModelPart(rFullName);
}

/***********************************************************************************/
/***********************************************************************************/

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

/***********************************************************************************/
/***********************************************************************************/

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

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMesherModeler::SetupModelPart()
{
    ExecuteRefinementOperations();
    ExecuteColoringOperations();
    ExecuteEntityGenerationOperations();
    ExecuteModelPartOperations();
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMesherModeler::ExecuteRefinementOperations()
{
    Internals::OctreeHybridMesherData& r_data = *mpData;

    // Dispatch every entry in refine_operations_list.
    // The first entry must be OctreeHybridRefineInterfaceCells, which builds the
    // initial octree from the surface and records mesh_type / projection settings
    // in r_data.  Subsequent entries add deeper local or uniform refinement.
    Dispatch<OctreeHybridRefineOperation>(
        "OctreeHybridRefineOperation", mParameters["refine_operations_list"],
        [&](const OctreeHybridRefineOperation& rProto, Parameters rParams) {
            rProto.Refine(*this, rParams); });

    KRATOS_ERROR_IF_NOT(r_data.mpOctree)
        << "OctreeHybridMesherModeler: no octree was built. "
        << "Ensure 'refine_operations_list' starts with an OctreeHybridRefineInterfaceCells entry."
        << std::endl;

    // 2:1 balancing + mesh extraction.
    r_data.mpOctree->StrongConstrain2To1();

    if (r_data.mMeshType == "dual") {
        OctreeHybridMeshUtility::ExtractDualHexMesh(
            *r_data.mpOctree, r_data.mNodes, r_data.mCells, r_data.mCellLevel);

        if (r_data.mProjectToSurface && !r_data.mTriangles.empty()) {
            OctreeHybridMeshUtility::RemoveOutsideElement(
                r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellLevel);
            OctreeHybridMeshUtility::ClearBufferZone(
                r_data.mNodes, r_data.mCells, r_data.mCellLevel);
            OctreeHybridMeshUtility::ProjectToIsoSurface(
                r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellLevel,
                r_data.mProjectionIterations, r_data.mProjectionSmoothing);
            r_data.mProjected = true;
        }
    } else if (r_data.mMeshType == "primal") {
        OctreeHybridMeshUtility::ExtractPrimalHexMesh(
            *r_data.mpOctree, r_data.mNodes, r_data.mCells, r_data.mCellLevel, r_data.mHanging);
    } else {
        KRATOS_ERROR << "OctreeHybridMesherModeler: unknown mesh_type '" << r_data.mMeshType
                     << "'. Use 'dual' or 'primal'." << std::endl;
    }

    r_data.mNodePtrs.assign(r_data.mNodes.size(), nullptr);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMesherModeler::ExecuteColoringOperations()
{
    Dispatch<OctreeHybridMesherColoring>(
        "OctreeHybridMesherColoring", mParameters["coloring_settings_list"],
        [&](const OctreeHybridMesherColoring& rProto, Parameters rParams) {
            rProto.Apply(*this, rParams); });
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMesherModeler::ExecuteEntityGenerationOperations()
{
    Dispatch<OctreeHybridMesherEntityGeneration>(
        "OctreeHybridMesherEntityGeneration", mParameters["entities_generator_list"],
        [&](const OctreeHybridMesherEntityGeneration& rProto, Parameters rParams) {
            rProto.Generate(*this, rParams); });
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMesherModeler::ExecuteModelPartOperations()
{
    Dispatch<OctreeHybridMesherOperation>(
        "OctreeHybridMesherOperation", mParameters["model_part_operations"],
        [&](const OctreeHybridMesherOperation& rProto, Parameters rParams) {
            rProto.Execute(*this, rParams); });
}

} // namespace Kratos
