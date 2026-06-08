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
#include "includes/kratos_components.h"
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/octree_hybrid_generate_boundary_conditions_by_face.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters OctreeHybridGenerateBoundaryConditionsByFace::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"               : "OctreeHybridGenerateBoundaryConditionsByFace",
        "model_part_name"    : "Undefined",
        "color"              : 1,
        "properties_id"      : 1,
        "generated_entity"   : "SurfaceCondition3D4N",
        "constraint_name"    : "LinearMasterSlaveConstraint",
        "variables"          : []
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridGenerateBoundaryConditionsByFace::Generate(
    OctreeHybridMesherModeler& rModeler,
    Parameters GenerationParameters) const
{
    auto& r_data = rModeler.GetData();
    KRATOS_ERROR_IF(!r_data.IsExtracted())
        << "OctreeHybridGenerateBoundaryConditionsByFace: hex mesh not yet extracted." << std::endl;

    ModelPart& r_mp = rModeler.CreateAndGetModelPart(GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_mp);

    const int want_color = GenerationParameters["color"].GetInt();
    const std::size_t properties_id = GenerationParameters["properties_id"].GetInt();
    Properties::Pointer p_props = r_mp.HasProperties(properties_id)
        ? r_mp.pGetProperties(properties_id)
        : r_mp.CreateNewProperties(properties_id);
    const Condition& r_proto = KratosComponents<Condition>::Get(
        GenerationParameters["generated_entity"].GetString());

    // Build a colour-filtered cell list matching the hex generator's skip logic.
    // ExtractBoundaryFaces needs only connectivity (same indices as mNodes).
    std::vector<std::array<int,8>> active_cells;
    active_cells.reserve(r_data.mCells.size());
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_color) continue;
        active_cells.push_back(r_data.mCells[c]);
    }

    const auto bfaces = OctreeHybridMeshUtility::ExtractBoundaryFaces(active_cells);

    ModelPart::NodesContainerType new_nodes;
    ModelPart::ConditionsContainerType new_conditions;
    Condition::NodesArrayType face_nodes(4);
    for (const auto& bf : bfaces) {
        for (int k = 0; k < 4; ++k) {
            Node::Pointer p_node = rModeler.GenerateOrRetrieveNode(r_mp, new_nodes, bf[k]);
            face_nodes(k) = p_node;
            // If the node was created by a prior stage (e.g. hex generator) it won't
            // be in new_nodes, so push it explicitly so it lands in this ModelPart too.
            new_nodes.push_back(p_node);
        }
        new_conditions.push_back(r_proto.Create(rModeler.NextConditionId(), face_nodes, p_props));
    }

    new_nodes.Unique();
    ModelPartUtils::AddNodesFromOrderedContainer(r_mp, new_nodes.begin(), new_nodes.end());
    r_mp.AddConditions(new_conditions.begin(), new_conditions.end());

    // Optionally generate hanging-node constraints for primal meshes.
    // Triggered only when 'variables' is non-empty and there are 2:1 transitions.
    const auto& r_var_list = GenerationParameters["variables"];
    if (r_var_list.size() == 0 || r_data.mHanging.empty()) return;

    const std::string constraint_name = GenerationParameters["constraint_name"].GetString();
    const int n_vars = static_cast<int>(r_var_list.size());
    std::vector<const Variable<double>*> vars(n_vars);
    for (int vi = 0; vi < n_vars; ++vi) {
        const std::string& vname = r_var_list[vi].GetString();
        KRATOS_ERROR_IF_NOT(KratosComponents<Variable<double>>::Has(vname))
            << "OctreeHybridGenerateBoundaryConditionsByFace: variable '" << vname
            << "' is not registered as a scalar variable." << std::endl;
        vars[vi] = &KratosComponents<Variable<double>>::Get(vname);
    }

    for (const auto& hc : r_data.mHanging) {
        Node::Pointer p_slave = r_data.mNodePtrs[hc.SlaveNode];
        if (!p_slave) continue;

        std::vector<Node::Pointer> master_ptrs(hc.NumMasters);
        bool all_masters_exist = true;
        for (int m = 0; m < hc.NumMasters; ++m) {
            master_ptrs[m] = r_data.mNodePtrs[hc.MasterNodes[m]];
            if (!master_ptrs[m]) { all_masters_exist = false; break; }
        }
        if (!all_masters_exist) continue;

        for (int vi = 0; vi < n_vars; ++vi) {
            const Variable<double>& r_var = *vars[vi];
            p_slave->AddDof(r_var);
            for (int m = 0; m < hc.NumMasters; ++m) master_ptrs[m]->AddDof(r_var);
            for (int m = 0; m < hc.NumMasters; ++m) {
                r_mp.CreateNewMasterSlaveConstraint(
                    constraint_name, rModeler.NextConstraintId(),
                    *master_ptrs[m], r_var,
                    *p_slave, r_var,
                    hc.Weights[m], 0.0);
            }
        }
    }
}

} // namespace Kratos
