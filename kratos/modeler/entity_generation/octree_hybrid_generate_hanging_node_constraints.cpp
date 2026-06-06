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
#include "containers/model.h"
#include "includes/variables.h"
#include "modeler/entity_generation/octree_hybrid_generate_hanging_node_constraints.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos 
{

const Parameters OctreeHybridGenerateHangingNodeConstraints::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"             : "OctreeHybridGenerateHangingNodeConstraints",
        "model_part_name"  : "Undefined",
        "constraint_name"  : "LinearMasterSlaveConstraint",
        "variables"        : []
    })");
}

/***********************************************************************************/
/***********************************************************************************/


void OctreeHybridGenerateHangingNodeConstraints::Generate(
    OctreeHybridMesherModeler& rModeler,
    Parameters GenerationParameters) const
{
    auto& r_data = rModeler.GetData();

    KRATOS_ERROR_IF(r_data.mHanging.empty() && r_data.mCells.empty())
        << "OctreeHybridGenerateHangingNodeConstraints: no primal mesh data. "
        << "Set mesh_type to 'primal' on the octree_generator." << std::endl;

    if (r_data.mHanging.empty()) return;  // all-same-level mesh — no transitions

    ModelPart& r_mp = rModeler.CreateAndGetModelPart(
        GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_mp);

    const std::string constraint_name = GenerationParameters["constraint_name"].GetString();

    // Collect the list of Kratos variables to constrain
    const auto& r_var_list = GenerationParameters["variables"];
    const int n_vars = static_cast<int>(r_var_list.size());
    std::vector<const Variable<double>*> vars(n_vars);
    for (int vi = 0; vi < n_vars; ++vi) {
        const std::string& vname = r_var_list[vi].GetString();
        KRATOS_ERROR_IF_NOT(KratosComponents<Variable<double>>::Has(vname))
            << "OctreeHybridGenerateHangingNodeConstraints: variable '" << vname
            << "' is not registered as a scalar variable." << std::endl;
        vars[vi] = &KratosComponents<Variable<double>>::Get(vname);
    }

    // Build constraints: one LinearMasterSlaveConstraint per
    // (hanging_node × master_node × variable) — the 1-1 form.
    // The hex generator must have run first so mNodePtrs are filled.
    for (const auto& hc : r_data.mHanging) {
        // Skip if the slave node wasn't created (e.g. outside cells carved away)
        Node::Pointer p_slave = r_data.mNodePtrs[hc.SlaveNode];
        if (!p_slave) continue;

        // Gather master node pointers (all must exist)
        std::vector<Node::Pointer> master_ptrs(hc.NumMasters);
        bool all_masters_exist = true;
        for (int m = 0; m < hc.NumMasters; ++m) {
            master_ptrs[m] = r_data.mNodePtrs[hc.MasterNodes[m]];
            if (!master_ptrs[m]) { all_masters_exist = false; break; }
        }
        if (!all_masters_exist) continue;

        for (int vi = 0; vi < n_vars; ++vi) {
            const Variable<double>& r_var = *vars[vi];

            // Ensure DOFs exist before using the node+variable overload
            p_slave->AddDof(r_var);
            for (int m = 0; m < hc.NumMasters; ++m) master_ptrs[m]->AddDof(r_var);

            // One 1×1 constraint per master (matches model_part_io MPC format)
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
