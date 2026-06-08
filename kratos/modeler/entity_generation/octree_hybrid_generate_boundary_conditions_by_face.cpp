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
    // Retrieve an existing Properties object if present (e.g. the hex-generation
    // step already created it on the same ModelPart), otherwise create a new one.
    Properties::Pointer p_props = r_mp.HasProperties(properties_id)
        ? r_mp.pGetProperties(properties_id)
        : r_mp.CreateNewProperties(properties_id);
    const Condition& r_proto = KratosComponents<Condition>::Get(
        GenerationParameters["generated_entity"].GetString());

    // Build a colour-filtered cell list so ExtractBoundaryFaces sees the same
    // cell set that the hex generator emitted.  A boundary face is one owned by
    // exactly one cell in this filtered set — it is the outer closed surface of
    // the carved mesh within that colour region.
    // ExtractBoundaryFaces uses only connectivity indices (into mNodes), so there
    // is no need to copy the actual node coordinates here.
    std::vector<std::array<int,8>> active_cells;
    active_cells.reserve(r_data.mCells.size());
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_color) continue;
        active_cells.push_back(r_data.mCells[c]);
    }

    const auto bfaces = OctreeHybridMeshUtility::ExtractBoundaryFaces(active_cells);

    // Accumulate nodes and conditions locally, then batch-add at the end so the
    // containers are sorted once rather than after every individual insertion.
    ModelPart::NodesContainerType new_nodes;
    ModelPart::ConditionsContainerType new_conditions;
    Condition::NodesArrayType face_nodes(4);
    for (const auto& bf : bfaces) {
        for (int k = 0; k < 4; ++k) {
            Node::Pointer p_node = rModeler.GenerateOrRetrieveNode(r_mp, new_nodes, bf[k]);
            face_nodes(k) = p_node;
            // GenerateOrRetrieveNode only appends to new_nodes on first creation.
            // Nodes created by a prior stage (e.g. the hex generator) already exist
            // in mNodePtrs and are returned without being pushed into new_nodes, so
            // this ModelPart would not own them.  Pushing explicitly here guarantees
            // every boundary node is registered in this ModelPart regardless of when
            // it was first created.
            new_nodes.push_back(p_node);
        }
        new_conditions.push_back(r_proto.Create(rModeler.NextConditionId(), face_nodes, p_props));
    }

    // Deduplicate: shared face-corner nodes appear multiple times in new_nodes
    // (once per incident boundary face that pushed them).  Unique() removes the
    // duplicates before the batch add.
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
        // A null pointer means the node index was never materialised as a Kratos
        // Node — e.g. it belongs to a cell excluded by the color filter.
        // Skip silently; it is valid for a hanging node or one of its masters to
        // fall outside the boundary condition region.
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
            // DOFs must be registered on the node before a constraint can
            // reference them — AddDof is a no-op if the DOF already exists.
            p_slave->AddDof(r_var);
            for (int m = 0; m < hc.NumMasters; ++m) master_ptrs[m]->AddDof(r_var);
            // One binary LinearMasterSlaveConstraint per (master, variable) pair.
            // The slave satisfies u_slave = Σ weight_m · u_master_m; each term
            // becomes one binary constraint whose contributions accumulate during
            // assembly.
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
