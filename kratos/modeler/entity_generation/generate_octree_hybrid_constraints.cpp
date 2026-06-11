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
#include "modeler/entity_generation/generate_octree_hybrid_constraints.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos
{

const Parameters GenerateOctreeHybridConstraints::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                  : "GenerateOctreeHybridConstraints",
        "model_part_name"       : "Undefined",
        "color"                 : -1,
        "face_color"            : -1,
        "generated_entity"      : "LinearMasterSlaveConstraint",
        "constrained_variables" : ["TEMPERATURE"],
        "initial_node_id"       : 0,
        "initial_constraint_id" : 0,
        "echo_level"            : 0
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void GenerateOctreeHybridConstraints::Generate(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters GenerationParameters
    ) const
{
    // Validate and assign defaults to the parameters.
    auto& r_data = rModeler.GetData();
    KRATOS_ERROR_IF(!r_data.IsExtracted()) << "GenerateOctreeHybridConstraints: hex mesh not yet extracted." << std::endl;

    ModelPart& r_model_part = rModeler.CreateAndGetModelPart(GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_model_part);
    rModeler.OverrideStartNodeId(GenerationParameters["initial_node_id"].GetInt());
    rModeler.OverrideStartConstraintId(GenerationParameters["initial_constraint_id"].GetInt());

    // Get echo level for info messages.
    const int echo_level = GenerationParameters["echo_level"].GetInt();

    // Extract parameters.
    const std::string constraint_type = GenerationParameters["generated_entity"].GetString();
    KRATOS_ERROR_IF(constraint_type.empty())
        << "GenerateOctreeHybridConstraints: \"generated_entity\" must not be empty." << std::endl;

    const auto& r_var_list = GenerationParameters["constrained_variables"];
    if (r_data.mHanging.empty() || r_var_list.size() == 0) {
        KRATOS_INFO_IF("GenerateOctreeHybridConstraints", echo_level > 0)
            << "Generated 0 \"" << constraint_type << "\" constraints in ModelPart \""
            << r_model_part.FullName() << "\" (no hanging nodes or no constrained variables)." << std::endl;
        return;
    }

    const unsigned int n_vars = r_var_list.size();
    std::vector<const Variable<double>*> vars(n_vars);
    for (unsigned int vi = 0; vi < n_vars; ++vi) {
        const std::string& vname = r_var_list[vi].GetString();
        KRATOS_ERROR_IF_NOT(KratosComponents<Variable<double>>::Has(vname))
            << "GenerateOctreeHybridConstraints: variable '" << vname
            << "' is not registered as a scalar variable." << std::endl;
        vars[vi] = &KratosComponents<Variable<double>>::Get(vname);
    }

    // Optional colour filter: keep hanging records whose slave node is a corner of a
    // cell coloured "color".
    const int want_color = GenerationParameters["color"].GetInt();
    std::vector<bool> node_in_color;
    if (want_color != -1 && !r_data.mCellColor.empty()) {
        node_in_color.assign(r_data.mNodes.size(), false);
        for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
            if (r_data.mCellColor[c] != want_color) continue;
            for (int k = 0; k < 8; ++k) node_in_color[r_data.mCells[c][k]] = true;
        }
    }

    // Optional face-colour filter: keep hanging records whose slave node lies on the
    // boundary of the cells coloured "face_color".
    const int want_face_color = GenerationParameters["face_color"].GetInt();
    std::vector<bool> node_on_face_color;
    if (want_face_color != -1) {
        std::vector<std::array<int, 8>> active_cells;
        active_cells.reserve(r_data.mCells.size());
        for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
            if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_face_color) continue;
            active_cells.push_back(r_data.mCells[c]);
        }
        const auto bfaces = OctreeHybridMeshUtility::ExtractBoundaryFaces(active_cells);
        node_on_face_color.assign(r_data.mNodes.size(), false);
        for (const auto& r_bf : bfaces) {
            for (int k = 0; k < 4; ++k) node_on_face_color[r_bf[k]] = true;
        }
    }

    // Generate one binary "generated_entity" constraint per (master, variable) pair for
    // every 2:1 hanging-node transition that passes the colour filters.
    std::size_t n_constraints = 0;
    for (const auto& hc : r_data.mHanging) {
        if (!node_in_color.empty() && !node_in_color[hc.SlaveNode]) continue;
        if (!node_on_face_color.empty() && !node_on_face_color[hc.SlaveNode]) continue;

        // A null pointer means the node index was never materialised as a Kratos
        // Node — e.g. it belongs to a cell excluded by a prior generator's colour
        // filter. Skip the record silently.
        Node::Pointer p_slave = r_data.mNodePtrs[hc.SlaveNode];
        if (!p_slave) continue;

        std::vector<Node::Pointer> master_ptrs(hc.NumMasters);
        bool all_masters_exist = true;
        for (int m = 0; m < hc.NumMasters; ++m) {
            master_ptrs[m] = r_data.mNodePtrs[hc.MasterNodes[m]];
            if (!master_ptrs[m]) { all_masters_exist = false; break; }
        }
        if (!all_masters_exist) continue;

        for (unsigned int vi = 0; vi < n_vars; ++vi) {
            const Variable<double>& r_var = *vars[vi];
            // DOFs must be registered on the node before a constraint can
            // reference them — AddDof is a no-op if the DOF already exists.
            p_slave->AddDof(r_var);
            for (int m = 0; m < hc.NumMasters; ++m) master_ptrs[m]->AddDof(r_var);
            // One binary LinearMasterSlaveConstraint per (master, variable) pair.
            // The slave DOF satisfies: u_slave = Sum weight_m * u_master_m, which
            // requires one constraint per master to express the full interpolation.
            for (int m = 0; m < hc.NumMasters; ++m) {
                r_model_part.CreateNewMasterSlaveConstraint(
                    constraint_type, rModeler.NextConstraintId(),
                    *master_ptrs[m], r_var,
                    *p_slave, r_var,
                    hc.Weights[m], 0.0);
                ++n_constraints;
            }
        }
    }

    KRATOS_INFO_IF("GenerateOctreeHybridConstraints", echo_level > 0)
        << "Generated " << n_constraints << " \"" << constraint_type
        << "\" hanging-node constraints in ModelPart \"" << r_model_part.FullName() << "\"." << std::endl;
}

} // namespace Kratos
