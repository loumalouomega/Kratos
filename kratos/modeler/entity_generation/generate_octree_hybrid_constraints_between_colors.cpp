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
#include <map>

// External includes

// Project includes
#include "includes/kratos_components.h"
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/generate_octree_hybrid_constraints_between_colors.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos
{

namespace
{
    /// Coordinates within this distance are considered geometrically coincident.
    constexpr double GeometricTolerance = 1e-9;

    /// Rounds a world-space coordinate to a grid key for coincidence look-ups.
    std::array<long long, 3> RoundToGridKey(const std::array<double, 3>& rPoint)
    {
        return {
            static_cast<long long>(std::llround(rPoint[0] / GeometricTolerance)),
            static_cast<long long>(std::llround(rPoint[1] / GeometricTolerance)),
            static_cast<long long>(std::llround(rPoint[2] / GeometricTolerance))
        };
    }
} // anonymous namespace

const Parameters GenerateOctreeHybridConstraintsBetweenColors::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                  : "GenerateOctreeHybridConstraintsBetweenColors",
        "model_part_name"       : "Undefined",
        "color"                 : -1,
        "color_block_a"         : -1,
        "color_block_b"         : -1,
        "generated_entity"      : "",
        "constrained_variables" : [],
        "initial_node_id"       : 0,
        "initial_constraint_id" : 0,
        "echo_level"            : 0
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void GenerateOctreeHybridConstraintsBetweenColors::Generate(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters GenerationParameters
    ) const
{
    // Validate and assign defaults to the parameters.
    auto& r_data = rModeler.GetData();
    KRATOS_ERROR_IF(!r_data.IsExtracted()) << "GenerateOctreeHybridConstraintsBetweenColors: hex mesh not yet extracted." << std::endl;

    ModelPart& r_model_part = rModeler.CreateAndGetModelPart(GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_model_part);
    rModeler.OverrideStartNodeId(GenerationParameters["initial_node_id"].GetInt());
    rModeler.OverrideStartConstraintId(GenerationParameters["initial_constraint_id"].GetInt());

    // Get echo level for info messages.
    const int echo_level = GenerationParameters["echo_level"].GetInt();

    // Extract parameters.
    const std::string constraint_type = GenerationParameters["generated_entity"].GetString();
    KRATOS_ERROR_IF(constraint_type.empty())
        << "GenerateOctreeHybridConstraintsBetweenColors: \"generated_entity\" must not be empty." << std::endl;

    const auto& r_var_list = GenerationParameters["constrained_variables"];
    if (r_data.mCellColor.empty() || r_var_list.size() == 0) {
        KRATOS_INFO_IF("GenerateOctreeHybridConstraintsBetweenColors", echo_level > 0)
            << "Generated 0 \"" << constraint_type << "\" constraints in ModelPart \""
            << r_model_part.FullName() << "\" (no cell colours or no constrained variables)." << std::endl;
        return;
    }

    const unsigned int n_vars = r_var_list.size();
    std::vector<const Variable<double>*> vars(n_vars);
    for (unsigned int vi = 0; vi < n_vars; ++vi) {
        const std::string& vname = r_var_list[vi].GetString();
        KRATOS_ERROR_IF_NOT(KratosComponents<Variable<double>>::Has(vname))
            << "GenerateOctreeHybridConstraintsBetweenColors: variable '" << vname
            << "' is not registered as a scalar variable." << std::endl;
        vars[vi] = &KratosComponents<Variable<double>>::Get(vname);
    }

    // Build per-block node membership sets from the cell connectivity.
    const int color_a = GenerationParameters["color_block_a"].GetInt();
    const int color_b = GenerationParameters["color_block_b"].GetInt();
    const int want_color = GenerationParameters["color"].GetInt();

    const std::size_t n_nodes = r_data.mNodes.size();
    std::vector<bool> in_a(n_nodes, false);
    std::vector<bool> in_b(n_nodes, false);
    std::vector<bool> in_region;
    if (want_color != -1) in_region.assign(n_nodes, false);

    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        const int cell_color = r_data.mCellColor[c];
        if (cell_color == color_a) {
            for (int k = 0; k < 8; ++k) in_a[r_data.mCells[c][k]] = true;
        }
        if (cell_color == color_b) {
            for (int k = 0; k < 8; ++k) in_b[r_data.mCells[c][k]] = true;
        }
        if (want_color != -1 && cell_color == want_color) {
            for (int k = 0; k < 8; ++k) in_region[r_data.mCells[c][k]] = true;
        }
    }

    // Bucket the candidate "block A" nodes (in A, not in B, optionally in the region)
    // by their rounded coordinates so coincident "block B" nodes can be found in O(1).
    std::map<std::array<long long, 3>, std::vector<std::size_t>> a_buckets;
    for (std::size_t i = 0; i < n_nodes; ++i) {
        if (!in_a[i] || in_b[i]) continue;
        if (!in_region.empty() && !in_region[i]) continue;
        a_buckets[RoundToGridKey(r_data.mNodes[i])].push_back(i);
    }

    // Accumulate any newly created nodes locally, then batch-add them at the end.
    ModelPart::NodesContainerType new_nodes;

    std::size_t n_constraints = 0;
    for (std::size_t ib = 0; ib < n_nodes; ++ib) {
        if (!in_b[ib] || in_a[ib]) continue;
        if (!in_region.empty() && !in_region[ib]) continue;

        const auto it = a_buckets.find(RoundToGridKey(r_data.mNodes[ib]));
        if (it == a_buckets.end()) continue;

        for (const std::size_t ia : it->second) {
            Node::Pointer p_master = rModeler.GenerateOrRetrieveNode(r_model_part, new_nodes, ia);
            Node::Pointer p_slave = rModeler.GenerateOrRetrieveNode(r_model_part, new_nodes, ib);

            for (unsigned int vi = 0; vi < n_vars; ++vi) {
                const Variable<double>& r_var = *vars[vi];
                // DOFs must be registered on the node before a constraint can
                // reference them — AddDof is a no-op if the DOF already exists.
                p_master->AddDof(r_var);
                p_slave->AddDof(r_var);
                // u_slave = u_master (weight 1.0, constant 0.0).
                r_model_part.CreateNewMasterSlaveConstraint(
                    constraint_type, rModeler.NextConstraintId(),
                    *p_master, r_var,
                    *p_slave, r_var,
                    1.0, 0.0);
                ++n_constraints;
            }
        }
    }

    // Deduplicate and bulk-insert any nodes created by GenerateOrRetrieveNode above.
    new_nodes.Unique();
    const std::size_t n_new_nodes = new_nodes.size();
    ModelPartUtils::AddNodesFromOrderedContainer(r_model_part, new_nodes.begin(), new_nodes.end());

    KRATOS_INFO_IF("GenerateOctreeHybridConstraintsBetweenColors", echo_level > 0)
        << "Generated " << n_constraints << " \"" << constraint_type
        << "\" constraints and " << n_new_nodes << " nodes in ModelPart \""
        << r_model_part.FullName() << "\"." << std::endl;
}

} // namespace Kratos
