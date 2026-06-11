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
#include "utilities/model_part_utils.h"
#include "modeler/entity_generation/generate_hybrid_octree_hexahedra_elements_with_cell_color.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos 
{

const Parameters GenerateHybridOctreeHexahedraElementsWithCellColor::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                  : "GenerateHybridOctreeHexahedraElementsWithCellColor",
        "model_part_name"       : "Undefined",
        "color"                 : 1,
        "properties_id"         : 1,
        "generated_entity"      : "Element3D8N",
        "tag_refinement_level"  : true,
        "constraint_type"       : "",
        "constrained_variables" : [],
        "initial_node_id"       : 0,
        "initial_element_id"    : 0,
        "initial_constraint_id" : 0,
        "echo_level"            : 0
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void GenerateHybridOctreeHexahedraElementsWithCellColor::Generate(
    OctreeHybridMeshGeneratorModeler& rModeler, 
    Parameters GenerationParameters
    ) const
{
    // Validate and assign defaults to the parameters.
    auto& r_data = rModeler.GetData();
    ModelPart& r_model_part = rModeler.CreateAndGetModelPart(GenerationParameters["model_part_name"].GetString());
    rModeler.SetStartIds(r_model_part);
    rModeler.OverrideStartNodeId(GenerationParameters["initial_node_id"].GetInt());
    rModeler.OverrideStartElementId(GenerationParameters["initial_element_id"].GetInt());
    rModeler.OverrideStartConstraintId(GenerationParameters["initial_constraint_id"].GetInt());

    // Get echo level for info messages.
    const int echo_level = GenerationParameters["echo_level"].GetInt();

    // Extract parameters.
    const int want_color = GenerationParameters["color"].GetInt();
    const std::size_t properties_id = GenerationParameters["properties_id"].GetInt();
    // Retrieve an existing Properties object if present (e.g. a prior generation
    // step already created it on the same ModelPart), otherwise create a new one.
    Properties::Pointer p_properties = r_model_part.HasProperties(properties_id)
        ? r_model_part.pGetProperties(properties_id)
        : r_model_part.CreateNewProperties(properties_id);
    const std::string entity_name = GenerationParameters["generated_entity"].GetString();
    KRATOS_ERROR_IF(!KratosComponents<Element>::Has(entity_name))
        << "GenerateHybridOctreeHexahedraElementsWithCellColor: element type '" << entity_name
        << "' is not registered in KratosComponents." << std::endl;
    const Element& r_prototype_element = KratosComponents<Element>::Get(entity_name);

    // Optional: whether to tag the generated elements with their refinement level. This is useful for visualization and debugging, but not needed for the meshing process itself.
    const bool tag_level = GenerationParameters["tag_refinement_level"].GetBool();

    // Accumulate nodes and elements locally, then batch-add them at the end.
    // Batch addition is required because ModelPart::AddNodes / AddElements
    // expects the containers to be sorted — inserting one-by-one would trigger
    // repeated re-sorting, which is O(N²) for large meshes.
    ModelPart::NodesContainerType new_nodes;
    ModelPart::ElementsContainerType new_elements;
    Element::NodesArrayType cell_nodes(8);

    // The engine corner ordering (CX/CY/CZ) already matches the Kratos
    // Hexahedra3D8 node order, so corner k maps directly to local node k.
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        if (!r_data.mCellColor.empty() && r_data.mCellColor[c] != want_color) continue;
        for (int k = 0; k < 8; ++k) {
            cell_nodes(k) = rModeler.GenerateOrRetrieveNode(r_model_part, new_nodes, r_data.mCells[c][k]);
        }
        auto p_el = r_prototype_element.Create(rModeler.NextElementId(), cell_nodes, p_properties);
        if (tag_level && !r_data.mCellLevel.empty())
            p_el->SetValue(REFINEMENT_LEVEL, r_data.mCellLevel[c]);
        new_elements.push_back(p_el);
    }

    // Deduplicate: each cell corner pushes the same node pointer for all cells
    // that share it, so new_nodes may contain the same pointer multiple times.
    // Unique() removes duplicates before the batch add.
    new_nodes.Unique();
    const std::size_t n_new_nodes = new_nodes.size();
    ModelPartUtils::AddNodesFromOrderedContainer(r_model_part, new_nodes.begin(), new_nodes.end());
    r_model_part.AddElements(new_elements.begin(), new_elements.end());

    KRATOS_INFO_IF("GenerateHybridOctreeHexahedraElementsWithCellColor", echo_level > 0)
        << "Generated " << new_elements.size() << " elements and " << n_new_nodes
        << " nodes in ModelPart \"" << r_model_part.FullName() << "\"." << std::endl;

    // Optionally generate hanging-node constraints for primal meshes.
    // Triggered only when 'constraint_type' is non-empty, 'constrained_variables' is
    // non-empty, and there are 2:1 transitions.
    const std::string constraint_type = GenerationParameters["constraint_type"].GetString();
    const auto& r_var_list = GenerationParameters["constrained_variables"];
    if (constraint_type.empty() || r_var_list.size() == 0 || r_data.mHanging.empty()) return;

    const unsigned int n_vars = r_var_list.size();
    std::vector<const Variable<double>*> vars(n_vars);
    for (unsigned int vi = 0; vi < n_vars; ++vi) {
        const std::string& vname = r_var_list[vi].GetString();
        KRATOS_ERROR_IF_NOT(KratosComponents<Variable<double>>::Has(vname))
            << "GenerateHybridOctreeHexahedraElementsWithCellColor: variable '" << vname
            << "' is not registered as a scalar variable." << std::endl;
        vars[vi] = &KratosComponents<Variable<double>>::Get(vname);
    }

    // Generate one binary LinearMasterSlaveConstraint per (master, variable) pair. The slave
    std::size_t n_constraints = 0;
    for (const auto& hc : r_data.mHanging) {
        // A null pointer means the node index was never materialised as a Kratos
        // Node — e.g. it belongs to a cell excluded by the color filter.
        // Skip the constraint silently; it is valid for a hanging node or one of
        // its masters to fall outside the colored region.
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
            // The slave DOF satisfies: u_slave = Σ weight_m · u_master_m, which
            // requires one constraint per master to express the full interpolation
            // (Kratos's binary constraint accumulates the contributions at assembly).
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

    KRATOS_INFO_IF("GenerateHybridOctreeHexahedraElementsWithCellColor", echo_level > 0)
        << "Generated " << n_constraints << " \"" << constraint_type
        << "\" hanging-node constraints in ModelPart \"" << r_model_part.FullName() << "\"." << std::endl;
}

} // namespace Kratos
