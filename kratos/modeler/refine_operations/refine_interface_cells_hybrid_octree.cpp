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
#include "modeler/refine_operations/refine_interface_cells_hybrid_octree.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters RefineInterfaceCellsOctreeHybrid::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                     : "RefineInterfaceCellsOctreeHybrid",
        "model_part_name"          : "",
        "refinement_depth"         : 5,
        "refined_cell_size"        : 0.0,
        "adaptive"                 : true,
        "mesh_type"                : "dual",
        "project_to_surface"       : false,
        "projection_iterations"    : 20000,
        "projection_smoothing"     : 1000,
        "enforce_minimum_cell_size": true, // NOTE: To be compatible with octree mesher. DOES NOTHING. To be implemented if required in the future.
        "echo_level"               : 1
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void RefineInterfaceCellsOctreeHybrid::Refine(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters RefineParameters
    ) const
{
    // Get the echo level for this operation (default 1) and override the modeler's echo level if set.
    const int echo_level = RefineParameters["echo_level"].GetInt();

    // If no octree exists yet, build it now using this entry's settings
    Internals::OctreeHybridMesherData& r_data = rModeler.GetData();

    // If no octree exists yet, build it now using this entry's settings
    // (refinement_depth, adaptive, mesh_type, project_to_surface, model_part_name, ...).
    // The octree is built exactly once; every subsequent entry to this Refine method
    // adds deeper refinement without rebuilding from scratch.
    if (!r_data.mpOctree) {
        rModeler.EnsureOctreeBuilt(RefineParameters);
        return;
    }

    // Subsequent calls: selectively subdivide cells near the interface at a finer
    // level.  refined_cell_size takes priority over refinement_depth when set (> 0).
    const double refined_cell_size = RefineParameters["refined_cell_size"].GetDouble();
    const std::size_t target_depth = (refined_cell_size > 0.0)
        ? OctreeHybridMeshUtility::ElementSizeToDepth(*r_data.mpOctree, refined_cell_size, false)
        : static_cast<std::size_t>(RefineParameters["refinement_depth"].GetInt());

    KRATOS_INFO_IF("RefineInterfaceCellsOctreeHybrid", echo_level > 0 && refined_cell_size > 0.0)
        << "RefineInterfaceCellsOctreeHybrid: refined_cell_size=" << refined_cell_size
        << " corresponds to refinement depth " << target_depth << "." << std::endl;

    // Extend the octree's internal grid tables if the requested depth exceeds the
    // current maximum; without this, SubdivideCellByIdAndLevel would silently stop
    // at the old maximum depth. IncreaseDepth preserves all previously refined cells.
    if (target_depth > r_data.mpOctree->GetDepth())
        r_data.mpOctree->IncreaseDepth(target_depth);

    // When no surface is specified for the sub-refinement pass, reuse the triangles
    // from the initial build (the main geometry).  A non-empty name allows adding
    // refinement driven by a different feature surface (e.g. a local refinement zone)
    // without rebuilding the octree.
    const int leaf_count_before = r_data.mpOctree->GetLeafCount();
    const std::string op_surface_name = RefineParameters["model_part_name"].GetString();
    if (op_surface_name.empty()) {
        OctreeHybridMeshUtility::RefineInterfaceCells(*r_data.mpOctree, r_data.mTriangles, target_depth);
    } else {
        ModelPart& r_surface = rModeler.GetModel().GetModelPart(op_surface_name);
        const OctreeHybridMeshUtility::TriangleSoup triangles = OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
        OctreeHybridMeshUtility::RefineInterfaceCells(*r_data.mpOctree, triangles, target_depth);
    }

    KRATOS_INFO_IF("RefineInterfaceCellsOctreeHybrid", echo_level > 0)
        << "Interface cell refinement to depth " << target_depth << " finished: leaf cells "
        << leaf_count_before << " -> " << r_data.mpOctree->GetLeafCount() << "." << std::endl;
}

} // namespace Kratos
