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
#include "modeler/refine_operations/refine_uniform_hybrid_octree.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters RefineUniformOctreeHybrid::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"             : "RefineUniformOctreeHybrid",
        "refinement_depth" : 5,
        "model_part_name"  : "Undefined", // NOTE: Only used if the octree has not been built yet (see EnsureOctreeBuilt).
        "max_voxel_size"   : 0.0,
        "echo_level"       : 1
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void RefineUniformOctreeHybrid::Refine(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters RefineParameters
    ) const
{
    // Get the echo level for this operation (default 1) and override the modeler's echo level if set.
    const int echo_level = RefineParameters["echo_level"].GetInt();

    // If no octree exists yet, build it now using this entry's settings
    Internals::OctreeHybridMesherData& r_data = rModeler.GetData();

    // If no octree exists yet, build it now using this entry's settings
    // (refinement_depth, model_part_name, ...) and the modeler's defaults for
    // the remaining build options (adaptive, mesh_type, etc.).
    if (!r_data.mpOctree) {
        rModeler.EnsureOctreeBuilt(RefineParameters);
    }

    // max_voxel_size takes priority over refinement_depth when explicitly set (> 0).
    // The default value of 0.0 signals "not specified", so the explicit depth
    // parameter is used as the fallback.
    const double max_voxel_size = RefineParameters["max_voxel_size"].GetDouble();
    const std::size_t target_depth = (max_voxel_size > 0.0)
        ? OctreeHybridMeshUtility::ElementSizeToDepth(*r_data.mpOctree, max_voxel_size, false)
        : static_cast<std::size_t>(RefineParameters["refinement_depth"].GetInt());

    KRATOS_INFO_IF("RefineUniformOctreeHybrid", echo_level > 0 && max_voxel_size > 0.0)
        << "RefineUniformOctreeHybrid: max_voxel_size=" << max_voxel_size
        << " corresponds to refinement depth " << target_depth << "." << std::endl;

    // OctreeType is constructed with a fixed maximum depth at build time.  If
    // the requested depth exceeds that limit, extend the internal grid tables
    // to the new depth without discarding the subdivision performed so far.
    if (target_depth > r_data.mpOctree->GetDepth()) {
        r_data.mpOctree->IncreaseDepth(target_depth);
    }

    // Subdivide all cells to the requested depth.  This is a no-op for any cell that has already been subdivided to the requested depth or deeper, so it is safe to call this multiple times with the same or increasing depth without rebuilding the octree.
    OctreeHybridMeshUtility::RefineAllCells(*r_data.mpOctree, target_depth);
}

} // namespace Kratos
