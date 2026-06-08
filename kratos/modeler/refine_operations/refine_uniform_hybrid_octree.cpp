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
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters OctreeHybridRefineUniform::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"             : "OctreeHybridRefineUniform",
        "refinement_depth" : 5,
        "element_size"     : 0.0
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridRefineUniform::Refine(
    OctreeHybridMesherModeler& rModeler,
    Parameters RefineParameters) const
{
    Internals::OctreeHybridMesherData& r_data = rModeler.GetData();
    KRATOS_ERROR_IF_NOT(r_data.mpOctree)
        << "OctreeHybridRefineUniform: octree has not been built yet." << std::endl;

    // element_size takes priority over refinement_depth when explicitly set (> 0).
    // The default value of 0.0 signals "not specified", so the explicit depth
    // parameter is used as the fallback.
    const double element_size = RefineParameters["element_size"].GetDouble();
    const std::size_t target_depth = (element_size > 0.0)
        ? OctreeHybridMeshUtility::ElementSizeToDepth(*r_data.mpOctree, element_size, false)
        : static_cast<std::size_t>(RefineParameters["refinement_depth"].GetInt());

    // OctreeType is constructed with a fixed maximum depth at build time.  If
    // the requested depth exceeds that limit, Initialize re-creates the internal
    // grid at the new depth before any subdivision can reach that level.
    if (target_depth > r_data.mpOctree->GetDepth()) {
        r_data.mpOctree->Initialize(target_depth);
    }

    OctreeHybridMeshUtility::RefineAllCells(*r_data.mpOctree, target_depth);
}

} // namespace Kratos
