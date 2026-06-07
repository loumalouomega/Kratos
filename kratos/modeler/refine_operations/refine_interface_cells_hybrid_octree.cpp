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
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters OctreeHybridRefineInterfaceCells::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"                  : "OctreeHybridRefineInterfaceCells",
        "refinement_depth"      : 5,
        "element_size"          : 0.0,
        "input_model_part_name" : ""
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridRefineInterfaceCells::Refine(
    OctreeHybridMesherModeler& rModeler,
    Parameters RefineParameters) const
{
    Internals::OctreeHybridMesherData& r_data = rModeler.GetData();
    KRATOS_ERROR_IF_NOT(r_data.mpOctree)
        << "OctreeHybridRefineInterfaceCells: octree has not been built yet." << std::endl;

    const double element_size = RefineParameters["element_size"].GetDouble();
    const std::size_t target_depth = (element_size > 0.0)
        ? OctreeHybridMeshUtility::ElementSizeToDepth(*r_data.mpOctree, element_size)
        : static_cast<std::size_t>(RefineParameters["refinement_depth"].GetInt());

    const std::string mp_name = RefineParameters["input_model_part_name"].GetString();
    if (mp_name.empty()) {
        // Reuse the main surface triangle soup already extracted by the modeler.
        OctreeHybridMeshUtility::RefineInterfaceCells(
            *r_data.mpOctree, r_data.mTriangles, target_depth);
    } else {
        ModelPart& r_surface = rModeler.GetModel().GetModelPart(mp_name);
        const OctreeHybridMeshUtility::TriangleSoup triangles =
            OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
        OctreeHybridMeshUtility::RefineInterfaceCells(
            *r_data.mpOctree, triangles, target_depth);
    }
}

} // namespace Kratos
