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
        "input_model_part_name" : "",
        "refinement_depth"      : 5,
        "element_size"          : 0.0,
        "adaptive"              : true,
        "mesh_type"             : "dual",
        "project_to_surface"    : false,
        "projection_iterations" : 20000,
        "projection_smoothing"  : 1000
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridRefineInterfaceCells::Refine(
    OctreeHybridMesherModeler& rModeler,
    Parameters RefineParameters) const
{
    Internals::OctreeHybridMesherData& r_data = rModeler.GetData();

    const std::string op_surface_name = RefineParameters["input_model_part_name"].GetString();

    if (!r_data.mpOctree) {
        // First call in the pipeline: build the initial octree from the surface.
        std::string surface_name = op_surface_name;
        if (surface_name.empty()) surface_name = rModeler.GetInputModelPartName();
        KRATOS_ERROR_IF(surface_name.empty())
            << "OctreeHybridRefineInterfaceCells: no input surface model part specified. "
            << "Set 'input_model_part_name' on the operation or the modeler's top-level key."
            << std::endl;

        ModelPart& r_surface = rModeler.GetModel().GetModelPart(surface_name);
        r_data.mTriangles = OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
        r_data.mpOctree   = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
            r_surface,
            RefineParameters["refinement_depth"].GetInt(),
            RefineParameters["adaptive"].GetBool());
        r_data.mMeshType             = RefineParameters["mesh_type"].GetString();
        r_data.mProjectToSurface     = RefineParameters["project_to_surface"].GetBool();
        r_data.mProjectionIterations = RefineParameters["projection_iterations"].GetInt();
        r_data.mProjectionSmoothing  = RefineParameters["projection_smoothing"].GetInt();
        return;
    }

    // Subsequent calls: selectively subdivide cells near the interface.
    const double element_size = RefineParameters["element_size"].GetDouble();
    const std::size_t target_depth = (element_size > 0.0)
        ? OctreeHybridMeshUtility::ElementSizeToDepth(*r_data.mpOctree, element_size, false)
        : static_cast<std::size_t>(RefineParameters["refinement_depth"].GetInt());

    if (target_depth > r_data.mpOctree->GetDepth())
        r_data.mpOctree->Initialize(target_depth);

    if (op_surface_name.empty()) {
        OctreeHybridMeshUtility::RefineInterfaceCells(
            *r_data.mpOctree, r_data.mTriangles, target_depth);
    } else {
        ModelPart& r_surface = rModeler.GetModel().GetModelPart(op_surface_name);
        const OctreeHybridMeshUtility::TriangleSoup triangles =
            OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
        OctreeHybridMeshUtility::RefineInterfaceCells(
            *r_data.mpOctree, triangles, target_depth);
    }
}

} // namespace Kratos
