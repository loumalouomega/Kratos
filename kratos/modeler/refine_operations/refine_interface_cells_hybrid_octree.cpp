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
        "input_model_part_name"    : "",
        "refinement_depth"         : 5,
        "refined_cell_size"        : 0.0,
        "adaptive"                 : true,
        "mesh_type"                : "dual",
        "project_to_surface"       : false,
        "projection_iterations"    : 20000,
        "projection_smoothing"     : 1000,
        "enforce_minimum_cell_size": true // NOTE: To be compatible with octree mesher. DOES NOTHING. To be implemented if required in the future.
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void RefineInterfaceCellsOctreeHybrid::Refine(
    OctreeHybridMeshGeneratorModeler& rModeler,
    Parameters RefineParameters) const
{
    Internals::OctreeHybridMesherData& r_data = rModeler.GetData();

    const std::string op_surface_name = RefineParameters["input_model_part_name"].GetString();

    if (!r_data.mpOctree) {
        // First call in the pipeline: build the initial octree from the surface.
        // The octree is built exactly once; every subsequent entry to this Refine
        // method adds deeper refinement without rebuilding from scratch.
        std::string surface_name = op_surface_name;
        if (surface_name.empty()) surface_name = rModeler.GetInputModelPartName();
        KRATOS_ERROR_IF(surface_name.empty())
            << "RefineInterfaceCellsOctreeHybrid: no input surface model part specified. "
            << "Set 'input_model_part_name' on the operation or the modeler's top-level key."
            << std::endl;

        ModelPart& r_surface = rModeler.GetModel().GetModelPart(surface_name);
        // Cache the triangle soup in r_data so downstream stages
        // (RemoveOutsideElement, ClassifyInsideOutside, ProjectToIsoSurface) can
        // reuse it without re-parsing the ModelPart.
        r_data.mTriangles = OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
        const BoundingBox<Point>* p_bounding_box_override =
            rModeler.HasOctreeBoundingBox() ? &rModeler.GetOctreeBoundingBox() : nullptr;
        r_data.mpOctree   = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
            r_surface,
            RefineParameters["refinement_depth"].GetInt(),
            RefineParameters["adaptive"].GetBool(),
            p_bounding_box_override);
        // Mesh-type and projection settings are fixed at octree construction and
        // must not be overwritten by subsequent refinement entries in the list.
        r_data.mMeshType             = RefineParameters["mesh_type"].GetString();
        r_data.mProjectToSurface     = RefineParameters["project_to_surface"].GetBool();
        r_data.mProjectionIterations = RefineParameters["projection_iterations"].GetInt();
        r_data.mProjectionSmoothing  = RefineParameters["projection_smoothing"].GetInt();
        return;
    }

    // Subsequent calls: selectively subdivide cells near the interface at a finer
    // level.  refined_cell_size takes priority over refinement_depth when set (> 0).
    const double refined_cell_size = RefineParameters["refined_cell_size"].GetDouble();
    const std::size_t target_depth = (refined_cell_size > 0.0)
        ? OctreeHybridMeshUtility::ElementSizeToDepth(*r_data.mpOctree, refined_cell_size, false)
        : static_cast<std::size_t>(RefineParameters["refinement_depth"].GetInt());

    // Re-initialize the octree's internal grid if the requested depth exceeds the
    // maximum set at construction; without this, SubdivideCellByIdAndLevel would
    // silently stop at the old maximum depth.
    if (target_depth > r_data.mpOctree->GetDepth())
        r_data.mpOctree->Initialize(target_depth);

    // When no surface is specified for the sub-refinement pass, reuse the triangles
    // from the initial build (the main geometry).  A non-empty name allows adding
    // refinement driven by a different feature surface (e.g. a local refinement zone)
    // without rebuilding the octree.
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
