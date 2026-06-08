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
#include "testing/testing.h"
#include "modeler/refine_operations/refine_uniform_hybrid_octree.h"
#include "modeler/refine_operations/refine_interface_cells_hybrid_octree.h"
#include "modeler/refine_operations/generate_from_surface_hybrid_octree.h"
#include "modeler/coloring/octree_hybrid_classify_cells_inside_outside.h"
#include "modeler/entity_generation/octree_hybrid_generate_hexes_by_cell_color.h"
#include "modeler/entity_generation/octree_hybrid_generate_boundary_conditions_by_face.h"
#include "modeler/entity_generation/octree_hybrid_generate_hanging_node_constraints.h"
#include "modeler/operation/octree_hybrid_report_mesh_quality.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"

namespace Kratos::Testing {

// ===========================================================================
// OctreeHybridRefineUniform — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridRefineUniform op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridRefineUniform"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformDefaultParametersHasRefinementDepth, KratosCoreFastSuite)
{
    OctreeHybridRefineUniform op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("refinement_depth"));
    KRATOS_EXPECT_EQ(p["refinement_depth"].GetInt(), 5);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformDefaultParametersHasElementSize, KratosCoreFastSuite)
{
    OctreeHybridRefineUniform op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("element_size"));
    KRATOS_EXPECT_NEAR(p["element_size"].GetDouble(), 0.0, 1e-15);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridRefineUniform op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridRefineOperation"});
}

// ===========================================================================
// OctreeHybridRefineInterfaceCells — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridRefineInterfaceCells op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridRefineInterfaceCells"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsDefaultParametersHasInputModelPartName, KratosCoreFastSuite)
{
    OctreeHybridRefineInterfaceCells op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("input_model_part_name"));
    KRATOS_EXPECT_EQ(p["input_model_part_name"].GetString(), std::string{""});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridRefineInterfaceCells op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridRefineOperation"});
}

// ===========================================================================
// OctreeHybridGenerateFromSurface — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateFromSurfaceDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridGenerateFromSurface op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridGenerateFromSurface"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateFromSurfaceDefaultParametersAllKeys, KratosCoreFastSuite)
{
    OctreeHybridGenerateFromSurface op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("input_model_part_name"));
    KRATOS_EXPECT_TRUE(p.Has("refinement_depth"));
    KRATOS_EXPECT_TRUE(p.Has("adaptive"));
    KRATOS_EXPECT_TRUE(p.Has("mesh_type"));
    KRATOS_EXPECT_TRUE(p.Has("project_to_surface"));
    KRATOS_EXPECT_TRUE(p.Has("projection_iterations"));
    KRATOS_EXPECT_TRUE(p.Has("projection_smoothing"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateFromSurfaceDefaultParametersDefaults, KratosCoreFastSuite)
{
    OctreeHybridGenerateFromSurface op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["refinement_depth"].GetInt(), 5);
    KRATOS_EXPECT_TRUE(p["adaptive"].GetBool());
    KRATOS_EXPECT_EQ(p["mesh_type"].GetString(), std::string{"dual"});
    KRATOS_EXPECT_FALSE(p["project_to_surface"].GetBool());
    KRATOS_EXPECT_EQ(p["projection_iterations"].GetInt(), 20000);
    KRATOS_EXPECT_EQ(p["projection_smoothing"].GetInt(), 1000);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateFromSurfaceInfoReturnsOwnClassName, KratosCoreFastSuite)
{
    OctreeHybridGenerateFromSurface op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridGenerateFromSurface"});
}

// ===========================================================================
// OctreeHybridClassifyCellsInsideOutside — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridClassifyCellsInsideOutsideDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridClassifyCellsInsideOutside op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridClassifyCellsInsideOutside"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridClassifyCellsInsideOutsideDefaultParametersOnlyTypeKey, KratosCoreFastSuite)
{
    OctreeHybridClassifyCellsInsideOutside op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p.size(), std::size_t{1});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridClassifyCellsInsideOutsideInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridClassifyCellsInsideOutside op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridMesherColoring"});
}

// ===========================================================================
// OctreeHybridGenerateHexesByCellColor — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHexesByCellColorDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridGenerateHexesByCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridGenerateHexesByCellColor"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHexesByCellColorDefaultParametersAllKeys, KratosCoreFastSuite)
{
    OctreeHybridGenerateHexesByCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_TRUE(p.Has("color"));
    KRATOS_EXPECT_TRUE(p.Has("properties_id"));
    KRATOS_EXPECT_TRUE(p.Has("generated_entity"));
    KRATOS_EXPECT_TRUE(p.Has("tag_refinement_level"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHexesByCellColorDefaultParametersDefaults, KratosCoreFastSuite)
{
    OctreeHybridGenerateHexesByCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["properties_id"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["generated_entity"].GetString(), std::string{"Element3D8N"});
    KRATOS_EXPECT_TRUE(p["tag_refinement_level"].GetBool());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHexesByCellColorInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridGenerateHexesByCellColor op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridMesherEntityGeneration"});
}

// ===========================================================================
// OctreeHybridGenerateBoundaryConditionsByFace — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateBoundaryConditionsByFaceDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridGenerateBoundaryConditionsByFace op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridGenerateBoundaryConditionsByFace"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateBoundaryConditionsByFaceDefaultParametersAllKeys, KratosCoreFastSuite)
{
    OctreeHybridGenerateBoundaryConditionsByFace op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_TRUE(p.Has("color"));
    KRATOS_EXPECT_TRUE(p.Has("properties_id"));
    KRATOS_EXPECT_TRUE(p.Has("generated_entity"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateBoundaryConditionsByFaceDefaultParametersDefaults, KratosCoreFastSuite)
{
    OctreeHybridGenerateBoundaryConditionsByFace op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["properties_id"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["generated_entity"].GetString(), std::string{"SurfaceCondition3D4N"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateBoundaryConditionsByFaceInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridGenerateBoundaryConditionsByFace op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridMesherEntityGeneration"});
}

// ===========================================================================
// OctreeHybridGenerateHangingNodeConstraints — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHangingNodeConstraintsDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridGenerateHangingNodeConstraints op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridGenerateHangingNodeConstraints"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHangingNodeConstraintsDefaultParametersAllKeys, KratosCoreFastSuite)
{
    OctreeHybridGenerateHangingNodeConstraints op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_TRUE(p.Has("constraint_name"));
    KRATOS_EXPECT_TRUE(p.Has("variables"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHangingNodeConstraintsDefaultParametersDefaults, KratosCoreFastSuite)
{
    OctreeHybridGenerateHangingNodeConstraints op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["constraint_name"].GetString(), std::string{"LinearMasterSlaveConstraint"});
    KRATOS_EXPECT_EQ(p["variables"].size(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateHangingNodeConstraintsInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridGenerateHangingNodeConstraints op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridMesherEntityGeneration"});
}

// ===========================================================================
// OctreeHybridReportMeshQuality — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridReportMeshQualityDefaultParametersTypeKey, KratosCoreFastSuite)
{
    OctreeHybridReportMeshQuality op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridReportMeshQuality"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridReportMeshQualityDefaultParametersHasModelPartName, KratosCoreFastSuite)
{
    OctreeHybridReportMeshQuality op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridReportMeshQualityInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridReportMeshQuality op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridMesherOperation"});
}

// ===========================================================================
// OctreeHybridMesherData — IsExtracted
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherDataIsExtractedFalseWhenEmpty, KratosCoreFastSuite)
{
    Internals::OctreeHybridMesherData data;
    KRATOS_EXPECT_FALSE(data.IsExtracted());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherDataIsExtractedTrueAfterFillingCells, KratosCoreFastSuite)
{
    Internals::OctreeHybridMesherData data;
    data.mCells.push_back({0, 1, 2, 3, 4, 5, 6, 7});
    KRATOS_EXPECT_TRUE(data.IsExtracted());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherDataIsExtractedFalseAfterClearingCells, KratosCoreFastSuite)
{
    Internals::OctreeHybridMesherData data;
    data.mCells.push_back({0, 1, 2, 3, 4, 5, 6, 7});
    KRATOS_EXPECT_TRUE(data.IsExtracted());
    data.mCells.clear();
    KRATOS_EXPECT_FALSE(data.IsExtracted());
}

} // namespace Kratos::Testing
