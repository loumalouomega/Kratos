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
#include "modeler/coloring/octree_hybrid_classify_cells_inside_outside.h"
#include "modeler/entity_generation/generate_hybrid_octree_hexahedra_elements_with_cell_color.h"
#include "modeler/entity_generation/generate_hybrid_octree_quadrilateral_conditions_with_face_color.h"
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

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformDefaultParametersHasRefinedCellSize, KratosCoreFastSuite)
{
    OctreeHybridRefineUniform op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("refined_cell_size"));
    KRATOS_EXPECT_NEAR(p["refined_cell_size"].GetDouble(), 0.0, 1e-15);
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

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsDefaultParametersAllKeys, KratosCoreFastSuite)
{
    OctreeHybridRefineInterfaceCells op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("refinement_depth"));
    KRATOS_EXPECT_TRUE(p.Has("refined_cell_size"));
    KRATOS_EXPECT_TRUE(p.Has("adaptive"));
    KRATOS_EXPECT_TRUE(p.Has("mesh_type"));
    KRATOS_EXPECT_TRUE(p.Has("project_to_surface"));
    KRATOS_EXPECT_TRUE(p.Has("projection_iterations"));
    KRATOS_EXPECT_TRUE(p.Has("projection_smoothing"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsDefaultParametersDefaults, KratosCoreFastSuite)
{
    OctreeHybridRefineInterfaceCells op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["refinement_depth"].GetInt(), 5);
    KRATOS_EXPECT_NEAR(p["refined_cell_size"].GetDouble(), 0.0, 1e-15);
    KRATOS_EXPECT_TRUE(p["adaptive"].GetBool());
    KRATOS_EXPECT_EQ(p["mesh_type"].GetString(), std::string{"dual"});
    KRATOS_EXPECT_FALSE(p["project_to_surface"].GetBool());
    KRATOS_EXPECT_EQ(p["projection_iterations"].GetInt(), 20000);
    KRATOS_EXPECT_EQ(p["projection_smoothing"].GetInt(), 1000);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridRefineInterfaceCells op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridRefineOperation"});
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
    // Only "type" key — verify no other known keys exist
    KRATOS_EXPECT_TRUE(p.Has("type"));
    KRATOS_EXPECT_FALSE(p.Has("model_part_name"));
    KRATOS_EXPECT_FALSE(p.Has("refinement_depth"));
    KRATOS_EXPECT_FALSE(p.Has("color"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridClassifyCellsInsideOutsideInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    OctreeHybridClassifyCellsInsideOutside op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridMesherColoring"});
}

// ===========================================================================
// GenerateHybridOctreeHexahedraElementsWithCellColor — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorDefaultParametersTypeKey, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"GenerateHybridOctreeHexahedraElementsWithCellColor"});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorDefaultParametersAllKeys, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_TRUE(p.Has("color"));
    KRATOS_EXPECT_TRUE(p.Has("properties_id"));
    KRATOS_EXPECT_TRUE(p.Has("generated_entity"));
    KRATOS_EXPECT_TRUE(p.Has("tag_refinement_level"));
    KRATOS_EXPECT_TRUE(p.Has("constraint_type"));
    KRATOS_EXPECT_TRUE(p.Has("constrained_variables"));
    KRATOS_EXPECT_TRUE(p.Has("initial_node_id"));
    KRATOS_EXPECT_TRUE(p.Has("initial_element_id"));
    KRATOS_EXPECT_TRUE(p.Has("initial_constraint_id"));
    KRATOS_EXPECT_TRUE(p.Has("echo_level"));
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorDefaultParametersDefaults, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["properties_id"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["generated_entity"].GetString(), std::string{"Element3D8N"});
    KRATOS_EXPECT_TRUE(p["tag_refinement_level"].GetBool());
    KRATOS_EXPECT_EQ(p["constraint_type"].GetString(), std::string{""});
    KRATOS_EXPECT_EQ(p["constrained_variables"].size(), std::size_t{0});
    KRATOS_EXPECT_EQ(p["initial_node_id"].GetInt(), 0);
    KRATOS_EXPECT_EQ(p["initial_element_id"].GetInt(), 0);
    KRATOS_EXPECT_EQ(p["initial_constraint_id"].GetInt(), 0);
    KRATOS_EXPECT_EQ(p["echo_level"].GetInt(), 0);
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    KRATOS_EXPECT_EQ(op.Info(), std::string{"OctreeHybridMesherEntityGeneration"});
}

// ===========================================================================
// GenerateHybridOctreeQuadrilateralConditionsWithFaceColor — GetDefaultParameters / Info
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeQuadrilateralConditionsWithFaceColorDefaultParametersTypeKey, KratosCoreFastSuite)
{
    GenerateHybridOctreeQuadrilateralConditionsWithFaceColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor"});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeQuadrilateralConditionsWithFaceColorDefaultParametersAllKeys, KratosCoreFastSuite)
{
    GenerateHybridOctreeQuadrilateralConditionsWithFaceColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_TRUE(p.Has("color"));
    KRATOS_EXPECT_TRUE(p.Has("properties_id"));
    KRATOS_EXPECT_TRUE(p.Has("generated_entity"));
    KRATOS_EXPECT_TRUE(p.Has("constraint_type"));
    KRATOS_EXPECT_TRUE(p.Has("constrained_variables"));
    KRATOS_EXPECT_TRUE(p.Has("initial_node_id"));
    KRATOS_EXPECT_TRUE(p.Has("initial_condition_id"));
    KRATOS_EXPECT_TRUE(p.Has("initial_constraint_id"));
    KRATOS_EXPECT_TRUE(p.Has("echo_level"));
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeQuadrilateralConditionsWithFaceColorDefaultParametersDefaults, KratosCoreFastSuite)
{
    GenerateHybridOctreeQuadrilateralConditionsWithFaceColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["properties_id"].GetInt(), 1);
    KRATOS_EXPECT_EQ(p["generated_entity"].GetString(), std::string{"SurfaceCondition3D4N"});
    KRATOS_EXPECT_EQ(p["constraint_type"].GetString(), std::string{""});
    KRATOS_EXPECT_EQ(p["constrained_variables"].size(), std::size_t{0});
    KRATOS_EXPECT_EQ(p["initial_node_id"].GetInt(), 0);
    KRATOS_EXPECT_EQ(p["initial_condition_id"].GetInt(), 0);
    KRATOS_EXPECT_EQ(p["initial_constraint_id"].GetInt(), 0);
    KRATOS_EXPECT_EQ(p["echo_level"].GetInt(), 0);
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeQuadrilateralConditionsWithFaceColorInfoReturnsBaseClassName, KratosCoreFastSuite)
{
    GenerateHybridOctreeQuadrilateralConditionsWithFaceColor op;
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
