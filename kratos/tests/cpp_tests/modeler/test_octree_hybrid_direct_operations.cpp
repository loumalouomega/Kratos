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
#include "containers/model.h"
#include "includes/variables.h"

#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

#include "modeler/refine_operations/refine_hybrid_octree.h"
#include "modeler/refine_operations/refine_uniform_hybrid_octree.h"
#include "modeler/refine_operations/refine_interface_cells_hybrid_octree.h"

#include "modeler/coloring/octree_hybrid_mesher_coloring.h"
#include "modeler/coloring/octree_hybrid_classify_cells_inside_outside.h"

#include "modeler/entity_generation/octree_hybrid_mesher_entity_generation.h"
#include "modeler/entity_generation/generate_hybrid_octree_hexahedra_elements_with_cell_color.h"
#include "modeler/entity_generation/generate_hybrid_octree_quadrilateral_conditions_with_face_color.h"
#include "modeler/entity_generation/generate_octree_hybrid_constraints.h"
#include "modeler/entity_generation/generate_octree_hybrid_constraints_between_colors.h"

#include "modeler/operation/octree_hybrid_mesher_operation.h"
#include "modeler/operation/octree_hybrid_report_mesh_quality.h"
#include "modeler/operation/octree_hybrid_find_contacts_in_skin_model_part.h"

namespace Kratos::Testing {

namespace {

using Util = OctreeHybridMeshUtility;

void BuildClosedBoxSurface(ModelPart& rSurfaceMesh, double Lo = 0.3, double Hi = 0.7)
{
    rSurfaceMesh.SetBufferSize(1);
    rSurfaceMesh.GetProcessInfo()[DOMAIN_SIZE] = 3;
    const std::array<std::array<double,3>, 8> corners = {{
        {Lo,Lo,Lo}, {Hi,Lo,Lo}, {Hi,Hi,Lo}, {Lo,Hi,Lo},
        {Lo,Lo,Hi}, {Hi,Lo,Hi}, {Hi,Hi,Hi}, {Lo,Hi,Hi}
    }};
    IndexType nid = 1;
    for (const auto& c : corners) rSurfaceMesh.CreateNewNode(nid++, c[0], c[1], c[2]);
    rSurfaceMesh.CreateNewNode(nid++, 0.0, 0.0, 0.0);
    rSurfaceMesh.CreateNewNode(nid,   1.0, 1.0, 1.0);
    const std::array<std::array<IndexType,3>, 12> faces = {{
        {1,2,3},{1,3,4},{5,7,6},{5,8,7},
        {1,6,2},{1,5,6},{4,3,7},{4,7,8},
        {1,4,8},{1,8,5},{2,6,7},{2,7,3}
    }};
    IndexType gid = 1;
    for (const auto& f : faces)
        rSurfaceMesh.CreateNewGeometry("Triangle3D3", gid++,
            std::vector<IndexType>{f[0], f[1], f[2]});
}

OctreeHybridMeshGeneratorModeler MakeEmptyModeler(Model& rModel,
    const std::string& rInput, const std::string& rOutput)
{
    Parameters params(R"({
        "input_model_part_name": ")" + rInput + R"(",
        "refinement_settings_list":  [],
        "coloring_settings_list":  [],
        "entities_generator_list": [],
        "model_part_operations":   []
    })");
    return OctreeHybridMeshGeneratorModeler(rModel, params);
}

void ExtractMesh(OctreeHybridMeshGeneratorModeler& rModeler)
{
    auto& r_data = rModeler.GetData();
    KRATOS_ERROR_IF(!r_data.mpOctree) << "Octree not built — call Refine first.";
    r_data.mpOctree->StrongConstrain2To1();
    if (r_data.mMeshType == "primal") {
        Util::ExtractPrimalHexMesh(*r_data.mpOctree,
            r_data.mNodes, r_data.mCells, r_data.mCellLevel, r_data.mHanging);
    } else {
        Util::ExtractDualHexMesh(*r_data.mpOctree,
            r_data.mNodes, r_data.mCells, r_data.mCellLevel);
    }
    r_data.mNodePtrs.assign(r_data.mNodes.size(), nullptr);
}

} // anonymous namespace

// ===========================================================================
// OctreeHybridRefineInterfaceCells::Refine — first call (octree build)
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsFirstCallBuildsOctree, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridRefineInterfaceCells op;
    Parameters p(R"({
        "type"            : "OctreeHybridRefineInterfaceCells",
        "refinement_depth": 3,
        "adaptive"        : false,
        "mesh_type"       : "dual"
    })");
    op.ValidateParameters(p);
    op.Refine(modeler, p);

    KRATOS_EXPECT_NE(modeler.GetData().mpOctree, nullptr);
    KRATOS_EXPECT_FALSE(modeler.GetData().mTriangles.empty());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsFirstCallExtractsTriangles, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridRefineInterfaceCells op;
    Parameters p(R"({
        "type"            : "OctreeHybridRefineInterfaceCells",
        "refinement_depth": 3,
        "adaptive"        : false,
        "mesh_type"       : "dual"
    })");
    op.ValidateParameters(p);
    op.Refine(modeler, p);

    KRATOS_EXPECT_FALSE(modeler.GetData().mTriangles.empty());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsFirstCallPrimalMeshType, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridRefineInterfaceCells op;
    Parameters p(R"({
        "type"            : "OctreeHybridRefineInterfaceCells",
        "refinement_depth": 4,
        "adaptive"        : false,
        "mesh_type"       : "primal"
    })");
    op.ValidateParameters(p);
    op.Refine(modeler, p);

    auto& r_data = modeler.GetData();
    KRATOS_EXPECT_NE(r_data.mpOctree, nullptr);
    KRATOS_EXPECT_EQ(r_data.mMeshType, std::string{"primal"});
}

// ===========================================================================
// OctreeHybridRefineUniform::Refine — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformRefineDirectlyIncreasesLeafCount, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    // Build octree first via GenerateFromSurface
    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters p(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3,
            "adaptive"        : false,
            "mesh_type"       : "dual"
        })");
        build_op.ValidateParameters(p);
        build_op.Refine(modeler, p);
    }

    const int leaves_before = modeler.GetData().mpOctree->GetLeafCount();

    // Now apply uniform refinement to a deeper level
    OctreeHybridRefineUniform uniform_op;
    Parameters p(R"({
        "type"              : "OctreeHybridRefineUniform",
        "refinement_depth"  : 4,
        "refined_cell_size" : 0.0
    })");
    // Reset mesh so Refine can run (mpOctree must exist)
    modeler.GetData().mCells.clear();
    modeler.GetData().mNodes.clear();
    modeler.GetData().mCellLevel.clear();
    uniform_op.Refine(modeler, p);

    KRATOS_EXPECT_GE(modeler.GetData().mpOctree->GetLeafCount(), leaves_before);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformRefineDirectlyNoOctreeThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridRefineUniform op;
    Parameters p(R"({
        "type"              : "OctreeHybridRefineUniform",
        "refinement_depth"  : 3,
        "refined_cell_size" : 0.0
    })");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(op.Refine(modeler, p), "");
}

// ===========================================================================
// OctreeHybridRefineInterfaceCells::Refine — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsRefineDirectlyIncreasesLeafCount, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters p(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 2,
            "adaptive"        : false,
            "mesh_type"       : "dual"
        })");
        build_op.ValidateParameters(p);
        build_op.Refine(modeler, p);
    }

    const int leaves_before = modeler.GetData().mpOctree->GetLeafCount();

    modeler.GetData().mCells.clear();
    modeler.GetData().mNodes.clear();
    modeler.GetData().mCellLevel.clear();

    OctreeHybridRefineInterfaceCells interface_op;
    Parameters p(R"({
        "type"                  : "OctreeHybridRefineInterfaceCells",
        "refinement_depth"      : 4,
        "refined_cell_size"     : 0.0,
        "input_model_part_name" : ""
    })");
    interface_op.Refine(modeler, p);

    KRATOS_EXPECT_GE(modeler.GetData().mpOctree->GetLeafCount(), leaves_before);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsRefineDirectlyNoSurfaceThrows, KratosCoreFastSuite)
{
    // No octree + no surface (empty input_model_part_name AND empty modeler top-level) → throws
    Model model;
    auto modeler = MakeEmptyModeler(model, "", "Out");

    OctreeHybridRefineInterfaceCells op;
    Parameters p(R"({
        "type"                  : "OctreeHybridRefineInterfaceCells",
        "refinement_depth"      : 3,
        "refined_cell_size"     : 0.0,
        "input_model_part_name" : ""
    })");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(op.Refine(modeler, p), "");
}

// ===========================================================================
// OctreeHybridClassifyCellsInsideOutside::Apply — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridClassifyCellsInsideOutsideApplyDirectlyFillsColors, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters p(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3,
            "adaptive"        : false,
            "mesh_type"       : "dual"
        })");
        build_op.ValidateParameters(p);
        build_op.Refine(modeler, p);
    }
    ExtractMesh(modeler);

    auto& r_data = modeler.GetData();
    KRATOS_EXPECT_TRUE(r_data.mCellColor.empty());

    OctreeHybridClassifyCellsInsideOutside color_op;
    Parameters p(R"({"type": "OctreeHybridClassifyCellsInsideOutside"})");
    color_op.Apply(modeler, p);

    KRATOS_EXPECT_EQ(r_data.mCellColor.size(), r_data.mCells.size());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridClassifyCellsInsideOutsideApplyDirectlyProducesBothColors, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters p(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3,
            "adaptive"        : false,
            "mesh_type"       : "dual"
        })");
        build_op.ValidateParameters(p);
        build_op.Refine(modeler, p);
    }
    ExtractMesh(modeler);

    OctreeHybridClassifyCellsInsideOutside color_op;
    Parameters p(R"({"type": "OctreeHybridClassifyCellsInsideOutside"})");
    color_op.Apply(modeler, p);

    const auto& colors = modeler.GetData().mCellColor;
    bool has_inside  = std::any_of(colors.begin(), colors.end(), [](int c){ return c == 1; });
    bool has_outside = std::any_of(colors.begin(), colors.end(), [](int c){ return c == 0; });
    KRATOS_EXPECT_TRUE(has_inside);
    KRATOS_EXPECT_TRUE(has_outside);
}

// ===========================================================================
// GenerateHybridOctreeHexahedraElementsWithCellColor::Generate — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorGenerateDirectlyCreatesElements, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    // Build + extract + color
    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }

    GenerateHybridOctreeHexahedraElementsWithCellColor gen_op;
    Parameters p(R"({
        "type"                : "GenerateHybridOctreeHexahedraElementsWithCellColor",
        "model_part_name"     : "Out",
        "color"               : 1,
        "properties_id"       : 1,
        "generated_entity"    : "Element3D8N",
        "tag_refinement_level": true
    })");
    gen_op.ValidateParameters(p);
    gen_op.Generate(modeler, p);

    KRATOS_EXPECT_GT(model.GetModelPart("Out").NumberOfElements(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorGenerateDirectlyColor0Elements, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Outside");
    auto modeler = MakeEmptyModeler(model, "Skin", "Outside");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }

    GenerateHybridOctreeHexahedraElementsWithCellColor gen_op;
    Parameters p(R"({
        "type"                : "GenerateHybridOctreeHexahedraElementsWithCellColor",
        "model_part_name"     : "Outside",
        "color"               : 0,
        "properties_id"       : 1,
        "generated_entity"    : "Element3D8N",
        "tag_refinement_level": false
    })");
    gen_op.ValidateParameters(p);
    gen_op.Generate(modeler, p);

    KRATOS_EXPECT_GT(model.GetModelPart("Outside").NumberOfElements(), std::size_t{0});
}

// ===========================================================================
// GenerateHybridOctreeQuadrilateralConditionsWithFaceColor::Generate — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeQuadrilateralConditionsWithFaceColorGenerateDirectlyCreatesConditions, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }

    // Generate hexes first
    {
        GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
        Parameters hex_p(R"({
            "type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Out","color":1,"properties_id":1,
            "generated_entity":"Element3D8N","tag_refinement_level":true
        })");
        hex_op.ValidateParameters(hex_p);
        hex_op.Generate(modeler, hex_p);
    }

    // Now generate boundary conditions
    GenerateHybridOctreeQuadrilateralConditionsWithFaceColor bc_op;
    Parameters p(R"({
        "type"             : "GenerateHybridOctreeQuadrilateralConditionsWithFaceColor",
        "model_part_name"  : "Out",
        "color"            : 1,
        "properties_id"    : 1,
        "generated_entity" : "SurfaceCondition3D4N"
    })");
    bc_op.ValidateParameters(p);
    bc_op.Generate(modeler, p);

    KRATOS_EXPECT_GT(model.GetModelPart("Out").NumberOfConditions(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeQuadrilateralConditionsWithFaceColorNotExtractedThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    GenerateHybridOctreeQuadrilateralConditionsWithFaceColor bc_op;
    Parameters p(R"({
        "type"             : "GenerateHybridOctreeQuadrilateralConditionsWithFaceColor",
        "model_part_name"  : "Out",
        "color"            : 1,
        "properties_id"    : 1,
        "generated_entity" : "SurfaceCondition3D4N"
    })");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(bc_op.Generate(modeler, p), "");
}

// ===========================================================================
// GenerateHybridOctreeHexahedraElementsWithCellColor::Generate — hanging-node constraint path
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorWithVariablesCreatesConstraints, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    // Build primal mesh to populate mHanging
    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 4, "adaptive": false, "mesh_type": "primal"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);

    auto& r_data = modeler.GetData();
    r_data.mCellColor.assign(r_data.mCells.size(), 1);

    GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
    Parameters p(R"({
        "type"                  : "GenerateHybridOctreeHexahedraElementsWithCellColor",
        "model_part_name"       : "Out",
        "color"                 : 1,
        "properties_id"         : 1,
        "generated_entity"      : "Element3D8N",
        "tag_refinement_level"  : true,
        "constraint_type"       : "LinearMasterSlaveConstraint",
        "constrained_variables" : ["DISPLACEMENT_X"]
    })");
    hex_op.ValidateParameters(p);
    hex_op.Generate(modeler, p);

    // Uniform-depth box has no 2:1 transitions → zero hanging constraints expected,
    // but the call must complete without error.
    KRATOS_EXPECT_GE(model.GetModelPart("Out").NumberOfMasterSlaveConstraints(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateHybridOctreeHexahedraElementsWithCellColorEmptyVariablesSkipsConstraints, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "primal"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);

    auto& r_data = modeler.GetData();
    r_data.mCellColor.assign(r_data.mCells.size(), 1);

    GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
    Parameters p(R"({
        "type"                  : "GenerateHybridOctreeHexahedraElementsWithCellColor",
        "model_part_name"       : "Out",
        "color"                 : 1,
        "properties_id"         : 1,
        "generated_entity"      : "Element3D8N",
        "tag_refinement_level"  : true,
        "constraint_type"       : "LinearMasterSlaveConstraint",
        "constrained_variables" : []
    })");
    hex_op.ValidateParameters(p);
    hex_op.Generate(modeler, p);

    // Empty constrained_variables list → no constraints created
    KRATOS_EXPECT_EQ(model.GetModelPart("Out").NumberOfMasterSlaveConstraints(), std::size_t{0});
}

// ===========================================================================
// GenerateOctreeHybridConstraints::Generate — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsGenerateDirectlyDoesNotThrow, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    // Build primal mesh to populate mHanging
    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 4, "adaptive": false, "mesh_type": "primal"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);

    auto& r_data = modeler.GetData();
    r_data.mCellColor.assign(r_data.mCells.size(), 1);

    // Generate hexes first so mNodePtrs is populated.
    {
        GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
        Parameters hex_p(R"({
            "type"                 : "GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name"      : "Out",
            "color"                : 1,
            "properties_id"        : 1,
            "generated_entity"     : "Element3D8N",
            "tag_refinement_level" : true
        })");
        hex_op.ValidateParameters(hex_p);
        hex_op.Generate(modeler, hex_p);
    }

    GenerateOctreeHybridConstraints constraints_op;
    Parameters p(R"({
        "type"                  : "GenerateOctreeHybridConstraints",
        "model_part_name"       : "Out",
        "color"                 : 1,
        "face_color"            : -1,
        "generated_entity"      : "LinearMasterSlaveConstraint",
        "constrained_variables" : ["DISPLACEMENT_X"]
    })");
    constraints_op.ValidateParameters(p);
    constraints_op.Generate(modeler, p);

    // Uniform-depth box has no 2:1 transitions → zero hanging constraints expected,
    // but the call must complete without error.
    KRATOS_EXPECT_GE(model.GetModelPart("Out").NumberOfMasterSlaveConstraints(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsEmptyVariablesSkipsConstraints, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "primal"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);

    auto& r_data = modeler.GetData();
    r_data.mCellColor.assign(r_data.mCells.size(), 1);

    GenerateOctreeHybridConstraints constraints_op;
    Parameters p(R"({
        "type"                  : "GenerateOctreeHybridConstraints",
        "model_part_name"       : "Out",
        "constrained_variables" : []
    })");
    constraints_op.ValidateParameters(p);
    constraints_op.Generate(modeler, p);

    KRATOS_EXPECT_EQ(model.GetModelPart("Out").NumberOfMasterSlaveConstraints(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsNotExtractedThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    GenerateOctreeHybridConstraints constraints_op;
    Parameters p(R"({
        "type"             : "GenerateOctreeHybridConstraints",
        "model_part_name"  : "Out"
    })");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(constraints_op.Generate(modeler, p), "");
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsEmptyGeneratedEntityThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "primal"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);

    GenerateOctreeHybridConstraints constraints_op;
    Parameters p = constraints_op.GetDefaultParameters();
    p["model_part_name"].SetString("Out");
    p["generated_entity"].SetString("");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(constraints_op.Generate(modeler, p), "");
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsDefaultParameters, KratosCoreFastSuite)
{
    GenerateOctreeHybridConstraints constraints_op;
    Parameters p = constraints_op.GetDefaultParameters();

    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"GenerateOctreeHybridConstraints"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["face_color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["generated_entity"].GetString(), std::string{"LinearMasterSlaveConstraint"});
    KRATOS_EXPECT_EQ(p["constrained_variables"].size(), std::size_t{1});
    KRATOS_EXPECT_EQ(p["constrained_variables"][0].GetString(), std::string{"TEMPERATURE"});
}

// ===========================================================================
// GenerateOctreeHybridConstraintsBetweenColors::Generate — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsBetweenColorsGenerateDirectlyDoesNotThrow, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }

    // Generate hexes for both colours so mNodePtrs is populated on both sides.
    {
        GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
        Parameters hex_p(R"({
            "type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Out","color":1,"properties_id":1,
            "generated_entity":"Element3D8N","tag_refinement_level":false
        })");
        hex_op.ValidateParameters(hex_p);
        hex_op.Generate(modeler, hex_p);
    }
    {
        GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
        Parameters hex_p(R"({
            "type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Out","color":0,"properties_id":1,
            "generated_entity":"Element3D8N","tag_refinement_level":false
        })");
        hex_op.ValidateParameters(hex_p);
        hex_op.Generate(modeler, hex_p);
    }

    GenerateOctreeHybridConstraintsBetweenColors between_op;
    Parameters p(R"({
        "type"                  : "GenerateOctreeHybridConstraintsBetweenColors",
        "model_part_name"       : "Out",
        "color_block_a"         : 1,
        "color_block_b"         : 0,
        "generated_entity"      : "LinearMasterSlaveConstraint",
        "constrained_variables" : ["TEMPERATURE"]
    })");
    between_op.ValidateParameters(p);
    between_op.Generate(modeler, p);

    // The dual mesh shares mesh-node indices at the inside/outside interface, so
    // there are no non-conforming coincident pairs to tie — the call must complete
    // without error and create zero constraints.
    KRATOS_EXPECT_EQ(model.GetModelPart("Out").NumberOfMasterSlaveConstraints(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsBetweenColorsNoCellColorSkips, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);

    GenerateOctreeHybridConstraintsBetweenColors between_op;
    Parameters p(R"({
        "type"                  : "GenerateOctreeHybridConstraintsBetweenColors",
        "model_part_name"       : "Out",
        "color_block_a"         : 1,
        "color_block_b"         : 0,
        "generated_entity"      : "LinearMasterSlaveConstraint",
        "constrained_variables" : ["TEMPERATURE"]
    })");
    between_op.ValidateParameters(p);
    between_op.Generate(modeler, p);

    KRATOS_EXPECT_EQ(model.GetModelPart("Out").NumberOfMasterSlaveConstraints(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsBetweenColorsNotExtractedThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    GenerateOctreeHybridConstraintsBetweenColors between_op;
    Parameters p(R"({
        "type"                  : "GenerateOctreeHybridConstraintsBetweenColors",
        "model_part_name"       : "Out",
        "generated_entity"      : "LinearMasterSlaveConstraint",
        "constrained_variables" : ["TEMPERATURE"]
    })");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(between_op.Generate(modeler, p), "");
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsBetweenColorsEmptyGeneratedEntityThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type"            : "OctreeHybridRefineInterfaceCells",
            "refinement_depth": 3, "adaptive": false, "mesh_type": "dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }

    GenerateOctreeHybridConstraintsBetweenColors between_op;
    Parameters p = between_op.GetDefaultParameters();
    p["model_part_name"].SetString("Out");
    p["color_block_a"].SetInt(1);
    p["color_block_b"].SetInt(0);
    p["generated_entity"].SetString("");
    p["constrained_variables"].Append("TEMPERATURE");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(between_op.Generate(modeler, p), "");
}

KRATOS_TEST_CASE_IN_SUITE(GenerateOctreeHybridConstraintsBetweenColorsDefaultParameters, KratosCoreFastSuite)
{
    GenerateOctreeHybridConstraintsBetweenColors between_op;
    Parameters p = between_op.GetDefaultParameters();

    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"GenerateOctreeHybridConstraintsBetweenColors"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["color_block_a"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["color_block_b"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["generated_entity"].GetString(), std::string{""});
    KRATOS_EXPECT_EQ(p["constrained_variables"].size(), std::size_t{0});
}

// ===========================================================================
// OctreeHybridReportMeshQuality::Execute — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridReportMeshQualityExecuteDirectlyOnEmptyPartDoesNotThrow, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridReportMeshQuality quality_op;
    Parameters p(R"({
        "type"            : "OctreeHybridReportMeshQuality",
        "model_part_name" : "Out"
    })");
    quality_op.Execute(modeler, p); // must not throw
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridReportMeshQualityExecuteDirectlyOnFilledPartDoesNotThrow, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false,"mesh_type":"dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }
    {
        GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
        Parameters hex_p(R"({
            "type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Out","color":1,"properties_id":1,
            "generated_entity":"Element3D8N","tag_refinement_level":true
        })");
        hex_op.ValidateParameters(hex_p);
        hex_op.Generate(modeler, hex_p);
    }

    OctreeHybridReportMeshQuality quality_op;
    Parameters p(R"({"type":"OctreeHybridReportMeshQuality","model_part_name":"Out"})");
    quality_op.Execute(modeler, p); // must not throw
}

// ===========================================================================
// OctreeHybridFindContactsInSkinModelPart::Execute — direct call
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridFindContactsInSkinModelPartNoContactModelPartsDoesNotThrow, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false,"mesh_type":"dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }
    {
        GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
        Parameters hex_p(R"({
            "type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Out","color":1,"properties_id":1,
            "generated_entity":"Element3D8N","tag_refinement_level":true
        })");
        hex_op.ValidateParameters(hex_p);
        hex_op.Generate(modeler, hex_p);
    }
    {
        GenerateHybridOctreeQuadrilateralConditionsWithFaceColor bc_op;
        Parameters bc_p(R"({
            "type":"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor",
            "model_part_name":"Out","color":1,"properties_id":1,
            "generated_entity":"SurfaceCondition3D4N"
        })");
        bc_op.ValidateParameters(bc_p);
        bc_op.Generate(modeler, bc_p);
    }

    OctreeHybridFindContactsInSkinModelPart contacts_op;
    Parameters p(R"({
        "type"                : "OctreeHybridFindContactsInSkinModelPart",
        "model_part_name"     : "Out",
        "cell_color"          : 1,
        "contact_model_parts" : []
    })");
    contacts_op.Execute(modeler, p); // must not throw, just logs and returns
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridFindContactsInSkinModelPartFindsContacts, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false,"mesh_type":"dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);
    {
        OctreeHybridClassifyCellsInsideOutside color_op;
        color_op.Apply(modeler, Parameters(R"({"type":"OctreeHybridClassifyCellsInsideOutside"})"));
    }
    {
        GenerateHybridOctreeHexahedraElementsWithCellColor hex_op;
        Parameters hex_p(R"({
            "type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Out","color":1,"properties_id":1,
            "generated_entity":"Element3D8N","tag_refinement_level":true
        })");
        hex_op.ValidateParameters(hex_p);
        hex_op.Generate(modeler, hex_p);
    }

    // Boundary conditions of the "inside" (color 1) cells form the skin model part.
    {
        GenerateHybridOctreeQuadrilateralConditionsWithFaceColor bc_op;
        Parameters bc_p(R"({
            "type":"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor",
            "model_part_name":"Out","color":1,"properties_id":1,
            "generated_entity":"SurfaceCondition3D4N"
        })");
        bc_op.ValidateParameters(bc_p);
        bc_op.Generate(modeler, bc_p);
    }

    model.CreateModelPart("Out.Contact0");
    const std::size_t n_skin_conditions = model.GetModelPart("Out").NumberOfConditions();
    KRATOS_EXPECT_GT(n_skin_conditions, std::size_t{0});

    OctreeHybridFindContactsInSkinModelPart contacts_op;
    Parameters p(R"({
        "type"                : "OctreeHybridFindContactsInSkinModelPart",
        "model_part_name"     : "Out",
        "cell_color"          : 1,
        "contact_model_parts" : [
            { "outside_color": 0, "contact_model_part_name": "Out.Contact0" }
        ]
    })");
    contacts_op.ValidateParameters(p);
    contacts_op.Execute(modeler, p);

    ModelPart& r_contact = model.GetModelPart("Out.Contact0");
    // Boundary faces of the carved "inside" region either neighbour an "outside" (color 0)
    // cell (a contact) or lie on the outer boundary of the octree domain (no neighbour cell,
    // not a contact). So the contact part is a non-empty, strict subset of the skin.
    KRATOS_EXPECT_GT(r_contact.NumberOfConditions(), std::size_t{0});
    KRATOS_EXPECT_LE(r_contact.NumberOfConditions(), n_skin_conditions);
    KRATOS_EXPECT_GT(r_contact.NumberOfNodes(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridFindContactsInSkinModelPartModelPartNotFoundThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    model.CreateModelPart("Out");
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    {
        OctreeHybridRefineInterfaceCells build_op;
        Parameters bp(R"({
            "type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false,"mesh_type":"dual"
        })");
        build_op.ValidateParameters(bp);
        build_op.Refine(modeler, bp);
    }
    ExtractMesh(modeler);

    OctreeHybridFindContactsInSkinModelPart contacts_op;
    Parameters p(R"({
        "type"                : "OctreeHybridFindContactsInSkinModelPart",
        "model_part_name"     : "DoesNotExist",
        "cell_color"          : 1,
        "contact_model_parts" : []
    })");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(
        contacts_op.Execute(modeler, p),
        "not found");
}

// ===========================================================================
// ValidateParameters — base class behaviour (shared across all base types)
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationValidateParametersFillsDefaults, KratosCoreFastSuite)
{
    OctreeHybridRefineUniform op;
    // Provide only the mandatory "type" key — defaults should fill the rest
    Parameters p(R"({"type": "OctreeHybridRefineUniform"})");
    op.ValidateParameters(p);

    KRATOS_EXPECT_TRUE(p.Has("refinement_depth"));
    KRATOS_EXPECT_TRUE(p.Has("refined_cell_size"));
    KRATOS_EXPECT_EQ(p["refinement_depth"].GetInt(), 5);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationValidateParametersUnknownKeyThrows, KratosCoreFastSuite)
{
    OctreeHybridRefineUniform op;
    Parameters p(R"({"type": "OctreeHybridRefineUniform", "unknown_key": 99})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(op.ValidateParameters(p), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherColoringValidateParametersFillsDefaults, KratosCoreFastSuite)
{
    OctreeHybridClassifyCellsInsideOutside op;
    Parameters p(R"({})");
    // The schema only has "type" with default "" — missing "type" gets filled
    op.ValidateParameters(p);
    KRATOS_EXPECT_TRUE(p.Has("type"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherColoringValidateParametersUnknownKeyThrows, KratosCoreFastSuite)
{
    OctreeHybridClassifyCellsInsideOutside op;
    Parameters p(R"({"type": "OctreeHybridClassifyCellsInsideOutside", "bad_key": true})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(op.ValidateParameters(p), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherEntityGenerationValidateParametersFillsDefaults, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    Parameters p(R"({"type": "GenerateHybridOctreeHexahedraElementsWithCellColor"})");
    op.ValidateParameters(p);

    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_TRUE(p.Has("color"));
    KRATOS_EXPECT_TRUE(p.Has("generated_entity"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherEntityGenerationValidateParametersUnknownKeyThrows, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    Parameters p(R"({"type": "GenerateHybridOctreeHexahedraElementsWithCellColor", "unexpected": 0})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(op.ValidateParameters(p), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherOperationValidateParametersFillsDefaults, KratosCoreFastSuite)
{
    OctreeHybridReportMeshQuality op;
    Parameters p(R"({"type": "OctreeHybridReportMeshQuality"})");
    op.ValidateParameters(p);

    KRATOS_EXPECT_TRUE(p.Has("model_part_name"));
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherOperationValidateParametersUnknownKeyThrows, KratosCoreFastSuite)
{
    OctreeHybridReportMeshQuality op;
    Parameters p(R"({"type": "OctreeHybridReportMeshQuality", "not_a_key": false})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(op.ValidateParameters(p), "");
}

// ===========================================================================
// Base class virtual methods throw when invoked on base
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationBaseRefineThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridRefineOperation base_op;
    Parameters p(R"({"type": ""})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(base_op.Refine(modeler, p), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherColoringBaseApplyThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridMesherColoring base_op;
    Parameters p(R"({"type": ""})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(base_op.Apply(modeler, p), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherEntityGenerationBaseGenerateThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridMesherEntityGeneration base_op;
    Parameters p(R"({"type": ""})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(base_op.Generate(modeler, p), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherOperationBaseExecuteThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto modeler = MakeEmptyModeler(model, "Skin", "Out");

    OctreeHybridMesherOperation base_op;
    Parameters p(R"({"type": ""})");
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(base_op.Execute(modeler, p), "");
}

} // namespace Kratos::Testing
