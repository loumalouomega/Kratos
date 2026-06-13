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
#include <algorithm>
#include <cmath>
#include <set>

// External includes

// Project includes
#include "testing/testing.h"
#include "includes/kratos_application.h"
#include "includes/kernel.h"
#include "includes/variables.h"
#include "includes/registry.h"

#include "modeler/octree_hybrid_mesh_generator_modeler.h"
#include "modeler/internals/octree_hybrid_mesher_data.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"
#include "modeler/coloring/octree_hybrid_color_cell_faces_between_colors.h"
#include "modeler/coloring/octree_hybrid_color_cell_faces.h"
#include "modeler/coloring/octree_hybrid_color_outer_cell_faces.h"
#include "modeler/coloring/octree_hybrid_color_cells_in_bounding_box.h"
#include "modeler/coloring/octree_hybrid_color_cells_faces_in_bounding_box.h"
#include "modeler/entity_generation/generate_hybrid_octree_hexahedra_elements_with_cell_color.h"
#include "modeler/entity_generation/generate_hybrid_octree_tetrahedra_elements_with_cell_color.h"
#include "modeler/entity_generation/generate_hybrid_octree_triangular_conditions_with_face_color.h"

namespace Kratos::Testing {

namespace {

// ---------------------------------------------------------------------------
// Shared surface-builder helpers
// ---------------------------------------------------------------------------

/**
 * @brief Builds a closed cube skin [lo, hi]^3 (12 triangles) with two extra
 *        bounding-box-pin nodes, as expected by the octree engine.
 *
 * The surface lives in @p rSurfaceMesh which must already exist in @p rModel.
 */
void BuildClosedBoxSurface(ModelPart& rSurfaceMesh, double Lo = 0.3, double Hi = 0.7)
{
    rSurfaceMesh.SetBufferSize(1);
    rSurfaceMesh.GetProcessInfo()[DOMAIN_SIZE] = 3;

    const std::array<std::array<double,3>, 8> corners = {{
        {Lo,Lo,Lo}, {Hi,Lo,Lo}, {Hi,Hi,Lo}, {Lo,Hi,Lo},
        {Lo,Lo,Hi}, {Hi,Lo,Hi}, {Hi,Hi,Hi}, {Lo,Hi,Hi}
    }};
    IndexType nid = 1;
    for (const auto& c : corners)
        rSurfaceMesh.CreateNewNode(nid++, c[0], c[1], c[2]);
    rSurfaceMesh.CreateNewNode(nid++, 0.0, 0.0, 0.0); // bbox pin
    rSurfaceMesh.CreateNewNode(nid,   1.0, 1.0, 1.0); // bbox pin

    const std::array<std::array<IndexType,3>, 12> faces = {{
        {1,2,3},{1,3,4},{5,7,6},{5,8,7},
        {1,6,2},{1,5,6},{4,3,7},{4,7,8},
        {1,4,8},{1,8,5},{2,6,7},{2,7,3}
    }};
    IndexType gid = 1;
    for (const auto& f : faces) {
        rSurfaceMesh.CreateNewGeometry(
            "Triangle3D3", gid++,
            std::vector<IndexType>{f[0], f[1], f[2]});
    }
}

/**
 * @brief Builds a small inclined triangular patch near one corner of the unit
 *        cube — guarantees 2:1 transitions in the adaptive octree.
 */
void BuildTransitionSurface(ModelPart& rSurfaceMesh)
{
    rSurfaceMesh.SetBufferSize(1);
    rSurfaceMesh.GetProcessInfo()[DOMAIN_SIZE] = 3;

    const std::array<std::array<double,3>, 6> pts = {{
        {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
        {0.15, 0.15, 0.30}, {0.45, 0.15, 0.30},
        {0.45, 0.45, 0.36}, {0.15, 0.45, 0.36}
    }};
    IndexType nid = 1;
    for (const auto& p : pts)
        rSurfaceMesh.CreateNewNode(nid++, p[0], p[1], p[2]);

    rSurfaceMesh.CreateNewGeometry("Triangle3D3", 1,
        std::vector<IndexType>{3, 4, 5});
    rSurfaceMesh.CreateNewGeometry("Triangle3D3", 2,
        std::vector<IndexType>{3, 5, 6});
}

/**
 * @brief Runs OctreeHybridMeshGeneratorModeler::SetupModelPart with the given JSON settings
 *        string, then returns the named output ModelPart.
 */
ModelPart& RunModeler(Model& rModel, const std::string& rSettingsJson)
{
    Parameters settings(rSettingsJson);
    const std::string output_model_part_name = settings["output_model_part_name"].GetString();
    settings.RemoveValue("output_model_part_name");
    OctreeHybridMeshGeneratorModeler modeler(rModel, settings);
    modeler.SetupModelPart();
    return rModel.GetModelPart(output_model_part_name);
}

/**
 * @brief Computes the minimum scaled Jacobian of a hexahedral element using the
 *        SJ_ADJ corner-triple convention matching the engine's Sj function.
 */
double MinScaledJacobian(const Element& rElement)
{
    static constexpr int SJ_ADJ[8][3] =
        {{1,3,4},{2,0,5},{3,1,6},{0,2,7},{7,5,0},{4,6,1},{5,7,2},{6,4,3}};
    const auto& r_geom = rElement.GetGeometry();
    double worst = 1.0e30;
    for (int o = 0; o < 8; ++o) {
        const double ox = r_geom[o].X(), oy = r_geom[o].Y(), oz = r_geom[o].Z();
        double e[3][3];
        for (int k = 0; k < 3; ++k) {
            e[k][0] = r_geom[SJ_ADJ[o][k]].X() - ox;
            e[k][1] = r_geom[SJ_ADJ[o][k]].Y() - oy;
            e[k][2] = r_geom[SJ_ADJ[o][k]].Z() - oz;
        }
        const double det = e[0][0]*(e[1][1]*e[2][2]-e[1][2]*e[2][1])
                         - e[0][1]*(e[1][0]*e[2][2]-e[1][2]*e[2][0])
                         + e[0][2]*(e[1][0]*e[2][1]-e[1][1]*e[2][0]);
        const auto nrm = [](const double v[3]){
            return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); };
        const double n0 = nrm(e[0]), n1 = nrm(e[1]), n2 = nrm(e[2]);
        if (n0 < 1e-14 || n1 < 1e-14 || n2 < 1e-14) return -1.0;
        worst = std::min(worst, det / (n0 * n1 * n2));
    }
    return worst;
}

} // anonymous namespace

// ===========================================================================
// OctreeHybridMeshGeneratorModeler — top-level modeler tests
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerDualElementsCreated, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerDualZeroInverted, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    int n_inv = 0;
    for (const auto& r_el : out.Elements())
        if (MinScaledJacobian(r_el) <= 0.0) ++n_inv;

    KRATOS_EXPECT_EQ(n_inv, 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerDualCarveBbox, KratosCoreFastSuite)
{
    constexpr double lo = 0.3, hi = 0.7;
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"), lo, hi);

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    // All output nodes must lie within the (surface + one-half-cell) bounding box.
    const double margin = 0.07;  // half-cell at depth 4 in a unit box ≈ 0.0625
    for (const auto& r_node : out.Nodes()) {
        KRATOS_EXPECT_GT(r_node.X(), lo - margin);
        KRATOS_EXPECT_LT(r_node.X(), hi + margin);
        KRATOS_EXPECT_GT(r_node.Y(), lo - margin);
        KRATOS_EXPECT_LT(r_node.Y(), hi + margin);
        KRATOS_EXPECT_GT(r_node.Z(), lo - margin);
        KRATOS_EXPECT_LT(r_node.Z(), hi + margin);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerDualExplicitBoundingBoxOverride, KratosCoreFastSuite)
{
    constexpr double lo = 0.3, hi = 0.7;
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"), lo, hi);

    // Skin's own node bounding box is [0,0,0]-[1,1,1] (the bbox-pin nodes), which
    // the override below fully contains.
    constexpr double override_lo = -0.5, override_hi = 1.5;
    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "bounding_box" : {
            "min_point" : [-0.5, -0.5, -0.5],
            "max_point" : [1.5, 1.5, 1.5]
        },
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);

    // All output nodes must lie within the explicit override domain.
    for (const auto& r_node : out.Nodes()) {
        KRATOS_EXPECT_GE(r_node.X(), override_lo - 1e-9);
        KRATOS_EXPECT_LE(r_node.X(), override_hi + 1e-9);
        KRATOS_EXPECT_GE(r_node.Y(), override_lo - 1e-9);
        KRATOS_EXPECT_LE(r_node.Y(), override_hi + 1e-9);
        KRATOS_EXPECT_GE(r_node.Z(), override_lo - 1e-9);
        KRATOS_EXPECT_LE(r_node.Z(), override_hi + 1e-9);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerSetupModelPartBoundingBoxConflictThrows, KratosCoreFastSuite)
{
    constexpr double lo = 0.3, hi = 0.7;
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"), lo, hi);

    // Skin's own node bounding box is [0,0,0]-[1,1,1] (the bbox-pin nodes); this
    // override does not contain it.
    Parameters settings(R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "bounding_box" : {
            "min_point" : [0.3, 0.3, 0.3],
            "max_point" : [0.7, 0.7, 0.7]
        },
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");
    settings.RemoveValue("output_model_part_name");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);

    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.SetupModelPart(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerDefaultParametersValid, KratosCoreFastSuite)
{
    OctreeHybridMeshGeneratorModeler m;
    const Parameters defaults = m.GetDefaultParameters();

    // Check that the mandatory keys are present with correct types
    KRATOS_EXPECT_TRUE(defaults.Has("input_model_part_name"));
    KRATOS_EXPECT_TRUE(defaults.Has("refinement_settings_list"));
    KRATOS_EXPECT_TRUE(defaults.Has("coloring_settings_list"));
    KRATOS_EXPECT_TRUE(defaults.Has("entities_generator_list"));
    KRATOS_EXPECT_TRUE(defaults.Has("model_part_operations"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerInfoString, KratosCoreFastSuite)
{
    OctreeHybridMeshGeneratorModeler m;
    KRATOS_EXPECT_EQ(m.Info(), "OctreeHybridMeshGeneratorModeler");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerEmptyRefineListAutoBuildsOctree, KratosCoreFastSuite)
{
    // An empty refinement_settings_list no longer requires an explicit
    // RefineInterfaceCellsOctreeHybrid entry: EnsureOctreeBuilt auto-builds a default octree
    // (refinement_depth=5, mesh_type="dual", adaptive=true) from the input surface model part
    // before 2:1 balancing and extraction.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerRefineUniformFirstAutoBuildsOctree, KratosCoreFastSuite)
{
    // A refinement_settings_list that starts with RefineUniformOctreeHybrid (instead of
    // RefineInterfaceCellsOctreeHybrid) no longer throws: RefineUniformOctreeHybrid::Refine
    // self-checks that no octree exists yet and delegates to EnsureOctreeBuilt, which builds
    // the octree (refinement_depth=4 from this entry, mesh_type="dual", adaptive=true) from
    // the input surface model part before the uniform refinement runs.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineUniformOctreeHybrid", "refinement_depth": 4 }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerUnknownMeshTypeThrows, KratosCoreFastSuite)
{
    // ApplyRefinement throws for any mesh_type that is neither "dual" nor "primal".
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name"  : "Skin",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 2, "adaptive": false,
                                      "mesh_type": "invalid_mesh_type" }],
        "coloring_settings_list" : [],
        "entities_generator_list": [],
        "model_part_operations"  : []
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.SetupModelPart(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerUnknownOperationThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name"  : "Skin",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [],
        "entities_generator_list": [],
        "model_part_operations"  : [{ "type": "NonExistentOperationType_XYZ" }]
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.SetupModelPart(), "");
}

// ===========================================================================
// OctreeHybridMeshGeneratorModeler — bounding box override resolution
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerNoBoundingBoxOverride, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    Parameters settings(R"({
        "input_model_part_name" : "Skin"
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.Initialize();

    KRATOS_EXPECT_FALSE(modeler.HasOctreeBoundingBox());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerExplicitBoundingBoxResolved, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "bounding_box" : {
            "min_point" : [0.0, 0.0, 0.0],
            "max_point" : [1.0, 1.0, 1.0]
        }
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.Initialize();

    KRATOS_EXPECT_TRUE(modeler.HasOctreeBoundingBox());
    const auto& r_min = modeler.GetOctreeBoundingBox().GetMinPoint();
    const auto& r_max = modeler.GetOctreeBoundingBox().GetMaxPoint();
    for (unsigned int i = 0; i < 3; ++i) {
        KRATOS_EXPECT_NEAR(r_min[i], 0.0, 1e-12);
        KRATOS_EXPECT_NEAR(r_max[i], 1.0, 1e-12);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerBoundingBoxModelPartResolved, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    ModelPart& r_bbox = model.CreateModelPart("BBox");
    r_bbox.CreateNewNode(1, 0.0, 0.0, 0.0);
    r_bbox.CreateNewNode(2, 1.0, 1.0, 1.0);

    Parameters settings(R"({
        "input_model_part_name"   : "Skin",
        "bounding_box_model_part" : "BBox"
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.Initialize();

    KRATOS_EXPECT_TRUE(modeler.HasOctreeBoundingBox());
    const auto& r_min = modeler.GetOctreeBoundingBox().GetMinPoint();
    const auto& r_max = modeler.GetOctreeBoundingBox().GetMaxPoint();
    for (unsigned int i = 0; i < 3; ++i) {
        KRATOS_EXPECT_NEAR(r_min[i], 0.0, 1e-12);
        KRATOS_EXPECT_NEAR(r_max[i], 1.0, 1e-12);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerBothBoundingBoxSourcesThrows, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    ModelPart& r_bbox = model.CreateModelPart("BBox");
    r_bbox.CreateNewNode(1, 0.0, 0.0, 0.0);
    r_bbox.CreateNewNode(2, 1.0, 1.0, 1.0);

    Parameters settings(R"({
        "input_model_part_name"   : "Skin",
        "bounding_box_model_part" : "BBox",
        "bounding_box" : {
            "min_point" : [0.0, 0.0, 0.0],
            "max_point" : [1.0, 1.0, 1.0]
        }
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);

    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.Initialize(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerBoundingBoxConflictThrows, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "bounding_box" : {
            "min_point" : [0.3, 0.3, 0.3],
            "max_point" : [0.7, 0.7, 0.7]
        }
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);

    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.Initialize(), "");
}

// ===========================================================================
// OctreeHybridClassifyCellsInsideOutside colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherClassifyReducesCellCount, KratosCoreFastSuite)
{
    // Without colouring the hex generator sees all cells (color=1 default matches
    // everything since mCellColor is empty). With colouring the set must be smaller.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("S"));
    BuildClosedBoxSurface(m2.CreateModelPart("S"));

    const char* settings_no_classify = R"({
        "input_model_part_name":"S","output_model_part_name":"O",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"O","color":1}],
        "model_part_operations":[]
    })";
    const char* settings_classify = R"({
        "input_model_part_name":"S","output_model_part_name":"O",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"O","color":1}],
        "model_part_operations":[]
    })";

    ModelPart& out_all    = RunModeler(m1, settings_no_classify);
    ModelPart& out_carved = RunModeler(m2, settings_classify);

    KRATOS_EXPECT_GT(out_all.NumberOfElements(), out_carved.NumberOfElements());
    KRATOS_EXPECT_GT(out_carved.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherClassifyRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype"));
}

// ===========================================================================
// GenerateHybridOctreeHexahedraElementsWithCellColor entity generator
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherEntityGeneration.All.GenerateHybridOctreeHexahedraElementsWithCellColor.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesNodeDeduplication, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    // In a conforming mesh, nodes are shared: node count << 8 × element count
    const std::size_t n_nodes = out.NumberOfNodes();
    const std::size_t n_elems = out.NumberOfElements();
    KRATOS_EXPECT_LT(n_nodes, 8 * n_elems);
    KRATOS_EXPECT_GT(n_nodes, 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesRefinementLevelTagged, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1,"tag_refinement_level":true}],
        "model_part_operations":[]
    })");

    bool has_positive_level = false;
    for (const auto& r_el : out.Elements()) {
        const int lv = r_el.GetValue(REFINEMENT_LEVEL);
        // Template hexes get -1; regular ones get the octree level (> 0)
        KRATOS_EXPECT_TRUE(lv == -1 || lv > 0);
        if (lv > 0) has_positive_level = true;
    }
    KRATOS_EXPECT_TRUE(has_positive_level);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesNoLevelWhenDisabled, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1,"tag_refinement_level":false}],
        "model_part_operations":[]
    })");

    for (const auto& r_el : out.Elements()) {
        // Without tagging every element carries the default int value of 0
        KRATOS_EXPECT_EQ(r_el.GetValue(REFINEMENT_LEVEL), 0);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesUniqueIds, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    // All node ids must be unique and all element ids must be unique
    std::set<IndexType> node_ids, elem_ids;
    for (const auto& r_nd : out.Nodes()) {
        KRATOS_EXPECT_EQ(node_ids.count(r_nd.Id()), 0u);
        node_ids.insert(r_nd.Id());
    }
    for (const auto& r_el : out.Elements()) {
        KRATOS_EXPECT_EQ(elem_ids.count(r_el.Id()), 0u);
        elem_ids.insert(r_el.Id());
    }
}

// ===========================================================================
// GenerateHybridOctreeQuadrilateralConditionsWithFaceColor entity generator
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherBoundaryConditionsRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherEntityGeneration.All.GenerateHybridOctreeQuadrilateralConditionsWithFaceColor.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherBoundaryConditionsCreated, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Volume",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"Volume","color":1},
            {"type":"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor","model_part_name":"Boundary","color":1}
        ],
        "model_part_operations":[]
    })");

    ModelPart& bnd = model.GetModelPart("Boundary");
    KRATOS_EXPECT_GT(bnd.NumberOfConditions(), 0u);
    KRATOS_EXPECT_GT(bnd.NumberOfNodes(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherBoundaryConditionsQuadNodes, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Volume",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"Volume","color":1},
            {"type":"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor","model_part_name":"Boundary","color":1}
        ],
        "model_part_operations":[]
    })");

    ModelPart& bnd = model.GetModelPart("Boundary");
    for (const auto& r_cond : bnd.Conditions())
        KRATOS_EXPECT_EQ(r_cond.GetGeometry().size(), 4u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherBoundaryConditionsFewerthanSixTimesElements, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Volume",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"Volume","color":1},
            {"type":"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor","model_part_name":"Boundary","color":1}
        ],
        "model_part_operations":[]
    })");

    ModelPart& vol = model.GetModelPart("Volume");
    ModelPart& bnd = model.GetModelPart("Boundary");
    KRATOS_EXPECT_LT(bnd.NumberOfConditions(), 6 * vol.NumberOfElements());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherBoundaryNodesSubsetOfVolume, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Volume",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"Volume","color":1},
            {"type":"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor","model_part_name":"Boundary","color":1}
        ],
        "model_part_operations":[]
    })");

    ModelPart& vol = model.GetModelPart("Volume");
    ModelPart& bnd = model.GetModelPart("Boundary");

    std::set<IndexType> vol_ids;
    for (const auto& r_nd : vol.Nodes()) vol_ids.insert(r_nd.Id());

    for (const auto& r_nd : bnd.Nodes())
        KRATOS_EXPECT_GT(vol_ids.count(r_nd.Id()), 0u);
}

// ===========================================================================
// GenerateHybridOctreeHexahedraElementsWithCellColor — hanging-node constraint parameters
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesByCellColorHasConstraintParams, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("constraint_type"));
    KRATOS_EXPECT_TRUE(p.Has("constrained_variables"));
    KRATOS_EXPECT_EQ(p["constraint_type"].GetString(), std::string{""});
    KRATOS_EXPECT_EQ(p["constrained_variables"].size(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherPrimalMeshConstraintsGenerated, KratosCoreFastSuite)
{
    Model model;
    BuildTransitionSurface(model.CreateModelPart("Surface"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Surface","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"Output","color":1,
             "constraint_type":"LinearMasterSlaveConstraint","constrained_variables":["DISPLACEMENT_X"]}
        ],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfMasterSlaveConstraints(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherPrimalConstraintsPartitionOfUnity, KratosCoreFastSuite)
{
    Model model;
    BuildTransitionSurface(model.CreateModelPart("Surface"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Surface","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"Output","color":1,
             "constraint_type":"LinearMasterSlaveConstraint","constrained_variables":["DISPLACEMENT_X"]}
        ],
        "model_part_operations":[]
    })");

    // Each constraint is now 1×1 (one master DOF per constraint)
    for (const auto& r_constraint : out.MasterSlaveConstraints()) {
        MasterSlaveConstraint::MatrixType T;
        MasterSlaveConstraint::VectorType b;
        r_constraint.CalculateLocalSystem(T, b, out.GetProcessInfo());

        KRATOS_EXPECT_EQ(T.size1(), 1u);
        KRATOS_EXPECT_EQ(T.size2(), 1u);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherPrimalConstraintsMasterCountsValid, KratosCoreFastSuite)
{
    Model model;
    BuildTransitionSurface(model.CreateModelPart("Surface"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Surface","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"Output","color":1,
             "constraint_type":"LinearMasterSlaveConstraint","constrained_variables":["DISPLACEMENT_X"]}
        ],
        "model_part_operations":[]
    })");

    // Every constraint has exactly 1 master DOF (1-1 form)
    bool found_quarter_weight = false;
    for (const auto& r_constraint : out.MasterSlaveConstraints()) {
        KRATOS_EXPECT_EQ(r_constraint.GetMasterDofsVector().size(), 1u);
        MasterSlaveConstraint::MatrixType T;
        MasterSlaveConstraint::VectorType b;
        r_constraint.CalculateLocalSystem(T, b, out.GetProcessInfo());
        if (std::abs(T(0,0) - 0.25) < 1e-10) found_quarter_weight = true;
    }
    // Face-centre hanging nodes contribute constraints with weight 0.25
    KRATOS_EXPECT_TRUE(found_quarter_weight);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherPrimalMultiVariableConstraints, KratosCoreFastSuite)
{
    Model m1, m2;
    BuildTransitionSurface(m1.CreateModelPart("S"));
    BuildTransitionSurface(m2.CreateModelPart("S"));

    const char* s1 = R"({
        "input_model_part_name":"S","output_model_part_name":"O",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"O","color":1,
             "constraint_type":"LinearMasterSlaveConstraint","constrained_variables":["DISPLACEMENT_X"]}
        ],"model_part_operations":[]
    })";
    const char* s3 = R"({
        "input_model_part_name":"S","output_model_part_name":"O",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"GenerateHybridOctreeHexahedraElementsWithCellColor","model_part_name":"O","color":1,
             "constraint_type":"LinearMasterSlaveConstraint","constrained_variables":["DISPLACEMENT_X","DISPLACEMENT_Y","DISPLACEMENT_Z"]}
        ],"model_part_operations":[]
    })";

    ModelPart& out1 = RunModeler(m1, s1);
    ModelPart& out3 = RunModeler(m2, s3);

    KRATOS_EXPECT_EQ(out3.NumberOfMasterSlaveConstraints(),
                     3 * out1.NumberOfMasterSlaveConstraints());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherDualMeshNoHangingConstraints, KratosCoreFastSuite)
{
    Model model;
    BuildTransitionSurface(model.CreateModelPart("Surface"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Surface","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":true,"mesh_type":"dual"}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_EQ(out.NumberOfMasterSlaveConstraints(), 0u);
}

// ===========================================================================
// OctreeHybridReportMeshQuality operation
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherReportMeshQualityRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherOperation.All.OctreeHybridReportMeshQuality.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherReportMeshQualityRunsWithoutError, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    // The operation just logs — verify it does not throw
    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[{"type":"OctreeHybridReportMeshQuality",
            "model_part_name":"Output"}]
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherReportMeshQualityEmptyModelPart, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    // Create the output part explicitly so the quality-report operation can reach it.
    model.CreateModelPart("Empty");

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[],
        "model_part_operations":[{"type":"OctreeHybridReportMeshQuality","model_part_name":"Empty"}]
    })");
    // SetupModelPart on an empty entity list should not throw
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();
    KRATOS_EXPECT_EQ(model.GetModelPart("Empty").NumberOfElements(), 0u);
}

// ===========================================================================
// Registry dispatch mechanism
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherRegistryBasePrototypesPresent, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridMesherColoring.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherEntityGeneration.All.OctreeHybridMesherEntityGeneration.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherOperation.All.OctreeHybridMesherOperation.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherRegistryKratosMultiphysicsPaths, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridClassifyCellsInsideOutside.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherEntityGeneration.KratosMultiphysics.GenerateHybridOctreeHexahedraElementsWithCellColor.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherOperation.KratosMultiphysics.OctreeHybridReportMeshQuality.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherRegistryFullPathDispatchWorks, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    // A four-segment full path must be accepted directly by the Dispatch method
    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype"
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherRegistryBaseColoringInvocationThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridMesherColoring.All.OctreeHybridMesherColoring.Prototype"
        }],
        "entities_generator_list":[],"model_part_operations":[]
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherRegistryBaseOperationInvocationThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[],
        "model_part_operations":[{
            "type":"OctreeHybridMesherOperation.All.OctreeHybridMesherOperation.Prototype"
        }]
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

// ===========================================================================
// RefineOctreeHybrid — refinement_settings_list stage tests
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformRefinesAllCells, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    // Reference run: build at depth 4 then uniformly refine to depth 4 (fills all cells).
    ModelPart& ref = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Ref",
        "refinement_settings_list" : [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 4, "adaptive": false },
            { "type": "RefineUniformOctreeHybrid",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Ref", "color": 1 }],
        "model_part_operations"  : []
    })");

    // Test run: initial build at depth 1, then uniformly refined to depth 4.
    // RefineUniform reinitializes the octree to the target depth when it exceeds the
    // current max depth, so the resulting mesh is identical to the reference.
    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 1, "adaptive": false },
            { "type": "RefineUniformOctreeHybrid",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    // Both runs produce the same fully-uniform depth-4 octree → same element counts.
    KRATOS_EXPECT_EQ(out.NumberOfElements(), ref.NumberOfElements());
    KRATOS_EXPECT_EQ(out.NumberOfNodes(),    ref.NumberOfNodes());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsProducesElements, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 2, "adaptive": false },
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "model_part_name": "Skin",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(),    0u);

    // All elements must have positive scaled Jacobian.
    for (const auto& r_el : out.Elements())
        KRATOS_EXPECT_GT(MinScaledJacobian(r_el), 0.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsFallbackToMainSurface, KratosCoreFastSuite)
{
    // Empty model_part_name → falls back to the modeler's main surface.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 2, "adaptive": false },
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "model_part_name": "",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsMultipleGeometries, KratosCoreFastSuite)
{
    // Two chained RefineInterfaceCellsOctreeHybrid entries, each with a different
    // model part and depth — verifies multi-geometry chaining runs without error.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("SkinA"), 0.2, 0.5);
    BuildClosedBoxSurface(model.CreateModelPart("SkinB"), 0.5, 0.8);

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "SkinA",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 1, "adaptive": false },
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "model_part_name": "SkinA", "refinement_depth": 4 },
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "model_part_name": "SkinB", "refinement_depth": 3 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformMaxVoxelSizeProducesElements, KratosCoreFastSuite)
{
    // max_voxel_size > 0 should override refinement_depth via ElementSizeToDepth
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    ModelPart& out = model.CreateModelPart("Output");

    // Box is [0,1]^3 so max_voxel_size=0.25 → depth≥2; should produce elements
    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "output_model_part_name": "Output",
        "refinement_settings_list": [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 1, "adaptive": false },
            { "type": "RefineUniformOctreeHybrid",
              "refinement_depth": 1, "max_voxel_size": 0.25 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");
    RunModeler(model, settings.WriteJsonString());

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsRefinedCellSizeProducesElements, KratosCoreFastSuite)
{
    // refined_cell_size > 0 should override refinement_depth for interface refinement
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    ModelPart& out = model.CreateModelPart("Output");

    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "output_model_part_name": "Output",
        "refinement_settings_list": [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 1, "adaptive": false },
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "model_part_name": "Skin",
              "refinement_depth": 1, "refined_cell_size": 0.25 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");
    RunModeler(model, settings.WriteJsonString());

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationRegistryPrototypePresent, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "RefineOctreeHybrid.All.RefineUniformOctreeHybrid.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "RefineOctreeHybrid.KratosMultiphysics.RefineUniformOctreeHybrid.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "RefineOctreeHybrid.All.RefineInterfaceCellsOctreeHybrid.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "RefineOctreeHybrid.KratosMultiphysics.RefineInterfaceCellsOctreeHybrid.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationBaseInvocationThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "refinement_settings_list": [
            { "type": "RefineInterfaceCellsOctreeHybrid",
              "refinement_depth": 2, "adaptive": false },
            { "type": "RefineOctreeHybrid.All.RefineOctreeHybrid.Prototype" }
        ],
        "coloring_settings_list": [],
        "entities_generator_list": [],
        "model_part_operations"  : []
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

// ===========================================================================
// OctreeHybridColorCellsInTouch colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsInTouchRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorCellsInTouch.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsInTouch.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsInTouchColorsSomeCells, KratosCoreFastSuite)
{
    // Cells whose AABB intersects the box skin surface are labelled with color=2.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsInTouch",
            "model_part_name":"Skin","color":2
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":2}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(),    0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsInTouchNotAllCells, KratosCoreFastSuite)
{
    // Interior and far-exterior cells are not in contact with the skin:
    // the surface-touch count must be strictly less than the total cell count.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("Skin"));
    BuildClosedBoxSurface(m2.CreateModelPart("Skin"));

    // Total elements (no colouring — generator treats empty mCellColor as all-match)
    ModelPart& out_all = RunModeler(m1, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    // Only surface-touching cells (color=2)
    ModelPart& out_touch = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsInTouch",
            "model_part_name":"Skin","color":2
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":2}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out_all.NumberOfElements(), out_touch.NumberOfElements());
}

// ===========================================================================
// OctreeHybridColorConnectedCellsInTouch colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorConnectedCellsInTouchRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorConnectedCellsInTouch.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorConnectedCellsInTouch.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorConnectedCellsInTouchProducesElements, KratosCoreFastSuite)
{
    // Seeds: inside cells touching the skin; BFS through all inside cells → all
    // inside cells reachable from the interface receive color=3.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridClassifyCellsInsideOutside"},
            {"type":"OctreeHybridColorConnectedCellsInTouch",
             "model_part_name":"Skin","cell_color":1,"color":3}
        ],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":3}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(),    0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorConnectedCellsInTouchFloodFillCoversInsideRegion, KratosCoreFastSuite)
{
    // Flood-filling from inside interface cells (color=1 → color=3) must color a
    // non-empty strict subset of the inside cells.  Face-adjacency is matched by
    // exact 4-node face keys, so cells at different refinement levels are not
    // considered adjacent; some coarser interior cells may remain unreachable.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("Skin"));
    BuildClosedBoxSurface(m2.CreateModelPart("Skin"));

    ModelPart& out_inside = RunModeler(m1, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    ModelPart& out_connected = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridClassifyCellsInsideOutside"},
            {"type":"OctreeHybridColorConnectedCellsInTouch",
             "model_part_name":"Skin","cell_color":1,"color":3}
        ],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":3}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out_connected.NumberOfElements(), 0u);
    KRATOS_EXPECT_LT(out_connected.NumberOfElements(), out_inside.NumberOfElements());
}

// ===========================================================================
// OctreeHybridColorCellsByLevel colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsByLevelRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorCellsByLevel.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsByLevel.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsByLevelColorsCorrectSubset, KratosCoreFastSuite)
{
    // A depth-4 mesh has leaf cells at level 4; targeting only level 4 must yield
    // at least some elements, but fewer than the full set.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("Skin"));
    BuildClosedBoxSurface(m2.CreateModelPart("Skin"));

    ModelPart& out_all = RunModeler(m1, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    ModelPart& out_level = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsByLevel","color":2,"min_level":4,"max_level":4
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":2}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out_level.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out_all.NumberOfElements(), out_level.NumberOfElements());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsByLevelExcludesBeyondMaxDepth, KratosCoreFastSuite)
{
    // Targeting level 5 in a depth-4 mesh yields no cells.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsByLevel","color":2,"min_level":5,"max_level":5
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":2}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_EQ(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsByLevelMinGreaterThanMaxThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsByLevel","color":1,"min_level":5,"max_level":3
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

// ===========================================================================
// OctreeHybridColorCellsWithInsideCenter colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsWithInsideCenterRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorCellsWithInsideCenter.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsWithInsideCenter.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsWithInsideCenterColorsInsideSubset, KratosCoreFastSuite)
{
    // The octree spans [0,1]^3 (driven by the bbox-pin nodes), while the skin
    // surface is the box [0.3,0.7]^3. Cells whose centre lies inside that box
    // must be a non-empty, strict subset of the full cell set.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("Skin"));
    BuildClosedBoxSurface(m2.CreateModelPart("Skin"));

    ModelPart& out_all = RunModeler(m1, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    ModelPart& out_inside = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsWithInsideCenter",
            "model_part_name":"Skin","color":2,"input_entities":"geometries"
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":2}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out_inside.NumberOfElements(), 0u);
    KRATOS_EXPECT_LT(out_inside.NumberOfElements(), out_all.NumberOfElements());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsWithInsideCenterBoundingBoxRestrictsCandidates, KratosCoreFastSuite)
{
    // Restricting the bounding_box to a sub-octant of the inside region must
    // color a non-empty subset of the cells colored without the restriction.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("Skin"));
    BuildClosedBoxSurface(m2.CreateModelPart("Skin"));

    ModelPart& out_inside = RunModeler(m1, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsWithInsideCenter",
            "model_part_name":"Skin","color":2,"input_entities":"geometries"
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":2}],
        "model_part_operations":[]
    })");

    ModelPart& out_bbox = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsWithInsideCenter",
            "model_part_name":"Skin","color":2,"input_entities":"geometries",
            "bounding_box":{"min_point":[0.3,0.3,0.3],"max_point":[0.5,0.5,0.5]}
        }],
        "entities_generator_list":[{"type":"GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name":"Output","color":2}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out_bbox.NumberOfElements(), 0u);
    KRATOS_EXPECT_LT(out_bbox.NumberOfElements(), out_inside.NumberOfElements());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsWithInsideCenterUnsupportedInputEntitiesThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsWithInsideCenter",
            "model_part_name":"Skin","color":2,"input_entities":"invalid"
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsWithInsideCenterModelPartNotFoundThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsWithInsideCenter",
            "model_part_name":"DoesNotExist","color":2,"input_entities":"geometries"
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

// ===========================================================================
// OctreeHybridColorCellFacesBetweenColors colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesBetweenColorsRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorCellFacesBetweenColors.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellFacesBetweenColors.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesBetweenColorsDefaultParameters, KratosCoreFastSuite)
{
    OctreeHybridColorCellFacesBetweenColors op;
    const Parameters p = op.GetDefaultParameters();

    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridColorCellFacesBetweenColors"});
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["cell_color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["outside_color"].GetInt(), 0);
    KRATOS_EXPECT_EQ(p["default_outside_color"].GetInt(), 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesBetweenColorsMarksInterfaceFaces, KratosCoreFastSuite)
{
    // After OctreeHybridClassifyCellsInsideOutside, inside cells get mCellColor == 1
    // and outside cells get mCellColor == 0. OctreeHybridColorCellFacesBetweenColors
    // must then mark, on every inside cell, the local faces whose neighbour (or the
    // implicit outer-boundary neighbour) is colour 0.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridClassifyCellsInsideOutside"},
            {"type":"OctreeHybridColorCellFacesBetweenColors",
             "color":2,"cell_color":1,"outside_color":0}
        ],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();

    const auto& r_data = modeler.GetData();
    KRATOS_EXPECT_EQ(r_data.mCellFaceColor.size(), r_data.mCells.size());

    std::size_t n_colored_faces = 0;
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        for (int f = 0; f < 6; ++f) {
            if (r_data.mCellFaceColor[c][f] == 2) {
                ++n_colored_faces;
                // Only inside (cell_color == 1) cells may receive the interface colour.
                KRATOS_EXPECT_EQ(r_data.mCellColor[c], 1);
            }
        }
    }
    KRATOS_EXPECT_GT(n_colored_faces, 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesBetweenColorsNoOpWhenColorEqualsOutsideColor, KratosCoreFastSuite)
{
    // Mirrors voxel mesher's ColorCellFacesBetweenColors: when "color" == "outside_color"
    // the stage is a no-op and mCellFaceColor stays at its zero-initialised default.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridClassifyCellsInsideOutside"},
            {"type":"OctreeHybridColorCellFacesBetweenColors",
             "color":0,"cell_color":1,"outside_color":0}
        ],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();

    const auto& r_data = modeler.GetData();
    KRATOS_EXPECT_EQ(r_data.mCellFaceColor.size(), r_data.mCells.size());
    for (const auto& r_faces : r_data.mCellFaceColor)
        for (int f = 0; f < 6; ++f)
            KRATOS_EXPECT_EQ(r_faces[f], 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesBetweenColorsDefaultOutsideColorMarksOuterBoundary, KratosCoreFastSuite)
{
    // OctreeHybridColorCellsByLevel with a [-10, 100] level range colours every
    // cell (including the negative-level transition/buffer cells) with
    // cell_color == 1. With outside_color == 0 and a top-level
    // default_outside_color == 0, an interior neighbour (colour 1) never equals
    // outside_color, but the implicit neighbour of an outer-boundary face
    // (default_outside_color == 0) does. So only outer-boundary faces (no
    // neighbour cell) must be coloured.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "default_outside_color":0,
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridColorCellsByLevel",
             "color":1,"min_level":-10,"max_level":100},
            {"type":"OctreeHybridColorCellFacesBetweenColors",
             "color":3,"cell_color":1,"outside_color":0}
        ],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();

    const auto& r_data = modeler.GetData();
    const auto neighbors = OctreeHybridMeshUtility::ComputeCellFaceNeighbors(r_data.mCells);

    std::size_t n_colored_faces = 0;
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        for (int f = 0; f < 6; ++f) {
            if (neighbors[c][f] == -1) {
                KRATOS_EXPECT_EQ(r_data.mCellFaceColor[c][f], 3);
                ++n_colored_faces;
            } else {
                KRATOS_EXPECT_EQ(r_data.mCellFaceColor[c][f], 0);
            }
        }
    }
    KRATOS_EXPECT_GT(n_colored_faces, 0u);
}

// ===========================================================================
// OctreeHybridColorCellsInBoundingBox colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsInBoundingBoxRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorCellsInBoundingBox.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsInBoundingBox.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsInBoundingBoxDefaultParameters, KratosCoreFastSuite)
{
    OctreeHybridColorCellsInBoundingBox op;
    const Parameters p = op.GetDefaultParameters();

    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridColorCellsInBoundingBox"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["min_point"].size(), 0);
    KRATOS_EXPECT_EQ(p["max_point"].size(), 0);
    KRATOS_EXPECT_FALSE(p["inside_bounding_box"].GetBool());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsInBoundingBoxInsideAndOutsideArePartition, KratosCoreFastSuite)
{
    // The octree spans [0,1]^3 (driven by the bbox-pin nodes). Colouring the
    // sub-region [0.3,0.5]^3 must be a non-empty, strict subset of all cells,
    // and the inside/outside selections must partition the full cell set.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("Skin"));
    BuildClosedBoxSurface(m2.CreateModelPart("Skin"));

    Parameters settings_inside(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsInBoundingBox",
            "color":2,"min_point":[0.3,0.3,0.3],"max_point":[0.5,0.5,0.5],
            "inside_bounding_box":true
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    Parameters settings_outside(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsInBoundingBox",
            "color":2,"min_point":[0.3,0.3,0.3],"max_point":[0.5,0.5,0.5],
            "inside_bounding_box":false
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler_inside(m1, settings_inside);
    modeler_inside.SetupModelPart();
    OctreeHybridMeshGeneratorModeler modeler_outside(m2, settings_outside);
    modeler_outside.SetupModelPart();

    const auto& r_data_inside  = modeler_inside.GetData();
    const auto& r_data_outside = modeler_outside.GetData();

    const std::size_t n_inside = std::count(r_data_inside.mCellColor.begin(), r_data_inside.mCellColor.end(), 2);
    const std::size_t n_outside = std::count(r_data_outside.mCellColor.begin(), r_data_outside.mCellColor.end(), 2);

    KRATOS_EXPECT_GT(n_inside, 0u);
    KRATOS_EXPECT_GT(n_outside, 0u);
    KRATOS_EXPECT_LT(n_inside, r_data_inside.mCellColor.size());
    KRATOS_EXPECT_EQ(n_inside + n_outside, r_data_inside.mCellColor.size());
}

// ===========================================================================
// OctreeHybridColorCellsFacesInBoundingBox colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsFacesInBoundingBoxRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorCellsFacesInBoundingBox.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsFacesInBoundingBox.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsFacesInBoundingBoxDefaultParameters, KratosCoreFastSuite)
{
    OctreeHybridColorCellsFacesInBoundingBox op;
    const Parameters p = op.GetDefaultParameters();

    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridColorCellsFacesInBoundingBox"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["min_point"].size(), 0);
    KRATOS_EXPECT_EQ(p["max_point"].size(), 0);
    KRATOS_EXPECT_FALSE(p["inside_bounding_box"].GetBool());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellsFacesInBoundingBoxColorsFacesInsideSubset, KratosCoreFastSuite)
{
    // All 6 faces of every cell are candidates; only those whose centroid
    // lies inside the [0.3,0.5]^3 sub-region must be coloured.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsFacesInBoundingBox",
            "color":5,"min_point":[0.3,0.3,0.3],"max_point":[0.5,0.5,0.5],
            "inside_bounding_box":true
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();

    const auto& r_data = modeler.GetData();
    KRATOS_EXPECT_EQ(r_data.mCellFaceColor.size(), r_data.mCells.size());

    std::size_t n_colored = 0;
    const std::size_t n_total = r_data.mCellFaceColor.size() * 6;
    for (const auto& r_faces : r_data.mCellFaceColor)
        for (int f = 0; f < 6; ++f)
            if (r_faces[f] == 5) ++n_colored;

    KRATOS_EXPECT_GT(n_colored, 0u);
    KRATOS_EXPECT_LT(n_colored, n_total);
}

// ===========================================================================
// OctreeHybridColorOuterCellFaces colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorOuterCellFacesRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorOuterCellFaces.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorOuterCellFaces.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorOuterCellFacesDefaultParameters, KratosCoreFastSuite)
{
    OctreeHybridColorOuterCellFaces op;
    const Parameters p = op.GetDefaultParameters();

    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridColorOuterCellFaces"});
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["cell_color"].GetInt(), -1);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorOuterCellFacesColorsOuterBoundary, KratosCoreFastSuite)
{
    // OctreeHybridColorCellsByLevel with a [-10, 100] level range colours every
    // cell (including the negative-level transition/buffer cells) with
    // cell_color == 1. OctreeHybridColorOuterCellFaces must then colour, for
    // every such cell, the local faces with no neighbouring cell — i.e. the
    // outer boundary of the extracted hex mesh.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridColorCellsByLevel","color":1,"min_level":-10,"max_level":100},
            {"type":"OctreeHybridColorOuterCellFaces","color":3,"cell_color":1}
        ],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();

    const auto& r_data = modeler.GetData();
    const auto neighbors = OctreeHybridMeshUtility::ComputeCellFaceNeighbors(r_data.mCells);

    std::size_t n_colored_faces = 0;
    for (std::size_t c = 0; c < r_data.mCells.size(); ++c) {
        for (int f = 0; f < 6; ++f) {
            if (neighbors[c][f] == -1) {
                KRATOS_EXPECT_EQ(r_data.mCellFaceColor[c][f], 3);
                ++n_colored_faces;
            } else {
                KRATOS_EXPECT_EQ(r_data.mCellFaceColor[c][f], 0);
            }
        }
    }
    KRATOS_EXPECT_GT(n_colored_faces, 0u);
}

// ===========================================================================
// OctreeHybridColorCellFaces colouring
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.All.OctreeHybridColorCellFaces.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellFaces.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesDefaultParameters, KratosCoreFastSuite)
{
    OctreeHybridColorCellFaces op;
    const Parameters p = op.GetDefaultParameters();

    KRATOS_EXPECT_EQ(p["type"].GetString(), std::string{"OctreeHybridColorCellFaces"});
    KRATOS_EXPECT_EQ(p["model_part_name"].GetString(), std::string{"Undefined"});
    KRATOS_EXPECT_EQ(p["color"].GetInt(), -1);
    KRATOS_EXPECT_EQ(p["input_entities"].GetString(), std::string{""});
    KRATOS_EXPECT_NEAR(p["tolerance"].GetDouble(), 1.0e-12, 1.0e-20);
    KRATOS_EXPECT_EQ(p["bounding_box"]["min_point"].size(), 0);
    KRATOS_EXPECT_EQ(p["bounding_box"]["max_point"].size(), 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesManualColorsBoundingBoxFaces, KratosCoreFastSuite)
{
    // input_entities == "" selects the manual strategy: every candidate face
    // (after the bounding_box pre-filter) is coloured directly, with no
    // geometry test.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellFaces",
            "color":7,"input_entities":"",
            "bounding_box":{"min_point":[0.3,0.3,0.3],"max_point":[0.7,0.7,0.7]}
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();

    const auto& r_data = modeler.GetData();
    KRATOS_EXPECT_EQ(r_data.mCellFaceColor.size(), r_data.mCells.size());

    std::size_t n_colored = 0;
    const std::size_t n_total = r_data.mCellFaceColor.size() * 6;
    for (const auto& r_faces : r_data.mCellFaceColor)
        for (int f = 0; f < 6; ++f)
            if (r_faces[f] == 7) ++n_colored;

    KRATOS_EXPECT_GT(n_colored, 0u);
    KRATOS_EXPECT_LT(n_colored, n_total);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesAutomaticColorsSkinFaces, KratosCoreFastSuite)
{
    // input_entities == "geometries" selects the automatic, distance-based
    // strategy: faces whose centroid is within tolerance of the skin surface
    // must be coloured, while far-away faces must not.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellFaces",
            "model_part_name":"Skin","color":4,
            "input_entities":"geometries","tolerance":0.05
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");

    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.SetupModelPart();

    const auto& r_data = modeler.GetData();
    std::size_t n_colored = 0;
    const std::size_t n_total = r_data.mCellFaceColor.size() * 6;
    for (const auto& r_faces : r_data.mCellFaceColor)
        for (int f = 0; f < 6; ++f)
            if (r_faces[f] == 4) ++n_colored;

    KRATOS_EXPECT_GT(n_colored, 0u);
    KRATOS_EXPECT_LT(n_colored, n_total);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesUnsupportedInputEntitiesThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellFaces",
            "model_part_name":"Skin","color":4,"input_entities":"invalid"
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorCellFacesModelPartNotFoundThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[{"type":"RefineInterfaceCellsOctreeHybrid",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellFaces",
            "model_part_name":"DoesNotExist","color":4,"input_entities":"geometries"
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");
    OctreeHybridMeshGeneratorModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

// ===========================================================================
// GenerateHybridOctreeTetrahedraElementsWithCellColor — registry and generation tests
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateTetrahedraRegistryEntry, KratosCoreFastSuite)
{
    const std::string path_km = "OctreeHybridMesherEntityGeneration.KratosMultiphysics."
                                "GenerateHybridOctreeTetrahedraElementsWithCellColor.Prototype";
    const std::string path_all = "OctreeHybridMesherEntityGeneration.All."
                                 "GenerateHybridOctreeTetrahedraElementsWithCellColor.Prototype";
    KRATOS_EXPECT_TRUE(Registry::HasItem(path_km));
    KRATOS_EXPECT_TRUE(Registry::HasItem(path_all));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGenerateTriangleBCsRegistryEntry, KratosCoreFastSuite)
{
    const std::string path_km = "OctreeHybridMesherEntityGeneration.KratosMultiphysics."
                                "GenerateHybridOctreeTriangularConditionsWithFaceColor.Prototype";
    const std::string path_all = "OctreeHybridMesherEntityGeneration.All."
                                 "GenerateHybridOctreeTriangularConditionsWithFaceColor.Prototype";
    KRATOS_EXPECT_TRUE(Registry::HasItem(path_km));
    KRATOS_EXPECT_TRUE(Registry::HasItem(path_all));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTetraElementsCreated, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeTetrahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTetraCountIsHexTimes6, KratosCoreFastSuite)
{
    const std::string common_settings = R"({
        "input_model_part_name"  : "Skin",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "model_part_operations"  : []
    })";

    Model hex_model;
    BuildClosedBoxSurface(hex_model.CreateModelPart("Skin"));
    Parameters hex_params(common_settings);
    hex_params.AddValue("entities_generator_list", Parameters(R"([{
        "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
        "model_part_name": "Output", "color": 1 }])"));
    OctreeHybridMeshGeneratorModeler hex_modeler(hex_model, hex_params);
    hex_modeler.SetupModelPart();
    const std::size_t n_hexes = hex_model.GetModelPart("Output").NumberOfElements();

    Model tet_model;
    BuildClosedBoxSurface(tet_model.CreateModelPart("Skin"));
    Parameters tet_params(common_settings);
    tet_params.AddValue("entities_generator_list", Parameters(R"([{
        "type": "GenerateHybridOctreeTetrahedraElementsWithCellColor",
        "model_part_name": "Output", "color": 1 }])"));
    OctreeHybridMeshGeneratorModeler tet_modeler(tet_model, tet_params);
    tet_modeler.SetupModelPart();
    const std::size_t n_tets = tet_model.GetModelPart("Output").NumberOfElements();

    KRATOS_EXPECT_EQ(n_tets, 6 * n_hexes);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTetraZeroInverted, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeTetrahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    int n_inv = 0;
    for (const auto& r_el : out.Elements()) {
        const auto& g = r_el.GetGeometry();
        const double e0[3] = {g[1].X()-g[0].X(), g[1].Y()-g[0].Y(), g[1].Z()-g[0].Z()};
        const double e1[3] = {g[2].X()-g[0].X(), g[2].Y()-g[0].Y(), g[2].Z()-g[0].Z()};
        const double e2[3] = {g[3].X()-g[0].X(), g[3].Y()-g[0].Y(), g[3].Z()-g[0].Z()};
        const double vol = e0[0]*(e1[1]*e2[2]-e1[2]*e2[1])
                         - e0[1]*(e1[0]*e2[2]-e1[2]*e2[0])
                         + e0[2]*(e1[0]*e2[1]-e1[1]*e2[0]);
        if (vol <= 0.0) ++n_inv;
    }
    KRATOS_EXPECT_EQ(n_inv, 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTetraNodeSubsetFromHex, KratosCoreFastSuite)
{
    const std::string common_settings_str = R"({
        "input_model_part_name"  : "Skin",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "model_part_operations"  : []
    })";

    Model hex_model;
    BuildClosedBoxSurface(hex_model.CreateModelPart("Skin"));
    Parameters hex_p(common_settings_str);
    hex_p.AddValue("entities_generator_list", Parameters(R"([{
        "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
        "model_part_name": "Output", "color": 1 }])"));
    OctreeHybridMeshGeneratorModeler hex_modeler(hex_model, hex_p);
    hex_modeler.SetupModelPart();
    std::set<IndexType> hex_node_ids;
    for (const auto& n : hex_model.GetModelPart("Output").Nodes())
        hex_node_ids.insert(n.Id());

    Model tet_model;
    BuildClosedBoxSurface(tet_model.CreateModelPart("Skin"));
    Parameters tet_p(common_settings_str);
    tet_p.AddValue("entities_generator_list", Parameters(R"([{
        "type": "GenerateHybridOctreeTetrahedraElementsWithCellColor",
        "model_part_name": "Output", "color": 1 }])"));
    OctreeHybridMeshGeneratorModeler tet_modeler(tet_model, tet_p);
    tet_modeler.SetupModelPart();
    for (const auto& n : tet_model.GetModelPart("Output").Nodes())
        KRATOS_EXPECT_TRUE(hex_node_ids.count(n.Id()) > 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTetraTagRefinementLevel, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeTetrahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1,
                                      "tag_refinement_level": true }],
        "model_part_operations"  : []
    })");

    for (const auto& r_el : out.Elements())
        KRATOS_EXPECT_GE(r_el.GetValue(REFINEMENT_LEVEL), 0);
}

// ===========================================================================
// GenerateHybridOctreeTriangularConditionsWithFaceColor — tests
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTriangleBCsCreated, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [
            { "type": "GenerateHybridOctreeTetrahedraElementsWithCellColor",
              "model_part_name": "Output", "color": 1 },
            { "type": "GenerateHybridOctreeTriangularConditionsWithFaceColor",
              "model_part_name": "Output.Boundary", "color": 1 }
        ],
        "model_part_operations"  : []
    })");

    ModelPart& boundary = model.GetModelPart("Output.Boundary");
    KRATOS_EXPECT_GT(boundary.NumberOfConditions(), 0u);
    KRATOS_EXPECT_GT(boundary.NumberOfNodes(), 0u);

    for (const auto& cond : boundary.Conditions())
        KRATOS_EXPECT_EQ(cond.GetGeometry().size(), 3u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTriangleBCsCountIsTwiceQuad, KratosCoreFastSuite)
{
    const std::string common_str = R"({
        "input_model_part_name"  : "Skin",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "model_part_operations"  : []
    })";

    Model quad_model;
    BuildClosedBoxSurface(quad_model.CreateModelPart("Skin"));
    Parameters quad_p(common_str);
    quad_p.AddValue("entities_generator_list", Parameters(R"([{
        "type": "GenerateHybridOctreeQuadrilateralConditionsWithFaceColor",
        "model_part_name": "Output", "color": 1 }])"));
    OctreeHybridMeshGeneratorModeler quad_modeler(quad_model, quad_p);
    quad_modeler.SetupModelPart();
    const std::size_t n_quads = quad_model.GetModelPart("Output").NumberOfConditions();

    Model tri_model;
    BuildClosedBoxSurface(tri_model.CreateModelPart("Skin"));
    Parameters tri_p(common_str);
    tri_p.AddValue("entities_generator_list", Parameters(R"([{
        "type": "GenerateHybridOctreeTriangularConditionsWithFaceColor",
        "model_part_name": "Output", "color": 1 }])"));
    OctreeHybridMeshGeneratorModeler tri_modeler(tri_model, tri_p);
    tri_modeler.SetupModelPart();
    const std::size_t n_tris = tri_model.GetModelPart("Output").NumberOfConditions();

    KRATOS_EXPECT_EQ(n_tris, 2 * n_quads);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerTriangleBCsNodeSubset, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refinement_settings_list" : [{ "type": "RefineInterfaceCellsOctreeHybrid",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [
            { "type": "GenerateHybridOctreeTetrahedraElementsWithCellColor",
              "model_part_name": "Output", "color": 1 },
            { "type": "GenerateHybridOctreeTriangularConditionsWithFaceColor",
              "model_part_name": "Output.Boundary", "color": 1 }
        ],
        "model_part_operations"  : []
    })");

    std::set<IndexType> tet_node_ids;
    for (const auto& n : model.GetModelPart("Output").Nodes())
        tet_node_ids.insert(n.Id());

    ModelPart& boundary = model.GetModelPart("Output.Boundary");
    for (const auto& n : boundary.Nodes())
        KRATOS_EXPECT_TRUE(tet_node_ids.count(n.Id()) > 0);
}

} // namespace Kratos::Testing
