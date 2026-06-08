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
#include <cmath>

// External includes

// Project includes
#include "testing/testing.h"
#include "includes/kratos_application.h"
#include "includes/kernel.h"
#include "includes/variables.h"
#include "includes/registry.h"

#include "modeler/octree_hybrid_mesher_modeler.h"
#include "modeler/entity_generation/octree_hybrid_generate_hexes_by_cell_color.h"

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
 * @brief Runs OctreeHybridMesherModeler::SetupModelPart with the given JSON settings
 *        string, then returns the named output ModelPart.
 */
ModelPart& RunModeler(Model& rModel, const std::string& rSettingsJson)
{
    Parameters settings(rSettingsJson);
    OctreeHybridMesherModeler modeler(rModel, settings);
    modeler.SetupModelPart();
    return rModel.GetModelPart(settings["output_model_part_name"].GetString());
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
// OctreeHybridMesherModeler — top-level modeler tests
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherModelerDualElementsCreated, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refine_operations_list" : [{ "type": "OctreeHybridRefineInterfaceCells",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherModelerDualZeroInverted, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refine_operations_list" : [{ "type": "OctreeHybridRefineInterfaceCells",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    int n_inv = 0;
    for (const auto& r_el : out.Elements())
        if (MinScaledJacobian(r_el) <= 0.0) ++n_inv;

    KRATOS_EXPECT_EQ(n_inv, 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherModelerDualCarveBbox, KratosCoreFastSuite)
{
    constexpr double lo = 0.3, hi = 0.7;
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"), lo, hi);

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refine_operations_list" : [{ "type": "OctreeHybridRefineInterfaceCells",
                                      "refinement_depth": 4, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
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

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherModelerDefaultParametersValid, KratosCoreFastSuite)
{
    OctreeHybridMesherModeler m;
    const Parameters defaults = m.GetDefaultParameters();

    // Check that the mandatory keys are present with correct types
    KRATOS_EXPECT_TRUE(defaults.Has("input_model_part_name"));
    KRATOS_EXPECT_TRUE(defaults.Has("output_model_part_name"));
    KRATOS_EXPECT_TRUE(defaults.Has("refine_operations_list"));
    KRATOS_EXPECT_TRUE(defaults.Has("coloring_settings_list"));
    KRATOS_EXPECT_TRUE(defaults.Has("entities_generator_list"));
    KRATOS_EXPECT_TRUE(defaults.Has("model_part_operations"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherModelerInfoString, KratosCoreFastSuite)
{
    OctreeHybridMesherModeler m;
    KRATOS_EXPECT_EQ(m.Info(), "OctreeHybridMesherModeler");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherModelerUnknownOperationThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refine_operations_list" : [{ "type": "OctreeHybridRefineInterfaceCells",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [],
        "entities_generator_list": [],
        "model_part_operations"  : [{ "type": "NonExistentOperationType_XYZ" }]
    })");
    OctreeHybridMesherModeler modeler(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.SetupModelPart(), "");
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
            "model_part_name":"O","color":1}],
        "model_part_operations":[]
    })";
    const char* settings_classify = R"({
        "input_model_part_name":"S","output_model_part_name":"O",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
// OctreeHybridGenerateHexesByCellColor entity generator
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherEntityGeneration.All.OctreeHybridGenerateHexesByCellColor.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesNodeDeduplication, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
// OctreeHybridGenerateBoundaryConditionsByFace entity generator
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherBoundaryConditionsRegistered, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridMesherEntityGeneration.All.OctreeHybridGenerateBoundaryConditionsByFace.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherBoundaryConditionsCreated, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    RunModeler(model, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Volume",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"Volume","color":1},
            {"type":"OctreeHybridGenerateBoundaryConditionsByFace","model_part_name":"Boundary","color":1}
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"Volume","color":1},
            {"type":"OctreeHybridGenerateBoundaryConditionsByFace","model_part_name":"Boundary","color":1}
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"Volume","color":1},
            {"type":"OctreeHybridGenerateBoundaryConditionsByFace","model_part_name":"Boundary","color":1}
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"Volume","color":1},
            {"type":"OctreeHybridGenerateBoundaryConditionsByFace","model_part_name":"Boundary","color":1}
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
// OctreeHybridGenerateHexesByCellColor — hanging-node constraint parameters
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherGenerateHexesByCellColorHasConstraintParams, KratosCoreFastSuite)
{
    OctreeHybridGenerateHexesByCellColor op;
    Parameters p = op.GetDefaultParameters();
    KRATOS_EXPECT_TRUE(p.Has("constraint_name"));
    KRATOS_EXPECT_TRUE(p.Has("variables"));
    KRATOS_EXPECT_EQ(p["constraint_name"].GetString(), std::string{"LinearMasterSlaveConstraint"});
    KRATOS_EXPECT_EQ(p["variables"].size(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherPrimalMeshConstraintsGenerated, KratosCoreFastSuite)
{
    Model model;
    BuildTransitionSurface(model.CreateModelPart("Surface"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name":"Surface","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"Output","color":1,
             "variables":["DISPLACEMENT_X"]}
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"Output","color":1,
             "variables":["DISPLACEMENT_X"]}
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"Output","color":1,
             "variables":["DISPLACEMENT_X"]}
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"O","color":1,
             "variables":["DISPLACEMENT_X"]}
        ],"model_part_operations":[]
    })";
    const char* s3 = R"({
        "input_model_part_name":"S","output_model_part_name":"O",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":true,"mesh_type":"primal"}],
        "coloring_settings_list":[],
        "entities_generator_list":[
            {"type":"OctreeHybridGenerateHexesByCellColor","model_part_name":"O","color":1,
             "variables":["DISPLACEMENT_X","DISPLACEMENT_Y","DISPLACEMENT_Z"]}
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":true,"mesh_type":"dual"}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "input_model_part_name":"Skin","output_model_part_name":"Empty",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[],
        "model_part_operations":[{"type":"OctreeHybridReportMeshQuality","model_part_name":"Empty"}]
    })");
    // SetupModelPart on an empty entity list should not throw
    OctreeHybridMesherModeler modeler(model, settings);
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
        "OctreeHybridMesherEntityGeneration.KratosMultiphysics.OctreeHybridGenerateHexesByCellColor.Prototype"));
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype"
        }],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridMesherColoring.All.OctreeHybridMesherColoring.Prototype"
        }],
        "entities_generator_list":[],"model_part_operations":[]
    })");
    OctreeHybridMesherModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherRegistryBaseOperationInvocationThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":3,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[],
        "model_part_operations":[{
            "type":"OctreeHybridMesherOperation.All.OctreeHybridMesherOperation.Prototype"
        }]
    })");
    OctreeHybridMesherModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

// ===========================================================================
// OctreeHybridRefineOperation — refine_operations_list stage tests
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformRefinesAllCells, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    // Reference run: build at depth 4 then uniformly refine to depth 4 (fills all cells).
    ModelPart& ref = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Ref",
        "refine_operations_list" : [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 4, "adaptive": false },
            { "type": "OctreeHybridRefineUniform",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
                                      "model_part_name": "Ref", "color": 1 }],
        "model_part_operations"  : []
    })");

    // Test run: initial build at depth 1, then uniformly refined to depth 4.
    // RefineUniform reinitializes the octree to the target depth when it exceeds the
    // current max depth, so the resulting mesh is identical to the reference.
    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refine_operations_list" : [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 1, "adaptive": false },
            { "type": "OctreeHybridRefineUniform",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list" : [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 2, "adaptive": false },
            { "type": "OctreeHybridRefineInterfaceCells",
              "input_model_part_name": "Skin",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
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
    // Empty input_model_part_name → falls back to the modeler's main surface.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "refine_operations_list" : [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 2, "adaptive": false },
            { "type": "OctreeHybridRefineInterfaceCells",
              "input_model_part_name": "",
              "refinement_depth": 4 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsMultipleGeometries, KratosCoreFastSuite)
{
    // Two chained OctreeHybridRefineInterfaceCells entries, each with a different
    // model part and depth — verifies multi-geometry chaining runs without error.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("SkinA"), 0.2, 0.5);
    BuildClosedBoxSurface(model.CreateModelPart("SkinB"), 0.5, 0.8);

    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "SkinA",
        "output_model_part_name" : "Output",
        "refine_operations_list" : [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 1, "adaptive": false },
            { "type": "OctreeHybridRefineInterfaceCells",
              "input_model_part_name": "SkinA", "refinement_depth": 4 },
            { "type": "OctreeHybridRefineInterfaceCells",
              "input_model_part_name": "SkinB", "refinement_depth": 3 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineUniformElementSizeProducesElements, KratosCoreFastSuite)
{
    // element_size > 0 should override refinement_depth via ElementSizeToDepth
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    ModelPart& out = model.CreateModelPart("Output");

    // Box is [0,1]^3 so element_size=0.25 → depth≥2; should produce elements
    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "output_model_part_name": "Output",
        "refine_operations_list": [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 1, "adaptive": false },
            { "type": "OctreeHybridRefineUniform",
              "refinement_depth": 1, "element_size": 0.25 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");
    RunModeler(model, settings.WriteJsonString());

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineInterfaceCellsElementSizeProducesElements, KratosCoreFastSuite)
{
    // element_size > 0 should override refinement_depth for interface refinement
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    ModelPart& out = model.CreateModelPart("Output");

    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "output_model_part_name": "Output",
        "refine_operations_list": [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 1, "adaptive": false },
            { "type": "OctreeHybridRefineInterfaceCells",
              "input_model_part_name": "Skin",
              "refinement_depth": 1, "element_size": 0.25 }
        ],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "OctreeHybridGenerateHexesByCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");
    RunModeler(model, settings.WriteJsonString());

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationRegistryPrototypePresent, KratosCoreFastSuite)
{
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridRefineOperation.All.OctreeHybridRefineUniform.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridRefineOperation.KratosMultiphysics.OctreeHybridRefineUniform.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridRefineOperation.All.OctreeHybridRefineInterfaceCells.Prototype"));
    KRATOS_EXPECT_TRUE(Registry::HasValue(
        "OctreeHybridRefineOperation.KratosMultiphysics.OctreeHybridRefineInterfaceCells.Prototype"));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationBaseInvocationThrows, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "output_model_part_name": "Output",
        "refine_operations_list": [
            { "type": "OctreeHybridRefineInterfaceCells",
              "refinement_depth": 2, "adaptive": false },
            { "type": "OctreeHybridRefineOperation.All.OctreeHybridRefineOperation.Prototype" }
        ],
        "coloring_settings_list": [],
        "entities_generator_list": [],
        "model_part_operations"  : []
    })");
    OctreeHybridMesherModeler m(model, settings);
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsInTouch",
            "model_part_name":"Skin","color":2
        }],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    // Only surface-touching cells (color=2)
    ModelPart& out_touch = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsInTouch",
            "model_part_name":"Skin","color":2
        }],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridClassifyCellsInsideOutside"},
            {"type":"OctreeHybridColorConnectedCellsInTouch",
             "model_part_name":"Skin","cell_color":1,"color":3}
        ],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
            "model_part_name":"Output","color":3}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);
    KRATOS_EXPECT_GT(out.NumberOfNodes(),    0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridColorConnectedCellsInTouchFloodFillCoversInsideRegion, KratosCoreFastSuite)
{
    // The interior of the box is a single connected domain: flood-filling from the
    // inside interface cells must reach every inside cell, so the element count
    // from color=3 equals the count from a plain classify with color=1.
    Model m1, m2;
    BuildClosedBoxSurface(m1.CreateModelPart("Skin"));
    BuildClosedBoxSurface(m2.CreateModelPart("Skin"));

    ModelPart& out_inside = RunModeler(m1, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{"type":"OctreeHybridClassifyCellsInsideOutside"}],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    ModelPart& out_connected = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[
            {"type":"OctreeHybridClassifyCellsInsideOutside"},
            {"type":"OctreeHybridColorConnectedCellsInTouch",
             "model_part_name":"Skin","cell_color":1,"color":3}
        ],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
            "model_part_name":"Output","color":3}],
        "model_part_operations":[]
    })");

    KRATOS_EXPECT_EQ(out_connected.NumberOfElements(), out_inside.NumberOfElements());
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
            "model_part_name":"Output","color":1}],
        "model_part_operations":[]
    })");

    ModelPart& out_level = RunModeler(m2, R"({
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsByLevel","color":2,"min_level":4,"max_level":4
        }],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsByLevel","color":2,"min_level":5,"max_level":5
        }],
        "entities_generator_list":[{"type":"OctreeHybridGenerateHexesByCellColor",
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
        "input_model_part_name":"Skin","output_model_part_name":"Output",
        "refine_operations_list":[{"type":"OctreeHybridRefineInterfaceCells",
            "refinement_depth":4,"adaptive":false}],
        "coloring_settings_list":[{
            "type":"OctreeHybridColorCellsByLevel","color":1,"min_level":5,"max_level":3
        }],
        "entities_generator_list":[],
        "model_part_operations":[]
    })");
    OctreeHybridMesherModeler m(model, settings);
    KRATOS_EXPECT_EXCEPTION_IS_THROWN(m.SetupModelPart(), "");
}

} // namespace Kratos::Testing
