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
#include "modeler/utilities/octree_hybrid_mesh_utility.h"
#include "modeler/refine_operations/refine_hybrid_octree.h"
#include "modeler/refine_operations/refine_uniform_hybrid_octree.h"
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"
#include "modeler/coloring/octree_hybrid_classify_cells_inside_outside.h"
#include "modeler/entity_generation/octree_hybrid_mesher_entity_generation.h"
#include "modeler/entity_generation/generate_hybrid_octree_hexahedra_elements_with_cell_color.h"
#include "modeler/operation/octree_hybrid_mesher_operation.h"
#include "modeler/operation/octree_hybrid_report_mesh_quality.h"
#include "modeler/octree_hybrid_mesh_generator_modeler.h"

namespace Kratos::Testing {

namespace {

using OctreeType = OctreeHybridMeshUtility::OctreeType;

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

std::string ReadFirstLine(const std::string& rFilename)
{
    std::ifstream f(rFilename);
    std::string line;
    std::getline(f, line);
    return line;
}

bool FileExists(const std::string& rFilename)
{
    std::ifstream f(rFilename);
    return f.is_open();
}

} // anonymous namespace

// ===========================================================================
// OctreeHybridMeshUtility::WriteHexVtk
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityWriteHexVtkProducesVtkHeader, KratosCoreFastSuite)
{
    std::vector<std::array<double,3>> nodes = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    std::vector<std::array<int,8>> cells = {{{0,1,2,3,4,5,6,7}}};
    std::vector<int> levels = {1};

    const std::string fname = "/tmp/kratos_test_write_hex_vtk.vtk";
    OctreeHybridMeshUtility::WriteHexVtk(fname, nodes, cells, levels);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    KRATOS_EXPECT_EQ(ReadFirstLine(fname), std::string{"# vtk DataFile Version 2.0"});
    std::remove(fname.c_str());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityWriteHexVtkEmptyCellsWritesEmptyMesh, KratosCoreFastSuite)
{
    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>>   cells;
    std::vector<int> levels;

    const std::string fname = "/tmp/kratos_test_write_hex_vtk_empty.vtk";
    OctreeHybridMeshUtility::WriteHexVtk(fname, nodes, cells, levels);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    std::remove(fname.c_str());
}

// ===========================================================================
// OctreeHybridMeshUtility::WriteDualHexVtk
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityWriteDualHexVtkProducesFile, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto p_octree = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
        model.GetModelPart("Skin"), 3, /*Adaptive=*/false);

    const std::string fname = "/tmp/kratos_test_write_dual_hex_vtk.vtk";
    OctreeHybridMeshUtility::WriteDualHexVtk(*p_octree, fname);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    KRATOS_EXPECT_EQ(ReadFirstLine(fname), std::string{"# vtk DataFile Version 2.0"});
    std::remove(fname.c_str());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityWriteDualHexVtkWithCarveProducesFile, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto p_octree = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
        model.GetModelPart("Skin"), 3, /*Adaptive=*/false);
    auto triangles = OctreeHybridMeshUtility::ExtractTriangleSoup(model.GetModelPart("Skin"));

    const std::string fname = "/tmp/kratos_test_write_dual_hex_vtk_carved.vtk";
    OctreeHybridMeshUtility::WriteDualHexVtk(*p_octree, fname, &triangles);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    std::remove(fname.c_str());
}

// ===========================================================================
// OctreeHybridMeshUtility::WritePrimalVtk
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityWritePrimalVtkProducesFile, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));
    auto p_octree = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
        model.GetModelPart("Skin"), 3, /*Adaptive=*/false);

    const std::string fname = "/tmp/kratos_test_write_primal_vtk.vtk";
    OctreeHybridMeshUtility::WritePrimalVtk(*p_octree, fname);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    KRATOS_EXPECT_EQ(ReadFirstLine(fname), std::string{"# vtk DataFile Version 2.0"});
    std::remove(fname.c_str());
}

// ===========================================================================
// OctreeHybridMeshUtility::BuildAndWriteVtk
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityBuildAndWriteVtkProducesFile, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    const std::string fname = "/tmp/kratos_test_build_and_write_vtk.vtk";
    OctreeHybridMeshUtility::BuildAndWriteVtk(model.GetModelPart("Skin"), fname, 3, /*Adaptive=*/false);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    KRATOS_EXPECT_EQ(ReadFirstLine(fname), std::string{"# vtk DataFile Version 2.0"});
    std::remove(fname.c_str());
}

// ===========================================================================
// OctreeHybridMeshUtility::BuildCarveAndWriteVtk
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityBuildCarveAndWriteVtkProducesFile, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    const std::string fname = "/tmp/kratos_test_build_carve_write_vtk.vtk";
    OctreeHybridMeshUtility::BuildCarveAndWriteVtk(model.GetModelPart("Skin"), fname, 3, /*Adaptive=*/false);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    KRATOS_EXPECT_EQ(ReadFirstLine(fname), std::string{"# vtk DataFile Version 2.0"});
    std::remove(fname.c_str());
}

// ===========================================================================
// OctreeHybridMeshUtility::BuildCarveProjectAndWriteVtk
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityBuildCarveProjectAndWriteVtkProducesFile, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    const std::string fname = "/tmp/kratos_test_build_carve_project_vtk.vtk";
    // Use minimal iterations to keep the test fast
    OctreeHybridMeshUtility::BuildCarveProjectAndWriteVtk(
        model.GetModelPart("Skin"), fname,
        /*RefinementDepth=*/3,
        /*ProjIters=*/1, /*ProjSmooth=*/1,
        /*Adaptive=*/false);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    KRATOS_EXPECT_EQ(ReadFirstLine(fname), std::string{"# vtk DataFile Version 2.0"});
    std::remove(fname.c_str());
}

// ===========================================================================
// OctreeHybridMeshUtility::WriteOctreeForReference
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityWriteOctreeForReferenceProducesFile, KratosCoreFastSuite)
{
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"));

    const std::string fname = "/tmp/kratos_test_write_octree_reference.vtk";
    OctreeHybridMeshUtility::WriteOctreeForReference(model.GetModelPart("Skin"), fname, 3);

    KRATOS_EXPECT_TRUE(FileExists(fname));
    KRATOS_EXPECT_EQ(ReadFirstLine(fname), std::string{"# vtk DataFile Version 2.0"});
    std::remove(fname.c_str());
}

// ===========================================================================
// OctreeHybridMeshUtility::ClearBufferZone
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClearBufferZoneEmptyMeshIsNoop, KratosCoreFastSuite)
{
    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>>   cells;
    std::vector<int> levels;

    OctreeHybridMeshUtility::ClearBufferZone(nodes, cells, levels);
    KRATOS_EXPECT_EQ(cells.size(), std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClearBufferZoneConvexHexNotRemoved, KratosCoreFastSuite)
{
    // A single unit cube: all boundary vertices have outward normals from 3
    // faces that are satisfiable by the corner-diagonal probe direction,
    // so no vertex is "folded" and no cell should be deleted.
    std::vector<std::array<double,3>> nodes = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    std::vector<std::array<int,8>> cells = {{{0,1,2,3,4,5,6,7}}};
    std::vector<int> levels = {1};

    OctreeHybridMeshUtility::ClearBufferZone(nodes, cells, levels);
    KRATOS_EXPECT_EQ(cells.size(), std::size_t{1});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClearBufferZonePreservesConvexBlockMesh, KratosCoreFastSuite)
{
    // A 2x1x1 block of two adjacent hexes — convex boundary, nothing removed.
    std::vector<std::array<double,3>> nodes = {
        {0,0,0},{1,0,0},{2,0,0},
        {0,1,0},{1,1,0},{2,1,0},
        {0,0,1},{1,0,1},{2,0,1},
        {0,1,1},{1,1,1},{2,1,1}
    };
    std::vector<std::array<int,8>> cells = {
        {{0,1,4,3,6,7,10,9}},
        {{1,2,5,4,7,8,11,10}}
    };
    std::vector<int> levels = {1, 1};

    OctreeHybridMeshUtility::ClearBufferZone(nodes, cells, levels);
    KRATOS_EXPECT_EQ(cells.size(), std::size_t{2});
}

// ===========================================================================
// OctreeHybridMeshUtility::ProjectToIsoSurface
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityProjectToIsoSurfaceEmptyTrianglesIsNoop, KratosCoreFastSuite)
{
    std::vector<std::array<double,3>> nodes = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    std::vector<std::array<int,8>> cells = {{{0,1,2,3,4,5,6,7}}};
    std::vector<int> levels = {1};

    OctreeHybridMeshUtility::TriangleSoup empty_soup;
    const auto nodes_before = nodes;

    OctreeHybridMeshUtility::ProjectToIsoSurface(empty_soup, nodes, cells, levels, /*TotalIters=*/1, /*SmoothEvery=*/1);

    // Nothing should change when triangle soup is empty
    KRATOS_EXPECT_EQ(nodes.size(), nodes_before.size());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityProjectToIsoSurfaceRunsWithoutError, KratosCoreFastSuite)
{
    // Build a carved dual-hex mesh from a box surface, then project it.
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    auto p_octree = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
        model.GetModelPart("Skin"), 3, /*Adaptive=*/false);
    auto triangles = OctreeHybridMeshUtility::ExtractTriangleSoup(model.GetModelPart("Skin"));

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    OctreeHybridMeshUtility::ExtractDualHexMesh(*p_octree, nodes, cells, levels);
    OctreeHybridMeshUtility::RemoveOutsideElement(triangles, nodes, cells, levels);

    KRATOS_EXPECT_GT(cells.size(), std::size_t{0});

    // Minimal iterations so the test is fast; verify it does not throw.
    OctreeHybridMeshUtility::ClearBufferZone(nodes, cells, levels);
    OctreeHybridMeshUtility::ProjectToIsoSurface(triangles, nodes, cells, levels, /*TotalIters=*/2, /*SmoothEvery=*/1);

    // After projection nodes should still be finite (non-NaN, non-Inf)
    for (const auto& nd : nodes)
        for (int d = 0; d < 3; ++d)
            KRATOS_EXPECT_TRUE(std::isfinite(nd[d]));
}

// ===========================================================================
// operator<< — base class stream operators
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineOperationStreamOperatorNonEmpty, KratosCoreFastSuite)
{
    RefineUniformOctreeHybrid op;
    std::ostringstream ss;
    ss << static_cast<const RefineOctreeHybrid&>(op);
    KRATOS_EXPECT_FALSE(ss.str().empty());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherColoringStreamOperatorNonEmpty, KratosCoreFastSuite)
{
    OctreeHybridClassifyCellsInsideOutside op;
    std::ostringstream ss;
    ss << static_cast<const OctreeHybridMesherColoring&>(op);
    KRATOS_EXPECT_FALSE(ss.str().empty());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherEntityGenerationStreamOperatorNonEmpty, KratosCoreFastSuite)
{
    GenerateHybridOctreeHexahedraElementsWithCellColor op;
    std::ostringstream ss;
    ss << static_cast<const OctreeHybridMesherEntityGeneration&>(op);
    KRATOS_EXPECT_FALSE(ss.str().empty());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMesherOperationStreamOperatorNonEmpty, KratosCoreFastSuite)
{
    OctreeHybridReportMeshQuality op;
    std::ostringstream ss;
    ss << static_cast<const OctreeHybridMesherOperation&>(op);
    KRATOS_EXPECT_FALSE(ss.str().empty());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerStreamOperatorNonEmpty, KratosCoreFastSuite)
{
    Model model;
    model.CreateModelPart("Skin");
    model.CreateModelPart("Out");
    Parameters params(R"({
        "input_model_part_name":"Skin",
        "refinement_settings_list":[],"coloring_settings_list":[],
        "entities_generator_list":[],"model_part_operations":[]
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, params);
    std::ostringstream ss;
    ss << modeler;
    KRATOS_EXPECT_FALSE(ss.str().empty());
}

} // namespace Kratos::Testing
