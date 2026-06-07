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
#include <numeric>
#include <set>

// External includes

// Project includes
#include "testing/testing.h"
#include "containers/model.h"
#include "includes/variables.h"
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

namespace Kratos::Testing {

namespace {

using Util = OctreeHybridMeshUtility;
using OctreeType = Util::OctreeType;

void BuildBoxSurface(ModelPart& rSurfaceMesh, double Lo = 0.3, double Hi = 0.7)
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

// Build the 8 corners of a unit cube [0,1]^3 in VTK HEXAHEDRON node order.
// Corner 0=(0,0,0), 1=(1,0,0), 2=(1,1,0), 3=(0,1,0),
// Corner 4=(0,0,1), 5=(1,0,1), 6=(1,1,1), 7=(0,1,1).
void MakeUnitHex(double p[8][3])
{
    p[0][0]=0; p[0][1]=0; p[0][2]=0;
    p[1][0]=1; p[1][1]=0; p[1][2]=0;
    p[2][0]=1; p[2][1]=1; p[2][2]=0;
    p[3][0]=0; p[3][1]=1; p[3][2]=0;
    p[4][0]=0; p[4][1]=0; p[4][2]=1;
    p[5][0]=1; p[5][1]=0; p[5][2]=1;
    p[6][0]=1; p[6][1]=1; p[6][2]=1;
    p[7][0]=0; p[7][1]=1; p[7][2]=1;
}

// Inverted hex: swap first 4 node positions (reverses bottom-face winding).
void MakeInvertedHex(double p[8][3])
{
    MakeUnitHex(p);
    std::swap(p[0][0], p[3][0]); std::swap(p[0][1], p[3][1]); std::swap(p[0][2], p[3][2]);
    std::swap(p[1][0], p[2][0]); std::swap(p[1][1], p[2][1]); std::swap(p[1][2], p[2][2]);
    std::swap(p[4][0], p[7][0]); std::swap(p[4][1], p[7][1]); std::swap(p[4][2], p[7][2]);
    std::swap(p[5][0], p[6][0]); std::swap(p[5][1], p[6][1]); std::swap(p[5][2], p[6][2]);
}

} // anonymous namespace

// ===========================================================================
// Section A — Geometry helpers
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilitySqDistKnownValue, KratosCoreFastSuite)
{
    const double a[3] = {0, 0, 0};
    const double b[3] = {3, 4, 0};
    KRATOS_EXPECT_DOUBLE_EQ(Util::SqDist(a, b), 25.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilitySqDistSamePoint, KratosCoreFastSuite)
{
    const double p[3] = {1.5, -2.3, 7.0};
    KRATOS_EXPECT_DOUBLE_EQ(Util::SqDist(p, p), 0.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityTriAreaEquilateral, KratosCoreFastSuite)
{
    // Equilateral triangle with side length 2: area = sqrt(3) ≈ 1.73205
    KRATOS_EXPECT_NEAR(Util::TriArea(2.0, 2.0, 2.0), std::sqrt(3.0), 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityTriAreaDegenerate, KratosCoreFastSuite)
{
    // Collinear: a+b=c → Heron gives 0 (clamped ≥ 0)
    KRATOS_EXPECT_NEAR(Util::TriArea(1.0, 1.0, 2.0), 0.0, 1e-12);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityTriRayIntersectHitsCenter, KratosCoreFastSuite)
{
    // Triangle in XY plane: a=(0,0,0), b=(2,0,0), c=(0,2,0)
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    // Ray from (0.5,0.5,1) straight down
    const double p[3] = {0.5, 0.5, 1.0};
    const double dir[3] = {0, 0, -1};
    double e[3]; double alpha;
    const int res = Util::TriRayIntersect(a, b, c, p, dir, e, alpha);
    KRATOS_EXPECT_EQ(res, 1);
    KRATOS_EXPECT_NEAR(e[0], 0.5, 1e-10);
    KRATOS_EXPECT_NEAR(e[1], 0.5, 1e-10);
    KRATOS_EXPECT_NEAR(e[2], 0.0, 1e-10);
    KRATOS_EXPECT_NEAR(alpha, 1.0, 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityTriRayIntersectMisses, KratosCoreFastSuite)
{
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    const double p[3] = {5.0, 5.0, 1.0}; // far from triangle
    const double dir[3] = {0, 0, -1};
    double e[3]; double alpha;
    KRATOS_EXPECT_EQ(Util::TriRayIntersect(a, b, c, p, dir, e, alpha), 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityTriRayIntersectParallel, KratosCoreFastSuite)
{
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    const double p[3] = {0.5, 0.5, 1.0};
    const double dir[3] = {1, 0, 0}; // in-plane
    double e[3]; double alpha;
    KRATOS_EXPECT_EQ(Util::TriRayIntersect(a, b, c, p, dir, e, alpha), -1);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityPointToTriInside, KratosCoreFastSuite)
{
    // Triangle a=(0,0,0), b=(2,0,0), c=(0,2,0); p above centroid at height 1.5
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    const double p[3] = {0.5, 0.5, 1.5};
    const double dist = Util::PointToTri(a, b, c, p, 100.0);
    KRATOS_EXPECT_NEAR(dist, 1.5, 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityPointToTriOutside, KratosCoreFastSuite)
{
    // Point beyond vertex b=(2,0,0): closest is b itself, dist = 1.0
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    const double p[3] = {3, 0, 0};
    const double dist = Util::PointToTri(a, b, c, p, 100.0);
    KRATOS_EXPECT_NEAR(dist, 1.0, 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityTripleProductUnitBasis, KratosCoreFastSuite)
{
    const double e0[3] = {1,0,0}, e1[3] = {0,1,0}, e2[3] = {0,0,1};
    KRATOS_EXPECT_DOUBLE_EQ(Util::TripleProduct(e0, e1, e2), 1.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityTripleProductDegenerate, KratosCoreFastSuite)
{
    const double e0[3] = {1,0,0}, e1[3] = {1,0,0}, e2[3] = {0,0,1};
    KRATOS_EXPECT_DOUBLE_EQ(Util::TripleProduct(e0, e1, e2), 0.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityHexEdgeTripleCorner0, KratosCoreFastSuite)
{
    // Unit cube, corner=0: SJ_ADJ[0]={1,3,4} → e0=(1,0,0), e1=(0,1,0), e2=(0,0,1)
    double p[8][3];
    MakeUnitHex(p);
    double e0[3], e1[3], e2[3];
    Util::HexEdgeTriple(p, 0, e0, e1, e2);
    KRATOS_EXPECT_DOUBLE_EQ(e0[0], 1.0); KRATOS_EXPECT_DOUBLE_EQ(e0[1], 0.0); KRATOS_EXPECT_DOUBLE_EQ(e0[2], 0.0);
    KRATOS_EXPECT_DOUBLE_EQ(e1[0], 0.0); KRATOS_EXPECT_DOUBLE_EQ(e1[1], 1.0); KRATOS_EXPECT_DOUBLE_EQ(e1[2], 0.0);
    KRATOS_EXPECT_DOUBLE_EQ(e2[0], 0.0); KRATOS_EXPECT_DOUBLE_EQ(e2[1], 0.0); KRATOS_EXPECT_DOUBLE_EQ(e2[2], 1.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityHexEdgeTripleBodyCenter, KratosCoreFastSuite)
{
    // Unit cube, corner=8 (body center): triple product must be > 0 (valid orientation)
    double p[8][3];
    MakeUnitHex(p);
    double e0[3], e1[3], e2[3];
    Util::HexEdgeTriple(p, 8, e0, e1, e2);
    KRATOS_EXPECT_GT(Util::TripleProduct(e0, e1, e2), 0.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClosestPointOnTriangleInterior, KratosCoreFastSuite)
{
    // p above interior: foot = projection onto plane
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    const double p[3] = {0.5, 0.5, 1.0};
    double q[3];
    Util::ClosestPointOnTriangle(a, b, c, p, q);
    KRATOS_EXPECT_NEAR(q[0], 0.5, 1e-10);
    KRATOS_EXPECT_NEAR(q[1], 0.5, 1e-10);
    KRATOS_EXPECT_NEAR(q[2], 0.0, 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClosestPointOnTriangleEdge, KratosCoreFastSuite)
{
    // p = (1, -1, 0), nearest point on edge ab is (1,0,0) (foot of perpendicular, clamped in)
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    const double p[3] = {1.0, -1.0, 0.0};
    double q[3];
    Util::ClosestPointOnTriangle(a, b, c, p, q);
    KRATOS_EXPECT_NEAR(q[0], 1.0, 1e-10);
    KRATOS_EXPECT_NEAR(q[1], 0.0, 1e-10);
    KRATOS_EXPECT_NEAR(q[2], 0.0, 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClosestPointOnTriangleVertex, KratosCoreFastSuite)
{
    // p far beyond vertex b: closest point is b
    const double a[3] = {0,0,0}, b[3] = {2,0,0}, c[3] = {0,2,0};
    const double p[3] = {5.0, -3.0, 0.0};
    double q[3];
    Util::ClosestPointOnTriangle(a, b, c, p, q);
    KRATOS_EXPECT_NEAR(q[0], 2.0, 1e-10);
    KRATOS_EXPECT_NEAR(q[1], 0.0, 1e-10);
    KRATOS_EXPECT_NEAR(q[2], 0.0, 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClosestPointOnSoupTwoTriangles, KratosCoreFastSuite)
{
    // Two triangles in XY plane, well-separated
    Util::TriangleSoup soup(2);
    soup[0] = {{{0,0,0}}, {{2,0,0}}, {{0,2,0}}};  // centered near (0.67, 0.67, 0)
    soup[1] = {{{5,0,0}}, {{7,0,0}}, {{5,2,0}}};  // centered near (5.67, 0.67, 0)

    // Query near soup[0]
    {
        const double p[3] = {0.5, 0.5, 0.1};
        double q[3]; int tri = -1;
        const double sq = Util::ClosestPointOnSoup(soup, p, q, tri);
        KRATOS_EXPECT_EQ(tri, 0);
        KRATOS_EXPECT_NEAR(sq, 0.01, 1e-10); // height^2 = 0.1^2
        KRATOS_EXPECT_NEAR(q[0], 0.5, 1e-10);
        KRATOS_EXPECT_NEAR(q[1], 0.5, 1e-10);
        KRATOS_EXPECT_NEAR(q[2], 0.0, 1e-10);
    }

    // Query near soup[1]
    {
        const double p[3] = {5.5, 0.5, 0.1};
        double q[3]; int tri = -1;
        const double sq = Util::ClosestPointOnSoup(soup, p, q, tri);
        KRATOS_EXPECT_EQ(tri, 1);
        KRATOS_EXPECT_NEAR(sq, 0.01, 1e-10);
        KRATOS_EXPECT_NEAR(q[0], 5.5, 1e-10);
        KRATOS_EXPECT_NEAR(q[1], 0.5, 1e-10);
        KRATOS_EXPECT_NEAR(q[2], 0.0, 1e-10);
    }
}

// ===========================================================================
// Section B — Jacobian quality
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityScaledJacobianMinUnitCube, KratosCoreFastSuite)
{
    double p[8][3];
    MakeUnitHex(p);
    KRATOS_EXPECT_NEAR(Util::ScaledJacobianMin(p), 1.0, 1e-10);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityScaledJacobianMinInverted, KratosCoreFastSuite)
{
    double p[8][3];
    MakeInvertedHex(p);
    KRATOS_EXPECT_LT(Util::ScaledJacobianMin(p), 0.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityJacobianMinUnitCube, KratosCoreFastSuite)
{
    double p[8][3];
    MakeUnitHex(p);
    KRATOS_EXPECT_GT(Util::JacobianMin(p), 0.0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityJacobianMinInverted, KratosCoreFastSuite)
{
    double p[8][3];
    MakeInvertedHex(p);
    KRATOS_EXPECT_LT(Util::JacobianMin(p), 0.0);
}

// ===========================================================================
// Section C — Signed distance and carving
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityComputeNodeSignedDistanceInsideIsPositive, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    const Util::TriangleSoup soup = Util::ExtractTriangleSoup(model.GetModelPart("Skin"));

    // Node at center of box
    const std::vector<std::array<double,3>> nodes = {{{0.5, 0.5, 0.5}}};
    const auto dist = Util::ComputeNodeSignedDistance(soup, nodes);
    KRATOS_EXPECT_GT(dist[0], 0.0); // inside → positive
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityComputeNodeSignedDistanceOutsideIsNegative, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    const Util::TriangleSoup soup = Util::ExtractTriangleSoup(model.GetModelPart("Skin"));

    // Node clearly outside box
    const std::vector<std::array<double,3>> nodes = {{{0.0, 0.0, 0.0}}};
    const auto dist = Util::ComputeNodeSignedDistance(soup, nodes);
    KRATOS_EXPECT_LT(dist[0], 0.0); // outside → negative
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityKeepCarvedCellAllInside, KratosCoreFastSuite)
{
    const std::vector<double> signed_dist = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const std::array<int,8> cell = {0, 1, 2, 3, 4, 5, 6, 7};
    KRATOS_EXPECT_TRUE(Util::KeepCarvedCell(cell, signed_dist));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityKeepCarvedCellAllOutside, KratosCoreFastSuite)
{
    const std::vector<double> signed_dist = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
    const std::array<int,8> cell = {0, 1, 2, 3, 4, 5, 6, 7};
    KRATOS_EXPECT_FALSE(Util::KeepCarvedCell(cell, signed_dist));
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityKeepCarvedCellMixed, KratosCoreFastSuite)
{
    // 1 corner slightly outside, 7 clearly inside → keep (n_out=1<3, ratio ok)
    const std::vector<double> signed_dist = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.05};
    const std::array<int,8> cell = {0, 1, 2, 3, 4, 5, 6, 7};
    KRATOS_EXPECT_TRUE(Util::KeepCarvedCell(cell, signed_dist));
}

// ===========================================================================
// Section D — Classification
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityClassifyInsideOutsideProducesBothColors, KratosCoreFastSuite)
{
    // Build octree for box [0.3,0.7]^3 then extract and classify
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    const Util::TriangleSoup soup = Util::ExtractTriangleSoup(model.GetModelPart("Skin"));

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    Util::ExtractDualHexMesh(*octree, nodes, cells, levels);

    std::vector<int> colors;
    Util::ClassifyInsideOutside(soup, nodes, cells, colors);

    KRATOS_EXPECT_EQ(colors.size(), cells.size());
    const bool has_inside  = std::any_of(colors.begin(), colors.end(), [](int c){ return c == 1; });
    const bool has_outside = std::any_of(colors.begin(), colors.end(), [](int c){ return c == 0; });
    KRATOS_EXPECT_TRUE(has_inside);
    KRATOS_EXPECT_TRUE(has_outside);
}

// ===========================================================================
// Section E — Boundary face extraction
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractBoundaryFacesSingleHex, KratosCoreFastSuite)
{
    // A single hex with indices 0..7
    const std::vector<std::array<int,8>> cells = {{{0, 1, 2, 3, 4, 5, 6, 7}}};
    const auto faces = Util::ExtractBoundaryFaces(cells);
    KRATOS_EXPECT_EQ(faces.size(), 6u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractBoundaryFacesTwoAdjacentHexes, KratosCoreFastSuite)
{
    // Two hexes sharing nodes {1,2,5,6}: hex1=[0..7], hex2=[1,8,9,2,5,10,11,6]
    // Total 12 face-entries, 1 shared face (owned by both → 2 entries cancel) → 10 boundary faces
    const std::vector<std::array<int,8>> cells = {
        {{0, 1, 2, 3, 4,  5,  6,  7}},
        {{1, 8, 9, 2, 5, 10, 11,  6}}
    };
    const auto faces = Util::ExtractBoundaryFaces(cells);
    KRATOS_EXPECT_EQ(faces.size(), 10u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractBoundaryFacesOwnerIndexInBounds, KratosCoreFastSuite)
{
    const std::vector<std::array<int,8>> cells = {
        {{0, 1, 2, 3, 4, 5, 6, 7}},
        {{1, 8, 9, 2, 5, 10, 11, 6}}
    };
    const auto faces = Util::ExtractBoundaryFaces(cells);
    const int n_cells = static_cast<int>(cells.size());
    for (const auto& f : faces) {
        KRATOS_EXPECT_GE(f[4], 0);
        KRATOS_EXPECT_LT(f[4], n_cells);
    }
}

// ===========================================================================
// Section F — Mesh extraction
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractDualHexMeshNonEmpty, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    Util::ExtractDualHexMesh(*octree, nodes, cells, levels);

    KRATOS_EXPECT_GT(nodes.size(), 0u);
    KRATOS_EXPECT_GT(cells.size(), 0u);
    KRATOS_EXPECT_EQ(cells.size(), levels.size());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractDualHexMeshConnectivityInBounds, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    Util::ExtractDualHexMesh(*octree, nodes, cells, levels);

    const int n = static_cast<int>(nodes.size());
    for (const auto& c : cells)
        for (int ni : c) {
            KRATOS_EXPECT_GE(ni, 0);
            KRATOS_EXPECT_LT(ni, n);
        }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractDualHexMeshZeroInverted, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    Util::ExtractDualHexMesh(*octree, nodes, cells, levels);

    int n_inv = 0;
    for (const auto& c : cells) {
        double p[8][3];
        for (int k = 0; k < 8; ++k)
            for (int d = 0; d < 3; ++d)
                p[k][d] = nodes[c[k]][d];
        if (Util::ScaledJacobianMin(p) <= 0.0) ++n_inv;
    }
    KRATOS_EXPECT_EQ(n_inv, 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractDualHexMeshLevelTagsAssigned, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    Util::ExtractDualHexMesh(*octree, nodes, cells, levels);

    for (int lv : levels) {
        // Plain-dual hexes: level > 0; template hexes: level == -1
        KRATOS_EXPECT_TRUE(lv == -1 || lv > 0);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractPrimalHexMeshNonEmpty, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    std::vector<Util::HangingConstraint> hanging;
    Util::ExtractPrimalHexMesh(*octree, nodes, cells, levels, hanging);

    KRATOS_EXPECT_GT(cells.size(), 0u);
    KRATOS_EXPECT_GT(nodes.size(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractPrimalHexMeshHangingConstraints, KratosCoreFastSuite)
{
    // Build a 2:1 transition by subdividing one L1 cell then balancing
    OctreeType octree(4);
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);
    octree.SubdivideCellByIdAndLevel(0, 0); // root → 8 L1 cells
    octree.SubdivideCellByIdAndLevel(1, 1); // (0,0,0)@L1 → 8 L2 cells
    octree.StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    std::vector<Util::HangingConstraint> hanging;
    Util::ExtractPrimalHexMesh(octree, nodes, cells, levels, hanging);

    KRATOS_EXPECT_GT(cells.size(), 0u);
    KRATOS_EXPECT_GT(hanging.size(), 0u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractPrimalHexMeshWeightsSumToOne, KratosCoreFastSuite)
{
    OctreeType octree(4);
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);
    octree.SubdivideCellByIdAndLevel(0, 0);
    octree.SubdivideCellByIdAndLevel(1, 1);
    octree.StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    std::vector<Util::HangingConstraint> hanging;
    Util::ExtractPrimalHexMesh(octree, nodes, cells, levels, hanging);

    for (const auto& hc : hanging) {
        double wsum = 0.0;
        for (int m = 0; m < hc.NumMasters; ++m)
            wsum += hc.Weights[m];
        KRATOS_EXPECT_NEAR(wsum, 1.0, 1e-10);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractTriangleSoupFromModelPart, KratosCoreFastSuite)
{
    // Build a ModelPart with 2 triangle geometries and verify soup has 2 entries
    Model model;
    auto& mp = model.CreateModelPart("Tris");
    mp.CreateNewNode(1, 0.0, 0.0, 0.0);
    mp.CreateNewNode(2, 1.0, 0.0, 0.0);
    mp.CreateNewNode(3, 0.0, 1.0, 0.0);
    mp.CreateNewNode(4, 1.0, 1.0, 0.0);
    mp.CreateNewGeometry("Triangle3D3", 1, std::vector<IndexType>{1, 2, 3});
    mp.CreateNewGeometry("Triangle3D3", 2, std::vector<IndexType>{2, 4, 3});

    const auto soup = Util::ExtractTriangleSoup(mp);
    KRATOS_EXPECT_EQ(soup.size(), 2u);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityExtractTriangleSoupPreservesCoordinates, KratosCoreFastSuite)
{
    Model model;
    auto& mp = model.CreateModelPart("Tris");
    mp.CreateNewNode(1, 1.0, 2.0, 3.0);
    mp.CreateNewNode(2, 4.0, 5.0, 6.0);
    mp.CreateNewNode(3, 7.0, 8.0, 9.0);
    mp.CreateNewGeometry("Triangle3D3", 1, std::vector<IndexType>{1, 2, 3});

    const auto soup = Util::ExtractTriangleSoup(mp);
    KRATOS_EXPECT_EQ(soup.size(), 1u);
    // Vertex 0 of the triangle matches node 1's coordinates
    KRATOS_EXPECT_NEAR(soup[0][0][0], 1.0, 1e-12);
    KRATOS_EXPECT_NEAR(soup[0][0][1], 2.0, 1e-12);
    KRATOS_EXPECT_NEAR(soup[0][0][2], 3.0, 1e-12);
}

// ===========================================================================
// Section G — Octree building and direct refinement
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityBuildFromSurfaceMeshNonNull, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    KRATOS_EXPECT_NE(octree, nullptr);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityBuildFromSurfaceMeshLeafCountGrowsWithDepth, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    auto& skin = model.GetModelPart("Skin");

    auto octree2 = Util::BuildFromSurfaceMesh(skin, 2, false);
    auto octree3 = Util::BuildFromSurfaceMesh(skin, 3, false);

    KRATOS_EXPECT_GT(octree3->GetLeafCount(), octree2->GetLeafCount());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityRefineAllCellsReachesTargetDepth, KratosCoreFastSuite)
{
    OctreeType octree(4);
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);

    Util::RefineAllCells(octree, 3);

    std::vector<OctreeType::cell_type*> leaves;
    octree.GetAllLeavesVector(leaves);
    for (const auto* c : leaves)
        KRATOS_EXPECT_GE(c->GetLevel(), 3);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityRefineAllCellsNopWhenAlreadyDeep, KratosCoreFastSuite)
{
    OctreeType octree(4);
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);

    Util::RefineAllCells(octree, 3); // bring to depth 3
    const int count_before = octree.GetLeafCount();

    Util::RefineAllCells(octree, 1); // target shallower → no-op
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), count_before);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityRefineInterfaceCellsOnlyNearSurface, KratosCoreFastSuite)
{
    OctreeType octree(4);
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);

    // Single triangle near the center
    Util::TriangleSoup soup(1);
    soup[0] = {{{0.3, 0.3, 0.5}}, {{0.7, 0.3, 0.5}}, {{0.5, 0.7, 0.5}}};

    Util::RefineInterfaceCells(octree, soup, 3);

    // Should have more leaves than just root (1), but fewer than full depth-3 refinement
    OctreeType full_octree(4);
    full_octree.SetBoundingBox(lo, hi);
    Util::RefineAllCells(full_octree, 3);

    KRATOS_EXPECT_GT(octree.GetLeafCount(), 1);
    KRATOS_EXPECT_LT(octree.GetLeafCount(), full_octree.GetLeafCount());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityElementSizeToDepthSmallSizeGivesLargeDepth, KratosCoreFastSuite)
{
    OctreeType octree(4); // MAX_DEPTH = 10 for KratosConfig but here use 4
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);

    // 0.01 is much smaller than 1/16 = 0.0625 at depth 4 → clamp to max depth
    const std::size_t d = Util::ElementSizeToDepth(octree, 0.01);
    KRATOS_EXPECT_EQ(d, octree.GetDepth());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityElementSizeToDepthLargeSizeGivesDepthZero, KratosCoreFastSuite)
{
    OctreeType octree(4);
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);

    // ElementSize exactly matching box size (1.0): norm_size=1.0 → -log2(1)=0 → depth=0
    const std::size_t d = Util::ElementSizeToDepth(octree, 1.0);
    KRATOS_EXPECT_EQ(d, std::size_t{0});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityElementSizeToDepthHalfBoxGivesDepthOne, KratosCoreFastSuite)
{
    OctreeType octree(4);
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    octree.SetBoundingBox(lo, hi);

    // 0.5 = exactly half the unit box → depth 1
    const std::size_t d = Util::ElementSizeToDepth(octree, 0.5);
    KRATOS_EXPECT_EQ(d, std::size_t{1});
}

// ===========================================================================
// Section H — RemoveOutsideElement
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityRemoveOutsideElementCarvesOutsideHexes, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    const Util::TriangleSoup soup = Util::ExtractTriangleSoup(model.GetModelPart("Skin"));

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    Util::ExtractDualHexMesh(*octree, nodes, cells, levels);

    const std::size_t n_before = cells.size();
    Util::RemoveOutsideElement(soup, nodes, cells, levels);
    const std::size_t n_after = cells.size();

    // Carving must remove some cells (those outside the box)
    KRATOS_EXPECT_LT(n_after, n_before);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityRemoveOutsideElementKeepsInsideHexes, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);
    const Util::TriangleSoup soup = Util::ExtractTriangleSoup(model.GetModelPart("Skin"));

    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false);
    octree->StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> levels;
    Util::ExtractDualHexMesh(*octree, nodes, cells, levels);

    Util::RemoveOutsideElement(soup, nodes, cells, levels);

    // At least some cells survive (those inside the box)
    KRATOS_EXPECT_GT(cells.size(), 0u);
}

} // namespace Kratos::Testing
