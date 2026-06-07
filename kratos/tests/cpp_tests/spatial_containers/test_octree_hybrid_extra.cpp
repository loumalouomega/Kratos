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
#include <vector>

// External includes

// Project includes
#include "testing/testing.h"
#include "spatial_containers/octree_hybrid.h"

namespace Kratos::Testing {

namespace {

struct OHExtraConfig {
    struct data_type { int value = 0; };
    using pointer_type = double*;

    static constexpr std::size_t MAX_DEPTH = 4;
    static constexpr std::size_t MIN_DEPTH = 1;
    static constexpr std::size_t DIMENSION = 3;

    static void DeleteData(data_type* p) { delete p; }

    static bool IsIntersected(pointer_type, double, const double*, const double*)
    { return true; }
};

using OHExtraCell   = OctreeHybridCell<OHExtraConfig>;
using OHExtraOctree = OctreeHybrid<OHExtraCell>;

} // anonymous namespace

// ===========================================================================
// Bounding-box constructor and GetDepth
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridBoundingBoxConstructor, KratosCoreFastSuite)
{
    double lo[3] = {-1.0, 0.0, 2.0};
    double hi[3] = { 3.0, 4.0, 8.0};
    OHExtraOctree octree(lo, hi, 4);

    double norm_lo[3], norm_hi[3];
    octree.NormalizeCoordinates(lo, norm_lo);
    octree.NormalizeCoordinates(hi, norm_hi);

    KRATOS_EXPECT_NEAR(norm_lo[0], 0.0, 1e-12);
    KRATOS_EXPECT_NEAR(norm_lo[1], 0.0, 1e-12);
    KRATOS_EXPECT_NEAR(norm_lo[2], 0.0, 1e-12);

    KRATOS_EXPECT_NEAR(norm_hi[0], 1.0, 1e-12);
    KRATOS_EXPECT_NEAR(norm_hi[1], 1.0, 1e-12);
    KRATOS_EXPECT_NEAR(norm_hi[2], 1.0, 1e-12);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGetDepthMatchesConstructorArg, KratosCoreFastSuite)
{
    KRATOS_EXPECT_EQ(OHExtraOctree(3).GetDepth(), std::size_t{3});
    KRATOS_EXPECT_EQ(OHExtraOctree(4).GetDepth(), std::size_t{4});
}

// ===========================================================================
// World-space Insert
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridInsertWorldSpaceRefinesToMinDepth, KratosCoreFastSuite)
{
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    OHExtraOctree octree(lo, hi, 4);
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 1);

    double pt[3] = {0.5, 0.5, 0.5};
    octree.Insert(pt); // MIN_DEPTH=1 → root is subdivided once

    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 8);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridInsertWorldSpaceLeafContainsPoint, KratosCoreFastSuite)
{
    double lo[3] = {0,0,0}, hi[3] = {1,1,1};
    OHExtraOctree octree(lo, hi, 4);

    double pt[3] = {0.7, 0.2, 0.9};
    octree.Insert(pt);

    double norm[3];
    octree.NormalizeCoordinates(pt, norm);
    OHExtraCell* cell = octree.pGetCellNormalized(norm);
    KRATOS_EXPECT_NE(cell, nullptr);
    KRATOS_EXPECT_TRUE(cell->IsLeaf());
}

// ===========================================================================
// SubdivideCell(cell_type*) pointer overload
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridSubdivideCellPointerForm, KratosCoreFastSuite)
{
    OHExtraOctree octree(4);
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 1);

    std::vector<OHExtraCell*> leaves;
    octree.GetAllLeavesVector(leaves);
    KRATOS_EXPECT_EQ(leaves.size(), 1u);

    octree.SubdivideCell(leaves[0]);
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 8);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridSubdivideCellPointerFormNopWhenAlreadySubdivided, KratosCoreFastSuite)
{
    OHExtraOctree octree(4);
    std::vector<OHExtraCell*> leaves;
    octree.GetAllLeavesVector(leaves);

    const int r0 = octree.SubdivideCell(leaves[0]);
    const int r1 = octree.SubdivideCell(leaves[0]); // second call → no-op
    KRATOS_EXPECT_EQ(r0, 0);
    KRATOS_EXPECT_EQ(r1, 1); // already subdivided → returns 1
}

// ===========================================================================
// pGetCellNormalized
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridpGetCellNormalizedReturnsLeaf, KratosCoreFastSuite)
{
    OHExtraOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0); // root → 8 L1 leaves

    // (0.25, 0.25, 0.25): key = floor(0.25*16) = 4; at L1: 4>>3 = 0 → child (0,0,0)
    const double pt[3] = {0.25, 0.25, 0.25};
    OHExtraCell* cell = octree.pGetCellNormalized(pt);

    KRATOS_EXPECT_NE(cell, nullptr);
    KRATOS_EXPECT_EQ(cell->GetLevel(), 1);
    KRATOS_EXPECT_TRUE(cell->IsLeaf());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridpGetCellNormalizedUndividedRootIsLeaf, KratosCoreFastSuite)
{
    OHExtraOctree octree(4);

    const double pt[3] = {0.5, 0.5, 0.5};
    OHExtraCell* cell = octree.pGetCellNormalized(pt);

    KRATOS_EXPECT_NE(cell, nullptr);
    KRATOS_EXPECT_EQ(cell->GetLevel(), 0); // root
    KRATOS_EXPECT_TRUE(cell->IsLeaf());
}

// ===========================================================================
// pGetCell(keys, level) — stops at requested level
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridpGetCellWithLevelStopsAtRequestedLevel, KratosCoreFastSuite)
{
    OHExtraOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0); // L0 → L1

    // Request level 0: loop exits immediately → root is returned
    OHExtraOctree::key_type keys[3] = {8, 8, 8};
    OHExtraCell* cell_l0 = octree.pGetCell(keys, 0);
    KRATOS_EXPECT_NE(cell_l0, nullptr);
    KRATOS_EXPECT_EQ(cell_l0->GetId(), 0);
    KRATOS_EXPECT_EQ(cell_l0->GetLevel(), 0);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridpGetCellWithLevelDescendsToL1, KratosCoreFastSuite)
{
    OHExtraOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0); // L0 → L1

    // Request level 1: descend once to the L1 child containing key (8,8,8)
    // At L1: cx = 8 >> (4-1) = 8>>3 = 1 → child (1,1,1)@L1
    OHExtraOctree::key_type keys[3] = {8, 8, 8};
    OHExtraCell* cell_l1 = octree.pGetCell(keys, 1);
    KRATOS_EXPECT_NE(cell_l1, nullptr);
    KRATOS_EXPECT_EQ(cell_l1->GetLevel(), 1);
}

// ===========================================================================
// GetNodeSubdivided
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGetNodeSubdividedTracksSubdivisions, KratosCoreFastSuite)
{
    OHExtraOctree octree(4);

    // Root not subdivided initially
    KRATOS_EXPECT_FALSE(octree.GetNodeSubdivided()[0]);

    octree.SubdivideCellByIdAndLevel(0, 0);

    // Root now marked subdivided; first L1 child (id=1) is not
    KRATOS_EXPECT_TRUE(octree.GetNodeSubdivided()[0]);
    KRATOS_EXPECT_FALSE(octree.GetNodeSubdivided()[1]);
}

// ===========================================================================
// CalcKeysNormalized
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCalcKeysNormalized3D, KratosCoreFastSuite)
{
    OHExtraOctree octree(4); // res = 2^4 = 16

    // Center of unit cube: normalized = (0.5, 0.5, 0.5) → key = floor(0.5*16) = 8
    const double coords[3] = {0.5, 0.5, 0.5};
    OHExtraOctree::key_type keys[3];
    octree.CalcKeysNormalized(coords, keys);

    KRATOS_EXPECT_EQ(keys[0], std::size_t{8});
    KRATOS_EXPECT_EQ(keys[1], std::size_t{8});
    KRATOS_EXPECT_EQ(keys[2], std::size_t{8});
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCalcKeysNormalizedCornerCase, KratosCoreFastSuite)
{
    OHExtraOctree octree(4); // res = 16

    // Origin: keys should be (0, 0, 0)
    const double origin[3] = {0.0, 0.0, 0.0};
    OHExtraOctree::key_type keys[3];
    octree.CalcKeysNormalized(origin, keys);

    KRATOS_EXPECT_EQ(keys[0], std::size_t{0});
    KRATOS_EXPECT_EQ(keys[1], std::size_t{0});
    KRATOS_EXPECT_EQ(keys[2], std::size_t{0});
}

// ===========================================================================
// CalculateCoordinates — world-space
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCalculateCoordinatesWorldSpace, KratosCoreFastSuite)
{
    // Bbox [-5,5] × [0,4] × [10,14]; MAX_DEPTH=4 → res=16
    double lo[3] = {-5, 0, 10};
    double hi[3] = { 5, 4, 14};
    OHExtraOctree octree(lo, hi, 4);

    // keys=(8,4,8): norm = (0.5, 0.25, 0.5)
    // world x = 0.5*(10) - 5 = 0.0
    // world y = 0.25*(4) + 0 = 1.0
    // world z = 0.5*(4) + 10 = 12.0
    OHExtraOctree::key_type keys[3] = {8, 4, 8};
    double coords[3];
    octree.CalculateCoordinates(keys, coords);

    KRATOS_EXPECT_NEAR(coords[0],  0.0, 1e-10);
    KRATOS_EXPECT_NEAR(coords[1],  1.0, 1e-10);
    KRATOS_EXPECT_NEAR(coords[2], 12.0, 1e-10);
}

// ===========================================================================
// NormalizeCoordinates — two-argument copy form
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridNormalizeCoordinatesCopyFormPreservesSource, KratosCoreFastSuite)
{
    double lo[3] = {0, 0, 0}, hi[3] = {2, 4, 8};
    OHExtraOctree octree(lo, hi, 4);

    double src[3] = {1.0, 2.0, 4.0}; // midpoints → normalized (0.5, 0.5, 0.5)
    double dst[3] = {0.0, 0.0, 0.0};
    octree.NormalizeCoordinates(src, dst);

    KRATOS_EXPECT_NEAR(dst[0], 0.5, 1e-12);
    KRATOS_EXPECT_NEAR(dst[1], 0.5, 1e-12);
    KRATOS_EXPECT_NEAR(dst[2], 0.5, 1e-12);

    // source unchanged
    KRATOS_EXPECT_NEAR(src[0], 1.0, 1e-12);
    KRATOS_EXPECT_NEAR(src[1], 2.0, 1e-12);
    KRATOS_EXPECT_NEAR(src[2], 4.0, 1e-12);
}

// ===========================================================================
// ScaleBackToOriginalCoordinate — two-argument copy form
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridScaleBackCopyFormRoundTrip, KratosCoreFastSuite)
{
    double lo[3] = {0, 0, 0}, hi[3] = {2, 4, 8};
    OHExtraOctree octree(lo, hi, 4);

    double pt[3] = {1.0, 2.0, 4.0};
    double norm[3];
    octree.NormalizeCoordinates(pt, norm); // norm ≈ (0.5, 0.5, 0.5)

    double back[3];
    octree.ScaleBackToOriginalCoordinate(norm, back);

    // Round-trip should recover original coordinates
    KRATOS_EXPECT_NEAR(back[0], 1.0, 1e-12);
    KRATOS_EXPECT_NEAR(back[1], 2.0, 1e-12);
    KRATOS_EXPECT_NEAR(back[2], 4.0, 1e-12);

    // norm source must be unchanged
    KRATOS_EXPECT_NEAR(norm[0], 0.5, 1e-12);
    KRATOS_EXPECT_NEAR(norm[1], 0.5, 1e-12);
    KRATOS_EXPECT_NEAR(norm[2], 0.5, 1e-12);
}

// ===========================================================================
// OctreeHybridCell — TransferObjectsToChildren
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellTransferObjectsToChildren, KratosCoreFastSuite)
{
    OHExtraCell parent(0, 0, 0, 0, 0); // root cell at level 0

    double obj1 = 1.0, obj2 = 2.0;
    parent.Insert(&obj1);
    parent.Insert(&obj2);
    KRATOS_EXPECT_EQ(parent.pGetObjects()->size(), 2u);

    // Children: default-constructed cells; IsIntersected always returns true
    OHExtraCell children[8];
    parent.TransferObjectsToChildren(children);

    // Parent cleared after transfer
    KRATOS_EXPECT_EQ(parent.pGetObjects()->size(), 0u);

    // Every child received both objects
    for (int i = 0; i < 8; ++i) {
        KRATOS_EXPECT_EQ(children[i].pGetObjects()->size(), 2u);
    }
}

// ===========================================================================
// OctreeHybridCell — pGetData / pGetDataPointer
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellDataPayloadNullInitially, KratosCoreFastSuite)
{
    OHExtraCell cell(0, 0, 0, 0, 0);
    KRATOS_EXPECT_EQ(cell.pGetData(), nullptr);
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellDataPayloadAssignAndRead, KratosCoreFastSuite)
{
    OHExtraCell cell(0, 0, 0, 0, 0);

    *cell.pGetDataPointer() = new OHExtraConfig::data_type{42};

    KRATOS_EXPECT_NE(cell.pGetData(), nullptr);
    KRATOS_EXPECT_EQ(cell.pGetData()->value, 42);
    // Cell destructor calls DeleteData → memory freed, no leak
}

} // namespace Kratos::Testing
