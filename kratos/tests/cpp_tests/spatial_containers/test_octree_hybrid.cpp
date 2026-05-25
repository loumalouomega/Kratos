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
#include <sstream>

// External includes

// Project includes
#include "testing/testing.h"
#include "spatial_containers/octree_hybrid.h"

namespace Kratos::Testing {

///@addtogroup HybridOctreeTests
///@{

///@name Configuration policy
///@{

/**
 * @brief Minimal TConfiguration for OctreeHybrid / OctreeHybridCell used in tests.
 *
 * MAX_DEPTH = 4 keeps the flat bit-array tiny (4 681 entries ≈ 584 bytes),
 * making every test fast and memory-safe.
 */
struct OHTestConfig {
    struct data_type { int value = 0; };
    using pointer_type = double*;

    static constexpr std::size_t MAX_DEPTH = 4;
    static constexpr std::size_t MIN_DEPTH = 1;
    static constexpr std::size_t DIMENSION = 3;

    static void DeleteData(data_type* p) { delete p; }

    static bool IsIntersected(pointer_type, double, const double*, const double*)
    { return true; }
};

using OHTestCell   = OctreeHybridCell<OHTestConfig>;
using OHTestOctree = OctreeHybrid<OHTestCell>;

///@}
///@name Level-table tests
///@{

/**
 * @brief mLevelRes[L] must equal 2^L for every level 0..MAX_DEPTH.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridLevelResMatchesPowerOfTwo, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    for (std::size_t L = 0; L <= 4; ++L) {
        KRATOS_EXPECT_EQ(octree.GetLevelRes(L), std::size_t{1} << L);
    }
}

/**
 * @brief mLevelId[L] = (8^L - 1)/7.
 *        Spot-checked against the reference StaticVars.h table.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridLevelIdMatchesGeometricSum, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    // Reference values from StaticVars.h: {0, 1, 9, 73, 585}
    const std::size_t expected[] = {0, 1, 9, 73, 585};
    for (std::size_t L = 0; L <= 4; ++L) {
        KRATOS_EXPECT_EQ(octree.GetLevelId(L), expected[L]);
    }
}

/**
 * @brief Total allocated nodes = levelId[depth+1] = (8^5-1)/7 = 4681.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridTotalNodeCountMatchesLevelId, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    KRATOS_EXPECT_EQ(octree.GetTotalNodeCount(), std::size_t{4681});
}

///@}
///@name Flat-index arithmetic tests
///@{

/**
 * @brief OctreeIdxToXyz and XyzToOctreeIdx are exact inverses.
 *        Tested for the root, selected level-1 cells, and all level-2 cells.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridFlatIndexRoundTrip, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    // Root
    {
        int x, y, z;
        octree.OctreeIdxToXyz(0, 0, x, y, z);
        KRATOS_EXPECT_EQ(x, 0); KRATOS_EXPECT_EQ(y, 0); KRATOS_EXPECT_EQ(z, 0);
    }

    // Level-1 spot checks.  At level 1, res=2: id = 1 + z*4 + y*2 + x
    KRATOS_EXPECT_EQ(octree.XyzToOctreeIdx(1, 0, 0, 0), 1);
    KRATOS_EXPECT_EQ(octree.XyzToOctreeIdx(1, 1, 0, 0), 2);
    KRATOS_EXPECT_EQ(octree.XyzToOctreeIdx(1, 0, 1, 0), 3);
    KRATOS_EXPECT_EQ(octree.XyzToOctreeIdx(1, 0, 0, 1), 5);
    KRATOS_EXPECT_EQ(octree.XyzToOctreeIdx(1, 1, 1, 1), 8);

    // Round-trip over all 64 level-2 cells.
    for (int id = 9; id <= 72; ++id) {
        int x, y, z;
        octree.OctreeIdxToXyz(id, 2, x, y, z);
        KRATOS_EXPECT_EQ(octree.XyzToOctreeIdx(2, x, y, z), id);
    }
}

/**
 * @brief Child() matches the expected flat indices for children of the root
 *        and of a level-1 cell.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridChildIdsAreCorrect, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    // Children of root (id=0, level=0).
    // id = levelId[1] + z*4 + y*2 + x; levelId[1]=1, res[1]=2.
    KRATOS_EXPECT_EQ(octree.Child(0, 0, 0), 1); // (0,0,0)@L1
    KRATOS_EXPECT_EQ(octree.Child(0, 0, 1), 2); // (1,0,0)@L1
    KRATOS_EXPECT_EQ(octree.Child(0, 0, 2), 3); // (0,1,0)@L1
    KRATOS_EXPECT_EQ(octree.Child(0, 0, 4), 5); // (0,0,1)@L1
    KRATOS_EXPECT_EQ(octree.Child(0, 0, 7), 8); // (1,1,1)@L1

    // Children of (0,0,0)@L1 (id=1).
    // id = levelId[2] + z*16 + y*4 + x; levelId[2]=9, res[2]=4.
    KRATOS_EXPECT_EQ(octree.Child(1, 1, 0), 9);  // (0,0,0)@L2
    KRATOS_EXPECT_EQ(octree.Child(1, 1, 7), 30); // (1,1,1)@L2 = 9+16+4+1
}

/**
 * @brief RefineBrothers for a level-1 cell returns exactly the 8 ids {1..8}
 *        (all children of the root). For the root it returns {0, -1, ..., -1}.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRefineBrothersReturnsAllSiblings, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    int sibs[8];

    // Siblings of (0,0,0)@L1 (id=1): all 8 children of root = ids 1..8.
    octree.RefineBrothers(1, 1, sibs);
    std::vector<int> sv(sibs, sibs + 8);
    std::sort(sv.begin(), sv.end());
    for (int i = 0; i < 8; ++i) KRATOS_EXPECT_EQ(sv[i], i + 1);

    // Siblings of (1,1,1)@L1 (id=8): same set.
    octree.RefineBrothers(8, 1, sibs);
    sv.assign(sibs, sibs + 8);
    std::sort(sv.begin(), sv.end());
    for (int i = 0; i < 8; ++i) KRATOS_EXPECT_EQ(sv[i], i + 1);

    // Root: no parent → siblings[0]=0, rest=-1.
    octree.RefineBrothers(0, 0, sibs);
    KRATOS_EXPECT_EQ(sibs[0], 0);
    for (int i = 1; i < 8; ++i) KRATOS_EXPECT_EQ(sibs[i], -1);
}

///@}
///@name Initialisation and subdivision tests
///@{

/**
 * @brief After construction the root (id=0) is the only leaf.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridInitializationHasOnlyRootAsLeaf, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 1);
    KRATOS_EXPECT_TRUE(octree.IsLeaf(0));

    const auto& ids = octree.GetLeafIds();
    KRATOS_EXPECT_EQ(ids.size(), std::size_t{1});
    KRATOS_EXPECT_EQ(ids[0], 0);
}

/**
 * @brief Subdividing the root marks it internal and produces 8 leaf children.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridSubdivideRootCreatesEightLeaves, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    KRATOS_EXPECT_EQ(octree.SubdivideCellByIdAndLevel(0, 0), 0);

    KRATOS_EXPECT_FALSE(octree.IsLeaf(0));
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 8);

    for (int id = 1; id <= 8; ++id) KRATOS_EXPECT_TRUE(octree.IsLeaf(id));
}

/**
 * @brief Subdividing a level-1 child gives 15 leaves total (7 at L1 + 8 at L2).
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridSubdivideChildCreatesNestedLeaves, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0);
    octree.SubdivideCellByIdAndLevel(1, 1);

    KRATOS_EXPECT_FALSE(octree.IsLeaf(1));
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 15);

    // Children of id=1 are ids 9..16.
    for (int id = 9; id <= 16; ++id) KRATOS_EXPECT_TRUE(octree.IsLeaf(id));
}

/**
 * @brief Calling SubdivideCellByIdAndLevel on an internal cell returns 1 and
 *        does not change the leaf count.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridSubdivideAlreadySubdividedIsNoop, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0);
    const int count_before = octree.GetLeafCount();

    KRATOS_EXPECT_EQ(octree.SubdivideCellByIdAndLevel(0, 0), 1);
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), count_before);
}

/**
 * @brief Attempting to subdivide a cell at MAX_DEPTH returns 1 with no effect.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridSubdivideAtMaxDepthIsNoop, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    // Descend all the way to level 4 from the root.
    octree.SubdivideCellByIdAndLevel(0, 0);
    octree.SubdivideCellByIdAndLevel(1, 1);
    octree.SubdivideCellByIdAndLevel(9, 2);
    const int l3_id = octree.XyzToOctreeIdx(3, 0, 0, 0);
    octree.SubdivideCellByIdAndLevel(l3_id, 3);

    const int l4_id = octree.XyzToOctreeIdx(4, 0, 0, 0);
    const int count_before = octree.GetLeafCount();

    KRATOS_EXPECT_EQ(octree.SubdivideCellByIdAndLevel(l4_id, 4), 1);
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), count_before);
}

///@}
///@name Leaf-list consistency tests
///@{

/**
 * @brief After several subdivisions, RebuildLeafListClean returns the same
 *        count as the tracked mLeafCount and contains no stale entries.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridRebuildLeafListIsConsistentAfterSubdivisions, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0);
    octree.SubdivideCellByIdAndLevel(1, 1);

    octree.RebuildLeafListClean();

    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 15);

    for (int id : octree.GetLeafIds()) {
        KRATOS_EXPECT_NE(id, 0);  // root must not appear
        KRATOS_EXPECT_NE(id, 1);  // subdivided cell must not appear
        KRATOS_EXPECT_TRUE(octree.IsLeaf(id));
    }
}

/**
 * @brief GetAllLeavesVector must not return internal cells even if their ids
 *        linger in the internal mLeafIds list.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGetAllLeavesVectorFiltersStaleEntries, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0);

    std::vector<OHTestCell*> leaves;
    octree.GetAllLeavesVector(leaves);

    KRATOS_EXPECT_EQ(static_cast<int>(leaves.size()), 8);
    for (OHTestCell* c : leaves) {
        KRATOS_EXPECT_NE(c, nullptr);
        KRATOS_EXPECT_TRUE(c->IsLeaf());
        KRATOS_EXPECT_EQ(c->GetLevel(), 1);
    }
}

///@}
///@name Level-lookup tests
///@{

/**
 * @brief GetLevelFromId must return the correct level for representative ids
 *        across all five levels (0–4).
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGetLevelFromIdReturnsBinarySearchResult, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    KRATOS_EXPECT_EQ(octree.GetLevelFromId(0),    0);  // root

    for (int id = 1;   id <= 8;   ++id) KRATOS_EXPECT_EQ(octree.GetLevelFromId(id), 1);
    for (int id = 9;   id <= 72;  ++id) KRATOS_EXPECT_EQ(octree.GetLevelFromId(id), 2);
    KRATOS_EXPECT_EQ(octree.GetLevelFromId(73),   3);
    KRATOS_EXPECT_EQ(octree.GetLevelFromId(584),  3);
    KRATOS_EXPECT_EQ(octree.GetLevelFromId(585),  4);
    KRATOS_EXPECT_EQ(octree.GetLevelFromId(4680), 4);
}

///@}
///@name Coordinate utility tests
///@{

/**
 * @brief CalcKeyNormalized maps [0,1] coordinates to integer keys in [0, 2^depth).
 *        At depth 4, the finest grid is 16 cells per axis.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCalcKeyNormalizedMapsToGrid, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    KRATOS_EXPECT_EQ(octree.CalcKeyNormalized(0.0),    std::size_t{0});
    KRATOS_EXPECT_EQ(octree.CalcKeyNormalized(0.25),   std::size_t{4});
    KRATOS_EXPECT_EQ(octree.CalcKeyNormalized(0.5),    std::size_t{8});
    KRATOS_EXPECT_EQ(octree.CalcKeyNormalized(0.9375), std::size_t{15});
    // 1.0 clamps to res-1 = 15.
    KRATOS_EXPECT_EQ(octree.CalcKeyNormalized(1.0),    std::size_t{15});
}

/**
 * @brief Cell sizes must equal 1/2^level at every level 0..MAX_DEPTH.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCalcSizeNormalizedMatchesInversePowerOfTwo, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    KRATOS_EXPECT_DOUBLE_EQ(octree.CalcSizeNormalizedAtLevel(0), 1.0);
    KRATOS_EXPECT_DOUBLE_EQ(octree.CalcSizeNormalizedAtLevel(1), 0.5);
    KRATOS_EXPECT_DOUBLE_EQ(octree.CalcSizeNormalizedAtLevel(2), 0.25);
    KRATOS_EXPECT_DOUBLE_EQ(octree.CalcSizeNormalizedAtLevel(3), 0.125);
    KRATOS_EXPECT_DOUBLE_EQ(octree.CalcSizeNormalizedAtLevel(4), 0.0625);
}

/**
 * @brief SetBoundingBox + NormalizeCoordinates maps the corners to (0,0,0) and
 *        (1,1,1); ScaleBackToOriginalCoordinate is the exact inverse.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridBoundingBoxNormalisationRoundTrip, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    const double low[3]  = {-10.0, 0.0, 5.0};
    const double high[3] = { 10.0, 4.0, 9.0};
    octree.SetBoundingBox(low, high);

    // Lower corner → (0, 0, 0)
    double c1[3] = {-10.0, 0.0, 5.0};
    octree.NormalizeCoordinates(c1);
    KRATOS_EXPECT_DOUBLE_EQ(c1[0], 0.0);
    KRATOS_EXPECT_DOUBLE_EQ(c1[1], 0.0);
    KRATOS_EXPECT_DOUBLE_EQ(c1[2], 0.0);

    // Upper corner → (1, 1, 1)
    double c2[3] = {10.0, 4.0, 9.0};
    octree.NormalizeCoordinates(c2);
    KRATOS_EXPECT_DOUBLE_EQ(c2[0], 1.0);
    KRATOS_EXPECT_DOUBLE_EQ(c2[1], 1.0);
    KRATOS_EXPECT_DOUBLE_EQ(c2[2], 1.0);

    // Midpoint (0.5, 0.5, 0.5) back to world.
    double n[3] = {0.5, 0.5, 0.5};
    octree.ScaleBackToOriginalCoordinate(n);
    KRATOS_EXPECT_DOUBLE_EQ(n[0],  0.0); // mid of [-10, 10]
    KRATOS_EXPECT_DOUBLE_EQ(n[1],  2.0); // mid of [0, 4]
    KRATOS_EXPECT_DOUBLE_EQ(n[2],  7.0); // mid of [5, 9]
}

/**
 * @brief CalculateCoordinatesNormalized(CalcKeyNormalized(p)) recovers p
 *        to within one voxel width (1/2^depth = 1/16 at depth 4).
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCoordinateKeyRoundTripIsWithinOneVoxel, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    const double p[3] = {0.375, 0.625, 0.125};
    std::size_t keys[3];
    keys[0] = octree.CalcKeyNormalized(p[0]);
    keys[1] = octree.CalcKeyNormalized(p[1]);
    keys[2] = octree.CalcKeyNormalized(p[2]);

    double recovered[3];
    octree.CalculateCoordinatesNormalized(keys, recovered);

    const double voxel = 1.0 / 16.0;
    KRATOS_EXPECT_NEAR(recovered[0], p[0], voxel);
    KRATOS_EXPECT_NEAR(recovered[1], p[1], voxel);
    KRATOS_EXPECT_NEAR(recovered[2], p[2], voxel);
}

///@}
///@name AABB overlap test
///@{

/**
 * @brief Collides() correctly identifies overlapping and disjoint boxes.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCollidesDetectsOverlap, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    const double lo1[3] = {0.0, 0.0, 0.0}, hi1[3] = {0.5, 0.5, 0.5};
    const double lo2[3] = {0.4, 0.4, 0.4}, hi2[3] = {1.0, 1.0, 1.0};
    const double lo3[3] = {0.6, 0.6, 0.6}, hi3[3] = {1.0, 1.0, 1.0};

    KRATOS_EXPECT_TRUE(octree.Collides(lo1, hi1, lo2, hi2));   // overlapping
    KRATOS_EXPECT_FALSE(octree.Collides(lo1, hi1, lo3, hi3));  // disjoint
    KRATOS_EXPECT_TRUE(octree.Collides(lo1, hi1, lo1, hi1));   // identical
}

///@}
///@name Point-query tests
///@{

/**
 * @brief Before any subdivision, any query returns the root cell.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridpGetCellReturnsRootBeforeSubdivision, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    const std::size_t k[3] = {0, 0, 0};
    OHTestCell* c = octree.pGetCell(k);
    KRATOS_EXPECT_NE(c, nullptr);
    KRATOS_EXPECT_EQ(c->GetId(), 0);
}

/**
 * @brief After subdividing the root, pGetCell returns the correct level-1 leaf.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridpGetCellReturnsCorrectLeafAfterSubdivision, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0);

    // Key (1,1,1): at level 1, key >> (4-1) = 0 → grid (0,0,0) → id=1.
    {
        const std::size_t k[3] = {1, 1, 1};
        OHTestCell* c = octree.pGetCell(k);
        KRATOS_EXPECT_NE(c, nullptr);
        KRATOS_EXPECT_EQ(c->GetLevel(), 1);
        KRATOS_EXPECT_EQ(c->GetGridX(), 0);
        KRATOS_EXPECT_EQ(c->GetGridY(), 0);
        KRATOS_EXPECT_EQ(c->GetGridZ(), 0);
    }

    // Key (9,9,9): at level 1, key >> 3 = 1 → grid (1,1,1) → id=8.
    {
        const std::size_t k[3] = {9, 9, 9};
        OHTestCell* c = octree.pGetCell(k);
        KRATOS_EXPECT_NE(c, nullptr);
        KRATOS_EXPECT_EQ(c->GetLevel(), 1);
        KRATOS_EXPECT_EQ(c->GetGridX(), 1);
        KRATOS_EXPECT_EQ(c->GetGridY(), 1);
        KRATOS_EXPECT_EQ(c->GetGridZ(), 1);
    }
}

/**
 * @brief InsertNormalized subdivides the path to MIN_DEPTH and leaves a
 *        leaf at the point's position.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridInsertNormalizedRefinesToMinDepth, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    const double p[3] = {0.25, 0.25, 0.25};
    octree.InsertNormalized(p);

    KRATOS_EXPECT_FALSE(octree.IsLeaf(0)); // root is internal
    KRATOS_EXPECT_GE(octree.GetLeafCount(), 8);

    std::size_t k[3];
    k[0] = octree.CalcKeyNormalized(p[0]);
    k[1] = octree.CalcKeyNormalized(p[1]);
    k[2] = octree.CalcKeyNormalized(p[2]);
    OHTestCell* c = octree.pGetCell(k);
    KRATOS_EXPECT_NE(c, nullptr);
    KRATOS_EXPECT_TRUE(c->IsLeaf());
}

///@}
///@name Bounding-box leaf query tests
///@{

/**
 * @brief Three bounding-box queries verify that only the expected leaves are
 *        returned after subdividing the root.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridGetLeavesInBBoxReturnsCorrectCells, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0);
    std::vector<OHTestCell*> found;

    // Lower octant [0,0.5)^3 → exactly 1 leaf.
    octree.GetLeavesInBoundingBoxNormalized(
        std::array<double,3>{0.01, 0.01, 0.01}.data(),
        std::array<double,3>{0.49, 0.49, 0.49}.data(),
        found);
    KRATOS_EXPECT_EQ(static_cast<int>(found.size()), 1);

    // Full domain → all 8 leaves.
    octree.GetLeavesInBoundingBoxNormalized(
        std::array<double,3>{0.0, 0.0, 0.0}.data(),
        std::array<double,3>{1.0, 1.0, 1.0}.data(),
        found);
    KRATOS_EXPECT_EQ(static_cast<int>(found.size()), 8);

    // Slab y ∈ [0.6, 1.0] → 4 cells with grid y=1.
    octree.GetLeavesInBoundingBoxNormalized(
        std::array<double,3>{0.0, 0.6, 0.0}.data(),
        std::array<double,3>{1.0, 1.0, 1.0}.data(),
        found);
    KRATOS_EXPECT_EQ(static_cast<int>(found.size()), 4);
}

///@}
///@name Balance-enforcement tests
///@{

/**
 * @brief After a 2-level imbalanced subdivision, Constrain2To1 ensures every
 *        pair of face-adjacent leaves has level difference at most 1.
 *
 * Setup:
 *   - root → L1 (8 cells, ids 1..8)
 *   - (0,0,0)@L1 (id=1) → L2 (8 cells, ids 9,10,13,14,25,26,29,30)
 *   - (1,0,0)@L2 (id=10) → L3 (8 cells at grid positions (2..3, 0..1, 0..1)@L3)
 *
 * The L3 cells at x=3 (i.e. (3,*,*)@L3) have their +X face at normalised
 * x = 0.5, which is shared with the unrefined L1 cell (1,0,0)@L1 (id=2).
 * Level difference = 3 - 1 = 2 → violation.
 *
 * Constrain2To1 must subdivide id=2 and all its L1 siblings (ids 3..8),
 * which increases the leaf count from 22 to at least 22 + 6*8 - 6 = 64.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridConstrain2To1EnforcesBalance, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    // (1,0,0)@L2 = levelId[2] + 0*16 + 0*4 + 1 = 9 + 1 = 10
    const int boundary_l2_id = 10;

    octree.SubdivideCellByIdAndLevel(0, 0);              // root → L1
    octree.SubdivideCellByIdAndLevel(1, 1);              // (0,0,0)@L1 → L2
    octree.SubdivideCellByIdAndLevel(boundary_l2_id, 2); // (1,0,0)@L2 → L3
    octree.RebuildLeafListClean();
    // Leaves: 7@L1 + 7@L2 + 8@L3 = 22
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 22);

    octree.Constrain2To1();

    // (1,0,0)@L1 and its 6 unsubdivided siblings must each be subdivided
    // → at least 6*8 - 6 = 42 more leaves.
    KRATOS_EXPECT_GT(octree.GetLeafCount(), 22);

    // Verify 2:1 rule over all face neighbours.
    for (int leaf_id : octree.GetLeafIds()) {
        const int level = octree.GetLevelFromId(leaf_id);
        const int res   = static_cast<int>(octree.GetLevelRes(static_cast<std::size_t>(level)));
        int lx, ly, lz;
        octree.OctreeIdxToXyz(leaf_id, level, lx, ly, lz);

        const int ddx[6] = {-1,1,0,0,0,0};
        const int ddy[6] = { 0,0,-1,1,0,0};
        const int ddz[6] = { 0,0,0,0,-1,1};

        for (int d = 0; d < 6; ++d) {
            const int nx = lx + ddx[d], ny = ly + ddy[d], nz = lz + ddz[d];
            if (nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res) continue;

            const double sz = octree.CalcSizeNormalizedAtLevel(static_cast<std::size_t>(level));
            const std::size_t pk[3] = {
                octree.CalcKeyNormalized((nx + 0.5) * sz),
                octree.CalcKeyNormalized((ny + 0.5) * sz),
                octree.CalcKeyNormalized((nz + 0.5) * sz)
            };

            const OHTestCell* nb = octree.pGetCell(pk);
            if (!nb) continue;
            KRATOS_EXPECT_LE(std::abs(level - nb->GetLevel()), 1);
        }
    }
}

/**
 * @brief After the same boundary-L2 imbalanced setup, StrongConstrain2To1
 *        leaves the tree balanced and the root internal.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridStrongConstrain2To1EnforcesBalance, KratosCoreFastSuite)
{
    OHTestOctree octree(4);

    const int boundary_l2_id = 10; // (1,0,0)@L2
    octree.SubdivideCellByIdAndLevel(0, 0);
    octree.SubdivideCellByIdAndLevel(1, 1);
    octree.SubdivideCellByIdAndLevel(boundary_l2_id, 2);
    octree.RebuildLeafListClean();
    KRATOS_EXPECT_EQ(octree.GetLeafCount(), 22);

    octree.StrongConstrain2To1();

    KRATOS_EXPECT_FALSE(octree.IsLeaf(0));
    KRATOS_EXPECT_GT(octree.GetLeafCount(), 22);
}

///@}
///@name I/O smoke tests
///@{

/**
 * @brief Info(), PrintInfo(), and PrintData() must produce non-empty strings.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridInfoAndPrintProduceNonEmptyOutput, KratosCoreFastSuite)
{
    OHTestOctree octree(4);
    octree.SubdivideCellByIdAndLevel(0, 0);

    KRATOS_EXPECT_FALSE(octree.Info().empty());

    std::ostringstream oss;
    octree.PrintInfo(oss);
    KRATOS_EXPECT_FALSE(oss.str().empty());

    oss.str("");
    octree.PrintData(oss);
    KRATOS_EXPECT_FALSE(oss.str().empty());
}

///@}
///@name OctreeHybridCell tests
///@{

/**
 * @brief The root cell (level=0, grid=(0,0,0)) must cover [0,1]^3 exactly.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellRootCoversUnitCube, KratosCoreFastSuite)
{
    OHTestCell root(0, 0, 0, 0, 0);
    double lo[3], hi[3];
    root.GetMinPointNormalized(lo);
    root.GetMaxPointNormalized(hi);

    KRATOS_EXPECT_DOUBLE_EQ(lo[0], 0.0); KRATOS_EXPECT_DOUBLE_EQ(lo[1], 0.0); KRATOS_EXPECT_DOUBLE_EQ(lo[2], 0.0);
    KRATOS_EXPECT_DOUBLE_EQ(hi[0], 1.0); KRATOS_EXPECT_DOUBLE_EQ(hi[1], 1.0); KRATOS_EXPECT_DOUBLE_EQ(hi[2], 1.0);
    KRATOS_EXPECT_DOUBLE_EQ(root.CalcSizeNormalized(), 1.0);
    KRATOS_EXPECT_TRUE(root.IsLeaf());
}

/**
 * @brief Level-1 cell (1,0,0) covers [0.5,1.0]×[0,0.5]×[0,0.5].
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellLevel1CoversCorrectOctant, KratosCoreFastSuite)
{
    // id = levelId[1] + 0*4 + 0*2 + 1 = 2
    OHTestCell c(2, 1, 1, 0, 0);
    double lo[3], hi[3];
    c.GetMinPointNormalized(lo);
    c.GetMaxPointNormalized(hi);

    KRATOS_EXPECT_DOUBLE_EQ(lo[0], 0.5); KRATOS_EXPECT_DOUBLE_EQ(lo[1], 0.0); KRATOS_EXPECT_DOUBLE_EQ(lo[2], 0.0);
    KRATOS_EXPECT_DOUBLE_EQ(hi[0], 1.0); KRATOS_EXPECT_DOUBLE_EQ(hi[1], 0.5); KRATOS_EXPECT_DOUBLE_EQ(hi[2], 0.5);
    KRATOS_EXPECT_DOUBLE_EQ(c.CalcSizeNormalized(), 0.5);
}

/**
 * @brief Level-2 cell (2,1,3) covers [0.5,0.75]×[0.25,0.5]×[0.75,1.0].
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellLevel2ArbitraryPosition, KratosCoreFastSuite)
{
    OHTestCell c(0, 2, 2, 1, 3);
    double lo[3], hi[3];
    c.GetMinPointNormalized(lo);
    c.GetMaxPointNormalized(hi);

    KRATOS_EXPECT_DOUBLE_EQ(lo[0], 0.50); KRATOS_EXPECT_DOUBLE_EQ(lo[1], 0.25); KRATOS_EXPECT_DOUBLE_EQ(lo[2], 0.75);
    KRATOS_EXPECT_DOUBLE_EQ(hi[0], 0.75); KRATOS_EXPECT_DOUBLE_EQ(hi[1], 0.50); KRATOS_EXPECT_DOUBLE_EQ(hi[2], 1.00);
}

/**
 * @brief The IsLeaf flag returned from GetAllLeavesVector reflects subdivision state.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellIsLeafFlagFollowsSubdivision, KratosCoreFastSuite)
{
    OHTestOctree oct(4);

    // Initially the root is a leaf.
    {
        std::vector<OHTestCell*> leaves;
        oct.GetAllLeavesVector(leaves);
        KRATOS_EXPECT_EQ(leaves.size(), std::size_t{1});
        KRATOS_EXPECT_TRUE(leaves[0]->IsLeaf());
    }

    oct.SubdivideCellByIdAndLevel(0, 0);

    // After subdivision: 8 leaves at level 1.
    {
        std::vector<OHTestCell*> leaves;
        oct.GetAllLeavesVector(leaves);
        KRATOS_EXPECT_EQ(leaves.size(), std::size_t{8});
        for (OHTestCell* c : leaves) KRATOS_EXPECT_TRUE(c->IsLeaf());
    }
}

/**
 * @brief Insert two objects into a leaf, verify size, then EmptyObjects clears them.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellObjectStorageInsertAndClear, KratosCoreFastSuite)
{
    OHTestOctree oct(4);
    oct.SubdivideCellByIdAndLevel(0, 0);

    std::vector<OHTestCell*> leaves;
    oct.GetAllLeavesVector(leaves);
    KRATOS_EXPECT_FALSE(leaves.empty());

    OHTestCell* c = leaves[0];
    KRATOS_EXPECT_EQ(c->pGetObjects()->size(), std::size_t{0});

    static double d1 = 1.0, d2 = 2.0;
    c->Insert(&d1);
    c->Insert(&d2);
    KRATOS_EXPECT_EQ(c->pGetObjects()->size(), std::size_t{2});

    c->EmptyObjects();
    KRATOS_EXPECT_EQ(c->pGetObjects()->size(), std::size_t{0});
}

/**
 * @brief OctreeHybridCell Info(), PrintInfo(), and PrintData() produce output.
 */
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridCellInfoAndPrintProduceNonEmptyOutput, KratosCoreFastSuite)
{
    OHTestCell cell(0, 0, 0, 0, 0);

    KRATOS_EXPECT_FALSE(cell.Info().empty());

    std::ostringstream oss;
    cell.PrintInfo(oss);
    KRATOS_EXPECT_FALSE(oss.str().empty());

    oss.str("");
    cell.PrintData(oss);
    KRATOS_EXPECT_FALSE(oss.str().empty());
}

///@}

} // namespace Kratos::Testing
