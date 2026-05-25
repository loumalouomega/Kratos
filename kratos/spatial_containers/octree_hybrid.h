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

#pragma once

// System includes
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Project includes
#include "octree_hybrid_cell.h"

/**
 * @file octree_hybrid.h
 * @brief Defines OctreeHybrid, a linearised complete-octree spatial container.
 *
 * ## Motivation
 *
 * OctreeBinary uses a pointer-based recursive tree: each cell owns a heap-
 * allocated array of eight children.  This is flexible, but the indirect
 * memory access pattern is cache-unfriendly and the per-cell overhead is high
 * when millions of cells are active.
 *
 * OctreeHybrid instead uses a **linearised complete octree** backed by a flat
 * bit-vector (`mNodeSubdivided`).  Every node that **could** exist in the
 * complete octree has a unique integer identifier (flat index), computed
 * entirely by arithmetic on (level, x, y, z) — no pointer chasing is required.
 * Only the leaf IDs and cells that carry data are kept in additional structures.
 *
 * ## Flat index arithmetic
 *
 * Precomputed tables (computed once in Initialize):
 * @code
 *   mLevelRes[L] = 2^L           // grid cells per axis at level L
 *   mLevelId[L]  = (8^L - 1)/7  // start of level L in mNodeSubdivided
 * @endcode
 *
 * The flat index of the cell at grid position (x, y, z) at level L is:
 * @code
 *   id = mLevelId[L] + z * mLevelRes[L]^2 + y * mLevelRes[L] + x
 * @endcode
 *
 * ## Level convention
 *
 * Unlike OctreeBinaryCell (where ROOT_LEVEL is the coarsest),
 * OctreeHybrid follows the reference HybridOctree_Hex paper:
 * - Level 0  = root / coarsest (1×1×1 grid)
 * - Level L  = grid of 2^L × 2^L × 2^L cells
 * - Level MAX_DEPTH = finest allowed leaf
 *
 * The maximum usable MAX_DEPTH is 10 (matching the reference code's VOXEL_SIZE).
 * Higher values increase memory for `mNodeSubdivided` rapidly:
 * | depth | flat-array size (bits) | approx. bytes        |
 * |-------|------------------------|----------------------|
 * |  4    | 37 449                 | ~4.6 KB              |
 * |  6    | 2 396 745              | ~292 KB              |
 * |  8    | 153 391 689            | ~18 MB               |
 * | 10    | 1 227 133 513 (!)      | ~149 MB              |
 *
 * `std::vector<bool>` packs bits, so each entry is 1 bit.
 *
 * ## Public API compatibility with OctreeBinary
 *
 * The following OctreeBinary methods are present with identical signatures:
 * - SetBoundingBox / NormalizeCoordinates / ScaleBackToOriginalCoordinate
 * - CalcKeyNormalized / CalculateCoordinatesNormalized / CalcSizeNormalized
 * - InsertNormalized / Insert
 * - SubdivideCell (signature differs slightly: takes id + level)
 * - Constrain2To1
 * - GetAllLeavesVector
 * - GetLeavesInBoundingBoxNormalized
 * - pGetCellNormalized / pGetCell
 * - Collides
 *
 * @tparam TCellType  Must be an instantiation of OctreeHybridCell<TConfiguration>.
 */
template<class TCellType>
class OctreeHybrid
{
public:
    ///@name Type Definitions
    ///@{

    using cell_type         = TCellType;
    using configuration_type = typename cell_type::configuration_type;
    using key_type          = typename cell_type::key_type;
    using coordinate_type   = double;

    static constexpr std::size_t CHILDREN_NUMBER = cell_type::CHILDREN_NUMBER;
    static constexpr std::size_t DIMENSION       = cell_type::DIMENSION;
    /// Finest level at which a leaf may exist.
    static constexpr std::size_t MAX_DEPTH       = cell_type::MAX_DEPTH;
    /// Coarsest level at which a leaf is permitted (refinement stops here).
    static constexpr std::size_t MIN_DEPTH       = cell_type::MIN_DEPTH;

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Constructs an OctreeHybrid and calls Initialize(max_depth).
     *
     * After construction the octree contains only the root cell (level 0,
     * id 0) as its sole leaf.  The normalisation parameters are set to the
     * identity mapping (scale = 1, offset = 0).
     *
     * @param max_depth  Maximum refinement depth.  Must be in [1, MAX_DEPTH].
     *                   Defaults to MAX_DEPTH (the coarsest allowed by the
     *                   configuration's compile-time constant).
     */
    explicit OctreeHybrid(std::size_t max_depth = MAX_DEPTH)
        : mDepth(max_depth)
        , mLeafCount(0)
    {
        for (std::size_t i = 0; i < DIMENSION; i++) {
            mScaleFactor[i] = 1.0;
            mOffset[i]      = 0.0;
        }
        Initialize(max_depth);
    }

    /**
     * @brief Constructs with explicit bounding-box normalisation.
     *
     * Convenience overload that calls SetBoundingBox immediately.
     *
     * @param Low        Pointer to the three lower-bound world coordinates.
     * @param High       Pointer to the three upper-bound world coordinates.
     * @param max_depth  Maximum refinement depth.
     */
    OctreeHybrid(
        const double* Low,
        const double* High,
        std::size_t   max_depth = MAX_DEPTH)
        : mDepth(max_depth)
        , mLeafCount(0)
    {
        Initialize(max_depth);
        SetBoundingBox(Low, High);
    }

    /// Destructor — all cells owned by mCells are freed automatically.
    ~OctreeHybrid() = default;

    // Non-copyable (owns unique_ptrs, large bit-array).
    OctreeHybrid(const OctreeHybrid&)            = delete;
    OctreeHybrid& operator=(const OctreeHybrid&) = delete;

    ///@}
    ///@name Initialisation
    ///@{

    /**
     * @brief Initialises (or re-initialises) the octree for the given depth.
     *
     * This method:
     *  1. Computes `mLevelId` and `mLevelRes` tables.
     *  2. Resizes `mNodeSubdivided` to the full complete-octree size and
     *     sets all entries to false (no cell is subdivided).
     *  3. Creates the root cell (level 0, id 0) and stores it in `mCells`.
     *  4. Sets `mLeafIds = {0}` so the root is the only leaf.
     *
     * Calling Initialize() on an already-populated octree discards all
     * previous subdivision state and object associations.
     *
     * @param depth  Maximum refinement depth.  Must be >= 1 and <= MAX_DEPTH.
     */
    void Initialize(std::size_t depth)
    {
        if (depth < 1 || depth > MAX_DEPTH) {
            throw std::invalid_argument(
                "OctreeHybrid::Initialize: depth must be in [1, MAX_DEPTH]");
        }
        mDepth = depth;

        // --- Precompute level tables -------------------------------------------
        // mLevelRes[L] = 2^L
        // mLevelId[L]  = (8^L - 1) / 7  (number of cells at levels 0..L-1)
        mLevelRes.resize(mDepth + 2);
        mLevelId.resize(mDepth + 2);
        mLevelRes[0] = 1;
        mLevelId[0]  = 0;
        for (std::size_t L = 1; L <= mDepth + 1; L++) {
            mLevelRes[L] = mLevelRes[L-1] * 2;
            // mLevelId[L] = mLevelId[L-1] + levelRes[L-1]^3 = mLevelId[L-1] + 8^(L-1)
            mLevelId[L]  = mLevelId[L-1] + mLevelRes[L-1] * mLevelRes[L-1] * mLevelRes[L-1];
        }

        // --- Allocate flat subdivision array -----------------------------------
        // Total size = mLevelId[depth+1] (all nodes from level 0 to level depth).
        const std::size_t total_nodes = mLevelId[mDepth + 1];
        mNodeSubdivided.assign(total_nodes, false);

        // --- Initialise leaf list and cell map --------------------------------
        mLeafIds.clear();
        mCells.clear();

        // Root cell: level 0, grid (0,0,0), flat id 0.
        mLeafIds.push_back(0);
        mLeafCount = 1;
        mCells[0] = std::make_unique<cell_type>(0, 0, 0, 0, 0);
    }

    ///@}
    ///@name Bounding box and coordinate utilities
    ///@{

    /**
     * @brief Sets the world-space bounding box and computes normalisation factors.
     *
     * After this call, NormalizeCoordinates() maps world points in
     * [Low[i], High[i]] to [0, 1] for each axis i.
     *
     * @param Low   Pointer to three lower-bound world coordinates.
     * @param High  Pointer to three upper-bound world coordinates.
     */
    void SetBoundingBox(const coordinate_type* Low, const coordinate_type* High)
    {
        for (std::size_t i = 0; i < DIMENSION; i++) {
            mScaleFactor[i] = 1.0 / (High[i] - Low[i]);
            mOffset[i]      = -Low[i];
        }
    }

    /**
     * @brief Normalises world coordinates in-place to [0, 1]^3.
     *
     * @param Coordinates  In/out array of length DIMENSION.
     */
    void NormalizeCoordinates(coordinate_type* Coordinates) const
    {
        for (std::size_t i = 0; i < DIMENSION; i++) {
            Coordinates[i] = (Coordinates[i] + mOffset[i]) * mScaleFactor[i];
        }
    }

    /**
     * @brief Normalises world coordinates into a separate output array.
     *
     * @param Coordinates            Input world coordinates (length DIMENSION).
     * @param NormalizedCoordinates  Output normalised coordinates (length DIMENSION).
     */
    void NormalizeCoordinates(
        const coordinate_type* Coordinates,
        coordinate_type*       NormalizedCoordinates) const
    {
        for (std::size_t i = 0; i < DIMENSION; i++) {
            NormalizedCoordinates[i] = (Coordinates[i] + mOffset[i]) * mScaleFactor[i];
        }
    }

    /**
     * @brief Scales normalised coordinates back to world space in-place.
     *
     * @param ThisCoordinates  In/out normalised coordinates (length DIMENSION).
     */
    void ScaleBackToOriginalCoordinate(coordinate_type* ThisCoordinates) const
    {
        for (std::size_t i = 0; i < DIMENSION; i++) {
            ThisCoordinates[i] = ThisCoordinates[i] / mScaleFactor[i] - mOffset[i];
        }
    }

    /**
     * @brief Scales normalised coordinates back to world space into a separate array.
     *
     * @param ThisCoordinates    Input normalised coordinates (length DIMENSION).
     * @param ResultCoordinates  Output world coordinates (length DIMENSION).
     */
    void ScaleBackToOriginalCoordinate(
        const coordinate_type* ThisCoordinates,
        coordinate_type*       ResultCoordinates) const
    {
        for (std::size_t i = 0; i < DIMENSION; i++) {
            ResultCoordinates[i] = ThisCoordinates[i] / mScaleFactor[i] - mOffset[i];
        }
    }

    /**
     * @brief Converts a single normalised coordinate in [0, 1] to an integer key.
     *
     * The key is in [0, 2^mDepth].  At level L the grid position of the key
     * is obtained by right-shifting by (mDepth - L):
     * @code
     *   grid_pos_at_L = key >> (mDepth - L)
     * @endcode
     *
     * @param coordinate  Normalised coordinate in [0, 1].
     * @return            Integer key in [0, 2^mDepth].
     */
    key_type CalcKeyNormalized(coordinate_type coordinate) const
    {
        assert(coordinate >= 0.0 && coordinate <= 1.0);
        const std::size_t res = mLevelRes[mDepth]; // 2^mDepth
        return static_cast<key_type>(
            std::min(static_cast<double>(res - 1),
                     std::floor(coordinate * static_cast<double>(res))));
    }

    /**
     * @brief Converts three normalised coordinates to integer keys.
     *
     * @param NormalizedCoordinates  Input normalised coordinates (length DIMENSION).
     * @param keys                   Output keys (length DIMENSION).
     */
    void CalcKeysNormalized(
        const coordinate_type* NormalizedCoordinates,
        key_type*              keys) const
    {
        for (std::size_t i = 0; i < DIMENSION; i++) {
            keys[i] = CalcKeyNormalized(NormalizedCoordinates[i]);
        }
    }

    /**
     * @brief Computes the normalised side length of a cell at the given level.
     *
     * @param level  Refinement level (0 = coarsest, MAX_DEPTH = finest).
     * @return       1.0 / 2^level.
     */
    double CalcSizeNormalizedAtLevel(std::size_t level) const
    {
        return 1.0 / static_cast<double>(mLevelRes[level]);
    }

    /**
     * @brief Computes the normalised side length of the given cell.
     *
     * @param cell  Pointer to the cell.
     * @return      1.0 / 2^cell->GetLevel().
     */
    double CalcSizeNormalized(const cell_type* cell) const
    {
        return CalcSizeNormalizedAtLevel(static_cast<std::size_t>(cell->GetLevel()));
    }

    /**
     * @brief Computes normalised coordinates from an array of keys.
     *
     * @param keys                   Input keys (length DIMENSION).
     * @param NormalizedCoordinates  Output normalised coordinates (length DIMENSION).
     */
    void CalculateCoordinatesNormalized(
        const key_type* keys,
        coordinate_type* NormalizedCoordinates) const
    {
        const double scale = 1.0 / static_cast<double>(mLevelRes[mDepth]);
        for (std::size_t i = 0; i < DIMENSION; i++) {
            NormalizedCoordinates[i] = static_cast<double>(keys[i]) * scale;
        }
    }

    /**
     * @brief Computes world-space coordinates from keys (reverses normalisation).
     *
     * @param keys             Input keys (length DIMENSION).
     * @param ResultCoordinates Output world-space coordinates (length DIMENSION).
     */
    void CalculateCoordinates(
        const key_type* keys,
        coordinate_type* ResultCoordinates) const
    {
        const double scale = 1.0 / static_cast<double>(mLevelRes[mDepth]);
        for (std::size_t i = 0; i < DIMENSION; i++) {
            ResultCoordinates[i] = static_cast<double>(keys[i]) * scale / mScaleFactor[i] - mOffset[i];
        }
    }

    ///@}
    ///@name Flat-index arithmetic
    ///@{

    /**
     * @brief Converts a flat octree id and its level to (x, y, z) grid coordinates.
     *
     * Inverse of the formula:
     * @code
     *   id = mLevelId[level] + z * res^2 + y * res + x,   res = 2^level
     * @endcode
     *
     * @param octree_id  Flat index in mNodeSubdivided.
     * @param level      Refinement level of the cell.
     * @param x          Output X grid coordinate.
     * @param y          Output Y grid coordinate.
     * @param z          Output Z grid coordinate.
     */
    void OctreeIdxToXyz(int octree_id, int level, int& x, int& y, int& z) const
    {
        const int idx = octree_id - static_cast<int>(mLevelId[level]);
        const int res = static_cast<int>(mLevelRes[level]);
        x = idx % res;
        y = (idx / res) % res;
        z = idx / (res * res);
    }

    /**
     * @brief Computes the flat id of a given (x, y, z) position at a given level.
     *
     * @param level  Refinement level.
     * @param x      X grid coordinate.
     * @param y      Y grid coordinate.
     * @param z      Z grid coordinate.
     * @return       Flat index in mNodeSubdivided.
     */
    int XyzToOctreeIdx(int level, int x, int y, int z) const
    {
        const int res = static_cast<int>(mLevelRes[level]);
        return static_cast<int>(mLevelId[level]) + z * res * res + y * res + x;
    }

    /**
     * @brief Returns the level of a node given its flat id.
     *
     * Uses binary search on mLevelId (O(log mDepth), effectively O(1)).
     *
     * @param octree_id  Flat index in mNodeSubdivided.
     * @return           Refinement level (0 to mDepth).
     */
    int GetLevelFromId(int octree_id) const
    {
        int lo = 0, hi = static_cast<int>(mDepth);
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (static_cast<int>(mLevelId[mid]) <= octree_id)
                lo = mid;
            else
                hi = mid - 1;
        }
        return lo;
    }

    /**
     * @brief Computes the flat id of a child of a given parent cell.
     *
     * Child indices map to (dx, dy, dz) offsets by bit-decomposition:
     * @code
     *   dx = child_index & 1
     *   dy = (child_index >> 1) & 1
     *   dz = (child_index >> 2) & 1
     * @endcode
     * The child grid position at level parent_level+1 is:
     * @code
     *   child_x = 2 * parent_x + dx
     * @endcode
     *
     * @param parent_id     Flat id of the parent cell.
     * @param parent_level  Level of the parent cell.
     * @param child_index   Child index in [0, 7].
     * @return              Flat id of the child cell.
     */
    int Child(int parent_id, int parent_level, int child_index) const
    {
        int px, py, pz;
        OctreeIdxToXyz(parent_id, parent_level, px, py, pz);

        const int dx = child_index & 1;
        const int dy = (child_index >> 1) & 1;
        const int dz = (child_index >> 2) & 1;

        const int child_level = parent_level + 1;
        const int child_x     = 2 * px + dx;
        const int child_y     = 2 * py + dy;
        const int child_z     = 2 * pz + dz;

        return XyzToOctreeIdx(child_level, child_x, child_y, child_z);
    }

    /**
     * @brief Fills `siblings` with the flat ids of all eight cells that share
     *        the same parent as `octree_id`.
     *
     * The list always includes `octree_id` itself at its correct position.
     * If `octree_id` is the root (level 0), only one sibling (itself) is valid
     * and the remaining seven entries are set to -1.
     *
     * @param octree_id  Flat id of any cell at level `level`.
     * @param level      Refinement level of the cell.
     * @param siblings   Output array of exactly 8 integers.
     */
    void RefineBrothers(int octree_id, int level, int siblings[8]) const
    {
        if (level == 0) {
            // Root has no parent; only itself.
            siblings[0] = octree_id;
            for (int i = 1; i < 8; i++) siblings[i] = -1;
            return;
        }

        // Parent grid position (integer division rounds toward zero).
        int cx, cy, cz;
        OctreeIdxToXyz(octree_id, level, cx, cy, cz);

        const int parent_level = level - 1;
        const int parent_x     = cx / 2;
        const int parent_y     = cy / 2;
        const int parent_z     = cz / 2;
        const int parent_id    = XyzToOctreeIdx(parent_level, parent_x, parent_y, parent_z);

        for (int i = 0; i < 8; i++) {
            siblings[i] = Child(parent_id, parent_level, i);
        }
    }

    /**
     * @brief Returns true if a cell is currently a leaf (not subdivided).
     *
     * @param octree_id  Flat id of the cell.
     * @return           true if mNodeSubdivided[octree_id] == false.
     */
    bool IsLeaf(int octree_id) const
    {
        return !mNodeSubdivided[static_cast<std::size_t>(octree_id)];
    }

    ///@}
    ///@name Insertion and subdivision
    ///@{

    /**
     * @brief Inserts a point given in normalised [0, 1]^3 coordinates.
     *
     * The octree is refined along the path to the point until either
     * MIN_DEPTH or an existing subdivision boundary is reached.
     *
     * @param point  Normalised coordinates (length DIMENSION).
     */
    void InsertNormalized(const coordinate_type* point)
    {
        key_type kx = CalcKeyNormalized(point[0]);
        key_type ky = CalcKeyNormalized(point[1]);
        key_type kz = CalcKeyNormalized(point[2]);

        int id    = 0;    // start at root
        int level = 0;

        while (level < static_cast<int>(MIN_DEPTH)) {
            if (IsLeaf(id)) {
                SubdivideCellByIdAndLevel(id, level);
            }
            level++;
            // Descend to the child containing the key.
            const int cx = static_cast<int>(kx >> (mDepth - static_cast<std::size_t>(level)));
            const int cy = static_cast<int>(ky >> (mDepth - static_cast<std::size_t>(level)));
            const int cz = static_cast<int>(kz >> (mDepth - static_cast<std::size_t>(level)));
            id = XyzToOctreeIdx(level, cx, cy, cz);
        }
    }

    /**
     * @brief Inserts a point given in world-space coordinates.
     *
     * Normalises the point first, then calls InsertNormalized.
     *
     * @param point  World-space coordinates (length DIMENSION).
     */
    void Insert(const coordinate_type* point)
    {
        coordinate_type norm[DIMENSION];
        NormalizeCoordinates(point, norm);
        InsertNormalized(norm);
    }

    /**
     * @brief Subdivides the cell identified by `octree_id` at `level`.
     *
     * The method:
     *  1. Sets `mNodeSubdivided[octree_id] = true`.
     *  2. Updates the cell in `mCells` to mark it as non-leaf.
     *  3. Creates eight child cells at `level+1` and registers them in
     *     `mCells` and `mLeafIds`.
     *  4. Decrements `mLeafCount` by 1 and increments it by 8.
     *
     * Calling this on an already-subdivided cell is a no-op.
     *
     * @param octree_id  Flat id of the cell to subdivide.
     * @param level      Refinement level of the cell.  Must be < mDepth.
     * @return           0 on success, 1 if the cell was already subdivided or at max depth.
     */
    int SubdivideCellByIdAndLevel(int octree_id, int level)
    {
        if (static_cast<std::size_t>(level) >= mDepth) return 1;  // at max depth
        if (mNodeSubdivided[static_cast<std::size_t>(octree_id)]) return 1; // already subdivided

        mNodeSubdivided[static_cast<std::size_t>(octree_id)] = true;

        // Mark the cached cell as non-leaf.
        auto it = mCells.find(octree_id);
        if (it != mCells.end()) {
            it->second->SetIsLeaf(false);
        }

        // Remove from leaf list (will be rebuilt lazily; just track count).
        mLeafCount--;

        // Create eight child cells.
        int x_p, y_p, z_p;
        OctreeIdxToXyz(octree_id, level, x_p, y_p, z_p);
        const int child_level = level + 1;

        for (int i = 0; i < 8; i++) {
            const int dx = i & 1;
            const int dy = (i >> 1) & 1;
            const int dz = (i >> 2) & 1;
            const int cx = 2 * x_p + dx;
            const int cy = 2 * y_p + dy;
            const int cz = 2 * z_p + dz;
            const int child_id = XyzToOctreeIdx(child_level, cx, cy, cz);

            mLeafIds.push_back(child_id);
            mLeafCount++;
            mCells[child_id] = std::make_unique<cell_type>(child_id, child_level, cx, cy, cz);
        }

        return 0;
    }

    /**
     * @brief Subdivides the cell pointed to by `p_cell`.
     *
     * Convenience overload compatible with OctreeBinary::SubdivideCell().
     *
     * @param p_cell  Pointer to the cell to subdivide.
     * @return        0 on success, 1 if already subdivided.
     */
    int SubdivideCell(cell_type* p_cell)
    {
        return SubdivideCellByIdAndLevel(p_cell->GetId(), p_cell->GetLevel());
    }

    ///@}
    ///@name Leaf management
    ///@{

    /**
     * @brief Rebuilds `mLeafIds` by traversing the octree from the root.
     *
     * After a batch of subdivisions (e.g. during balancing), entries for
     * cells that were subsequently subdivided may remain in `mLeafIds`.
     * This method removes all such stale entries and resets `mLeafCount`.
     *
     * Uses a depth-first traversal from the root; a cell is added to
     * `mLeafIds` if and only if it is reachable (its parent is subdivided
     * or it is the root) and it is not itself subdivided.
     *
     * Complexity: O(leaf_count * average_path_length) — essentially O(N).
     */
    void RebuildLeafListClean()
    {
        mLeafIds.clear();

        // Start from root, BFS/DFS to collect leaves.
        std::vector<int> stack;
        stack.push_back(0); // root id

        while (!stack.empty()) {
            const int id    = stack.back();
            stack.pop_back();
            const int level = GetLevelFromId(id);

            if (!mNodeSubdivided[static_cast<std::size_t>(id)]) {
                // Leaf
                mLeafIds.push_back(id);
            } else {
                // Internal: push all 8 children
                for (int i = 0; i < 8; i++) {
                    stack.push_back(Child(id, level, i));
                }
            }
        }

        mLeafCount = static_cast<int>(mLeafIds.size());
    }

    /**
     * @brief Collects all current leaf cells into the provided vector.
     *
     * The vector is cleared before filling.  Cells are returned as raw
     * pointers into the internal `mCells` map; they remain valid as long as
     * the octree is not modified.
     *
     * @param all_leaves  Output vector that will hold one pointer per leaf.
     * @return            Always 0 (for API compatibility with OctreeBinary).
     */
    int GetAllLeavesVector(std::vector<cell_type*>& all_leaves) const
    {
        all_leaves.clear();
        all_leaves.reserve(static_cast<std::size_t>(mLeafCount));
        for (const int id : mLeafIds) {
            // mLeafIds may contain stale entries for cells that were
            // subsequently subdivided.  Only emit actual leaves.
            if (!IsLeaf(id)) continue;
            auto it = mCells.find(id);
            if (it != mCells.end() && it->second) {
                all_leaves.push_back(it->second.get());
            }
        }
        return 0;
    }

    /**
     * @brief Returns the number of current leaf cells.
     */
    int GetLeafCount() const noexcept { return mLeafCount; }

    /**
     * @brief Returns a const reference to the flat leaf-id list.
     */
    const std::vector<int>& GetLeafIds() const noexcept { return mLeafIds; }

    ///@}
    ///@name Cell retrieval
    ///@{

    /**
     * @brief Returns the leaf cell that contains the given normalised point.
     *
     * Starting from the root, the method descends through the octree using
     * the flat-id arithmetic until it reaches a leaf (a cell whose
     * mNodeSubdivided entry is false).
     *
     * @param point  Normalised coordinates in [0, 1]^3 (length DIMENSION).
     * @return       Pointer to the leaf cell, or nullptr if the point is
     *               outside [0, 1]^3 or not in any allocated cell.
     */
    cell_type* pGetCellNormalized(const coordinate_type* point) const
    {
        key_type keys[DIMENSION];
        CalcKeysNormalized(point, keys);
        return pGetCell(keys);
    }

    /**
     * @brief Returns the leaf cell that contains the point encoded by the keys.
     *
     * @param keys  Integer keys (length DIMENSION), each in [0, 2^mDepth).
     * @return      Pointer to the leaf cell, or nullptr.
     */
    cell_type* pGetCell(const key_type* keys) const
    {
        int id    = 0;
        int level = 0;

        while (!IsLeaf(id)) {
            level++;
            if (static_cast<std::size_t>(level) > mDepth) break;
            const int cx = static_cast<int>(keys[0] >> (mDepth - static_cast<std::size_t>(level)));
            const int cy = static_cast<int>(keys[1] >> (mDepth - static_cast<std::size_t>(level)));
            const int cz = static_cast<int>(keys[2] >> (mDepth - static_cast<std::size_t>(level)));
            id = XyzToOctreeIdx(level, cx, cy, cz);
        }

        auto it = mCells.find(id);
        return (it != mCells.end()) ? it->second.get() : nullptr;
    }

    /**
     * @brief Returns the leaf cell at the given level or the deepest available
     *        ancestor if that level has not been refined to.
     *
     * @param keys   Integer keys (length DIMENSION).
     * @param level  Target level to retrieve.
     * @return       Pointer to the leaf at or above `level`, or nullptr.
     */
    cell_type* pGetCell(const key_type* keys, std::size_t level) const
    {
        int id       = 0;
        int cur_level = 0;

        while (!IsLeaf(id) && static_cast<std::size_t>(cur_level) < level) {
            cur_level++;
            const int cx = static_cast<int>(keys[0] >> (mDepth - static_cast<std::size_t>(cur_level)));
            const int cy = static_cast<int>(keys[1] >> (mDepth - static_cast<std::size_t>(cur_level)));
            const int cz = static_cast<int>(keys[2] >> (mDepth - static_cast<std::size_t>(cur_level)));
            id = XyzToOctreeIdx(cur_level, cx, cy, cz);
        }

        auto it = mCells.find(id);
        return (it != mCells.end()) ? it->second.get() : nullptr;
    }

    /**
     * @brief Collects all leaf cells whose bounding boxes intersect the given
     *        normalised axis-aligned box.
     *
     * Uses a depth-first traversal starting from the smallest ancestor of the
     * query box.
     *
     * @param coord1  Minimum corner of the query box (normalised, length DIMENSION).
     * @param coord2  Maximum corner of the query box (normalised, length DIMENSION).
     * @param leaves  Output vector of matching leaf cells.
     */
    void GetLeavesInBoundingBoxNormalized(
        const double* coord1,
        const double* coord2,
        std::vector<cell_type*>& leaves) const
    {
        leaves.clear();

        // DFS from root.
        std::vector<int> stack;
        stack.push_back(0);

        while (!stack.empty()) {
            const int id    = stack.back();
            stack.pop_back();
            const int level = GetLevelFromId(id);

            int x, y, z;
            OctreeIdxToXyz(id, level, x, y, z);
            const double cell_size = CalcSizeNormalizedAtLevel(static_cast<std::size_t>(level));
            const double lo[3] = { x * cell_size, y * cell_size, z * cell_size };
            const double hi[3] = { (x+1)*cell_size, (y+1)*cell_size, (z+1)*cell_size };

            if (!Collides(coord1, coord2, lo, hi)) continue;

            if (IsLeaf(id)) {
                auto it = mCells.find(id);
                if (it != mCells.end()) leaves.push_back(it->second.get());
            } else {
                for (int i = 0; i < 8; i++) {
                    stack.push_back(Child(id, level, i));
                }
            }
        }
    }

    ///@}
    ///@name 2:1 balance enforcement
    ///@{

    /**
     * @brief Enforces the 2:1 balance condition on the entire octree.
     *
     * Two adjacent leaf cells satisfy the 2:1 balance condition when their
     * refinement levels differ by at most 1.  If any pair of face-adjacent
     * leaves violates this condition, the coarser cell (and all its siblings)
     * is subdivided.  The process repeats until the octree is fully balanced.
     *
     * ## Algorithm
     *
     * For each leaf L at level level_L:
     *   For each of the 6 face directions:
     *     - Compute the face-neighbour flat id at level_L.
     *     - Find the actual leaf that contains that position (may be coarser).
     *     - If the neighbour leaf is at a level less than level_L - 1:
     *       - Subdivide the neighbour and all its 7 siblings.
     *       - Set a flag to re-check.
     *
     * This is repeated (convergence loop) until a full pass over all leaves
     * produces no new subdivisions.
     *
     * @note After this method returns, call RebuildLeafListClean() if you need
     *       a consistent leaf list (the method calls it internally).
     */
    void Constrain2To1()
    {
        bool changed = true;
        while (changed) {
            changed = false;

            // Snapshot of current leaf ids to iterate over.
            const std::vector<int> snapshot(mLeafIds);

            for (const int leaf_id : snapshot) {
                if (!IsLeaf(leaf_id)) continue; // may have been subdivided

                const int level = GetLevelFromId(leaf_id);
                if (level == 0) continue; // root has no face neighbours outside the box

                int lx, ly, lz;
                OctreeIdxToXyz(leaf_id, level, lx, ly, lz);
                const int res = static_cast<int>(mLevelRes[static_cast<std::size_t>(level)]);

                // Examine all 6 face directions.
                // Offsets: ±X, ±Y, ±Z
                const int dx[6] = {-1,  1,  0,  0,  0,  0};
                const int dy[6] = { 0,  0, -1,  1,  0,  0};
                const int dz[6] = { 0,  0,  0,  0, -1,  1};

                for (int dir = 0; dir < 6; dir++) {
                    const int nx = lx + dx[dir];
                    const int ny = ly + dy[dir];
                    const int nz = lz + dz[dir];

                    // Boundary check: skip if outside the [0, res-1]^3 grid.
                    if (nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res)
                        continue;

                    // Convert neighbour position (at current level) to a key.
                    // We probe the centre of the neighbour cell at level `level`.
                    const double cell_size = CalcSizeNormalizedAtLevel(static_cast<std::size_t>(level));
                    const double probe[3] = {
                        (nx + 0.5) * cell_size,
                        (ny + 0.5) * cell_size,
                        (nz + 0.5) * cell_size
                    };
                    key_type probe_keys[3];
                    CalcKeysNormalized(probe, probe_keys);

                    const cell_type* neighbour = pGetCell(probe_keys);
                    if (neighbour == nullptr) continue;

                    const int neighbour_level = neighbour->GetLevel();

                    // The balance condition requires |level - neighbour_level| <= 1.
                    // If the neighbour is coarser by more than 1 level, subdivide it.
                    if (level - neighbour_level > 1) {
                        // Subdivide the neighbour and all its siblings.
                        int siblings[8];
                        RefineBrothers(neighbour->GetId(), neighbour_level, siblings);
                        for (int i = 0; i < 8; i++) {
                            if (siblings[i] >= 0 && IsLeaf(siblings[i])) {
                                const int sib_level = GetLevelFromId(siblings[i]);
                                SubdivideCellByIdAndLevel(siblings[i], sib_level);
                            }
                        }
                        changed = true;
                    }
                }
            }

            // Synchronise the leaf list after each pass.
            RebuildLeafListClean();
        }
    }

    /**
     * @brief Strong-balance version of Constrain2To1.
     *
     * Port of StrongBalancedOctree() from the reference HexGen.cpp.
     *
     * In addition to face-neighbours (6 directions), this also checks
     * edge-neighbours (12 directions) and corner-neighbours (8 directions),
     * ensuring no two leaves at any shared topological feature differ by
     * more than one level.  This stricter condition is required for the
     * dual hex mesh extraction templates.
     *
     * The algorithm:
     *  1. Build a mapping from each octree grid vertex (corner of a cell) to
     *     the set of leaf cells sharing that corner.
     *  2. For each corner, find the maximum level among its incident leaves.
     *  3. Subdivide any leaf whose level is less than max_level - 1.
     *  4. Repeat until convergence.
     */
    void StrongConstrain2To1()
    {
        bool changed = true;
        while (changed) {
            changed = false;

            // Collect all corners of all leaf cells and record which leaves touch each corner.
            // Corner key = (cx, cy, cz) at the resolution of the FINEST current leaf.
            // We represent it at MAX depth resolution to avoid level-dependent indexing.
            std::unordered_map<long long, std::vector<int>> corner_to_leaves;

            for (const int leaf_id : mLeafIds) {
                if (!IsLeaf(leaf_id)) continue;

                const int level = GetLevelFromId(leaf_id);
                const int step  = static_cast<int>(mLevelRes[mDepth]) / static_cast<int>(mLevelRes[static_cast<std::size_t>(level)]);
                int lx, ly, lz;
                OctreeIdxToXyz(leaf_id, level, lx, ly, lz);

                // The 8 corners of this cell at MAX_DEPTH resolution.
                for (int ci = 0; ci < 8; ci++) {
                    const int cx = lx * step + (ci & 1) * step;
                    const int cy = ly * step + ((ci >> 1) & 1) * step;
                    const int cz = lz * step + ((ci >> 2) & 1) * step;
                    const long long key =
                        static_cast<long long>(cx)
                        + static_cast<long long>(cy) * (static_cast<long long>(mLevelRes[mDepth]) + 1)
                        + static_cast<long long>(cz) * (static_cast<long long>(mLevelRes[mDepth]) + 1)
                                                      * (static_cast<long long>(mLevelRes[mDepth]) + 1);
                    corner_to_leaves[key].push_back(leaf_id);
                }
            }

            // For each corner, find max level; subdivide leaves with level < max - 1.
            for (auto& [key, incident_leaves] : corner_to_leaves) {
                int max_level = 0;
                for (const int lid : incident_leaves) {
                    if (IsLeaf(lid)) {
                        max_level = std::max(max_level, GetLevelFromId(lid));
                    }
                }

                for (const int lid : incident_leaves) {
                    if (!IsLeaf(lid)) continue;
                    const int leaf_level = GetLevelFromId(lid);
                    if (leaf_level < max_level - 1) {
                        int siblings[8];
                        RefineBrothers(lid, leaf_level, siblings);
                        for (int i = 0; i < 8; i++) {
                            if (siblings[i] >= 0 && IsLeaf(siblings[i])) {
                                const int sib_level = GetLevelFromId(siblings[i]);
                                SubdivideCellByIdAndLevel(siblings[i], sib_level);
                            }
                        }
                        changed = true;
                    }
                }
            }

            if (changed) {
                RebuildLeafListClean();
            }
        }
    }

    ///@}
    ///@name Geometry utilities
    ///@{

    /**
     * @brief Returns true if two axis-aligned boxes overlap.
     *
     * @param Low1, High1  First box corners.
     * @param Low2, High2  Second box corners.
     */
    bool Collides(
        const double* Low1, const double* High1,
        const double* Low2, const double* High2) const
    {
        return Low1[0] <= High2[0] && Low2[0] <= High1[0]
            && Low1[1] <= High2[1] && Low2[1] <= High1[1]
            && Low1[2] <= High2[2] && Low2[2] <= High1[2];
    }

    ///@}
    ///@name Accessors
    ///@{

    /**
     * @brief Returns the maximum refinement depth of this octree.
     */
    std::size_t GetDepth() const noexcept { return mDepth; }

    /**
     * @brief Returns the grid resolution at the given level (2^level).
     */
    std::size_t GetLevelRes(std::size_t level) const { return mLevelRes[level]; }

    /**
     * @brief Returns the start flat index of the given level in mNodeSubdivided.
     */
    std::size_t GetLevelId(std::size_t level) const { return mLevelId[level]; }

    /**
     * @brief Returns the total number of nodes in the complete octree
     *        (the size of mNodeSubdivided).
     */
    std::size_t GetTotalNodeCount() const noexcept { return mNodeSubdivided.size(); }

    /**
     * @brief Returns the const flat subdivision bit-array.
     */
    const std::vector<bool>& GetNodeSubdivided() const noexcept { return mNodeSubdivided; }

    ///@}
    ///@name Input and output
    ///@{

    /**
     * @brief Returns a human-readable label for this class.
     */
    std::string Info() const
    {
        return "OctreeHybrid";
    }

    /**
     * @brief Prints a summary of octree parameters to the output stream.
     *
     * @param rOStream  Destination stream.
     */
    void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << Info()
                 << " [depth=" << mDepth
                 << ", total_nodes=" << mNodeSubdivided.size()
                 << ", leaf_count=" << mLeafCount
                 << "]";
    }

    /**
     * @brief Prints the full tree structure (leaf cells only) to the output stream.
     *
     * @param rOStream  Destination stream.
     */
    void PrintData(std::ostream& rOStream) const
    {
        rOStream << "Leaf cells:\n";
        for (const int id : mLeafIds) {
            auto it = mCells.find(id);
            if (it != mCells.end()) {
                it->second->PrintInfo(rOStream);
                it->second->PrintData(rOStream);
            }
        }
    }

    ///@}

private:
    ///@name Private member variables
    ///@{

    std::size_t mDepth;      ///< Maximum refinement depth (0 = unused, MAX_DEPTH = finest).

    /// mLevelRes[L] = 2^L: grid cells per axis at level L.
    std::vector<std::size_t> mLevelRes;
    /// mLevelId[L] = (8^L - 1)/7: start index of level L in mNodeSubdivided.
    std::vector<std::size_t> mLevelId;

    /// Flat subdivision bit-array.  mNodeSubdivided[id] == true means subdivided.
    /// Size = mLevelId[mDepth + 1] = total nodes in the complete octree to depth mDepth.
    std::vector<bool> mNodeSubdivided;

    /// IDs of all current leaf nodes (not yet subdivided).
    std::vector<int> mLeafIds;
    /// Cached count of leaf nodes.
    int mLeafCount;

    /// Cell objects keyed by flat octree id.  Only cells that have been accessed
    /// (via subdivision or explicit construction) appear here.
    std::unordered_map<int, std::unique_ptr<cell_type>> mCells;

    /// Coordinate normalisation: world → [0,1] via  norm_i = (coord_i + offset_i) * scale_i
    double mScaleFactor[DIMENSION];
    double mOffset[DIMENSION];

    ///@}
};

///@name Stream output operator
///@{

/**
 * @brief Stream insertion operator for OctreeHybrid.
 * @relates OctreeHybrid
 */
template<class TCellType>
inline std::ostream& operator<<(std::ostream& rOStream, const OctreeHybrid<TCellType>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << "\n";
    rThis.PrintData(rOStream);
    return rOStream;
}

///@}
