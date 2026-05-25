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
#include <cstddef>
#include <cassert>

/**
 * @file octree_hybrid_cell.h
 * @brief Defines OctreeHybridCell, the node type for OctreeHybrid.
 *
 * ## Overview
 *
 * OctreeHybridCell represents a single axis-aligned box ("voxel") inside a
 * linearised complete octree.  Unlike OctreeBinaryCell, which chains cells
 * through heap-allocated child arrays, OctreeHybridCell is **coordinate-
 * address-based**: it stores the level and integer grid position (x, y, z) of
 * the voxel rather than child pointers.  The actual subdivision state lives in
 * the flat bit-vector of the owning OctreeHybrid; the cell carries only a
 * cached `mIsLeaf` flag that the octree keeps up to date.
 *
 * ## Level convention
 *
 * Level numbers follow the reference HybridOctree_Hex code:
 *
 * | level | meaning              | cells per axis | cell size (normalised) |
 * |-------|----------------------|----------------|------------------------|
 * |  0    | root (coarsest)      | 1              | 1.0                    |
 * |  1    | first refinement     | 2              | 0.5                    |
 * |  L    | refinement level L   | 2^L            | 2^(-L)                 |
 * | MAX_DEPTH | finest leaves    | 2^MAX_DEPTH    | 2^(-MAX_DEPTH)         |
 *
 * This is the **inverse** of OctreeBinaryCell, where ROOT_LEVEL is the coarsest
 * and MIN_LEVEL is the finest.  The two cannot be used interchangeably, but
 * the public interface of OctreeHybrid deliberately mirrors OctreeBinary's
 * interface so that client code changes are minimal.
 *
 * ## Corner numbering
 *
 * All eight corners of the cell are numbered identically to OctreeBinaryCell
 * and the HybridOctree_Hex paper:
 *
 * @code
 *          7 ---- 6
 *         /|      /|
 *        4 ---- 5  |
 *        |  3 --|- 2
 *        | /    | /
 *        0 ---- 1
 * @endcode
 *
 * Corner 0 is at (x_min, y_min, z_min) and corner 6 at (x_max, y_max, z_max).
 *
 * ## Face index convention
 *
 * | face index | description | corners       |
 * |------------|-------------|---------------|
 * | 0          | bottom      | {0, 1, 2, 3}  |
 * | 1          | front       | {0, 1, 5, 4}  |
 * | 2          | left        | {0, 3, 7, 4}  |
 * | 3          | right       | {1, 2, 6, 5}  |
 * | 4          | back        | {3, 2, 6, 7}  |
 * | 5          | top         | {4, 5, 6, 7}  |
 *
 * @tparam TConfiguration  Configuration policy class.  Must provide:
 *   - `data_type`      : user payload stored per leaf cell
 *   - `pointer_type`   : pointer type used for inserted geometric objects
 *   - `MAX_DEPTH`      : compile-time maximum refinement depth (std::size_t)
 *   - `MIN_DEPTH`      : compile-time minimum refinement depth (std::size_t)
 *   - `DIMENSION`      : must be 3 (std::size_t)
 *   - `static void DeleteData(data_type*)` : frees a data_type pointer
 *   - `static bool IsIntersected(pointer_type, double tolerance,
 *                                const double* box_min, const double* box_max)`
 *     : returns true when the object intersects the axis-aligned box
 */
template<class TConfiguration>
class OctreeHybridCell
{
public:
    ///@name Type Definitions
    ///@{

    using data_type             = typename TConfiguration::data_type;
    using configuration_type    = TConfiguration;
    using pointer_type          = typename TConfiguration::pointer_type;
    using object_container_type = std::vector<pointer_type>;
    /// Integer type used for grid coordinates and flat-array keys.
    using key_type              = std::size_t;

    static constexpr std::size_t CHILDREN_NUMBER = 8; ///< Always 8 for an octree.
    static constexpr std::size_t DIMENSION       = TConfiguration::DIMENSION;
    /// Maximum refinement depth (level of the finest possible leaf).
    static constexpr std::size_t MAX_DEPTH       = TConfiguration::MAX_DEPTH;
    /// Minimum depth at which leaves are allowed (refinement stops here).
    static constexpr std::size_t MIN_DEPTH       = TConfiguration::MIN_DEPTH;

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Constructs an OctreeHybridCell with explicit coordinates.
     *
     * @param octree_id  Flat index of this cell in the owning OctreeHybrid's
     *                   subdivision bit-array.
     * @param level      Refinement level (0 = root, MAX_DEPTH = finest).
     * @param grid_x     Integer X coordinate at this level (0 to 2^level - 1).
     * @param grid_y     Integer Y coordinate at this level.
     * @param grid_z     Integer Z coordinate at this level.
     */
    explicit OctreeHybridCell(
        int octree_id = 0,
        int level     = 0,
        int grid_x    = 0,
        int grid_y    = 0,
        int grid_z    = 0
    )
        : mId(octree_id)
        , mLevel(level)
        , mGridX(grid_x)
        , mGridY(grid_y)
        , mGridZ(grid_z)
        , mIsLeaf(true)
        , mData(nullptr)
    {}

    /// Destructor — frees user data if present.
    ~OctreeHybridCell()
    {
        if (mData) {
            configuration_type::DeleteData(mData);
        }
    }

    // Non-copyable (owns mData and mObjects).
    OctreeHybridCell(const OctreeHybridCell&)            = delete;
    OctreeHybridCell& operator=(const OctreeHybridCell&) = delete;

    // Movable (for insertion into containers).
    OctreeHybridCell(OctreeHybridCell&&) noexcept            = default;
    OctreeHybridCell& operator=(OctreeHybridCell&&) noexcept = default;

    ///@}
    ///@name Identification
    ///@{

    /**
     * @brief Returns the flat index of this cell in the parent octree's bit-array.
     *
     * The flat index is computed as:
     * @code
     *   id = levelId[level] + mGridZ * res^2 + mGridY * res + mGridX
     *   where res = 2^level
     * @endcode
     */
    int GetId() const noexcept { return mId; }

    /**
     * @brief Returns the refinement level (0 = coarsest root, MAX_DEPTH = finest).
     */
    int GetLevel() const noexcept { return mLevel; }

    ///@}
    ///@name Grid coordinates
    ///@{

    /**
     * @brief Returns the integer X grid coordinate at this cell's level.
     *
     * The coordinate is in the range [0, 2^level - 1].
     */
    int GetGridX() const noexcept { return mGridX; }

    /**
     * @brief Returns the integer Y grid coordinate at this cell's level.
     */
    int GetGridY() const noexcept { return mGridY; }

    /**
     * @brief Returns the integer Z grid coordinate at this cell's level.
     */
    int GetGridZ() const noexcept { return mGridZ; }

    ///@}
    ///@name Normalised geometry
    ///@{

    /**
     * @brief Writes the minimum corner of this cell in normalised [0, 1]^3 coordinates.
     *
     * With res = 2^level and cell_size = 1.0 / res:
     *   min_point[i] = grid_coord[i] * cell_size
     *
     * @param min_point  Output array of length 3.
     */
    void GetMinPointNormalized(double* min_point) const
    {
        const double cell_size = CalcSizeNormalized();
        min_point[0] = mGridX * cell_size;
        min_point[1] = mGridY * cell_size;
        min_point[2] = mGridZ * cell_size;
    }

    /**
     * @brief Writes the maximum corner of this cell in normalised [0, 1]^3 coordinates.
     *
     * max_point[i] = (grid_coord[i] + 1) * cell_size
     *
     * @param max_point  Output array of length 3.
     */
    void GetMaxPointNormalized(double* max_point) const
    {
        const double cell_size = CalcSizeNormalized();
        max_point[0] = (mGridX + 1) * cell_size;
        max_point[1] = (mGridY + 1) * cell_size;
        max_point[2] = (mGridZ + 1) * cell_size;
    }

    /**
     * @brief Returns the normalised side length of this cell.
     *
     * @return  1.0 / 2^level  (1.0 at the root, smallest at MAX_DEPTH).
     */
    double CalcSizeNormalized() const
    {
        // 1.0 / (1 << level): shift is safe for level <= 30 on 64-bit.
        return 1.0 / static_cast<double>(1 << mLevel);
    }

    ///@}
    ///@name Leaf state
    ///@{

    /**
     * @brief Returns true if this cell is currently a leaf (not subdivided).
     *
     * This flag is maintained by the owning OctreeHybrid:  whenever the
     * cell's entry in `mNodeSubdivided` changes, the octree calls
     * SetIsLeaf() on the cached cell.
     */
    bool IsLeaf() const noexcept { return mIsLeaf; }

    /**
     * @brief Sets the cached leaf flag.
     *
     * Called exclusively by the owning OctreeHybrid when a subdivision
     * event occurs.
     *
     * @param is_leaf  true if the cell should be treated as a leaf.
     */
    void SetIsLeaf(bool is_leaf) noexcept { mIsLeaf = is_leaf; }

    ///@}
    ///@name User data
    ///@{

    /**
     * @brief Returns a pointer to the user data attached to this cell.
     *
     * Returns nullptr if no data has been set.
     */
    data_type* pGetData() const noexcept { return mData; }

    /**
     * @brief Returns a mutable double-pointer to the user data.
     *
     * Allows the caller to assign new user data in place.
     */
    data_type** pGetDataPointer() noexcept { return &mData; }

    ///@}
    ///@name Object storage
    ///@{

    /**
     * @brief Inserts a geometric object into this cell's storage.
     *
     * Objects are typically stored only at leaf cells.  The OctreeHybrid
     * transfers them downward when a cell is subdivided via
     * TransferObjectsToChildren().
     *
     * @param object  Pointer to the object to store.
     */
    void Insert(pointer_type object)
    {
        mObjects.push_back(object);
    }

    /**
     * @brief Returns a const pointer to the container of stored objects.
     */
    const object_container_type* pGetObjects() const noexcept { return &mObjects; }

    /**
     * @brief Returns a mutable pointer to the container of stored objects.
     */
    object_container_type* pGetObjects() noexcept { return &mObjects; }

    /**
     * @brief Clears the stored objects without freeing them.
     *
     * Ownership of the object pointers remains with the caller.
     */
    void EmptyObjects()
    {
        object_container_type tmp;
        tmp.swap(mObjects);
    }

    /**
     * @brief Transfers stored objects to the appropriate child cells.
     *
     * For each object in this cell, the method tests intersection with each
     * child's bounding box (using TConfiguration::IsIntersected) and inserts
     * the object into every matching child.  After the transfer, this cell's
     * object list is cleared.
     *
     * This method must be called by OctreeHybrid after subdivision to
     * propagate objects from parent to children.
     *
     * @param children   Array of exactly CHILDREN_NUMBER child cells, ordered
     *                   with child i at grid offset:
     *                     dx = i & 1,  dy = (i >> 1) & 1,  dz = (i >> 2) & 1
     */
    void TransferObjectsToChildren(OctreeHybridCell* children)
    {
        if (mObjects.empty()) return;

        // Small tolerance: 0.1% of the cell size to avoid boundary misses.
        const double tolerance = 0.001 * CalcSizeNormalized();
        double child_min[DIMENSION], child_max[DIMENSION];

        for (std::size_t i = 0; i < CHILDREN_NUMBER; i++) {
            OctreeHybridCell& child = children[i];
            child.GetMinPointNormalized(child_min);
            child.GetMaxPointNormalized(child_max);

            for (const auto& obj : mObjects) {
                if (configuration_type::IsIntersected(obj, tolerance, child_min, child_max)) {
                    child.Insert(obj);
                }
            }
        }

        EmptyObjects();
    }

    ///@}
    ///@name Input and output
    ///@{

    /**
     * @brief Returns a human-readable label for this class.
     */
    std::string Info() const
    {
        return "OctreeHybridCell";
    }

    /**
     * @brief Prints level, flat id and grid position to the output stream.
     *
     * Indentation matches the refinement depth (two spaces per level step).
     *
     * @param rOStream  Destination stream.
     */
    void PrintInfo(std::ostream& rOStream) const
    {
        for (int i = 0; i < mLevel; i++) rOStream << "  ";
        rOStream << Info()
                 << " [id=" << mId
                 << ", level=" << mLevel
                 << ", pos=(" << mGridX << "," << mGridY << "," << mGridZ << ")]";
    }

    /**
     * @brief Prints the normalised bounding box and leaf/internal status.
     *
     * @param rOStream  Destination stream.
     */
    void PrintData(std::ostream& rOStream) const
    {
        double lo[3], hi[3];
        GetMinPointNormalized(lo);
        GetMaxPointNormalized(hi);
        rOStream << " [("
                 << lo[0] << "," << lo[1] << "," << lo[2] << ") -> ("
                 << hi[0] << "," << hi[1] << "," << hi[2] << ")] "
                 << (mIsLeaf ? "LEAF" : "INTERNAL")
                 << " objects=" << mObjects.size()
                 << "\n";
    }

    ///@}

private:
    ///@name Private member variables
    ///@{

    int  mId;     ///< Flat index in the owning OctreeHybrid's subdivision bit-array.
    int  mLevel;  ///< Refinement level (0 = root, MAX_DEPTH = finest).
    int  mGridX;  ///< Integer grid X coordinate at level `mLevel` (in [0, 2^mLevel - 1]).
    int  mGridY;  ///< Integer grid Y coordinate at level `mLevel`.
    int  mGridZ;  ///< Integer grid Z coordinate at level `mLevel`.
    bool mIsLeaf; ///< Cached leaf flag; true when mNodeSubdivided[mId] == false.

    data_type*            mData;    ///< Optional user payload; owned by this cell.
    object_container_type mObjects; ///< Geometric objects whose AABB intersects this cell.

    ///@}
};

///@name Stream output operator
///@{

/**
 * @brief Stream insertion operator for OctreeHybridCell.
 * @relates OctreeHybridCell
 */
template<class TConfiguration>
inline std::ostream& operator<<(
    std::ostream& rOStream,
    const OctreeHybridCell<TConfiguration>& rThis)
{
    rThis.PrintInfo(rOStream);
    rThis.PrintData(rOStream);
    return rOStream;
}

///@}
