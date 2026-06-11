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
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Project includes
#include "geometries/bounding_box.h"
#include "geometries/point.h"
#include "spatial_containers/octree_hybrid.h"
#include "spatial_containers/octree_hybrid_cell.h"
#include "spatial_containers/octree_hybrid_configure.h"

namespace Kratos {

// Forward declarations
class ModelPart;

/**
 * @brief Utility that builds an adaptive OctreeHybrid from a surface ModelPart
 *        and exports a conforming all-hexahedral mesh (the "hybrid mesh").
 *
 * ### Algorithm overview
 *
 * The mesh exported by WriteDualHexVtk is the **dual** of the strongly-balanced
 * octree (the "HybridOctree_Hex" algorithm, Gao et al.):
 *
 *  1. Apply StrongConstrain2To1 → no two adjacent leaves differ by more than 1 level.
 *  2. Build the **primal** mesh: the leaf cells are hexahedra sharing vertices
 *     at their corners.  Each primal vertex is keyed by its integer grid
 *     coordinates (at the finest level), so shared corners are merged.
 *  3. For every primal vertex touched by exactly 8 leaf cells, create one
 *     **dual hex** whose 8 nodes are the centres of those 8 cells.
 *     The node ordering follows idTransform[8] = {6,7,4,5,2,3,0,1} from
 *     the reference, which maps each cell's contributing corner to the
 *     correct VTK HEXAHEDRON position.
 *
 * In a locally-uniform region (all 8 cells the same size) the dual hex is a
 * perfect cuboid.  In transition regions (cells of two different sizes sharing
 * a vertex) the dual hex is distorted but valid, covering the space exactly —
 * the mesh is conforming with no hanging nodes.
 *
 * ### Transition templates
 *
 * For every face where one large cell meets four smaller cells, the utility
 * applies the transition templates from the HybridOctree_Hex paper, anchored at
 * the large cell whose in-plane grid indices are both even (the reference's
 * `stepI && stepJ` gate; the parity must be evaluated in grid-index space, not
 * in world coordinates which are offset by the bounding-box minimum):
 *
 *  - the 13-element base template that fills the face transition itself, and
 *  - the 4/3/5-element edge/corner sub-templates (t2/t22, t3/t32/t33/t34,
 *    t4/t42/t43/t44) that stitch adjacent transition regions together.
 *
 * Each template's bookkeeping mirrors the reference: the 13-element and
 * 4-element templates always emit; a 3-element template emits only if the
 * transition vertex it would replace is still unclaimed; a 5-element template
 * emits only if it introduces at least one new node.  A consumed vertex's plain
 * dual hex is then skipped (see `consume_at`).
 *
 * ### Node merging
 *
 * All hexes - plain dual and template - index into a single merged node array
 * (`nodes`): it is seeded with the cell centres (the dual nodes), and every
 * template point is looked up in a spatial hash so that a point coinciding with
 * an existing node reuses its id.  Template and plain-dual hexes therefore share
 * the same node at every interface where they geometrically meet, instead of
 * carrying coincident duplicates.
 *
 * @note All generated elements are valid (positive Jacobian) — verified by
 *       test_octree_hybrid_dual_mesh.py.  The element tiling matches the
 *       HybridOctree_Hex reference **cell-for-cell**: instrumented against the
 *       reference's own `DualFullHexMeshExtraction` output on the 2:1-transition
 *       test surface, the two meshes are identical (same hex set, zero gaps and
 *       zero overlaps) at refinement depths 3 through 7.  Two bugs that broke
 *       this were fixed: (a) the face-adjacency "fill opposite side" shortcut
 *       stamped valence-1 onto a coarse cell's transition face when a fine
 *       neighbour saw it first, hiding half the transitions (every `j=5`/high
 *       face) so only half the templates fired; (b) the 13-element base
 *       template's collectNum erase point used `z*4/15` instead of the
 *       reference's `0.5*(p21+p26+z*4/15)` = `z*2/15`, which mis-rounded and
 *       left the centre dual hex unconsumed once cells were large in finest-grid
 *       units (surfaced at depth >= 6).
 *
 *       This `DualFullHex` mesh is the reference's *intermediate* output: it is
 *       conforming in the node-sharing sense but is not by itself a closed
 *       2-manifold — it carries a small number of T-junctions and template
 *       overlaps at the refinement interface (e.g. depth 4: 2 overlapping faces,
 *       827 open boundary edges).  The reference yields the **identical** counts,
 *       so these are properties of the algorithm stage, not of this port; the
 *       reference removes them only in its later RemoveOutsideElement /
 *       ProjectToIsoSurface passes, which are not (yet) ported here.
 *
 * ### Usage
 * ```python
 * KM.OctreeHybridMeshUtility.BuildAndWriteVtk(surface_mp, "mesh.vtk", 8)
 * ```
 */
class OctreeHybridMeshUtility
{
public:
    ///@name Type Definitions
    ///@{

    using ConfigurationType = OctreeHybridKratosConfiguration;
    using CellType          = OctreeHybridCell<ConfigurationType>;
    using OctreeType        = OctreeHybrid<CellType>;
    using GeometryType      = Geometry<Node>;

    /// A flat list of surface triangles in world coordinates: one entry per
    /// triangle, each holding its 3 vertices (3 doubles each).  Used by the
    /// carving stage (RemoveOutsideElement) to test hex vertices against the
    /// input surface.
    using TriangleSoup = std::vector<std::array<std::array<double,3>,3>>;

    /// A hanging-node multipoint constraint at a 2:1 transition of the primal
    /// (leaf-hex) mesh: the slave node lies on a coarse hex face and is the
    /// bilinear interpolation of that face's master corners.  Node fields are
    /// indices into the primal node array; @ref NumMasters slots of
    /// @ref MasterNodes / @ref Weights are used (2 for an edge-midpoint slave,
    /// 4 for a face-centre slave) and the weights sum to one.
    struct HangingConstraint {
        int SlaveNode = -1;
        std::array<int, 4> MasterNodes{ -1, -1, -1, -1 };
        std::array<double, 4> Weights{ 0.0, 0.0, 0.0, 0.0 };
        int NumMasters = 0;
    };

    ///@}
    ///@name Static operations
    ///@{

    /**
     * @brief Builds an OctreeHybrid from a surface mesh, with optional adaptive refinement.
     *
     * @details Sets the bounding box from the surface geometry, inserts all surface
     *          triangle vertices into the octree, and refines the octree to
     *          @p RefinementDepth.  When @p Adaptive is `true`, the reference
     *          HybridOctree_Hex curvature + feature-thickness criterion is used
     *          (see @ref BuildAdaptiveFromSurfaceMesh); when `false`, every cell
     *          containing a surface vertex is uniformly subdivided to
     *          @p RefinementDepth via @ref RefineInterfaceCells.
     *          When @p pOverrideBoundingBox is non-null, its min/max points are used as the
     *          octree's world-space domain instead of the geometry-derived extents (no
     *          1% auto-padding is applied in the non-adaptive path; in the adaptive path
     *          the centred reference cube is derived from this box instead of the
     *          triangle-corner extents).
     *
     * @param rSurfaceMesh     ModelPart whose Geometries() container holds the
     *                         surface triangles (populated by StlIO::ReadModelPart).
     * @param RefinementDepth  Maximum refinement depth near the surface.
     *                         Must be in [1, OctreeHybridKratosConfiguration::MAX_DEPTH].
     * @param Adaptive         When `true` (default) use the curvature + thickness
     *                         criterion from the reference code so that leaf counts
     *                         match the reference for real geometries.  When `false`
     *                         use simple interface-cell refinement (faster but less
     *                         resolution near high-curvature features).
     * @param pOverrideBoundingBox  Optional world-space domain override (see @details).
     * @return Unique pointer to the built (but not yet 2:1-balanced) octree.
     */
    static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth,
        bool Adaptive = true,
        const BoundingBox<Point>* pOverrideBoundingBox = nullptr);

    /**
     * @brief Builds an OctreeHybrid using the reference HybridOctree_Hex **adaptive**
     *        refinement criterion (curvature + feature thickness).
     *
     * @details The criterion maps surface triangles to absolute octree levels based on
     *          two signals computed in a 100-unit-normalised coordinate system:
     *          - **Curvature** (`C_THRES = {0, 0, 0.4, 0.8, 1.6}`): sum of squared
     *            angle-defect at each triangle's vertices.  Triangles with high
     *            curvature are mapped to finer levels.
     *          - **Feature thickness** (`H_THRES = {16, 8, 4, 2, 1}`): ray-cast
     *            distance to the opposite surface.  Thin features are mapped to
     *            finer levels.
     *
     *          In addition, for each selected triangle all 8 siblings of any cell
     *          that cell belongs to are subdivided together (complete-octet
     *          propagation), matching the reference's convergence behaviour.
     *
     *          Use this overload when the generated leaf count must match the reference
     *          exactly (e.g. for validation).  For general use @ref BuildFromSurfaceMesh
     *          with `Adaptive = true` delegates here.
     *
     * @param rSurfaceMesh     ModelPart whose Geometries() container holds the triangles.
     * @param RefinementDepth  Maximum allowed refinement depth (clamped to
     *                         `OctreeHybridKratosConfiguration::MAX_DEPTH`).
     * @param pOverrideBoundingBox  Optional world-space domain override: when non-null, its
     *                              min/max points replace the triangle-corner extents used to
     *                              derive the centred reference cube (`cube_lo`/`cube_side`).
     * @return Unique pointer to the built (but not yet 2:1-balanced) octree.
     */
    static std::unique_ptr<OctreeType> BuildAdaptiveFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth,
        const BoundingBox<Point>* pOverrideBoundingBox = nullptr);

    /**
     * @brief Extracts the **dual hex mesh** (the proper HybridOctree_Hex output)
     *        into in-memory node/cell arrays.
     *
     * The octree must already be strongly 2:1-balanced (call
     * @ref OctreeHybrid::StrongConstrain2To1 first).
     *
     * @param rOctree     Octree (bounding box must be set, already 2:1-balanced).
     * @param rNodes      [out] Dual-hex node coordinates (world space).
     * @param rCells      [out] Hexahedra connectivity (indices into @p rNodes).
     * @param rCellLevel  [out] Per-cell refinement level (level of the leaf whose
     *                    centre is node 0 for plain-dual hexes; -1 for template
     *                    hexes), useful for colouring.
     */
    static void ExtractDualHexMesh(
        OctreeType& rOctree,
        std::vector<std::array<double,3>>& rNodes,
        std::vector<std::array<int,8>>& rCells,
        std::vector<int>& rCellLevel);

    /**
     * @brief Collects the surface triangles of @p rSurfaceMesh as a world-space
     *        @ref TriangleSoup.
     * @details Iterates `rSurfaceMesh.Geometries()` and copies the world-space
     *          coordinates of every triangle's three vertices into the returned
     *          soup.  Geometries with fewer than 3 points are silently skipped.
     * @param rSurfaceMesh  Read-only surface ModelPart (from StlIO::ReadModelPart).
     * @return              World-space triangle soup; one entry per triangle.
     */
    static TriangleSoup ExtractTriangleSoup(const ModelPart& rSurfaceMesh);

    /**
     * @brief Extracts the **primal (leaf-hex) mesh** plus the hanging-node
     *        master-slave constraints at 2:1 transitions.
     *
     * The primal mesh has one hexahedron per octree leaf, with shared
     * finest-grid corner nodes.  It is **non-conforming** at every 2:1
     * interface where a coarse face touches four finer faces.  The hanging
     * nodes that lie on those coarse faces are tied to the face's four
     * master corners by bilinear weights (0.5/0.5 for edge-midpoint slaves,
     * 0.25×4 for face-centre slaves) and reported in @p rHanging.
     *
     * The octree must already be 2:1-balanced (call
     * @ref OctreeHybrid::StrongConstrain2To1 first).
     *
     * @param rOctree    Octree (bounding box set, already 2:1-balanced).
     * @param rNodes     [out] Primal node coordinates (world space).
     * @param rCells     [out] Hexahedra connectivity (indices into @p rNodes).
     * @param rCellLevel [out] Per-cell refinement level (leaf level).
     * @param rHanging   [out] Hanging-node constraints (one per slave node at
     *                   each 2:1 transition face).
     */
    static void ExtractPrimalHexMesh(
        OctreeType& rOctree,
        std::vector<std::array<double,3>>& rNodes,
        std::vector<std::array<int,8>>&   rCells,
        std::vector<int>&                 rCellLevel,
        std::vector<HangingConstraint>&   rHanging);

    /**
     * @brief Balances the octree, extracts the dual hex mesh, optionally carves
     *        and projects it, then writes a legacy ASCII VTK file.
     *
     * @param rOctree     Octree (bounding box must be set).
     * @param rFilename   Output .vtk path.
     * @param pTriangles  When non-null, carves away hexes outside the surface.
     * @param Project     When true (and @p pTriangles non-null), also fits the
     *                    carved core mesh to the surface with Jacobian control.
     * @param ProjIters   Total projection iterations (default 20000).
     * @param ProjSmooth  Iterations between smoothing epochs (default 1000).
     */
    static void WriteDualHexVtk(
        OctreeType& rOctree,
        const std::string& rFilename,
        const TriangleSoup* pTriangles = nullptr,
        bool Project = false,
        int ProjIters = 20000,
        int ProjSmooth = 1000);

    /**
     * @brief Writes the raw primal octree cells as (non-conforming) hexahedra.
     *
     * Useful for debugging and for direct octree visualisation.
     * The cell-data field "level" encodes each cell's refinement level.
     */
    static void WritePrimalVtk(OctreeType& rOctree, const std::string& rFilename);

    /**
     * @brief Convenience wrapper: builds the octree and writes the uncarved dual hex mesh.
     * @details Equivalent to calling @ref BuildFromSurfaceMesh followed by
     *          @ref WriteDualHexVtk with no carving or projection.
     * @param rSurfaceMesh     Surface ModelPart (from StlIO).
     * @param rVtkFilename     Output .vtk path.
     * @param RefinementDepth  Maximum refinement depth (default 5).
     * @param Adaptive         Use adaptive (curvature-driven) refinement when `true`
     *                         (default), or simple interface-cell refinement when `false`.
     */
    static void BuildAndWriteVtk(
        ModelPart& rSurfaceMesh,
        const std::string& rVtkFilename,
        std::size_t RefinementDepth = 5,
        bool Adaptive = true);

    /**
     * @brief Builds the octree, extracts the dual hex mesh, carves away exterior
     *        hexes, and writes the result to a VTK file.
     * @param rSurfaceMesh     Surface ModelPart (from StlIO).
     * @param rVtkFilename     Output .vtk path.
     * @param RefinementDepth  Maximum refinement depth (default 5).
     * @param Adaptive         Use adaptive (curvature-driven) refinement when `true`
     *                         (default), or simple interface-cell refinement when `false`.
     */
    static void BuildCarveAndWriteVtk(
        ModelPart& rSurfaceMesh,
        const std::string& rVtkFilename,
        std::size_t RefinementDepth = 5,
        bool Adaptive = true);

    /**
     * @brief Builds the octree, carves the dual block, and then **fits it to the
     *        input surface with Jacobian control** before writing.
     *
     * @param rSurfaceMesh     Surface ModelPart (from StlIO).
     * @param rVtkFilename     Output .vtk path.
     * @param RefinementDepth  Maximum refinement depth (default 5).
     * @param ProjIters        Total projection/optimisation iterations (default 20000).
     * @param ProjSmooth       Iterations between smoothing/threshold updates (default 1000).
     */
    static void BuildCarveProjectAndWriteVtk(
        ModelPart& rSurfaceMesh,
        const std::string& rVtkFilename,
        std::size_t RefinementDepth = 5,
        int ProjIters = 20000,
        int ProjSmooth = 1000,
        bool Adaptive = true);

    /**
     * @brief Debug/validation helper: writes the strongly-balanced octree leaves
     *        in the exact VTK format the reference HybridOctree_Hex expects from
     *        its `ReadOctree`.
     */
    static void WriteOctreeForReference(
        ModelPart& rSurfaceMesh,
        const std::string& rFilename,
        std::size_t RefinementDepth);

    ///@}
    ///@name Octree refinement
    ///@{

    /**
     * @brief Refines all leaves of @p rOctree to at least @p TargetDepth by
     *        repeatedly subdividing every leaf that is shallower than the target.
     *
     * The method iterates over a snapshot of the leaf list and subdivides each
     * leaf whose level is strictly less than @p TargetDepth, repeating until no
     * such leaf remains.  Existing leaves already at or beyond @p TargetDepth are
     * left unchanged.  If @p TargetDepth exceeds the octree's maximum depth
     * (`OctreeHybrid::GetDepth()`), it is silently clamped.
     *
     * @note Call `OctreeHybrid::StrongConstrain2To1()` and re-extract the mesh
     *       **after** all refinement operations have run — not between them.
     *
     * @param rOctree     The octree to refine in-place.
     * @param TargetDepth The minimum leaf depth to reach.  Cells at this depth
     *                    are not subdivided further.
     */
    static void RefineAllCells(OctreeType& rOctree, std::size_t TargetDepth);

    /**
     * @brief Refines octree cells that contain surface-triangle vertices to at
     *        least @p TargetDepth, approximating the interface geometry.
     *
     * For every vertex in @p rTriangles the method normalises the world-space
     * coordinates and, for each level 0 … @p TargetDepth-1, computes the
     * integer grid coordinates of the enclosing cell and subdivides that cell if
     * it is still a leaf.  This drives refinement along the paths from the root
     * to the finest cells that contain surface vertices, refining the octree
     * near the interface without touching interior or exterior regions.
     *
     * Multiple calls with different triangle soups (different geometry model
     * parts) can be chained: each call adds refinement around its own surface,
     * and the combined result is 2:1-balanced in a single
     * `StrongConstrain2To1()` call at the end.
     *
     * @note Call `OctreeHybrid::StrongConstrain2To1()` and re-extract the mesh
     *       **after** all refinement operations have run — not between them.
     *
     * @param rOctree     The octree to refine in-place.
     * @param rTriangles  Surface triangles in world coordinates (one entry per
     *                    triangle, each with 3 vertices of 3 doubles).
     * @param TargetDepth Maximum depth to refine interface cells to.  Clamped to
     *                    the octree's `GetDepth()` if larger.
     */
    static void RefineInterfaceCells(
        OctreeType&         rOctree,
        const TriangleSoup& rTriangles,
        std::size_t         TargetDepth);

    /**
     * @brief Converts a desired world-space element size to an equivalent octree depth.
     * @details Computes the minimum number of refinement levels needed so that every leaf
     * cell has a characteristic size **at most** @p ElementSize in every dimension.
     *
     * The conversion uses the octree's normalisation transform to avoid direct access to
     * the private bounding-box members: two points separated by @p ElementSize in each
     * axis direction are normalised, and the resulting coordinate deltas give the
     * per-axis scale factors.  The target depth is then:
     * @code
     *   d = ceil( log2(1 / min_norm_delta) )
     * @endcode
     * where `min_norm_delta` = min over axes of `ElementSize * scale_axis`.
     * Using the minimum ensures that all three axes reach the requested resolution.
     *
     * The result is clamped to `[0, rOctree.GetDepth()]`.
     *
     * @param rOctree          The octree whose bounding box provides the conversion.
     * @param ElementSize      Desired maximum cell size in world-space units.  Must be > 0.
     * @param ClampToMaxDepth  When true (default), clamp the result to the octree's current
     *                         maximum depth.  Pass false to obtain the raw geometric depth.
     * @return Equivalent refinement depth.
     */
    static std::size_t ElementSizeToDepth(OctreeType& rOctree, double ElementSize,
                                          bool ClampToMaxDepth = true);

    ///@}
    ///@name Carving, projection and mesh extraction (reusable by callers)
    ///@{
    // These stateless static helpers operate on the in-memory node/cell arrays
    // produced by @ref ExtractDualHexMesh, so external drivers (e.g. the octree
    // mesher modeler) can run the carve / projection / quality phases directly.

    /**
     * @brief Squared Euclidean distance between two 3-D points.
     * @param a First point (array of 3 doubles).
     * @param b Second point (array of 3 doubles).
     * @return (a[i]-b[i])^2 summed over i=0,1,2.
     */
    static double SqDist(const double a[3], const double b[3]);

    /**
     * @brief Triangle area from its three edge lengths (Heron's formula, clamped to ≥ 0).
     * @param a Length of the first edge.
     * @param b Length of the second edge.
     * @param c Length of the third edge.
     * @return Non-negative area; returns 0 when the triangle is degenerate.
     */
    static double TriArea(double a, double b, double c);

    /**
     * @brief Ray/triangle intersection (port of HexGen.cpp::Intersect).
     *
     * @param[out] e      intersection point (when the return value is 1).
     * @param[out] alpha  ray parameter at the intersection.
     * @return 1 if the ray hits strictly inside the triangle; 0 if it misses;
     *         -1 if the ray is parallel to the plane or grazes an edge/vertex.
     */
    static int TriRayIntersect(
        const double a[3], const double b[3], const double c[3],
        const double p[3], const double dir[3], double e[3], double& alpha);

    /**
     * @brief Unsigned distance from point @p p to triangle (@p a,@p b,@p c)
     *        (port of HexGen.cpp::PointToTri).
     *
     * Returns the perpendicular plane distance when the foot of the perpendicular
     * lies inside the triangle, otherwise the distance to the nearest edge or
     * vertex.  Early-outs when that already exceeds @p currMin.
     */
    static double PointToTri(
        const double a[3], const double b[3], const double c[3],
        const double p[3], double currMin);

    /**
     * @brief Carves hexes lying outside the surface (reference stage 4,
     *        `RemoveOutsideElement`, inside/outside part).
     * @details For every hex the signed distance of each corner node is evaluated
     *          via a ray-cast parity test + closest-triangle distance.  The hex is
     *          discarded when the keep test (@ref KeepCarvedCell) fails.  @p rCells
     *          and @p rCellLevel are compacted in-place; @p rNodes is not changed.
     * @param rTriangles  Input surface triangle soup (world coordinates).
     * @param rNodes      Hex node coordinates (world space); read-only.
     * @param rCells      [in/out] Hex connectivity; compacted after carving.
     * @param rCellLevel  [in/out] Per-cell level tags; compacted in parallel with @p rCells.
     */
    static void RemoveOutsideElement(
        const TriangleSoup& rTriangles,
        const std::vector<std::array<double,3>>& rNodes,
        std::vector<std::array<int,8>>& rCells,
        std::vector<int>& rCellLevel);

    /**
     * @brief Classifies every hex cell as inside (1) or outside (0) the surface.
     * @details For each cell a representative point (centroid of its 8 nodes) is
     *          tested against the triangle soup with the same ray-cast parity + signed-
     *          distance logic as @ref RemoveOutsideElement.  Results are written into
     *          @p rCellColor, which is resized to `rCells.size()` before any entry is set.
     * @param rTriangles  Input surface triangle soup (world coordinates).
     * @param rNodes      Hex node coordinates (world space); read-only.
     * @param rCells      Hex connectivity (indices into @p rNodes); read-only.
     * @param rCellColor  [out] Per-cell colour: 1 = inside, 0 = outside.
     */
    static void ClassifyInsideOutside(
        const TriangleSoup& rTriangles,
        const std::vector<std::array<double,3>>& rNodes,
        const std::vector<std::array<int,8>>& rCells,
        std::vector<int>& rCellColor);

    /**
     * @brief Per-node signed distance to the surface (negative outside).
     *
     * Sign from ray-cast crossing parity (odd crossings ⇒ inside), magnitude from
     * the closest triangle.  Cost O(#nodes · #triangles); computed in parallel.
     */
    static std::vector<double> ComputeNodeSignedDistance(
        const TriangleSoup& rTriangles,
        const std::vector<std::array<double,3>>& rNodes);

    /**
     * @brief Keep test for a carved hex: at most two outside corners and a small
     *        outside excursion relative to the deepest inside corner.
     * @details Implements the reference's keep rule:
     *          `n_outside < 3 && (min_neg_dist + OUT_IN_RATIO * max_pos_dist) >= 0`,
     *          where `OUT_IN_RATIO = 0.15`, `min_neg_dist` is the most-negative
     *          signed distance among the 8 corners (deepest inside), and
     *          `max_pos_dist` is the largest positive distance (furthest outside).
     *          A hex with at most 2 outside corners and a small outside excursion
     *          relative to its deepest inside reach is retained.
     * @param rCell        Connectivity record (8 node indices into the mesh).
     * @param rSignedDist  Per-node signed distance (negative = inside, positive = outside).
     * @return `true` if the hex should be kept; `false` if it should be discarded.
     */
    static bool KeepCarvedCell(
        const std::array<int,8>& rCell,
        const std::vector<double>& rSignedDist);

    /**
     * @brief Minimum scaled Jacobian over a hex's body centre and 8 corners
     *        (port of HexGen.cpp::Sj).
     * @details For each of the 9 evaluation points (body centre + 8 corners) the
     *          three edge vectors are built via @ref HexEdgeTriple, and the scaled
     *          Jacobian `V / (|e0| * |e1| * |e2|)` is computed.  The minimum over
     *          all 9 points is returned.  A value > 0 indicates a valid (non-inverted)
     *          element; a value ≤ 0 indicates an inverted or degenerate element.
     * @param p  The 8 corner coordinates of the hex (world space), ordered by the
     *           Kratos Hexahedra3D8 local-node convention.
     * @return   Minimum scaled Jacobian; in (-∞, 1].
     */
    static double ScaledJacobianMin(const double p[8][3]);

    /**
     * @brief Minimum raw Jacobian (signed volume of the edge-triple) over a hex's
     *        body centre and 8 corners.
     * @details Evaluates the triple product `det(e0, e1, e2)` at all 9 evaluation
     *          points (body centre + 8 corners) and returns the minimum.  Unlike
     *          @ref ScaledJacobianMin, this is not normalised by edge lengths, so
     *          the result is scale-dependent.  Negative ⇒ inverted element.
     * @param p  The 8 corner coordinates of the hex (world space).
     * @return   Minimum raw Jacobian value.
     */
    static double JacobianMin(const double p[8][3]);

    /**
     * @brief Signed volume (scalar triple product) of three edge vectors.
     * @details Returns `det([e0, e1, e2])` = `e0 · (e1 × e2)`.  Positive when the
     *          three vectors form a right-handed frame; negative otherwise.
     * @param e0 First edge vector (length 3).
     * @param e1 Second edge vector (length 3).
     * @param e2 Third edge vector (length 3).
     * @return   Scalar triple product.
     */
    static double TripleProduct(const double e0[3], const double e1[3], const double e2[3]);

    /**
     * @brief Builds the edge triple of a hex at a specified evaluation point.
     * @details For the body centre (`corner == 8`) the three edge vectors span the
     *          full diagonal of the hex; for a corner (`corner` in 0…7) they follow
     *          the three edges emanating from that corner.  These vectors are the
     *          arguments expected by @ref TripleProduct and @ref ScaledJacobianMin.
     * @param p       The 8 corner coordinates of the hex (world space).
     * @param corner  Evaluation point: 0…7 for a corner, 8 for the body centre.
     * @param e0      [out] First edge vector.
     * @param e1      [out] Second edge vector.
     * @param e2      [out] Third edge vector.
     */
    static void HexEdgeTriple(const double p[8][3], int corner,
                              double e0[3], double e1[3], double e2[3]);

    /**
     * @brief Closest point on a triangle to a query point (Ericson, RTCD §5.1.5).
     * @details Projects @p p onto the plane of triangle (@p a, @p b, @p c), then
     *          clamps the result to the triangle using barycentric coordinates.
     *          Works correctly for degenerate (collinear) triangles.
     * @param a  First triangle vertex (length 3).
     * @param b  Second triangle vertex (length 3).
     * @param c  Third triangle vertex (length 3).
     * @param p  Query point (length 3).
     * @param q  [out] Closest point on the triangle to @p p (length 3).
     */
    static void ClosestPointOnTriangle(
        const double a[3], const double b[3], const double c[3],
        const double p[3], double q[3]);

    /**
     * @brief Closest point on the whole triangle soup to a query point.
     * @details Iterates all triangles in @p rTri and returns the global minimum
     *          squared distance.  The closest point coordinates are written to @p q
     *          and the index of the nearest triangle (0-based) is written to @p tri.
     * @param rTri  Surface triangle soup to search.
     * @param p     Query point in world space (length 3).
     * @param q     [out] Closest point on the soup (length 3).
     * @param tri   [out] Zero-based index of the nearest triangle in @p rTri.
     * @return      Squared distance from @p p to the closest point @p q.
     */
    static double ClosestPointOnSoup(
        const TriangleSoup& rTri, const double p[3], double q[3], int& tri);

    /**
     * @brief Extracts the boundary quad faces of a hex set.
     * @details A face is a boundary face when it is shared by exactly one
     *          hexahedron (i.e.\ it lies on the outer surface of the mesh).
     *          Each result entry is a 5-element array `{n0, n1, n2, n3, owner_hex}`,
     *          where `n0…n3` are the global node indices of the face and
     *          `owner_hex` is the index of the single hex that owns it.
     * @param rCells  Hexahedra connectivity (`N × 8` node indices into the mesh
     *                node array).
     * @return        Vector of boundary face descriptors; one per boundary quad.
     */
    static std::vector<std::array<int,5>> ExtractBoundaryFaces(
        const std::vector<std::array<int,8>>& rCells);

    /**
     * @brief Buffer-zone clearance — port of paper Section 2.3 hemisphere restriction.
     * @details Identifies boundary hexes that form a "fold" (their incident face
     *          normals do not fit any open hemisphere) and removes the most-boundary
     *          such hex.  The process repeats until no fold remains or @p MaxRounds is
     *          exhausted.  This step is critical for the surface-projection stage: folds
     *          in the buffer layer prevent convergence of the Jacobian optimiser.  In
     *          practice it eliminates the 11% inverted buffer hexes that would otherwise
     *          remain after extraction at depth 5.
     * @param rNodes      Hex node coordinates (world space); read-only.
     * @param rCells      [in/out] Hex connectivity; offending hexes are erased.
     * @param rCellLevel  [in/out] Per-cell level tags; compacted in parallel with @p rCells.
     * @param MaxRounds   Maximum number of removal iterations (default 50).
     */
    static void ClearBufferZone(
        const std::vector<std::array<double,3>>& rNodes,
        std::vector<std::array<int,8>>& rCells,
        std::vector<int>& rCellLevel,
        int MaxRounds = 50);

    /**
     * @brief Meshes the buffer zone and fits the carved core mesh to the input
     *        surface with Jacobian control — port of HexGen.cpp::ProjectToIsoSurface.
     *
     * @param rTriangles   Input surface triangles (world coordinates).
     * @param rNodes       Core-mesh nodes; duplicate shell nodes are appended.
     * @param rCells       Core hexes; buffer hexes are appended.
     * @param rCellLevel   Per-cell level tag; buffer hexes are tagged -2.
     * @param TotalIters   Total gradient iterations.
     * @param SmoothEvery  Iterations between smart-Laplacian/threshold updates.
     */
    static void ProjectToIsoSurface(
        const TriangleSoup& rTriangles,
        std::vector<std::array<double,3>>& rNodes,
        std::vector<std::array<int,8>>& rCells,
        std::vector<int>& rCellLevel,
        int TotalIters,
        int SmoothEvery);

    /**
     * @brief Writes a hex mesh to a legacy ASCII VTK file, compacting away any
     *        nodes not referenced by a cell.
     */
    static void WriteHexVtk(
        const std::string& rFilename,
        const std::vector<std::array<double,3>>& rNodes,
        const std::vector<std::array<int,8>>& rCells,
        const std::vector<int>& rCellLevel);

    ///@}
};

} // namespace Kratos
