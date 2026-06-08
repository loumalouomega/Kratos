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
     * @brief Builds an OctreeHybrid adaptively refined around a surface mesh.
     *
     * @param rSurfaceMesh     ModelPart whose Geometries() container holds the
     *                         surface triangles (populated by StlIO::ReadModelPart).
     * @param RefinementDepth  Maximum refinement depth near the surface.
     *                         Must be in [1, OctreeHybridKratosConfiguration::MAX_DEPTH].
     */
    static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth,
        bool Adaptive = true);

    /**
     * @brief Builds the octree with the reference HybridOctree_Hex **adaptive**
     *        refinement criterion (curvature + feature thickness), so the leaf
     *        set matches the reference for a real geometry such as the bunny.
     */
    static std::unique_ptr<OctreeType> BuildAdaptiveFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth);

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
     *        @ref TriangleSoup (the container carving/projection test against).
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
     * @brief Convenience wrapper: builds octree and writes the dual hex mesh.
     *
     * @param rSurfaceMesh     Surface ModelPart (from StlIO).
     * @param rVtkFilename     Output .vtk path.
     * @param RefinementDepth  Maximum refinement depth (default 5).
     */
    static void BuildAndWriteVtk(
        ModelPart& rSurfaceMesh,
        const std::string& rVtkFilename,
        std::size_t RefinementDepth = 5,
        bool Adaptive = true);

    /**
     * @brief Builds the octree, extracts the dual hex mesh, and **carves** it
     *        against the input surface before writing.
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

    /// Squared Euclidean distance between two points.
    static double SqDist(const double a[3], const double b[3]);

    /// Triangle area from its three edge lengths (Heron's formula, clamped ≥ 0).
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
     * @brief Carves hexes lying outside the surface — reference stage 4
     *        (`RemoveOutsideElement`), inside/outside part.
     */
    static void RemoveOutsideElement(
        const TriangleSoup& rTriangles,
        const std::vector<std::array<double,3>>& rNodes,
        std::vector<std::array<int,8>>& rCells,
        std::vector<int>& rCellLevel);

    /**
     * @brief Per-cell inside(1)/outside(0) classification against the surface.
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
     */
    static bool KeepCarvedCell(
        const std::array<int,8>& rCell,
        const std::vector<double>& rSignedDist);

    /**
     * @brief Minimum scaled Jacobian over a hex's body centre and 8 corners
     *        (port of HexGen.cpp::Sj).
     */
    static double ScaledJacobianMin(const double p[8][3]);

    /// Minimum raw Jacobian (signed volume) over a hex's body centre and 8 corners.
    static double JacobianMin(const double p[8][3]);

    /// Determinant (signed volume) of three edge vectors.
    static double TripleProduct(const double e0[3], const double e1[3], const double e2[3]);

    /// Builds the body-centre (corner=8) or corner (corner=0..7) edge triple of a hex.
    static void HexEdgeTriple(const double p[8][3], int corner,
                              double e0[3], double e1[3], double e2[3]);

    /// Closest point @p q on triangle (@p a,@p b,@p c) to @p p (Ericson, RTCD).
    static void ClosestPointOnTriangle(
        const double a[3], const double b[3], const double c[3],
        const double p[3], double q[3]);

    /// Closest point on the whole triangle soup to @p p; returns squared distance.
    static double ClosestPointOnSoup(
        const TriangleSoup& rTri, const double p[3], double q[3], int& tri);

    /// Boundary quad faces of a hex set (faces owned by exactly one hex).
    static std::vector<std::array<int,5>> ExtractBoundaryFaces(
        const std::vector<std::array<int,8>>& rCells);

    /**
     * @brief Buffer-zone clearance — port of the paper's Section 2.3 restriction.
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
