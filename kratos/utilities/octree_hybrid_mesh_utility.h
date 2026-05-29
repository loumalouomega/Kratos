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
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Project includes
#include "includes/model_part.h"
#include "geometries/point.h"
#include "spatial_containers/octree_hybrid.h"
#include "spatial_containers/octree_hybrid_cell.h"
#include "spatial_containers/octree_hybrid_configure.h"

namespace Kratos {

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
 *       test_octree_hybrid_dual_mesh.py.  The merged node array removes
 *       duplicate vertices, but the mesh is **still not fully watertight**: at
 *       transition regions a few percent of faces remain non-conforming (small
 *       geometric gaps where a plain-dual hex sits next to a template that does
 *       not cover the same vertices, plus a few overlaps).  This residue is a
 *       *geometric element-tiling* gap, not a node-id one — node merging alone
 *       does not remove it.  Closing it needs the plain-dual/template coverage
 *       to match the reference cell-for-cell (the strict uniform-region gating
 *       and exact consume tried here both regressed it).  Adequate for
 *       visualisation; needs a cleanup pass before FE analysis.
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
        std::size_t RefinementDepth)
    {
        KRATOS_ERROR_IF(RefinementDepth < 1 || RefinementDepth > ConfigurationType::MAX_DEPTH)
            << "OctreeHybridMeshUtility: RefinementDepth must be in [1, "
            << ConfigurationType::MAX_DEPTH << "], got " << RefinementDepth << std::endl;

        // --- Bounding box (1 % padding) ---
        double lo[3] = { std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max() };
        double hi[3] = { std::numeric_limits<double>::lowest(),
                         std::numeric_limits<double>::lowest(),
                         std::numeric_limits<double>::lowest() };

        for (const auto& r_node : rSurfaceMesh.Nodes()) {
            lo[0] = std::min(lo[0], r_node.X()); hi[0] = std::max(hi[0], r_node.X());
            lo[1] = std::min(lo[1], r_node.Y()); hi[1] = std::max(hi[1], r_node.Y());
            lo[2] = std::min(lo[2], r_node.Z()); hi[2] = std::max(hi[2], r_node.Z());
        }
        for (std::size_t d = 0; d < 3; ++d) {
            const double span = hi[d] - lo[d];
            lo[d] -= 0.01 * span;
            hi[d] += 0.01 * span;
        }

        auto p_octree = std::make_unique<OctreeType>(RefinementDepth);
        p_octree->SetBoundingBox(lo, hi);

        // --- Collect triangles ---
        std::vector<GeometryType*> triangles;
        triangles.reserve(rSurfaceMesh.NumberOfGeometries());
        for (auto& r_geom : rSurfaceMesh.Geometries()) triangles.push_back(&r_geom);

        // --- Adaptive refinement ---
        for (std::size_t iter = 0; iter < RefinementDepth; ++iter) {
            std::vector<CellType*> leaves;
            p_octree->GetAllLeavesVector(leaves);
            bool any_split = false;
            for (CellType* p_cell : leaves) {
                if (static_cast<std::size_t>(p_cell->GetLevel()) >= RefinementDepth) continue;
                double n_lo[3], n_hi[3], w_lo[3], w_hi[3];
                p_cell->GetMinPointNormalized(n_lo);
                p_cell->GetMaxPointNormalized(n_hi);
                p_octree->ScaleBackToOriginalCoordinate(n_lo, w_lo);
                p_octree->ScaleBackToOriginalCoordinate(n_hi, w_hi);
                const Point box_lo(w_lo[0], w_lo[1], w_lo[2]);
                const Point box_hi(w_hi[0], w_hi[1], w_hi[2]);
                for (GeometryType* p_tri : triangles) {
                    if (p_tri->HasIntersection(box_lo, box_hi)) {
                        p_octree->SubdivideCellByIdAndLevel(p_cell->GetId(), p_cell->GetLevel());
                        any_split = true;
                        break;
                    }
                }
            }
            if (!any_split) break;
        }
        return p_octree;
    }

    /**
     * @brief Writes the **dual hex mesh** (the proper HybridOctree_Hex output).
     *
     * The octree is first strongly 2:1-balanced, then the dual mesh is extracted
     * as described in the class documentation.  Every interior primal vertex
     * contributes exactly one conforming hexahedron.  The 13-element template
     * additionally fills face-transition regions with higher-quality elements.
     *
     * Cell data field "level" (in the VTK output) is the level of the cell
     * whose centre is at dual-hex node 0, useful for colouring in Paraview.
     *
     * @param rOctree   Octree (bounding box must be set).
     * @param rFilename Output .vtk path.
     */
    static void WriteDualHexVtk(OctreeType& rOctree, const std::string& rFilename)
    {
        // ------------------------------------------------------------------ //
        // 1. Balance + collect leaves
        // ------------------------------------------------------------------ //
        rOctree.StrongConstrain2To1();

        std::vector<CellType*> leaves;
        rOctree.GetAllLeavesVector(leaves);
        const int N = static_cast<int>(leaves.size());

        const std::size_t depth = rOctree.GetDepth();
        const std::size_t R    = std::size_t{1} << depth;  // cells per axis at max depth
        const std::size_t pts  = R + 1;                    // primal grid points per axis

        // ------------------------------------------------------------------ //
        // 2. Compute cell centres (dual mesh node positions)
        // ------------------------------------------------------------------ //
        std::vector<std::array<double,3>> centres(N);
        for (int i = 0; i < N; ++i) {
            double n_lo[3], n_hi[3];
            leaves[i]->GetMinPointNormalized(n_lo);
            leaves[i]->GetMaxPointNormalized(n_hi);
            double n_ctr[3] = { 0.5*(n_lo[0]+n_hi[0]),
                                 0.5*(n_lo[1]+n_hi[1]),
                                 0.5*(n_lo[2]+n_hi[2]) };
            rOctree.ScaleBackToOriginalCoordinate(n_ctr, centres[i].data());
        }

        // ------------------------------------------------------------------ //
        // 3. Build primal mesh: leaf cells as hexes with shared vertices
        //
        // Corner ordering (matches the reference HexGen.cpp):
        //   c=0 (x, y, z)     c=1 (x+s, y, z)     c=2 (x+s, y+s, z)
        //   c=3 (x, y+s, z)   c=4 (x, y, z+s)     c=5 (x+s, y, z+s)
        //   c=6 (x+s, y+s, z+s)  c=7 (x, y+s, z+s)
        // ------------------------------------------------------------------ //
        static constexpr int CX[8] = {0,1,1,0,0,1,1,0};
        static constexpr int CY[8] = {0,0,1,1,0,0,1,1};
        static constexpr int CZ[8] = {0,0,0,0,1,1,1,1};

        // Maps integer vertex key → sequential vertex id
        std::unordered_map<std::size_t, int> vid_map;
        vid_map.reserve(N * 4);

        // For each sequential vertex: list of (leaf_idx, corner_idx) pairs
        std::vector<std::vector<std::pair<int,int>>> vert_adj;
        vert_adj.reserve(N * 2);

        // primal_elem[i][c] = vertex id of corner c of leaf i
        std::vector<std::array<int,8>> primal_elem(N);

        for (int i = 0; i < N; ++i) {
            const auto* p = leaves[i];
            const int lv = p->GetLevel();
            const int gx = p->GetGridX(), gy = p->GetGridY(), gz = p->GetGridZ();
            const std::size_t stride = std::size_t{1} << (depth - static_cast<std::size_t>(lv));

            for (int c = 0; c < 8; ++c) {
                const std::size_t ix = static_cast<std::size_t>(gx + CX[c]) * stride;
                const std::size_t iy = static_cast<std::size_t>(gy + CY[c]) * stride;
                const std::size_t iz = static_cast<std::size_t>(gz + CZ[c]) * stride;
                const std::size_t key = iz * pts * pts + iy * pts + ix;

                auto [it, ins] = vid_map.emplace(key, static_cast<int>(vert_adj.size()));
                if (ins) vert_adj.push_back({});
                vert_adj[it->second].emplace_back(i, c);
                primal_elem[i][c] = it->second;
            }
        }

        const int NV = static_cast<int>(vert_adj.size());

        // Characteristic length: world size of one finest-level cell.  Used to
        // scale the node-merge tolerance so it is independent of the model's
        // absolute coordinate range.
        const double invR = 1.0 / static_cast<double>(R);
        double mc_w0[3], mc_w1[3];
        { const double n0[3] = {0,0,0}, n1[3] = {invR, invR, invR};
          rOctree.ScaleBackToOriginalCoordinate(n0, mc_w0);
          rOctree.ScaleBackToOriginalCoordinate(n1, mc_w1); }
        const double min_cell = std::min({ mc_w1[0]-mc_w0[0], mc_w1[1]-mc_w0[1], mc_w1[2]-mc_w0[2] });

        // ------------------------------------------------------------------ //
        // 4. Build face adjacency using O(N·depth) octree point queries.
        //
        // For each leaf i, for each face j (0..5), query 4 sub-quadrant points
        // just past that face using pGetCellNormalized.
        //
        //   adj[i][j].count = 0 (boundary) | 1 (same/larger nbr) | 4 (4 smaller)
        //   adj[i][j].ids[0..3]: leaf sequential indices of the neighbor(s)
        //
        // Sub-quadrant ordering matches the face corner ordering:
        //   Q0=[free1_lo, free2_lo], Q1=[free1_hi, free2_lo],
        //   Q2=[free1_hi, free2_hi], Q3=[free1_lo, free2_hi]
        // ------------------------------------------------------------------ //

        // Map octree flat-id → leaf sequential index
        std::unordered_map<int, int> id_to_idx;
        id_to_idx.reserve(N);
        for (int i = 0; i < N; ++i) id_to_idx[leaves[i]->GetId()] = i;

        // For each face: fixed axis, whether it's the hi boundary, and 2 free axes
        static constexpr int FACE_FIXED[6]    = {2,1,0,0,1,2};  // axis index
        static constexpr bool FACE_HI[6]      = {false,false,false,true,true,true};
        static constexpr int FACE_FREE1[6]    = {0,0,1,1,0,0};  // first free axis
        static constexpr int FACE_FREE2[6]    = {1,2,2,2,2,1};  // second free axis

        static constexpr double EPS = 1e-9;
        static constexpr double Q[2] = {0.25, 0.75};  // sub-quadrant centres

        struct FaceAdj { int count; std::array<int,4> ids; };
        std::vector<std::array<FaceAdj,6>> adj(N);
        for (auto& a : adj) for (auto& f : a) { f.count = 0; f.ids.fill(-1); }

        for (int i = 0; i < N; ++i) {
            double n_lo[3], n_hi[3];
            leaves[i]->GetMinPointNormalized(n_lo);
            leaves[i]->GetMaxPointNormalized(n_hi);

            for (int j = 0; j < 6; ++j) {
                if (adj[i][j].count != 0) continue;  // filled from opposite side

                const int fa  = FACE_FIXED[j];
                const int f1  = FACE_FREE1[j];
                const int f2  = FACE_FREE2[j];
                const double fixed_coord = FACE_HI[j]
                    ? n_hi[fa] + EPS : n_lo[fa] - EPS;

                // Boundary check
                if (fixed_coord < 0.0 || fixed_coord > 1.0) continue;

                const int opp = 5 - j;

                // Query 4 sub-quadrant centres just past this face
                int found[4] = {-1,-1,-1,-1};
                for (int q = 0; q < 4; ++q) {
                    const int qi = q & 1;       // Q[0] or Q[1] for free1
                    const int qj = (q >> 1) & 1; // Q[0] or Q[1] for free2
                    // Quadrant ordering: Q0=(0,0), Q1=(1,0), Q2=(1,1), Q3=(0,1)
                    const double q_f1 = n_lo[f1] + Q[qi] * (n_hi[f1]-n_lo[f1]);
                    const double q_f2 = n_lo[f2] + Q[qj] * (n_hi[f2]-n_lo[f2]);

                    double pt[3];
                    pt[fa] = fixed_coord;
                    pt[f1] = q_f1;
                    pt[f2] = q_f2;
                    if (pt[0] < 0||pt[0] > 1||pt[1] < 0||pt[1] > 1||pt[2] < 0||pt[2] > 1)
                        continue;

                    CellType* nb = rOctree.pGetCellNormalized(pt);
                    if (!nb || nb == leaves[i]) continue;
                    auto it = id_to_idx.find(nb->GetId());
                    if (it == id_to_idx.end()) continue;
                    found[q] = it->second;
                }

                // Determine valence: all same → 1 neighbour, 4 distinct → 4 smaller
                int first = -1;
                for (int q = 0; q < 4; ++q) if (found[q] >= 0) { first = found[q]; break; }
                if (first < 0) continue;  // boundary on all 4

                bool all_same = (found[0]==first && found[1]==first &&
                                 found[2]==first && found[3]==first);
                if (all_same) {
                    adj[i][j].count = 1; adj[i][j].ids[0] = first;
                    // Fill opposite face if not already done
                    if (adj[first][opp].count == 0) {
                        adj[first][opp].count = 1; adj[first][opp].ids[0] = i;
                    }
                } else {
                    // 4 (or fewer) distinct smaller neighbours
                    adj[i][j].count = 0;
                    for (int q = 0; q < 4; ++q) {
                        adj[i][j].ids[q] = found[q];
                        if (found[q] >= 0) {
                            adj[i][j].count++;
                            // Mark opposite face of smaller cell
                            if (adj[found[q]][opp].count == 0) {
                                adj[found[q]][opp].count = 1;
                                adj[found[q]][opp].ids[0] = i;
                            }
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------ //
        // 5. Extract dual hexes
        //
        // Dual mesh node positions are cell centres.
        // Node deduplication: key = leaf sequential index (each cell has exactly
        // one centre, so no additional deduplication is needed here).
        //
        // Output connectivity: each element is 8 leaf indices.
        // ------------------------------------------------------------------ //

        // idTransform: maps which corner k of a cell touches the primal vertex
        // to which dual hex node position that cell's centre should occupy.
        // From StaticVars.h: idTransform[8] = {6,7,4,5,2,3,0,1}
        static constexpr int idTransform[8] = {6,7,4,5,2,3,0,1};

        // dual_cells[e][n] = leaf index whose centre is dual hex node n
        std::vector<std::array<int,8>> dual_cells;
        dual_cells.reserve(N);

        // The 13-element transition templates (5b) are built FIRST.  Following
        // the reference (DualFullHexMeshExtraction), each template *replaces*
        // exactly one plain dual hex - the one at the transition vertex - while
        // its other 12 hexes fill the gap the plain dual leaves there.  The
        // template loop records that consumed vertex in `consumed[]`, and the
        // plain dual pass (5a, run afterwards) skips it.  Without this the two
        // meshes overlap in every transition region.
        std::vector<bool> consumed(NV, false);

        // -- 5b. 13-element template: fill face-transition regions --
        // For each leaf i and each face j where adj[i][j].count == 4 (4 smaller
        // neighbors), compute the 32 reference points and create 13 dual hexes.
        //
        // Template tables from StaticVars.h (t1Id[variant][13][8]):
        static constexpr int t1Id[2][13][8] = {
            {{0,1,5,4,18,16,21,20},{1,2,6,5,16,17,22,21},{2,3,7,6,17,19,23,22},
             {4,5,9,8,20,21,25,24},{5,6,10,9,21,22,26,25},{6,7,11,10,22,23,27,26},
             {8,9,13,12,24,25,30,28},{9,10,14,13,25,26,31,30},{10,11,15,14,26,27,29,31},
             {20,21,25,24,18,16,30,28},{22,23,27,26,17,19,29,31},
             {21,22,26,25,16,17,31,30},{16,17,31,30,18,19,29,28}},
            {{21,20,18,16,5,4,0,1},{22,21,16,17,6,5,1,2},{23,22,17,19,7,6,2,3},
             {25,24,20,21,9,8,4,5},{26,25,21,22,10,9,5,6},{27,26,22,23,11,10,6,7},
             {30,28,24,25,13,12,8,9},{31,30,25,26,14,13,9,10},{29,31,26,27,15,14,10,11},
             {30,28,18,16,25,24,20,21},{29,31,17,19,27,26,22,23},
             {31,30,16,17,26,25,21,22},{29,28,18,19,31,30,16,17}}
        };

        // Face direction axes for the 13-element template position check
        // xyz1[j], xyz2[j]: the two in-plane axes of face j
        static constexpr int xyz1[6] = {0,0,1,1,0,0};  // first in-plane axis index
        static constexpr int xyz2[6] = {1,2,2,2,2,1};  // second in-plane axis index

        // Single, merged output node array (the reference's hexMesh.v): every
        // hex - plain dual or template - references into `nodes`, and any two
        // points closer than the merge tolerance share one id.  Seeded with the
        // N cell centres (the dual node positions) so a template point landing on
        // a cell centre reuses the dual node id and the template stays conforming
        // with the plain dual hexes around it.  Dual node i == cell centre i.
        std::vector<std::array<double,3>> nodes = centres;
        std::vector<std::array<int,8>> tmpl_cells;
        tmpl_cells.reserve(N);

        // Tolerances scaled by the finest cell size (model-scale independent).
        const double merge_eps  = 1.0e-4 * min_cell;     // points within this merge
        const double merge_tol2 = merge_eps * merge_eps;
        const double bucket = 1.0e-2 * min_cell;          // spatial-hash bucket size
        const double QUANT = 1.0 / bucket;
        std::unordered_map<std::size_t, std::vector<int>> node_hash;

        auto cell_hash = [](long long a, long long b, long long c) -> std::size_t {
            std::size_t h = 1469598103934665603ULL;
            for (long long v : {a, b, c})
                h = (h ^ static_cast<std::size_t>(v)) * 1099511628211ULL;
            return h;
        };
        auto bkey = [&](const std::array<double,3>& p) -> std::array<long long,3> {
            return { std::llround(p[0]*QUANT), std::llround(p[1]*QUANT), std::llround(p[2]*QUANT) };
        };
        for (int i = 0; i < N; ++i) {
            const auto b = bkey(nodes[i]);
            node_hash[cell_hash(b[0],b[1],b[2])].push_back(i);
        }

        auto find_or_add_node = [&](const std::array<double,3>& pos) -> int {
            const auto b = bkey(pos);
            for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                auto it = node_hash.find(cell_hash(b[0]+dx, b[1]+dy, b[2]+dz));
                if (it == node_hash.end()) continue;
                for (int m : it->second) {
                    const auto& q = nodes[m];
                    double d2 = 0;
                    for (int d = 0; d < 3; ++d) d2 += (q[d]-pos[d])*(q[d]-pos[d]);
                    if (d2 < merge_tol2) return m;
                }
            }
            const int id = static_cast<int>(nodes.size());
            nodes.push_back(pos);
            node_hash[cell_hash(b[0],b[1],b[2])].push_back(id);
            return id;
        };

        // -- Edge/corner transition sub-templates (4/3/5-element) --------------
        // Tables from StaticVars.h.  Indexed [variant][cell][corner]; variant is
        // the same `tmpl` selector as the 13-element base.
        static constexpr int t2Id[2][4][8] = {
            {{0,8,9,2,1,12,13,3},{2,9,10,4,3,13,14,5},{3,13,14,5,1,12,15,6},{4,10,11,7,5,14,15,6}},
            {{13,3,1,12,9,2,0,8},{14,5,3,13,10,4,2,9},{15,6,1,12,14,5,3,13},{15,6,5,14,11,7,4,10}}};
        static constexpr int t22Id[2][4][8] = {
            {{8,0,2,9,14,1,3,12},{9,2,4,10,12,3,5,13},{12,3,5,13,14,1,6,15},{10,4,7,11,13,5,6,15}},
            {{3,12,14,1,2,9,8,0},{5,13,12,3,4,10,9,2},{6,15,14,1,5,13,12,3},{6,15,13,5,7,11,10,4}}};
        static constexpr int t3Id[2][3][8] = {
            {{0,8,9,2,1,12,13,3},{2,9,10,4,3,13,14,5},{4,10,11,6,5,14,15,7}},
            {{13,3,1,12,9,2,0,8},{14,5,3,13,10,4,2,9},{15,7,5,14,11,6,4,10}}};
        static constexpr int t32Id[2][3][8] = {
            {{0,2,9,8,1,3,12,14},{2,4,10,9,3,5,13,12},{4,6,11,10,5,7,15,13}},
            {{12,14,1,3,9,8,0,2},{13,12,3,5,10,9,2,4},{15,13,5,7,11,10,4,6}}};
        static constexpr int t33Id[2][3][8] = {
            {{8,0,2,9,12,1,3,13},{9,2,4,10,13,3,5,14},{10,4,6,11,14,5,7,15}},
            {{3,13,12,1,2,9,8,0},{5,14,13,3,4,10,9,2},{7,15,14,5,6,11,10,4}}};
        static constexpr int t34Id[2][3][8] = {
            {{8,9,2,0,12,14,3,1},{9,10,4,2,14,15,5,3},{10,11,6,4,15,13,7,5}},
            {{3,1,12,14,2,0,8,9},{5,3,14,15,4,2,9,10},{7,5,15,13,6,4,10,11}}};
        static constexpr int t4Id[2][5][8] = {
            {{0,8,9,2,1,12,13,3},{2,9,10,4,3,13,14,5},{4,10,11,6,5,14,15,7},{0,2,4,6,1,3,5,7},{3,13,14,5,1,12,15,7}},
            {{13,3,1,12,9,2,0,8},{14,5,3,13,10,4,2,9},{15,7,5,14,11,6,4,10},{5,7,1,3,4,6,0,2},{15,7,1,12,14,5,3,13}}};
        static constexpr int t42Id[2][5][8] = {
            {{0,2,9,8,1,3,12,14},{2,4,10,9,3,5,13,12},{4,6,11,10,5,7,15,13},{3,5,13,12,1,7,15,14},{0,6,4,2,1,7,5,3}},
            {{12,14,1,3,9,8,0,2},{13,12,3,5,10,9,2,4},{15,13,5,7,11,10,4,6},{15,14,1,7,13,12,3,5},{5,3,1,7,4,2,0,6}}};
        static constexpr int t43Id[2][5][8] = {
            {{8,0,2,9,12,1,3,13},{9,2,4,10,13,3,5,14},{10,4,6,11,14,5,7,15},{13,3,5,14,12,1,7,15},{2,0,6,4,3,1,7,5}},
            {{3,13,12,1,2,9,8,0},{5,14,13,3,4,10,9,2},{7,15,14,5,6,11,10,4},{7,15,12,1,5,14,13,3},{7,5,3,1,6,4,2,0}}};
        static constexpr int t44Id[2][5][8] = {
            {{0,8,9,2,1,12,14,3},{2,9,10,4,3,14,15,5},{4,10,11,6,5,15,13,7},{3,14,15,5,1,12,13,7},{0,2,4,6,1,3,5,7}},
            {{14,3,1,12,9,2,0,8},{15,5,3,14,10,4,2,9},{13,7,5,15,11,6,4,10},{13,7,1,12,15,5,3,14},{5,7,1,3,4,6,0,2}}};
        // "Side" faces of face j: pSId = the two faces whose transition edges run
        // alongside j; pS2Id = the two on the far (already-meshed) side.
        static constexpr int pSId[6][2]  = {{2,1},{2,0},{1,0},{1,0},{2,0},{2,1}};
        static constexpr int pS2Id[6][2] = {{3,4},{3,5},{4,5},{4,5},{3,5},{3,4}};

        // `available` == the reference's `collectNum`: interior (valence-8)
        // primal vertices whose 8 surrounding cells are NOT all the same size.
        // A transition template may consume one such vertex (replacing its plain
        // dual hex); once consumed it is no longer available to other templates.
        std::vector<bool> available(NV, false);
        for (int v = 0; v < NV; ++v) {
            if (vert_adj[v].size() != 8) continue;
            const int lv0 = leaves[vert_adj[v][0].first]->GetLevel();
            bool uniform = true;
            for (auto [ci, co] : vert_adj[v])
                if (leaves[ci]->GetLevel() != lv0) { uniform = false; break; }
            if (!uniform) available[v] = true;
        }

        // Neighbour helpers (mirror elementValenceNumber / elementValence, with
        // safe handling of boundary faces and missing neighbours).
        auto vcount = [&](int c, int fc) -> int { return c < 0 ? 0 : adj[c][fc].count; };
        auto nb     = [&](int c, int fc, int k) -> int { return c < 0 ? -1 : adj[c][fc].ids[k]; };

        // Consume the collectNum vertex a template replaces: round the template's
        // transition point to the finest grid and look it up among the primal
        // vertices.  Rounding (rather than a fixed-radius search) is what the
        // reference's off-grid `ptmp` positions need - it snaps to the intended
        // octree vertex.  Only an available (collectNum) vertex is consumed.
        auto consume_at = [&](const double* world) -> bool {
            double pn[3];
            rOctree.NormalizeCoordinates(world, pn);
            long long g[3];
            for (int d = 0; d < 3; ++d) {
                g[d] = std::llround(pn[d] * static_cast<double>(R));
                if (g[d] < 0 || g[d] > static_cast<long long>(R)) return false;
            }
            const std::size_t key = static_cast<std::size_t>(g[2])*pts*pts
                                  + static_cast<std::size_t>(g[1])*pts
                                  + static_cast<std::size_t>(g[0]);
            auto it = vid_map.find(key);
            if (it == vid_map.end() || !available[it->second]) return false;
            available[it->second] = false;
            consumed[it->second]  = true;
            return true;
        };

        // Emit a template's hexes.  P is the point array it indexes (p[] for the
        // 13-element base, p16[] for the sub-templates); `table` is the resolved
        // [cell][corner] connectivity; `ptmp` is the transition vertex used for
        // the collectNum bookkeeping.  Deletion modes (matching the reference):
        //   DEL_ALWAYS  : always keep (13-element and 4-element templates).
        //   DEL_IF_AVAIL: keep only if `ptmp`'s vertex was still available
        //                 (3-element templates; otherwise a neighbour built it).
        //   DEL_IF_NEW  : keep only if it introduced at least one new node
        //                 (5-element templates; otherwise fully overlapped).
        enum DelMode { DEL_ALWAYS, DEL_IF_AVAIL, DEL_IF_NEW };
        auto emit = [&](const double (*P)[3], const int (*table)[8], int nh,
                        const double* ptmp, DelMode mode) {
            const std::size_t before = nodes.size();
            std::array<std::array<int,8>,13> staged;
            for (int k = 0; k < nh; ++k)
                for (int l = 0; l < 8; ++l) {
                    const int idx = table[k][l];
                    const std::array<double,3> pt{ P[idx][0], P[idx][1], P[idx][2] };
                    staged[k][l] = find_or_add_node(pt);
                }
            const bool created_new = nodes.size() > before;
            bool keep = true;
            if (mode == DEL_IF_AVAIL)      keep = consume_at(ptmp);
            else if (mode == DEL_IF_NEW) { keep = created_new; if (keep) consume_at(ptmp); }
            else                           consume_at(ptmp);
            if (keep)
                for (int k = 0; k < nh; ++k) tmpl_cells.push_back(staged[k]);
        };

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (adj[i][j].count != 4) continue;

                const std::array<double,3>& ci = centres[i];
                // Sub-case parity must be computed in GRID-INDEX space (as the
                // reference does — it works in the octree's [0,1] grid that
                // starts at the origin).  Using world coordinates here is a bug:
                // they are offset by the bounding-box minimum and, for non-cubic
                // domains, scaled differently per axis, so the parity comes out
                // wrong and the template is fed the wrong cells.
                //
                // The reference loop `while(posI>0){posI-=delta; stepI=!stepI;}`
                // applied to a cell centre at (g+0.5)*delta toggles (g+1) times,
                // i.e. stepI == (g is even), where g is the cell's grid index at
                // its own level along that axis.
                const int gidx[3] = { leaves[i]->GetGridX(),
                                      leaves[i]->GetGridY(),
                                      leaves[i]->GetGridZ() };
                const bool sI = (gidx[xyz1[j]] % 2 == 0);
                const bool sJ = (gidx[xyz2[j]] % 2 == 0);
                if (!(sI && sJ)) continue;  // only the (even,even) cell anchors the template

                // IDs of the 4 smaller neighbours on face j, in the template's
                // (I,J) layout.  The face-adjacency probe fills ids[] in the
                // order  ids[0]=(Ilo,Jlo)  ids[1]=(Ihi,Jlo)  ids[2]=(Ilo,Jhi)
                // ids[3]=(Ihi,Jhi).  The template wants
                //   p0 = (Ilo,Jlo)   p1 = (Ihi,Jlo) = +I
                //   p4 = (Ilo,Jhi) = +J   p5 = (Ihi,Jhi) = +I+J
                // so p4/p5 come from ids[2]/ids[3].  (The reference numbers its
                // neighbours [LL,HL,HH,LH], which is why a naive ids[2]->p5,
                // ids[3]->p4 mapping silently swapped p4/p5 and inverted every
                // template hex.)
                const int sLL = adj[i][j].ids[0];  // (Ilo, Jlo)
                const int sHL = adj[i][j].ids[1];  // (Ihi, Jlo)  -> +I
                const int sLH = adj[i][j].ids[2];  // (Ilo, Jhi)  -> +J
                const int sHH = adj[i][j].ids[3];  // (Ihi, Jhi)  -> +I+J
                if (sLL<0 || sHL<0 || sLH<0 || sHH<0) continue;

                // Build the 32 reference points (p[0..31])
                double p[32][3];
                for (int d = 0; d < 3; ++d) {
                    p[0][d]  = centres[sLL][d];
                    p[1][d]  = centres[sHL][d];
                    p[2][d]  = 2*p[1][d]-p[0][d];
                    p[3][d]  = 2*p[2][d]-p[1][d];
                    p[4][d]  = centres[sLH][d];  // +J
                    p[5][d]  = centres[sHH][d];  // +I+J
                    p[6][d]  = 2*p[5][d]-p[4][d];
                    p[7][d]  = 2*p[6][d]-p[5][d];
                    p[8][d]  = 2*p[4][d]-p[0][d];
                    p[9][d]  = 2*p[5][d]-p[1][d];
                    p[10][d] = 2*p[5][d]-p[0][d];
                    p[11][d] = 2*p[10][d]-p[9][d];
                    p[12][d] = 2*p[8][d]-p[4][d];
                    p[13][d] = 2*p[9][d]-p[5][d];
                    p[14][d] = 2*p[13][d]-p[12][d];
                    p[15][d] = 2*p[14][d]-p[13][d];
                }
                // z: vector from face-centre to large cell centre
                double z[3];
                for (int d = 0; d < 3; ++d)
                    z[d] = ci[d] - 0.25*(p[0][d]+p[1][d]+p[4][d]+p[5][d]);

                for (int d = 0; d < 3; ++d) {
                    p[16][d] = p[1][d]+268.0/375*z[d]+(p[4][d]-p[0][d])*0.072;
                    p[17][d] = p[2][d]+268.0/375*z[d]+(p[4][d]-p[0][d])*0.072;
                    p[18][d] = ci[d];
                    p[19][d] = p[18][d]+2*(p[1][d]-p[0][d]);
                    p[20][d] = p[4][d]+268.0/375*z[d]+(p[1][d]-p[0][d])*0.072;
                    p[21][d] = p[5][d]+z[d]/5-(p[1][d]-p[0][d])*0.112+(p[4][d]-p[0][d])*0.056;
                    p[22][d] = p[6][d]+z[d]/5+(p[1][d]-p[0][d])*0.112+(p[4][d]-p[0][d])*0.056;
                    p[23][d] = p[7][d]+268.0/375*z[d]-(p[1][d]-p[0][d])*0.072;
                    p[24][d] = p[8][d]+268.0/375*z[d]+(p[1][d]-p[0][d])*0.072;
                    p[25][d] = p[9][d]+z[d]/5-(p[1][d]-p[0][d])*0.112-(p[4][d]-p[0][d])*0.056;
                    p[26][d] = p[10][d]+z[d]/5+(p[1][d]-p[0][d])*0.112-(p[4][d]-p[0][d])*0.056;
                    p[27][d] = p[11][d]+268.0/375*z[d]-(p[1][d]-p[0][d])*0.072;
                    p[28][d] = p[18][d]+2*(p[4][d]-p[0][d]);
                    p[29][d] = p[18][d]+2*(p[5][d]-p[0][d]);
                    p[30][d] = p[13][d]+268.0/375*z[d]-(p[4][d]-p[0][d])*0.072;
                    p[31][d] = p[14][d]+268.0/375*z[d]-(p[4][d]-p[0][d])*0.072;
                }

                // Template variant: stepI_tmpl = (j==1||j==3||j==5)
                const int tmpl = (j==1||j==3||j==5) ? 1 : 0;
                const int oj = 5 - j;  // opposite face

                // -- 13-element base template (always created) -----------------
                {
                    double ptmp[3];
                    for (int d = 0; d < 3; ++d)
                        ptmp[d] = 0.5*(p[21][d] + p[26][d]) + z[d]*4.0/15.0;
                    emit(p, t1Id[tmpl], 13, ptmp, DEL_ALWAYS);
                }

                // The sub-templates fill the edges and corners between adjacent
                // transition regions.  Each builds 16 working points p16[] from
                // the base points p[] and the up-axis z, optionally adjusts them
                // when the neighbouring region is itself a transition, then emits
                // its hexes.  Geometry and connectivity follow the reference
                // (HexGen.cpp DualFullHexMeshExtraction) verbatim.
                double p16[16][3];
                double ptmp[3];

                // -- 4-element template A (pSId[j][0] side) --------------------
                if (vcount(i, pSId[j][0]) == 1 && vcount(nb(i, pSId[j][0], 0), j) == 4) {
                    for (int d = 0; d < 3; ++d) {
                        p16[0][d]  = 2*p[0][d]-p[1][d];
                        p16[1][d]  = 2*p[18][d]-p[19][d];
                        p16[2][d]  = 2*p[4][d]-p[5][d];
                        p16[3][d]  = p[20][d]+1.144*(p[0][d]-p[1][d]);
                        p16[4][d]  = 2*p[8][d]-p[9][d];
                        p16[5][d]  = p[24][d]+1.144*(p[0][d]-p[1][d]);
                        p16[6][d]  = 2*p[28][d]-p[29][d];
                        p16[7][d]  = 2*p[12][d]-p[13][d];
                        p16[8][d]  = p[0][d];  p16[9][d]  = p[4][d];
                        p16[10][d] = p[8][d];  p16[11][d] = p[12][d];
                        p16[12][d] = p[18][d]; p16[13][d] = p[20][d];
                        p16[14][d] = p[24][d]; p16[15][d] = p[28][d];
                    }
                    for (int d = 0; d < 3; ++d) ptmp[d] = 0.5*(p16[2][d]+p[8][d]) + z[d]/3;
                    emit(p16, t2Id[tmpl], 4, ptmp, DEL_ALWAYS);
                }

                // -- 4-element template B (pSId[j][1] side) --------------------
                if (vcount(i, pSId[j][1]) == 1 && vcount(nb(i, pSId[j][1], 0), j) == 4) {
                    for (int d = 0; d < 3; ++d) {
                        p16[0][d]  = 2*p[0][d]-p[4][d];
                        p16[1][d]  = 2*p[18][d]-p[28][d];
                        p16[2][d]  = 2*p[1][d]-p[5][d];
                        p16[3][d]  = p[16][d]+1.144*(p[0][d]-p[4][d]);
                        p16[4][d]  = 2*p[2][d]-p[6][d];
                        p16[5][d]  = p[17][d]+1.144*(p[0][d]-p[4][d]);
                        p16[6][d]  = 2*p[19][d]-p[29][d];
                        p16[7][d]  = 2*p[3][d]-p[7][d];
                        p16[8][d]  = p[0][d];  p16[9][d]  = p[1][d];
                        p16[10][d] = p[2][d];  p16[11][d] = p[3][d];
                        p16[12][d] = p[16][d]; p16[13][d] = p[17][d];
                        p16[14][d] = p[18][d]; p16[15][d] = p[19][d];
                    }
                    for (int d = 0; d < 3; ++d) ptmp[d] = 0.5*(p16[4][d]+p[1][d]) + z[d]/3;
                    emit(p16, t22Id[tmpl], 4, ptmp, DEL_ALWAYS);
                }

                // -- 3-element template A (pSId[j][0] is itself a transition) --
                if (vcount(i, pSId[j][0]) == 4) {
                    for (int d = 0; d < 3; ++d) {
                        p16[0][d]  = 2*p[0][d]-p[1][d];
                        p16[1][d]  = p16[0][d]+2*z[d]/3;
                        p16[2][d]  = 2*p[4][d]-p[5][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[8][d]-p[9][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[6][d]  = 2*p[12][d]-p[13][d];
                        p16[7][d]  = p16[6][d]+2*z[d]/3;
                        p16[8][d]  = p[0][d];  p16[9][d]  = p[4][d];
                        p16[10][d] = p[8][d];  p16[11][d] = p[12][d];
                        p16[12][d] = p[18][d]; p16[13][d] = p[20][d];
                        p16[14][d] = p[24][d]; p16[15][d] = p[28][d];
                    }
                    const int sf = pSId[j][0];
                    bool far_trans = false;
                    for (int k = 0; k < 4; ++k)
                        if (vcount(nb(nb(i, sf, k), j, 0), oj) == 4) { far_trans = true; break; }
                    if (far_trans) for (int d = 0; d < 3; ++d) {
                        p16[0][d] = 2*p[18][d]-p[19][d]-4*z[d]/3;
                        p16[6][d] = 2*p[28][d]-p[29][d]-4*z[d]/3;
                        p16[2][d] = p16[5][d]+p[4][d]-p[24][d];
                        p16[4][d] = p16[5][d]+p[4][d]-p[20][d];
                    }
                    for (int d = 0; d < 3; ++d) ptmp[d] = 0.5*(p16[5][d]+p[4][d]);
                    emit(p16, t3Id[tmpl], 3, ptmp, DEL_IF_AVAIL);
                }

                // -- 3-element template B (pSId[j][1] is itself a transition) --
                if (vcount(i, pSId[j][1]) == 4) {
                    for (int d = 0; d < 3; ++d) {
                        p16[0][d]  = 2*p[0][d]-p[4][d];
                        p16[1][d]  = p16[0][d]+2*z[d]/3;
                        p16[2][d]  = 2*p[1][d]-p[5][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[2][d]-p[6][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[6][d]  = 2*p[3][d]-p[7][d];
                        p16[7][d]  = p16[6][d]+2*z[d]/3;
                        p16[8][d]  = p[0][d];  p16[9][d]  = p[1][d];
                        p16[10][d] = p[2][d];  p16[11][d] = p[3][d];
                        p16[12][d] = p[16][d]; p16[13][d] = p[17][d];
                        p16[14][d] = p[18][d]; p16[15][d] = p[19][d];
                    }
                    const int sf = pSId[j][1];
                    bool far_trans = false;
                    for (int k = 0; k < 4; ++k)
                        if (vcount(nb(nb(i, sf, k), j, 0), oj) == 4) { far_trans = true; break; }
                    if (far_trans) for (int d = 0; d < 3; ++d) {
                        p16[0][d] = 2*p[18][d]-p[28][d]-4*z[d]/3;
                        p16[6][d] = 2*p[19][d]-p[29][d]-4*z[d]/3;
                        p16[2][d] = p16[3][d]+p[2][d]-p[17][d];
                        p16[4][d] = p16[3][d]+p[2][d]-p[16][d];
                    }
                    for (int d = 0; d < 3; ++d) ptmp[d] = 0.5*(p16[3][d]+p[2][d]);
                    emit(p16, t32Id[tmpl], 3, ptmp, DEL_IF_AVAIL);
                }

                // -- 3-element template C (pS2Id[j][0] far side) ---------------
                if (vcount(nb(i, pS2Id[j][0], 0), pS2Id[j][0]) == 4) {
                    for (int d = 0; d < 3; ++d) {
                        p16[0][d]  = 2*p[3][d]-p[2][d];
                        p16[1][d]  = p16[0][d]+2*z[d]/3;
                        p16[2][d]  = 2*p[7][d]-p[6][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[11][d]-p[10][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[6][d]  = 2*p[15][d]-p[14][d];
                        p16[7][d]  = p16[6][d]+2*z[d]/3;
                        p16[8][d]  = p[3][d];  p16[9][d]  = p[7][d];
                        p16[10][d] = p[11][d]; p16[11][d] = p[15][d];
                        p16[12][d] = p[19][d]; p16[13][d] = p[23][d];
                        p16[14][d] = p[27][d]; p16[15][d] = p[29][d];
                    }
                    const int sf = pS2Id[j][0];
                    const int a  = nb(i, sf, 0);
                    bool far_trans = false;
                    for (int k = 0; k < 4; ++k)
                        if (vcount(nb(nb(a, sf, k), j, 0), oj) == 4) { far_trans = true; break; }
                    if (far_trans) for (int d = 0; d < 3; ++d) {
                        p16[0][d] = 2*p[19][d]-p[18][d]-4*z[d]/3;
                        p16[6][d] = 2*p[29][d]-p[28][d]-4*z[d]/3;
                        p16[2][d] = p16[3][d]+p[11][d]-p[27][d];
                        p16[4][d] = p16[3][d]+p[11][d]-p[23][d];
                    }
                    for (int d = 0; d < 3; ++d) ptmp[d] = 0.5*(p16[3][d]+p[11][d]);
                    emit(p16, t33Id[tmpl], 3, ptmp, DEL_IF_AVAIL);
                }

                // -- 3-element template D (pS2Id[j][1] far side) ---------------
                if (vcount(nb(i, pS2Id[j][1], 0), pS2Id[j][1]) == 4) {
                    for (int d = 0; d < 3; ++d) {
                        p16[0][d]  = 2*p[12][d]-p[8][d];
                        p16[1][d]  = p16[0][d]+2*z[d]/3;
                        p16[2][d]  = 2*p[13][d]-p[9][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[14][d]-p[10][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[6][d]  = 2*p[15][d]-p[11][d];
                        p16[7][d]  = p16[6][d]+2*z[d]/3;
                        p16[8][d]  = p[12][d]; p16[9][d]  = p[13][d];
                        p16[10][d] = p[14][d]; p16[11][d] = p[15][d];
                        p16[12][d] = p[28][d]; p16[13][d] = p[29][d];
                        p16[14][d] = p[30][d]; p16[15][d] = p[31][d];
                    }
                    const int sf = pS2Id[j][1];
                    const int a  = nb(i, sf, 0);
                    bool far_trans = false;
                    for (int k = 0; k < 4; ++k)
                        if (vcount(nb(nb(a, sf, k), j, 0), oj) == 4) { far_trans = true; break; }
                    if (far_trans) for (int d = 0; d < 3; ++d) {
                        p16[0][d] = 2*p[28][d]-p[18][d]-4*z[d]/3;
                        p16[6][d] = 2*p[29][d]-p[19][d]-4*z[d]/3;
                        p16[2][d] = p16[5][d]+p[13][d]-p[31][d];
                        p16[4][d] = p16[5][d]+p[13][d]-p[30][d];
                    }
                    for (int d = 0; d < 3; ++d) ptmp[d] = 0.5*(p16[5][d]+p[13][d]);
                    emit(p16, t34Id[tmpl], 3, ptmp, DEL_IF_AVAIL);
                }

                // -- 5-element template A (pSId[j][0] side, no further transition)
                if (vcount(i, pSId[j][0]) == 1 && vcount(nb(i, pSId[j][0], 0), j) == 1) {
                    for (int d = 0; d < 3; ++d) {
                        p16[1][d]  = 2*p[18][d]-p[19][d];
                        p16[0][d]  = p16[1][d]-4*z[d]/3;
                        p16[2][d]  = 2*p[4][d]-p[5][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[8][d]-p[9][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[7][d]  = 2*p[28][d]-p[29][d];
                        p16[6][d]  = p16[7][d]-4*z[d]/3;
                        p16[8][d]  = p[0][d];  p16[9][d]  = p[4][d];
                        p16[10][d] = p[8][d];  p16[11][d] = p[12][d];
                        p16[12][d] = p[18][d]; p16[13][d] = p[20][d];
                        p16[14][d] = p[24][d]; p16[15][d] = p[28][d];
                    }
                    for (int d = 0; d < 3; ++d) {
                        p16[2][d] += p[8][d]-p[24][d]+2*z[d]/3;
                        p16[4][d] += p[8][d]-p[24][d]+2*z[d]/3;
                        ptmp[d] = 0.5*(p16[2][d]+p[24][d]);
                    }
                    emit(p16, t4Id[tmpl], 5, ptmp, DEL_IF_NEW);
                }

                // -- 5-element template B (pSId[j][1] side) --------------------
                if (vcount(i, pSId[j][1]) == 1 && vcount(nb(i, pSId[j][1], 0), j) == 1) {
                    for (int d = 0; d < 3; ++d) {
                        p16[1][d]  = 2*p[18][d]-p[28][d];
                        p16[0][d]  = p16[1][d]-4*z[d]/3;
                        p16[2][d]  = 2*p[1][d]-p[5][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[2][d]-p[6][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[7][d]  = 2*p[19][d]-p[29][d];
                        p16[6][d]  = p16[7][d]-4*z[d]/3;
                        p16[8][d]  = p[0][d];  p16[9][d]  = p[1][d];
                        p16[10][d] = p[2][d];  p16[11][d] = p[3][d];
                        p16[12][d] = p[16][d]; p16[13][d] = p[17][d];
                        p16[14][d] = p[18][d]; p16[15][d] = p[19][d];
                    }
                    for (int d = 0; d < 3; ++d) {
                        p16[2][d] += p[1][d]-p[16][d]+2*z[d]/3;
                        p16[4][d] += p[1][d]-p[16][d]+2*z[d]/3;
                        ptmp[d] = 0.5*(p16[2][d]+p[17][d]);
                    }
                    emit(p16, t42Id[tmpl], 5, ptmp, DEL_IF_NEW);
                }

                // -- 5-element template C (pS2Id[j][0] far side) ---------------
                if (vcount(nb(i, pS2Id[j][0], 0), pS2Id[j][0]) == 1 &&
                    vcount(nb(nb(i, pS2Id[j][0], 0), pS2Id[j][0], 0), j) == 1) {
                    for (int d = 0; d < 3; ++d) {
                        p16[1][d]  = 2*p[19][d]-p[18][d];
                        p16[0][d]  = p16[1][d]-4*z[d]/3;
                        p16[2][d]  = 2*p[7][d]-p[6][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[11][d]-p[10][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[7][d]  = 2*p[29][d]-p[28][d];
                        p16[6][d]  = p16[7][d]-4*z[d]/3;
                        p16[8][d]  = p[3][d];  p16[9][d]  = p[7][d];
                        p16[10][d] = p[11][d]; p16[11][d] = p[15][d];
                        p16[12][d] = p[19][d]; p16[13][d] = p[23][d];
                        p16[14][d] = p[27][d]; p16[15][d] = p[29][d];
                    }
                    for (int d = 0; d < 3; ++d) {
                        p16[2][d] += p[7][d]-p[23][d]+2*z[d]/3;
                        p16[4][d] += p[7][d]-p[23][d]+2*z[d]/3;
                        ptmp[d] = 0.5*(p16[2][d]+p[27][d]);
                    }
                    emit(p16, t43Id[tmpl], 5, ptmp, DEL_IF_NEW);
                }

                // -- 5-element template D (pS2Id[j][1] far side) ---------------
                if (vcount(nb(i, pS2Id[j][1], 0), pS2Id[j][1]) == 1 &&
                    vcount(nb(nb(i, pS2Id[j][1], 0), pS2Id[j][1], 0), j) == 1) {
                    for (int d = 0; d < 3; ++d) {
                        p16[1][d]  = 2*p[28][d]-p[18][d];
                        p16[0][d]  = p16[1][d]-4*z[d]/3;
                        p16[2][d]  = 2*p[13][d]-p[9][d];
                        p16[3][d]  = p16[2][d]+2*z[d]/3;
                        p16[4][d]  = 2*p[14][d]-p[10][d];
                        p16[5][d]  = p16[4][d]+2*z[d]/3;
                        p16[7][d]  = 2*p[29][d]-p[19][d];
                        p16[6][d]  = p16[7][d]-4*z[d]/3;
                        p16[8][d]  = p[12][d]; p16[9][d]  = p[13][d];
                        p16[10][d] = p[14][d]; p16[11][d] = p[15][d];
                        p16[12][d] = p[28][d]; p16[13][d] = p[29][d];
                        p16[14][d] = p[30][d]; p16[15][d] = p[31][d];
                    }
                    for (int d = 0; d < 3; ++d) {
                        p16[2][d] += p[13][d]-p[30][d]+2*z[d]/3;
                        p16[4][d] += p[13][d]-p[30][d]+2*z[d]/3;
                        ptmp[d] = 0.5*(p16[2][d]+p[31][d]);
                    }
                    emit(p16, t44Id[tmpl], 5, ptmp, DEL_IF_NEW);
                }
            }
        }

        // -- 5a. Plain dual hexes: one per interior primal vertex (valence 8) --
        // idTransform maps the cell corner k touching the vertex to the dual-hex
        // node position that cell's centre occupies.  Vertices consumed by a
        // transition template above are skipped to avoid overlapping elements.
        for (int v = 0; v < NV; ++v) {
            if (consumed[v]) continue;
            if (vert_adj[v].size() != 8) continue;
            std::array<int,8> hex;
            hex.fill(-1);
            for (auto [ci, co] : vert_adj[v])
                hex[idTransform[co]] = ci;
            if (std::any_of(hex.begin(), hex.end(), [](int x){ return x < 0; })) continue;
            dual_cells.push_back(hex);
        }

        // ------------------------------------------------------------------ //
        // 6. Write VTK.  All hexes index into the single merged `nodes` array
        //    (dual node i == cell centre i == nodes[i]).
        // ------------------------------------------------------------------ //
        std::ofstream f(rFilename);
        KRATOS_ERROR_IF_NOT(f.is_open())
            << "OctreeHybridMeshUtility::WriteDualHexVtk: cannot open '"
            << rFilename << "'" << std::endl;

        const std::size_t n_nodes = nodes.size();
        const std::size_t n_cells = dual_cells.size() + tmpl_cells.size();

        f << "# vtk DataFile Version 2.0\n"
          << "OctreeHybrid dual hex mesh\n"
          << "ASCII\n"
          << "DATASET UNSTRUCTURED_GRID\n";

        f << "POINTS " << n_nodes << " double\n";
        f << std::scientific; f.precision(10);
        for (const auto& nd : nodes)
            f << nd[0] << ' ' << nd[1] << ' ' << nd[2] << '\n';

        f << "CELLS " << n_cells << ' ' << n_cells * 9 << '\n';
        for (const auto& h : dual_cells)
            f << "8 "<<h[0]<<' '<<h[1]<<' '<<h[2]<<' '<<h[3]
              <<' '<<h[4]<<' '<<h[5]<<' '<<h[6]<<' '<<h[7]<<'\n';
        for (const auto& h : tmpl_cells)
            f << "8 "<<h[0]<<' '<<h[1]<<' '<<h[2]<<' '<<h[3]
              <<' '<<h[4]<<' '<<h[5]<<' '<<h[6]<<' '<<h[7]<<'\n';

        f << "CELL_TYPES " << n_cells << '\n';
        for (std::size_t e = 0; e < n_cells; ++e) f << "12\n";

        // Cell-data: level of the cell whose centre is node 0 of each dual hex
        f << "CELL_DATA " << n_cells << '\n'
          << "SCALARS level int 1\nLOOKUP_TABLE default\n";
        for (const auto& h : dual_cells)
            f << leaves[h[0]]->GetLevel() << '\n';
        for (std::size_t e = 0; e < tmpl_cells.size(); ++e)
            f << "-1\n";  // template cells get level -1
    }

    /**
     * @brief Writes the raw primal octree cells as (non-conforming) hexahedra.
     *
     * Useful for debugging and for direct octree visualisation.
     * The cell-data field "level" encodes each cell's refinement level.
     */
    static void WritePrimalVtk(OctreeType& rOctree, const std::string& rFilename)
    {
        std::vector<CellType*> leaves;
        rOctree.GetAllLeavesVector(leaves);

        const std::size_t depth = rOctree.GetDepth();
        const std::size_t R    = std::size_t{1} << depth;
        const std::size_t pts  = R + 1;

        static constexpr int dx[8] = {0,1,1,0,0,1,1,0};
        static constexpr int dy[8] = {0,0,1,1,0,0,1,1};
        static constexpr int dz[8] = {0,0,0,0,1,1,1,1};

        std::unordered_map<std::size_t, std::size_t> node_map;
        std::vector<std::array<double,3>> node_coords;
        std::vector<std::array<std::size_t,8>> hex_conn;
        hex_conn.reserve(leaves.size());

        for (CellType* p : leaves) {
            const int lv = p->GetLevel();
            const int gx = p->GetGridX(), gy = p->GetGridY(), gz = p->GetGridZ();
            const std::size_t stride = std::size_t{1} << (depth - static_cast<std::size_t>(lv));
            std::array<std::size_t,8> conn{};
            for (int c = 0; c < 8; ++c) {
                const std::size_t ix = static_cast<std::size_t>(gx+dx[c])*stride;
                const std::size_t iy = static_cast<std::size_t>(gy+dy[c])*stride;
                const std::size_t iz = static_cast<std::size_t>(gz+dz[c])*stride;
                const std::size_t key = iz*pts*pts + iy*pts + ix;
                auto [it, ins] = node_map.emplace(key, node_coords.size());
                if (ins) {
                    const double sc = 1.0/static_cast<double>(R);
                    double nm[3] = {ix*sc, iy*sc, iz*sc}, wm[3];
                    rOctree.ScaleBackToOriginalCoordinate(nm, wm);
                    node_coords.push_back({wm[0],wm[1],wm[2]});
                }
                conn[c] = it->second;
            }
            hex_conn.push_back(conn);
        }

        std::ofstream f(rFilename);
        const std::size_t nc = hex_conn.size();
        f << "# vtk DataFile Version 2.0\nOctreeHybrid primal cells\nASCII\n"
          << "DATASET UNSTRUCTURED_GRID\n"
          << "POINTS " << node_coords.size() << " double\n";
        f << std::scientific; f.precision(10);
        for (auto& pt : node_coords) f << pt[0]<<' '<<pt[1]<<' '<<pt[2]<<'\n';
        f << "CELLS "<<nc<<' '<<nc*9<<'\n';
        for (auto& h : hex_conn) f<<"8 "<<h[0]<<' '<<h[1]<<' '<<h[2]<<' '<<h[3]
                                   <<' '<<h[4]<<' '<<h[5]<<' '<<h[6]<<' '<<h[7]<<'\n';
        f << "CELL_TYPES "<<nc<<'\n';
        for (std::size_t e = 0; e < nc; ++e) f << "12\n";
        f << "CELL_DATA "<<nc<<"\nSCALARS level int 1\nLOOKUP_TABLE default\n";
        for (CellType* p : leaves) f << p->GetLevel() << '\n';
    }

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
        std::size_t RefinementDepth = 5)
    {
        auto p_octree = BuildFromSurfaceMesh(rSurfaceMesh, RefinementDepth);
        WriteDualHexVtk(*p_octree, rVtkFilename);
    }

    /**
     * @brief Debug/validation helper: writes the strongly-balanced octree leaves
     *        in the exact VTK format the reference HybridOctree_Hex expects from
     *        its `ReadOctree` (so its dual extraction can be run on the identical
     *        octree and the two tilings compared).
     *
     * Vertex coordinates are written as (integer grid index) * (100 / 2^depth):
     * the reference forces START_POINT = 0 and BOX_LENGTH_RATIO = 100/voxelSize
     * with voxelSize = 2^depth, so `round(coord / RATIO)` recovers the integer
     * grid index.  Corner order matches the reference (0 = min, 6 = max).
     *
     * The octree is built and balanced exactly as in BuildAndWriteVtk, so the
     * leaves coincide with those used by the dual mesh written there.
     */
    static void WriteOctreeForReference(
        ModelPart& rSurfaceMesh,
        const std::string& rFilename,
        std::size_t RefinementDepth)
    {
        auto p_octree = BuildFromSurfaceMesh(rSurfaceMesh, RefinementDepth);
        p_octree->StrongConstrain2To1();

        std::vector<CellType*> leaves;
        p_octree->GetAllLeavesVector(leaves);

        const std::size_t depth = p_octree->GetDepth();
        const std::size_t R   = std::size_t{1} << depth;
        const std::size_t pts = R + 1;
        const double sc = 100.0 / static_cast<double>(R);  // reference RATIO

        static constexpr int dx[8] = {0,1,1,0,0,1,1,0};
        static constexpr int dy[8] = {0,0,1,1,0,0,1,1};
        static constexpr int dz[8] = {0,0,0,0,1,1,1,1};

        std::unordered_map<std::size_t, std::size_t> node_map;
        std::vector<std::array<double,3>> coords;
        std::vector<std::array<std::size_t,8>> conn;
        conn.reserve(leaves.size());

        for (CellType* c : leaves) {
            const int lv = c->GetLevel();
            const int gx = c->GetGridX(), gy = c->GetGridY(), gz = c->GetGridZ();
            const std::size_t stride = std::size_t{1} << (depth - static_cast<std::size_t>(lv));
            std::array<std::size_t,8> e{};
            for (int k = 0; k < 8; ++k) {
                const std::size_t ix = static_cast<std::size_t>(gx+dx[k])*stride;
                const std::size_t iy = static_cast<std::size_t>(gy+dy[k])*stride;
                const std::size_t iz = static_cast<std::size_t>(gz+dz[k])*stride;
                const std::size_t key = iz*pts*pts + iy*pts + ix;
                auto [it, ins] = node_map.emplace(key, coords.size());
                if (ins) coords.push_back({ ix*sc, iy*sc, iz*sc });
                e[k] = it->second;
            }
            conn.push_back(e);
        }

        std::ofstream f(rFilename);
        const std::size_t nc = conn.size();
        f << "# vtk DataFile Version 2.0\nOctreeHybrid leaves (reference format)\nASCII\n"
          << "DATASET UNSTRUCTURED_GRID\n"
          << "POINTS " << coords.size() << " double\n";
        f << std::scientific; f.precision(10);
        for (auto& p : coords) f << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        f << "CELLS " << nc << ' ' << nc*9 << '\n';
        for (auto& e : conn) f << "8 "<<e[0]<<' '<<e[1]<<' '<<e[2]<<' '<<e[3]
                               <<' '<<e[4]<<' '<<e[5]<<' '<<e[6]<<' '<<e[7]<<'\n';
        f << "CELL_TYPES " << nc << '\n';
        for (std::size_t i = 0; i < nc; ++i) f << "12\n";
    }

    ///@}
};

} // namespace Kratos
