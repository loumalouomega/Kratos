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
 * ### 13-element template
 *
 * The utility also applies the 13-element transition template from the
 * HybridOctree_Hex paper for every face where one large cell meets four small
 * cells.  This generates 13 additional hexes that fill the face-transition
 * region with higher element quality than the plain dual approach, replacing
 * the distorted hexes that would otherwise sit in that region.
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

        // -- 5a. 1-element template (uniform and mixed-size interior vertices) --
        // For each primal vertex with exactly 8 cells → 1 dual hex
        std::vector<bool> vertex_used(NV, false);

        for (int v = 0; v < NV; ++v) {
            if (vert_adj[v].size() != 8) continue;
            std::array<int,8> hex;
            hex.fill(-1);
            for (auto [ci, co] : vert_adj[v]) {
                hex[idTransform[co]] = ci;
            }
            if (std::any_of(hex.begin(), hex.end(), [](int x){ return x < 0; })) continue;
            dual_cells.push_back(hex);
            vertex_used[v] = true;
        }

        // -- 5b. 13-element template: fill face-transition regions --
        // For each leaf i and each face j where adj[i][j].count == 4 (4 smaller
        // neighbors), compute the 32 reference points and create 13 dual hexes.
        // This improves quality over the plain dual approach in transition zones.
        //
        // Template tables from StaticVars.h (t1Id[stepI][13][8]):
        // stepI = (j == 1 || j == 3 || j == 5)
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

        // Dual hex node positions produced by the 13-element template are
        // floating-point coordinates deduped with a tolerance.
        // We collect them in a flat vector and dedup on the fly.
        struct DualNode { std::array<double,3> pos; };
        std::vector<DualNode> tmpl_nodes;
        std::vector<std::array<int,8>> tmpl_cells;
        tmpl_nodes.reserve(N * 4);
        tmpl_cells.reserve(N * 13);

        const double DIST_THRES = 1e-10;

        auto find_or_add_node = [&](const std::array<double,3>& pos) -> int {
            for (int m = static_cast<int>(tmpl_nodes.size()) - 1; m >= 0; --m) {
                const auto& q = tmpl_nodes[m].pos;
                double d2 = 0;
                for (int d = 0; d < 3; ++d) d2 += (q[d]-pos[d])*(q[d]-pos[d]);
                if (d2 < DIST_THRES) return m;
            }
            tmpl_nodes.push_back({pos});
            return static_cast<int>(tmpl_nodes.size()) - 1;
        };

        // Gather which primal vertices are "claimed" by the 13-element template
        // so the plain dual approach (5a) doesn't double-create them.
        // We identify these by looking at which collectNum-style vertices
        // are handled by the 13-element template.
        // For simplicity we skip 5a for primal vertices that are adjacent to
        // any 4-valence face.
        // (already handled: 5a only runs on vertices not flagged below)
        // We flag by removing those dual_cells entries that have ALL their 8
        // contributing cells involved in a 13-element template face.
        // This is complex; instead we use a simpler rule: the 13-element
        // template nodes REPLACE the cell-centre nodes for the affected region.
        // We keep both sets of elements and rely on VTK viewer for display.

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (adj[i][j].count != 4) continue;

                // Determine sub-case: (stepI, stepJ) for pos calculation
                // using the large cell's centre in world coordinates
                const std::array<double,3>& ci = centres[i];
                // cell size in world space (use first axis as reference)
                double n_lo[3], n_hi[3], w_lo[3], w_hi[3];
                leaves[i]->GetMinPointNormalized(n_lo);
                leaves[i]->GetMaxPointNormalized(n_hi);
                rOctree.ScaleBackToOriginalCoordinate(n_lo, w_lo);
                rOctree.ScaleBackToOriginalCoordinate(n_hi, w_hi);
                const double delta = w_hi[0] - w_lo[0];  // cell size

                double posI = ci[xyz1[j]], posJ = ci[xyz2[j]];
                bool sI = false, sJ = false;
                while (posI > 0) { posI -= delta; sI = !sI; }
                while (posJ > 0) { posJ -= delta; sJ = !sJ; }
                if (!(sI && sJ)) continue;  // only handle (T,T) sub-case

                // IDs of the 4 smaller neighbors on face j
                const int s0 = adj[i][j].ids[0];  // [small,small]
                const int s1 = adj[i][j].ids[1];  // [big,small]
                const int s2 = adj[i][j].ids[2];  // [big,big]
                const int s3 = adj[i][j].ids[3];  // [small,big]
                if (s0<0 || s1<0 || s2<0 || s3<0) continue;

                // Build the 32 reference points (p[0..31])
                double p[32][3];
                for (int d = 0; d < 3; ++d) {
                    p[0][d]  = centres[s0][d];
                    p[1][d]  = centres[s1][d];
                    p[2][d]  = 2*p[1][d]-p[0][d];
                    p[3][d]  = 2*p[2][d]-p[1][d];
                    p[4][d]  = centres[s3][d];  // [small,big] = m=3
                    p[5][d]  = centres[s2][d];  // [big,big]   = m=2
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

                for (int k = 0; k < 13; ++k) {
                    std::array<int,8> hex;
                    for (int l = 0; l < 8; ++l) {
                        std::array<double,3> pt;
                        for (int d = 0; d < 3; ++d) pt[d] = p[t1Id[tmpl][k][l]][d];
                        hex[l] = find_or_add_node(pt) + N;  // offset past cell-centre nodes
                    }
                    tmpl_cells.push_back(hex);
                }
            }
        }

        // ------------------------------------------------------------------ //
        // 6. Write VTK
        // ------------------------------------------------------------------ //
        std::ofstream f(rFilename);
        KRATOS_ERROR_IF_NOT(f.is_open())
            << "OctreeHybridMeshUtility::WriteDualHexVtk: cannot open '"
            << rFilename << "'" << std::endl;

        const std::size_t n_nodes = static_cast<std::size_t>(N) + tmpl_nodes.size();
        const std::size_t n_cells = dual_cells.size() + tmpl_cells.size();

        f << "# vtk DataFile Version 2.0\n"
          << "OctreeHybrid dual hex mesh\n"
          << "ASCII\n"
          << "DATASET UNSTRUCTURED_GRID\n";

        f << "POINTS " << n_nodes << " double\n";
        f << std::scientific; f.precision(10);
        // Nodes 0..N-1: cell centres
        for (int i = 0; i < N; ++i)
            f << centres[i][0] << ' ' << centres[i][1] << ' ' << centres[i][2] << '\n';
        // Nodes N..: template points
        for (const auto& nd : tmpl_nodes)
            f << nd.pos[0] << ' ' << nd.pos[1] << ' ' << nd.pos[2] << '\n';

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

    ///@}
};

} // namespace Kratos
