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
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Project includes
#include "modeler/utilities/octree_hybrid_mesh_utility.h"
#include "includes/model_part.h"
#include "utilities/parallel_utilities.h"

namespace Kratos {

namespace {

constexpr double C_THRES[5] = { 0.0, 0.0, 0.4, 0.8, 1.6 };
constexpr double H_THRES[5] = { 16.0, 8.0, 4.0, 2.0, 1.0 };
constexpr double CELL_DETECT = 1.0;

struct AdaptiveRefineData {
    double cube_lo[3] = {0,0,0};
    double cube_side  = 0.0;
    std::vector<Geometry<Node>*> tri_geom;
    std::array<std::vector<int>,5> refine_tri;
};

AdaptiveRefineData BuildRefineSets(ModelPart& rSurfaceMesh)
{
    constexpr double PI = 3.1415926535897932384626433;
    AdaptiveRefineData data;

    // --- Gather triangle corners (world coords) and the bounding box ------
    std::vector<std::array<double,3>> corners;       // 3 per triangle
    corners.reserve(rSurfaceMesh.NumberOfGeometries() * 3);
    double lo[3] = { std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max() };
    double hi[3] = { std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest() };
    for (auto& g : rSurfaceMesh.Geometries()) {
        if (g.PointsNumber() < 3) continue;
        for (int k = 0; k < 3; ++k) {
            const double x = g[k].X(), y = g[k].Y(), z = g[k].Z();
            corners.push_back({x,y,z});
            lo[0]=std::min(lo[0],x); hi[0]=std::max(hi[0],x);
            lo[1]=std::min(lo[1],y); hi[1]=std::max(hi[1],y);
            lo[2]=std::min(lo[2],z); hi[2]=std::max(hi[2],z);
        }
        data.tri_geom.push_back(&g);
    }
    const int nTri = static_cast<int>(data.tri_geom.size());
    if (nTri == 0) return data;

    // --- Reference cube: centred, side = largest extent (START_POINT/BOX_LENGTH)
    double L = hi[0]-lo[0];
    L = std::max(L, hi[1]-lo[1]);
    L = std::max(L, hi[2]-lo[2]);
    for (int d = 0; d < 3; ++d) data.cube_lo[d] = 0.5*(lo[d]+hi[d]-L);
    data.cube_side = L;

    // --- Merge coincident corners into unique vertices (100-unit space) ---
    const double tol = 1e-6 * (L > 0.0 ? L : 1.0);
    std::map<std::array<long long,3>, int> vmap;
    std::vector<std::array<double,3>> v;             // merged, normalised
    std::vector<std::array<int,3>> e(nTri);          // triangle -> vertex ids
    for (int i = 0; i < nTri; ++i)
        for (int k = 0; k < 3; ++k) {
            const auto& c = corners[3*i + k];
            const std::array<long long,3> key{
                std::llround(c[0]/tol), std::llround(c[1]/tol), std::llround(c[2]/tol) };
            auto it = vmap.find(key);
            int idx;
            if (it == vmap.end()) {
                idx = static_cast<int>(v.size());
                vmap.emplace(key, idx);
                v.push_back({ (c[0]-data.cube_lo[0])*100.0/L,
                              (c[1]-data.cube_lo[1])*100.0/L,
                              (c[2]-data.cube_lo[2])*100.0/L });
            } else idx = it->second;
            e[i][k] = idx;
        }
    const int nV = static_cast<int>(v.size());

    auto cross = [](const double a[3], const double b[3], double o[3]) {
        o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0];
    };
    auto dot = [](const double a[3], const double b[3]) {
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    };

    // --- Per-vertex curvature: sum of squared dihedral-angle deviations ----
    std::vector<double> r(nV, 0.0);
    int pub[2];
    for (int j = 0; j < nTri-1; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = j+1; l < nTri; ++l)
          for (int m = 0; m < 3; ++m)
            if (e[l][m] == e[j][k]) {
                if      (e[j][(k+1)%3]==e[l][(m+1)%3]) { pub[0]=1; pub[1]=1; }
                else if (e[j][(k+1)%3]==e[l][(m+2)%3]) { pub[0]=1; pub[1]=2; }
                else if (e[j][(k+2)%3]==e[l][(m+1)%3]) { pub[0]=2; pub[1]=1; }
                else if (e[j][(k+2)%3]==e[l][(m+2)%3]) { pub[0]=2; pub[1]=2; }
                else continue;
                const auto& P  = v[e[j][k]];
                const auto& Q1 = v[e[j][(k+3-pub[0])%3]];
                const auto& Q2 = v[e[l][(m+3-pub[1])%3]];
                const auto& Pe = v[e[j][(k+pub[0])%3]];
                const double l1[3]={Q1[0]-P[0],Q1[1]-P[1],Q1[2]-P[2]};
                const double l2[3]={Q2[0]-P[0],Q2[1]-P[1],Q2[2]-P[2]};
                const double lp[3]={Pe[0]-P[0],Pe[1]-P[1],Pe[2]-P[2]};
                double c1[3], c2[3]; cross(l1,lp,c1); cross(l2,lp,c2);
                double ang = dot(c1,c2)/std::sqrt(dot(c1,c1)*dot(c2,c2));
                ang = (ang >= -1) ? ang : -1;
                ang = (ang <=  1) ? std::acos(ang) : 0;
                r[e[j][k]] += (ang-PI)*(ang-PI);
                break;
            }

    // --- Build the five nested refine-triangle sets -----------------------
    auto& R = data.refine_tri;
    for (int i = 0; i < nTri; ++i) {
        // curvature criterion (per vertex of the triangle)
        for (int j = 0; j < 3; ++j) {
            const double cv = r[e[i][j]];
            if (cv > C_THRES[0]) { R[0].push_back(i);
              if (cv > C_THRES[1]) { R[1].push_back(i);
                if (cv > C_THRES[2]) { R[2].push_back(i);
                  if (cv > C_THRES[3]) { R[3].push_back(i);
                    if (cv > C_THRES[4]) { R[4].push_back(i); }}}}}
        }
        // thickness criterion (normal-ray cast to the opposite sheet)
        const auto& A = v[e[i][0]]; const auto& B = v[e[i][1]]; const auto& C = v[e[i][2]];
        const double ab[3]={B[0]-A[0],B[1]-A[1],B[2]-A[2]};
        const double ac[3]={C[0]-A[0],C[1]-A[1],C[2]-A[2]};
        double dir[3]; cross(ab,ac,dir);
        const double nl = std::sqrt(dot(dir,dir));
        if (nl <= 0.0) continue;
        dir[0]/=nl; dir[1]/=nl; dir[2]/=nl;
        const double cen[3]={ (A[0]+B[0]+C[0])/3, (A[1]+B[1]+C[1])/3, (A[2]+B[2]+C[2])/3 };
        const double mc = std::max(std::max(std::abs(dir[0]),std::abs(dir[1])),std::abs(dir[2]));
        for (int j = i+1; j < nTri; ++j) {
            double hitp[3], alpha = 0.0;
            const int hit = OctreeHybridMeshUtility::TriRayIntersect(v[e[j][0]].data(), v[e[j][1]].data(), v[e[j][2]].data(),
                                            cen, dir, hitp, alpha);
            if (hit != 1) continue;
            const double len = mc * std::abs(alpha);
            if (len < H_THRES[0]) { R[0].push_back(i); R[0].push_back(j);
              if (len < H_THRES[1]) { R[1].push_back(i); R[1].push_back(j);
                if (len < H_THRES[2]) { R[2].push_back(i); R[2].push_back(j);
                  if (len < H_THRES[3]) { R[3].push_back(i); R[3].push_back(j);
                    if (len < H_THRES[4]) { R[4].push_back(i); R[4].push_back(j); }}}}}
        }
    }
    for (auto& s : R) { std::sort(s.begin(), s.end()); s.erase(std::unique(s.begin(), s.end()), s.end()); }
    return data;
}

} // anonymous namespace

/***********************************************************************************/
/***********************************************************************************/

auto OctreeHybridMeshUtility::BuildFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth,
    bool Adaptive) -> std::unique_ptr<OctreeType>
{
    KRATOS_ERROR_IF(RefinementDepth < 1 || RefinementDepth > ConfigurationType::MAX_DEPTH)
        << "OctreeHybridMeshUtility: RefinementDepth must be in [1, "
        << ConfigurationType::MAX_DEPTH << "], got " << RefinementDepth << std::endl;

    if (Adaptive)
        return BuildAdaptiveFromSurfaceMesh(rSurfaceMesh, RefinementDepth);

    // ----------------------------------------------------------------- //
    //  Uniform refinement (legacy path): every leaf whose box intersects
    //  any triangle is split to RefinementDepth.  Domain is the 1 %-padded
    //  axis-aligned bounding box.  Kept for the transition-template unit
    //  tests, whose synthetic flat patches carry no curvature and so would
    //  not refine under the adaptive criterion.
    // ----------------------------------------------------------------- //
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

/***********************************************************************************/
/***********************************************************************************/

auto OctreeHybridMeshUtility::BuildAdaptiveFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth) -> std::unique_ptr<OctreeType>
{
    const AdaptiveRefineData data = BuildRefineSets(rSurfaceMesh);
    KRATOS_ERROR_IF(data.tri_geom.empty())
        << "OctreeHybridMeshUtility: surface ModelPart has no triangles." << std::endl;

    const double clo[3] = { data.cube_lo[0], data.cube_lo[1], data.cube_lo[2] };
    const double chi[3] = { data.cube_lo[0] + data.cube_side,
                            data.cube_lo[1] + data.cube_side,
                            data.cube_lo[2] + data.cube_side };
    auto p_octree = std::make_unique<OctreeType>(RefinementDepth);
    p_octree->SetBoundingBox(clo, chi);

    // --- Level-3 "refine everything near the surface" grid (8^3) --------
    // pass3[i][j][k] = a level-3 cell that the reference would subdivide;
    // its ancestors at levels 0..2 are subdivided iff their subtree holds one
    // (the reference's bottom-up "child marked -> parent marked" rule).
    const double cs3 = data.cube_side / 8.0;
    std::array<bool, 8*8*8> pass3{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 8; ++k) {
                const double cen[3] = { clo[0]+(i+0.5)*cs3, clo[1]+(j+0.5)*cs3, clo[2]+(k+0.5)*cs3 };
                const Point blo(cen[0]-CELL_DETECT*cs3, cen[1]-CELL_DETECT*cs3, cen[2]-CELL_DETECT*cs3);
                const Point bhi(cen[0]+CELL_DETECT*cs3, cen[1]+CELL_DETECT*cs3, cen[2]+CELL_DETECT*cs3);
                bool hit = false;
                for (GeometryType* g : data.tri_geom)
                    if (g->HasIntersection(blo, bhi)) { hit = true; break; }
                pass3[(i*8+j)*8+k] = hit;
            }

    auto any_pass3_in_block = [&](int gi, int gj, int gk, int L) -> bool {
        const int span = 1 << (3 - L);                 // L < 3
        const int i0 = gi*span, j0 = gj*span, k0 = gk*span;
        for (int i = i0; i < i0+span && i < 8; ++i)
            for (int j = j0; j < j0+span && j < 8; ++j)
                for (int k = k0; k < k0+span && k < 8; ++k)
                    if (pass3[(i*8+j)*8+k]) return true;
        return false;
    };

    // Per-cell refinement test (reference ComputeCellValue geometry branch).
    auto should_sub = [&](CellType* p_cell, int L, int gi, int gj, int gk) -> bool {
        if (L >= 4) {
            const int idx = L - 4;
            if (idx > 4) return false;
            double n_lo2[3], n_hi[3], w_lo[3], w_hi[3];
            p_cell->GetMinPointNormalized(n_lo2);
            p_cell->GetMaxPointNormalized(n_hi);
            p_octree->ScaleBackToOriginalCoordinate(n_lo2, w_lo);
            p_octree->ScaleBackToOriginalCoordinate(n_hi, w_hi);
            const double cs = w_hi[0] - w_lo[0];
            const double cen[3] = { 0.5*(w_lo[0]+w_hi[0]), 0.5*(w_lo[1]+w_hi[1]), 0.5*(w_lo[2]+w_hi[2]) };
            const Point blo(cen[0]-CELL_DETECT*cs, cen[1]-CELL_DETECT*cs, cen[2]-CELL_DETECT*cs);
            const Point bhi(cen[0]+CELL_DETECT*cs, cen[1]+CELL_DETECT*cs, cen[2]+CELL_DETECT*cs);
            for (int t : data.refine_tri[idx])
                if (data.tri_geom[t]->HasIntersection(blo, bhi)) return true;
            return false;
        }
        if (L == 3) return pass3[(gi*8+gj)*8+gk];
        return any_pass3_in_block(gi, gj, gk, L);
    };

    // --- Refinement passes (top-down, one octree level per pass) --------
    // The reference refines in complete sibling octets: ComputeCellValue marks
    // all eight children of a cell subdivided as soon as any one of them is.
    // So a parent octet is refined iff any of its members would be, and then
    // every member subdivides.
    for (std::size_t iter = 0; iter < RefinementDepth; ++iter) {
        std::vector<CellType*> leaves;
        p_octree->GetAllLeavesVector(leaves);

        std::set<std::array<int,4>> octet_refine;       // (level, parent gi,gj,gk)
        std::vector<std::pair<CellType*,std::array<int,4>>> cand;
        cand.reserve(leaves.size());
        for (CellType* p_cell : leaves) {
            const int L = p_cell->GetLevel();
            if (static_cast<std::size_t>(L) >= RefinementDepth) continue;
            double n_lo[3];
            p_cell->GetMinPointNormalized(n_lo);
            const int gi = static_cast<int>(std::llround(n_lo[0] * (1 << L)));
            const int gj = static_cast<int>(std::llround(n_lo[1] * (1 << L)));
            const int gk = static_cast<int>(std::llround(n_lo[2] * (1 << L)));
            const std::array<int,4> key{ L, gi>>1, gj>>1, gk>>1 };
            cand.emplace_back(p_cell, key);
            if (should_sub(p_cell, L, gi, gj, gk)) octet_refine.insert(key);
        }

        bool any_split = false;
        for (auto& pc : cand) {
            if (!octet_refine.count(pc.second)) continue;
            p_octree->SubdivideCellByIdAndLevel(pc.first->GetId(), pc.second[0]);
            any_split = true;
        }
        if (!any_split) break;
    }
    return p_octree;
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::ExtractDualHexMesh(
    OctreeType& rOctree,
    std::vector<std::array<double,3>>& rNodes,
    std::vector<std::array<int,8>>& rCells,
    std::vector<int>& rCellLevel)
{
    // ------------------------------------------------------------------ //
    // 1. Collect leaves (the octree must already be 2:1-balanced)
    // ------------------------------------------------------------------ //
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
                // Fill opposite face only for a SAME-SIZE neighbour.  If
                // `first` is coarser (i is one of its 4 smaller children on
                // this side), its opposite face genuinely has valence 4 — we
                // must NOT stamp count=1 on it here, or it would be skipped
                // (line "filled from opposite side") and never detect its 4
                // smaller neighbours, dropping that whole transition template.
                if (leaves[i]->GetLevel() == leaves[first]->GetLevel() &&
                    adj[first][opp].count == 0) {
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
                    // Reference erase point is 0.5*(p21 + p26 + z*4/15),
                    // i.e. 0.5*(p21+p26) + z*2/15 — NOT z*4/15.  The doubled
                    // z-offset only mis-rounds (and leaves the centre dual hex
                    // unconsumed) once cells are large in finest-grid units,
                    // so it surfaced only at depth >= 6.
                    ptmp[d] = 0.5*(p[21][d] + p[26][d] + z[d]*4.0/15.0);
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
    // 6. Combine, optionally carve, and write.  All hexes index into the
    //    single merged `nodes` array (dual node i == cell centre i).
    // ------------------------------------------------------------------ //

    // Merge the plain-dual and template hexes into one element list, and
    // record each cell's "level" for VTK colouring (the level of the leaf
    // whose centre is node 0 for plain-dual hexes; -1 for template hexes).
    // The level MUST be captured here, before any carving re-indexes the
    // connectivity, because it is keyed by leaf index.
    std::vector<std::array<int,8>> cells;
    std::vector<int> cell_level;
    cells.reserve(dual_cells.size() + tmpl_cells.size());
    cell_level.reserve(dual_cells.size() + tmpl_cells.size());
    for (const auto& h : dual_cells) {
        cells.push_back(h);
        cell_level.push_back(leaves[h[0]]->GetLevel());
    }
    for (const auto& h : tmpl_cells) {
        cells.push_back(h);
        cell_level.push_back(-1);
    }

    rNodes     = std::move(nodes);
    rCells     = std::move(cells);
    rCellLevel = std::move(cell_level);
}

/***********************************************************************************/
/***********************************************************************************/

OctreeHybridMeshUtility::TriangleSoup OctreeHybridMeshUtility::ExtractTriangleSoup(const ModelPart& rSurfaceMesh)
{
    TriangleSoup triangles;
    triangles.reserve(rSurfaceMesh.NumberOfGeometries());
    for (const auto& r_geom : rSurfaceMesh.Geometries()) {
        if (r_geom.PointsNumber() < 3) continue;
        triangles.push_back({{
            {{ r_geom[0].X(), r_geom[0].Y(), r_geom[0].Z() }},
            {{ r_geom[1].X(), r_geom[1].Y(), r_geom[1].Z() }},
            {{ r_geom[2].X(), r_geom[2].Y(), r_geom[2].Z() }}
        }});
    }
    return triangles;
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::ExtractPrimalHexMesh(
    OctreeType& rOctree,
    std::vector<std::array<double,3>>& rNodes,
    std::vector<std::array<int,8>>&   rCells,
    std::vector<int>&                 rCellLevel,
    std::vector<HangingConstraint>&   rHanging)
{
    // ------------------------------------------------------------------ //
    // 1. Collect leaves (already 2:1-balanced)
    // ------------------------------------------------------------------ //
    std::vector<CellType*> leaves;
    rOctree.GetAllLeavesVector(leaves);
    const int N = static_cast<int>(leaves.size());
    const std::size_t depth = rOctree.GetDepth();
    const std::size_t R = std::size_t{1} << depth;
    const std::size_t pts = R + 1;

    static constexpr int PCX[8] = {0,1,1,0,0,1,1,0};
    static constexpr int PCY[8] = {0,0,1,1,0,0,1,1};
    static constexpr int PCZ[8] = {0,0,0,0,1,1,1,1};

    // ------------------------------------------------------------------ //
    // 2. Build primal vertex map and world-space positions
    // ------------------------------------------------------------------ //
    std::unordered_map<std::size_t,int> vid_map;
    vid_map.reserve(N * 4);
    std::vector<std::array<double,3>> primal_nodes;
    primal_nodes.reserve(N * 2);
    std::vector<std::array<int,8>> primal_elem(N);

    for (int i = 0; i < N; ++i) {
        const auto* p = leaves[i];
        const int lv = p->GetLevel();
        const int gx = p->GetGridX(), gy = p->GetGridY(), gz = p->GetGridZ();
        const std::size_t stride = std::size_t{1} << (depth - static_cast<std::size_t>(lv));
        for (int c = 0; c < 8; ++c) {
            const std::size_t ix = static_cast<std::size_t>(gx + PCX[c]) * stride;
            const std::size_t iy = static_cast<std::size_t>(gy + PCY[c]) * stride;
            const std::size_t iz = static_cast<std::size_t>(gz + PCZ[c]) * stride;
            const std::size_t key = iz * pts * pts + iy * pts + ix;
            auto [it, ins] = vid_map.emplace(key, static_cast<int>(primal_nodes.size()));
            if (ins) {
                double npt[3] = { static_cast<double>(ix) / static_cast<double>(R),
                                  static_cast<double>(iy) / static_cast<double>(R),
                                  static_cast<double>(iz) / static_cast<double>(R) };
                std::array<double,3> w;
                rOctree.ScaleBackToOriginalCoordinate(npt, w.data());
                primal_nodes.push_back(w);
            }
            primal_elem[i][c] = it->second;
        }
    }

    // ------------------------------------------------------------------ //
    // 3. Face adjacency (2:1 transition detection) — same algorithm as in
    //    ExtractDualHexMesh (see inline comments there).
    // ------------------------------------------------------------------ //
    std::unordered_map<int,int> id_to_idx;
    id_to_idx.reserve(N);
    for (int i = 0; i < N; ++i) id_to_idx[leaves[i]->GetId()] = i;

    static constexpr int P_FACE_FIXED[6] = {2,1,0,0,1,2};
    static constexpr bool P_FACE_HI[6]   = {false,false,false,true,true,true};
    static constexpr int P_FACE_FREE1[6] = {0,0,1,1,0,0};
    static constexpr int P_FACE_FREE2[6] = {1,2,2,2,2,1};
    static constexpr double PEPS = 1e-9;
    static constexpr double PQ[2] = {0.25, 0.75};

    struct PFaceAdj { int count; std::array<int,4> ids; };
    std::vector<std::array<PFaceAdj,6>> adj(N);
    for (auto& a : adj) for (auto& f : a) { f.count = 0; f.ids.fill(-1); }

    for (int i = 0; i < N; ++i) {
        double n_lo[3], n_hi[3];
        leaves[i]->GetMinPointNormalized(n_lo);
        leaves[i]->GetMaxPointNormalized(n_hi);
        for (int j = 0; j < 6; ++j) {
            if (adj[i][j].count != 0) continue;
            const int fa = P_FACE_FIXED[j];
            const int f1 = P_FACE_FREE1[j];
            const int f2 = P_FACE_FREE2[j];
            const double fc = P_FACE_HI[j] ? n_hi[fa]+PEPS : n_lo[fa]-PEPS;
            if (fc < 0.0 || fc > 1.0) continue;
            const int opp = 5 - j;
            int found[4] = {-1,-1,-1,-1};
            for (int q = 0; q < 4; ++q) {
                const int qi = q & 1, qj = (q >> 1) & 1;
                double pt[3];
                pt[fa] = fc;
                pt[f1] = n_lo[f1] + PQ[qi]*(n_hi[f1]-n_lo[f1]);
                pt[f2] = n_lo[f2] + PQ[qj]*(n_hi[f2]-n_lo[f2]);
                if (pt[0]<0||pt[0]>1||pt[1]<0||pt[1]>1||pt[2]<0||pt[2]>1) continue;
                CellType* nb = rOctree.pGetCellNormalized(pt);
                if (!nb || nb == leaves[i]) continue;
                auto it = id_to_idx.find(nb->GetId());
                if (it == id_to_idx.end()) continue;
                found[q] = it->second;
            }
            int first = -1;
            for (int q = 0; q < 4; ++q) if (found[q] >= 0) { first = found[q]; break; }
            if (first < 0) continue;
            const bool all_same = (found[0]==first&&found[1]==first&&
                                   found[2]==first&&found[3]==first);
            if (all_same) {
                adj[i][j].count = 1; adj[i][j].ids[0] = first;
                if (leaves[i]->GetLevel()==leaves[first]->GetLevel()&&
                    adj[first][opp].count==0) {
                    adj[first][opp].count = 1; adj[first][opp].ids[0] = i;
                }
            } else {
                adj[i][j].count = 0;
                for (int q = 0; q < 4; ++q) {
                    adj[i][j].ids[q] = found[q];
                    if (found[q] >= 0) {
                        adj[i][j].count++;
                        if (adj[found[q]][opp].count==0) {
                            adj[found[q]][opp].count = 1;
                            adj[found[q]][opp].ids[0] = i;
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ //
    // 4. Output primal hexes (one per leaf)
    // ------------------------------------------------------------------ //
    rNodes     = std::move(primal_nodes);
    rCells.clear();  rCells.reserve(N);
    rCellLevel.clear(); rCellLevel.reserve(N);
    for (int i = 0; i < N; ++i) {
        rCells.push_back(primal_elem[i]);
        rCellLevel.push_back(leaves[i]->GetLevel());
    }

    // ------------------------------------------------------------------ //
    // 5. Hanging-node constraints at 2:1 transitions
    //
    // For each coarse face j of leaf i with 4 finer neighbours (count==4):
    //   Masters: the 4 coarse corners on face j, ordered (LL,HL,HH,LH).
    //   Slaves:  4 edge midpoints (2 masters, w=0.5) + 1 face centre
    //            (4 masters, w=0.25) lying on the coarse face but not being
    //            coarse corners — identified from the fine cells' primal_elem.
    //
    // Sub-quadrant ordering q=0→(f1-lo,f2-lo), 1→(f1-hi,f2-lo),
    //   2→(f1-hi,f2-hi), 3→(f1-lo,f2-hi).  For sub-quadrant q at
    //   offset (qi,qj), fine-cell corner index c_pos ∈ {0,1,2,3}
    //   (mapped by FFCT[j]) maps to coarse-face position
    //   (qi+c_pos_f1_offset, qj+c_pos_f2_offset)*half:
    //     c_pos=0 → (qi, qj)*half   = LL of fine
    //     c_pos=1 → (qi+1,qj)*half  = HL of fine
    //     c_pos=2 → (qi+1,qj+1)*half = HH of fine
    //     c_pos=3 → (qi, qj+1)*half = LH of fine
    // ------------------------------------------------------------------ //

    // coarse face corner indices in (LL,HL,HH,LH) order per face j
    static constexpr int CFCT[6][4] = {
        {0,1,2,3}, {0,1,5,4}, {0,3,7,4},
        {1,2,6,5}, {3,2,6,7}, {4,5,6,7}
    };
    // fine face corner indices (those on the coarse face plane) in (LL,HL,HH,LH) order per face j
    static constexpr int FFCT[6][4] = {
        {4,5,6,7}, {3,2,6,7}, {1,2,6,5},
        {0,3,7,4}, {0,1,5,4}, {0,1,2,3}
    };

    // Collect the best (highest NumMasters) constraint per slave node.
    // A face-centre node (4 masters) may be seen first as an edge-midpoint
    // (2 masters) from an adjacent coarse face; the 4-master version is the
    // correct interpolation so it wins by replacing any weaker registration.
    rHanging.clear();
    std::unordered_map<int, HangingConstraint> best;

    auto maybe_add = [&](int slave, int nm,
                         std::array<int,4> masters,
                         std::array<double,4> weights) {
        if (slave < 0) return;
        auto it = best.find(slave);
        if (it == best.end() || it->second.NumMasters < nm) {
            HangingConstraint hc;
            hc.SlaveNode = slave; hc.NumMasters = nm;
            hc.MasterNodes = masters; hc.Weights = weights;
            best[slave] = hc;
        }
    };

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (adj[i][j].count != 4) continue;
            const auto& ids = adj[i][j].ids;
            const int M0 = primal_elem[i][CFCT[j][0]]; // LL master
            const int M1 = primal_elem[i][CFCT[j][1]]; // HL master
            const int M2 = primal_elem[i][CFCT[j][2]]; // HH master
            const int M3 = primal_elem[i][CFCT[j][3]]; // LH master

            auto gfc = [&](int q, int c) {
                return (ids[q] >= 0) ? primal_elem[ids[q]][FFCT[j][c]] : -1;
            };

            // Edge midpoints (2 masters, w=0.5):
            maybe_add(gfc(0,1), 2, {M0,M1,-1,-1}, {0.5,0.5,0.0,0.0}); // bottom (m,a)
            maybe_add(gfc(1,2), 2, {M1,M2,-1,-1}, {0.5,0.5,0.0,0.0}); // right  (b,m)
            maybe_add(gfc(2,3), 2, {M2,M3,-1,-1}, {0.5,0.5,0.0,0.0}); // top    (m,b)
            maybe_add(gfc(3,0), 2, {M3,M0,-1,-1}, {0.5,0.5,0.0,0.0}); // left   (a,m)
            // Face centre (4 masters, w=0.25):
            maybe_add(gfc(0,2), 4, {M0,M1,M2,M3}, {0.25,0.25,0.25,0.25});
        }
    }

    rHanging.reserve(best.size());
    for (auto& [k, hc] : best) rHanging.push_back(hc);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::WriteDualHexVtk(
    OctreeType& rOctree,
    const std::string& rFilename,
    const TriangleSoup* pTriangles,
    bool Project,
    int ProjIters,
    int ProjSmooth)
{
    rOctree.StrongConstrain2To1();

    std::vector<std::array<double,3>> nodes;
    std::vector<std::array<int,8>> cells;
    std::vector<int> cell_level;
    ExtractDualHexMesh(rOctree, nodes, cells, cell_level);

    // Carve away hexes outside the input surface (reference stage 4,
    // RemoveOutsideElement — inside/outside part).  Drops unused nodes.
    if (pTriangles && !pTriangles->empty())
        RemoveOutsideElement(*pTriangles, nodes, cells, cell_level);

    // Fit the carved core mesh to the surface and mesh the buffer zone with
    // Jacobian control (reference stage 5, ProjectToIsoSurface).  First clear
    // the buffer zone (paper Section 2.3): drop boundary hexes where the
    // carved surface folds, so the extruded shell does not self-intersect.
    if (Project && pTriangles && !pTriangles->empty()) {
        ClearBufferZone(nodes, cells, cell_level);
        ProjectToIsoSurface(*pTriangles, nodes, cells, cell_level, ProjIters, ProjSmooth);
    }

    WriteHexVtk(rFilename, nodes, cells, cell_level);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::WritePrimalVtk(OctreeType& rOctree, const std::string& rFilename)
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

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::BuildAndWriteVtk(
    ModelPart& rSurfaceMesh,
    const std::string& rVtkFilename,
    std::size_t RefinementDepth,
    bool Adaptive)
{
    auto p_octree = BuildFromSurfaceMesh(rSurfaceMesh, RefinementDepth, Adaptive);
    WriteDualHexVtk(*p_octree, rVtkFilename);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::BuildCarveAndWriteVtk(
    ModelPart& rSurfaceMesh,
    const std::string& rVtkFilename,
    std::size_t RefinementDepth,
    bool Adaptive)
{
    auto p_octree = BuildFromSurfaceMesh(rSurfaceMesh, RefinementDepth, Adaptive);

    // Collect surface triangles in world coordinates.
    TriangleSoup triangles = ExtractTriangleSoup(rSurfaceMesh);
    KRATOS_ERROR_IF(triangles.empty())
        << "OctreeHybridMeshUtility::BuildCarveAndWriteVtk: the surface "
        << "ModelPart has no triangles to carve against." << std::endl;

    WriteDualHexVtk(*p_octree, rVtkFilename, &triangles);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::BuildCarveProjectAndWriteVtk(
    ModelPart& rSurfaceMesh,
    const std::string& rVtkFilename,
    std::size_t RefinementDepth,
    int ProjIters,
    int ProjSmooth,
    bool Adaptive)
{
    auto p_octree = BuildFromSurfaceMesh(rSurfaceMesh, RefinementDepth, Adaptive);

    TriangleSoup triangles = ExtractTriangleSoup(rSurfaceMesh);
    KRATOS_ERROR_IF(triangles.empty())
        << "OctreeHybridMeshUtility::BuildCarveProjectAndWriteVtk: the surface "
        << "ModelPart has no triangles to fit against." << std::endl;

    WriteDualHexVtk(*p_octree, rVtkFilename, &triangles, /*Project=*/true,
                    ProjIters, ProjSmooth);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::WriteOctreeForReference(
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

/***********************************************************************************/
/***********************************************************************************/

double OctreeHybridMeshUtility::SqDist(const double a[3], const double b[3])
{
    const double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return dx*dx + dy*dy + dz*dz;
}

/***********************************************************************************/
/***********************************************************************************/

double OctreeHybridMeshUtility::TriArea(double a, double b, double c)
{
    const double s = 0.5*(a+b+c);
    const double area = s*(s-a)*(s-b)*(s-c);
    return std::sqrt(area < 0.0 ? 0.0 : area);
}

/***********************************************************************************/
/***********************************************************************************/

int OctreeHybridMeshUtility::TriRayIntersect(
    const double a[3], const double b[3], const double c[3],
    const double p[3], const double dir[3], double e[3], double& alpha)
{
    constexpr double DIST_THRES = 1e-12;
    const double A = (c[1]*b[2]-b[1]*c[2]+a[1]*c[2]-a[2]*c[1]-a[1]*b[2]+a[2]*b[1]);
    const double B = (a[0]*(b[2]-c[2])-b[0]*(a[2]-c[2])+c[0]*(a[2]-b[2]));
    const double C = (a[0]*(c[1]-b[1])-b[0]*(c[1]-a[1])+c[0]*(b[1]-a[1]));
    const double D = a[0]*(b[1]*c[2]-c[1]*b[2])-b[0]*(a[1]*c[2]-a[2]*c[1])+c[0]*(a[1]*b[2]-a[2]*b[1]);
    const double den = A*dir[0]+B*dir[1]+C*dir[2];
    if (std::abs(den) < DIST_THRES) return -1;          // parallel to plane
    alpha = (-A*p[0]-B*p[1]-C*p[2]-D)/den;
    e[0]=p[0]+dir[0]*alpha; e[1]=p[1]+dir[1]*alpha; e[2]=p[2]+dir[2]*alpha;
    const double AP[3]={e[0]-a[0],e[1]-a[1],e[2]-a[2]};
    const double AC[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
    const double AB[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
    auto dot=[](const double u[3],const double v[3]){ return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]; };
    const double fI=dot(AP,AC)*dot(AB,AB)-dot(AP,AB)*dot(AC,AB);
    const double fJ=dot(AP,AB)*dot(AC,AC)-dot(AP,AC)*dot(AB,AC);
    const double fD=dot(AC,AC)*dot(AB,AB)-dot(AC,AB)*dot(AC,AB);
    if (fI>0 && fJ>0 && fI+fJ<fD) return 1;             // inside
    if (fI==0 || fJ==0 || fI+fJ==fD) return -1;         // on the boundary
    return 0;                                            // outside
}

/***********************************************************************************/
/***********************************************************************************/

double OctreeHybridMeshUtility::PointToTri(
    const double a[3], const double b[3], const double c[3],
    const double p[3], double currMin)
{
    const double A = (c[1]*b[2]-b[1]*c[2]+a[1]*c[2]-a[2]*c[1]-a[1]*b[2]+a[2]*b[1]);
    const double B = (a[0]*(b[2]-c[2])-b[0]*(a[2]-c[2])+c[0]*(a[2]-b[2]));
    const double C = (a[0]*(c[1]-b[1])-b[0]*(c[1]-a[1])+c[0]*(b[1]-a[1]));
    const double D = a[0]*(b[1]*c[2]-c[1]*b[2])-b[0]*(a[1]*c[2]-a[2]*c[1])+c[0]*(a[1]*b[2]-a[2]*b[1]);
    const double sum = A*A+B*B+C*C;
    const double tmp = (-A*p[0]-B*p[1]-C*p[2]-D);
    const double alpha = std::abs(tmp/std::sqrt(sum));      // |plane distance|
    if (alpha >= currMin) return alpha;                     // early-out

    const double q[3] = { p[0]+A*tmp/sum, p[1]+B*tmp/sum, p[2]+C*tmp/sum };
    const double QA=std::sqrt(SqDist(q,a)), QB=std::sqrt(SqDist(q,b)), QC=std::sqrt(SqDist(q,c));
    const double AB=std::sqrt(SqDist(a,b)), AC=std::sqrt(SqDist(a,c)), BC=std::sqrt(SqDist(b,c));
    const double S1=TriArea(QA,QB,AB), S2=TriArea(QA,QC,AC), S3=TriArea(QB,QC,BC);

    const double AP[3]={p[0]-a[0],p[1]-a[1],p[2]-a[2]};
    const double ACl[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
    const double ABl[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
    auto dot=[](const double u[3],const double v[3]){ return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]; };
    const double fI=dot(AP,ACl)*dot(ABl,ABl)-dot(AP,ABl)*dot(ACl,ABl);
    const double fJ=dot(AP,ABl)*dot(ACl,ACl)-dot(AP,ACl)*dot(ABl,ACl);
    const double fD=dot(ACl,ACl)*dot(ABl,ABl)-dot(ACl,ABl)*dot(ACl,ABl);
    if (fI>=0 && fJ>=0 && fI+fJ<=fD) return alpha;          // foot inside triangle

    // Closest boundary feature: nearest of the 3 vertices or the 3 edge
    // projections (only edges whose foot lies between the endpoints).
    double beta = std::min({QA, QB, QC});
    const double kAB=((b[0]-a[0])*(q[0]-a[0])+(b[1]-a[1])*(q[1]-a[1])+(b[2]-a[2])*(q[2]-a[2]))/SqDist(a,b);
    const double kBC=((c[0]-b[0])*(q[0]-b[0])+(c[1]-b[1])*(q[1]-b[1])+(c[2]-b[2])*(q[2]-b[2]))/SqDist(b,c);
    const double kCA=((a[0]-c[0])*(q[0]-c[0])+(a[1]-c[1])*(q[1]-c[1])+(a[2]-c[2])*(q[2]-c[2]))/SqDist(c,a);
    if (kAB>0 && kAB<1) beta = std::min(beta, S1*2/AB);
    if (kBC>0 && kBC<1) beta = std::min(beta, S3*2/BC);
    if (kCA>0 && kCA<1) beta = std::min(beta, S2*2/AC);
    return std::sqrt(alpha*alpha + beta*beta);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::RemoveOutsideElement(
    const TriangleSoup& rTriangles,
    const std::vector<std::array<double,3>>& rNodes,
    std::vector<std::array<int,8>>& rCells,
    std::vector<int>& rCellLevel)
{
    const std::vector<double> signed_dist = ComputeNodeSignedDistance(rTriangles, rNodes);

    std::vector<std::array<int,8>> kept_cells;
    std::vector<int> kept_level;
    kept_cells.reserve(rCells.size());
    kept_level.reserve(rCells.size());
    for (std::size_t c = 0; c < rCells.size(); ++c) {
        if (KeepCarvedCell(rCells[c], signed_dist)) {
            kept_cells.push_back(rCells[c]);
            kept_level.push_back(rCellLevel[c]);
        }
    }
    rCells.swap(kept_cells);
    rCellLevel.swap(kept_level);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::ClassifyInsideOutside(
    const TriangleSoup& rTriangles,
    const std::vector<std::array<double,3>>& rNodes,
    const std::vector<std::array<int,8>>& rCells,
    std::vector<int>& rCellColor)
{
    const std::vector<double> signed_dist = ComputeNodeSignedDistance(rTriangles, rNodes);
    rCellColor.assign(rCells.size(), 0);
    for (std::size_t c = 0; c < rCells.size(); ++c)
        rCellColor[c] = KeepCarvedCell(rCells[c], signed_dist) ? 1 : 0;
}

/***********************************************************************************/
/***********************************************************************************/

auto OctreeHybridMeshUtility::ComputeNodeSignedDistance(
    const TriangleSoup& rTriangles,
    const std::vector<std::array<double,3>>& rNodes) -> std::vector<double>
{
    constexpr double DIST_THRES = 1e-12;
    const int NV = static_cast<int>(rNodes.size());
    const int NT = static_cast<int>(rTriangles.size());

    std::vector<double> signed_dist(NV);
    IndexPartition<int>(NV).for_each([&](int i) {
        const double p[3] = { rNodes[i][0], rNodes[i][1], rNodes[i][2] };

        // Deterministic per-node RNG (so the carve is reproducible and the
        // parallel loop is independent of scheduling).
        std::uint32_t s = 2654435761u * static_cast<std::uint32_t>(i + 1) + 12345u;
        auto nextf = [&s]() {
            s = s*1664525u + 1013904223u;
            return (s >> 8) * (1.0 / 16777216.0);   // uniform in [0, 1)
        };

        // Inside/outside via ray-cast crossing parity; perturb and retry if
        // the ray grazes a triangle edge/vertex.
        bool inside = false;
        for (int attempt = 0; attempt < 64; ++attempt) {
            inside = false;
            bool clean = true;
            double dir[3];
            for (int d = 0; d < 3; ++d)
                dir[d] = (nextf() + DIST_THRES) * (nextf() - 0.5 > 0 ? -1.0 : 1.0);
            double e[3], alpha;
            for (int t = 0; t < NT; ++t) {
                const int k = TriRayIntersect(
                    rTriangles[t][0].data(), rTriangles[t][1].data(),
                    rTriangles[t][2].data(), p, dir, e, alpha);
                if (k == 1 && alpha > 0) inside = !inside;
                else if (k == -1) { clean = false; break; }
            }
            if (clean) break;
        }

        // Distance magnitude: closest of all triangles (with early-out).
        double md = std::numeric_limits<double>::max();
        for (int t = 0; t < NT; ++t) {
            const double a = PointToTri(
                rTriangles[t][0].data(), rTriangles[t][1].data(),
                rTriangles[t][2].data(), p, md);
            if (a < md) md = a;
        }
        signed_dist[i] = inside ? md : -md;
    });
    return signed_dist;
}

/***********************************************************************************/
/***********************************************************************************/

bool OctreeHybridMeshUtility::KeepCarvedCell(
    const std::array<int,8>& rCell,
    const std::vector<double>& rSignedDist)
{
    constexpr double OUT_IN_RATIO = 0.15;
    double max_pos = 0.0, min_neg = 0.0;
    int n_out = 0;
    for (int j = 0; j < 8; ++j) {
        const double dv = rSignedDist[rCell[j]];
        if (dv > max_pos) max_pos = dv;
        else if (dv < 0.0) {
            if (++n_out > 2) return false;
            if (dv < min_neg) min_neg = dv;
        }
    }
    return min_neg + OUT_IN_RATIO * max_pos >= 0.0;
}

static constexpr int SJ_ADJ[8][3] =
    {{1,3,4},{2,0,5},{3,1,6},{0,2,7},{7,5,0},{4,6,1},{5,7,2},{6,4,3}};

/***********************************************************************************/
/***********************************************************************************/

double OctreeHybridMeshUtility::TripleProduct(const double e0[3], const double e1[3], const double e2[3])
{
    return e0[0]*(e1[1]*e2[2]-e1[2]*e2[1])
         + e0[1]*(e1[2]*e2[0]-e1[0]*e2[2])
         + e0[2]*(e1[0]*e2[1]-e1[1]*e2[0]);
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::HexEdgeTriple(const double p[8][3], int corner,
                          double e0[3], double e1[3], double e2[3])
{
    if (corner == 8) {  // body centre: opposite face-centre differences
        for (int d = 0; d < 3; ++d) {
            e0[d] = p[1][d]+p[2][d]+p[5][d]+p[6][d]-p[0][d]-p[3][d]-p[4][d]-p[7][d];
            e1[d] = p[2][d]+p[3][d]+p[6][d]+p[7][d]-p[0][d]-p[1][d]-p[4][d]-p[5][d];
            e2[d] = p[4][d]+p[5][d]+p[6][d]+p[7][d]-p[0][d]-p[1][d]-p[2][d]-p[3][d];
        }
    } else {
        for (int d = 0; d < 3; ++d) {
            e0[d] = p[SJ_ADJ[corner][0]][d]-p[corner][d];
            e1[d] = p[SJ_ADJ[corner][1]][d]-p[corner][d];
            e2[d] = p[SJ_ADJ[corner][2]][d]-p[corner][d];
        }
    }
}

/***********************************************************************************/
/***********************************************************************************/

double OctreeHybridMeshUtility::ScaledJacobianMin(const double p[8][3])
{
    constexpr double DIST_THRES = 1e-12;
    double mn = std::numeric_limits<double>::max();
    for (int c = 0; c <= 8; ++c) {
        double e0[3], e1[3], e2[3];
        HexEdgeTriple(p, c, e0, e1, e2);
        const double l0=e0[0]*e0[0]+e0[1]*e0[1]+e0[2]*e0[2];
        const double l1=e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2];
        const double l2=e2[0]*e2[0]+e2[1]*e2[1]+e2[2]*e2[2];
        if (l0<=DIST_THRES || l1<=DIST_THRES || l2<=DIST_THRES)
            return -std::numeric_limits<double>::max();
        const double s = TripleProduct(e0,e1,e2)/std::sqrt(l0*l1*l2);
        if (s < mn) mn = s;
    }
    return mn;
}

/***********************************************************************************/
/***********************************************************************************/

double OctreeHybridMeshUtility::JacobianMin(const double p[8][3])
{
    double mn = std::numeric_limits<double>::max();
    for (int c = 0; c <= 8; ++c) {
        double e0[3], e1[3], e2[3];
        HexEdgeTriple(p, c, e0, e1, e2);
        const double v = TripleProduct(e0,e1,e2);
        if (v < mn) mn = v;
    }
    return mn;
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::ClosestPointOnTriangle(
    const double a[3], const double b[3], const double c[3],
    const double p[3], double q[3])
{
    auto sub=[](const double u[3],const double v[3],double r[3]){ r[0]=u[0]-v[0]; r[1]=u[1]-v[1]; r[2]=u[2]-v[2]; };
    auto dot=[](const double u[3],const double v[3]){ return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]; };
    double ab[3],ac[3],ap[3]; sub(b,a,ab); sub(c,a,ac); sub(p,a,ap);
    const double d1=dot(ab,ap), d2=dot(ac,ap);
    if (d1<=0 && d2<=0) { q[0]=a[0]; q[1]=a[1]; q[2]=a[2]; return; }
    double bp[3]; sub(p,b,bp);
    const double d3=dot(ab,bp), d4=dot(ac,bp);
    if (d3>=0 && d4<=d3) { q[0]=b[0]; q[1]=b[1]; q[2]=b[2]; return; }
    const double vc=d1*d4-d3*d2;
    if (vc<=0 && d1>=0 && d3<=0) { const double v=d1/(d1-d3);
        for(int d=0;d<3;++d) q[d]=a[d]+v*ab[d];
        return; }
    double cp[3]; sub(p,c,cp);
    const double d5=dot(ab,cp), d6=dot(ac,cp);
    if (d6>=0 && d5<=d6) { q[0]=c[0]; q[1]=c[1]; q[2]=c[2]; return; }
    const double vb=d5*d2-d1*d6;
    if (vb<=0 && d2>=0 && d6<=0) { const double w=d2/(d2-d6);
        for(int d=0;d<3;++d) q[d]=a[d]+w*ac[d];
        return; }
    const double va=d3*d6-d5*d4;
    if (va<=0 && (d4-d3)>=0 && (d5-d6)>=0) { const double w=(d4-d3)/((d4-d3)+(d5-d6));
        for(int d=0;d<3;++d) q[d]=b[d]+w*(c[d]-b[d]);
        return; }
    const double denom=1.0/(va+vb+vc);
    const double v=vb*denom, w=vc*denom;
    for(int d=0;d<3;++d) q[d]=a[d]+ab[d]*v+ac[d]*w;
}

/***********************************************************************************/
/***********************************************************************************/

double OctreeHybridMeshUtility::ClosestPointOnSoup(
    const TriangleSoup& rTri, const double p[3], double q[3], int& tri)
{
    double best = std::numeric_limits<double>::max();
    for (int t = 0; t < static_cast<int>(rTri.size()); ++t) {
        double cand[3];
        ClosestPointOnTriangle(rTri[t][0].data(), rTri[t][1].data(),
                               rTri[t][2].data(), p, cand);
        const double dx=cand[0]-p[0], dy=cand[1]-p[1], dz=cand[2]-p[2];
        const double d2=dx*dx+dy*dy+dz*dz;
        if (d2 < best) { best=d2; q[0]=cand[0]; q[1]=cand[1]; q[2]=cand[2]; tri=t; }
    }
    return best;
}

static constexpr int FACE_FIDC[6][4] =
    {{0,1,2,3},{4,5,1,0},{4,0,3,7},{5,6,2,1},{6,7,3,2},{7,6,5,4}};

/***********************************************************************************/
/***********************************************************************************/

auto OctreeHybridMeshUtility::ExtractBoundaryFaces(
    const std::vector<std::array<int,8>>& rCells) -> std::vector<std::array<int,5>>
{
    std::map<std::array<int,4>, int> count;
    std::map<std::array<int,4>, std::array<int,5>> first;
    for (int c = 0; c < static_cast<int>(rCells.size()); ++c)
        for (int f = 0; f < 6; ++f) {
            std::array<int,4> q = { rCells[c][FACE_FIDC[f][0]], rCells[c][FACE_FIDC[f][1]],
                                    rCells[c][FACE_FIDC[f][2]], rCells[c][FACE_FIDC[f][3]] };
            std::array<int,4> key = q; std::sort(key.begin(), key.end());
            if (++count[key] == 1) first[key] = {q[0],q[1],q[2],q[3],c};
        }
    std::vector<std::array<int,5>> bfaces;
    for (const auto& kv : count) if (kv.second == 1) bfaces.push_back(first[kv.first]);
    return bfaces;
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::ClearBufferZone(
    const std::vector<std::array<double,3>>& rNodes,
    std::vector<std::array<int,8>>& rCells,
    std::vector<int>& rCellLevel,
    int MaxRounds)
{
    // ~128 unit probe directions on a Fibonacci sphere.
    constexpr int ND = 128;
    std::array<std::array<double,3>,ND> dir;
    const double ga = 3.39996322972865332;  // golden angle
    for (int i=0;i<ND;++i) {
        const double z = 1.0 - 2.0*(i+0.5)/ND;
        const double r = std::sqrt(std::max(0.0,1.0-z*z));
        const double a = ga*i;
        dir[i] = { r*std::cos(a), r*std::sin(a), z };
    }

    for (int round=0; round<MaxRounds; ++round) {
        auto bfaces = ExtractBoundaryFaces(rCells);
        if (bfaces.empty()) return;

        // Outward normal of each boundary face, and the faces / cells at each
        // boundary vertex.
        const int NN = static_cast<int>(rNodes.size());
        std::vector<std::vector<std::array<double,3>>> vnormals(NN);
        std::vector<std::map<int,int>> vcell_bfaces(NN);  // vertex -> (cell -> #boundary faces)
        for (const auto& bf : bfaces) {
            const int c0=bf[0],c1=bf[1],c2=bf[2],c3=bf[3], cell=bf[4];
            double cen[3]={0,0,0};
            for (int k=0;k<8;++k) for (int d=0;d<3;++d) cen[d]+=rNodes[rCells[cell][k]][d];
            for (int d=0;d<3;++d) cen[d]/=8.0;
            double fc[3],e1[3],e2[3],n[3];
            for (int d=0;d<3;++d){ fc[d]=0.25*(rNodes[c0][d]+rNodes[c1][d]+rNodes[c2][d]+rNodes[c3][d]);
                e1[d]=rNodes[c1][d]-rNodes[c0][d]; e2[d]=rNodes[c3][d]-rNodes[c0][d]; }
            n[0]=e1[1]*e2[2]-e1[2]*e2[1]; n[1]=e1[2]*e2[0]-e1[0]*e2[2]; n[2]=e1[0]*e2[1]-e1[1]*e2[0];
            const double L=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])+1e-300;
            for (int d=0;d<3;++d) n[d]/=L;
            if (n[0]*(fc[0]-cen[0])+n[1]*(fc[1]-cen[1])+n[2]*(fc[2]-cen[2]) < 0)
                for (int d=0;d<3;++d) n[d]=-n[d];
            for (int j=0;j<4;++j) {
                vnormals[bf[j]].push_back({n[0],n[1],n[2]});
                vcell_bfaces[bf[j]][cell]++;
            }
        }

        // A vertex is folded if no probe direction is positive against all of
        // its boundary-face normals.  At each folded vertex pick its most
        // exposed incident cell for deletion.
        std::set<int> to_delete;
        for (int v=0; v<NN; ++v) {
            if (vnormals[v].size() < 2) continue;
            bool ok=false;
            for (int i=0;i<ND && !ok;++i) {
                bool all=true;
                for (const auto& n : vnormals[v])
                    if (dir[i][0]*n[0]+dir[i][1]*n[1]+dir[i][2]*n[2] <= 1e-3) { all=false; break; }
                if (all) ok=true;
            }
            if (ok) continue;
            int best_cell=-1, best_cnt=-1;
            for (const auto& cb : vcell_bfaces[v])
                if (cb.second > best_cnt) { best_cnt=cb.second; best_cell=cb.first; }
            if (best_cell>=0) to_delete.insert(best_cell);
        }
        if (to_delete.empty()) return;

        std::vector<std::array<int,8>> kept; std::vector<int> kept_lv;
        kept.reserve(rCells.size());
        for (int c=0;c<static_cast<int>(rCells.size());++c)
            if (!to_delete.count(c)) { kept.push_back(rCells[c]); kept_lv.push_back(rCellLevel[c]); }
        rCells.swap(kept); rCellLevel.swap(kept_lv);
    }
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::ProjectToIsoSurface(
    const TriangleSoup& rTriangles,
    std::vector<std::array<double,3>>& rNodes,
    std::vector<std::array<int,8>>& rCells,
    std::vector<int>& rCellLevel,
    int TotalIters,
    int SmoothEvery)
{
    const int n_core_cells = static_cast<int>(rCells.size());
    if (n_core_cells == 0 || rTriangles.empty()) return;

    // --- 0. Normalise to the reference's 100-unit box --------------------
    // All of the optimisation constants below (learning rate, fitting weight,
    // scaled-Jacobian thresholds, tolerances) are the reference's, which runs
    // on a model rescaled so its largest extent is 100.  Working in the input
    // (e.g. unit) coordinates would make the scale-dependent gradients blow up,
    // so we rescale on entry and undo it on exit.
    double lo[3]={ 1e300, 1e300, 1e300}, hi[3]={-1e300,-1e300,-1e300};
    for (const auto& nd : rNodes)
        for (int d=0; d<3; ++d) { lo[d]=std::min(lo[d],nd[d]); hi[d]=std::max(hi[d],nd[d]); }
    double extent = std::max({hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2], 1e-300});
    const double S = 100.0 / extent;
    for (auto& nd : rNodes) for (int d=0;d<3;++d) nd[d]=(nd[d]-lo[d])*S;
    TriangleSoup tri = rTriangles;             // local, normalised copy
    for (auto& t : tri) for (auto& v : t) for (int d=0;d<3;++d) v[d]=(v[d]-lo[d])*S;
    const TriangleSoup& rTri = tri;            // shadow the argument below

    // --- 1. Boundary quad faces (owned by exactly one hex) ----------------
    static constexpr int FIDC[6][4] =
        {{0,1,2,3},{4,5,1,0},{4,0,3,7},{5,6,2,1},{6,7,3,2},{7,6,5,4}};
    std::map<std::array<int,4>, int>            face_count;
    std::map<std::array<int,4>, std::array<int,5>> face_first; // key -> {n0..n3, cell}
    for (int c = 0; c < n_core_cells; ++c)
        for (int f = 0; f < 6; ++f) {
            std::array<int,4> q = { rCells[c][FIDC[f][0]], rCells[c][FIDC[f][1]],
                                    rCells[c][FIDC[f][2]], rCells[c][FIDC[f][3]] };
            std::array<int,4> key = q; std::sort(key.begin(), key.end());
            if (++face_count[key] == 1) face_first[key] = {q[0],q[1],q[2],q[3],c};
        }
    std::vector<std::array<int,5>> bfaces;  // {n0,n1,n2,n3, owning cell}
    for (const auto& kv : face_count)
        if (kv.second == 1) bfaces.push_back(face_first[kv.first]);
    if (bfaces.empty()) return;

    // --- 2. Duplicate boundary vertices, build the buffer shell -----------
    std::unordered_map<int,int> dup_of;     // core boundary node -> dup node id
    for (const auto& bf : bfaces)
        for (int j = 0; j < 4; ++j)
            if (dup_of.find(bf[j]) == dup_of.end()) {
                const int id = static_cast<int>(rNodes.size());
                dup_of.emplace(bf[j], id);
                rNodes.push_back(rNodes[bf[j]]);   // duplicate starts coincident
            }

    // Cell centroid helper (to orient buffer hexes outward).
    auto centroid = [&](int c, double o[3]) {
        o[0]=o[1]=o[2]=0;
        for (int k=0;k<8;++k) for (int d=0;d<3;++d) o[d]+=rNodes[rCells[c][k]][d];
        for (int d=0;d<3;++d) o[d]/=8.0;
    };

    const int n_buffer_start = static_cast<int>(rCells.size());
    for (const auto& bf : bfaces) {
        const int c0=bf[0], c1=bf[1], c2=bf[2], c3=bf[3];
        const int d0=dup_of[c0], d1=dup_of[c1], d2=dup_of[c2], d3=dup_of[c3];
        // Outward normal of the quad (away from the owning cell centroid).
        double cen[3]; centroid(bf[4], cen);
        double fc[3]={0,0,0};
        for (int d=0;d<3;++d) fc[d]=0.25*(rNodes[c0][d]+rNodes[c1][d]+rNodes[c2][d]+rNodes[c3][d]);
        double e1[3],e2[3],nrm[3];
        for (int d=0;d<3;++d){ e1[d]=rNodes[c1][d]-rNodes[c0][d]; e2[d]=rNodes[c3][d]-rNodes[c0][d]; }
        nrm[0]=e1[1]*e2[2]-e1[2]*e2[1]; nrm[1]=e1[2]*e2[0]-e1[0]*e2[2]; nrm[2]=e1[0]*e2[1]-e1[1]*e2[0];
        double out[3]={fc[0]-cen[0],fc[1]-cen[1],fc[2]-cen[2]};
        const double nlen=std::sqrt(nrm[0]*nrm[0]+nrm[1]*nrm[1]+nrm[2]*nrm[2])+1e-300;
        for (int d=0;d<3;++d) nrm[d]/=nlen;
        if (nrm[0]*out[0]+nrm[1]*out[1]+nrm[2]*out[2] < 0)
            for (int d=0;d<3;++d) nrm[d]=-nrm[d];
        // Build the buffer hex (duplicates on the outward side, core on the
        // inner side); choose the winding that gives a positive test volume
        // when the duplicates are nudged a unit outward.
        double test[8][3];
        auto fill_test = [&](const std::array<int,8>& e){
            for (int k=0;k<8;++k) for (int d=0;d<3;++d) test[k][d]=rNodes[e[k]][d];
            for (int k=0;k<4;++k) for (int d=0;d<3;++d) test[k][d]+=nrm[d]; // nudge dups out
        };
        std::array<int,8> hexA = { d0,d1,d2,d3, c0,c1,c2,c3 };
        fill_test(hexA);
        std::array<int,8> hex = hexA;
        if (JacobianMin(test) <= 0) hex = { d0,d3,d2,d1, c0,c3,c2,c1 };
        rCells.push_back(hex);
        rCellLevel.push_back(-2);   // buffer-layer marker
    }

    // --- 3. Affected elements, optimizable vertices, adjacency ------------
    const int NN = static_cast<int>(rNodes.size());
    std::vector<std::vector<int>> node_cells(NN);     // node -> incident cells
    for (int c = 0; c < static_cast<int>(rCells.size()); ++c)
        for (int k = 0; k < 8; ++k) node_cells[rCells[c][k]].push_back(c);

    std::vector<char> is_dup(NN, 0), is_boundary(NN, 0);
    for (const auto& kv : dup_of) { is_boundary[kv.first]=1; is_dup[kv.second]=1; }

    std::vector<char> affected_cell(rCells.size(), 0);
    for (int c = n_buffer_start; c < static_cast<int>(rCells.size()); ++c) affected_cell[c]=1;
    for (int c = 0; c < n_core_cells; ++c)
        for (int k = 0; k < 8; ++k)
            if (is_boundary[rCells[c][k]]) { affected_cell[c]=1; break; }
    std::vector<int> affected;
    for (int c = 0; c < static_cast<int>(rCells.size()); ++c)
        if (affected_cell[c]) affected.push_back(c);

    std::vector<char> optimizable(NN, 0);
    for (int c : affected) for (int k=0;k<8;++k) optimizable[rCells[c][k]]=1;

    // Smoothing neighbours: core-boundary points average their core
    // neighbours; duplicate points average their duplicate-ring neighbours.
    std::vector<std::vector<int>> smooth_nbr(NN);
    {
        std::vector<std::set<int>> s(NN);
        for (const auto& bf : bfaces) {
            for (int j=0;j<4;++j) {
                const int dj=dup_of[bf[j]];
                s[dj].insert(dup_of[bf[(j+1)%4]]);
                s[dj].insert(dup_of[bf[(j+3)%4]]);
            }
        }
        static constexpr int ADJ[8][3] =
            {{1,3,4},{0,2,5},{1,3,6},{0,2,7},{0,5,7},{1,4,6},{2,5,7},{3,4,6}};
        // Core-boundary points smooth toward their edge-adjacent corners in
        // ALL incident elements, *including the buffer hexes* (reference cP2):
        // this pulls the core boundary toward the on-surface duplicates and so
        // straightens the buffer prisms, which dominate the quality budget.
        for (int c=0;c<static_cast<int>(rCells.size());++c)
            for (int k=0;k<8;++k) {
                const int v=rCells[c][k];
                if (!is_boundary[v]) continue;
                for (int a=0;a<3;++a) s[v].insert(rCells[c][ADJ[k][a]]);
            }
        for (int i=0;i<NN;++i) smooth_nbr[i].assign(s[i].begin(), s[i].end());
    }

    // Closest triangle / projection target for each duplicate node.
    std::vector<int> dup_tri(NN, -1);
    auto load_hex = [&](int c, double p[8][3]) {
        for (int k=0;k<8;++k) for (int d=0;d<3;++d) p[k][d]=rNodes[rCells[c][k]][d];
    };
    for (int i=0;i<NN;++i) if (is_dup[i]) {
        double q[3]={0,0,0}; int tri=-1;
        ClosestPointOnSoup(rTri, rNodes[i].data(), q, tri);
        dup_tri[i]=tri;
    }

    // --- 4. Optimisation --------------------------------------------------
    constexpr double LR  = 5.0e-4;     // learning rate for the surface attractor
    constexpr double LRQ = 2.0e-3;     // learning rate for the quality (untangling) gradient
    constexpr double H   = 1.0e-4;     // finite-difference step
    constexpr double TOL = 1.0e-3;     // shell-on-surface tolerance (100-unit box)
    double eps_sj = 0.01;              // scaled-Jacobian gate (reference ELEM_THRES)
    constexpr double EPS_TARGET = 0.50;// escalation ceiling (reference drives toward >0.5)
    constexpr double EPS_STEP   = 0.03;// per-window escalation increment
    constexpr int    STALL_MAX  = 8;   // windows allowed to recover before freezing eps
    std::vector<std::array<double,3>> grad(NN);   // surface attractor
    std::vector<std::array<double,3>> gq(NN);     // quality / untangling gradient

    // Per-node gradient of an element's quality metric (scaled Jacobian when
    // the element is non-inverted, raw Jacobian when inverted), accumulated
    // for every optimisable corner via central differences.
    auto accumulate_quality_grad = [&](int c) {
        double p[8][3]; load_hex(c, p);
        const bool inverted = JacobianMin(p) <= 0.0;
        for (int k=0;k<8;++k) {
            const int v=rCells[c][k];
            if (!optimizable[v]) continue;
            for (int d=0; d<3; ++d) {
                const double save=p[k][d];
                p[k][d]=save+H; const double fp = inverted ? JacobianMin(p) : ScaledJacobianMin(p);
                p[k][d]=save-H; const double fm = inverted ? JacobianMin(p) : ScaledJacobianMin(p);
                p[k][d]=save;
                gq[v][d] += (fp-fm)/(2*H);   // ascent on quality
            }
        }
    };

    // Smart-Laplacian gated move: accept a Laplacian move only if it keeps
    // every incident affected element's scaled Jacobian above the threshold.
    auto try_move = [&](int v, const double tgt[3]) {
        std::array<double,3> old = rNodes[v];
        rNodes[v] = {tgt[0],tgt[1],tgt[2]};
        for (int c : node_cells[v]) {
            if (!affected_cell[c]) continue;
            double p[8][3]; load_hex(c,p);
            if (ScaledJacobianMin(p) <= eps_sj) { rNodes[v]=old; return; }
        }
    };
    // One smoothing sweep: surface (duplicate) points smooth toward their
    // surface ring and re-project onto the input surface; inner boundary
    // points smooth toward their incident-corner ring — which, crucially,
    // spans the buffer hexes too (see smooth_nbr), so the core boundary
    // follows the on-surface duplicates and the buffer prisms stay regular.
    // Every move is scaled-Jacobian-gated, so smoothing never tangles a cell.
    auto smoothing_sweep = [&]() {
        for (int v=0; v<NN; ++v) {
            if (!optimizable[v] || smooth_nbr[v].empty()) continue;
            double avg[3]={0,0,0};
            for (int nb : smooth_nbr[v]) for (int d=0;d<3;++d) avg[d]+=rNodes[nb][d];
            for (int d=0;d<3;++d) avg[d]/=smooth_nbr[v].size();
            if (is_dup[v]) {
                double q[3]={0,0,0}; int tri=-1;
                ClosestPointOnSoup(rTri, avg, q, tri);
                try_move(v, q);
            } else if (is_boundary[v]) {
                try_move(v, avg);
            }
        }
    };

    // The reference runs ProjectToIsoSurface as an endless loop whose quality
    // comes from the Sj-gated smoothing plus a threshold escalation that drives
    // the worst element up toward the paper's >0.5 target.  A single 0.01->0.53
    // jump diverges under the finite-difference gradient (it makes dozens of
    // cells bad at once and strands the duplicates off the surface), so we climb
    // the gate with three stabilisers:
    //   * a *gradual* ramp of eps_sj (small steps, only while the mesh is valid
    //     at the current gate) rather than one discrete jump;
    //   * an *always-on* surface attractor on the duplicates (full strength when
    //     valid, attenuated while untangling) so the shell never drifts off the
    //     geometry during a ramp window;
    //   * a best-valid snapshot, restored on exit, so escalation can only ever
    //     raise the worst element, never degrade the converged mesh.
    auto global_min_sj = [&]() {
        double m = 1.0e30;
        for (int c : affected) { double p[8][3]; load_hex(c, p); m = std::min(m, ScaledJacobianMin(p)); }
        return m;
    };

    std::vector<std::array<double,3>> best_nodes = rNodes;
    double best_min_sj = global_min_sj();
    int drag_count = 0;
    int stall = 0;
    for (int it = 1; it <= TotalIters; ++it) {
        for (int i=0;i<NN;++i) { grad[i]={0,0,0}; gq[i]={0,0,0}; }

        int bad = 0;
        for (int c : affected) {
            double p[8][3]; load_hex(c, p);
            if (ScaledJacobianMin(p) <= eps_sj) { ++bad; accumulate_quality_grad(c); }
        }
        // Geometry-fitting attractor: always pull the duplicated shell onto the
        // input surface so escalation cannot strand it off the geometry.  Full
        // strength once the mesh is valid at the current gate, attenuated while
        // sub-threshold cells remain (so untangling leads).
        const double w_attr = (bad == 0) ? 3.0 : 1.0;
        for (int i=0;i<NN;++i) if (is_dup[i]) {
            double q[3];
            ClosestPointOnTriangle(rTri[dup_tri[i]][0].data(),
                rTri[dup_tri[i]][1].data(), rTri[dup_tri[i]][2].data(),
                rNodes[i].data(), q);
            for (int d=0;d<3;++d) grad[i][d] += -w_attr*(rNodes[i][d]-q[d]);
        }
        for (int i=0;i<NN;++i) if (optimizable[i])
            for (int d=0;d<3;++d) rNodes[i][d] += LR*grad[i][d] + LRQ*gq[i][d];

        if (it % SmoothEvery == 0) {
            smoothing_sweep();

            // Refresh closest triangles, find the duplicate sitting furthest
            // from the surface, and drag it on (ungated) so the shell keeps
            // closing onto the geometry where smoothing alone cannot.
            double max_dist = 0.0; int max_dist_node = -1;
            for (int i=0;i<NN;++i) if (is_dup[i]) {
                double q[3]={0,0,0}; int tri=-1;
                const double d2 = ClosestPointOnSoup(rTri, rNodes[i].data(), q, tri);
                dup_tri[i]=tri;
                const double dd=std::sqrt(d2);
                if (dd>max_dist) { max_dist=dd; max_dist_node=i; }
            }
            if (max_dist_node >= 0 && max_dist >= TOL && (drag_count++ % 4 == 0)) {
                double q[3]={0,0,0}; int tri=-1;
                ClosestPointOnSoup(rTri, rNodes[max_dist_node].data(), q, tri);
                for (int d=0;d<3;++d) rNodes[max_dist_node][d] = q[d];
            }

            // Snapshot the best valid mesh (worst element highest), then ramp
            // the gate: a valid window raises eps_sj one step; a window that
            // cannot regain validity counts toward a stall budget, after which
            // the gate backs off one step so the remaining iterations settle at
            // the best reachable worst element.
            const double gmin = global_min_sj();
            if (gmin > best_min_sj) { best_min_sj = gmin; best_nodes = rNodes; }
            int bad_now = 0;
            for (int c : affected) {
                double p[8][3]; load_hex(c,p);
                if (ScaledJacobianMin(p) <= eps_sj) { bad_now = 1; break; }
            }
            if (bad_now == 0) {
                stall = 0;
                if (eps_sj < EPS_TARGET) eps_sj = std::min(EPS_TARGET, eps_sj + EPS_STEP);
            } else if (++stall >= STALL_MAX) {
                eps_sj = std::max(0.01, eps_sj - EPS_STEP);  // back off one step
                stall = 0;
            }
        }
    }

    // Restore the best valid mesh found during escalation.
    if (global_min_sj() < best_min_sj) rNodes = best_nodes;

    // --- 5. Undo the normalisation --------------------------------------
    for (auto& nd : rNodes) for (int d=0; d<3; ++d) nd[d] = nd[d]/S + lo[d];
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshUtility::WriteHexVtk(
    const std::string& rFilename,
    const std::vector<std::array<double,3>>& rNodes,
    const std::vector<std::array<int,8>>& rCells,
    const std::vector<int>& rCellLevel)
{
    std::ofstream f(rFilename);
    KRATOS_ERROR_IF_NOT(f.is_open())
        << "OctreeHybridMeshUtility::WriteHexVtk: cannot open '"
        << rFilename << "'" << std::endl;

    // Keep only the nodes that some cell references, remapping ids.
    std::vector<int> remap(rNodes.size(), -1);
    std::vector<std::array<double,3>> out_nodes;
    out_nodes.reserve(rNodes.size());
    for (const auto& h : rCells)
        for (int k = 0; k < 8; ++k) {
            int& r = remap[h[k]];
            if (r < 0) { r = static_cast<int>(out_nodes.size()); out_nodes.push_back(rNodes[h[k]]); }
        }

    const std::size_t nn = out_nodes.size();
    const std::size_t nc = rCells.size();

    f << "# vtk DataFile Version 2.0\n"
      << "OctreeHybrid dual hex mesh\n"
      << "ASCII\n"
      << "DATASET UNSTRUCTURED_GRID\n";

    f << "POINTS " << nn << " double\n";
    f << std::scientific; f.precision(10);
    for (const auto& nd : out_nodes)
        f << nd[0] << ' ' << nd[1] << ' ' << nd[2] << '\n';

    f << "CELLS " << nc << ' ' << nc * 9 << '\n';
    for (const auto& h : rCells)
        f << "8 "<<remap[h[0]]<<' '<<remap[h[1]]<<' '<<remap[h[2]]<<' '<<remap[h[3]]
          <<' '<<remap[h[4]]<<' '<<remap[h[5]]<<' '<<remap[h[6]]<<' '<<remap[h[7]]<<'\n';

    f << "CELL_TYPES " << nc << '\n';
    for (std::size_t e = 0; e < nc; ++e) f << "12\n";

    if (nc > 0) {
        f << "CELL_DATA " << nc << '\n'
          << "SCALARS level int 1\nLOOKUP_TABLE default\n";
        for (int lv : rCellLevel) f << lv << '\n';
    }
}

} // namespace Kratos
