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
 *        (read with StlIO) and exports the resulting hex mesh.
 *
 * ### Workflow
 * ```
 * model_part ──StlIO──► surface_mp
 * OctreeHybridMeshUtility::BuildAndWriteVtk(surface_mp, "mesh.vtk", depth=5)
 * ```
 *
 * ### What it does
 * 1. Computes the bounding box of the surface mesh (with 1 % padding).
 * 2. Constructs an OctreeHybrid and sets its bounding box.
 * 3. Iteratively subdivides every leaf that intersects a surface triangle,
 *    down to the requested target depth.
 * 4. Writes a legacy ASCII VTK file (UNSTRUCTURED_GRID, VTK_HEXAHEDRON cells)
 *    with a cell-data field "level" for Paraview colouring.
 *
 * All octree leaves — coarse interior/exterior cells and fine boundary cells —
 * are written as hexahedra.  Shared faces between adjacent cells at different
 * levels produce hanging nodes (a non-conforming mesh), which is correct for
 * visualising the octree structure.
 *
 * ### Node deduplication
 * Nodes are keyed by their integer grid coordinates at the finest level
 * (resolution = 2^depth per axis).  Two adjacent cells at the same level share
 * a face and thus the same integer coordinates on that face, so they
 * automatically share nodes.  Cells at different levels that share a corner
 * likewise reuse the same node.
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
     * @param rSurfaceMesh     ModelPart containing the surface triangles
     *                         (populated by StlIO::ReadModelPart).
     * @param RefinementDepth  Maximum refinement level applied near the surface.
     *                         Must be in [1, OctreeHybridKratosConfiguration::MAX_DEPTH].
     * @return                 Owning pointer to the constructed octree.
     */
    static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth)
    {
        KRATOS_ERROR_IF(RefinementDepth < 1 || RefinementDepth > ConfigurationType::MAX_DEPTH)
            << "OctreeHybridMeshUtility: RefinementDepth must be in [1, "
            << ConfigurationType::MAX_DEPTH << "], got " << RefinementDepth << std::endl;

        // --- 1. Bounding box ---------------------------------------------------
        double lo[3] = { std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max() };
        double hi[3] = { std::numeric_limits<double>::lowest(),
                         std::numeric_limits<double>::lowest(),
                         std::numeric_limits<double>::lowest() };

        for (const auto& r_node : rSurfaceMesh.Nodes()) {
            lo[0] = std::min(lo[0], r_node.X());
            lo[1] = std::min(lo[1], r_node.Y());
            lo[2] = std::min(lo[2], r_node.Z());
            hi[0] = std::max(hi[0], r_node.X());
            hi[1] = std::max(hi[1], r_node.Y());
            hi[2] = std::max(hi[2], r_node.Z());
        }

        // 1 % padding so surface triangles on the boundary are not clipped.
        for (std::size_t i = 0; i < 3; ++i) {
            const double span = hi[i] - lo[i];
            lo[i] -= 0.01 * span;
            hi[i] += 0.01 * span;
        }

        // --- 2. Build octree ---------------------------------------------------
        auto p_octree = std::make_unique<OctreeType>(RefinementDepth);
        p_octree->SetBoundingBox(lo, hi);

        // --- 3. Collect triangles ---------------------------------------------
        std::vector<GeometryType*> triangles;
        triangles.reserve(rSurfaceMesh.NumberOfGeometries());
        for (auto& r_geom : rSurfaceMesh.Geometries()) {
            triangles.push_back(&r_geom);
        }

        // --- 4. Adaptive refinement -------------------------------------------
        // Iterate level by level so that newly created children are refined
        // in subsequent iterations.
        for (std::size_t iter = 0; iter < RefinementDepth; ++iter) {
            std::vector<CellType*> leaves;
            p_octree->GetAllLeavesVector(leaves);

            bool any_split = false;
            for (CellType* p_cell : leaves) {
                if (static_cast<std::size_t>(p_cell->GetLevel()) >= RefinementDepth)
                    continue;

                // World-space bounding box of this cell.
                double norm_lo[3], norm_hi[3];
                p_cell->GetMinPointNormalized(norm_lo);
                p_cell->GetMaxPointNormalized(norm_hi);
                double world_lo[3], world_hi[3];
                p_octree->ScaleBackToOriginalCoordinate(norm_lo, world_lo);
                p_octree->ScaleBackToOriginalCoordinate(norm_hi, world_hi);

                const Point box_lo(world_lo[0], world_lo[1], world_lo[2]);
                const Point box_hi(world_hi[0], world_hi[1], world_hi[2]);

                for (GeometryType* p_tri : triangles) {
                    if (p_tri->HasIntersection(box_lo, box_hi)) {
                        p_octree->SubdivideCellByIdAndLevel(
                            p_cell->GetId(), p_cell->GetLevel());
                        any_split = true;
                        break; // one hit is enough to trigger subdivision
                    }
                }
            }

            if (!any_split) break; // converged early
        }

        return p_octree;
    }

    /**
     * @brief Writes all octree leaves as hexahedra to a legacy ASCII VTK file.
     *
     * The output is directly loadable by Paraview.  Cell-data field "level"
     * encodes the refinement level of each hexahedron (0 = coarsest).
     *
     * VTK HEXAHEDRON (cell type 12) node order:
     * @code
     *     7 ─── 6
     *    /|    /|
     *   4 ─── 5 |
     *   | 3 ──┼─2
     *   |/    |/
     *   0 ─── 1
     * @endcode
     *
     * @param rOctree   The built octree (bounding box must be set).
     * @param rFilename Path to the output .vtk file.
     */
    static void WriteVtk(OctreeType& rOctree, const std::string& rFilename)
    {
        std::vector<CellType*> leaves;
        rOctree.GetAllLeavesVector(leaves);

        const std::size_t depth = rOctree.GetDepth();
        // Resolution at the finest level: 2^depth cells per axis.
        const std::size_t R = std::size_t{1} << depth;
        // Number of grid points per axis at finest resolution.
        const std::size_t pts = R + 1;

        // VTK HEXAHEDRON corner offsets (dx, dy, dz) in the same order as
        // the figure above, matching OctreeBinaryCell corner indices.
        static constexpr int dx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
        static constexpr int dy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
        static constexpr int dz[8] = {0, 0, 0, 0, 1, 1, 1, 1};

        // --- Node deduplication -----------------------------------------------
        // Key = iz*(pts^2) + iy*pts + ix  (integer grid coords at finest level).
        std::unordered_map<std::size_t, std::size_t> node_map;
        std::vector<std::array<double, 3>> node_coords;

        // Connectivity table: one array of 8 node IDs per hex.
        std::vector<std::array<std::size_t, 8>> hex_connectivity;
        hex_connectivity.reserve(leaves.size());

        for (CellType* p_cell : leaves) {
            const int level  = p_cell->GetLevel();
            const int gx     = p_cell->GetGridX();
            const int gy     = p_cell->GetGridY();
            const int gz     = p_cell->GetGridZ();
            // How many finest-level grid steps correspond to one cell edge.
            const std::size_t stride = std::size_t{1} << (depth - static_cast<std::size_t>(level));

            std::array<std::size_t, 8> conn{};
            for (int c = 0; c < 8; ++c) {
                const std::size_t ix = static_cast<std::size_t>(gx + dx[c]) * stride;
                const std::size_t iy = static_cast<std::size_t>(gy + dy[c]) * stride;
                const std::size_t iz = static_cast<std::size_t>(gz + dz[c]) * stride;

                const std::size_t key = iz * pts * pts + iy * pts + ix;
                auto [it, inserted] = node_map.emplace(key, node_coords.size());
                if (inserted) {
                    // Convert finest-grid integer coords → normalised → world.
                    const double scale = 1.0 / static_cast<double>(R);
                    double norm[3] = {
                        static_cast<double>(ix) * scale,
                        static_cast<double>(iy) * scale,
                        static_cast<double>(iz) * scale
                    };
                    double world[3];
                    rOctree.ScaleBackToOriginalCoordinate(norm, world);
                    node_coords.push_back({world[0], world[1], world[2]});
                }
                conn[c] = it->second;
            }
            hex_connectivity.push_back(conn);
        }

        // --- Write VTK --------------------------------------------------------
        std::ofstream f(rFilename);
        KRATOS_ERROR_IF_NOT(f.is_open())
            << "OctreeHybridMeshUtility::WriteVtk: cannot open '" << rFilename << "'" << std::endl;

        const std::size_t n_nodes = node_coords.size();
        const std::size_t n_cells = hex_connectivity.size();

        f << "# vtk DataFile Version 2.0\n"
          << "OctreeHybrid adaptive hex mesh\n"
          << "ASCII\n"
          << "DATASET UNSTRUCTURED_GRID\n";

        f << "POINTS " << n_nodes << " double\n";
        f << std::scientific;
        f.precision(10);
        for (const auto& pt : node_coords) {
            f << pt[0] << ' ' << pt[1] << ' ' << pt[2] << '\n';
        }

        // CELLS section: each hex has 8 nodes + 1 count = 9 integers.
        f << "CELLS " << n_cells << ' ' << n_cells * 9 << '\n';
        for (const auto& conn : hex_connectivity) {
            f << "8 " << conn[0] << ' ' << conn[1] << ' ' << conn[2] << ' ' << conn[3]
              << ' '  << conn[4] << ' ' << conn[5] << ' ' << conn[6] << ' ' << conn[7] << '\n';
        }

        f << "CELL_TYPES " << n_cells << '\n';
        for (std::size_t i = 0; i < n_cells; ++i) {
            f << "12\n"; // VTK_HEXAHEDRON
        }

        // Level as cell scalar for Paraview colouring.
        f << "CELL_DATA " << n_cells << '\n'
          << "SCALARS level int 1\n"
          << "LOOKUP_TABLE default\n";
        for (const CellType* p_cell : leaves) {
            f << p_cell->GetLevel() << '\n';
        }
    }

    /**
     * @brief Convenience wrapper: builds the octree and immediately writes VTK.
     *
     * @param rSurfaceMesh     Surface ModelPart (from StlIO).
     * @param rVtkFilename     Output .vtk path.
     * @param RefinementDepth  Maximum refinement depth near the surface.
     */
    static void BuildAndWriteVtk(
        ModelPart& rSurfaceMesh,
        const std::string& rVtkFilename,
        std::size_t RefinementDepth = 5)
    {
        auto p_octree = BuildFromSurfaceMesh(rSurfaceMesh, RefinementDepth);
        WriteVtk(*p_octree, rVtkFilename);
    }

    ///@}
};

} // namespace Kratos
