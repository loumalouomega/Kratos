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
#include <cmath>

// External includes

// Project includes
#include "modeler/operation/octree_hybrid_report_mesh_quality.h"
#include "modeler/octree_hybrid_mesher_modeler.h"
#include "utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

const Parameters OctreeHybridReportMeshQuality::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"             : "OctreeHybridReportMeshQuality",
        "model_part_name"  : "Undefined"
    })");
}

/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridReportMeshQuality::Execute(OctreeHybridMesherModeler& rModeler, Parameters OperationParameters) const
{
    ModelPart& r_mp = rModeler.GetModel().GetModelPart(
        OperationParameters["model_part_name"].GetString());

    const int n = static_cast<int>(r_mp.NumberOfElements());
    if (n == 0) {
        KRATOS_INFO("OctreeHybridReportMeshQuality")
            << r_mp.FullName() << ": no elements." << std::endl;
        return;
    }

    // Compute min scaled Jacobian for each hex using the engine's SJ_ADJ convention.
    // The SJ_ADJ table uses the same corner ordering as Hexahedra3D8.
    double sj_min = 1.0e30, sj_sum = 0.0;
    int n_inverted = 0;

    for (const auto& r_el : r_mp.Elements()) {
        const auto& r_geom = r_el.GetGeometry();
        if (r_geom.size() != 8) continue;

        double p[8][3];
        for (int k = 0; k < 8; ++k) {
            p[k][0] = r_geom[k].X();
            p[k][1] = r_geom[k].Y();
            p[k][2] = r_geom[k].Z();
        }
        const double sj = OctreeHybridMeshUtility::ScaledJacobianMin(p);
        sj_sum += sj;
        if (sj < sj_min) sj_min = sj;
        if (sj <= 0.0) ++n_inverted;
    }

    const double sj_mean = sj_sum / n;

    KRATOS_INFO("OctreeHybridReportMeshQuality")
        << r_mp.FullName() << " (" << n << " hexes): "
        << "minSJ=" << sj_min
        << "  meanSJ=" << sj_mean
        << "  inverted=" << n_inverted
        << " (" << 100.0 * n_inverted / n << "%)" << std::endl;
}

} // namespace Kratos
